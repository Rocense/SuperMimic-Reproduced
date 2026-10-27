"""Initialize student BPTT from the selected teacher and its distilled student."""
# ruff: noqa: I001 -- GPU pinning must run before torch imports.

from pathlib import Path

import supermimic.scripts.train  # noqa: F401 - applies torchrun pinning before CUDA imports
import torch
import tyro

from supermimic.perception.paired import restore_config, validate_student_initialization


def main(student: Path, teacher: Path, teacher_filter: Path, paired_dir: Path,
         run_name: str = "perceptive_student", log_root: Path = Path("logs"),
         num_envs: int = 4096, total_iterations: int = 320000, device: str = "cuda:0"):
    checkpoint = torch.load(student, map_location="cpu", weights_only=False)
    cfg = restore_config(checkpoint["config"])
    if cfg.stage != "student" or cfg.integration_test:
        raise ValueError("formal fine-tuning requires a distilled perceptive student")
    cfg.run_name, cfg.log_root, cfg.device = run_name, log_root, device
    cfg.data.paired_dir, cfg.data.teacher_filter = paired_dir, teacher_filter
    cfg.env.num_envs = num_envs
    cfg.schedule.total_iterations = total_iterations
    cfg.resume = None
    cfg.init_policy, cfg.init_world_model = student, teacher
    cfg.policy_trainer.backbone_lr_scale = 0.3
    validate_student_initialization(cfg, checkpoint)
    del checkpoint
    from supermimic.training.loop import Trainer

    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    tyro.cli(main)
