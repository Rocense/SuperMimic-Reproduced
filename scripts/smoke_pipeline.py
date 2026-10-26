"""Bounded integration test: TCRS -> teacher -> DAgger -> student BPTT."""

import copy
from pathlib import Path
import subprocess
import sys

import torch
import tyro

from supermimic.config import TrainConfig
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.perception.distillation import DistillConfig, distill
from supermimic.perception.env import goal_for_library
from supermimic.training.loop import Trainer


def main(npz_dir: Path, output: Path = Path("logs/pipeline_smoke"), compile: bool = False):
    if output.exists():
        raise FileExistsError("choose a fresh smoke output directory")
    output.mkdir(parents=True)
    pairs = output / "pairs"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("synthesize_terrain.py")),
            "--npz-dir",
            str(npz_dir),
            "--output",
            str(pairs),
            "--clips",
            "walk1_subject1",
            "--start-frame",
            "200",
            "--max-frames",
            "128",
            "--terrain.kind",
            "stairs",
            "--terrain.size",
            "6",
            "--terrain.resolution",
            "0.1",
            "--terrain.height",
            "0.025",
            "--tcrs.samples",
            "24",
            "--tcrs.sampling-iterations",
            "3",
            "--tcrs.ik-iterations",
            "30",
        ],
        check=True,
    )
    cfg = TrainConfig(stage="teacher", run_name="teacher", log_root=output)
    cfg.data.paired_dir = pairs
    cfg.env.num_envs = 8
    cfg.env.replay_capacity = 64
    cfg.env.sampling_mode = "start"
    cfg.env.obs_noise = cfg.env.reset_noise = cfg.env.domain_randomization = False
    cfg.env.action_noise_std = 0
    cfg.world_model.hidden = (64, 64)
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (64,)
    cfg.critic.hidden = (32,)
    cfg.wm_trainer.window = 2
    cfg.wm_trainer.batch_size = 16
    cfg.wm_trainer.batches_per_update = 1
    cfg.policy_trainer.batch_size = 8
    cfg.policy_trainer.horizon = 3
    cfg.policy_trainer.critic_epochs = 1
    cfg.policy_trainer.critic_minibatches = 2
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = "default" if compile else None
    cfg.schedule.total_iterations = 18
    cfg.schedule.warmup_iterations = 12
    cfg.schedule.updates_per_env_step = 1
    cfg.schedule.wm_warmup_updates = 1
    cfg.schedule.log_interval = cfg.schedule.checkpoint_interval = 18
    tr = Trainer(cfg)
    try:
        assert tr.policy.cfg.terrain_dim == 0
        assert tr.wm.terrain is not None
        tr.run()
        assert tr.num_updates == 7
    finally:
        tr.env.close()
    teacher_path = output / "teacher/ckpt_latest.pt"
    distilled = distill(
        DistillConfig(
            teacher=teacher_path,
            paired_dir=pairs,
            output=output / "distilled.pt",
            steps=4,
            num_envs=8,
            buffer_steps=4,
            batch_size=16,
            decay_steps=3,
            log_interval=4,
            obs_noise=False,
        )
    )
    scfg = copy.deepcopy(cfg)
    scfg.stage, scfg.run_name = "student", "student"
    scfg.init_policy, scfg.init_world_model = distilled, teacher_path
    st = Trainer(scfg)
    try:
        obs, _ = st.env.reset()
        assert st.policy.cfg.terrain_dim == 374
        torch.testing.assert_close(
            obs["goal"], goal_for_library(st.command, st.lib, scfg.env.anchor_obs)
        )
        assert not torch.allclose(
            obs["goal"],
            goal_for_library(st.command, st.command.tracking_library, scfg.env.anchor_obs),
        )
        torch.testing.assert_close(
            obs["terrain"], st.wm.terrain.observation(st.recorder.state_io.read())
        )
        st.run()
        assert st.policy.terrain_action_gate.abs().sum() > 0
        assert all(torch.isfinite(p).all() for p in st.policy.parameters())
        loaded, saved = load_policy_from_checkpoint(output / "student/ckpt_latest.pt", cfg.device)
        obs, _ = st.env.reset()
        torch.testing.assert_close(
            loaded(obs["policy"], obs["goal"], terrain=obs["terrain"]),
            st.policy(obs["policy"], obs["goal"], terrain=obs["terrain"]),
        )
        assert saved["_control_contract"]["joint_names"]
    finally:
        st.env.close()
    print(
        "PASS: TCRS, blind teacher, PD-aligned DAgger, raw-reference student, BPTT and checkpoint reload"
    )


if __name__ == "__main__":
    tyro.cli(main)
