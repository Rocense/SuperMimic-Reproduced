"""Teacher or student training on validated paired terrain references."""

import tyro
from supermimic.scripts.train import TrainConfig


def main():
    cfg = TrainConfig(stage="teacher", run_name="terrain_teacher")
    cfg.env.num_envs = 256
    cfg.env.replay_capacity = 256
    cfg.wm_trainer.batch_size = 256
    cfg.policy_trainer.batch_size = 256
    cfg.schedule.wm_warmup_updates = 200
    cfg = tyro.cli(TrainConfig, default=cfg)
    from supermimic.training.loop import Trainer

    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
