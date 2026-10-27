"""Teacher or student training on validated paired terrain references."""

import tyro
from supermimic.scripts.train import TrainConfig


def main():
    cfg = TrainConfig(stage="teacher", run_name="terrain_teacher")
    cfg = tyro.cli(TrainConfig, default=cfg)
    from supermimic.training.loop import Trainer

    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
