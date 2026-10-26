"""Distill the adapted-reference teacher into a raw-reference student."""

import tyro
from supermimic.perception.distillation import DistillConfig, distill

if __name__ == "__main__":
    distill(tyro.cli(DistillConfig))
