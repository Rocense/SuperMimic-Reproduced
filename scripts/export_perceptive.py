"""Export a standalone three-input TorchScript actor and its input contract."""

import json
from pathlib import Path

import torch
import tyro

from supermimic.eval.benchmark import load_policy_from_checkpoint


class Actor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, proprio_history, raw_motion_goal, terrain_scan):
        return self.policy(proprio_history, raw_motion_goal, terrain=terrain_scan)


def main(ckpt: Path, out: Path):
    policy, cfg = load_policy_from_checkpoint(ckpt, "cpu")
    if not policy.cfg.terrain_dim:
        raise ValueError("checkpoint has no terrain input")
    actor = Actor(policy).eval()
    shapes = (93 * policy.cfg.history, policy.cfg.goal_dim, policy.cfg.terrain_dim)
    example = tuple(torch.zeros(1, n) for n in shapes)
    with torch.no_grad():
        traced = torch.jit.trace(actor, example)
        check = tuple(torch.randn(2, n) for n in shapes)
        torch.testing.assert_close(traced(*check), actor(*check))
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": dict(zip(("proprio_history", "raw_motion_goal", "terrain_scan"), shapes)),
                "output": "29 clean normalized joint-position actions; apply the original G1_ACTION_SCALE and default offsets",
                "policy": cfg["policy"],
                "terrain": cfg["terrain"],
                "control_contract": cfg.get("_control_contract"),
                "terrain_layout": "x-major, then y; flattened heights followed by flattened validity mask",
                "height_units": "clip((world_height - root_z + nominal_root_height), +/-scan_clip) / scan_clip; invalid heights zero",
                "frame": "robot root XY and root yaw; gravity-aligned grid, not full torso orientation",
                "note": "Supply an onboard elevation map with this contract; no mapping or camera model is exported.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported and checked {out}")


if __name__ == "__main__":
    tyro.cli(main)
