"""Evaluate a frozen teacher on every TCRS reference and save a checked selection receipt."""

from pathlib import Path

import tyro

from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark
from supermimic.perception.filtering import write_teacher_filter
from supermimic.perception.paired import load_pairs, sha256, terrain_config


def main(teacher: Path, paired_dir: Path, output: Path, num_envs: int = 128, device: str = "cuda:0"):
    if output.exists():
        raise FileExistsError(output)
    teacher_hash = sha256(teacher)
    manifest_hash = sha256(paired_dir / "manifest.json")
    _, adapted, terrain, manifest = load_pairs(paired_dir)
    if manifest.get("split") in ("validation", "test"):
        raise ValueError("validation/test references must stay unfiltered")
    policy, cfg = load_policy_from_checkpoint(teacher, device)
    if cfg.get("stage") != "teacher" or policy.cfg.terrain_dim:
        raise ValueError("selection requires the frozen blind adapted-reference teacher")
    if terrain != terrain_config(cfg["terrain"]):
        raise ValueError("teacher terrain differs from the reference terrain")
    results = run_benchmark(
        policy, adapted, device=device, num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False),
        terrain=terrain, reference_is_adapted=True, pmt_tracking_evaluation=True,
    )
    if teacher_hash != sha256(teacher) or manifest_hash != sha256(paired_dir / "manifest.json"):
        raise ValueError("teacher or paired manifest changed during evaluation; discard the incomplete selection")
    receipt = write_teacher_filter(paired_dir, teacher, results, output)
    print(f"Kept {receipt['kept_count']}/{receipt['candidate_count']} references; receipt: {output}")


if __name__ == "__main__":
    tyro.cli(main)
