"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
"""

from __future__ import annotations

from pathlib import Path

import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import (
    load_policy_from_checkpoint,
    run_benchmark,
    save_results,
    summarize,
)
from supermimic.perception.paired import load_pairs


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    clip_list: Path | None = None,
    manifest: Path | None = None,
    num_envs: int = 1024,
    device: str = "cuda:0",
    out: Path | None = None,
    paired_dir: Path | None = None,
    map_ablation: str = "none",
    library: Path | None = None,
    pmt_tracking_evaluation: bool = False,
) -> None:
    if clip_list is not None:
        with clip_list.open() as stream:
            clips = [line.strip() for line in stream if line.strip()]
    if library is not None:
        if clips is not None or manifest is not None:
            raise ValueError("packed library evaluation covers all clips; do not pass an NPZ filter")
        files = [str(library)]
    else:
        files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    terrain, target_files = None, None
    is_adapted = cfg.get("stage", "flat") in ("teacher", "student")
    if is_adapted:
        if library is not None:
            raise ValueError("terrain evaluation requires paired references, not a raw packed library")
        paired_dir = paired_dir or cfg["data"].get("paired_dir")
        if paired_dir is None:
            raise ValueError("terrain evaluation requires a paired dataset")
        raw, adapted, terrain, _ = load_pairs(paired_dir, clips)
        files = adapted if cfg["stage"] == "teacher" else raw
        target_files = adapted if cfg["stage"] == "student" else None
    results = run_benchmark(
        policy,
        files,
        device=device,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        anchor_obs=cfg["env"].get("anchor_obs", False),
        terrain=terrain,
        target_npz_files=target_files,
        reference_is_adapted=is_adapted,
        map_ablation=map_ablation,
        pmt_tracking_evaluation=pmt_tracking_evaluation,
    )
    s = summarize(results)
    print(
        f"\n{s['clips']} clips | failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f}"
    )
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(
            f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}"
        )
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}_{map_ablation}.json"
    save_results(results, out)
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
