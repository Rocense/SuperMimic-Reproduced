"""PMT coarse IK acceptance and teacher-conditioned reference selection.

Source: Mondo-Robotics/PMT, commit 1a92390077d329d22cbcdbf89c4a5c982003fbc8.
The public batch generator rejects >8 frames with max(root, ankle) error >200 mm.
Teacher filtering follows paper Algorithm 1; it is not a fixed rejection quota.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.kinematics.rotations import (
    projected_gravity,
    quat_apply,
    quat_inv,
    quat_mul,
    yaw_quat,
)

PMT_COMMIT = "1a92390077d329d22cbcdbf89c4a5c982003fbc8"
PMT_REPOSITORY = "https://github.com/Mondo-Robotics/PMT"


def pmt_ik_acceptance(errors_m, threshold_mm=200.0, max_bad_frames=8):
    errors = np.asarray(errors_m, dtype=np.float64)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("IK filter requires finite nonnegative per-frame errors")
    bad = int(np.count_nonzero(errors * 1000 > threshold_mm))
    return {"ik_error_max_m": float(errors.max()), "ik_bad_frame_count": bad,
            "batch_reject_ik_mm": float(threshold_mm),
            "batch_reject_ik_bad_frames": int(max_bad_frames),
            "accepted": max_bad_frames < 0 or bad <= max_bad_frames,
            "acceptance_rule": "pmt_batch", "upstream_commit": PMT_COMMIT}


@dataclass(frozen=True)
class TrackingThresholds:
    # Official stepping_stone.py + inherited tracking_env_cfg.py termination terms.
    anchor_position_m: float = 0.35
    anchor_gravity_z: float = 0.8
    end_effector_position_m: float = 0.35
    end_effectors: tuple[str, ...] = (
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    )


def tracking_errors(robot_pos, robot_quat, ref_pos, ref_quat, anchor_index, ee_indices):
    """Official XYZ anchor, gravity-z orientation, and yaw-reanchored EE errors."""
    rp, rq = robot_pos[:, anchor_index], robot_quat[:, anchor_index]
    tp, tq = ref_pos[:, anchor_index], ref_quat[:, anchor_index]
    shift = torch.cat([rp[:, :2], tp[:, 2:3]], dim=-1)
    yaw = yaw_quat(quat_mul(rq, quat_inv(tq)))
    aligned = shift[:, None] + quat_apply(
        yaw[:, None].expand(-1, ref_pos.shape[1], -1), ref_pos - tp[:, None]
    )
    return torch.stack([
        torch.linalg.vector_norm(tp - rp, dim=-1),
        (projected_gravity(tq)[:, 2] - projected_gravity(rq)[:, 2]).abs(),
        torch.linalg.vector_norm(aligned[:, ee_indices] - robot_pos[:, ee_indices], dim=-1).amax(-1),
    ], dim=-1)


def tracking_failures(errors, cfg=None):
    cfg = cfg or TrackingThresholds()
    limits = errors.new_tensor([cfg.anchor_position_m, cfg.anchor_gravity_z, cfg.end_effector_position_m])
    return (errors > limits) | ~torch.isfinite(errors)


def pmt_tracking_termination(env, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    library = getattr(cmd, "tracking_library", cmd.library)
    ref = library.bodies(cmd.frame)
    ee = [cmd.cfg.body_names.index(name) for name in TrackingThresholds().end_effectors]
    errors = tracking_errors(
        cmd.robot_body_pos_w - env.scene.env_origins[:, None], cmd.robot_body_quat_w,
        ref.pos_w, ref.quat_w, cmd.motion_anchor_body_index, ee,
    )
    # Retain pre-reset diagnostics for the benchmark. Auto-reset does not erase these.
    cmd.pmt_tracking_errors = errors.detach()
    cmd.pmt_tracking_failures = tracking_failures(errors)
    return cmd.pmt_tracking_failures.any(-1)


def write_teacher_filter(directory, teacher, results, output):
    """Bind complete per-reference results to immutable teacher and pair identities."""
    from supermimic.perception.paired import load_pairs, sha256

    directory, teacher, output = Path(directory), Path(teacher), Path(output)
    _, _, _, manifest = load_pairs(directory)
    if manifest.get("split") in ("validation", "test"):
        raise ValueError("do not prune validation/test data using teacher outcomes")
    records = {r["name"]: r for r in manifest["clips"] if r["metrics"]["accepted"]}
    if len(results) != len(records) or {r.name for r in results} != set(records):
        raise ValueError("teacher filter requires exactly one result for every candidate reference")
    if output.exists():
        raise FileExistsError("teacher selection receipts are immutable; choose a new output")
    selected = []
    for result in results:
        record = records[result.name]
        if result.frames != record["metrics"]["frames"]:
            raise ValueError("evaluation and paired reference frame counts differ")
        if result.tracking_max is None or result.failure_reasons is None:
            raise ValueError("use PMT tracking evaluation, not legacy survival-only results")
        if not math.isfinite(result.norm_return) or not all(math.isfinite(v) for v in result.tracking_max.values()):
            raise ValueError("nonfinite teacher evaluation cannot certify a dataset")
        limits = {k: v for k, v in asdict(TrackingThresholds()).items() if k != "end_effectors"}
        if set(result.tracking_max) != set(limits):
            raise ValueError("incomplete tracking-error evidence")
        keep = (not result.failed and not result.failure_reasons
                and result.steps >= result.frames - 1
                and all(result.tracking_max[k] <= v for k, v in limits.items()))
        selected.append({"name": result.name, "frames": result.frames, "steps": result.steps,
                         "keep": keep, "reason": "completed" if keep else "teacher_tracking_failure",
                         "failure_reasons": result.failure_reasons,
                         "tracking_max": result.tracking_max, "norm_return": result.norm_return,
                         "raw_sha256": record["raw_sha256"],
                         "adapted_sha256": record["adapted_sha256"]})
    receipt = {
        "format": "supermimic-pmt-teacher-filter-v1", "upstream_commit": PMT_COMMIT,
        "teacher_sha256": sha256(teacher), "teacher_path": str(teacher.resolve()),
        "paired_manifest_sha256": sha256(directory / "manifest.json"),
        "source_coarse_filter": manifest.get("tcrs", {}).get("acceptance_rule", "legacy"),
        "tracking_thresholds": asdict(TrackingThresholds()),
        "evaluation_protocol": "one deterministic full-clip rollout per reference; all candidates counted",
        "selection_rule": "no tracking failure before clip end; no return cutoff or rejection quota",
        "clips": sorted(selected, key=lambda row: row["name"]),
        "candidate_count": len(selected), "kept_count": sum(r["keep"] for r in selected),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def selected_reference_names(directory, receipt_path, teacher=None):
    from supermimic.perception.paired import sha256

    directory = Path(directory)
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("format") != "supermimic-pmt-teacher-filter-v1":
        raise ValueError("missing PMT teacher selection receipt")
    if receipt["paired_manifest_sha256"] != sha256(directory / "manifest.json"):
        raise ValueError("teacher filter belongs to a different paired dataset")
    if teacher is not None and receipt["teacher_sha256"] != sha256(teacher):
        raise ValueError("teacher filter belongs to a different teacher checkpoint")
    expected = json.loads(json.dumps(asdict(TrackingThresholds())))
    if receipt["tracking_thresholds"] != expected:
        raise ValueError("teacher filter uses different tracking criteria")
    manifest = json.loads((directory / "manifest.json").read_text())
    records = {r["name"]: r for r in manifest["clips"] if r["metrics"]["accepted"]}
    rows = receipt["clips"]
    if len(rows) != len(records) or {r["name"] for r in rows} != set(records):
        raise ValueError("teacher filter has missing or duplicate reference outcomes")
    for row in rows:
        record = records[row["name"]]
        if any(row[k] != record[k] for k in ("raw_sha256", "adapted_sha256")):
            raise ValueError("teacher filter reference fingerprint mismatch")
        limits = {k: v for k, v in expected.items() if k != "end_effectors"}
        if row["frames"] != record["metrics"]["frames"] or set(row["tracking_max"]) != set(limits):
            raise ValueError("teacher filter has incomplete tracking evidence")
        keep = (not row["failure_reasons"] and row["steps"] >= row["frames"] - 1
                and all(math.isfinite(row["tracking_max"][k]) and row["tracking_max"][k] <= v
                        for k, v in limits.items()))
        if row["keep"] != keep:
            raise ValueError("teacher selection contradicts its tracking evidence")
    names = [r["name"] for r in rows if r["keep"]]
    if not names:
        raise ValueError("teacher cannot track any candidate references; do not start distillation")
    return names
