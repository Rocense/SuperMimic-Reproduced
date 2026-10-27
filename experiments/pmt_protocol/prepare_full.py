"""Resumable full-corpus TCRS generation from original BONES motion CSVs.

Every source frame is kept. One pairing per motion is the explicit first-pass
default; this is not the paper's approximately 100 terrain pairings per motion.
No neural-network training is performed by this program.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import tarfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    ResampledMotion,
    forward_kinematics,
    load_csv,
    quat_mul,
    resample_motion,
)
from supermimic.perception.paired import sha256
from supermimic.perception.tcrs import Synthesizer, TCRSCfg
from supermimic.perception.terrain import TerrainCfg, TerrainField

_STOP = False
_PACK = _SYNTH = _PLAN = _ROOT = _CSV_INDEX = None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def copy_verified(source, destination, expected_sha, progress=None):
    """Snapshot input on the output disk; never replace an existing different input."""
    if destination.exists():
        if sha256(destination) != expected_sha:
            raise ValueError(f"input snapshot changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    copied = reported = 0
    with source.open("rb") as src, temporary.open("wb") as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            if _STOP:
                raise InterruptedError("stopped while copying source snapshot")
            dst.write(block)
            digest.update(block)
            copied += len(block)
            if progress is not None and copied - reported >= 512 * 1024**2:
                progress(source.name, copied, source.stat().st_size)
                reported = copied
        dst.flush()
        os.fsync(dst.fileno())
    if digest.hexdigest() != expected_sha:
        raise ValueError("source library differs from the audited full corpus")
    temporary.replace(destination)
    if progress is not None:
        progress(source.name, copied, source.stat().st_size)


def extract_selected_csv(root, names, archive_sha, progress):
    """Extract only selected regular CSVs to numeric paths; never trust tar paths."""
    destination = root / "inputs/bones_csv"
    destination.mkdir(parents=True, exist_ok=True)
    index_path = root / "inputs/bones_csv_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["archive_sha256"] != archive_sha or set(index["files"]) != set(names):
            raise ValueError("CSV snapshot index differs from the full selected corpus")
        for row in index["files"].values():
            if (destination / row["file"]).stat().st_size != row["bytes"]:
                raise ValueError("CSV snapshot size changed")
        return
    expected = {name: i for i, name in enumerate(names)}
    if len(expected) != len(names):
        raise ValueError("duplicate source motion names")
    rows = {}
    total = 0
    with tarfile.open(root / "inputs/g1.tar.gz", mode="r|gz") as archive:
        for member in archive:
            if _STOP:
                raise InterruptedError("stopped while extracting source CSVs")
            name = Path(member.name).stem
            if not member.isfile() or not member.name.endswith(".csv") or name not in expected:
                continue
            if name in rows:
                raise ValueError(f"duplicate archive motion: {name}")
            path = destination / f"{expected[name]:06d}.csv"
            temporary = path.with_suffix(".partial")
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            with stream, temporary.open("wb") as out:
                for block in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
            temporary.replace(path)
            rows[name] = {"file": path.name, "bytes": member.size, "sha256": digest.hexdigest()}
            total += member.size
            if len(rows) % 1000 == 0:
                progress(len(rows), len(expected), total)
    if set(rows) != set(expected):
        raise ValueError(f"archive lacks {len(set(expected) - set(rows))} selected motions")
    atomic_json(index_path, {"archive_sha256": archive_sha, "files": rows})
    progress(len(rows), len(expected), total)


def placement(source_name, pairing, seed):
    digest = hashlib.sha256(f"{seed}:{source_name}:{pairing}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    # Both sides of the stair pyramid, with varied headings and distance to edges.
    x = float(rng.uniform(0.9, 2.8) * rng.choice([-1, 1]))
    return x, float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-np.pi, np.pi))


def packed_motion(ref, model, transform):
    """Recover exactly the G1 joint order, world velocity units, and full timeline."""
    r = np.asarray(ref, dtype=np.float64).copy()
    if r.ndim != 2 or r.shape[1] != 71 or len(r) < 3 or not np.isfinite(r).all():
        raise ValueError("invalid compact reference")
    x, y, yaw = transform
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    root = r[:, :3].copy()
    origin = root[0, :2].copy()
    root[:, :2] -= origin
    root = root @ rotation.T
    root[:, :2] += [x, y]
    q = r[:, 3:7] / np.linalg.norm(r[:, 3:7], axis=1, keepdims=True)
    q = quat_mul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), q)
    motion = ResampledMotion(50, root, q, r[:, 7:36], r[:, 36:39] @ rotation.T,
                             r[:, 39:42] @ rotation.T, r[:, 42:71])
    bodies = forward_kinematics(model, motion)
    return {"fps": np.array([50], dtype=np.float32),
            "input_fps": np.array([50], dtype=np.float32),
            "joint_pos": motion.joint_pos.astype(np.float32),
            "joint_vel": motion.joint_vel.astype(np.float32),
            "body_pos_w": bodies.pos_w.astype(np.float32),
            "body_quat_w": bodies.quat_w.astype(np.float32),
            "body_lin_vel_w": bodies.lin_vel_w.astype(np.float32),
            "body_ang_vel_w": bodies.ang_vel_w.astype(np.float32),
            "joint_names": np.array(G1_JOINT_NAMES), "body_names": np.array(bodies.body_names)}, origin


def npz_ref(path):
    with np.load(path) as source:
        names = source["body_names"].astype(str).tolist()
        pelvis = names.index("pelvis")
        if float(source["fps"].reshape(-1)[0]) != 50:
            raise ValueError("validation motions must be 50 Hz")
        if tuple(source["joint_names"].astype(str)) != G1_JOINT_NAMES:
            raise ValueError("validation joint ordering differs from G1")
        return np.concatenate([source["body_pos_w"][:, pelvis], source["body_quat_w"][:, pelvis],
                               source["joint_pos"], source["body_lin_vel_w"][:, pelvis],
                               source["body_ang_vel_w"][:, pelvis], source["joint_vel"]], axis=1)


def initialize(root):
    global _PACK, _SYNTH, _PLAN, _ROOT, _CSV_INDEX
    torch.set_num_threads(1)
    _ROOT = Path(root)
    _PLAN = json.loads((_ROOT / "plan.json").read_text())
    _PACK = torch.load(_ROOT / "inputs/bones.pt", map_location="cpu", weights_only=True, mmap=True)
    _CSV_INDEX = json.loads((_ROOT / "inputs/bones_csv_index.json").read_text())["files"]
    _SYNTH = Synthesizer(TerrainField(TerrainCfg(**_PLAN["terrain"])), TCRSCfg(**_PLAN["tcrs"]))


def one(job):
    split, index, pairing = job
    name = f"{split}_{index:06d}_p{pairing:03d}"
    destination = _ROOT / split
    record_path = destination / "records" / f"{name}.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        for key in ("raw", "adapted"):
            if record["metrics"]["accepted"] and sha256(destination / key / f"{name}.npz") != record[f"{key}_sha256"]:
                raise ValueError(f"completed output changed: {name}")
        return record
    if split == "train":
        source_name = _PACK["names"][index]
        start, frames = int(_PACK["starts"][index]), int(_PACK["lengths"][index])
        source = _CSV_INDEX[source_name]
        path = _ROOT / "inputs/bones_csv" / source["file"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"source CSV changed: {source_name}")
        motion = resample_motion(load_csv(path), input_fps=120, output_fps=50)
        ref = np.concatenate([motion.root_pos, motion.root_quat, motion.joint_pos,
                              motion.root_lin_vel_w, motion.root_ang_vel_w, motion.joint_vel], axis=1)
        if len(ref) != frames:
            raise ValueError(f"source resampling frame count differs from audited library: {source_name}")
        source_hash = source["sha256"]
    else:
        spec = _PLAN["validation_files"][index]
        source_name, source_hash = spec["name"], spec["sha256"]
        ref = npz_ref(_ROOT / "inputs/lafan1" / f"{source_name}.npz")
        start, frames = 0, len(ref)
    transform = placement(f"{split}:{source_name}", pairing, _PLAN["seed"])
    raw, origin = packed_motion(ref, _SYNTH.model, transform)
    raw["source"] = np.array(source_name)
    raw["input_fps"] = np.array([120 if split == "train" else 50], dtype=np.float32)
    _SYNTH.rng.manual_seed(_PLAN["seed"] + index * _PLAN["pairings_per_motion"] + pairing)
    try:
        adapted, metrics = _SYNTH.synthesize(raw)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        adapted = dict(raw, rejection_error=np.array(str(exc)))
        metrics = {"accepted": False, "frames": frames, "reason": "synthesis_failure", "error": str(exc)}
    record = {"name": name, "source_name": source_name, "source_index": index,
              "source_sha256": source_hash, "source_frame_start": start,
              "frames": frames, "pairing": pairing, "split": split,
              "source_origin_xy": origin.tolist(), "placement_xy_yaw": list(transform),
              "metrics": metrics}
    outputs = (("raw", raw), ("adapted", adapted)) if metrics["accepted"] else (("rejected", adapted),)
    for kind, values in outputs:
        path = destination / kind / f"{name}.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **values)
        temporary.replace(path)
        if metrics["accepted"]:
            record[f"{kind}_sha256"] = sha256(path)
    atomic_json(record_path, record)
    return record


def main(library, inventory, archive, archive_sha256, output, workers=8,
         pairings_per_motion=1, reserve_gib=150, seed=20261028):
    global _STOP
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if workers < 1 or pairings_per_motion < 1:
        raise ValueError("worker and pairing counts must be positive")
    source = json.loads(inventory.read_text())
    total_frames = source["bones"]["frames"] + source["lafan1"]["frames"]
    # Include an uncompressed CSV allowance (120 Hz, <=1000 bytes/input frame),
    # immutable input snapshots, full paired NPZ outputs and a separate free reserve.
    estimated_bytes = (total_frames * pairings_per_motion * 4000 + library.stat().st_size
                       + archive.stat().st_size + source["bones"]["frames"] * 2.4 * 1000)
    if not (output / "plan.json").exists() and shutil.disk_usage(output).free < estimated_bytes + reserve_gib * 2**30:
        raise OSError("not enough space for the full requested pairing count plus reserve; no partial corpus will be substituted")
    terrain = TerrainCfg(enabled=True, kind="stairs", stairs_surface="boxes", height=0.15,
                         size=128, resolution=0.1, stair_width=0.5, stair_count=4)
    plan = {"format": "supermimic-full-tcrs-v1", "seed": seed,
            "bones_sha256": source["bones"]["library"]["sha256"],
            "archive_sha256": archive_sha256,
            "bones_clips": source["bones"]["kept_clips"], "bones_frames": source["bones"]["frames"],
            "validation_files": source["lafan1"]["files"],
            "pairings_per_motion": pairings_per_motion, "terrain": asdict(terrain),
            "tcrs": asdict(TCRSCfg()), "generator_sha256": sha256(Path(__file__)),
            "validation_scope": "different motion source, same fixed terrain geometry",
            "note": "All full-length BONES train clips plus all 40 source-held-out LAFAN1 validation clips; no motion/frame slicing. BONES poses are regenerated from original 120 Hz CSVs in full precision; the fp16 library is only the audited selection/length index. First-pass one terrain pairing is not the paper's approximately 100 pairings."}
    plan = json.loads(json.dumps(plan))
    if (output / "plan.json").exists() and json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("resume configuration/source differs from the immutable plan")
    atomic_json(output / "plan.json", plan)

    def stop(signum, frame):
        global _STOP
        _STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.time()
    state = {"phase": "snapshotting_inputs", "pid": os.getpid(), "pgid": os.getpgrp(),
             "workers": workers, "planned_pairs": (plan["bones_clips"] + len(plan["validation_files"])) * pairings_per_motion,
             "planned_source_frames": total_frames, "pairings_per_motion": pairings_per_motion,
             "estimated_output_gib_upper": estimated_bytes / 2**30, "started_unix": started}

    def publish():
        state.update(updated_unix=time.time(), disk_free_gib=shutil.disk_usage(output).free / 2**30)
        atomic_json(output / "status.json", state)

    publish()
    def copy_progress(name, copied, total):
        state.update(snapshot_file=name, snapshot_bytes=copied, snapshot_total_bytes=total)
        publish()

    copy_verified(library, output / "inputs/bones.pt", plan["bones_sha256"], copy_progress)
    copy_verified(archive, output / "inputs/g1.tar.gz", plan["archive_sha256"], copy_progress)
    for row in plan["validation_files"]:
        copy_verified(Path(row["path"]), output / "inputs/lafan1" / f"{row['name']}.npz", row["sha256"])
    metadata = torch.load(output / "inputs/bones.pt", map_location="cpu", weights_only=True, mmap=True)
    if len(metadata["names"]) != plan["bones_clips"] or sum(metadata["lengths"]) != plan["bones_frames"]:
        raise ValueError("full-library membership/frames differ from audited inventory")
    state["phase"] = "extracting_full_precision_csv"
    def extraction_progress(done, total, size):
        state.update(extracted_csvs=done, expected_csvs=total, extracted_csv_bytes=size)
        publish()

    extract_selected_csv(output, metadata["names"], plan["archive_sha256"], extraction_progress)
    del metadata
    jobs = []
    for split, count in (("train", plan["bones_clips"]), ("validation", len(plan["validation_files"]))):
        for sub in ("raw", "adapted", "rejected", "records"):
            (output / split / sub).mkdir(parents=True, exist_ok=True)
        jobs.extend((split, index, pairing) for index in range(count) for pairing in range(pairings_per_motion))
    completed = {"train": [], "validation": []}
    remaining = []
    for job in jobs:
        split, index, pairing = job
        path = output / split / "records" / f"{split}_{index:06d}_p{pairing:03d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["name"] != path.stem or record["source_index"] != index or record["pairing"] != pairing:
                raise ValueError(f"completed record identity changed: {path}")
            if record["metrics"]["accepted"]:
                for kind in ("raw", "adapted"):
                    if sha256(output / split / kind / f"{path.stem}.npz") != record[f"{kind}_sha256"]:
                        raise ValueError(f"completed paired data changed: {path}")
            completed[split].append(record)
        else:
            remaining.append(job)
    state["phase"] = "generating_tcrs"
    iterator = iter(remaining)
    exhausted = False
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(output),)) as pool:
        pending = set()
        while pending or not exhausted:
            if (output / "stop.request").exists() or shutil.disk_usage(output).free < reserve_gib * 2**30:
                _STOP = True
                state["stop_reason"] = "stop_request_or_disk_reserve"
            if _STOP:
                exhausted = True
            while not exhausted and len(pending) < workers:
                job = next(iterator, None)
                if job is None:
                    exhausted = True
                    break
                pending.add(pool.submit(one, job))
            done, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                completed[record["split"]].append(record)
                print(json.dumps({k: record[k] for k in ("name", "source_name", "frames", "metrics")}), flush=True)
            for split, rows in completed.items():
                state[split] = {"completed_pairs": len(rows), "accepted_pairs": sum(r["metrics"]["accepted"] for r in rows),
                                "processed_frames": sum(r["frames"] for r in rows),
                                "accepted_frames": sum(r["frames"] for r in rows if r["metrics"]["accepted"])}
            state["phase"] = "stopping" if _STOP else "generating_tcrs"
            publish()
    if _STOP:
        state["phase"] = "stopped_resumable"
        publish()
        return
    for split, rows in completed.items():
        manifest = {"format": "supermimic-tcrs-v1", "split": split,
                    "terrain": plan["terrain"], "tcrs": plan["tcrs"],
                    "candidate_count": len(rows), "accepted_count": sum(r["metrics"]["accepted"] for r in rows),
                    "clips": sorted(rows, key=lambda r: r["name"])}
        atomic_json(output / split / "manifest.json", manifest)
    state["ready_for_teacher"] = all(state[split]["accepted_pairs"] > 0 for split in completed)
    state["phase"] = "tcrs_complete_teacher_not_started" if state["ready_for_teacher"] else "tcrs_complete_insufficient_references"
    publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pairings-per-motion", type=int, default=1)
    parser.add_argument("--reserve-gib", type=float, default=150)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    try:
        main(args.library, args.inventory, args.archive, args.archive_sha256, args.output,
             args.workers, args.pairings_per_motion,
             args.reserve_gib, args.seed)
    except BlockingIOError:
        raise SystemExit("TCRS preparation is already running for this output")
    except BaseException as exc:
        status = args.output / "status.json"
        state = json.loads(status.read_text()) if status.exists() else {}
        state.update(phase="failed", error=str(exc), resumable=True, updated_unix=time.time())
        atomic_json(status, state)
        traceback.print_exc()
        raise
