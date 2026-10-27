"""Audit and link the complete existing corpus without duplicating its large files.

This prepares flat tracking inputs. It does not claim that TCRS pairs or a
perceptive student have been generated from the compact BONES library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from supermimic.config import TrainConfig


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def link(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source:
            raise FileExistsError(f"refusing to replace {target}")
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def prepare(source: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    source, output = source.resolve(), output.resolve()
    library = source / "library/bones_seed_compact.pt"
    manifest = source / "library/bones_seed_compact.manifest.csv"
    receipt = source / "library/bones_seed_compact.ready.json"
    ready = json.loads(receipt.read_text())
    if ready["status"] != "ready" or ready["fps"] != 50:
        raise ValueError("source library is not ready at the required 50 Hz")
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    kept = {r["name"]: int(r["frames"]) for r in rows
            if r["keep"] in ("True", "true", "1")}
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("source manifest contains duplicate names")

    pack = torch.load(library, map_location="cpu", weights_only=True, mmap=True)
    ref = pack["ref"]
    lengths = [int(n) for n in pack["lengths"]]
    starts = [int(n) for n in pack["starts"]]
    names = pack["names"]
    if (len(names) != len(set(names)) or len(names) != len(lengths)
            or len(names) != len(starts) or dict(zip(names, lengths)) != kept):
        raise ValueError("packed clips differ from the complete keep manifest")
    if any(n < 3 for n in lengths) or starts != np.cumsum([0] + lengths[:-1]).tolist():
        raise ValueError("invalid or truncated packed motion timeline")
    if ref.dtype != torch.float16 or tuple(ref.shape) != (sum(lengths), 71):
        raise ValueError("packed reference shape/dtype mismatch")
    if (len(rows) != ready["counts"]["raw"] or len(names) != ready["counts"]["kept"]
            or sum(lengths) != ready["frames"] or library.stat().st_size != ready["bytes"]):
        raise ValueError("packed corpus differs from its preparation receipt")
    for start in range(0, len(ref), 100_000):
        block = ref[start:start + 100_000]
        if not bool(torch.isfinite(block).all()):
            raise ValueError(f"nonfinite reference state near frame {start}")
        norm = torch.linalg.vector_norm(block[:, 3:7].float(), dim=-1)
        if not bool(((norm - 1).abs() < 0.002).all()):
            raise ValueError(f"invalid wxyz quaternion near frame {start}")
    print(f"Validated all {len(names):,} BONES clips / {len(ref):,} frames", flush=True)

    lafan = []
    for path in sorted((source / "npz/lafan1").glob("*.npz")):
        with np.load(path) as clip:
            frames = len(clip["joint_pos"])
            fps = float(clip["fps"].item())
            if fps != 50 or frames < 3:
                raise ValueError(f"invalid LAFAN1 timeline: {path}")
            lafan.append({"name": path.stem, "frames": frames, "fps": fps,
                          **fingerprint(path)})
    if len(lafan) != 40:
        raise ValueError(f"expected all 40 G1 LAFAN1 clips, found {len(lafan)}")
    print("Validated all 40 full-length LAFAN1 files; hashing BONES library", flush=True)

    report = {
        "status": "full_raw_corpus_ready",
        "training_started": False,
        "bones": {"raw_clips": len(rows), "kept_clips": len(names),
                  "excluded_clips": len(rows) - len(names), "frames": len(ref),
                  "hours": len(ref) / 50 / 3600, "fps": 50,
                  "filter_reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
                  "library": fingerprint(library), "manifest": fingerprint(manifest),
                  "receipt": fingerprint(receipt)},
        "lafan1": {"clips": len(lafan), "frames": sum(r["frames"] for r in lafan),
                   "hours": sum(r["frames"] for r in lafan) / 50 / 3600,
                   "files": lafan},
        "sampling": "All retained source frames; no start_frame/max_frames/clips/limit slicing.",
        "split": "BONES full filtered training corpus; all 40 LAFAN1 clips are source-held-out evaluation for that baseline.",
        "limitations": [
            "Filtering approximates SuperMimic Appendix E; the actual count differs from the paper's 129514.",
            "TCRS teacher/student stages still require validated paired NPZs; this receipt covers raw full-corpus access only.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for path in (library, manifest, receipt):
        link(path, output / path.name)
    link(source / "npz/lafan1", output / "lafan1")
    cfg = TrainConfig(run_name="bones_full_flat", stage="flat")
    cfg.data.library = output / library.name
    cfg.data.npz_dir = output / "lafan1"
    cfg.env.compact_library = True
    # Keep the original network/optimizer sizes. Fewer archived checkpoints limit
    # disk growth; this does not change optimizer steps or data coverage.
    cfg.schedule.checkpoint_interval = 20_000
    (output / "bones_flat_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
    cfg.run_name = "lafan1_full_flat"
    cfg.data.library = None
    (output / "lafan1_flat_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="existing data/ directory")
    parser.add_argument("--output", type=Path, default=Path("data/full_corpus"))
    args = parser.parse_args()
    result = prepare(args.source, args.output)
    print(json.dumps({k: {a: b for a, b in result[k].items() if a not in ("files", "filter_reasons")}
                      for k in ("bones", "lafan1")}, indent=2), flush=True)
