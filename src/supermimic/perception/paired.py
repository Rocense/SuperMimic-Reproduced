"""Validated, frame-aligned raw/TCRS data and stage configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from supermimic.perception.terrain import TerrainCfg


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def terrain_config(values):
    values = dict(values)
    for key in ("scan_x", "scan_y", "scan_shape"):
        if key in values:
            values[key] = tuple(values[key])
    cfg = TerrainCfg(**values)
    cfg.validate()
    return cfg


def load_pairs(directory, names=None, limit=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "supermimic-tcrs-v1":
        raise ValueError("unsupported paired dataset format")
    records = [r for r in manifest["clips"] if r["metrics"]["accepted"]]
    if names is not None:
        wanted = set(names)
        missing = wanted - {r["name"] for r in records}
        if missing:
            raise ValueError(f"missing/rejected paired clips: {sorted(missing)}")
        records = [r for r in records if r["name"] in wanted]
    records = records[:limit] if limit is not None else records
    if not records:
        raise ValueError("paired dataset contains no accepted clips")
    raw_files, adapted_files = [], []
    for record in records:
        name = record["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("invalid clip name in paired manifest")
        raw, adapted = directory / "raw" / f"{name}.npz", directory / "adapted" / f"{name}.npz"
        for path, key in ((raw, "raw_sha256"), (adapted, "adapted_sha256")):
            if sha256(path) != record[key]:
                raise ValueError(f"paired reference changed: {path}")
        with np.load(raw) as r, np.load(adapted) as a:
            if (
                r["joint_pos"].shape != a["joint_pos"].shape
                or len(r["joint_pos"]) != record["metrics"]["frames"]
            ):
                raise ValueError(f"raw/adapted frame mismatch for {name}")
            if not np.array_equal(r["joint_names"], a["joint_names"]):
                raise ValueError(f"raw/adapted joint order mismatch for {name}")
            if not np.allclose(r["fps"], 50) or not np.allclose(a["fps"], 50):
                raise ValueError("paired references must run at 50 Hz")
        raw_files.append(str(raw))
        adapted_files.append(str(adapted))
    return raw_files, adapted_files, terrain_config(manifest["terrain"]), manifest


def configure_stage(cfg):
    if cfg.stage == "flat":
        if cfg.terrain.enabled:
            raise ValueError("use stage teacher/student with a paired TCRS dataset")
        return None
    if cfg.data.paired_dir is None:
        raise ValueError(
            "teacher/student stages require data.paired_dir from synthesize_terrain.py"
        )
    if cfg.data.library is not None or cfg.data.manifest is not None:
        raise ValueError("paired stages use their own manifest, not a packed library or CSV filter")
    raw, adapted, terrain, manifest = load_pairs(
        cfg.data.paired_dir, cfg.data.clips, cfg.data.limit
    )
    if cfg.terrain.enabled and cfg.terrain != terrain:
        raise ValueError("training terrain must match the paired dataset manifest")
    cfg.terrain = terrain
    return (adapted, None) if cfg.stage == "teacher" else (raw, adapted)


def restore_config(values):
    """Restore a saved dataclass tree, rejecting unknown fields instead of guessing."""
    from supermimic.config import TrainConfig

    def fill(obj, values):
        known = {f.name for f in fields(obj)}
        if set(values) - known:
            raise ValueError(f"unknown saved fields: {set(values) - known}")
        for name, value in values.items():
            old = getattr(obj, name)
            if is_dataclass(old):
                fill(old, value)
            else:
                setattr(obj, name, tuple(value) if isinstance(old, tuple) else value)
        return obj

    return fill(TrainConfig(), values)
