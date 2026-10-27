"""Generate raw/adapted paired NPZs and a checked TCRS manifest offline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Literal

import numpy as np
import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.perception.paired import sha256
from supermimic.perception.tcrs import Synthesizer, TCRSCfg
from supermimic.perception.terrain import TerrainCfg, TerrainField


@dataclass
class Config:
    npz_dir: Path
    output: Path
    split: Literal["train", "validation", "test"] = "train"
    clips: list[str] | None = None
    max_frames: int | None = None  # Shorten both sides identically for integration tests.
    start_frame: int = 0
    terrain: TerrainCfg = field(default_factory=lambda: TerrainCfg(enabled=True))
    tcrs: TCRSCfg = field(default_factory=TCRSCfg)


def generate(cfg: Config):
    files = discover_npz(cfg.npz_dir, names=cfg.clips)
    if not files:
        raise FileNotFoundError("no source motion clips")
    if cfg.output.exists() and any(cfg.output.iterdir()):
        raise FileExistsError("choose an empty output directory; existing datasets are immutable")
    cfg.terrain.enabled = True
    cfg.output.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "adapted", "rejected"):
        (cfg.output / sub).mkdir()
    synth = Synthesizer(TerrainField(cfg.terrain), cfg.tcrs)
    manifest = {
        "format": "supermimic-tcrs-v1",
        "split": cfg.split,
        "terrain": asdict(cfg.terrain),
        "tcrs": asdict(cfg.tcrs),
        "clips": [],
    }
    frame_keys = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    for path in files:
        with np.load(path) as source:
            raw = {k: source[k] for k in source.files}
        if cfg.start_frame < 0 or (cfg.max_frames is not None and cfg.max_frames < 3):
            raise ValueError("start_frame must be nonnegative and max_frames at least three")
        end = None if cfg.max_frames is None else cfg.start_frame + cfg.max_frames
        for key in frame_keys:
            raw[key] = raw[key][cfg.start_frame : end]
        record = {"name": path.stem, "source_sha256": sha256(path),
                  "source": path.name, "start_frame": cfg.start_frame}
        try:
            adapted, metrics = synth.synthesize(raw)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            # A failed synthesis attempt is an explicit rejected candidate, not
            # an unreported omission or a reason to discard the rest of the corpus.
            adapted = dict(raw, rejection_error=np.array(str(exc)))
            metrics = {"accepted": False, "frames": len(raw["joint_pos"]),
                       "reason": "synthesis_failure", "error": str(exc)}
        record["metrics"] = metrics
        if metrics["accepted"]:
            raw_path = cfg.output / "raw" / path.name
            adapted_path = cfg.output / "adapted" / path.name
            np.savez_compressed(raw_path, **raw)
            np.savez_compressed(adapted_path, **adapted)
            record.update(raw_sha256=sha256(raw_path), adapted_sha256=sha256(adapted_path))
        else:
            # Retain diagnostics for rejected references, but never train on them implicitly.
            np.savez_compressed(cfg.output / "rejected" / path.name, **adapted)
        manifest["clips"].append(record)
        manifest["candidate_count"] = len(manifest["clips"])
        manifest["accepted_count"] = sum(r["metrics"]["accepted"] for r in manifest["clips"])
        (cfg.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(path.stem, json.dumps(metrics), flush=True)
    if not any(r["metrics"]["accepted"] for r in manifest["clips"]):
        raise RuntimeError(
            "all TCRS references failed quality checks; inspect rejected diagnostics"
        )
    return cfg.output


if __name__ == "__main__":
    generate(tyro.cli(Config))
