"""Run the audited, unsliced baseline; CPU preflight is the default."""

import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path

# Apply the existing torchrun GPU/CPU pinning before importing CUDA-capable modules.
import supermimic.scripts.train  # noqa: F401
from supermimic.perception.paired import restore_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/full_corpus/bones_flat_config.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true", help="start GPU training after preflight")
    args = parser.parse_args()
    cfg = restore_config(json.loads(args.config.read_text()))
    if cfg.stage != "flat" or cfg.data.clips or cfg.data.limit is not None or cfg.data.manifest:
        raise ValueError("full-corpus baseline cannot silently select a motion subset")
    if cfg.data.paired_dir or cfg.terrain.enabled:
        raise ValueError("raw-corpus readiness is not TCRS-pair readiness")
    report = json.loads((args.config.parent / "inventory.json").read_text())
    if report["status"] != "full_raw_corpus_ready":
        raise ValueError("run full-corpus preparation first")
    corpus = "bones" if cfg.data.library else "lafan1"
    if cfg.data.library:
        expected = report["bones"]["library"]
        path = Path(cfg.data.library)
        if str(path.resolve()) != expected["path"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("the audited packed library has been replaced")
    else:
        expected = {r["name"] for r in report["lafan1"]["files"]}
        if {p.stem for p in Path(cfg.data.npz_dir).glob("*.npz")} != expected:
            raise ValueError("LAFAN1 file membership changed after the full audit")
    cfg.device = args.device
    cfg.run_name = args.run_name or cfg.run_name
    cfg.resume = args.resume
    print(json.dumps({"corpus": corpus, "frames": report[corpus]["frames"],
                      "hours": report[corpus]["hours"], "device": cfg.device,
                      "execute": args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    if (int(os.environ.get("RANK", "0")) == 0
            and (Path(cfg.log_root) / cfg.run_name).exists() and args.resume is None):
        raise FileExistsError("choose a fresh run name or explicitly resume")
    Path(cfg.log_root).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cfg.log_root).free < 8 * 1024**3:
        raise OSError("full-size checkpoints need at least 8 GiB free before starting")
    import torch

    from supermimic.training.loop import Trainer

    class FullCorpusTrainer(Trainer):
        def __init__(self, config):
            self.stop_requested = False
            self.stop_reason = None
            self.started = time.time()
            self.first_update_saved = False
            super().__init__(config)

        def write_status(self, phase, **extra):
            if not self.is_main:
                return
            status = {"phase": phase, "iteration": self.iteration,
                      "updates": self.num_updates, "world_size": self.world,
                      "corpus_clips": self.lib.num_clips, "corpus_frames": self.lib.num_frames,
                      "updated_unix": time.time(), "pid": os.getpid(),
                      "disk_free_gib": shutil.disk_usage(self.log_dir).free / 2**30, **extra}
            temporary = self.log_dir / ".status.tmp.json"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(self.log_dir / "status.json")

        def act(self, obs):
            if self.iteration and self.iteration % self.cfg.schedule.log_interval == 0:
                self.write_status("training")
                if self.num_updates and not self.first_update_saved:
                    self.save()
                    self.first_update_saved = True
            return super().act(obs)

        def should_stop(self):
            if self.stop_requested or (self.log_dir / "stop.request").exists():
                self.stop_reason = "stop_requested"
            elif self.is_main and shutil.disk_usage(self.log_dir).free < 5 * 1024**3:
                self.stop_reason = "disk_reserve"
            return self.stop_reason is not None

        def save(self, name="ckpt_latest.pt"):
            if not self.is_main:
                return
            for model in (self.policy, self.wm, self.critic):
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("nonfinite parameters; keeping the previous checkpoint")
            temporary = f".{name}.tmp.pt"
            super().save(temporary)
            (self.log_dir / temporary).replace(self.log_dir / name)
            self.write_status("training", checkpoint=str(self.log_dir / name))

    trainer = FullCorpusTrainer(cfg)
    def stop(signum, frame):
        trainer.stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        trainer.write_status("initialized")
        if args.resume is None:
            trainer.save("ckpt_initial.pt")
        trainer.run()
        trainer.write_status("completed" if trainer.iteration >= cfg.schedule.total_iterations
                             else "stopped", reason=trainer.stop_reason)
    except BaseException as exc:
        trainer.write_status("failed", error=str(exc))
        raise
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


if __name__ == "__main__":
    main()
