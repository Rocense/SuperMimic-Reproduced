"""Guard against reusing a sliced/corrupt corpus as a full-data experiment."""

import csv
import json

import numpy as np
import pytest
import torch

from experiments.full_corpus.prepare import prepare
from experiments.full_corpus.train import main


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source"
    (source / "library").mkdir(parents=True)
    (source / "npz/lafan1").mkdir(parents=True)
    ref = torch.zeros(11, 71, dtype=torch.float16)
    ref[:, 3] = 1
    library = source / "library/bones_seed_compact.pt"
    torch.save({"ref": ref, "names": ["walk", "dance"], "starts": [0, 4],
                "lengths": [4, 7]}, library)
    with (source / "library/bones_seed_compact.manifest.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerows([("name", "frames", "keep", "reason"), ("walk", 4, True, ""),
                          ("dance", 7, True, ""), ("chair", 0, False, "name:chair")])
    ready = {"status": "ready", "fps": 50, "counts": {"raw": 3, "kept": 2},
             "frames": 11, "bytes": library.stat().st_size}
    (source / "library/bones_seed_compact.ready.json").write_text(json.dumps(ready))
    for i in range(40):
        np.savez(source / f"npz/lafan1/motion_{i}.npz", joint_pos=np.zeros((i + 3, 29)), fps=50)
    return source, tmp_path / "linked"


def test_full_preparation_links_files_and_preserves_every_frame(corpus, monkeypatch):
    source, output = corpus
    report = prepare(source, output)
    assert report["bones"]["frames"] == 11
    assert report["lafan1"]["frames"] == sum(range(3, 43))
    assert report["bones"]["excluded_clips"] == 1
    assert (output / "bones_seed_compact.pt").samefile(source / "library/bones_seed_compact.pt")
    config_path = output / "bones_flat_config.json"
    config = json.loads(config_path.read_text())
    assert config["data"]["clips"] is None and config["data"]["limit"] is None
    assert config["policy"]["encoder_hidden"] == [2048, 1024, 512, 256]
    monkeypatch.setattr("sys.argv", ["train.py", "--config", str(config_path)])
    main()  # CPU preflight must never construct a simulator or train.
    config["data"]["limit"] = 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="silently select"):
        main()


def test_preparation_rejects_a_subset_even_if_the_source_is_called_full(corpus):
    source, output = corpus
    path = source / "library/bones_seed_compact.pt"
    pack = torch.load(path, weights_only=True)
    pack.update(names=["walk"], starts=[0], lengths=[4], ref=pack["ref"][:4])
    torch.save(pack, path)
    with pytest.raises(ValueError, match="complete keep manifest"):
        prepare(source, output)
    assert not output.exists()


def test_preparation_rejects_corrupt_frames_outside_the_first_clip(corpus):
    source, output = corpus
    path = source / "library/bones_seed_compact.pt"
    pack = torch.load(path, weights_only=True)
    pack["ref"][-1, 15] = float("nan")
    torch.save(pack, path)
    receipt = source / "library/bones_seed_compact.ready.json"
    ready = json.loads(receipt.read_text())
    ready["bytes"] = path.stat().st_size
    receipt.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="nonfinite"):
        prepare(source, output)
    assert not output.exists()
