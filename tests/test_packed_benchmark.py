"""A packed file represents many clips, not one evaluation environment."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.eval import benchmark


def test_packed_benchmark_uses_clip_count_and_compact_storage(tmp_path, monkeypatch):
    path = tmp_path / "all.pt"
    torch.save({"names": ["walk", "dance", "jump"], "ref": torch.zeros(9, 71),
                "starts": [0, 3, 6], "lengths": [3, 3, 3]}, path)
    received = []

    class ConfigCaptured(Exception):
        pass

    def capture(args, device):
        received.append(args)
        raise ConfigCaptured

    monkeypatch.setattr(benchmark, "make_env", capture)
    with pytest.raises(ConfigCaptured):
        benchmark.run_benchmark(SimpleNamespace(cfg=SimpleNamespace(history=10)),
                                [str(path)], num_envs=8, device="cpu")
    assert received[0].num_envs == 3
    assert received[0].compact_library
    assert received[0].npz_files == [str(path)]


def test_raw_pack_cannot_be_used_as_adapted_evaluation(tmp_path):
    with pytest.raises(ValueError, match="paired TCRS"):
        benchmark.run_benchmark(None, [str(tmp_path / "raw.pt")], reference_is_adapted=True)
