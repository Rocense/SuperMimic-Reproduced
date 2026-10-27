"""Full-corpus preprocessing must preserve frames, placement and completed outputs."""

import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.pmt_protocol import prepare_full as preparation
from supermimic.data.motion_convert import G1_JOINT_NAMES, load_g1_model
from supermimic.perception.paired import sha256


def reference(frames=7):
    ref = np.zeros((frames, 71), dtype=np.float16)
    ref[:, 0] = np.linspace(10, 10.2, frames)
    ref[:, 1] = 20
    ref[:, 2] = 0.75
    ref[:, 3] = 1
    ref[:, 36] = 0.5
    return ref


def test_packed_motion_preserves_all_frames_joints_and_world_velocity():
    ref = reference()
    raw, origin = preparation.packed_motion(ref, load_g1_model(), (1.5, -0.3, np.pi / 2))
    names = raw["body_names"].tolist()
    root = names.index("pelvis")
    assert len(raw["joint_pos"]) == 7
    assert tuple(raw["joint_names"]) == G1_JOINT_NAMES
    np.testing.assert_allclose(origin, [10, 20])
    np.testing.assert_allclose(raw["body_pos_w"][0, root], [1.5, -0.3, 0.75], atol=1e-6)
    np.testing.assert_allclose(raw["body_pos_w"][:, root, 1], -0.3 + ref[:, 0].astype(float) - 10, atol=1e-6)
    np.testing.assert_allclose(raw["body_lin_vel_w"][:, root, :2], np.tile([0, 0.5], (7, 1)), atol=1e-6)
    np.testing.assert_array_equal(raw["joint_pos"], ref[:, 7:36])


def test_placement_is_reproducible_and_varies_between_pairings():
    assert preparation.placement("walk", 0, 11) == preparation.placement("walk", 0, 11)
    assert preparation.placement("walk", 0, 11) != preparation.placement("walk", 1, 11)


def test_completed_pair_is_reused_and_tampering_is_rejected(tmp_path, monkeypatch):
    for folder in ("raw", "adapted", "rejected", "records"):
        (tmp_path / "train" / folder).mkdir(parents=True)
    monkeypatch.setattr(preparation, "_ROOT", tmp_path)
    monkeypatch.setattr(preparation, "_PLAN", {"seed": 1, "pairings_per_motion": 1, "bones_sha256": "source"})
    monkeypatch.setattr(preparation, "_PACK", {"ref": torch.from_numpy(reference()),
                                               "names": ["fixture"], "starts": [0], "lengths": [7]})
    csv = tmp_path / "inputs/bones_csv/000000.csv"
    csv.parent.mkdir(parents=True)
    data = np.zeros((16, 36))
    data[:, 0] = np.arange(16)
    data[:, 1] = np.linspace(1000, 1020, 16)
    data[:, 2:4] = [2000, 75]
    columns = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
               "root_rotateX", "root_rotateY", "root_rotateZ"] + [f"{n}_dof" for n in G1_JOINT_NAMES]
    np.savetxt(csv, data, delimiter=",", header=",".join(columns), comments="")
    monkeypatch.setattr(preparation, "_CSV_INDEX", {"fixture": {"file": csv.name, "sha256": sha256(csv)}})
    calls = []

    def synthesize(raw):
        calls.append(True)
        return raw, {"accepted": True, "frames": len(raw["joint_pos"])}

    monkeypatch.setattr(preparation, "_SYNTH", SimpleNamespace(
        model=load_g1_model(), rng=torch.Generator(), synthesize=synthesize))
    row = preparation.one(("train", 0, 0))
    assert row["frames"] == 7 and row["source_name"] == "fixture"
    assert preparation.one(("train", 0, 0)) == row and len(calls) == 1
    record = json.loads((tmp_path / "train/records/train_000000_p000.json").read_text())
    assert record["raw_sha256"] and record["adapted_sha256"]
    with (tmp_path / "train/raw/train_000000_p000.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="completed output changed"):
        preparation.one(("train", 0, 0))


def test_archive_extraction_uses_numeric_paths_and_only_selected_regular_files(tmp_path):
    (tmp_path / "inputs").mkdir()
    content = b"CSV content"
    with tarfile.open(tmp_path / "inputs/g1.tar.gz", "w:gz") as archive:
        for name in ("g1/csv/fixture.csv", "../../outside_tcrs_test.csv", "unused.csv"):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        info = tarfile.TarInfo("fixture.csv")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        archive.addfile(info)
    preparation.extract_selected_csv(tmp_path, ["fixture", "outside_tcrs_test"], "archive", lambda *args: None)
    assert (tmp_path / "inputs/bones_csv/000000.csv").read_bytes() == content
    assert (tmp_path / "inputs/bones_csv/000001.csv").read_bytes() == content
    assert not (tmp_path / "outside_tcrs_test.csv").exists()
    index = json.loads((tmp_path / "inputs/bones_csv_index.json").read_text())
    assert set(index["files"]) == {"fixture", "outside_tcrs_test"}
