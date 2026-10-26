"""Tests of the TCRS/DAgger contracts, beyond tensor-shape smoke checks."""

import numpy as np
import pytest
import torch

from supermimic.perception.control import aligned_action
from supermimic.perception.distillation import AggregationBuffer
from supermimic.perception.tcrs import FOOT_POINTS, MID_FOOT, Synthesizer, TCRSCfg, detect_contacts
from supermimic.perception.terrain import TerrainCfg, TerrainField


def test_action_alignment_preserves_the_pd_setpoint():
    teacher = {"joint_names": ["hip", "knee"], "scale": [0.2, 0.4], "offset": [0.1, 0.5]}
    student = {"joint_names": ["hip", "knee"], "scale": [0.3, 0.2], "offset": [-0.1, 0.7]}
    action = torch.tensor([[0.3, -0.4], [1.0, 2.0]])
    aligned = aligned_action(action, teacher, student)
    torch.testing.assert_close(
        torch.tensor(student["offset"]) + torch.tensor(student["scale"]) * aligned,
        torch.tensor(teacher["offset"]) + torch.tensor(teacher["scale"]) * action,
    )
    student["joint_names"].reverse()
    with pytest.raises(ValueError, match="ordering"):
        aligned_action(action, teacher, student)


def test_dagger_aggregation_wraps_and_detaches_labels():
    buffer = AggregationBuffer(4, {"x": torch.zeros(3, 2)})
    buffer.add({"x": torch.ones(3, 2, requires_grad=True)})
    buffer.add({"x": torch.full((3, 2), 2.0)})
    assert buffer.size == 4 and buffer.cursor == 2
    assert not buffer.values["x"].requires_grad
    assert (buffer.values["x"][:, 0] == 2).sum() == 3
    assert set(buffer.sample(128)["x"][:, 0].tolist()) == {1.0, 2.0}


def test_contact_detector_preserves_timeline_and_finds_swing():
    points = np.tile(FOOT_POINTS, (40, 2, 1, 1))
    points[..., 2] += 0.03
    points[10:30, 0, :, 2] += (0.18 * np.sin(np.linspace(0, np.pi, 20)))[:, None]
    contacts = detect_contacts(points, 50, TCRSCfg())
    assert contacts.shape == (40, 2)
    assert contacts[:, 1].all()
    assert not contacts[20, 0]
    assert contacts[0, 0] and contacts[-1, 0]


def test_swing_optimization_clears_obstacle_and_keeps_endpoints():
    field = TerrainField(TerrainCfg(kind="flat", size=4, resolution=0.1))
    field.heights[:, 20:24] = 0.10
    cfg = TCRSCfg(samples=32, sampling_iterations=6, seed=3)
    synth = Synthesizer(field, cfg)
    raw = np.zeros((35, 3))
    raw[:, 0] = np.linspace(-0.6, 0.8, len(raw))
    raw[:, 2] = 0.04
    offsets = np.tile(FOOT_POINTS - MID_FOOT, (len(raw), 1, 1))
    adapted = synth.swing(raw, offsets, raw[0], raw[-1])
    np.testing.assert_allclose(adapted[[0, -1]], raw[[0, -1]], atol=1e-6)

    def penetration(path):
        points = path[:, None] + offsets
        return np.maximum(synth.height(points[:, 1:, :2]) - points[:, 1:, 2], 0).mean()

    assert penetration(adapted) < 0.5 * penetration(raw)


def test_adapted_targets_are_not_lifted_a_second_time():
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY

    terrain = TerrainField(TerrainCfg(kind="stairs", size=6, resolution=0.1, height=0.1))
    heads = AnalyticHeads(
        G1Kinematics(load_g1_model()),
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        torch.tensor([[-3.0, 3.0]] * 29),
        torch.zeros(29),
        terrain=terrain,
        reference_is_adapted=True,
    )
    pos = torch.tensor([[2.0, 0.0, 0.9]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    ref = ReferenceFrame(None, None, None, None, pos.clone(), quat.clone())
    torch.testing.assert_close(heads.termination_features(pos, quat, ref), torch.zeros(1, 2))
    assert heads.adapt_reference(ref) is ref
    torch.testing.assert_close(heads.body_weights, torch.ones(len(TRACKED_BODIES)))
