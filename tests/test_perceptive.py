"""Perception invariants and real MuJoCo geometry checks (CPU runnable)."""

from dataclasses import asdict

import mujoco
import numpy as np
import pytest
import torch

from supermimic.perception.terrain import TerrainCfg, TerrainField, add_collision_heightfield
from supermimic.policy.nets import Policy, PolicyCfg


def states(n=2):
    s = torch.zeros(n, 71)
    s[:, 2], s[:, 3] = 0.75, 1
    return s


def test_scan_translation_heading_and_mask():
    field = TerrainField(TerrainCfg(size=4, resolution=0.1, kind="flat"))
    axis = torch.linspace(-2, 2, 41)
    field.heights.copy_(0.2 * axis[None, :].expand(41, 41) + 0.4)
    s = states()
    s[1, 0] = 0.2
    scan = field.observation(s)
    torch.testing.assert_close(
        scan[1, :187] - scan[0, :187], torch.full((187,), 0.04), atol=1e-6, rtol=0
    )
    # Turn 90 degrees: the x ramp now varies along the scan's local y axis.
    s = states(1)
    s[:, 3], s[:, 6] = 2**-0.5, 2**-0.5
    z = field.observation(s)[0, :187].reshape(17, 11)
    torch.testing.assert_close(z[0], z[-1], atol=1e-6, rtol=0)
    assert z[0, 0] > z[0, -1]
    s[:, 0] = 10
    assert field.observation(s).count_nonzero() == 0


def test_map_query_position_gradients():
    field = TerrainField(TerrainCfg(kind="slope", size=6, resolution=0.1, height=0.1))
    xy = torch.tensor([[1.43, 0.13]], requires_grad=True)
    field.height(xy).sum().backward()
    torch.testing.assert_close(xy.grad, torch.tensor([[0.1, 0.0]]), atol=1e-6, rtol=0)
    s = states(1).requires_grad_()
    field.observation(s).sum().backward()
    assert torch.isfinite(s.grad).all() and s.grad[:, 2].abs().sum() > 0


@pytest.mark.parametrize("kind", ["flat", "boxes", "stairs", "slope"])
def test_collision_height_matches_map_vertices(kind):
    cfg = TerrainCfg(kind=kind, size=6, resolution=0.1, box_count=8)
    field = TerrainField(cfg)
    spec = mujoco.MjSpec()
    add_collision_heightfield(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = [(10, 20), (36, 30), (45, 25), (25, 40), (55, 35)]
    points += np.random.default_rng(42).uniform(1, 59, (30, 2)).tolist()
    for ix, iy in points:
        x, y = -3 + 0.1 * ix, -3 + 0.1 * iy
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([x, y, 3.0]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            -1,
            np.array([-1], dtype=np.int32),
        )
        assert distance >= 0
        expected = field.height(torch.tensor([[x, y]])).item()
        assert 3 - distance == pytest.approx(expected, abs=2e-5)


def small_policy(terrain_dim=0):
    return PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,), terrain_dim=terrain_dim)


def test_zero_gates_preserve_tracker_then_learn_perception():
    torch.manual_seed(4)
    flat = Policy(small_policy())
    perceptive = Policy(small_policy(374))
    missing, unexpected = perceptive.load_state_dict(flat.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("terrain_") for k in missing)
    obs, goal = torch.randn(2, 930), torch.randn(2, 680)
    scan = torch.randn(2, 374)
    torch.testing.assert_close(perceptive(obs, goal, terrain=scan), flat(obs, goal))
    perceptive(obs, goal, terrain=scan).sum().backward()
    assert perceptive.terrain_action_gate.grad.abs().sum() > 0
    with torch.no_grad():
        perceptive.terrain_action_gate.fill_(0.5)
    assert not torch.allclose(
        perceptive(obs, goal, terrain=scan), perceptive(obs, goal, terrain=torch.zeros_like(scan))
    )
    with pytest.raises(ValueError, match="terrain observation"):
        perceptive(obs, goal)


def test_perceptive_checkpoint_round_trip(tmp_path):
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    p = Policy(small_policy(374))
    path = tmp_path / "checkpoint.pt"
    cfg = {"policy": asdict(p.cfg), "terrain": asdict(TerrainCfg(enabled=True))}
    torch.save({"policy": p.state_dict(), "config": cfg}, path)
    loaded, restored = load_policy_from_checkpoint(path, "cpu")
    assert restored["terrain"]["enabled"]
    for k, v in p.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)


def test_terrain_world_model_inputs_and_gradients():
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    terrain = TerrainField(TerrainCfg(kind="stairs", size=6, resolution=0.1))
    wm = WorldModel(WorldModelCfg(hidden=(32,)), None, "cpu", terrain=terrain)
    s = states()
    xi = torch.zeros(2, 33)
    xi[:, 0] = 0.75
    a = torch.randn(2, 29, requires_grad=True)
    # Remove the identity initialization only to verify the differentiation path.
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=0.01)
    context = wm.dynamics_context(s, xi)
    torch.testing.assert_close(context[:, 33:], terrain.observation(s))
    moved = s.clone()
    moved[:, 0] += 1.5
    assert not torch.allclose(context[:, 33:], wm.dynamics_context(moved, xi)[:, 33:])
    wm.step(s, a, xi).sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_terrain_height_termination_uses_adapted_root():
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY

    terrain = TerrainField(TerrainCfg(kind="stairs", size=6, resolution=0.1, height=0.15))
    heads = AnalyticHeads(
        G1Kinematics(load_g1_model()),
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        torch.tensor([[-3.0, 3.0]] * 29),
        torch.zeros(29),
        terrain=terrain,
    )
    s = states(1)
    s[:, 0] = 2.3
    ref = ReferenceFrame(None, None, None, None, s[:, :3].clone(), s[:, 3:7].clone())
    s[:, 2] += terrain.height(s[:, :2])
    torch.testing.assert_close(
        heads.termination_features(s[:, :3], s[:, 3:7], ref), torch.zeros(1, 2)
    )
