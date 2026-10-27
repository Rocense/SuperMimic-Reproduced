"""Stair shape and saved-map compatibility, beyond matching heightfield vertices."""
from dataclasses import asdict

import mujoco
import numpy as np
import pytest
import torch

from supermimic.perception.paired import restore_config, terrain_config
from supermimic.perception.terrain import TerrainCfg, TerrainField, add_collision_terrain, height_samples


def cast(model, data, origin, direction):
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array(origin, dtype=float),
                             np.array(direction, dtype=float), None, 1, -1, geom)
    return distance, int(geom[0])


@pytest.mark.parametrize('riser_height', [0.025, 0.15])
def test_box_stairs_have_vertical_risers_and_no_coplanar_heightfield(riser_height):
    cfg = TerrainCfg(kind='stairs', size=12, resolution=0.1, height=riser_height)
    field = TerrainField(cfg)
    spec = mujoco.MjSpec()
    add_collision_terrain(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nhfield == 0
    assert np.count_nonzero(model.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)) == 1
    assert np.count_nonzero(model.geom_type == int(mujoco.mjtGeom.mjGEOM_BOX)) == 14
    edge = cfg.spawn_radius + cfg.stair_width
    for z in (cfg.height * 0.1, cfg.height * 0.5, cfg.height * 0.9):
        distance, geom = cast(model, data, [edge - 0.2, 0, z], [1, 0, 0])
        assert distance == pytest.approx(0.2, abs=1e-8)
        assert model.geom_type[geom] == int(mujoco.mjtGeom.mjGEOM_BOX)
    xy = torch.tensor([[edge - 1e-4, 0], [edge + 1e-4, 0], [edge + 0.2, 0]], requires_grad=True)
    height = field.height(xy)
    torch.testing.assert_close(height, torch.tensor([0.0, cfg.height, cfg.height]))
    height.sum().backward()
    torch.testing.assert_close(xy.grad, torch.zeros_like(xy))
    # Both sides, ascent and descent, lateral boundary and outside the finite stair footprint.
    for x, y in [(1.3001, 0), (-1.3001, 0), (2.91, 0), (3.51, 0), (4.79, 0),
                 (4.81, 0), (2, 5.99), (2, 6.01), (8, 0)]:
        distance, _ = cast(model, data, [x, y, 2], [0, 0, -1])
        expected = field.height(torch.tensor([[x, y]])).item()
        assert 2 - distance == pytest.approx(expected, abs=2e-6)


def test_saved_legacy_stairs_retain_the_original_ramp_surface():
    original = asdict(TerrainCfg(kind='stairs', size=12, resolution=0.1, height=0.025))
    original.pop('stairs_surface')
    for cfg in (terrain_config(original), restore_config({'terrain': original}).terrain):
        assert cfg.stairs_surface == 'heightfield'
        heights = height_samples(cfg)
        middle = heights.shape[0] // 2
        row = heights[middle]
        left = next(i for i in range(middle, len(row) - 1) if row[i + 1] > row[i])
        midpoint = -cfg.size / 2 + (left + 0.5) * cfg.resolution
        value = TerrainField(cfg).height(torch.tensor([[midpoint, 0]])).item()
        assert value == pytest.approx(float((row[left] + row[left + 1]) / 2), abs=2e-6)
        assert 0 < value < cfg.height
    assert terrain_config(asdict(TerrainCfg(kind='stairs'))).stairs_surface == 'boxes'


def test_new_paired_geometry_rejects_old_world_model_surface():
    old = terrain_config({'kind': 'stairs', 'enabled': True})
    new = TerrainCfg(kind='stairs', enabled=True)
    assert old != new  # Existing terrain-equality guards must distinguish these environments.
