"""A fixed height field, queried at every real or imagined robot pose.

The same samples build MuJoCo's collision surface and the differentiable map.
World coordinates are origin-free; perceptive scenes use coincident origins in
independent MuJoCo Warp worlds. Maps are immutable during a run, so replay only
needs the state, not a stale robot-centred crop or an episode-dependent map ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from supermimic.kinematics.rotations import heading_yaw


@dataclass
class TerrainCfg:
    enabled: bool = False
    kind: Literal["flat", "boxes", "stairs", "slope"] = "boxes"
    seed: int = 17
    size: float = 16.0
    resolution: float = 0.05
    height: float = 0.08
    box_count: int = 32
    box_width: float = 0.65
    spawn_radius: float = 0.8
    stair_width: float = 0.5
    stair_count: int = 4
    scan_x: tuple[float, float] = (-0.5, 1.1)
    scan_y: tuple[float, float] = (-0.5, 0.5)
    scan_shape: tuple[int, int] = (17, 11)
    nominal_root_height: float = 0.75
    scan_clip: float = 1.0
    obs_noise_std: float = 0.01
    dropout: float = 0.02
    lower_body_weight: float = 0.25
    foot_clearance: float = 0.025
    clearance_weight: float = 2.0

    @property
    def obs_dim(self) -> int:
        return 2 * self.scan_shape[0] * self.scan_shape[1]

    def validate(self) -> None:
        if self.size <= 0 or self.resolution <= 0 or self.height < 0:
            raise ValueError("terrain size/resolution must be positive and height nonnegative")
        if self.size / self.resolution < 2 or self.size / self.resolution > 2048:
            raise ValueError("terrain must have between 3 and 2049 samples per side")
        if any(n < 2 for n in self.scan_shape) or self.scan_clip <= 0:
            raise ValueError("scan dimensions must be >= 2 and scan_clip positive")
        if self.scan_x[0] >= self.scan_x[1] or self.scan_y[0] >= self.scan_y[1]:
            raise ValueError("scan bounds must be increasing")
        if not 0 <= self.dropout < 1 or self.obs_noise_std < 0:
            raise ValueError("invalid terrain observation noise")
        if (
            self.box_width <= 0
            or self.box_count < 0
            or self.stair_width <= 0
            or self.stair_count < 1
        ):
            raise ValueError("invalid obstacle dimensions/counts")
        if (
            not 0 < self.lower_body_weight <= 1
            or self.foot_clearance < 0
            or self.clearance_weight < 0
        ):
            raise ValueError("invalid terrain reward settings")
        if self.kind not in ("flat", "boxes", "stairs", "slope"):
            raise ValueError(f"unknown terrain kind: {self.kind}")


def height_samples(cfg: TerrainCfg) -> np.ndarray:
    cfg.validate()
    n = round(cfg.size / cfg.resolution) + 1
    axis = np.linspace(-cfg.size / 2, cfg.size / 2, n, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    h = np.zeros_like(x)
    if cfg.kind == "boxes":
        rng = np.random.default_rng(cfg.seed)
        for _ in range(cfg.box_count):
            cx, cy = rng.uniform(-cfg.size / 2 + cfg.box_width, cfg.size / 2 - cfg.box_width, 2)
            mask = (np.abs(x - cx) <= cfg.box_width / 2) & (np.abs(y - cy) <= cfg.box_width / 2)
            h[mask] = np.maximum(h[mask], rng.uniform(0.4, 1.0) * cfg.height)
    elif cfg.kind == "stairs":
        # A finite staircase up and back down, with a flat central spawn area.
        distance = np.maximum(np.abs(x) - cfg.spawn_radius, 0)
        steps = np.floor(distance / cfg.stair_width)
        level = np.minimum(steps, 2 * cfg.stair_count - steps).clip(0, cfg.stair_count)
        h = (cfg.height * level).astype(np.float32)
    elif cfg.kind == "slope":
        h = (cfg.height * np.maximum(np.abs(x) - cfg.spawn_radius, 0)).astype(np.float32)
    h[x * x + y * y < cfg.spawn_radius**2] = 0
    # Match the surrounding ground plane at the finite map boundary.
    h[[0, -1], :] = 0
    h[:, [0, -1]] = 0
    return h


class TerrainField(nn.Module):
    def __init__(self, cfg: TerrainCfg, device="cpu"):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.register_buffer("heights", torch.from_numpy(height_samples(cfg)).to(device))
        x = torch.linspace(*cfg.scan_x, cfg.scan_shape[0], device=device)
        y = torch.linspace(*cfg.scan_y, cfg.scan_shape[1], device=device)
        gx, gy = torch.meshgrid(x, y, indexing="ij")
        self.register_buffer("scan_offsets", torch.stack([gx.flatten(), gy.flatten()], -1))

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """MuJoCo's piecewise-linear triangle samples, differentiable in xy.

        Each cell is split along its bottom-left to top-right diagonal. Outside
        the finite field, the collision scene has a flat ground plane.
        """
        n = self.heights.shape[0]
        uv = (xy / self.cfg.size + 0.5) * (n - 1)
        uv = uv.clamp(0, n - 1 - 1e-4)
        ij = uv.floor().long()
        f = uv - ij
        ix, iy = ij.unbind(-1)
        fx, fy = f.unbind(-1)
        h = self.heights
        h00, h10, h01, h11 = h[iy, ix], h[iy, ix + 1], h[iy + 1, ix], h[iy + 1, ix + 1]
        z = torch.where(
            fy <= fx,
            h00 + fx * (h10 - h00) + fy * (h11 - h10),
            h00 + fy * (h01 - h00) + fx * (h11 - h01),
        )
        return torch.where((xy.abs() <= self.cfg.size / 2).all(-1), z, torch.zeros_like(z))

    def observation(self, state: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        """Flattened [relative heights, validity], in a heading-aligned grid."""
        yaw = heading_yaw(state[..., 3:7])
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        ox, oy = self.scan_offsets.unbind(-1)
        xy = state[..., None, :2] + torch.stack([c * ox - s * oy, s * ox + c * oy], -1)
        valid = (xy.abs() <= self.cfg.size / 2).all(-1)
        z = self.height(xy) - state[..., 2:3] + self.cfg.nominal_root_height
        if noisy:
            z = z + torch.randn_like(z) * self.cfg.obs_noise_std
            valid = valid & (torch.rand_like(z) >= self.cfg.dropout)
        z = z.clamp(-self.cfg.scan_clip, self.cfg.scan_clip) / self.cfg.scan_clip
        return torch.cat([torch.where(valid, z, torch.zeros_like(z)), valid.to(z.dtype)], -1)


def add_collision_heightfield(spec, cfg: TerrainCfg) -> None:
    """SceneCfg.spec_fn callback, before MuJoCo compilation."""
    import mujoco

    heights = height_samples(cfg)
    scale = max(float(heights.max()), 1e-4)
    hf = spec.add_hfield(
        name="perceptive_heightfield",
        nrow=heights.shape[0],
        ncol=heights.shape[1],
        size=[cfg.size / 2, cfg.size / 2, scale, 0.1],
    )
    hf.userdata = (heights / scale).ravel().tolist()
    spec.worldbody.add_geom(
        name="perceptive_ground",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="perceptive_heightfield",
        contype=1,
        conaffinity=1,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.35, 0.42, 0.32, 1],
    )
