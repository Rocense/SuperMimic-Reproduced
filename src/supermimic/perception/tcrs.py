"""Offline, phase-preserving terrain-conformal reference synthesis for G1.

An independent implementation of the TCRS stages: contact latching, sampled
cubic swing optimization, support-aware root reconstruction, and multi-point
damped Jacobian IK with lower-leg collision repair and continuity guards.
This produces kinematic supervision, not a proof of dynamic feasibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import mujoco
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    ResampledMotion,
    _so3_derivative,
    forward_kinematics,
    load_g1_model,
)
from supermimic.perception.terrain import TerrainField


@dataclass
class TCRSCfg:
    contact_height: float = 0.045
    contact_speed: float = 0.20
    hysteresis: float = 1.5
    clearance: float = 0.035
    edge_tolerance: float = 0.10
    knots: int = 6
    samples: int = 96
    sampling_iterations: int = 12
    sampling_noise: float = 0.04
    temperature: float = 0.03
    max_horizontal_shift: float = 0.18
    root_smoothing: float = 2.0
    ik_iterations: int = 30
    damping: float = 1e-4
    posture_weight: float = 0.001
    continuity_weight: float = 0.002
    max_joint_step: float = 0.12
    max_foot_error: float = 0.055
    max_penetration: float = 0.025
    max_joint_jump: float = 0.45
    seed: int = 0
    acceptance_rule: Literal["pmt_batch", "legacy"] = "pmt_batch"
    batch_reject_ik_mm: float = 200.0
    batch_reject_ik_bad_frames: int = 8

    def validate(self):
        if self.acceptance_rule not in ("pmt_batch", "legacy") or self.batch_reject_ik_mm <= 0:
            raise ValueError("invalid TCRS acceptance rule")
        if (
            self.knots < 2
            or self.samples < 2
            or self.sampling_iterations < 1
            or self.ik_iterations < 1
        ):
            raise ValueError("TCRS needs >=2 knots/samples and positive iteration counts")
        if min(self.temperature, self.sampling_noise, self.damping, self.max_joint_step) <= 0:
            raise ValueError("TCRS temperature, noise, damping and step limits must be positive")


# Ankle origin plus sole toe and heel samples, in the ankle-roll link frame.
FOOT_POINTS = np.array([[0.0, 0.0, 0.0], [0.12, 0.0, -0.03], [-0.06, 0.0, -0.03]])
MID_FOOT = (FOOT_POINTS[1] + FOOT_POINTS[2]) / 2


def intervals(mask):
    """Inclusive/exclusive contiguous true intervals."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect_contacts(points, fps, cfg):
    sole = points[..., 1:, :].mean(-2)  # T, feet, xyz
    speed = np.linalg.norm(np.gradient(sole, 1 / fps, axis=0), axis=-1)
    height = points[..., 1:, 2].min(-1)
    floor = np.quantile(height, 0.05)
    contacts = np.zeros(height.shape, dtype=bool)
    for t in range(len(points)):
        factor = np.where(contacts[t - 1], cfg.hysteresis, 1.0) if t else 1.0
        contacts[t] = (height[t] < floor + cfg.contact_height * factor) & (
            speed[t] < cfg.contact_speed * factor
        )
    return contacts


class Synthesizer:
    def __init__(self, terrain: TerrainField, cfg: TCRSCfg | None = None):
        self.terrain, self.cfg = terrain.cpu(), cfg or TCRSCfg()
        self.cfg.validate()
        self.model = load_g1_model()
        self.data = mujoco.MjData(self.model)
        self.foot_ids = [
            self.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")
        ]
        self.knee_ids = [self.model.body(f"{side}_knee_link").id for side in ("left", "right")]
        self.leg_ids = [self.model.joint(name).id for name in G1_JOINT_NAMES[:12]]
        self.qadr = self.model.jnt_qposadr[self.leg_ids]
        self.vadr = self.model.jnt_dofadr[self.leg_ids]
        self.limits = self.model.jnt_range[self.leg_ids]
        self.rng = torch.Generator().manual_seed(self.cfg.seed)

    def height(self, xy):
        with torch.no_grad():
            return (
                self.terrain.height(torch.as_tensor(xy, dtype=torch.float32))
                .numpy()
                .astype(np.float64)
            )

    def points(self):
        return np.stack(
            [
                self.data.xpos[b] + FOOT_POINTS @ self.data.xmat[b].reshape(3, 3).T
                for b in self.foot_ids
            ]
        )

    def pose(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def swing(self, raw, offsets, start, end):
        """MPPI-style softmin updates of cubic control knots; endpoints fixed."""
        cfg = self.cfg
        n = len(raw)
        k = min(cfg.knots, n)
        basis = torch.tensor(
            CubicSpline(np.linspace(0, 1, k), np.eye(k), axis=0)(np.linspace(0, 1, n)),
            dtype=torch.float32,
        )
        base = torch.tensor(raw, dtype=torch.float32)
        off = torch.tensor(offsets, dtype=torch.float32)
        phase = torch.linspace(0, 1, n)
        desired = base.clone()
        desired += (1 - phase[:, None]) * (torch.tensor(start, dtype=torch.float32) - base[0])
        desired += phase[:, None] * (torch.tensor(end, dtype=torch.float32) - base[-1])
        sole = desired[:, None, :] + off[:, 1:, :]
        required = (self.terrain.height(sole[..., :2]) - off[:, 1:, 2]).amax(-1)
        desired[:, 2] = torch.maximum(
            desired[:, 2], required + cfg.clearance * torch.sin(torch.pi * phase)
        )
        knots = desired[torch.linspace(0, n - 1, k).round().long()].clone()
        knots[0], knots[-1] = torch.tensor(start), torch.tensor(end)
        best, best_cost = knots.clone(), float("inf")
        for _ in range(cfg.sampling_iterations):
            noise = torch.randn(cfg.samples, k, 3, generator=self.rng) * cfg.sampling_noise
            noise[:, [0, -1]] = 0
            noise[0] = 0
            candidates = knots[None] + noise
            path = torch.einsum("tk,skd->std", basis, candidates)
            points = path[:, :, None] + off[None]
            heights = self.terrain.height(points[..., :2])
            gap = (
                heights[..., 1:]
                + cfg.clearance * torch.sin(torch.pi * phase)[None, :, None]
                - points[..., 1:, 2]
            ).clamp_min(0)
            # Toe/heel queries account for vertical faces across the whole foot footprint.
            edge = (heights.amax(-1) - points[..., 1:, 2].amin(-1)).clamp_min(0)
            smooth = (
                (path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2]).square().mean((1, 2))
                if n > 2
                else torch.zeros(cfg.samples)
            )
            drift = (
                (path[..., :2] - base[None, :, :2])
                .abs()
                .sub(cfg.max_horizontal_shift)
                .clamp_min(0)
                .square()
                .mean((1, 2))
            )
            cost = (
                (path - desired[None]).square().mean((1, 2))
                + 20 * smooth
                + 100 * gap.square().mean((1, 2))
                + 20 * edge.square().mean(1)
                + 200 * drift
            )
            idx = int(cost.argmin())
            if float(cost[idx]) < best_cost:
                best, best_cost = candidates[idx].clone(), float(cost[idx])
            weights = torch.softmax(-(cost - cost.min()) / cfg.temperature, 0)
            knots = (weights[:, None, None] * candidates).sum(0)
            knots[0], knots[-1] = torch.tensor(start), torch.tensor(end)
        return (basis @ best).numpy().astype(np.float64)

    def _ik(self, initial, raw, previous, target, weights):
        cfg = self.cfg
        q = initial.copy()
        lo = np.maximum(self.limits[:, 0], previous[self.qadr] - 0.9 * cfg.max_joint_jump)
        hi = np.minimum(self.limits[:, 1], previous[self.qadr] + 0.9 * cfg.max_joint_jump)
        q[self.qadr] = q[self.qadr].clip(lo, hi)
        best, best_cost = q.copy(), np.inf
        for _ in range(cfg.ik_iterations):
            self.pose(q)
            actual = self.points()
            jac, errors, ws = [], [], []
            for f, body in enumerate(self.foot_ids):
                for point in range(3):
                    jp = np.zeros((3, self.model.nv))
                    mujoco.mj_jac(self.model, self.data, jp, None, actual[f, point], body)
                    jac.append(jp[:, self.vadr])
                    errors.append(target[f, point] - actual[f, point])
                    ws.extend([weights[f, point]] * 3)
            # Repair foot and shin penetration using active inequality residuals.
            collision_points = []
            for f, body in enumerate(self.foot_ids):
                collision_points.extend((p, body, 0.0) for p in actual[f, 1:])
                knee = self.knee_ids[f]
                collision_points.append((self.data.xpos[knee].copy(), knee, 0.04))
                collision_points.append(
                    ((self.data.xpos[knee] + self.data.xpos[body]) / 2, knee, 0.04)
                )
            penetration = []
            for point, body, radius in collision_points:
                error = float(self.height(point[:2])) + radius - point[2]
                penetration.append(max(error, 0))
                if error > 0:
                    jp = np.zeros((3, self.model.nv))
                    mujoco.mj_jac(self.model, self.data, jp, None, point, body)
                    jac.append(jp[2:3, self.vadr])
                    errors.append(np.array([error]))
                    ws.append(15.0)
            J, e, w = np.vstack(jac), np.concatenate(errors), np.asarray(ws)
            cost = float(
                np.mean((actual - target) ** 2)
                + 10 * np.mean(np.square(penetration))
                + cfg.continuity_weight * np.mean((q[self.qadr] - previous[self.qadr]) ** 2)
                + cfg.posture_weight * np.mean((q[self.qadr] - raw[self.qadr]) ** 2)
            )
            if cost < best_cost:
                best, best_cost = q.copy(), cost
            A = J.T @ (w[:, None] * J) + (
                cfg.damping + cfg.posture_weight + cfg.continuity_weight
            ) * np.eye(12)
            b = (
                J.T @ (w * e)
                + cfg.posture_weight * (raw[self.qadr] - q[self.qadr])
                + cfg.continuity_weight * (previous[self.qadr] - q[self.qadr])
            )
            dq = np.linalg.solve(A, b).clip(-cfg.max_joint_step, cfg.max_joint_step)
            q[self.qadr] = (q[self.qadr] + dq).clip(lo, hi)
            if np.max(np.abs(dq)) < 1e-5:
                break
        return best, best_cost

    def synthesize(self, raw: dict) -> tuple[dict, dict]:
        cfg = self.cfg
        fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
        if not np.isclose(fps, 50):
            raise ValueError("SuperMimic paired references must use 50 Hz")
        names = list(raw["body_names"].astype(str))
        root_idx = names.index("pelvis")
        qpos = np.concatenate(
            [raw["body_pos_w"][:, root_idx], raw["body_quat_w"][:, root_idx], raw["joint_pos"]], -1
        ).astype(np.float64)
        n = len(qpos)
        if n < 3 or not np.isfinite(qpos).all():
            raise ValueError("TCRS requires at least three finite frames")
        if "joint_names" in raw and tuple(raw["joint_names"].astype(str)) != G1_JOINT_NAMES:
            raise ValueError("input joint order differs from G1")
        raw_points = []
        for q in qpos:
            self.pose(q)
            raw_points.append(self.points())
        raw_points = np.stack(raw_points)
        if np.any(np.abs(raw_points[..., :2]) > self.terrain.cfg.size / 2):
            raise ValueError("reference leaves the terrain map; enlarge terrain.size")
        contacts = detect_contacts(raw_points, fps, cfg)
        floor = np.quantile(raw_points[..., 1:, 2].min(-1), 0.05)
        if not contacts.any():
            raise ValueError("no support contacts found; this TCRS preset targets locomotion")
        target = raw_points.copy()
        mid = raw_points[..., 1:, :].mean(-2)
        offsets = raw_points - mid[:, :, None]
        weights = np.ones((n, 2, 3))
        weights[..., 0] = 0.5
        for f in range(2):
            for lo, hi in intervals(contacts[:, f]):
                # Latched support in XY; toe/heel heights follow the support surface.
                centre = np.median(mid[lo:hi, f, :2], axis=0)
                target[lo:hi, f, :, :2] += centre - mid[lo:hi, f, None, :2]
                h = self.height(target[lo:hi, f, :, :2])
                # Preserve heel/toe roll during support; flattening both sole points
                # creates a discontinuous ankle target at liftoff and touchdown.
                target[lo:hi, f, 1:, 2] = h[:, 1:] + raw_points[lo:hi, f, 1:, 2] - floor + 0.002
                target[lo:hi, f, 0, 2] = (
                    h[:, 1:].mean(-1) + raw_points[lo:hi, f, 0, 2] - floor + 0.002
                )
                weights[lo:hi, f, 1:] *= np.exp(
                    -np.square((raw_points[lo:hi, f, 1:, 2] - floor) / 0.035)
                ).clip(0.1, 1)
                edge = np.ptp(h[:, 1:], axis=-1) > cfg.edge_tolerance
                weights[lo:hi, f, 1:][edge] *= 0.2
                weights[lo:hi, f] *= 3
            for lo, hi in intervals(~contacts[:, f]):
                start = max(0, lo - 1)
                end = min(n - 1, hi)
                if end <= start:
                    continue
                first = target[start, f, 1:].mean(0)
                last = target[end, f, 1:].mean(0)
                if not contacts[start, f]:
                    first[2] += float(self.height(first[:2]))
                if not contacts[end, f]:
                    last[2] += float(self.height(last[:2]))
                path = self.swing(mid[start : end + 1, f], offsets[start : end + 1, f], first, last)
                target[lo:hi, f] = path[lo - start : hi - start, None] + offsets[lo:hi, f]
        # Support-weighted root offset. Flight intervals interpolate between supports.
        shift = target[..., 1:, 2].mean(-1) - mid[..., 2]
        support_count = contacts.sum(-1)
        known = np.flatnonzero(support_count)
        dz = (shift * contacts).sum(-1) / np.maximum(support_count, 1)
        dz = np.interp(np.arange(n), known, dz[known])
        dz = gaussian_filter1d(dz, cfg.root_smoothing) if cfg.root_smoothing > 0 else dz
        solved = qpos.copy()
        solved[:, 2] += dz
        # Preserve the source's root-to-support reach envelope (plus 5 cm).
        reach = np.max(qpos[:, None, 2] - raw_points[:, :, 0, 2]) + 0.05
        cap = np.where(contacts, target[:, :, 0, 2] + reach, np.inf).min(-1)
        solved[:, 2] = np.minimum(solved[:, 2], cap)
        errors, penetrations, ankle_errors = [], [], []
        for t in range(n):
            previous = solved[t - 1] if t else solved[t]
            seed = solved[t].copy()
            if t:
                seed[self.qadr] = previous[self.qadr]
            solution, score = self._ik(seed, qpos[t], previous, target[t], weights[t])
            self.pose(solution)
            err = np.linalg.norm(self.points() - target[t], axis=-1).max()
            if err > cfg.max_foot_error:
                retry, retry_score = self._ik(solved[t], qpos[t], previous, target[t], weights[t])
                if retry_score < score:
                    solution = retry
            solved[t] = solution
            self.pose(solution)
            actual = self.points()
            errors.append(float(np.linalg.norm(actual - target[t], axis=-1).max()))
            # Official coarse filtering measures root/ankle origins, not toe/heel.
            # Our root stays fixed to its reconstructed target during leg IK.
            ankle_errors.append(float(np.linalg.norm(actual[:, 0] - target[t, :, 0], axis=-1).max()))
            penetration = np.maximum(self.height(actual[:, 1:, :2]) - actual[:, 1:, 2], 0).max()
            for f, knee in enumerate(self.knee_ids):
                for point in (
                    self.data.xpos[knee],
                    (self.data.xpos[knee] + self.data.xpos[self.foot_ids[f]]) / 2,
                ):
                    penetration = max(penetration, float(self.height(point[:2])) + 0.04 - point[2])
            penetrations.append(float(max(0, penetration)))
        motion = ResampledMotion(
            fps,
            solved[:, :3],
            solved[:, 3:7],
            solved[:, 7:],
            np.gradient(solved[:, :3], 1 / fps, axis=0),
            _so3_derivative(solved[:, 3:7], 1 / fps),
            np.gradient(solved[:, 7:], 1 / fps, axis=0),
        )
        bodies = forward_kinematics(self.model, motion)
        result = dict(raw)
        result.update(
            joint_pos=motion.joint_pos.astype(np.float32),
            joint_vel=motion.joint_vel.astype(np.float32),
            body_pos_w=bodies.pos_w.astype(np.float32),
            body_quat_w=bodies.quat_w.astype(np.float32),
            body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
            body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
            body_names=np.array(bodies.body_names),
            tcrs_contacts=contacts,
            tcrs_foot_targets=target.astype(np.float32),
            tcrs_root_offset=dz.astype(np.float32),
            tcrs_ik_position_error_m=np.asarray(ankle_errors, dtype=np.float32),
        )
        jump = float(np.abs(np.diff(solved[:, self.qadr], axis=0)).max())
        metrics = dict(
            frames=n,
            contact_fraction=float(contacts.mean()),
            foot_error_max=max(errors),
            penetration_max=max(penetrations),
            joint_jump_max=jump,
            upper_body_joint_error=float(np.abs(solved[:, 19:] - qpos[:, 19:]).max()),
            root_orientation_error=float(np.abs(solved[:, 3:7] - qpos[:, 3:7]).max()),
        )
        legacy_accepted = bool(
            max(errors) <= cfg.max_foot_error
            and max(penetrations) <= cfg.max_penetration
            and jump <= cfg.max_joint_jump
        )
        if cfg.acceptance_rule == "pmt_batch":
            from supermimic.perception.filtering import pmt_ik_acceptance

            metrics.update(pmt_ik_acceptance(
                ankle_errors, cfg.batch_reject_ik_mm, cfg.batch_reject_ik_bad_frames
            ))
            metrics["legacy_geometric_diagnostic_pass"] = legacy_accepted
        else:
            metrics.update(accepted=legacy_accepted, acceptance_rule="legacy")
        return result, metrics
