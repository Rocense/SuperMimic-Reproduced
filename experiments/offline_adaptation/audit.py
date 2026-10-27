"""Offline raw/TCRS geometry audit, with real collision queries and no policy training."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from supermimic.perception.paired import load_pairs
from supermimic.perception.tcrs import FOOT_POINTS, Synthesizer, TCRSCfg, intervals
from supermimic.perception.terrain import TerrainField

ROOT = Path("data/overnight_walk_20261027")
OUT = Path("logs/offline_adaptation_20261027")


def load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def points(motion):
    names = motion["body_names"].astype(str).tolist()
    ids = [names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    quat = motion["body_quat_w"][:, ids]
    rot = Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
    rot = rot.reshape(len(quat), 2, 3, 3)
    return motion["body_pos_w"][:, ids, None] + np.einsum("tfij,pj->tfpi", rot, FOOT_POINTS)


def collision_model(terrain):
    from mjlab.scene import Scene

    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import make_env_cfg
    args = SuperMimicEnvCfgArgs(npz_files=[], num_envs=1, terrain=terrain,
                               record_transitions=False)
    model = Scene(make_env_cfg(args).scene, "cpu").compile()
    assert model.nq == 36
    return model


def collision_metrics(motion, model):
    data = mujoco.MjData(model)
    names = motion["body_names"].astype(str).tolist()
    pelvis = names.index("pelvis")
    qpos = np.concatenate([motion["body_pos_w"][:, pelvis],
                           motion["body_quat_w"][:, pelvis], motion["joint_pos"]], axis=-1)
    # The compiled training scene namespaces names, but retains XML joint order.
    joints = [model.joint(i).name.split("/")[-1] for i in range(1, model.njnt)]
    assert joints == motion["joint_names"].astype(str).tolist()
    deepest = np.zeros(len(qpos))
    nonfoot = np.zeros(len(qpos))
    worst = {"depth_m": 0.0}
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for contact in data.contact[:data.ncon]:
            g1, g2 = map(int, contact.geom)
            world1, world2 = model.geom_bodyid[g1] == 0, model.geom_bodyid[g2] == 0
            if world1 == world2:
                continue
            depth = max(0.0, -float(contact.dist))
            robot_geom = model.geom(g2 if world1 else g1).name
            deepest[t] = max(deepest[t], depth)
            if "foot" not in robot_geom:
                nonfoot[t] = max(nonfoot[t], depth)
            if depth > worst["depth_m"]:
                worst = {"depth_m": depth, "frame": t, "geom": robot_geom}
    return {"max_geom_penetration_m": float(deepest.max()),
            "mean_frame_deepest_penetration_m": float(deepest.mean()),
            "geom_penetration_frames_gt_1mm": float((deepest > 0.001).mean()),
            "nonfoot_max_geom_penetration_m": float(nonfoot.max()), "worst": worst}


def quality(motion, contacts, field, model):
    foot = points(motion)
    height = field.height(torch.tensor(foot[..., :2], dtype=torch.float32)).numpy()
    gap = foot[..., 2] - height
    # Toe/heel signed clearance, foot-level contact uses the lower of the two.
    sole_gap = gap[..., 1:].min(-1)
    names = motion["body_names"].astype(str).tolist()
    knees = motion["body_pos_w"][:, [names.index(f"{s}_knee_link") for s in ("left", "right")]]
    shin = np.stack([knees, (knees + foot[:, :, 0]) / 2], axis=2)
    shin_gap = shin[..., 2] - 0.04 - field.height(torch.tensor(shin[..., :2], dtype=torch.float32)).numpy()
    collision_gaps = np.concatenate([gap[..., 1:], shin_gap], axis=-1)
    acceleration = np.diff(foot[:, :, 0], n=2, axis=0) * 50**2
    slip = []
    for f in range(2):
        mid = foot[:, f, 1:].mean(1)
        for lo, hi in intervals(contacts[:, f]):
            slip.append(float(np.linalg.norm(mid[lo:hi, :2] - mid[lo, :2], axis=-1).max()))
    return {
        "point_penetration_mean_m": float(np.maximum(-collision_gaps, 0).mean()),
        "point_penetration_max_m": float(np.maximum(-collision_gaps, 0).max()),
        "stance_contact_abs_error_m": float(np.abs(sole_gap[contacts]).mean()),
        "stance_float_fraction_gt_2cm": float((sole_gap[contacts] > 0.02).mean()),
        "swing_clearance_violation_fraction_lt_3_5cm": float((sole_gap[~contacts] < 0.035).mean()),
        "foot_acceleration_mean_m_s2": float(np.linalg.norm(acceleration, axis=-1).mean()),
        "stance_slip_max_m": max(slip, default=0),
        "terrain_height_range_m": float(np.ptp(height)),
        "terrain_height_max_m": float(height.max()),
        **collision_metrics(motion, model),
    }


def compare(raw, adapted, cfg, model):
    field = TerrainField(cfg)
    contacts = adapted["tcrs_contacts"]
    names = raw["body_names"].astype(str).tolist()
    root = names.index("pelvis")
    upper = [i for i, name in enumerate(names) if any(p in name for p in ("torso", "shoulder", "elbow", "wrist"))]
    joint_delta = adapted["joint_pos"] - raw["joint_pos"]
    root_dz = adapted["body_pos_w"][:, root, 2] - raw["body_pos_w"][:, root, 2]
    return {"raw": quality(raw, contacts, field, model),
            "adapted": quality(adapted, contacts, field, model),
            "preservation": {
                "frames": len(raw["joint_pos"]),
                "upper_joint_max_error_rad": float(np.abs(joint_delta[:, 12:]).max()),
                "root_orientation_max_abs_error": float(np.abs(adapted["body_quat_w"][:, root] - raw["body_quat_w"][:, root]).max()),
                "upper_world_displacement_mean_m": float(np.linalg.norm(adapted["body_pos_w"][:, upper] - raw["body_pos_w"][:, upper], axis=-1).mean()),
                "leg_joint_change_rms_rad": float(np.sqrt(np.mean(joint_delta[:, :12]**2))),
                "root_z_offset_range_m": [float(root_dz.min()), float(root_dz.max())],
            }}


def stress(spec):
    torch.set_num_threads(1)
    split, name, height = spec
    _, _, old, _ = load_pairs(ROOT / split, [name])
    cfg = replace(old, stairs_surface="boxes", height=height)
    raw = load(ROOT / split / "raw" / f"{name}.npz")
    adapted, metrics = Synthesizer(TerrainField(cfg), TCRSCfg(acceptance_rule="legacy")).synthesize(raw)
    folder = OUT / "probes" / f"{split}_{name}_{round(height * 1000)}mm"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "adapted.npz", **adapted)
    result = {"split": split, "name": name, "terrain": asdict(cfg), "metrics": metrics,
              "raw_path": str(ROOT / split / "raw" / f"{name}.npz"),
              "adapted_path": str(folder / "adapted.npz"),
              **compare(raw, adapted, cfg, collision_model(cfg))}
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    historical = []
    probes = []
    for split in ("train", "validation"):
        raw_paths, adapted_paths, cfg, manifest = load_pairs(ROOT / split)
        model = collision_model(cfg)
        for rp, ap, record in zip(raw_paths, adapted_paths, manifest["clips"]):
            row = {"split": split, "name": Path(rp).stem, "metrics": record["metrics"],
                   **compare(load(rp), load(ap), cfg, model)}
            historical.append(row)
        # Four independent source windows, chosen for the largest terrain height
        # variation. This is an explicit stress probe, not an unbiased pass rate.
        selected, seen = [], set()
        for row in sorted([r for r in historical if r["split"] == split],
                          key=lambda r: r["raw"]["terrain_height_range_m"], reverse=True):
            key = row["name"].rsplit("_p", 1)[0]
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) == 2:
                break
        probes.extend((split, row["name"], height) for row in selected for height in (0.025, 0.15))
        print(split, "historical audit done", len(raw_paths), flush=True)
    (OUT / "historical.json").write_text(json.dumps(historical, indent=2) + "\n")
    (OUT / "probe_selection.json").write_text(json.dumps(probes, indent=2) + "\n")
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(stress, spec) for spec in probes]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            (OUT / "probes.json").write_text(json.dumps(results, indent=2) + "\n")
            print(row["split"], row["name"], row["terrain"]["height"], row["metrics"], flush=True)


if __name__ == "__main__":
    main()
