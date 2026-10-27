"""Render audited kinematic references, not a simulated controller rollout."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.offline_adaptation.audit import OUT, collision_model, load, points
from supermimic.perception.paired import terrain_config


def main():
    history = json.loads((OUT / "historical.json").read_text())
    probes = json.loads((OUT / "probes.json").read_text())
    names = [name for _, name in sorted({(r["split"], r["name"]) for r in probes})]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for j, height in enumerate((0.025, 0.15)):
        rows = {r["name"]: r for r in probes if r["terrain"]["height"] == height}
        x = np.arange(len(names)) + (j - 0.5) * 0.36
        for ax, values in zip(axes, (
            [rows[n]["metrics"]["foot_error_max"] * 100 for n in names],
            [rows[n]["adapted"]["max_geom_penetration_m"] * 100 for n in names],
        )):
            bars = ax.bar(x, values, width=0.34, label=f"{height * 100:g} cm risers")
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    axes[0].axhline(5.5, color="#a82030", linestyle="--", label="IK rejection threshold")
    axes[0].set_title("Maximum foot target error")
    axes[1].set_title("Actual MuJoCo geometry penetration")
    for ax in axes:
        ax.set_xticks(range(4), ["train A", "train B", "validation A", "validation B"])
        ax.set_ylabel("cm (lower is better)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    fig.suptitle("Offline adaptation stress probe: 4 crossing clips, no policy training")
    fig.savefig(OUT / "height_probe.png", dpi=150)
    plt.close(fig)
    summary = {"historical_clips": len(history), "historical_accepted": sum(r["metrics"]["accepted"] for r in history),
               "aggregation": "Per-clip mean unless explicitly marked global_max; each historical clip has 256 frames.",
               "float_threshold_m": 0.02, "swing_clearance_threshold_m": 0.035,
               "probe_selection": "Four independent source windows with largest terrain height variation, two per split; not a population pass rate."}
    for variant in ("raw", "adapted"):
        summary[variant] = {k: float(np.mean([r[variant][k] for r in history]))
                            for k in history[0][variant] if k != "worst"}
        summary[variant]["global_max_geom_penetration_m"] = max(r[variant]["max_geom_penetration_m"] for r in history)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    row = next(r for r in probes if r["name"] == "walk2_subject1_f1200_p1" and r["terrain"]["height"] == 0.15)
    raw, adapted = load(row["raw_path"]), load(row["adapted_path"])
    model = collision_model(terrain_config(row["terrain"]))
    data = mujoco.MjData(model)
    root = raw["body_names"].astype(str).tolist().index("pelvis")
    body = adapted["body_pos_w"][:, root]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [float(body[:, 0].mean()), float(body[:, 1].mean()), 0.65]
    camera.azimuth, camera.elevation, camera.distance = 0, -15, 4.0
    model.vis.global_.offwidth, model.vis.global_.offheight = 512, 384
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    targets = adapted["tcrs_foot_targets"]
    error = np.linalg.norm(points(adapted) - targets, axis=-1).max(axis=(1, 2))
    worst_frame = int(error.argmax())
    frames = []
    with mujoco.Renderer(model, width=512, height=384) as renderer:
        def render(t):
            canvas = Image.new("RGB", (1024, 424), "#10151e")
            for j, motion in enumerate((raw, adapted)):
                data.qpos[:] = np.r_[motion["body_pos_w"][t, root], motion["body_quat_w"][t, root], motion["joint_pos"][t]]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                if j == 1:
                    for point in targets[t].reshape(-1, 3):
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                           np.full(3, 0.015), point.astype(float),
                                           np.eye(3).ravel(), np.array([1., 0.05, 0.05, 1.]))
                        renderer.scene.ngeom += 1
                canvas.paste(Image.fromarray(renderer.render()), (j * 512, 25))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 4), "RAW: flat-ground motion placed on stairs", font=font, fill="white")
            draw.text((524, 4), "TCRS: 15 cm stairs / REJECTED by IK checks", font=font, fill="#ffb2a8")
            draw.text((12, 408), f"Kinematic replay only | frame {t:03d}/255 | no controller or training", font=font, fill="white")
            draw.text((604, 408), "Red dots: requested foot targets", font=font, fill="#ffb2a8")
            return canvas
        for t in range(0, len(body), 4):
            frames.append(render(t))
        render(worst_frame).save(OUT / "reference_comparison.png")
    frames[0].save(OUT / "reference_comparison.gif", save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    print(json.dumps({"worst_frame": worst_frame, "max_foot_error_m": float(error.max()),
                      "plot": str(OUT / "height_probe.png"),
                      "replay": str(OUT / "reference_comparison.gif")}, indent=2))


if __name__ == "__main__":
    main()
