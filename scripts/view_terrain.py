"""Inspect stair geometry without loading a policy or changing a saved experiment."""
from pathlib import Path
import time
from typing import Literal

import mujoco
import tyro
from mjlab.scene import Scene

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.perception.terrain import TerrainCfg


def main(
    height: float = 0.15,
    stair_width: float = 0.5,
    stair_count: int = 4,
    size: float = 12.0,
    resolution: float = 0.1,
    stairs_surface: Literal['boxes', 'heightfield'] = 'boxes',
    output: Path | None = None,
):
    terrain = TerrainCfg(enabled=True, kind='stairs', height=height, stair_width=stair_width,
                         stair_count=stair_count, size=size, resolution=resolution,
                         stairs_surface=stairs_surface)
    terrain.validate()
    args = SuperMimicEnvCfgArgs(npz_files=[], num_envs=1, terrain=terrain,
                                record_transitions=False)
    model = Scene(make_env_cfg(args).scene, 'cpu').compile()
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def set_camera(camera):
        camera.lookat[:] = [1.9, 0, 0.35]
        camera.distance, camera.azimuth, camera.elevation = 5.5, 140, -20

    print(f'Static preview: surface={stairs_surface}, riser={height:g} m, '
          f'tread={stair_width:g} m, peak={height * stair_count:g} m. '
          'The stationary robot is a scale reference, not a policy rollout.')
    if output is not None:
        from PIL import Image
        model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 800
        camera = mujoco.MjvCamera()
        set_camera(camera)
        with mujoco.Renderer(model, width=1280, height=800) as renderer:
            renderer.update_scene(data, camera=camera)
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(output)
        print(output)
        return
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        set_camera(viewer.cam)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 30)


if __name__ == '__main__':
    tyro.cli(main)
