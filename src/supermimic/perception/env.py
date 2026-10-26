"""Environment bridge: raw student commands, privileged adapted supervision."""

from dataclasses import dataclass, field

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.state_io import StateIO
from supermimic.perception.terrain import TerrainCfg, TerrainField
from supermimic.world_model.wm_trainer import reference_at


class PerceptiveMotionCommand(MultiClipMotionCommand):
    def __init__(self, cfg, env):
        self.terrain = TerrainField(cfg.terrain, env.device)
        super().__init__(cfg, env)
        self.tracking_library = self.library
        if cfg.target_npz_files is not None:
            self.tracking_library = MotionLibrary(
                cfg.target_npz_files,
                cfg.body_names,
                self.robot.data.default_joint_pos[0],
                device=env.device,
                bin_frames=cfg.bin_frames,
                compact=cfg.compact_library,
            )
            raw_layout = [(c.name, c.start, c.length) for c in self.library.clips]
            target_layout = [(c.name, c.start, c.length) for c in self.tracking_library.clips]
            if raw_layout != target_layout:
                raise ValueError(
                    "raw/adapted libraries must have identical names, order and frame counts"
                )
        # Training-only side channel. goal_window() still reads the raw library.
        self.library.target_library = self.tracking_library
        from supermimic.data.motion_convert import load_g1_model
        from supermimic.kinematics.g1_fk import G1Kinematics
        from supermimic.world_model.heads import AnalyticHeads

        self.state_io = StateIO(env)
        self.heads = AnalyticHeads(
            G1Kinematics(load_g1_model(), device=env.device),
            cfg.body_names,
            cfg.anchor_body_name,
            cfg.root_body_name,
            self.robot.data.soft_joint_pos_limits[0],
            self.robot.data.default_joint_pos[0],
            terrain=self.terrain,
            reference_is_adapted=cfg.reference_is_adapted,
        )

    def _write_reference_state_to_sim(
        self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    ):
        # Reference-state initialization uses the feasible target while preserving reset noise.
        # The actor's command after reset remains raw for the student.
        if self.tracking_library is not self.library:
            raw = self.library.ref(self.frame[env_ids])
            target = self.tracking_library.ref(self.frame[env_ids])
            root_pos = root_pos + target.root_pos_w - raw.root_pos_w
            root_lin_vel = root_lin_vel + target.root_lin_vel_w - raw.root_lin_vel_w
            root_ang_vel = root_ang_vel + target.root_ang_vel_w - raw.root_ang_vel_w
            joint_pos = joint_pos + target.joint_pos - raw.joint_pos
            joint_vel = joint_vel + target.joint_vel - raw.joint_vel
        super()._write_reference_state_to_sim(
            env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
        )


@dataclass(kw_only=True)
class PerceptiveMotionCommandCfg(MultiClipMotionCommandCfg):
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    target_npz_files: list[str] | None = None
    reference_is_adapted: bool = False

    def build(self, env):
        return PerceptiveMotionCommand(self, env)


def terrain_observation(env, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    return cmd.terrain.observation(
        cmd.state_io.read(), noisy=env.cfg.observations["terrain"].enable_corruption
    )


def terrain_reward(env, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    terms = cmd.heads.reward(
        cmd.state_io.read(),
        env.action_manager.action,
        env.action_manager.prev_action,
        reference_at(cmd.tracking_library, cmd.frame),
    )
    return terms["total"] / env.step_dt


def terrain_height_termination(env, command_name="motion", threshold=0.25):
    cmd = env.command_manager.get_term(command_name)
    ref = reference_at(cmd.tracking_library, cmd.frame)
    root = cmd.robot_root_pos_w - env.scene.env_origins
    return cmd.heads.termination_features(root, cmd.robot_root_quat_w, ref)[:, 0].abs() > threshold


def goal_for_library(cmd, library, anchor_obs: bool):
    """Teacher command at the student's current simulator state and motion frame."""
    goal = library.goal_window(cmd.frame, cmd.cfg.future_frames)
    if anchor_obs:
        from supermimic.world_model.heads import anchor_relative_obs

        ref = library.bodies(cmd.frame)
        ai = cmd.motion_anchor_body_index
        relative = anchor_relative_obs(
            cmd.robot_anchor_pos_w - cmd._env.scene.env_origins,
            cmd.robot_anchor_quat_w,
            ref.pos_w[:, ai],
            ref.quat_w[:, ai],
        )
        goal = torch.cat([goal, relative], -1)
    return goal
