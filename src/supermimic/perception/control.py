"""Align teacher/student labels through the physical joint-position target."""

import torch


def control_contract(env):
    action = env.action_manager.get_term("joint_pos")

    def row(value):
        t = torch.as_tensor(value).detach().cpu()
        return (t[0] if t.ndim > 1 else t).reshape(-1).tolist()

    return {
        "joint_names": list(env.scene["robot"].joint_names),
        "scale": row(action._scale),
        "offset": row(action._offset),
        "control_dt": env.step_dt,
        "action_type": "joint_position",
    }


def aligned_action(action, teacher, student):
    if teacher["joint_names"] != student["joint_names"]:
        raise ValueError("teacher/student joint ordering differs")
    ts, to = action.new_tensor(teacher["scale"]), action.new_tensor(teacher["offset"])
    ss, so = action.new_tensor(student["scale"]), action.new_tensor(student["offset"])
    if (ss.abs() < 1e-8).any():
        raise ValueError("student action scale contains zero")
    return (to + ts * action - so) / ss


def pd_target_loss(prediction, aligned_target, student_contract):
    """MSE in physical joint-target radians, independent of normalized action scales."""
    scale = prediction.new_tensor(student_contract["scale"])
    return ((prediction - aligned_target) * scale).square().mean()
