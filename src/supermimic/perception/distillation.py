"""DAgger: adapted-reference teacher labels for a raw-reference student."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.perception.control import aligned_action, control_contract, pd_target_loss
from supermimic.perception.env import goal_for_library
from supermimic.perception.paired import load_pairs, restore_config, sha256
from supermimic.policy.nets import Policy


@dataclass
class DistillConfig:
    teacher: Path
    paired_dir: Path
    output: Path
    steps: int = 240000  # Official main distillation config: 10000 iterations x 24 steps.
    num_envs: int = 256
    buffer_steps: int = 24
    batch_size: int = 512
    updates_per_step: int = 1
    lr: float = 1e-4
    teacher_probability_start: float = 1.0
    teacher_probability_end: float = 0.0
    decay_steps: int = 2400  # Official teacher_mix_anneal_iters=100, 24 steps/iteration.
    obs_noise: bool = True
    log_interval: int = 100
    seed: int = 0
    device: str = "cuda:0"
    teacher_filter: Path | None = None
    integration_test: bool = False
    rollout_steps: int = 24
    learning_epochs: int = 4


class AggregationBuffer:
    def __init__(self, capacity, example):
        self.capacity, self.cursor, self.size = capacity, 0, 0
        self.values = {
            k: torch.empty((capacity,) + v.shape[1:], device=v.device, dtype=v.dtype)
            for k, v in example.items()
        }

    @torch.no_grad()
    def add(self, values):
        first = next(iter(values.values()))
        n = first.shape[0]
        if n > self.capacity:
            raise ValueError("buffer is smaller than one environment batch")
        idx = (torch.arange(n, device=first.device) + self.cursor) % self.capacity
        for k, v in values.items():
            self.values[k][idx] = v.detach()
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, count):
        idx = torch.randint(self.size, (count,), device=next(iter(self.values.values())).device)
        return {k: v[idx] for k, v in self.values.items()}

    def minibatches(self, batch_size):
        order = torch.randperm(self.size, device=next(iter(self.values.values())).device)
        for idx in order.split(batch_size):
            yield {k: v[idx] for k, v in self.values.items()}


class EpisodeTeacherMix:
    """Official PMT hard switch: choose control per episode, resample only on reset."""

    def __init__(self, count, probability, device):
        self.mask = torch.rand(count, 1, device=device) < probability

    def apply(self, teacher, student):
        return torch.where(self.mask, teacher, student)

    def reset(self, done, probability):
        ids = done.nonzero().flatten()
        self.mask[ids] = torch.rand(len(ids), 1, device=self.mask.device) < probability


def distill(cfg: DistillConfig):
    if cfg.integration_test and (cfg.steps > 16 or cfg.num_envs > 16):
        raise ValueError("integration-test distillation is limited to <=16 steps/environments")
    if cfg.teacher_filter is None and not cfg.integration_test:
        raise ValueError("run filter_teacher_references.py before formal distillation")
    if cfg.buffer_steps < cfg.rollout_steps or cfg.updates_per_step != 1:
        raise ValueError("PMT distillation uses complete rollout batches and learning_epochs; updates_per_step must be 1")
    if (
        min(
            cfg.steps,
            cfg.num_envs,
            cfg.buffer_steps,
            cfg.batch_size,
            cfg.updates_per_step,
            cfg.rollout_steps,
            cfg.learning_epochs,
            cfg.decay_steps,
            cfg.log_interval,
        )
        < 1
    ):
        raise ValueError("DAgger counts must be positive")
    if not 0 <= cfg.teacher_probability_end <= cfg.teacher_probability_start <= 1:
        raise ValueError("teacher probabilities must anneal within [0, 1]")
    if cfg.output.exists():
        raise FileExistsError("choose a new distillation checkpoint path")
    torch.manual_seed(cfg.seed)
    ck = torch.load(cfg.teacher, map_location=cfg.device, weights_only=False)
    tc = restore_config(ck["config"])
    if tc.stage != "teacher" or tc.policy.terrain_dim:
        raise ValueError("DAgger requires a blind adapted-reference teacher")
    names = tc.data.clips
    if not cfg.integration_test:
        from supermimic.perception.filtering import selected_reference_names

        names = selected_reference_names(cfg.paired_dir, cfg.teacher_filter, cfg.teacher)
    raw, adapted, terrain, _ = load_pairs(
        cfg.paired_dir, names, tc.data.limit if cfg.integration_test else None
    )
    if tc.terrain != terrain:
        raise ValueError("teacher and dataset terrain differ")
    teacher_contract = ck.get("control_contract")
    if teacher_contract is None:
        raise ValueError("teacher is missing its action/PD contract")
    teacher = Policy(tc.policy).to(cfg.device).eval()
    teacher.load_state_dict(ck["policy"])
    teacher.requires_grad_(False)
    student_cfg = copy.deepcopy(tc)
    student_cfg.stage = "student"
    student_cfg.data.paired_dir = cfg.paired_dir
    student_cfg.data.teacher_filter = cfg.teacher_filter
    student_cfg.data.clips = names
    student_cfg.data.limit = None
    student_cfg.integration_test = cfg.integration_test
    student_cfg.policy_trainer.backbone_lr_scale = 0.3
    student_cfg.policy.terrain_dim = terrain.obs_dim
    student_cfg.resume = student_cfg.init_policy = student_cfg.init_world_model = None
    student = Policy(student_cfg.policy).to(cfg.device)
    missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)
    if unexpected or any(not k.startswith("terrain_") for k in missing):
        raise ValueError("teacher/student backbone mismatch")
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=raw,
            target_npz_files=adapted,
            terrain=terrain,
            reference_is_adapted=True,
            num_envs=cfg.num_envs,
            future_frames=tc.env.future_frames,
            history_length=tc.policy.history,
            episode_length_s=tc.env.episode_length_s,
            action_noise_std=0.0,
            obs_noise=cfg.obs_noise,
            reset_noise=tc.env.reset_noise,
            domain_randomization=tc.env.domain_randomization,
            sampling_mode="uniform",
            anchor_obs=tc.env.anchor_obs,
            record_transitions=False,
            pmt_tracking_evaluation=not cfg.integration_test,
            seed=cfg.seed,
        ),
        device=cfg.device,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    metrics = []
    try:
        command = env.command_manager.get_term("motion")
        student_contract = control_contract(env)
        obs, _ = env.reset()
        buffer = None
        mixer = EpisodeTeacherMix(cfg.num_envs, cfg.teacher_probability_start, cfg.device)
        last_loss = None
        for step in range(cfg.steps):
            with torch.no_grad():
                teacher_goal = goal_for_library(
                    command, command.tracking_library, tc.env.anchor_obs
                )
                target = aligned_action(
                    teacher(obs["policy"], teacher_goal), teacher_contract, student_contract
                )
                values = {
                    "history": obs["policy"],
                    "goal": obs["goal"],
                    "terrain": obs["terrain"],
                    "target": target,
                }
                if buffer is None:
                    buffer = AggregationBuffer(cfg.num_envs * cfg.buffer_steps, values)
                buffer.add(values)
            rollout_start = step // cfg.rollout_steps * cfg.rollout_steps
            beta = cfg.teacher_probability_start + min(rollout_start / cfg.decay_steps, 1.0) * (
                cfg.teacher_probability_end - cfg.teacher_probability_start
            )
            with torch.no_grad():
                student_action = student(obs["policy"], obs["goal"], terrain=obs["terrain"])
                applied = mixer.apply(target, student_action)
                teacher_fraction = float(mixer.mask.float().mean())
                obs, _, terminated, timeout, _ = env.step(applied)
                mixer.reset(terminated | timeout, beta)
            if (step + 1) % cfg.rollout_steps == 0 or step + 1 == cfg.steps:
                for _ in range(cfg.learning_epochs):
                    for batch in buffer.minibatches(cfg.batch_size):
                        pred = student(batch["history"], batch["goal"], terrain=batch["terrain"])
                        loss = pd_target_loss(pred, batch["target"], student_contract)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                        optimizer.step()
                        last_loss = float(loss.detach())
                buffer.cursor = buffer.size = 0
            if (step + 1) % cfg.log_interval == 0 or step + 1 == cfg.steps:
                row = {
                    "step": step + 1,
                    "pd_target_mse_rad2": last_loss,
                    "teacher_probability": beta,
                    "teacher_control_fraction": teacher_fraction,
                    "buffer_samples": buffer.size,
                }
                metrics.append(row)
                print(json.dumps(row), flush=True)
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": student.state_dict(),
                "config": asdict(student_cfg),
                "control_contract": student_contract,
                "stage": "distilled_student",
                "teacher_checkpoint": str(cfg.teacher),
                "teacher_filter_sha256": sha256(cfg.teacher_filter)
                    if cfg.teacher_filter is not None else None,
                "distillation": asdict(cfg),
                "metrics": metrics,
            },
            cfg.output,
        )
        return cfg.output
    finally:
        env.close()
