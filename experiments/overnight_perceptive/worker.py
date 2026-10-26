"""One isolated training/evaluation stage for the bounded overnight experiment."""
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import torch

from supermimic.config import TrainConfig
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results
from supermimic.perception.distillation import DistillConfig, distill
from supermimic.perception.paired import load_pairs, restore_config
from supermimic.training.loop import Trainer


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def extend_resume_budget(schedule, completed_iteration):
    """Allow another segment without moving the original LR decay endpoint."""
    decay_steps = schedule.total_iterations * schedule.lr_decay_fraction
    while schedule.total_iterations < completed_iteration + 50000:
        schedule.total_iterations *= 2
    schedule.lr_decay_fraction = decay_steps / schedule.total_iterations


class CheckedTrainer(Trainer):
    """Atomic snapshots, bounded retention, and cooperative stopping for this run."""
    def __init__(self, cfg, deadline):
        self.deadline = deadline
        super().__init__(cfg)

    def act(self, obs):
        if self.iteration % 50 == 0:
            if time.time() >= self.deadline or (self.log_dir / 'stop.request').exists():
                self.cfg.schedule.max_hours = 0.0
            if shutil.disk_usage(self.log_dir).free < 2 * 1024**3:
                self.cfg.schedule.max_hours = 0.0
        return super().act(obs)

    def save(self, name='ckpt_latest.pt'):
        for model in (self.policy, self.wm, self.critic):
            if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise FloatingPointError('Nonfinite model parameters; preserving last valid snapshot')
        temporary = '.checkpoint.tmp.pt'
        super().save(temporary)
        (self.log_dir / temporary).replace(self.log_dir / name)
        snapshots = sorted(self.log_dir.glob('ckpt_[0-9]*.pt'))
        for old in snapshots[:-2]:
            old.unlink()
        write_json(self.log_dir / 'progress.json', dict(
            phase='training', iteration=self.iteration, updates=self.num_updates,
            updated_unix=time.time(), checkpoint=str(self.log_dir / name),
            disk_free_gib=shutil.disk_usage(self.log_dir).free / 2**30))


def train(job):
    directory = Path(job['directory'])
    if job['stage'] == 'student':
        ck = torch.load(job['init_policy'], map_location='cpu', weights_only=False)
        cfg = restore_config(ck['config'])
        cfg.init_policy = Path(job['init_policy'])
        cfg.init_world_model = Path(job['init_world_model'])
        cfg.resume = None
    else:
        cfg = TrainConfig(stage='teacher')
        cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (512, 256, 128)
        cfg.world_model.hidden = (512, 512, 512)
    cfg.stage = job['stage']
    cfg.run_name, cfg.log_root = directory.name, directory.parent
    cfg.seed = job.get('seed', 0)
    cfg.device = 'cuda:0'
    cfg.data.paired_dir = Path(job['pairs'])
    cfg.env.num_envs = 512
    cfg.env.replay_capacity = 768
    cfg.env.anchor_obs = True
    cfg.env.obs_noise = cfg.env.reset_noise = cfg.env.domain_randomization = False
    cfg.env.action_noise_std = 0.05
    cfg.env.sampling_mode = 'return'
    cfg.wm_trainer.batch_size = 512
    cfg.wm_trainer.batches_per_update = 10
    cfg.wm_trainer.lr = 2e-4
    cfg.policy_trainer.batch_size = 256
    cfg.policy_trainer.horizon = 16
    cfg.policy_trainer.lr = job['lr']
    cfg.policy_trainer.critic_epochs = 3
    cfg.policy_trainer.use_critic = job.get('use_critic', True)
    cfg.schedule.total_iterations = 50000
    cfg.schedule.lr_decay_fraction = 1.0
    cfg.schedule.lr_final_ratio = 0.1
    cfg.schedule.warmup_iterations = 100
    cfg.schedule.wm_warmup_updates = job.get('wm_warmup_updates', 500)
    cfg.schedule.checkpoint_interval = 1000
    cfg.schedule.log_interval = 50
    cfg.schedule.max_hours = max(0.001, (job['deadline'] - time.time()) / 3600)
    if job.get('resume'):
        cfg.resume = Path(job['resume'])
        cfg.init_policy = cfg.init_world_model = None
        metadata = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        extend_resume_budget(cfg.schedule, int(metadata['iteration']))
        del metadata
    trainer = CheckedTrainer(cfg, job['deadline'])
    try:
        if cfg.resume is None:
            trainer.save('ckpt_initial.pt')
        trainer.run()
        return dict(iteration=trainer.iteration, updates=trainer.num_updates,
                    checkpoint=str(directory / 'ckpt_latest.pt'))
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


def evaluate(job):
    checkpoint = Path(job['checkpoint'])
    policy, cfg = load_policy_from_checkpoint(checkpoint, 'cuda:0')
    output = {}
    for split, directory in job['datasets'].items():
        raw, adapted, terrain, _ = load_pairs(directory)
        student = cfg['stage'] == 'student'
        for ablation in job.get('ablations', ['none']):
            result = run_benchmark(
                policy, raw if student else adapted, target_npz_files=adapted if student else None,
                terrain=terrain, reference_is_adapted=True, num_envs=min(32, len(raw)),
                anchor_obs=cfg['env']['anchor_obs'], future_frames=cfg['env']['future_frames'],
                map_ablation=ablation)
            suffix = split if ablation == 'none' else f'{split}_{ablation}'
            result_path = Path(job['directory']) / f'{suffix}.json'
            save_results(result, result_path)
            output[suffix] = dict(
                clips=len(result), completed=sum(not r.failed for r in result),
                completion_rate=sum(not r.failed for r in result) / len(result),
                mean_frame_fraction=sum(min(1.0, r.steps / r.frames) for r in result) / len(result),
                mean_norm_return=sum(r.norm_return for r in result) / len(result),
                report=str(result_path))
            if not all(math.isfinite(v) for v in output[suffix].values() if isinstance(v, float)):
                raise FloatingPointError('Nonfinite evaluation metric')
    return dict(checkpoint=str(checkpoint), metrics=output)


def distillation(job):
    output = Path(job['directory']) / 'distilled.pt'
    distill(DistillConfig(teacher=Path(job['teacher']), paired_dir=Path(job['pairs']),
                         output=output, steps=10000, num_envs=512, buffer_steps=128,
                         batch_size=1024, updates_per_step=2, decay_steps=8000,
                         obs_noise=False, lr=5e-5))
    ck = torch.load(output, map_location='cpu', weights_only=False)
    if not all(torch.isfinite(t).all().item() for t in ck['policy'].values()):
        raise FloatingPointError('Nonfinite distilled policy')
    return dict(checkpoint=str(output), steps=10000, teacher_probability_end=0.0)


def main(job):
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    base = dict(pid=os.getpid(), started_unix=time.time(), job=job)
    write_json(directory / 'status.json', dict(base, phase='running'))
    try:
        result = {'train': train, 'evaluate': evaluate, 'distill': distillation}[job['kind']](job)
        write_json(directory / 'status.json', dict(base, phase='completed',
                   ended_unix=time.time(), result=result))
    except BaseException as exc:
        write_json(directory / 'status.json', dict(base, phase='failed', ended_unix=time.time(),
                   error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
