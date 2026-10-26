"""Bounded three-teacher comparison, quality gate, DAgger, BPTT and evaluation.

Run from the repository root with the prepared paired dataset on local SSD.
The process manages only this experiment's verified workers; PHP and other jobs are untouched.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'logs/overnight_20261027'
DATA = ROOT / 'data/overnight_walk_20261027'
PYTHON = ROOT / '.venv/bin/python'
RUN.mkdir(parents=True, exist_ok=True)
LOCK = (RUN / 'supervisor.lock').open('w')
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
PLAN = json.loads((RUN / 'plan.json').read_text())
children = {}
state = dict(phase='starting', pid=os.getpid(), started_unix=time.time(), jobs={}, evaluations=[], plan=PLAN)
stopping = False


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def save():
    state['updated_unix'] = time.time()
    state['disk_free_gib'] = shutil.disk_usage(ROOT).free / 2**30
    write_json(RUN / 'status.json', state)


def environment(gpu):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1',
               PYTHONNOUSERSITE='1', VIRTUAL_ENV=str(ROOT / '.venv'), CUDA_VISIBLE_DEVICES=str(gpu),
               MUJOCO_GL='egl', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TORCHINDUCTOR_COMPILE_THREADS='2', TMPDIR=str(ROOT / '.tmp'),
               XDG_CACHE_HOME=str(ROOT / '.cache'), WARP_CACHE_PATH=str(ROOT / '.cache/warp'),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / '.cache/torchinductor'),
               TRITON_CACHE_DIR=str(ROOT / '.cache/triton'), HF_HOME=str(ROOT / '.cache/huggingface'),
               CUDA_CACHE_PATH=str(ROOT / '.cache/nv'))
    env['PATH'] = str(ROOT / '.venv/bin') + ':' + str(ROOT / '.venv/lib/python3.12/site-packages/ninja/data/bin') + ':' + env['PATH']
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    return env


def launch(name, job, gpu):
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    job_file = directory / 'job.json'
    write_json(job_file, job)
    with (directory / 'console.log').open('a') as log:
        proc = subprocess.Popen([str(PYTHON), str(HERE / 'worker.py'), str(job_file)],
                                cwd=ROOT, env=environment(gpu), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children[name] = proc
    state['jobs'][name] = dict(pid=proc.pid, gpu=gpu, job=job, status='running', started_unix=time.time())
    print('START', name, 'GPU', gpu, 'PID', proc.pid, flush=True)
    save()


def collect():
    for name, proc in list(children.items()):
        code = proc.poll()
        if code is None:
            continue
        entry = state['jobs'][name]
        entry.update(returncode=code, status='completed' if code == 0 else 'failed', ended_unix=time.time())
        report = Path(entry['job']['directory']) / 'status.json'
        if report.exists():
            entry['report'] = json.loads(report.read_text())
        del children[name]
        print('EXIT', name, code, flush=True)


class AdoptedWorker:
    """Track a worker that outlived its supervisor without launching it again."""
    def __init__(self, entry):
        self.pid = entry['pid']
        self.directory = Path(entry['job']['directory'])
        self.start_time = None
        try:
            fields = self.process_fields()
            if fields[0] != 'Z':
                if not self.matches_command():
                    raise RuntimeError(f'Worker identity mismatch for PID {self.pid}')
                self.start_time = fields[19]
        except (FileNotFoundError, ProcessLookupError):
            pass

    def process_fields(self):
        return Path(f'/proc/{self.pid}/stat').read_text().rsplit(')', 1)[1].split()

    def matches_command(self):
        proc = Path(f'/proc/{self.pid}')
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        return (proc.stat().st_uid == os.getuid()
                and str(HERE / 'worker.py').encode() in argv
                and str(self.directory / 'job.json').encode() in argv
                and os.getpgid(self.pid) == self.pid)

    def poll(self):
        try:
            fields = self.process_fields()
            if (self.start_time is not None and fields[19] == self.start_time
                    and fields[0] != 'Z' and self.matches_command()):
                return None
        except (FileNotFoundError, ProcessLookupError):
            pass
        report = self.directory / 'status.json'
        if report.exists() and json.loads(report.read_text()).get('phase') == 'completed':
            return 0
        return 1


def restore_teacher_supervision():
    global state
    saved = json.loads((RUN / 'status.json').read_text())
    if saved['phase'] != 'teachers' or saved.get('selected_teacher_evaluation'):
        raise RuntimeError('Supervisor adoption currently supports the teacher phase only')
    adopted = {name: AdoptedWorker(entry) for name, entry in saved['jobs'].items()
               if entry['status'] == 'running'}
    saved.setdefault('supervisor_handoffs', []).append(dict(
        previous_pid=saved['pid'], new_pid=os.getpid(), unix=time.time()))
    saved['pid'] = os.getpid()
    state = saved
    children.update(adopted)
    print('ADOPTED', {name: proc.pid for name, proc in children.items()}, flush=True)
    save()


def request_stop_teachers():
    for name in children:
        if name.startswith('teacher_'):
            (Path(state['jobs'][name]['job']['directory']) / 'stop.request').touch()


def terminate(signum=None, frame=None):
    global stopping
    stopping = True
    for proc in children.values():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)


def qualified(metrics):
    return (metrics['train']['completion_rate'] >= 0.8
            and metrics['train']['mean_norm_return'] >= 0.65
            and metrics['validation']['completion_rate'] >= 0.6
            and metrics['validation']['mean_norm_return'] >= 0.6)


def score(metrics):
    return sum(m['completion_rate'] * 2 + m['mean_frame_fraction'] + m['mean_norm_return']
               for split, m in metrics.items() if split in ('train', 'validation'))


def snapshot_candidate(names, seen):
    candidates = []
    last_evaluated = {record['source']: index for index, record in enumerate(state['evaluations'])}
    for name in names:
        job = state['jobs'].get(name)
        if job is None:
            continue
        directory = Path(job['job']['directory'])
        numbered = sorted(directory.glob('ckpt_[0-9]*.pt'))
        if job['status'] == 'completed' and (directory / 'ckpt_latest.pt').exists():
            checkpoint = directory / 'ckpt_latest.pt'
            key = str(checkpoint)
        elif numbered:
            checkpoint = numbered[-1]
            key = str(checkpoint)
        else:
            continue
        if key not in seen:
            # Checkpoint age favors slow writers and can starve a faster teacher.
            candidates.append((last_evaluated.get(name, -1), name, checkpoint, key))
    return min(candidates) if candidates else None


def main(resume_teachers=False):
    if not (DATA / 'ready.json').exists():
        raise RuntimeError('TCRS dataset has not completed quality checks')
    if shutil.disk_usage(ROOT).free < 4 * 2**30:
        raise RuntimeError('At least 4 GiB free SSD space is required')
    if resume_teachers:
        restore_teacher_supervision()
        latest_per_gpu = {}
        for name, entry in state['jobs'].items():
            if name.startswith('teacher_'):
                latest_per_gpu[entry['gpu']] = name
        teacher_names = list(latest_per_gpu.values())
    else:
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if len(usage.splitlines()) < 4 or any(int(v) > 500 for v in usage.splitlines()[:4]):
            raise RuntimeError('This experiment requires four currently idle GPUs')
        teacher_names = []
        for gpu, spec in enumerate(PLAN['teachers']):
            name = spec['name']
            teacher_names.append(name)
            job = dict(kind='train', stage='teacher', directory=str(RUN / name),
                       pairs=str(DATA / 'train'), deadline=PLAN['teacher_deadline'], **spec)
            launch(name, job, gpu)
        state['phase'] = 'teachers'
    evaluated = {entry['job']['source_checkpoint'] for entry in state['jobs'].values()
                 if 'source_checkpoint' in entry['job']}
    processed = {record['evaluation'] for record in state['evaluations']}
    pending = [name for name in state['jobs'] if name.startswith('evaluation_') and name not in processed]
    if len(pending) > 1:
        raise RuntimeError('Multiple unprocessed evaluations require inspection')
    evaluation_name = pending[0] if pending else None
    counter = max((int(name.rsplit('_', 1)[1]) for name in state['jobs'] if name.startswith('evaluation_')), default=0)
    best_teacher, best_student = state.get('best_teacher'), state.get('best_student')
    selected_teacher = None
    last_evaluation = 0
    extended = bool(state.get('teacher_extension'))
    while not stopping:
        collect()
        if shutil.disk_usage(ROOT).free < 1.5 * 2**30:
            raise RuntimeError('SSD free space below 1.5 GiB; preserving saved checkpoints')
        if time.time() >= PLAN['hard_deadline']:
            raise TimeoutError('Overnight deadline reached; preserving all retained checkpoints')
        if evaluation_name and evaluation_name not in children:
            entry = state['jobs'][evaluation_name]
            if entry['status'] != 'completed':
                raise RuntimeError(f'Evaluation failed: {evaluation_name}')
            report = entry['report']['result']
            metrics = report['metrics']
            group = 'teacher' if entry['job']['source'].startswith('teacher_') else 'student'
            record = dict(source=entry['job']['source'], evaluation=evaluation_name,
                          metrics=metrics, score=score(metrics), qualified=qualified(metrics))
            state['evaluations'].append(record)
            best = best_teacher if group == 'teacher' else best_student
            if best is None or record['score'] > best['score']:
                destination = RUN / f'best_{group}.pt'
                shutil.copy2(report['checkpoint'], destination.with_suffix('.tmp.pt'))
                destination.with_suffix('.tmp.pt').replace(destination)
                record['checkpoint'] = str(destination)
                if group == 'teacher':
                    best_teacher = record
                    state['best_teacher'] = record
                else:
                    best_student = record
                    state['best_student'] = record
            if group == 'teacher' and record['qualified'] and selected_teacher is None:
                selected_teacher = RUN / 'selected_teacher.pt'
                shutil.copy2(report['checkpoint'], selected_teacher)
                state['selected_teacher_evaluation'] = record
                request_stop_teachers()
                launch('dagger', dict(kind='distill', directory=str(RUN / 'dagger'),
                       teacher=str(selected_teacher), pairs=str(DATA / 'train')), 3)
                state['phase'] = 'dagger'
            Path(report['checkpoint']).unlink()
            processed.add(evaluation_name)
            evaluation_name = None
        if state['phase'] == 'dagger' and 'dagger' not in children:
            entry = state['jobs']['dagger']
            if entry['status'] != 'completed':
                raise RuntimeError('DAgger failed; teacher remains available')
            if any(name.startswith('teacher_') for name in children):
                save()
                time.sleep(30)
                continue
            launch('student', dict(kind='train', stage='student', directory=str(RUN / 'student'),
                   pairs=str(DATA / 'train'), deadline=PLAN['student_deadline'], lr=1e-5,
                   wm_warmup_updates=200, init_policy=entry['report']['result']['checkpoint'],
                   init_world_model=str(selected_teacher)), 0)
            state['phase'] = 'student'
        if state['phase'] in ('teachers', 'student') and not evaluation_name:
            names = teacher_names if state['phase'] == 'teachers' else ['student']
            all_finished = all(name not in children for name in names)
            candidate = snapshot_candidate(names, evaluated)
            if candidate and (time.time() - last_evaluation >= PLAN['evaluation_interval_seconds'] or all_finished):
                _, source, checkpoint, key = candidate
                counter += 1
                evaluation_name = f'evaluation_{counter:03d}'
                directory = RUN / evaluation_name
                directory.mkdir(exist_ok=True)
                snapshot = directory / 'snapshot.pt'
                try:
                    shutil.copy2(checkpoint, snapshot)
                except FileNotFoundError:
                    evaluation_name = None
                    continue
                evaluated.add(key)
                launch(evaluation_name, dict(kind='evaluate', directory=str(directory),
                       checkpoint=str(snapshot), source=source, source_checkpoint=str(checkpoint),
                       datasets={split: str(DATA / split) for split in ('train', 'validation')}), 3)
                last_evaluation = time.time()
            elif all_finished and not candidate:
                if state['phase'] == 'teachers':
                    if not extended and best_teacher and time.time() < PLAN['student_deadline'] - 3600:
                        extended = True
                        new_names = []
                        for gpu, name in enumerate(teacher_names):
                            previous = state['jobs'][name]
                            checkpoint = Path(previous['job']['directory']) / 'ckpt_latest.pt'
                            if not checkpoint.exists():
                                continue
                            new_name = name + '_extended'
                            job = dict(previous['job'], directory=str(RUN / new_name),
                                       resume=str(checkpoint), deadline=PLAN['student_deadline'],
                                       lr=previous['job']['lr'] * 0.5)
                            launch(new_name, job, gpu)
                            new_names.append(new_name)
                        if not new_names:
                            raise RuntimeError('No valid teacher checkpoint for continuation')
                        teacher_names = new_names
                        state['teacher_extension'] = 'No teacher met the gate; continuing at half learning rate'
                    else:
                        state['phase'] = 'teacher_quality_not_met'
                        save()
                        return
                else:
                    if best_student is None:
                        raise RuntimeError('No valid student evaluation')
                    directory = RUN / 'final_evaluation'
                    launch('final_evaluation', dict(kind='evaluate', directory=str(directory),
                           checkpoint=best_student['checkpoint'], source='student',
                           datasets={split: str(DATA / split) for split in ('train', 'validation')},
                           ablations=['none', 'zero', 'shuffle']), 3)
                    state['phase'] = 'final_evaluation'
        if state['phase'] == 'final_evaluation' and 'final_evaluation' not in children:
            entry = state['jobs']['final_evaluation']
            if entry['status'] != 'completed':
                raise RuntimeError('Final evaluation failed')
            subprocess.run([str(PYTHON), str(ROOT / 'scripts/export_perceptive.py'),
                            '--ckpt', best_student['checkpoint'], '--out', str(RUN / 'student.ts')],
                           cwd=ROOT, env=environment(''), check=True, timeout=180)
            state.update(phase='completed', final_evaluation=entry['report']['result'],
                         exported_actor=str(RUN / 'student.ts'))
            save()
            return
        if state['phase'] == 'teachers' and all(name not in children for name in teacher_names):
            if all(state['jobs'][name]['status'] == 'failed' for name in teacher_names):
                raise RuntimeError('All teacher jobs failed; inspect their console logs')
        save()
        time.sleep(30)


if __name__ == '__main__':
    resume_teachers = sys.argv[1:] == ['--resume-teachers']
    if sys.argv[1:] and not resume_teachers:
        raise SystemExit('Usage: supervise.py [--resume-teachers]')
    if (RUN / 'status.json').exists() and not resume_teachers:
        raise SystemExit('Existing run state: inspect it before relaunching this supervisor')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        main(resume_teachers=resume_teachers)
    except BaseException as exc:
        state.update(phase='failed', error=str(exc), traceback=traceback.format_exc())
        save()
        terminate()
        raise
    finally:
        if stopping:
            state['stopped_unix'] = time.time()
            save()
