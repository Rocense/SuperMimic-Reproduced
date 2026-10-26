"""Regression checks for evaluation fairness and worker adoption."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    script = Path(__file__).with_name('supervise.py')
    monkeypatch.chdir(tmp_path)
    run = tmp_path / 'logs/overnight_20261027'
    run.mkdir(parents=True)
    (run / 'plan.json').write_text('{}')
    spec = importlib.util.spec_from_file_location('overnight_supervisor_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module.LOCK.close()


def test_fast_checkpoint_writer_is_not_starved(supervisor, tmp_path):
    jobs = {}
    for index, name in enumerate(('slow', 'middle', 'fast')):
        directory = tmp_path / name
        directory.mkdir()
        checkpoint = directory / 'ckpt_0001000.pt'
        checkpoint.write_bytes(b'test snapshot')
        os.utime(checkpoint, (100 + index, 100 + index))
        jobs[name] = dict(status='running', job=dict(directory=str(directory)))
    supervisor.state.update(jobs=jobs, evaluations=[dict(source='slow'), dict(source='middle')])
    assert supervisor.snapshot_candidate(list(jobs), set())[1] == 'fast'
    supervisor.state['evaluations'].append(dict(source='fast'))
    assert supervisor.snapshot_candidate(list(jobs), set())[1] == 'slow'
    # A completed worker's final snapshot takes precedence over numbered snapshots.
    final = tmp_path / 'slow/ckpt_latest.pt'
    final.write_bytes(b'final')
    jobs['slow']['status'] = 'completed'
    assert supervisor.snapshot_candidate(list(jobs), set())[2] == final
    assert supervisor.snapshot_candidate(list(jobs), {str(final)})[1] == 'middle'


def test_adoption_preserves_running_worker_and_uses_its_completion_report(supervisor, tmp_path):
    supervisor.HERE = tmp_path / 'worker_source'
    supervisor.HERE.mkdir()
    worker = supervisor.HERE / 'worker.py'
    worker.write_text('import time\ntime.sleep(30)\n')
    directory = tmp_path / 'teacher'
    directory.mkdir()
    job = directory / 'job.json'
    job.write_text('{}')
    proc = subprocess.Popen([sys.executable, str(worker), str(job)], start_new_session=True)
    try:
        entry = dict(pid=proc.pid, status='running', job=dict(directory=str(directory)), gpu=0)
        saved = dict(phase='teachers', pid=123, jobs=dict(teacher_test=entry), evaluations=[])
        (supervisor.RUN / 'status.json').write_text(json.dumps(saved))
        supervisor.restore_teacher_supervision()
        adopted = supervisor.children['teacher_test']
        assert adopted.pid == proc.pid and adopted.poll() is None
        assert proc.poll() is None
        assert supervisor.state['supervisor_handoffs'][-1]['previous_pid'] == 123
        (directory / 'status.json').write_text(json.dumps(dict(phase='completed')))
        proc.terminate()
        proc.wait(timeout=5)
        assert adopted.poll() == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_adoption_rejects_unrelated_pid(supervisor, tmp_path):
    with pytest.raises(RuntimeError, match='identity mismatch'):
        supervisor.AdoptedWorker(dict(pid=os.getpid(), job=dict(directory=str(tmp_path))))
