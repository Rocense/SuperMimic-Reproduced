"""A checkpoint at the original step cap must still be able to continue."""
import importlib.util
from pathlib import Path

import pytest

from supermimic.config import ScheduleCfg
from supermimic.training.schedule import cosine_lr


@pytest.mark.parametrize('completed', [0, 47000, 50000, 100000])
def test_continuation_retains_decay_endpoint_and_has_remaining_steps(completed):
    path = Path(__file__).with_name('worker.py')
    spec = importlib.util.spec_from_file_location('overnight_worker_test', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    schedule = ScheduleCfg(total_iterations=50000, lr_decay_fraction=1.0, lr_final_ratio=0.1)
    worker.extend_resume_budget(schedule, completed)
    assert schedule.total_iterations - completed >= 50000
    for step in (0, 20000, 49999, 50000, 90000):
        assert cosine_lr(step, schedule.total_iterations, 3e-5, 0.1,
                         schedule.lr_decay_fraction) == pytest.approx(
            cosine_lr(step, 50000, 3e-5, 0.1, 1.0), rel=1e-12)
