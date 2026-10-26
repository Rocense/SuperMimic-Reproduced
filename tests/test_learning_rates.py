"""Actor and critic schedules must preserve their independently configured rates."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.training.schedule import cosine_lr


@pytest.mark.parametrize('actor_lr,critic_lr', [(2e-4, 2e-4), (1e-5, 2e-4), (3e-5, 2e-4), (0.0, 2e-4)])
def test_optimizer_schedules_keep_the_critic_rate(actor_lr, critic_lr):
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    trainer = SimpleNamespace(
        opt=torch.optim.Adam(actor.parameters(), lr=actor_lr),
        critic_opt=torch.optim.Adam(critic.parameters(), lr=critic_lr),
    )
    wm_lr = 2e-4
    for iteration in (0, 15000, 50000):
        scheduled_wm_lr = cosine_lr(iteration, 50000, wm_lr, 0.1, 1.0)
        PolicyTrainer.set_lr(trainer, scheduled_wm_lr * actor_lr / wm_lr,
                             scheduled_wm_lr * critic_lr / wm_lr)
        assert trainer.opt.param_groups[0]['lr'] == pytest.approx(
            cosine_lr(iteration, 50000, actor_lr, 0.1, 1.0))
        assert trainer.critic_opt.param_groups[0]['lr'] == pytest.approx(
            cosine_lr(iteration, 50000, critic_lr, 0.1, 1.0))
