from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for module_name in ("config", "behavior", "world_model", "imagination"):
    sys.modules.pop(module_name, None)
from behavior import Critic, GaussianActor  # noqa: E402
from imagination import imagine, lambda_returns  # noqa: E402
from world_model import FrozenLatentWorldModel  # noqa: E402


def test_actor_critic_and_imagination_shapes():
    torch.manual_seed(2)
    world = FrozenLatentWorldModel().freeze()
    actor, critic = GaussianActor(), Critic()
    initial = world.encode(torch.randn(5, 4))
    trajectory = imagine(world, actor, initial, 6)
    assert trajectory["latents"].shape == (5, 7, 16)
    assert trajectory["actions"].shape == (5, 6, 2)
    assert trajectory["rewards"].shape == (5, 6)
    assert critic(trajectory["latents"]).shape == (5, 7)
    assert trajectory["actions"].abs().max() <= 1.0


def test_lambda_return_matches_one_step_and_monte_carlo_limits():
    rewards = torch.tensor([[1.0, 2.0]])
    next_values = torch.tensor([[10.0, 20.0]])
    one_step = lambda_returns(rewards, next_values, discount=0.5, lambda_=0.0)
    torch.testing.assert_close(one_step, torch.tensor([[6.0, 12.0]]))
    monte_carlo = lambda_returns(rewards, next_values, discount=0.5, lambda_=1.0)
    torch.testing.assert_close(monte_carlo, torch.tensor([[7.0, 12.0]]))


def test_actor_receives_gradient_through_frozen_world():
    torch.manual_seed(3)
    world = FrozenLatentWorldModel().freeze()
    actor = GaussianActor()
    target_critic = Critic()
    for parameter in target_critic.parameters():
        parameter.requires_grad_(False)
    initial = world.encode(torch.randn(8, 4))
    trajectory = imagine(world, actor, initial, 4)
    returns = lambda_returns(trajectory["rewards"], target_critic(trajectory["latents"][:, 1:]), 0.97, 0.95)
    (-returns.mean()).backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in actor.parameters())
    assert all(parameter.grad is None for parameter in world.parameters())


def test_critic_receives_finite_gradient():
    critic = Critic()
    latents = torch.randn(4, 5, 16)
    target = torch.randn(4, 5)
    loss = torch.nn.functional.mse_loss(critic(latents), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in critic.parameters())
