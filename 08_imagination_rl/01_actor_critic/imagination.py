from __future__ import annotations

import torch

from behavior import GaussianActor
from world_model import FrozenLatentWorldModel


def imagine(
    world_model: FrozenLatentWorldModel,
    actor: GaussianActor,
    initial_latent: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    latent = initial_latent
    latents = [latent]
    actions, rewards, entropies = [], [], []
    for _ in range(horizon):
        action, entropy = actor.sample(latent)
        latent = world_model.transition(latent, action)
        actions.append(action)
        rewards.append(world_model.reward(latent))
        entropies.append(entropy)
        latents.append(latent)
    return {
        "latents": torch.stack(latents, dim=1),
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "entropies": torch.stack(entropies, dim=1),
    }


def lambda_returns(rewards: torch.Tensor, next_values: torch.Tensor, discount: float, lambda_: float) -> torch.Tensor:
    """Backward-view lambda return; rewards and next_values are [B,H]."""
    if rewards.shape != next_values.shape:
        raise ValueError("rewards and next_values must have equal [B,H] shapes")
    returns = []
    accumulated = next_values[:, -1]
    for time_index in reversed(range(rewards.shape[1])):
        bootstrap = (1.0 - lambda_) * next_values[:, time_index] + lambda_ * accumulated
        accumulated = rewards[:, time_index] + discount * bootstrap
        returns.append(accumulated)
    return torch.stack(list(reversed(returns)), dim=1)
