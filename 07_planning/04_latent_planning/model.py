from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class TaskOrientedLatentModel(nn.Module):
    """Decoder-free latent transition with reward and terminal-value heads."""

    def __init__(self, observation_dim: int = 4, action_dim: int = 2, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim), nn.Tanh()
        )
        self.dynamics = mlp(latent_dim + action_dim, hidden_dim, latent_dim)
        self.reward_head = mlp(latent_dim, hidden_dim, 1)
        self.value_head = mlp(latent_dim, hidden_dim, 1)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.dynamics(torch.cat((latent, action), dim=-1))
        return torch.tanh(latent + delta)

    def reward(self, next_latent: torch.Tensor) -> torch.Tensor:
        return self.reward_head(next_latent).squeeze(-1)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        return self.value_head(latent).squeeze(-1)

    def rollout(self, initial_latent: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = initial_latent
        latents, rewards = [], []
        for time_index in range(actions.shape[-2]):
            latent = self.transition(latent, actions[..., time_index, :])
            latents.append(latent)
            rewards.append(self.reward(latent))
        return {"latents": torch.stack(latents, dim=-2), "rewards": torch.stack(rewards, dim=-1)}

    def evaluate_action_sequences(
        self, observation: torch.Tensor, action_sequences: torch.Tensor, discount: float
    ) -> dict[str, torch.Tensor]:
        initial_latent = self.encode(observation).expand(action_sequences.shape[0], -1)
        rollout = self.rollout(initial_latent, action_sequences)
        powers = discount ** torch.arange(action_sequences.shape[1], device=action_sequences.device)
        scores = (rollout["rewards"] * powers).sum(dim=-1)
        scores = scores + discount ** action_sequences.shape[1] * self.value(rollout["latents"][:, -1])
        return {**rollout, "scores": scores}
