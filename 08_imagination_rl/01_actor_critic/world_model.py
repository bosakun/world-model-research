from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class FrozenLatentWorldModel(nn.Module):
    """Architecture-compatible copy of Phase 07's learned model."""

    def __init__(self, observation_dim: int = 4, action_dim: int = 2, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim), nn.Tanh()
        )
        self.dynamics = mlp(latent_dim + action_dim, hidden_dim, latent_dim)
        self.reward_head = mlp(latent_dim, hidden_dim, 1)
        self.value_head = mlp(latent_dim, hidden_dim, 1)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent + self.dynamics(torch.cat((latent, action), dim=-1)))

    def reward(self, next_latent: torch.Tensor) -> torch.Tensor:
        return self.reward_head(next_latent).squeeze(-1)

    def freeze(self) -> "FrozenLatentWorldModel":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
