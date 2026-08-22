from __future__ import annotations

import torch
from torch import nn


class SimpleDynamics(nn.Module):
    """Memory-free baseline: (z_t, a_t) -> predicted z_{t+1}."""

    def __init__(self, latent_dim: int = 16, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((latents, actions), dim=-1))

    def rollout(self, initial_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        latent = initial_latent
        predictions = []
        for step in range(actions.shape[1]):
            latent = self(latent, actions[:, step])
            predictions.append(latent)
        return torch.stack(predictions, dim=1)

