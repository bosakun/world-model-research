from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(self, latent_dim: int = 16, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, latent: torch.Tensor) -> Normal:
        features = self.trunk(latent)
        mean = self.mean(features)
        std = self.log_std(features).clamp(-4.0, 1.0).exp()
        return Normal(mean, std)

    def sample(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(latent)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        entropy = distribution.entropy().sum(dim=-1)
        return action, entropy

    def mode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(latent).mean)


class Critic(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent).squeeze(-1)
