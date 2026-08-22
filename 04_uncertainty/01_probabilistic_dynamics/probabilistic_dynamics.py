from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GaussianPrediction:
    mean: torch.Tensor
    log_variance: torch.Tensor

    @property
    def variance(self) -> torch.Tensor:
        return self.log_variance.exp()

    @property
    def std(self) -> torch.Tensor:
        return (0.5 * self.log_variance).exp()

    def sample(self, stochastic: bool = True) -> torch.Tensor:
        if not stochastic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)


class ProbabilisticDynamics(nn.Module):
    """Diagonal Gaussian next-state model with bounded learned log variance."""

    def __init__(self, state_dim: int = 2, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_delta_head = nn.Linear(hidden_dim, state_dim)
        self.raw_log_variance_head = nn.Linear(hidden_dim, state_dim)
        self.max_log_variance = nn.Parameter(torch.full((state_dim,), 0.5))
        self.min_log_variance = nn.Parameter(torch.full((state_dim,), -10.0))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> GaussianPrediction:
        features = self.backbone(torch.cat((states, actions), dim=-1))
        mean = states + self.mean_delta_head(features)
        raw = self.raw_log_variance_head(features)
        upper_bounded = self.max_log_variance - F.softplus(self.max_log_variance - raw)
        log_variance = self.min_log_variance + F.softplus(
            upper_bounded - self.min_log_variance
        )
        return GaussianPrediction(mean, log_variance)

    def rollout(
        self, initial_states: torch.Tensor, actions: torch.Tensor, stochastic: bool = True
    ) -> dict[str, torch.Tensor]:
        states = initial_states
        sampled_states, means, standard_deviations = [], [], []
        for time_index in range(actions.shape[1]):
            prediction = self(states, actions[:, time_index])
            states = prediction.sample(stochastic)
            sampled_states.append(states)
            means.append(prediction.mean)
            standard_deviations.append(prediction.std)
        return {
            "states": torch.stack(sampled_states, dim=1),
            "means": torch.stack(means, dim=1),
            "stds": torch.stack(standard_deviations, dim=1),
        }
