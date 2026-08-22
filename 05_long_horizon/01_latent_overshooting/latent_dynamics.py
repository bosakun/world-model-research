from __future__ import annotations

import torch
from torch import nn


class LatentDynamics(nn.Module):
    def __init__(self, state_dim: int = 2, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return states + self.network(torch.cat((states, actions), dim=-1))

    def rollout(self, initial_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = initial_states
        predictions = []
        for time_index in range(actions.shape[1]):
            states = self(states, actions[:, time_index])
            predictions.append(states)
        return torch.stack(predictions, dim=1)
