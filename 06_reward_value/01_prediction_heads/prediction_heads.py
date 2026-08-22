from __future__ import annotations

import torch
from torch import nn


class RewardValueContinuationHeads(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.transition_encoder = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.SiLU()
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.continuation_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.state_encoder(states)
        transition_features = self.transition_encoder(torch.cat((features, actions), dim=-1))
        return {
            "reward": self.reward_head(transition_features).squeeze(-1),
            "continuation_logit": self.continuation_head(transition_features).squeeze(-1),
            "value": self.value_head(features).squeeze(-1),
            "state_features": features,
        }
