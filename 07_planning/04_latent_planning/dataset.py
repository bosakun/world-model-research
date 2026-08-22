from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


PLANNING_ROOT = Path(__file__).resolve().parents[1]
if str(PLANNING_ROOT) not in sys.path:
    sys.path.append(str(PLANNING_ROOT))
from planning_core import PointWorldModel  # noqa: E402


class LatentPlanningSequenceDataset(Dataset):
    """Sequences covering both random and noisy goal-directed actions."""

    def __init__(self, sequences: int, horizon: int, seed: int, action_scale: float = 0.2):
        generator = torch.Generator().manual_seed(seed)
        dynamics = PointWorldModel(action_scale=action_scale)
        agents = torch.empty(sequences, 2).uniform_(-0.9, 0.9, generator=generator)
        goals = torch.empty(sequences, 2).uniform_(-0.9, 0.9, generator=generator)
        states = torch.cat((agents, goals), dim=-1)
        observations = [states]
        actions, rewards, values = [], [], [-torch.linalg.vector_norm(agents - goals, dim=-1)]
        for _ in range(horizon):
            desired = (states[:, 2:] - states[:, :2]) / action_scale
            noisy_greedy = desired + 0.35 * torch.randn(sequences, 2, generator=generator)
            random_actions = torch.empty(sequences, 2).uniform_(-1.0, 1.0, generator=generator)
            use_greedy = torch.rand(sequences, 1, generator=generator) < 0.6
            action = torch.where(use_greedy, noisy_greedy, random_actions).clamp(-1.0, 1.0)
            states = dynamics.transition(states, action)
            actions.append(action)
            rewards.append(dynamics.reward(states))
            observations.append(states)
            values.append(-torch.linalg.vector_norm(states[:, :2] - states[:, 2:], dim=-1))
        self.observations = torch.stack(observations, dim=1)
        self.actions = torch.stack(actions, dim=1)
        self.rewards = torch.stack(rewards, dim=1)
        self.values = torch.stack(values, dim=1)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "rewards": self.rewards[index],
            "values": self.values[index],
        }
