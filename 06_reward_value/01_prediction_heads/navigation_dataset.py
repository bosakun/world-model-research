from __future__ import annotations

import torch
from torch.utils.data import Dataset


ACTION_DELTAS = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.long)


def discounted_returns(
    rewards: torch.Tensor, continuations: torch.Tensor, discount: float
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    future = torch.zeros(rewards.shape[0], dtype=rewards.dtype)
    for time_index in range(rewards.shape[1] - 1, -1, -1):
        future = rewards[:, time_index] + discount * continuations[:, time_index] * future
        returns[:, time_index] = future
    return returns


class GoalNavigationSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, num_sequences: int, horizon: int, discount: float, seed: int):
        generator = torch.Generator().manual_seed(seed)
        agent = torch.randint(0, 5, (num_sequences, 2), generator=generator)
        goal = torch.randint(0, 5, (num_sequences, 2), generator=generator)
        same = (agent == goal).all(dim=-1)
        goal[same, 0] = (goal[same, 0] + 1) % 5
        done = torch.zeros(num_sequences, dtype=torch.bool)
        state_history, action_history = [], []
        rewards, continuations, valid = [], [], []

        for _ in range(horizon):
            state_history.append(torch.cat((agent, goal), dim=-1).float() / 4.0)
            difference = goal - agent
            preferred = torch.where(
                difference[:, 0] < 0,
                0,
                torch.where(
                    difference[:, 0] > 0,
                    1,
                    torch.where(difference[:, 1] < 0, 2, 3),
                ),
            )
            random_actions = torch.randint(0, 4, (num_sequences,), generator=generator)
            goal_directed = torch.rand(num_sequences, generator=generator) < 0.8
            action_indices = torch.where(goal_directed, preferred, random_actions)
            action_indices = torch.where(done, torch.zeros_like(action_indices), action_indices)
            was_valid = ~done
            proposed = (agent + ACTION_DELTAS[action_indices]).clamp(0, 4)
            agent = torch.where(was_valid[:, None], proposed, agent)
            reached = was_valid & (agent == goal).all(dim=-1)
            reward = torch.where(reached, 1.0, torch.where(was_valid, -0.05, 0.0))
            continuation = (was_valid & ~reached).float()
            action_history.append(action_indices)
            rewards.append(reward)
            continuations.append(continuation)
            valid.append(was_valid.float())
            done = done | reached

        state_history.append(torch.cat((agent, goal), dim=-1).float() / 4.0)
        self.states = torch.stack(state_history, dim=1)
        self.action_indices = torch.stack(action_history, dim=1)
        self.actions = torch.nn.functional.one_hot(self.action_indices, 4).float()
        self.rewards = torch.stack(rewards, dim=1)
        self.continuations = torch.stack(continuations, dim=1)
        self.valid = torch.stack(valid, dim=1)
        self.value_targets = discounted_returns(self.rewards, self.continuations, discount)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.states[index],
            "actions": self.actions[index],
            "action_indices": self.action_indices[index],
            "rewards": self.rewards[index],
            "continuations": self.continuations[index],
            "valid": self.valid[index],
            "value_targets": self.value_targets[index],
        }
