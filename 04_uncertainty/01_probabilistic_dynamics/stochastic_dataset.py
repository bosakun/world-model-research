from __future__ import annotations

import torch
from torch.utils.data import Dataset


ACTION_DELTAS = torch.tensor(
    [[-0.12, 0.0], [0.12, 0.0], [0.0, -0.12], [0.0, 0.12]], dtype=torch.float32
)


def transition_noise_std(states: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
    """Known input-dependent irreducible noise used only to evaluate calibration."""
    horizontal = 0.01 + 0.09 * torch.sigmoid(6.0 * states[..., 0])
    vertical_action = (action_indices >= 2).to(states.dtype)
    vertical = 0.015 + 0.055 * vertical_action
    return torch.stack((horizontal, vertical), dim=-1)


def stochastic_transition(
    states: torch.Tensor,
    action_indices: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    std = transition_noise_std(states, action_indices)
    noise = torch.randn(states.shape, generator=generator, dtype=states.dtype) * std
    return states + ACTION_DELTAS[action_indices] + noise, std


class HeteroscedasticTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, num_transitions: int, seed: int, state_limit: float = 0.8):
        generator = torch.Generator().manual_seed(seed)
        self.states = (2.0 * torch.rand(num_transitions, 2, generator=generator) - 1.0) * state_limit
        self.action_indices = torch.randint(0, 4, (num_transitions,), generator=generator)
        self.actions = torch.nn.functional.one_hot(self.action_indices, 4).float()
        self.next_states, self.true_noise_std = stochastic_transition(
            self.states, self.action_indices, generator
        )

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.states[index],
            "actions": self.actions[index],
            "action_indices": self.action_indices[index],
            "next_states": self.next_states[index],
            "true_noise_std": self.true_noise_std[index],
        }


class StochasticPointSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, num_sequences: int, horizon: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        states = (2.0 * torch.rand(num_sequences, 2, generator=generator) - 1.0) * 0.5
        state_history = [states]
        action_indices = torch.randint(0, 4, (num_sequences, horizon), generator=generator)
        noise_history = []
        for time_index in range(horizon):
            states, std = stochastic_transition(states, action_indices[:, time_index], generator)
            state_history.append(states)
            noise_history.append(std)
        self.states = torch.stack(state_history, dim=1)
        self.action_indices = action_indices
        self.actions = torch.nn.functional.one_hot(action_indices, 4).float()
        self.true_noise_std = torch.stack(noise_history, dim=1)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.states[index],
            "actions": self.actions[index],
            "action_indices": self.action_indices[index],
            "true_noise_std": self.true_noise_std[index],
        }
