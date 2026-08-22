from __future__ import annotations

import torch
from torch.utils.data import Dataset


ACTION_ACCELERATIONS = torch.tensor([-0.04, -0.015, 0.015, 0.04])


def oscillator_transition(states: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
    position, velocity = states.unbind(dim=-1)
    acceleration = ACTION_ACCELERATIONS[action_indices]
    next_velocity = 0.97 * velocity + acceleration - 0.02 * torch.sin(3.0 * position)
    next_position = position + next_velocity
    return torch.stack((next_position, next_velocity), dim=-1)


class ControlledOscillatorSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, num_sequences: int, sequence_length: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        position = 2.0 * torch.rand(num_sequences, generator=generator) - 1.0
        velocity = 0.2 * torch.rand(num_sequences, generator=generator) - 0.1
        states = torch.stack((position, velocity), dim=-1)
        action_indices = torch.randint(
            0, 4, (num_sequences, sequence_length), generator=generator
        )
        history = [states]
        for time_index in range(sequence_length):
            states = oscillator_transition(states, action_indices[:, time_index])
            history.append(states)
        self.states = torch.stack(history, dim=1)
        self.action_indices = action_indices
        self.actions = torch.nn.functional.one_hot(action_indices, 4).float()

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.states[index],
            "actions": self.actions[index],
            "action_indices": self.action_indices[index],
        }
