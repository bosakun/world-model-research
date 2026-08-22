from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from env import FullyObservableGridWorld


class GridSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-generated deterministic trajectories for reproducible sequence learning."""

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        grid_size: int = 5,
        cell_size: int = 4,
        seed: int = 0,
    ):
        if num_sequences < 1 or sequence_length < 1:
            raise ValueError("num_sequences and sequence_length must be positive")
        rng = np.random.default_rng(seed)
        observations, actions, states = [], [], []
        for sequence_index in range(num_sequences):
            env = FullyObservableGridWorld(grid_size, cell_size, seed + sequence_index)
            observation = env.reset()
            sequence_observations = [observation]
            sequence_states = [(env.state.row, env.state.col)]
            sequence_actions = []
            for _ in range(sequence_length):
                action = int(rng.integers(0, 4))
                observation, _, _, info = env.step(action)
                sequence_actions.append(action)
                sequence_observations.append(observation)
                sequence_states.append(info["state"])
            observations.append(np.stack(sequence_observations))
            actions.append(np.asarray(sequence_actions, dtype=np.int64))
            states.append(np.asarray(sequence_states, dtype=np.int64))

        self.observations = torch.from_numpy(np.stack(observations))
        self.action_indices = torch.from_numpy(np.stack(actions))
        self.actions = torch.nn.functional.one_hot(
            self.action_indices, num_classes=4
        ).float()
        self.states = torch.from_numpy(np.stack(states))

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "action_indices": self.action_indices[index],
            "states": self.states[index],
        }

