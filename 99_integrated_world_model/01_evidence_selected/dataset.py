from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PARTIAL_ROOT = Path(__file__).resolve().parents[2] / "03_memory" / "02_partial_observation"
if str(PARTIAL_ROOT) not in sys.path:
    sys.path.append(str(PARTIAL_ROOT))

from partial_env import (  # noqa: E402
    DOWN,
    LEFT,
    RIGHT,
    UP,
    PartialObservationGridWorld,
)


def greedy_action(state: np.ndarray, rng: np.random.Generator) -> int:
    """Choose one action that reduces Manhattan distance to the hidden goal."""
    agent_row, agent_col, goal_row, goal_col = state
    options: list[int] = []
    if agent_row < goal_row:
        options.append(DOWN)
    if agent_row > goal_row:
        options.append(UP)
    if agent_col < goal_col:
        options.append(RIGHT)
    if agent_col > goal_col:
        options.append(LEFT)
    return int(rng.choice(options)) if options else int(rng.integers(0, 4))


class IntegratedNavigationDataset(Dataset):
    """Paired partial-view trajectories with task targets for integrated training."""

    def __init__(self, sequences: int = 768, length: int = 12, seed: int = 331):
        observations = []
        actions = []
        states = []
        rewards = []
        values = []
        continuations = []
        goals = ((2, 3), (3, 2))

        for index in range(sequences):
            # Adjacent sequences share the same random stream and differ only in goal.
            rng = np.random.default_rng(seed + index // 2)
            environment = PartialObservationGridWorld()
            observation = environment.reset((2, 2), goals[index % 2])

            sequence_observations = [observation]
            sequence_states = [environment.true_state_array()]
            sequence_actions = []
            sequence_rewards = []
            sequence_values = []
            sequence_continuations = []

            for time in range(length):
                # The two LEFT moves hide either goal while preserving an aliased view.
                if time < 2:
                    action = LEFT
                elif rng.random() < 0.7:
                    action = greedy_action(environment.true_state_array(), rng)
                else:
                    action = int(rng.integers(0, 4))

                observation, _, done, info = environment.step(action)
                state = np.asarray(info["true_state"])
                distance = abs(state[0] - state[2]) + abs(state[1] - state[3])

                sequence_actions.append(action)
                sequence_rewards.append(-distance / 4.0 + float(done))
                # Planning target: a state potential, not a behavior-policy return.
                sequence_values.append(-distance / 4.0)
                sequence_continuations.append(0.0 if done else 1.0)
                sequence_observations.append(observation)
                sequence_states.append(state)

            observations.append(np.stack(sequence_observations))
            actions.append(sequence_actions)
            states.append(np.stack(sequence_states))
            rewards.append(sequence_rewards)
            values.append(sequence_values)
            continuations.append(sequence_continuations)

        self.observations = torch.from_numpy(np.stack(observations)).float()
        self.action_indices = torch.tensor(np.stack(actions))
        self.actions = torch.nn.functional.one_hot(self.action_indices, 4).float()
        self.true_states = torch.from_numpy(np.stack(states))
        self.rewards = torch.tensor(np.stack(rewards)).float()
        self.values = torch.tensor(np.stack(values)).float()
        self.continuations = torch.tensor(np.stack(continuations)).float()

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index):
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "action_indices": self.action_indices[index],
            "true_states": self.true_states[index],
            "rewards": self.rewards[index],
            "values": self.values[index],
            "continuations": self.continuations[index],
        }
