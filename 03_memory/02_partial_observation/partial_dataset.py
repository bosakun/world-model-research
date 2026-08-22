from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from partial_env import LEFT, PartialObservationGridWorld


class PartialObservationSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Paired sequences with visual aliasing at `alias_time`.

    Each adjacent even/odd pair begins from the same agent location and uses the
    same actions. It has a different initially visible goal location. After two
    left moves, both goals lie outside the 3x3 view, yielding bitwise-identical
    current observations but distinct true goal coordinates.
    """

    alias_time = 2
    initial_agent = (2, 2)
    paired_goals = ((2, 3), (3, 2))  # right and down: both initially visible

    def __init__(self, num_sequences: int = 32, sequence_length: int = 6, seed: int = 17):
        if num_sequences < 2 or num_sequences % 2:
            raise ValueError("num_sequences must be an even integer >= 2 so aliasing pairs are complete")
        if sequence_length < self.alias_time:
            raise ValueError(f"sequence_length must be >= {self.alias_time}")
        observations, full_worlds, actions, states, goal_visible = [], [], [], [], []
        for sequence_index in range(num_sequences):
            pair_index = sequence_index // 2
            goal = self.paired_goals[sequence_index % 2]
            env = PartialObservationGridWorld()
            observation = env.reset(self.initial_agent, goal)
            # Pair members share all actions. The prefix makes t=2 an alias.
            pair_rng = np.random.default_rng(seed + pair_index)
            trailing = pair_rng.integers(0, 4, size=sequence_length - self.alias_time, dtype=np.int64)
            action_indices = np.concatenate((np.asarray([LEFT, LEFT], dtype=np.int64), trailing))

            sequence_observations = [observation]
            sequence_full_worlds = [env.render_full_world()]
            sequence_states = [env.true_state_array()]
            sequence_goal_visible = [env.goal_is_visible()]
            for action in action_indices:
                observation, _, _, info = env.step(int(action))
                sequence_observations.append(observation)
                sequence_full_worlds.append(env.render_full_world())
                sequence_states.append(np.asarray(info["true_state"], dtype=np.int64))
                sequence_goal_visible.append(bool(info["goal_visible"]))
            observations.append(np.stack(sequence_observations))
            full_worlds.append(np.stack(sequence_full_worlds))
            actions.append(action_indices)
            states.append(np.stack(sequence_states))
            goal_visible.append(np.asarray(sequence_goal_visible, dtype=np.bool_))

        self.observations = torch.from_numpy(np.stack(observations))
        self.full_worlds = torch.from_numpy(np.stack(full_worlds))
        self.action_indices = torch.from_numpy(np.stack(actions))
        self.actions = torch.nn.functional.one_hot(self.action_indices, num_classes=4).float()
        self.true_states = torch.from_numpy(np.stack(states))
        self.goal_visible = torch.from_numpy(np.stack(goal_visible))

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observations": self.observations[index],
            "full_worlds": self.full_worlds[index],
            "actions": self.actions[index],
            "action_indices": self.action_indices[index],
            "true_states": self.true_states[index],
            "goal_visible": self.goal_visible[index],
        }
