import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from partial_dataset import PartialObservationSequenceDataset  # noqa: E402
from partial_env import AGENT_COLOR, GOAL_COLOR, LEFT, PartialObservationGridWorld  # noqa: E402


def _cell_pixels(image: np.ndarray, row: int, col: int) -> np.ndarray:
    return image[:, row * 4 + 1 : row * 4 + 4, col * 4 + 1 : col * 4 + 4]


def test_true_state_updates_and_goal_stays_fixed() -> None:
    env = PartialObservationGridWorld()
    env.reset(agent=(2, 2), goal=(2, 3))
    _, _, _, info = env.step(LEFT)
    assert tuple(info["true_state"]) == (2, 1, 2, 3)
    assert env.state.goal == (2, 3)
    assert env.state.agent == (2, 1)


def test_partial_observation_range_is_agent_centred() -> None:
    env = PartialObservationGridWorld()
    partial = env.reset(agent=(2, 2), goal=(2, 3))
    assert env.visible_world_coordinates() == {
        (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)
    }
    assert np.allclose(_cell_pixels(partial, 2, 2), np.asarray(AGENT_COLOR)[:, None, None])
    assert np.allclose(_cell_pixels(partial, 2, 3), np.asarray(GOAL_COLOR)[:, None, None])


def test_outside_goal_does_not_leak_into_partial_observation() -> None:
    env = PartialObservationGridWorld()
    partial = env.reset(agent=(2, 0), goal=(2, 3))
    assert not env.goal_is_visible()
    green = partial[1]
    assert float(green.max()) < GOAL_COLOR[1]
    assert not np.allclose(_cell_pixels(partial, 2, 4), np.asarray(GOAL_COLOR)[:, None, None])


def test_goal_is_visible_then_disappears_by_t2() -> None:
    env = PartialObservationGridWorld()
    env.reset(agent=(2, 2), goal=(2, 3))
    assert env.goal_is_visible()
    env.step(LEFT)
    env.step(LEFT)
    assert not env.goal_is_visible()


def test_paired_dataset_has_identical_alias_observation_but_different_true_goal() -> None:
    dataset = PartialObservationSequenceDataset(num_sequences=2, sequence_length=6, seed=17)
    alias_time = dataset.alias_time
    assert torch.equal(dataset.observations[0, alias_time], dataset.observations[1, alias_time])
    assert not bool(dataset.goal_visible[0, alias_time])
    assert not bool(dataset.goal_visible[1, alias_time])
    assert not torch.equal(dataset.true_states[0, alias_time, 2:], dataset.true_states[1, alias_time, 2:])
    assert bool(dataset.goal_visible[0, 0]) and bool(dataset.goal_visible[1, 0])


def test_dataset_temporal_transition_actions_and_shapes() -> None:
    dataset = PartialObservationSequenceDataset(num_sequences=4, sequence_length=6, seed=19)
    sample = dataset[0]
    assert sample["observations"].shape == (7, 3, 20, 20)
    assert sample["full_worlds"].shape == (7, 3, 20, 20)
    assert sample["actions"].shape == (6, 4)
    assert sample["true_states"].shape == (7, 4)
    assert sample["goal_visible"].shape == (7,)
    assert torch.allclose(sample["actions"].sum(-1), torch.ones(6))
    assert sample["action_indices"][0].item() == LEFT
    assert tuple(sample["true_states"][1, :2].tolist()) == (2, 1)
    assert tuple(sample["true_states"][2, :2].tolist()) == (2, 0)

