import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from dataset import GridSequenceDataset  # noqa: E402
from env import FullyObservableGridWorld  # noqa: E402


def test_environment_boundary_and_transition() -> None:
    env = FullyObservableGridWorld(seed=0)
    observation = env.reset((0, 0))
    assert observation.shape == (3, 20, 20)
    _, _, _, info = env.step(0)
    assert info["state"] == (0, 0)
    _, _, _, info = env.step(3)
    assert info["state"] == (0, 1)


def test_dataset_shapes_one_hot_and_consistency() -> None:
    dataset = GridSequenceDataset(6, 8, seed=3)
    sample = dataset[0]
    assert sample["observations"].shape == (9, 3, 20, 20)
    assert sample["actions"].shape == (8, 4)
    assert sample["states"].shape == (9, 2)
    assert torch.allclose(sample["actions"].sum(-1), torch.ones(8))
    assert torch.isfinite(sample["observations"]).all()

