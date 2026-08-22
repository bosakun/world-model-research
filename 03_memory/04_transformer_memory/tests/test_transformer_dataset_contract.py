import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from transformer_dataset import build_transformer_dataset  # noqa: E402


def test_partial_observation_dataset_contract_for_transformer_memory() -> None:
    dataset = build_transformer_dataset(4, 6, 29)
    assert dataset.observations.shape == (4, 7, 3, 20, 20)
    assert dataset.actions.shape == (4, 6, 4)
    assert dataset.true_states.shape == (4, 7, 4)
    assert torch.equal(dataset.observations[0, 2], dataset.observations[1, 2])
