from __future__ import annotations

import sys
from pathlib import Path

import torch


PROBABILISTIC_EXPERIMENT = Path(__file__).resolve().parents[1] / "01_probabilistic_dynamics"
if str(PROBABILISTIC_EXPERIMENT) not in sys.path:
    sys.path.append(str(PROBABILISTIC_EXPERIMENT))

from stochastic_dataset import (  # noqa: E402
    HeteroscedasticTransitionDataset,
    StochasticPointSequenceDataset,
    transition_noise_std,
)


def bootstrap_indices(
    dataset_size: int, ensemble_size: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, dataset_size, (ensemble_size, dataset_size), generator=generator)


__all__ = [
    "HeteroscedasticTransitionDataset",
    "StochasticPointSequenceDataset",
    "bootstrap_indices",
    "transition_noise_std",
]
