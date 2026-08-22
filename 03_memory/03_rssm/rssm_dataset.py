from __future__ import annotations

import sys
from pathlib import Path


PARTIAL_EXPERIMENT = Path(__file__).resolve().parents[1] / "02_partial_observation"
if str(PARTIAL_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(PARTIAL_EXPERIMENT))

from partial_dataset import PartialObservationSequenceDataset  # noqa: E402


def build_rssm_dataset(num_sequences: int, sequence_length: int, seed: int):
    """Use the versioned partial-observation sequence contract unchanged."""
    return PartialObservationSequenceDataset(num_sequences, sequence_length, seed)
