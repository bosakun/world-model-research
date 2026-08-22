import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from partial_dataset import PartialObservationSequenceDataset  # noqa: E402
from model_adapters import validate_model_compatibility  # noqa: E402


def test_completed_simple_and_gru_models_accept_partial_observation_sequences() -> None:
    dataset = PartialObservationSequenceDataset(num_sequences=2, sequence_length=6)
    batch = {
        "observations": dataset.observations,
        "actions": dataset.actions,
    }
    shapes = validate_model_compatibility(batch)
    assert shapes == {
        "encoded_latents": (2, 7, 16),
        "simple_predictions": (2, 6, 16),
        "gru_predictions": (2, 6, 16),
        "gru_hidden_states": (2, 6, 64),
    }
    assert torch.isfinite(dataset.observations).all()
