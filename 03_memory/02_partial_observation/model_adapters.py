"""Explicit compatibility bridge to the completed 01_gru model interfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


EXPERIMENT_01 = Path(__file__).resolve().parents[1] / "01_gru"
if str(EXPERIMENT_01) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_01))

from baseline import SimpleDynamics  # noqa: E402
from model import GRUWorldModel, VisualEncoder  # noqa: E402


def build_compatible_models(
    latent_dim: int = 16, action_dim: int = 4, hidden_dim: int = 64
) -> tuple[VisualEncoder, SimpleDynamics, GRUWorldModel]:
    """Return unmodified 01_gru classes for this 20x20 partial-observation data."""
    encoder = VisualEncoder(latent_dim)
    simple_dynamics = SimpleDynamics(latent_dim, action_dim, hidden_dim)
    gru_world_model = GRUWorldModel(latent_dim, action_dim, hidden_dim)
    return encoder, simple_dynamics, gru_world_model


def validate_model_compatibility(batch: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    """Run both existing model paths without training or altering their source."""
    encoder, simple_dynamics, gru_world_model = build_compatible_models()
    observations, actions = batch["observations"], batch["actions"]
    latents = encoder(observations)
    simple_predictions = simple_dynamics(latents[:, :-1], actions)
    gru_outputs = gru_world_model(observations, actions)
    return {
        "encoded_latents": tuple(latents.shape),
        "simple_predictions": tuple(simple_predictions.shape),
        "gru_predictions": tuple(gru_outputs["predicted_next_latents"].shape),
        "gru_hidden_states": tuple(gru_outputs["hidden_states"].shape),
    }

