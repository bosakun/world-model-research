import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from baseline import SimpleDynamics  # noqa: E402
from losses import world_model_loss  # noqa: E402
from model import GRUDynamics, GRUWorldModel  # noqa: E402


def test_forward_shapes_and_no_nan() -> None:
    model = GRUWorldModel(latent_dim=16, action_dim=4, hidden_dim=64)
    observations = torch.rand(3, 9, 3, 20, 20)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (3, 8)), 4).float()
    outputs = model(observations, actions)
    assert outputs["latents"].shape == (3, 9, 16)
    assert outputs["reconstructions"].shape == observations.shape
    assert outputs["predicted_next_latents"].shape == (3, 8, 16)
    assert outputs["hidden_states"].shape == (3, 8, 64)
    assert outputs["final_hidden"].shape == (3, 64)
    assert all(torch.isfinite(tensor).all() for tensor in outputs.values())


def test_rollout_shapes() -> None:
    dynamics = GRUDynamics(16, 4, 64)
    initial = torch.randn(2, 16)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 5)), 4).float()
    predictions, hidden_states, final_hidden = dynamics.rollout(initial, actions)
    assert predictions.shape == (2, 5, 16)
    assert hidden_states.shape == (2, 5, 64)
    assert final_hidden.shape == (2, 64)


def test_gradients_reach_encoder_decoder_gru_and_head() -> None:
    model = GRUWorldModel()
    observations = torch.rand(2, 5, 3, 20, 20)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 4)), 4).float()
    loss = world_model_loss(model(observations, actions), observations)["total"]
    loss.backward()
    components = [model.encoder, model.decoder, model.dynamics.cell, model.dynamics.prediction_head]
    for component in components:
        assert any(parameter.grad is not None for parameter in component.parameters())
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in component.parameters()
            if parameter.grad is not None
        )


def test_simple_dynamics_is_retained() -> None:
    baseline = SimpleDynamics()
    assert baseline(torch.randn(2, 3, 16), torch.randn(2, 3, 4)).shape == (2, 3, 16)

