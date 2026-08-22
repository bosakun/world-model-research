import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from transformer_losses import transformer_world_model_loss  # noqa: E402
from transformer_memory import TransformerMemoryDynamics, TransformerMemoryWorldModel  # noqa: E402


def _batch(batch_size: int = 3, steps: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.rand(batch_size, steps + 1, 3, 20, 20)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (batch_size, steps)), 4).float()
    return observations, actions


def test_world_model_shapes_and_finite_values() -> None:
    model = TransformerMemoryWorldModel()
    observations, actions = _batch()
    outputs = model(observations, actions)
    assert outputs["latents"].shape == (3, 7, 16)
    assert outputs["predicted_next_latents"].shape == (3, 6, 16)
    assert outputs["context_tokens"].shape == (3, 6, 64)
    assert outputs["attention_maps"].shape == (2, 3, 4, 6, 6)
    assert outputs["predicted_next_observations"].shape == (3, 6, 3, 20, 20)
    assert outputs["goal_logits"].shape == (3, 6, 10)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_causal_outputs_do_not_depend_on_future_tokens() -> None:
    torch.manual_seed(1)
    model = TransformerMemoryDynamics(dropout=0.0).eval()
    latents = torch.randn(2, 6, 16)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 6)), 4).float()
    changed = latents.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100.0
    original_outputs = model(latents, actions)["predicted_next_latents"]
    changed_outputs = model(changed, actions)["predicted_next_latents"]
    assert torch.allclose(original_outputs[:, :3], changed_outputs[:, :3], atol=1e-6)
    assert not torch.allclose(original_outputs[:, 3:], changed_outputs[:, 3:])


def test_attention_has_exactly_zero_weight_on_future_positions() -> None:
    model = TransformerMemoryDynamics(dropout=0.0).eval()
    latents = torch.randn(2, 5, 16)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 5)), 4).float()
    attention = model(latents, actions)["attention_maps"]
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.equal(attention[..., future], torch.zeros_like(attention[..., future]))


def test_position_embedding_distinguishes_identical_content() -> None:
    model = TransformerMemoryDynamics()
    latents = torch.zeros(1, 4, 16)
    actions = torch.zeros(1, 4, 4)
    tokens = model.tokenize(latents, actions)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])


def test_loss_backpropagates_through_all_core_components_without_nan() -> None:
    model = TransformerMemoryWorldModel()
    observations, actions = _batch(batch_size=2, steps=4)
    loss = transformer_world_model_loss(model(observations, actions), observations)["total"]
    loss.backward()
    components = [
        model.encoder,
        model.decoder,
        model.dynamics.token_projection,
        model.dynamics.position_embedding,
        model.dynamics.blocks,
        model.dynamics.prediction_head,
        model.goal_head,
    ]
    assert torch.isfinite(loss)
    for component in components:
        gradients = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_rollout_is_autoregressive_and_has_expected_shapes() -> None:
    model = TransformerMemoryWorldModel().eval()
    observations, actions = _batch(batch_size=2, steps=5)
    with torch.no_grad():
        rollout = model.rollout(observations[:, 0], actions)
    assert rollout["predicted_next_latents"].shape == (2, 5, 16)
    assert rollout["predicted_next_observations"].shape == (2, 5, 3, 20, 20)
    assert rollout["goal_logits"].shape == (2, 5, 10)
    assert rollout["last_attention_maps"].shape == (2, 2, 4, 5, 5)


def test_context_limit_is_enforced() -> None:
    model = TransformerMemoryDynamics(max_context=4)
    latents = torch.zeros(1, 5, 16)
    actions = torch.zeros(1, 5, 4)
    try:
        model(latents, actions)
    except ValueError as error:
        assert "max_context" in str(error)
    else:
        raise AssertionError("expected context length validation")
