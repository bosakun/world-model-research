import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from rssm import DiagonalGaussian, RSSMState, RecurrentStateSpaceModel  # noqa: E402
from rssm_losses import diagonal_gaussian_kl, goal_class_targets, rssm_loss  # noqa: E402


def _batch(batch_size: int = 3, steps: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.rand(batch_size, steps + 1, 3, 20, 20)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (batch_size, steps)), 4).float()
    return observations, actions


def test_observe_shapes_positive_std_and_finite_values() -> None:
    model = RecurrentStateSpaceModel()
    observations, actions = _batch()
    outputs = model.observe(observations, actions)
    assert outputs["deterministic_states"].shape == (3, 6, 64)
    assert outputs["stochastic_states"].shape == (3, 6, 16)
    assert outputs["prior_means"].shape == (3, 6, 16)
    assert outputs["posterior_means"].shape == (3, 6, 16)
    assert outputs["reconstructions"].shape == observations.shape
    assert torch.all(outputs["prior_stds"] > 0.1)
    assert torch.all(outputs["posterior_stds"] > 0.1)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_reparameterized_sample_has_gradient_to_mean_and_std() -> None:
    mean = torch.zeros(2, 4, requires_grad=True)
    std = torch.ones(2, 4, requires_grad=True)
    DiagonalGaussian(mean, std).sample(stochastic=True).sum().backward()
    assert mean.grad is not None and std.grad is not None


def test_kl_is_zero_for_equal_diagonal_gaussians_and_nonnegative() -> None:
    mean, std = torch.randn(2, 3), torch.rand(2, 3) + 0.1
    equal_kl = diagonal_gaussian_kl(mean, std, mean, std)
    shifted_kl = diagonal_gaussian_kl(mean + 1.0, std, mean, std)
    assert torch.allclose(equal_kl, torch.zeros_like(equal_kl), atol=1e-6)
    assert torch.all(shifted_kl >= 0)


def test_world_model_loss_backpropagates_through_all_core_components() -> None:
    model = RecurrentStateSpaceModel()
    observations, actions = _batch(batch_size=2, steps=3)
    loss = rssm_loss(model.observe(observations, actions), observations)["total"]
    loss.backward()
    components = [
        model.encoder,
        model.recurrent_transition,
        model.prior,
        model.posterior,
        model.decoder,
        model.goal_head,
    ]
    for component in components:
        assert any(parameter.grad is not None for parameter in component.parameters())
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in component.parameters()
            if parameter.grad is not None
        )


def test_imagination_uses_prior_without_future_observations() -> None:
    model = RecurrentStateSpaceModel()
    initial = RSSMState(torch.zeros(2, 64), torch.zeros(2, 16))
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 5)), 4).float()
    imagined = model.imagine(initial, actions, stochastic=False)
    assert imagined["deterministic_states"].shape == (2, 5, 64)
    assert imagined["stochastic_states"].shape == (2, 5, 16)
    assert imagined["observations"].shape == (2, 5, 3, 20, 20)
    assert "posterior_means" not in imagined


def test_posterior_depends_on_observation_but_prior_does_not_at_t0() -> None:
    model = RecurrentStateSpaceModel()
    actions = torch.empty(1, 0, 4)
    dark = torch.zeros(1, 1, 3, 20, 20)
    bright = torch.ones(1, 1, 3, 20, 20)
    dark_outputs = model.observe(dark, actions, stochastic=False)
    bright_outputs = model.observe(bright, actions, stochastic=False)
    assert torch.allclose(dark_outputs["prior_means"], bright_outputs["prior_means"])
    assert not torch.allclose(dark_outputs["posterior_means"], bright_outputs["posterior_means"])


def test_goal_class_target_distinguishes_visible_location_and_absence() -> None:
    observations = torch.zeros(2, 1, 3, 20, 20)
    observations[0, 0, 1, 9:12, 13:16] = 1.0  # displayed local cell row=2,col=3 -> class 5
    targets = goal_class_targets(observations)
    assert targets.tolist() == [5, 9]
