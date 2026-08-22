import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from ensemble_dataset import bootstrap_indices  # noqa: E402
from probabilistic_ensemble import ProbabilisticEnsemble  # noqa: E402


def _inputs(batch_size: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.randn(batch_size, 2)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (batch_size,)), 4).float()
    return states, actions


def test_ensemble_shapes_variances_and_finite_values() -> None:
    ensemble = ProbabilisticEnsemble(ensemble_size=5)
    states, actions = _inputs()
    outputs = ensemble(states, actions)
    assert outputs["member_means"].shape == (5, 7, 2)
    assert outputs["member_variances"].shape == (5, 7, 2)
    assert outputs["mean"].shape == (7, 2)
    assert torch.all(outputs["aleatoric_variance"] > 0)
    assert torch.all(outputs["epistemic_variance"] >= 0)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_total_variance_is_aleatoric_plus_epistemic() -> None:
    member_means = torch.tensor([[[0.0]], [[2.0]]])
    member_variances = torch.tensor([[[3.0]], [[5.0]]])
    outputs = ProbabilisticEnsemble.decompose(member_means, member_variances)
    assert torch.equal(outputs["aleatoric_variance"], torch.tensor([[4.0]]))
    assert torch.equal(outputs["epistemic_variance"], torch.tensor([[1.0]]))
    assert torch.equal(outputs["total_variance"], torch.tensor([[5.0]]))


def test_identical_member_means_have_zero_epistemic_variance() -> None:
    means = torch.ones(5, 3, 2)
    variances = torch.ones_like(means)
    outputs = ProbabilisticEnsemble.decompose(means, variances)
    assert torch.equal(outputs["epistemic_variance"], torch.zeros(3, 2))


def test_bootstrap_members_receive_different_resampled_indices() -> None:
    indices = bootstrap_indices(100, 5, seed=11)
    assert indices.shape == (5, 100)
    assert not torch.equal(indices[0], indices[1])
    assert torch.all((indices >= 0) & (indices < 100))


def test_loss_can_backpropagate_through_every_member() -> None:
    ensemble = ProbabilisticEnsemble(ensemble_size=3)
    states, actions = _inputs(batch_size=5)
    outputs = ensemble(states, actions)
    outputs["member_means"].square().mean().backward()
    for member in ensemble.members:
        assert any(parameter.grad is not None for parameter in member.parameters())


def test_ts_infinity_keeps_model_per_particle_and_ts1_can_change() -> None:
    torch.manual_seed(3)
    ensemble = ProbabilisticEnsemble(ensemble_size=5)
    initial = torch.zeros(1, 2)
    actions = torch.nn.functional.one_hot(torch.zeros(1, 8, dtype=torch.long), 4).float()
    ts_infinity = ensemble.rollout(
        initial, actions, particles=32, propagation="ts_infinity", generator=torch.Generator().manual_seed(7)
    )
    ts1 = ensemble.rollout(
        initial, actions, particles=32, propagation="ts1", generator=torch.Generator().manual_seed(7)
    )
    assert ts_infinity["states"].shape == (1, 32, 8, 2)
    assert ts_infinity["model_indices"].shape == (1, 32, 8)
    assert torch.equal(
        ts_infinity["model_indices"], ts_infinity["model_indices"][:, :, :1].expand(-1, -1, 8)
    )
    assert torch.any(ts1["model_indices"][:, :, 1:] != ts1["model_indices"][:, :, :-1])


def test_invalid_propagation_mode_is_rejected() -> None:
    ensemble = ProbabilisticEnsemble()
    try:
        ensemble.rollout(torch.zeros(1, 2), torch.zeros(1, 2, 4), 4, "invalid")
    except ValueError as error:
        assert "propagation" in str(error)
    else:
        raise AssertionError("expected propagation validation")
