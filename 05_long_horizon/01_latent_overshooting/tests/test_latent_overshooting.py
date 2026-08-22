import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from latent_dynamics import LatentDynamics  # noqa: E402
from overshooting_losses import latent_overshooting_loss, long_horizon_loss  # noqa: E402
from sequence_dataset import ControlledOscillatorSequenceDataset, oscillator_transition  # noqa: E402


def test_environment_transition_matches_dataset_sequence() -> None:
    dataset = ControlledOscillatorSequenceDataset(8, 10, seed=2)
    expected = oscillator_transition(dataset.states[:, 0], dataset.action_indices[:, 0])
    assert torch.allclose(expected, dataset.states[:, 1])
    assert dataset.states.shape == (8, 11, 2)
    assert dataset.actions.shape == (8, 10, 4)


def test_forward_and_rollout_shapes_are_finite() -> None:
    model = LatentDynamics()
    dataset = ControlledOscillatorSequenceDataset(4, 12, seed=3)
    one_step = model(dataset.states[:, :-1], dataset.actions)
    rollout = model.rollout(dataset.states[:, 0], dataset.actions)
    assert one_step.shape == rollout.shape == (4, 12, 2)
    assert torch.isfinite(one_step).all() and torch.isfinite(rollout).all()


def test_overshooting_distance_one_matches_one_step_loss() -> None:
    model = LatentDynamics()
    dataset = ControlledOscillatorSequenceDataset(4, 6, seed=4)
    one_step = torch.nn.functional.mse_loss(
        model(dataset.states[:, :-1], dataset.actions), dataset.states[:, 1:]
    )
    overshooting, by_distance = latent_overshooting_loss(
        model, dataset.states, dataset.actions, max_distance=1
    )
    assert torch.allclose(overshooting, one_step, atol=1e-7)
    assert by_distance.shape == (1,)


def test_overshooting_loss_backpropagates_through_recursive_predictions() -> None:
    model = LatentDynamics()
    dataset = ControlledOscillatorSequenceDataset(4, 8, seed=5)
    losses = long_horizon_loss(model, dataset.states, dataset.actions, 5, 0.5)
    losses["total"].backward()
    assert losses["overshooting_by_distance"].shape == (5,)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_rollout_uses_predictions_not_future_true_states() -> None:
    model = LatentDynamics()
    dataset = ControlledOscillatorSequenceDataset(2, 5, seed=6)
    baseline = model.rollout(dataset.states[:, 0], dataset.actions)
    changed_future = dataset.states.clone()
    changed_future[:, 1:] += 1000.0
    assert torch.equal(baseline, model.rollout(changed_future[:, 0], dataset.actions))
