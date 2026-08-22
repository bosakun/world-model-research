import math
import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from probabilistic_dynamics import GaussianPrediction, ProbabilisticDynamics  # noqa: E402
from probabilistic_losses import diagonal_gaussian_nll, probabilistic_dynamics_loss  # noqa: E402
from stochastic_dataset import (  # noqa: E402
    HeteroscedasticTransitionDataset,
    StochasticPointSequenceDataset,
    transition_noise_std,
)


def test_transition_dataset_shapes_and_alignment() -> None:
    dataset = HeteroscedasticTransitionDataset(32, seed=3)
    assert dataset.states.shape == (32, 2)
    assert dataset.actions.shape == (32, 4)
    assert dataset.next_states.shape == (32, 2)
    assert dataset.true_noise_std.shape == (32, 2)
    assert torch.allclose(dataset.actions.sum(-1), torch.ones(32))


def test_known_noise_is_heteroscedastic() -> None:
    states = torch.tensor([[-0.8, 0.0], [0.8, 0.0]])
    horizontal_actions = torch.tensor([0, 0])
    std = transition_noise_std(states, horizontal_actions)
    assert std[1, 0] > 5.0 * std[0, 0]
    vertical = transition_noise_std(states[:1], torch.tensor([2]))
    assert vertical[0, 1] > std[0, 1]


def test_model_outputs_positive_finite_variance_with_expected_shape() -> None:
    model = ProbabilisticDynamics()
    states = torch.randn(7, 2)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (7,)), 4).float()
    prediction = model(states, actions)
    assert prediction.mean.shape == (7, 2)
    assert prediction.log_variance.shape == (7, 2)
    assert torch.all(prediction.variance > 0)
    assert torch.isfinite(prediction.mean).all() and torch.isfinite(prediction.std).all()


def test_gaussian_nll_matches_standard_normal_at_its_mean() -> None:
    prediction = GaussianPrediction(torch.zeros(3, 2), torch.zeros(3, 2))
    loss = diagonal_gaussian_nll(prediction, torch.zeros(3, 2))
    assert torch.allclose(loss, torch.tensor(math.log(2.0 * math.pi)), atol=1e-6)


def test_reparameterized_sample_has_gradient_to_mean_and_log_variance() -> None:
    mean = torch.zeros(4, 2, requires_grad=True)
    log_variance = torch.zeros(4, 2, requires_grad=True)
    GaussianPrediction(mean, log_variance).sample(stochastic=True).sum().backward()
    assert mean.grad is not None and log_variance.grad is not None


def test_loss_backpropagates_without_nan_through_mean_and_variance_paths() -> None:
    model = ProbabilisticDynamics()
    dataset = HeteroscedasticTransitionDataset(16, seed=5)
    prediction = model(dataset.states, dataset.actions)
    loss = probabilistic_dynamics_loss(model, prediction, dataset.next_states)["total"]
    loss.backward()
    assert torch.isfinite(loss)
    for component in (model.backbone, model.mean_delta_head, model.raw_log_variance_head):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_sequence_dataset_and_rollout_shapes() -> None:
    dataset = StochasticPointSequenceDataset(8, horizon=6, seed=7)
    model = ProbabilisticDynamics()
    rollout = model.rollout(dataset.states[:, 0], dataset.actions, stochastic=False)
    assert dataset.states.shape == (8, 7, 2)
    assert dataset.actions.shape == (8, 6, 4)
    assert rollout["states"].shape == (8, 6, 2)
    assert rollout["means"].shape == (8, 6, 2)
    assert rollout["stds"].shape == (8, 6, 2)
    assert torch.isfinite(rollout["states"]).all()


def test_stochastic_rollout_changes_with_sampling_while_mean_is_repeatable() -> None:
    model = ProbabilisticDynamics()
    initial = torch.zeros(2, 2)
    actions = torch.nn.functional.one_hot(torch.zeros(2, 4, dtype=torch.long), 4).float()
    mean_a = model.rollout(initial, actions, stochastic=False)["states"]
    mean_b = model.rollout(initial, actions, stochastic=False)["states"]
    sample_a = model.rollout(initial, actions, stochastic=True)["states"]
    sample_b = model.rollout(initial, actions, stochastic=True)["states"]
    assert torch.equal(mean_a, mean_b)
    assert not torch.equal(sample_a, sample_b)
