import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from navigation_dataset import GoalNavigationSequenceDataset, discounted_returns  # noqa: E402
from prediction_heads import RewardValueContinuationHeads  # noqa: E402
from prediction_losses import prediction_head_loss  # noqa: E402


def test_dataset_shapes_padding_and_terminal_transition() -> None:
    dataset = GoalNavigationSequenceDataset(64, horizon=20, discount=0.95, seed=2)
    assert dataset.states.shape == (64, 21, 4)
    assert dataset.actions.shape == (64, 20, 4)
    assert dataset.rewards.shape == dataset.continuations.shape == dataset.valid.shape == (64, 20)
    terminal = (dataset.continuations == 0) & (dataset.valid == 1)
    assert terminal.any()
    assert torch.all(dataset.rewards[dataset.valid == 0] == 0)


def test_discounted_return_respects_continuation_boundary() -> None:
    rewards = torch.tensor([[1.0, 5.0, 7.0]])
    continuations = torch.tensor([[0.0, 1.0, 1.0]])
    returns = discounted_returns(rewards, continuations, discount=0.9)
    assert returns[0, 0] == 1.0
    assert torch.allclose(returns[0, 1], torch.tensor(11.3))


def test_model_output_shapes_and_finite_values() -> None:
    model = RewardValueContinuationHeads()
    states = torch.rand(3, 5, 4)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (3, 5)), 4).float()
    outputs = model(states, actions)
    assert outputs["reward"].shape == (3, 5)
    assert outputs["value"].shape == (3, 5)
    assert outputs["continuation_logit"].shape == (3, 5)
    assert outputs["state_features"].shape == (3, 5, 64)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_all_three_losses_backpropagate_to_corresponding_heads() -> None:
    dataset = GoalNavigationSequenceDataset(8, 10, 0.95, seed=4)
    model = RewardValueContinuationHeads()
    outputs = model(dataset.states[:, :-1], dataset.actions)
    losses = prediction_head_loss(
        outputs,
        dataset.rewards,
        dataset.value_targets,
        dataset.continuations,
        dataset.valid,
    )
    losses["total"].backward()
    for component in (model.state_encoder, model.transition_encoder, model.reward_head, model.value_head, model.continuation_head):
        assert any(parameter.grad is not None for parameter in component.parameters())


def test_invalid_padding_values_do_not_change_masked_loss() -> None:
    dataset = GoalNavigationSequenceDataset(8, 20, 0.95, seed=5)
    model = RewardValueContinuationHeads()
    outputs = model(dataset.states[:, :-1], dataset.actions)
    original = prediction_head_loss(
        outputs, dataset.rewards, dataset.value_targets, dataset.continuations, dataset.valid
    )["total"]
    changed_rewards = dataset.rewards + (1.0 - dataset.valid) * 1000.0
    changed_values = dataset.value_targets - (1.0 - dataset.valid) * 1000.0
    changed = prediction_head_loss(
        outputs, changed_rewards, changed_values, dataset.continuations, dataset.valid
    )["total"]
    assert torch.equal(original, changed)
