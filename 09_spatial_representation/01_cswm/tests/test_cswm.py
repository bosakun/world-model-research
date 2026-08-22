from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
for name in ("config", "dataset", "model", "losses"): sys.modules.pop(name, None)
from dataset import TwoObjectTransitionDataset, relational_transition, render_objects  # noqa: E402
from losses import contrastive_world_model_loss  # noqa: E402
from model import ContrastiveStructuredWorldModel  # noqa: E402


def test_dataset_shapes_and_true_relational_transition():
    data = TwoObjectTransitionDataset(9, 2)
    assert data.images.shape == (9, 3, 16, 16)
    assert data.actions.shape == (9, 2, 2)
    torch.testing.assert_close(data.next_positions, relational_transition(data.positions, data.actions))
    assert render_objects(data.positions).amin() >= 0 and render_objects(data.positions).amax() <= 1


def test_object_slots_and_relational_transition_shapes():
    model = ContrastiveStructuredWorldModel()
    data = TwoObjectTransitionDataset(4, 3)
    slots, predicted = model.predict(data.images, data.actions)
    assert slots.shape == predicted.shape == (4, 2, 8)


def test_contrastive_loss_has_finite_gradients():
    model = ContrastiveStructuredWorldModel(); data = TwoObjectTransitionDataset(8, 4)
    losses = contrastive_world_model_loss(model, data.images, data.actions, data.next_images)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_relational_effect_is_equal_and_opposite_without_actions():
    positions = torch.tensor([[[-0.1, 0.0], [0.1, 0.0]]])
    next_positions = relational_transition(positions, torch.zeros_like(positions))
    displacement = next_positions - positions
    torch.testing.assert_close(displacement[:, 0], -displacement[:, 1])
