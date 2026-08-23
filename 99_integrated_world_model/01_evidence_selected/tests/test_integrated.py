from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for module_name in ("config", "dataset", "model", "losses", "planner"):
    sys.modules.pop(module_name, None)

from config import IntegratedConfig
from dataset import IntegratedNavigationDataset
from losses import integrated_loss
from model import IntegratedWorldModel
from planner import DiscreteActionGuard, RiskAwarePlanner


def test_dataset_shapes_and_prefix_alias():
    dataset = IntegratedNavigationDataset(8, 12, 1)
    assert dataset.observations.shape == (8, 13, 3, 20, 20)
    assert dataset.actions.shape == (8, 12, 4)
    assert torch.equal(dataset.observations[0, 2], dataset.observations[1, 2])


def test_model_sequence_shapes_and_ensemble():
    dataset = IntegratedNavigationDataset(4, 5, 2)
    model = IntegratedWorldModel()
    outputs = model.observe(dataset.observations, dataset.actions)
    assert outputs["feature"].shape == (4, 6, 80)
    assert outputs["prior_mean"].shape == (3, 4, 6, 16)
    assert outputs["reward"].shape == (4, 5)


def test_all_losses_and_gradients_finite():
    config = IntegratedConfig(sequence_length=5, train_sequences=4)
    batch = IntegratedNavigationDataset(4, 5, 3)[:]
    model = IntegratedWorldModel()
    losses, _ = integrated_loss(model, batch, config)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_checkpoint_schema_and_planner_interface(tmp_path):
    model = IntegratedWorldModel()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "format_version": 1}, checkpoint)
    restored = IntegratedWorldModel()
    restored.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])

    state = restored.initial(1, torch.device("cpu"))
    plan = RiskAwarePlanner(restored, 3, 32, seed=4).plan(state)
    assert plan.actions.shape == (3,)
    assert 0 <= int(plan.actions[0]) < 4

    guard = DiscreteActionGuard()
    assert guard.filter(2) == 2
    assert guard.filter(2, enabled=False) == 0
