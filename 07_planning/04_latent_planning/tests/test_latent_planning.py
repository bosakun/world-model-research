from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Independent experiment folders intentionally reuse descriptive module names.
# Clear earlier experiments' cached modules before resolving this folder.
for module_name in ("dataset", "model", "losses", "planner"):
    sys.modules.pop(module_name, None)
from dataset import LatentPlanningSequenceDataset  # noqa: E402
from losses import latent_model_loss  # noqa: E402
from model import TaskOrientedLatentModel  # noqa: E402
from planner import LatentCEMPlanner  # noqa: E402


def test_dataset_shapes_and_transition_alignment():
    data = LatentPlanningSequenceDataset(7, 5, 3)
    assert data.observations.shape == (7, 6, 4)
    assert data.actions.shape == (7, 5, 2)
    expected = (data.observations[:, :-1, :2] + 0.2 * torch.tanh(data.actions)).clamp(-1.0, 1.0)
    torch.testing.assert_close(data.observations[:, 1:, :2], expected)
    torch.testing.assert_close(data.observations[:, 1:, 2:], data.observations[:, :-1, 2:])


def test_model_shapes_and_decoder_free_interface():
    model = TaskOrientedLatentModel()
    observations = torch.randn(4, 6, 4)
    actions = torch.randn(4, 5, 2)
    latent = model.encode(observations[:, 0])
    rollout = model.rollout(latent, actions)
    assert latent.shape == (4, 16)
    assert rollout["latents"].shape == (4, 5, 16)
    assert rollout["rewards"].shape == (4, 5)
    assert not hasattr(model, "decoder")


def test_joint_loss_is_finite_and_all_components_receive_gradients():
    data = LatentPlanningSequenceDataset(8, 4, 9)
    model = TaskOrientedLatentModel()
    losses = latent_model_loss(model, data.observations, data.actions, data.rewards, data.values)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    for module in (model.encoder, model.dynamics, model.reward_head, model.value_head):
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_latent_cem_returns_bounded_plan_without_decoder():
    torch.manual_seed(4)
    model = TaskOrientedLatentModel()
    planner = LatentCEMPlanner(model, horizon=4, candidates=32, elites=4, iterations=2)
    result = planner.plan(torch.tensor([-0.5, -0.5, 0.5, 0.5]))
    assert result.actions.shape == (4, 2)
    assert result.predicted_latents.shape == (4, 16)
    assert result.actions.abs().max() <= 1.0
    assert torch.isfinite(result.iteration_best_scores).all()
