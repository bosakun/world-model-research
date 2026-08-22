import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
PLANNING_ROOT = EXPERIMENT.parents[0]
sys.path.insert(0, str(EXPERIMENT))
sys.path.append(str(PLANNING_ROOT))

from planning_core import PointWorldEnvironment, PointWorldModel  # noqa: E402
from random_shooting import RandomShootingPlanner  # noqa: E402


def test_world_model_shapes_and_terminal_value() -> None:
    model = PointWorldModel()
    state = torch.tensor([-0.9, -0.8, 0.8, 0.7])
    actions = torch.zeros(5, 7, 2)
    outputs = model.evaluate_action_sequences(state, actions, 0.97)
    assert outputs["states"].shape == (5, 7, 4)
    assert outputs["scores"].shape == (5,)
    assert torch.isfinite(outputs["scores"]).all()


def test_planner_returns_best_sampled_candidate() -> None:
    planner = RandomShootingPlanner(PointWorldModel(), horizon=8, candidates=512, seed=3)
    result = planner.plan(torch.tensor([-0.9, -0.8, 0.8, 0.7]))
    assert result.action_sequence.shape == (8, 2)
    assert result.predicted_states.shape == (8, 4)
    assert result.score == result.candidate_scores.max()


def test_planned_sequence_reduces_goal_distance() -> None:
    state = torch.tensor([-0.9, -0.8, 0.8, 0.7])
    planner = RandomShootingPlanner(PointWorldModel(), horizon=10, candidates=4096, seed=5)
    result = planner.plan(state)
    initial_distance = torch.linalg.vector_norm(state[:2] - state[2:])
    final_distance = torch.linalg.vector_norm(result.predicted_states[-1, :2] - state[2:])
    assert final_distance < initial_distance * 0.6


def test_environment_transition_matches_model() -> None:
    environment = PointWorldEnvironment()
    state = environment.reset()
    action = torch.tensor([0.4, -0.2])
    predicted = environment.model.transition(state, action)
    actual, _, _ = environment.step(action)
    assert torch.equal(predicted, actual)
