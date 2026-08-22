import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
PLANNING_ROOT = EXPERIMENT.parents[0]
sys.path.insert(0, str(EXPERIMENT))
sys.path.append(str(PLANNING_ROOT))

from cem import CEMPlanner  # noqa: E402
from planning_core import PointWorldModel  # noqa: E402


def test_cem_shapes_and_distribution_contraction() -> None:
    planner = CEMPlanner(PointWorldModel(), horizon=8, candidates=256, elites=32, iterations=4, seed=3)
    result = planner.plan(torch.tensor([-0.9, -0.8, 0.8, 0.7]))
    assert result.action_sequence.shape == result.mean.shape == result.std.shape == (8, 2)
    assert result.iteration_best_scores.shape == (4,)
    assert torch.all(result.std < 1.0)


def test_cem_improves_or_preserves_iteration_best_score() -> None:
    planner = CEMPlanner(PointWorldModel(), candidates=512, elites=64, iterations=5, seed=5)
    scores = planner.plan(torch.tensor([-0.9, -0.8, 0.8, 0.7])).iteration_best_scores
    assert scores[-1] >= scores[0]


def test_cem_rejects_invalid_elite_count() -> None:
    try:
        CEMPlanner(PointWorldModel(), candidates=4, elites=5)
    except ValueError as error:
        assert "elites" in str(error)
    else:
        raise AssertionError("expected elite validation")


def test_cem_plan_reduces_goal_distance() -> None:
    state = torch.tensor([-0.9, -0.8, 0.8, 0.7])
    result = CEMPlanner(PointWorldModel(), seed=7).plan(state)
    initial = torch.linalg.vector_norm(state[:2] - state[2:])
    final = torch.linalg.vector_norm(result.predicted_states[-1, :2] - state[2:])
    assert final < initial * 0.3
