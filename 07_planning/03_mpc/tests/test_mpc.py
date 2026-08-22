import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
PLANNING_ROOT = EXPERIMENT.parents[0]
CEM_EXPERIMENT = PLANNING_ROOT / "02_cem"
sys.path.insert(0, str(EXPERIMENT))
sys.path.append(str(PLANNING_ROOT))
sys.path.append(str(CEM_EXPERIMENT))

from cem import CEMPlanner  # noqa: E402
from mpc import RecedingHorizonMPC  # noqa: E402
from planning_core import PointWorldEnvironment, PointWorldModel  # noqa: E402


def test_mpc_replans_and_reaches_goal() -> None:
    planner = CEMPlanner(PointWorldModel(), horizon=8, candidates=256, elites=32, iterations=4, seed=3)
    result = RecedingHorizonMPC(planner, max_steps=20).run(PointWorldEnvironment(max_steps=20))
    assert result.success
    assert result.states.shape[0] == result.actions.shape[0] + 1
    assert result.planned_scores.shape[0] == result.actions.shape[0]


def test_mpc_executes_only_first_action_of_each_plan() -> None:
    planner = CEMPlanner(PointWorldModel(), horizon=6, candidates=128, elites=16, iterations=3, seed=5)
    result = RecedingHorizonMPC(planner, max_steps=3).run(PointWorldEnvironment(max_steps=20))
    assert result.actions.shape == (3, 2)
    assert result.states.shape == (4, 4)


def test_mpc_distance_decreases() -> None:
    planner = CEMPlanner(PointWorldModel(), horizon=8, candidates=256, elites=32, iterations=4, seed=7)
    result = RecedingHorizonMPC(planner, max_steps=20).run(PointWorldEnvironment(max_steps=20))
    distances = torch.linalg.vector_norm(result.states[:, :2] - result.states[:, 2:], dim=-1)
    assert distances[-1] < distances[0]
