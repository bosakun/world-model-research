from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


PLANNING_ROOT = Path(__file__).resolve().parents[1]
CEM_EXPERIMENT = PLANNING_ROOT / "02_cem"
for path in (PLANNING_ROOT, CEM_EXPERIMENT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from cem import CEMPlanner  # noqa: E402
from planning_core import PointWorldEnvironment  # noqa: E402


@dataclass
class MPCResult:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    planned_scores: torch.Tensor
    success: bool


class RecedingHorizonMPC:
    def __init__(self, planner: CEMPlanner, max_steps: int = 20):
        self.planner = planner
        self.max_steps = max_steps

    def run(self, environment: PointWorldEnvironment) -> MPCResult:
        state = environment.reset()
        states = [state]
        actions, rewards, scores = [], [], []
        success = False
        for _ in range(self.max_steps):
            plan = self.planner.plan(state)
            action = plan.action_sequence[0]
            state, reward, done = environment.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            scores.append(plan.score)
            success = bool(environment.model.continuation(state) == 0)
            if done:
                break
        return MPCResult(
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards),
            torch.stack(scores),
            success,
        )
