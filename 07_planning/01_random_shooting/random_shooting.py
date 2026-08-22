from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


PLANNING_ROOT = Path(__file__).resolve().parents[1]
if str(PLANNING_ROOT) not in sys.path:
    sys.path.append(str(PLANNING_ROOT))

from planning_core import PointWorldModel  # noqa: E402


@dataclass
class RandomShootingResult:
    action_sequence: torch.Tensor
    score: torch.Tensor
    candidate_scores: torch.Tensor
    predicted_states: torch.Tensor


class RandomShootingPlanner:
    def __init__(
        self,
        model: PointWorldModel,
        horizon: int = 10,
        candidates: int = 2048,
        discount: float = 0.97,
        seed: int = 67,
    ):
        self.model = model
        self.horizon = horizon
        self.candidates = candidates
        self.discount = discount
        self.generator = torch.Generator().manual_seed(seed)

    def plan(self, state: torch.Tensor) -> RandomShootingResult:
        action_sequences = 2.0 * torch.rand(
            self.candidates, self.horizon, 2, generator=self.generator
        ) - 1.0
        evaluation = self.model.evaluate_action_sequences(state, action_sequences, self.discount)
        best_index = evaluation["scores"].argmax()
        return RandomShootingResult(
            action_sequences[best_index],
            evaluation["scores"][best_index],
            evaluation["scores"],
            evaluation["states"][best_index],
        )
