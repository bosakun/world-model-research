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
class CEMResult:
    action_sequence: torch.Tensor
    score: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    iteration_best_scores: torch.Tensor
    predicted_states: torch.Tensor


class CEMPlanner:
    def __init__(
        self,
        model: PointWorldModel,
        horizon: int = 10,
        candidates: int = 512,
        elites: int = 64,
        iterations: int = 5,
        discount: float = 0.97,
        momentum: float = 0.1,
        seed: int = 71,
    ):
        if not 0 < elites <= candidates:
            raise ValueError("elites must be in [1,candidates]")
        self.model = model
        self.horizon = horizon
        self.candidates = candidates
        self.elites = elites
        self.iterations = iterations
        self.discount = discount
        self.momentum = momentum
        self.generator = torch.Generator().manual_seed(seed)

    def plan(self, state: torch.Tensor) -> CEMResult:
        mean = torch.zeros(self.horizon, 2)
        std = torch.ones_like(mean)
        best_sequence = mean.clone()
        best_score = torch.tensor(float("-inf"))
        best_states = torch.empty(self.horizon, 4)
        iteration_scores = []
        for _ in range(self.iterations):
            noise = torch.randn(self.candidates, self.horizon, 2, generator=self.generator)
            samples = (mean + std * noise).clamp(-1.0, 1.0)
            evaluation = self.model.evaluate_action_sequences(state, samples, self.discount)
            elite_indices = evaluation["scores"].topk(self.elites).indices
            elites = samples[elite_indices]
            new_mean, new_std = elites.mean(dim=0), elites.std(dim=0, unbiased=False).clamp_min(0.05)
            mean = self.momentum * mean + (1.0 - self.momentum) * new_mean
            std = self.momentum * std + (1.0 - self.momentum) * new_std
            iteration_best = evaluation["scores"].max()
            iteration_scores.append(iteration_best)
            if iteration_best > best_score:
                index = evaluation["scores"].argmax()
                best_score = iteration_best
                best_sequence = samples[index]
                best_states = evaluation["states"][index]
        return CEMResult(best_sequence, best_score, mean, std, torch.stack(iteration_scores), best_states)
