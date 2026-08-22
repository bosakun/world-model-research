from __future__ import annotations

from dataclasses import dataclass

import torch

from model import TaskOrientedLatentModel


@dataclass
class LatentCEMResult:
    actions: torch.Tensor
    predicted_latents: torch.Tensor
    score: float
    iteration_best_scores: torch.Tensor


class LatentCEMPlanner:
    def __init__(
        self,
        model: TaskOrientedLatentModel,
        horizon: int = 10,
        candidates: int = 512,
        elites: int = 64,
        iterations: int = 6,
        discount: float = 0.97,
        seed: int = 101,
    ):
        self.model = model
        self.horizon = horizon
        self.candidates = candidates
        self.elites = elites
        self.iterations = iterations
        self.discount = discount
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def plan(self, observation: torch.Tensor) -> LatentCEMResult:
        mean = torch.zeros(self.horizon, self.model.action_dim)
        std = torch.ones_like(mean)
        best_score = torch.tensor(float("-inf"))
        best_actions = mean.clone()
        best_latents = torch.empty(self.horizon, self.model.encoder[-2].out_features)
        iteration_scores = []
        self.model.eval()
        for _ in range(self.iterations):
            noise = torch.randn(self.candidates, self.horizon, self.model.action_dim, generator=self.generator)
            candidates = (mean + std * noise).clamp(-1.0, 1.0)
            evaluation = self.model.evaluate_action_sequences(observation, candidates, self.discount)
            elite_indices = evaluation["scores"].topk(self.elites).indices
            elite_actions = candidates[elite_indices]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0, unbiased=False).clamp_min(0.05)
            iteration_best = evaluation["scores"].max()
            iteration_scores.append(iteration_best)
            if iteration_best > best_score:
                index = evaluation["scores"].argmax()
                best_score = iteration_best
                best_actions = candidates[index]
                best_latents = evaluation["latents"][index]
        return LatentCEMResult(best_actions, best_latents, float(best_score), torch.stack(iteration_scores))
