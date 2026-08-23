from __future__ import annotations

from dataclasses import dataclass

import torch

from model import State


@dataclass
class Plan:
    actions: torch.Tensor
    score: float
    mean_return: float
    epistemic_std: float


class RiskAwarePlanner:
    """Discrete random-shooting MPC scored across prior ensemble members."""

    def __init__(
        self,
        model,
        horizon: int = 6,
        candidates: int = 512,
        discount: float = 0.97,
        penalty: float = 0.5,
        seed: int = 331,
    ):
        self.model = model
        self.horizon = horizon
        self.candidates = candidates
        self.discount = discount
        self.penalty = penalty
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def plan(self, state: State) -> Plan:
        action_indices = torch.randint(
            0,
            4,
            (self.candidates, self.horizon),
            generator=self.generator,
        )
        actions = torch.nn.functional.one_hot(action_indices, 4).float()
        ensemble_returns = []

        for member in range(len(self.model.priors)):
            imagined_state = State(
                state.deterministic.expand(self.candidates, -1).clone(),
                state.stochastic.expand(self.candidates, -1).clone(),
            )
            candidate_return = torch.zeros(self.candidates)
            discount = 1.0
            for time in range(self.horizon):
                imagined_state, _ = self.model.prior_step(
                    imagined_state, actions[:, time], member
                )
                feature = self.model.feature(imagined_state)
                candidate_return += discount * self.model.reward(feature).squeeze(-1)
                discount *= self.discount
            terminal_value = self.model.value(self.model.feature(imagined_state)).squeeze(-1)
            candidate_return += discount * terminal_value
            ensemble_returns.append(candidate_return)

        ensemble_returns = torch.stack(ensemble_returns)
        mean_return = ensemble_returns.mean(dim=0)
        epistemic_std = ensemble_returns.std(dim=0, unbiased=False)
        risk_adjusted_score = mean_return - self.penalty * epistemic_std
        best = risk_adjusted_score.argmax()
        return Plan(
            actions=action_indices[best],
            score=float(risk_adjusted_score[best]),
            mean_return=float(mean_return[best]),
            epistemic_std=float(epistemic_std[best]),
        )


class DiscreteActionGuard:
    """Final boundary that rejects malformed actions and supports a dead-man switch."""

    def filter(self, action: int, enabled: bool = True) -> int:
        if not enabled:
            return 0
        if not isinstance(action, int) or not 0 <= action < 4:
            raise ValueError("discrete action must be integer in [0,3]")
        return action
