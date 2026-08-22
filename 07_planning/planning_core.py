from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PointWorldState:
    vector: torch.Tensor
    steps: int = 0


class PointWorldModel:
    """Exact compact model used to isolate planner behavior."""

    def __init__(self, action_scale: float = 0.2, success_radius: float = 0.08):
        self.action_scale = action_scale
        self.success_radius = success_radius

    def transition(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        next_agent = (states[..., :2] + self.action_scale * torch.tanh(actions)).clamp(-1.0, 1.0)
        return torch.cat((next_agent, states[..., 2:]), dim=-1)

    def reward(self, next_states: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(next_states[..., :2] - next_states[..., 2:], dim=-1)
        return -distance + (distance <= self.success_radius).to(distance.dtype)

    def continuation(self, next_states: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(next_states[..., :2] - next_states[..., 2:], dim=-1)
        return (distance > self.success_radius).to(next_states.dtype)

    def terminal_value(self, states: torch.Tensor) -> torch.Tensor:
        return -torch.linalg.vector_norm(states[..., :2] - states[..., 2:], dim=-1)

    def evaluate_action_sequences(
        self, initial_state: torch.Tensor, action_sequences: torch.Tensor, discount: float
    ) -> dict[str, torch.Tensor]:
        candidates, horizon = action_sequences.shape[:2]
        states = initial_state.expand(candidates, -1).clone()
        alive = torch.ones(candidates, dtype=states.dtype, device=states.device)
        score = torch.zeros_like(alive)
        state_history, reward_history, continuation_history = [], [], []
        discount_power = 1.0
        for time_index in range(horizon):
            proposed = self.transition(states, action_sequences[:, time_index])
            next_states = torch.where(alive[:, None].bool(), proposed, states)
            rewards = alive * self.reward(next_states)
            continuation = alive * self.continuation(next_states)
            score = score + discount_power * rewards
            states = next_states
            alive = continuation
            state_history.append(states)
            reward_history.append(rewards)
            continuation_history.append(continuation)
            discount_power *= discount
        score = score + discount_power * alive * self.terminal_value(states)
        return {
            "scores": score,
            "states": torch.stack(state_history, dim=1),
            "rewards": torch.stack(reward_history, dim=1),
            "continuations": torch.stack(continuation_history, dim=1),
        }


class PointWorldEnvironment:
    def __init__(
        self,
        start: tuple[float, float] = (-0.9, -0.8),
        goal: tuple[float, float] = (0.8, 0.7),
        max_steps: int = 20,
    ):
        self.model = PointWorldModel()
        self.initial = torch.tensor((*start, *goal), dtype=torch.float32)
        self.max_steps = max_steps
        self.state = PointWorldState(self.initial.clone())

    def reset(self) -> torch.Tensor:
        self.state = PointWorldState(self.initial.clone())
        return self.state.vector.clone()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        next_state = self.model.transition(self.state.vector, action)
        self.state = PointWorldState(next_state, self.state.steps + 1)
        reached = bool(self.model.continuation(next_state) == 0)
        done = reached or self.state.steps >= self.max_steps
        return next_state.clone(), float(self.model.reward(next_state)), done
