from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


PROBABILISTIC_EXPERIMENT = Path(__file__).resolve().parents[1] / "01_probabilistic_dynamics"
if str(PROBABILISTIC_EXPERIMENT) not in sys.path:
    sys.path.append(str(PROBABILISTIC_EXPERIMENT))

from probabilistic_dynamics import ProbabilisticDynamics  # noqa: E402


class ProbabilisticEnsemble(nn.Module):
    def __init__(
        self,
        ensemble_size: int = 5,
        state_dim: int = 2,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.members = nn.ModuleList(
            ProbabilisticDynamics(state_dim, action_dim, hidden_dim)
            for _ in range(ensemble_size)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        predictions = [member(states, actions) for member in self.members]
        member_means = torch.stack([prediction.mean for prediction in predictions], dim=0)
        member_variances = torch.stack([prediction.variance for prediction in predictions], dim=0)
        return self.decompose(member_means, member_variances)

    @staticmethod
    def decompose(
        member_means: torch.Tensor, member_variances: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        mean = member_means.mean(dim=0)
        aleatoric_variance = member_variances.mean(dim=0)
        epistemic_variance = member_means.var(dim=0, unbiased=False)
        return {
            "mean": mean,
            "member_means": member_means,
            "member_variances": member_variances,
            "aleatoric_variance": aleatoric_variance,
            "epistemic_variance": epistemic_variance,
            "total_variance": aleatoric_variance + epistemic_variance,
        }

    def rollout(
        self,
        initial_states: torch.Tensor,
        actions: torch.Tensor,
        particles: int,
        propagation: str,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """PETS-style trajectory sampling with TS1 or TS-infinity model assignment."""
        if propagation not in {"ts1", "ts_infinity"}:
            raise ValueError("propagation must be 'ts1' or 'ts_infinity'")
        batch_size, horizon = actions.shape[:2]
        states = initial_states[:, None].expand(batch_size, particles, -1).clone()
        fixed_indices = torch.randint(
            self.ensemble_size, (batch_size, particles), generator=generator
        )
        trajectory, selected_models = [], []
        for time_index in range(horizon):
            model_indices = (
                fixed_indices
                if propagation == "ts_infinity"
                else torch.randint(
                    self.ensemble_size, (batch_size, particles), generator=generator
                )
            )
            repeated_actions = actions[:, time_index, None].expand(-1, particles, -1)
            flattened_states = states.reshape(batch_size * particles, -1)
            flattened_actions = repeated_actions.reshape(batch_size * particles, -1)
            predictions = self(flattened_states, flattened_actions)
            flat_indices = model_indices.reshape(-1)
            particle_indices = torch.arange(flat_indices.numel(), device=states.device)
            mean = predictions["member_means"][flat_indices, particle_indices]
            variance = predictions["member_variances"][flat_indices, particle_indices]
            noise = torch.randn(mean.shape, generator=generator, device=mean.device)
            states = (mean + variance.sqrt() * noise).reshape(batch_size, particles, -1)
            trajectory.append(states)
            selected_models.append(model_indices)
        return {
            "states": torch.stack(trajectory, dim=2),
            "model_indices": torch.stack(selected_models, dim=2),
        }
