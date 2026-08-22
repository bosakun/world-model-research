from __future__ import annotations

import torch
from torch.nn import functional as F

from latent_dynamics import LatentDynamics


def latent_overshooting_loss(
    model: LatentDynamics,
    states: torch.Tensor,
    actions: torch.Tensor,
    max_distance: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []
    counts_by_distance = torch.zeros(max_distance, device=states.device)
    sums_by_distance = torch.zeros(max_distance, device=states.device)
    horizon = actions.shape[1]
    for start in range(horizon):
        predicted = states[:, start]
        available = min(max_distance, horizon - start)
        for offset in range(available):
            predicted = model(predicted, actions[:, start + offset])
            error = F.mse_loss(predicted, states[:, start + offset + 1])
            losses.append(error)
            sums_by_distance[offset] = sums_by_distance[offset] + error.detach()
            counts_by_distance[offset] += 1
    mean_by_distance = sums_by_distance / counts_by_distance.clamp_min(1)
    return torch.stack(losses).mean(), mean_by_distance


def long_horizon_loss(
    model: LatentDynamics,
    states: torch.Tensor,
    actions: torch.Tensor,
    max_distance: int,
    overshooting_weight: float,
) -> dict[str, torch.Tensor]:
    one_step_predictions = model(states[:, :-1], actions)
    one_step = F.mse_loss(one_step_predictions, states[:, 1:])
    overshooting, by_distance = latent_overshooting_loss(model, states, actions, max_distance)
    return {
        "total": one_step + overshooting_weight * overshooting,
        "one_step": one_step,
        "overshooting": overshooting,
        "overshooting_by_distance": by_distance,
    }
