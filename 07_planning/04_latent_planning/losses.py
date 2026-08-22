from __future__ import annotations

import torch
import torch.nn.functional as F

from model import TaskOrientedLatentModel


def latent_model_loss(
    model: TaskOrientedLatentModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
    reward_targets: torch.Tensor,
    value_targets: torch.Tensor,
    consistency_weight: float = 1.0,
    reward_weight: float = 2.0,
    value_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    encoded = model.encode(observations)
    latent = encoded[:, 0]
    consistency_terms, reward_terms, value_terms = [], [], []
    value_terms.append(F.mse_loss(model.value(latent), value_targets[:, 0]))
    for time_index in range(actions.shape[1]):
        predicted = model.transition(latent, actions[:, time_index])
        target = encoded[:, time_index + 1].detach()
        consistency_terms.append(F.mse_loss(predicted, target))
        reward_terms.append(F.mse_loss(model.reward(predicted), reward_targets[:, time_index]))
        value_terms.append(F.mse_loss(model.value(predicted), value_targets[:, time_index + 1]))
        latent = predicted
    consistency = torch.stack(consistency_terms).mean()
    reward = torch.stack(reward_terms).mean()
    value = torch.stack(value_terms).mean()
    total = consistency_weight * consistency + reward_weight * reward + value_weight * value
    return {"total": total, "consistency": consistency, "reward": reward, "value": value}
