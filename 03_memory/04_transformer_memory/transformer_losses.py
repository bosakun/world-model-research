from __future__ import annotations

import torch
from torch.nn import functional as F


def observation_goal_evidence(images: torch.Tensor) -> torch.Tensor:
    green_score = images[..., 1, :, :] - torch.maximum(images[..., 0, :, :], images[..., 2, :, :])
    flat = green_score.reshape(-1, 1, 20, 20)
    return 10.0 * F.avg_pool2d(flat, kernel_size=4, stride=4)[:, 0, 1:4, 1:4].flatten(1)


def goal_class_targets(observations: torch.Tensor) -> torch.Tensor:
    evidence = observation_goal_evidence(observations)
    visible_score, visible_class = evidence.max(dim=-1)
    return torch.where(visible_score > 4.0, visible_class, torch.full_like(visible_class, 9))


def weighted_image_mse(
    predictions: torch.Tensor, targets: torch.Tensor, green_channel_weight: float
) -> torch.Tensor:
    weights = targets.new_tensor((1.0, green_channel_weight, 1.0)).reshape(
        *((1,) * (targets.ndim - 3)), 3, 1, 1
    )
    return ((predictions - targets).square() * weights).mean()


def transformer_world_model_loss(
    outputs: dict[str, torch.Tensor],
    observations: torch.Tensor,
    latent_prediction_weight: float = 0.5,
    goal_classification_weight: float = 0.1,
    green_channel_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    reconstruction = weighted_image_mse(
        outputs["reconstructions"], observations, green_channel_weight
    )
    prediction_image = weighted_image_mse(
        outputs["predicted_next_observations"], observations[:, 1:], green_channel_weight
    )
    latent_prediction = F.mse_loss(
        outputs["predicted_next_latents"], outputs["latents"][:, 1:].detach()
    )
    goal_classification = F.cross_entropy(
        outputs["goal_logits"].reshape(-1, 10), goal_class_targets(observations[:, 1:])
    )
    total = (
        reconstruction
        + prediction_image
        + latent_prediction_weight * latent_prediction
        + goal_classification_weight * goal_classification
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "prediction_image": prediction_image,
        "latent_prediction": latent_prediction,
        "goal_classification": goal_classification,
    }
