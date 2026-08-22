from __future__ import annotations

import torch
from torch.nn import functional as F


def observation_goal_evidence(images: torch.Tensor) -> torch.Tensor:
    """Extract 9 local-cell green evidences for target construction only."""
    green_score = images[..., 1, :, :] - torch.maximum(images[..., 0, :, :], images[..., 2, :, :])
    flat = green_score.reshape(-1, 1, 20, 20)
    cells = F.avg_pool2d(flat, kernel_size=4, stride=4)[:, 0, 1:4, 1:4].flatten(1)
    return 10.0 * cells


def goal_class_targets(observations: torch.Tensor) -> torch.Tensor:
    evidence = observation_goal_evidence(observations)
    visible_score, visible_class = evidence.max(dim=-1)
    return torch.where(visible_score > 4.0, visible_class, torch.full_like(visible_class, 9))


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p), summed across stochastic dimensions."""
    variance_ratio = posterior_std.square() / prior_std.square()
    mean_difference = (posterior_mean - prior_mean).square() / prior_std.square()
    elementwise = torch.log(prior_std / posterior_std) + 0.5 * (variance_ratio + mean_difference - 1.0)
    return elementwise.sum(dim=-1)


def rssm_loss(
    outputs: dict[str, torch.Tensor],
    observations: torch.Tensor,
    kl_weight: float = 1e-3,
    free_nats: float = 1.0,
    goal_classification_weight: float = 0.1,
    green_channel_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    channel_weight = observations.new_tensor((1.0, green_channel_weight, 1.0)).reshape(
        1, 1, 3, 1, 1
    )
    reconstruction = ((outputs["reconstructions"] - observations).square() * channel_weight).mean()
    goal_classification = F.cross_entropy(
        outputs["goal_logits"].reshape(-1, 10), goal_class_targets(observations)
    )
    kl_per_state = diagonal_gaussian_kl(
        outputs["posterior_means"],
        outputs["posterior_stds"],
        outputs["prior_means"],
        outputs["prior_stds"],
    )
    kl_raw = kl_per_state.mean()
    kl = torch.clamp(kl_per_state, min=free_nats).mean()
    total = reconstruction + kl_weight * kl + goal_classification_weight * goal_classification
    return {
        "total": total,
        "reconstruction": reconstruction,
        "goal_classification": goal_classification,
        "kl": kl,
        "kl_raw": kl_raw,
    }
