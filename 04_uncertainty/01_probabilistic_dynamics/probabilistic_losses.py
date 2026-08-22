from __future__ import annotations

import math

import torch

from probabilistic_dynamics import GaussianPrediction, ProbabilisticDynamics


def diagonal_gaussian_nll(
    prediction: GaussianPrediction, targets: torch.Tensor
) -> torch.Tensor:
    elementwise = 0.5 * (
        prediction.log_variance
        + (targets - prediction.mean).square() / prediction.variance
        + math.log(2.0 * math.pi)
    )
    return elementwise.sum(dim=-1).mean()


def probabilistic_dynamics_loss(
    model: ProbabilisticDynamics,
    prediction: GaussianPrediction,
    targets: torch.Tensor,
    bound_regularizer_weight: float = 1e-4,
) -> dict[str, torch.Tensor]:
    negative_log_likelihood = diagonal_gaussian_nll(prediction, targets)
    bound_regularizer = model.max_log_variance.sum() - model.min_log_variance.sum()
    total = negative_log_likelihood + bound_regularizer_weight * bound_regularizer
    return {
        "total": total,
        "negative_log_likelihood": negative_log_likelihood,
        "bound_regularizer": bound_regularizer,
    }
