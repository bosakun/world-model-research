from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def prediction_head_loss(
    outputs: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    values: torch.Tensor,
    continuations: torch.Tensor,
    valid: torch.Tensor,
    reward_weight: float = 1.0,
    value_weight: float = 1.0,
    continuation_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    reward = masked_mean((outputs["reward"] - rewards).square(), valid)
    value = masked_mean((outputs["value"] - values).square(), valid)
    continuation = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["continuation_logit"], continuations, reduction="none"
        ),
        valid,
    )
    total = reward_weight * reward + value_weight * value + continuation_weight * continuation
    return {"total": total, "reward": reward, "value": value, "continuation": continuation}
