from __future__ import annotations

import torch

from model import ContrastiveStructuredWorldModel


def contrastive_world_model_loss(
    model: ContrastiveStructuredWorldModel,
    images: torch.Tensor,
    actions: torch.Tensor,
    next_images: torch.Tensor,
    margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    _, predicted = model.predict(images, actions)
    target = model.encoder(next_images)
    positive_energy = ((predicted - target) ** 2).mean(dim=(1, 2))
    negative_target = target.roll(shifts=1, dims=0)
    negative_energy = ((predicted - negative_target) ** 2).mean(dim=(1, 2))
    hinge = torch.relu(margin + positive_energy - negative_energy)
    return {
        "total": (positive_energy + hinge).mean(),
        "positive_energy": positive_energy.mean(),
        "negative_energy": negative_energy.mean(),
        "hinge": hinge.mean(),
    }
