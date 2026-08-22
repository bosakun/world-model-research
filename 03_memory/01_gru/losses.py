import torch
from torch.nn import functional as F


def world_model_loss(
    outputs: dict[str, torch.Tensor],
    observations: torch.Tensor,
    reconstruction_weight: float = 1.0,
    dynamics_weight: float = 2.0,
    position_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    reconstruction = F.mse_loss(outputs["reconstructions"], observations)

    def cell_logits(images: torch.Tensor) -> torch.Tensor:
        """Turn decoded red-vs-other evidence into 25 exclusive cell logits."""
        red_score = images[..., 0, :, :] - torch.maximum(images[..., 1, :, :], images[..., 2, :, :])
        flat = red_score.reshape(-1, 1, 20, 20)
        return 10.0 * F.avg_pool2d(flat, kernel_size=4, stride=4).flatten(1)

    target_cells = cell_logits(observations).argmax(dim=-1)
    position = F.cross_entropy(cell_logits(outputs["reconstructions"]), target_cells)
    # Detaching the target prevents the encoder from making moving targets merely easier
    # for dynamics; reconstruction remains the representation-learning signal.
    target_next_latents = outputs["latents"][:, 1:].detach()
    dynamics = F.mse_loss(outputs["predicted_next_latents"], target_next_latents)
    total = (
        reconstruction_weight * reconstruction
        + dynamics_weight * dynamics
        + position_weight * position
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "dynamics": dynamics,
        "position": position,
    }
