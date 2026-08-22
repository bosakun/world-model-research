from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import CSWMConfig
from dataset import TwoObjectTransitionDataset
from model import ContrastiveStructuredWorldModel


ROOT = Path(__file__).resolve().parent


def linear_probe(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    augmented = torch.cat((features, torch.ones(features.shape[0], 1)), dim=-1)
    return torch.linalg.lstsq(augmented, targets).solution


def evaluate(output_dir: Path = ROOT / "outputs") -> dict[str, object]:
    config = CSWMConfig(); checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=False)
    model = ContrastiveStructuredWorldModel(config.num_objects, config.action_dim, config.slot_dim, config.hidden_dim, config.image_size)
    model.load_state_dict(checkpoint["model"]); model.eval()
    probe_data = TwoObjectTransitionDataset(512, config.seed + 20_000, config.image_size)
    test_data = TwoObjectTransitionDataset(128, config.seed + 30_000, config.image_size)
    with torch.no_grad():
        probe_slots = model.encoder(probe_data.images)
        test_slots = model.encoder(test_data.images)
        _, predicted_slots = model.predict(test_data.images, test_data.actions)
    weights = linear_probe(probe_slots.reshape(-1, config.slot_dim), probe_data.positions.reshape(-1, 2))
    def decode(slots):
        flat = slots.reshape(-1, config.slot_dim); augmented = torch.cat((flat, torch.ones(flat.shape[0], 1)), dim=-1)
        return (augmented @ weights).reshape(-1, config.num_objects, 2)
    decoded_current, decoded_next = decode(test_slots), decode(predicted_slots)
    current_rmse = torch.sqrt(((decoded_current - test_data.positions) ** 2).mean())
    next_rmse = torch.sqrt(((decoded_next - test_data.next_positions) ** 2).mean())
    index = 0
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].imshow(test_data.images[index].permute(1, 2, 0)); axes[0].set_title("current image")
    axes[1].imshow(test_data.next_images[index].permute(1, 2, 0)); axes[1].set_title("true next image")
    axes[2].scatter(test_data.next_positions[index, :, 0], test_data.next_positions[index, :, 1], label="true", marker="o")
    axes[2].scatter(decoded_next[index, :, 0], decoded_next[index, :, 1], label="latent probe", marker="x")
    axes[2].set(xlim=(-1, 1), ylim=(-1, 1), title="object-wise next state"); axes[2].legend(); axes[2].grid(alpha=0.3)
    for axis in axes[:2]: axis.axis("off")
    figure.tight_layout(); figure.savefig(output_dir / "object_transition.png", dpi=170); plt.close(figure)
    metrics = {"dataset_version": config.dataset_version, "seed": config.seed,
               "current_position_probe_rmse": float(current_rmse), "predicted_next_position_probe_rmse": float(next_rmse),
               "slot_assignment": "fixed by color channel", "evaluation_entry_point": "python 09_spatial_representation/01_cswm/evaluate.py"}
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n"); print(json.dumps(metrics, indent=2)); return metrics


if __name__ == "__main__": evaluate()
