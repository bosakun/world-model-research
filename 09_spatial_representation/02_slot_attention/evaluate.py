from __future__ import annotations

import itertools, json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import SlotAttentionConfig
from dataset import TwoObjectImagesDataset
from model import SlotAttentionAutoencoder

ROOT = Path(__file__).resolve().parent


def best_permutation_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_labels = predicted.argmax(dim=1); target_labels = target.argmax(dim=1); scores = []
    for batch_index in range(predicted.shape[0]):
        best = 0.0
        for permutation in itertools.permutations(range(predicted.shape[1])):
            mapped = torch.empty_like(predicted_labels[batch_index])
            for predicted_index, target_index in enumerate(permutation): mapped[predicted_labels[batch_index] == predicted_index] = target_index
            per_slot = []
            for slot in range(predicted.shape[1]):
                intersection = ((mapped == slot) & (target_labels[batch_index] == slot)).sum()
                union = ((mapped == slot) | (target_labels[batch_index] == slot)).sum().clamp_min(1)
                per_slot.append(intersection.float() / union)
            best = max(best, float(torch.stack(per_slot).mean()))
        scores.append(best)
    return torch.tensor(scores)


def evaluate(output_dir: Path = ROOT / "outputs"):
    config = SlotAttentionConfig(); checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=False)
    model = SlotAttentionAutoencoder(config.image_size, config.num_slots, config.slot_dim, config.iterations)
    model.load_state_dict(checkpoint["model"]); model.eval(); data = TwoObjectImagesDataset(128, config.seed + 20_000, config.image_size)
    torch.manual_seed(config.seed)
    with torch.no_grad(): output = model(data.images, True)
    mse = torch.nn.functional.mse_loss(output["reconstruction"], data.images); iou = best_permutation_iou(output["masks"], data.masks)
    index = int(iou.argmax()); figure, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes[0,0].imshow(data.images[index].permute(1,2,0)); axes[0,0].set_title("input")
    axes[0,1].imshow(output["reconstruction"][index].permute(1,2,0)); axes[0,1].set_title("reconstruction")
    axes[0,2].imshow(data.masks[index].argmax(dim=0), cmap="tab10", vmin=0, vmax=2); axes[0,2].set_title("true components")
    axes[0,3].imshow(output["masks"][index, :, 0].argmax(dim=0), cmap="tab10", vmin=0, vmax=2); axes[0,3].set_title("predicted slots")
    for slot in range(3):
        axes[1,slot].imshow(output["masks"][index,slot,0], cmap="viridis", vmin=0, vmax=1); axes[1,slot].set_title(f"slot mask {slot}")
    axes[1,3].imshow((output["slot_rgb"][index] * output["masks"][index]).sum(dim=0).permute(1,2,0)); axes[1,3].set_title("slot composition")
    for axis in axes.flat: axis.axis("off")
    figure.tight_layout(); figure.savefig(output_dir / "slot_decomposition.png", dpi=170); plt.close(figure)
    metrics = {"dataset_version": config.dataset_version, "seed": config.seed, "reconstruction_mse": float(mse),
               "mean_best_permutation_iou": float(iou.mean()), "best_example_iou": float(iou[index]),
               "evaluation_entry_point": "python 09_spatial_representation/02_slot_attention/evaluate.py"}
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n"); print(json.dumps(metrics, indent=2)); return metrics


if __name__ == "__main__": evaluate()
