from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from config import TransformerMemoryConfig
from train import build_model
from transformer_dataset import build_transformer_dataset
from transformer_losses import goal_class_targets
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def _image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TransformerMemoryConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = build_transformer_dataset(
        config.val_sequences, config.sequence_length, config.seed + 20_000
    )
    observations, actions = dataset.observations, dataset.actions

    with torch.no_grad():
        teacher_forced = model(observations, actions)
        reconstruction_mse = F.mse_loss(teacher_forced["reconstructions"], observations).item()
        one_step_pixel_mse = F.mse_loss(
            teacher_forced["predicted_next_observations"], observations[:, 1:]
        ).item()
        one_step_goal_accuracy = (
            teacher_forced["goal_logits"].argmax(-1)
            == goal_class_targets(observations[:, 1:]).reshape(
                observations.shape[0], config.sequence_length
            )
        ).float().mean().item()
        rollout = model.rollout(observations[:, 0], actions)
        rollout_mse_by_horizon = (
            (rollout["predicted_next_observations"] - observations[:, 1:])
            .square()
            .mean(dim=(0, 2, 3, 4))
            .tolist()
        )
        rollout_targets = goal_class_targets(observations[:, 1:]).reshape(
            observations.shape[0], config.sequence_length
        )
        rollout_goal_accuracy_by_horizon = (
            (rollout["goal_logits"].argmax(-1) == rollout_targets).float().mean(dim=0).tolist()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, config.sequence_length, figsize=(2 * config.sequence_length, 4))
    for time_index in range(config.sequence_length):
        axes[0, time_index].imshow(_image(observations[0, time_index + 1]))
        axes[0, time_index].set_title(f"true t={time_index + 1}")
        axes[1, time_index].imshow(
            _image(teacher_forced["predicted_next_observations"][0, time_index])
        )
        axes[1, time_index].set_title(f"causal t={time_index + 1}")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "one_step_prediction.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, config.sequence_length + 1, figsize=(14, 4))
    axes[0, 0].imshow(_image(observations[0, 0]))
    axes[1, 0].imshow(_image(observations[0, 0]))
    axes[0, 0].set_title("true t=0")
    axes[1, 0].set_title("rollout seed")
    for time_index in range(config.sequence_length):
        axes[0, time_index + 1].imshow(_image(observations[0, time_index + 1]))
        axes[1, time_index + 1].imshow(
            _image(rollout["predicted_next_observations"][0, time_index])
        )
        axes[0, time_index + 1].set_title(f"true t={time_index + 1}")
        axes[1, time_index + 1].set_title(f"rollout t={time_index + 1}")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "sequence_rollout.png", dpi=150)
    plt.close(figure)

    attention = teacher_forced["attention_maps"][-1, 0].mean(dim=0).cpu().numpy()
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(attention, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set(
        xlabel="attended history token (key)",
        ylabel="prediction token (query)",
        title="Final-layer causal attention (head mean)",
        xticks=range(config.sequence_length),
        yticks=range(config.sequence_length),
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "attention_map.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, config.sequence_length + 1), rollout_mse_by_horizon, marker="o")
    axis.set(
        xlabel="autoregressive rollout horizon",
        ylabel="pixel MSE",
        title="Transformer memory rollout error",
    )
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_error.png", dpi=170)
    plt.close(figure)

    metrics: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "evaluation_sequences": len(dataset),
        "autoencoder_reconstruction_mse": reconstruction_mse,
        "teacher_forced_one_step_pixel_mse": one_step_pixel_mse,
        "teacher_forced_state_head_goal_class_accuracy": one_step_goal_accuracy,
        "autoregressive_rollout_mean_pixel_mse": float(np.mean(rollout_mse_by_horizon)),
        "autoregressive_rollout_pixel_mse_by_horizon": rollout_mse_by_horizon,
        "autoregressive_rollout_mean_state_head_goal_class_accuracy": float(
            np.mean(rollout_goal_accuracy_by_horizon)
        ),
        "autoregressive_rollout_state_head_goal_class_accuracy_by_horizon": (
            rollout_goal_accuracy_by_horizon
        ),
        "latent_shape": list(teacher_forced["latents"].shape),
        "token_shape": list(teacher_forced["context_tokens"].shape),
        "attention_shape": list(teacher_forced["attention_maps"].shape),
        "parameter_count": parameter_count(model),
        "evaluation_entry_point": "python 03_memory/04_transformer_memory/evaluate.py",
    }
    save_json(output_dir / "evaluation_metrics.json", metrics)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    evaluate(arguments.checkpoint, arguments.output_dir)
