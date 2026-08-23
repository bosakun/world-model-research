from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from baseline import SimpleDynamics
from config import ExperimentConfig
from dataset import GridSequenceDataset
from model import GRUWorldModel
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def _image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _agent_cells(images: torch.Tensor, cell_size: int, grid_size: int) -> torch.Tensor:
    """Locate the red agent by maximum red-vs-other channel score per cell."""
    red_score = images[..., 0, :, :] - torch.maximum(images[..., 1, :, :], images[..., 2, :, :])
    leading = red_score.shape[:-2]
    cells = red_score.reshape(*leading, grid_size, cell_size, grid_size, cell_size).mean(dim=(-1, -3))
    return cells.reshape(*leading, grid_size * grid_size).argmax(dim=-1)


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = GRUWorldModel(config.latent_dim, config.action_dim, config.hidden_dim)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = GridSequenceDataset(
        config.val_sequences, config.sequence_length, config.grid_size, config.cell_size, config.seed + 20_000
    )
    observations, actions = dataset.observations, dataset.actions

    with torch.no_grad():
        latents = model.encoder(observations)
        one_step_latents, hidden_states, _ = model.dynamics(latents[:, :-1], actions)
        one_step_images = model.decoder(one_step_latents)
        rollout_latents, rollout_hidden, _ = model.dynamics.rollout(latents[:, 0], actions)
        rollout_images = model.decoder(rollout_latents)
        one_step_latent_mse = F.mse_loss(one_step_latents, latents[:, 1:]).item()
        one_step_pixel_mse = F.mse_loss(one_step_images, observations[:, 1:]).item()
        horizon_pixel_mse = (
            (rollout_images - observations[:, 1:]).square().mean(dim=(0, 2, 3, 4)).tolist()
        )
        rollout_pixel_mse = float(np.mean(horizon_pixel_mse))
        true_cells = _agent_cells(observations[:, 1:], config.cell_size, config.grid_size)
        one_step_cells = _agent_cells(one_step_images, config.cell_size, config.grid_size)
        rollout_cells = _agent_cells(rollout_images, config.cell_size, config.grid_size)
        one_step_position_accuracy = (one_step_cells == true_cells).float().mean().item()
        rollout_position_accuracy_by_horizon = (
            (rollout_cells == true_cells).float().mean(dim=0).tolist()
        )
        rollout_position_accuracy = float(np.mean(rollout_position_accuracy_by_horizon))

        start = time.perf_counter()
        for _ in range(100):
            model.dynamics.rollout(latents[:, 0], actions)
        elapsed = time.perf_counter() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, config.sequence_length + 1, figsize=(2 * (config.sequence_length + 1), 4))
    axes[0, 0].imshow(_image(observations[0, 0]))
    axes[1, 0].imshow(_image(observations[0, 0]))
    axes[0, 0].set_title("true t=0")
    axes[1, 0].set_title("rollout seed")
    for step in range(config.sequence_length):
        axes[0, step + 1].imshow(_image(observations[0, step + 1]))
        axes[1, step + 1].imshow(_image(rollout_images[0, step]))
        axes[0, step + 1].set_title(f"true t={step + 1}")
        axes[1, step + 1].set_title(f"pred t={step + 1}")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_comparison.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, config.sequence_length + 1), horizon_pixel_mse, marker="o")
    axis.set(xlabel="rollout horizon", ylabel="pixel MSE", title="Autoregressive rollout error")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_error.png", dpi=160)
    plt.close(figure)

    # A randomly initialized baseline is recorded as an interface/parameter reference only;
    # it is deliberately not presented as a trained performance comparison.
    baseline = SimpleDynamics(config.latent_dim, config.action_dim, config.hidden_dim)
    metrics: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "evaluation_sequences": len(dataset),
        "one_step_latent_mse": one_step_latent_mse,
        "one_step_pixel_mse": one_step_pixel_mse,
        "one_step_agent_position_accuracy": one_step_position_accuracy,
        "rollout_mean_pixel_mse": rollout_pixel_mse,
        "rollout_pixel_mse_by_horizon": horizon_pixel_mse,
        "rollout_agent_position_accuracy": rollout_position_accuracy,
        "rollout_agent_position_accuracy_by_horizon": rollout_position_accuracy_by_horizon,
        "hidden_shape": list(hidden_states.shape),
        "rollout_hidden_shape": list(rollout_hidden.shape),
        "gru_dynamics_parameter_count": parameter_count(model.dynamics),
        "simple_dynamics_parameter_count": parameter_count(baseline),
        "mean_rollout_seconds_per_sequence": elapsed / (100 * len(dataset)),
        "baseline_performance": "not measured; baseline is retained for a controlled later comparison",
        "evaluation_entry_point": "python 03_memory/01_gru/evaluate.py",
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
