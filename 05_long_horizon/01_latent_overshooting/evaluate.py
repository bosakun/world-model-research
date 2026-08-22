from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from config import OvershootingConfig
from overshooting_losses import latent_overshooting_loss
from sequence_dataset import ControlledOscillatorSequenceDataset
from train import build_model
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = OvershootingConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = ControlledOscillatorSequenceDataset(
        config.val_sequences, config.sequence_length, config.seed + 20_000
    )

    with torch.no_grad():
        one_step = model(dataset.states[:, :-1], dataset.actions)
        rollout = model.rollout(dataset.states[:, 0], dataset.actions)
        one_step_mse = F.mse_loss(one_step, dataset.states[:, 1:]).item()
        rollout_mse_by_horizon = (
            (rollout - dataset.states[:, 1:]).square().mean(dim=(0, 2)).tolist()
        )
        overshooting_mse, by_distance = latent_overshooting_loss(
            model, dataset.states, dataset.actions, config.overshooting_distance
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    time = np.arange(config.sequence_length + 1)
    predicted = torch.cat((dataset.states[:1, :1], rollout[:1]), dim=1)[0]
    axes[0].plot(time, dataset.states[0, :, 0], label="true position")
    axes[0].plot(time, predicted[:, 0], label="rollout position", linestyle="--")
    axes[1].plot(time, dataset.states[0, :, 1], label="true velocity")
    axes[1].plot(time, predicted[:, 1], label="rollout velocity", linestyle="--")
    axes[0].set(ylabel="position", title="Thirty-step autoregressive latent rollout")
    axes[1].set(xlabel="time", ylabel="velocity")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "long_rollout.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(range(1, config.sequence_length + 1), rollout_mse_by_horizon, marker="o")
    axis.axhline(one_step_mse, color="tab:red", linestyle="--", label="teacher-forced one-step MSE")
    axis.set(xlabel="rollout horizon", ylabel="state MSE (log scale)", title="Compounding error")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "compounding_error.png", dpi=170)
    plt.close(figure)

    metrics: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "evaluation_sequences": len(dataset),
        "teacher_forced_one_step_mse": one_step_mse,
        "overshooting_mean_mse": float(overshooting_mse),
        "overshooting_mse_by_distance_1_to_5": by_distance.tolist(),
        "autoregressive_rollout_mean_mse": float(np.mean(rollout_mse_by_horizon)),
        "autoregressive_rollout_mse_by_horizon": rollout_mse_by_horizon,
        "rollout_mse_horizon_5": rollout_mse_by_horizon[4],
        "rollout_mse_horizon_10": rollout_mse_by_horizon[9],
        "rollout_mse_horizon_30": rollout_mse_by_horizon[29],
        "parameter_count": parameter_count(model),
        "evaluation_entry_point": "python 05_long_horizon/01_latent_overshooting/evaluate.py",
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
