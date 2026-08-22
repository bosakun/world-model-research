from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from config import TemporalAbstractionConfig
from macro_dataset import MacroSequenceDataset
from train import build_model
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TemporalAbstractionConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = MacroSequenceDataset(
        config.val_sequences,
        config.sequence_length,
        config.chunk_size,
        config.seed + 20_000,
    )
    with torch.no_grad():
        teacher_forced = model(dataset.boundary_states[:, :-1], dataset.action_chunks)
        rollout = model.rollout(dataset.boundary_states[:, 0], dataset.action_chunks)
        teacher_mse = F.mse_loss(teacher_forced, dataset.boundary_states[:, 1:]).item()
        rollout_mse = (
            (rollout - dataset.boundary_states[:, 1:]).square().mean(dim=(0, 2)).tolist()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    macro_times = np.arange(0, config.sequence_length + 1, config.chunk_size)
    predicted = torch.cat((dataset.boundary_states[:1, :1], rollout[:1]), dim=1)[0]
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for dimension, label in enumerate(("position", "velocity")):
        axes[dimension].plot(macro_times, dataset.boundary_states[0, :, dimension], marker="o", label="true boundary")
        axes[dimension].plot(macro_times, predicted[:, dimension], marker="x", linestyle="--", label="macro rollout")
        axes[dimension].set(ylabel=label)
        axes[dimension].legend()
        axes[dimension].grid(alpha=0.3)
    axes[0].set_title("Thirty primitive steps as six macro transitions")
    axes[1].set_xlabel("primitive time")
    figure.tight_layout()
    figure.savefig(output_dir / "macro_rollout.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(macro_times[1:], rollout_mse, marker="o")
    axis.axhline(teacher_mse, color="tab:red", linestyle="--", label="teacher-forced macro MSE")
    axis.set(xlabel="equivalent primitive horizon", ylabel="boundary state MSE", title="Macro rollout compounding error")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "macro_error.png", dpi=170)
    plt.close(figure)

    metrics: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "evaluation_sequences": len(dataset),
        "chunk_size": config.chunk_size,
        "primitive_horizon": config.sequence_length,
        "macro_rollout_steps": config.sequence_length // config.chunk_size,
        "teacher_forced_macro_mse": teacher_mse,
        "macro_rollout_mean_mse": float(np.mean(rollout_mse)),
        "macro_rollout_mse_by_primitive_horizon": {
            str(int(horizon)): error for horizon, error in zip(macro_times[1:], rollout_mse, strict=True)
        },
        "macro_rollout_mse_horizon_30": rollout_mse[-1],
        "parameter_count": parameter_count(model),
        "evaluation_entry_point": "python 05_long_horizon/02_temporal_abstraction/evaluate.py",
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
