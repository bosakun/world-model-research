from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ProbabilisticDynamicsConfig
from probabilistic_losses import diagonal_gaussian_nll
from stochastic_dataset import (
    HeteroscedasticTransitionDataset,
    StochasticPointSequenceDataset,
    transition_noise_std,
)
from train import build_model
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ProbabilisticDynamicsConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    validation = HeteroscedasticTransitionDataset(config.val_transitions, config.seed + 20_000)

    with torch.no_grad():
        prediction = model(validation.states, validation.actions)
        errors = validation.next_states - prediction.mean
        normalized_errors = errors.abs() / prediction.std
        rmse = errors.square().mean().sqrt().item()
        nll = diagonal_gaussian_nll(prediction, validation.next_states).item()
        coverage_1 = (normalized_errors <= 1.0).float().mean().item()
        coverage_2 = (normalized_errors <= 2.0).float().mean().item()
        std_correlation = float(
            np.corrcoef(
                prediction.std.flatten().numpy(), validation.true_noise_std.flatten().numpy()
            )[0, 1]
        )

        x_grid = torch.linspace(-1.0, 1.0, 200)
        probe_states = torch.stack((x_grid, torch.zeros_like(x_grid)), dim=-1)
        probe_action_indices = torch.zeros(200, dtype=torch.long)
        probe_actions = torch.nn.functional.one_hot(probe_action_indices, 4).float()
        probe_prediction = model(probe_states, probe_actions)
        probe_true_std = transition_noise_std(probe_states, probe_action_indices)

        sequences = StochasticPointSequenceDataset(1, config.rollout_horizon, config.seed + 30_000)
        samples = []
        for _ in range(64):
            samples.append(
                model.rollout(sequences.states[:, 0], sequences.actions, stochastic=True)["states"][0]
            )
        rollout_samples = torch.stack(samples)
        mean_rollout = model.rollout(
            sequences.states[:, 0], sequences.actions, stochastic=False
        )["states"][0]

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x_grid, probe_true_std[:, 0], label="true horizontal noise std")
    axis.plot(x_grid, probe_prediction.std[:, 0], label="predicted horizontal std")
    axis.set(
        xlabel="state x coordinate",
        ylabel="standard deviation",
        title="Input-dependent aleatoric uncertainty",
    )
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "aleatoric_std.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    for sample in rollout_samples:
        path = torch.cat((sequences.states[0, :1], sample), dim=0)
        axis.plot(path[:, 0], path[:, 1], color="tab:blue", alpha=0.08)
    true_path = sequences.states[0]
    mean_path = torch.cat((sequences.states[0, :1], mean_rollout), dim=0)
    axis.plot(true_path[:, 0], true_path[:, 1], color="black", marker="o", label="one true rollout")
    axis.plot(mean_path[:, 0], mean_path[:, 1], color="tab:red", marker="x", label="mean rollout")
    axis.set(xlabel="state x", ylabel="state y", title="Sampled probabilistic rollouts")
    axis.legend()
    axis.grid(alpha=0.3)
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(output_dir / "sampled_rollouts.png", dpi=170)
    plt.close(figure)

    metrics: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "evaluation_transitions": len(validation),
        "next_state_rmse": rmse,
        "gaussian_nll": nll,
        "coverage_within_1_std": coverage_1,
        "coverage_within_2_std": coverage_2,
        "predicted_true_std_correlation": std_correlation,
        "mean_predicted_std": prediction.std.mean(dim=0).tolist(),
        "mean_true_std": validation.true_noise_std.mean(dim=0).tolist(),
        "rollout_horizon": config.rollout_horizon,
        "rollout_samples": rollout_samples.shape[0],
        "parameter_count": parameter_count(model),
        "evaluation_entry_point": "python 04_uncertainty/01_probabilistic_dynamics/evaluate.py",
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
