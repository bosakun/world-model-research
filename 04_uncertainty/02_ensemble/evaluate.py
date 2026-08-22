from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import EnsembleConfig
from ensemble_dataset import (
    HeteroscedasticTransitionDataset,
    StochasticPointSequenceDataset,
    transition_noise_std,
)
from train import build_model
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def moment_gaussian_nll(
    mean: torch.Tensor, variance: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return (
        0.5 * (variance.log() + (targets - mean).square() / variance + math.log(2.0 * math.pi))
    ).sum(dim=-1).mean()


def _ood_inputs(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    signs = torch.where(torch.arange(count) % 2 == 0, 1.0, -1.0)
    x = signs * (1.1 + 0.4 * torch.rand(count, generator=generator))
    y = 3.0 * torch.rand(count, generator=generator) - 1.5
    action_indices = torch.randint(0, 4, (count,), generator=generator)
    return torch.stack((x, y), dim=-1), torch.nn.functional.one_hot(action_indices, 4).float()


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = EnsembleConfig(**checkpoint["config"])
    seed_everything(config.seed)
    ensemble = build_model(config)
    ensemble.load_state_dict(checkpoint["model"])
    ensemble.eval()
    validation = HeteroscedasticTransitionDataset(config.val_transitions, config.seed + 20_000)
    ood_states, ood_actions = _ood_inputs(config.val_transitions, config.seed + 30_000)

    with torch.no_grad():
        in_distribution = ensemble(validation.states, validation.actions)
        out_of_distribution = ensemble(ood_states, ood_actions)
        errors = validation.next_states - in_distribution["mean"]
        total_std = in_distribution["total_variance"].sqrt()
        coverage_1 = (errors.abs() <= total_std).float().mean().item()
        coverage_2 = (errors.abs() <= 2.0 * total_std).float().mean().item()
        nll = moment_gaussian_nll(
            in_distribution["mean"], in_distribution["total_variance"], validation.next_states
        ).item()
        aleatoric_std = in_distribution["aleatoric_variance"].sqrt()
        aleatoric_correlation = float(
            np.corrcoef(
                aleatoric_std.flatten().numpy(), validation.true_noise_std.flatten().numpy()
            )[0, 1]
        )
        id_epistemic = in_distribution["epistemic_variance"].mean().sqrt().item()
        ood_epistemic = out_of_distribution["epistemic_variance"].mean().sqrt().item()

        grid_axis = torch.linspace(-1.5, 1.5, 101)
        grid_y, grid_x = torch.meshgrid(grid_axis, grid_axis, indexing="ij")
        grid_states = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
        right_actions = torch.nn.functional.one_hot(
            torch.ones(grid_states.shape[0], dtype=torch.long), 4
        ).float()
        grid_prediction = ensemble(grid_states, right_actions)
        epistemic_map = grid_prediction["epistemic_variance"].mean(dim=-1).sqrt().reshape(101, 101)

        x_probe = torch.linspace(-1.5, 1.5, 250)
        probe_states = torch.stack((x_probe, torch.zeros_like(x_probe)), dim=-1)
        probe_actions = torch.nn.functional.one_hot(torch.zeros(250, dtype=torch.long), 4).float()
        probe = ensemble(probe_states, probe_actions)
        true_std = transition_noise_std(probe_states, torch.zeros(250, dtype=torch.long))[:, 0]

        sequence = StochasticPointSequenceDataset(1, config.rollout_horizon, config.seed + 40_000)
        ts_infinity = ensemble.rollout(
            sequence.states[:, 0],
            sequence.actions,
            config.rollout_particles,
            "ts_infinity",
            torch.Generator().manual_seed(config.seed + 50_000),
        )
        ts1 = ensemble.rollout(
            sequence.states[:, 0],
            sequence.actions,
            config.rollout_particles,
            "ts1",
            torch.Generator().manual_seed(config.seed + 50_000),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        epistemic_map.numpy(),
        origin="lower",
        extent=(-1.5, 1.5, -1.5, 1.5),
        cmap="magma",
        aspect="equal",
    )
    axis.add_patch(plt.Rectangle((-0.8, -0.8), 1.6, 1.6, fill=False, color="cyan", linewidth=2))
    axis.set(
        xlabel="state x",
        ylabel="state y",
        title="Ensemble epistemic std (cyan = training support)",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "epistemic_map.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(x_probe, true_std, label="true aleatoric std", color="black", linestyle="--")
    axis.plot(x_probe, probe["aleatoric_variance"][:, 0].sqrt(), label="ensemble aleatoric")
    axis.plot(x_probe, probe["epistemic_variance"][:, 0].sqrt(), label="ensemble epistemic")
    axis.plot(x_probe, probe["total_variance"][:, 0].sqrt(), label="total (variance sum)")
    axis.axvspan(-0.8, 0.8, alpha=0.1, color="green", label="training x support")
    axis.set(xlabel="state x", ylabel="standard deviation", title="Uncertainty decomposition")
    axis.legend(ncol=2)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "uncertainty_decomposition.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, trajectories, title in (
        (axes[0], ts_infinity["states"][0], "TS-infinity: fixed model per particle"),
        (axes[1], ts1["states"][0], "TS1: resample model each step"),
    ):
        for trajectory in trajectories:
            path = torch.cat((sequence.states[0, :1], trajectory), dim=0)
            axis.plot(path[:, 0], path[:, 1], color="tab:blue", alpha=0.05)
        axis.plot(sequence.states[0, :, 0], sequence.states[0, :, 1], color="black", marker="o")
        axis.set(xlabel="state x", ylabel="state y", title=title)
        axis.grid(alpha=0.3)
        axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_sampling.png", dpi=170)
    plt.close(figure)

    metrics: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "ensemble_size": config.ensemble_size,
        "evaluation_transitions": len(validation),
        "moment_matched_gaussian_nll": nll,
        "next_state_rmse": errors.square().mean().sqrt().item(),
        "coverage_within_1_total_std": coverage_1,
        "coverage_within_2_total_std": coverage_2,
        "aleatoric_predicted_true_std_correlation": aleatoric_correlation,
        "in_distribution_epistemic_std": id_epistemic,
        "out_of_distribution_epistemic_std": ood_epistemic,
        "ood_to_id_epistemic_ratio": ood_epistemic / id_epistemic,
        "mean_in_distribution_aleatoric_std": aleatoric_std.mean().item(),
        "rollout_horizon": config.rollout_horizon,
        "rollout_particles_per_method": config.rollout_particles,
        "propagation_methods": ["ts_infinity", "ts1"],
        "parameter_count": parameter_count(ensemble),
        "evaluation_entry_point": "python 04_uncertainty/02_ensemble/evaluate.py",
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
