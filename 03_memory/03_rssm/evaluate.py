from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from config import RSSMConfig
from rssm import RSSMState
from rssm_dataset import build_rssm_dataset
from rssm_losses import goal_class_targets
from train import build_model
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def _image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = RSSMConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = build_rssm_dataset(config.val_sequences, config.sequence_length, config.seed + 20_000)
    observations, actions = dataset.observations, dataset.actions

    with torch.no_grad():
        posterior = model.observe(observations, actions, stochastic=False)
        posterior_reconstruction_mse = F.mse_loss(
            posterior["reconstructions"], observations
        ).item()
        targets = goal_class_targets(observations)
        posterior_goal_accuracy = (
            posterior["goal_logits"].reshape(-1, 10).argmax(-1) == targets
        ).float().mean().item()
        prior_images = model.decode(posterior["deterministic_states"], posterior["prior_means"])
        one_step_prior_mse = F.mse_loss(prior_images[:, 1:], observations[:, 1:]).item()
        prior_goal_logits = model.predict_goal(
            posterior["deterministic_states"], posterior["prior_means"]
        )
        one_step_prior_goal_accuracy = (
            prior_goal_logits[:, 1:].reshape(-1, 10).argmax(-1)
            == goal_class_targets(observations[:, 1:])
        ).float().mean().item()

        initial_filter = model.observe(observations[:, :1], actions[:, :0], stochastic=False)
        initial_state = RSSMState(
            initial_filter["deterministic_states"][:, 0],
            initial_filter["posterior_means"][:, 0],
        )
        imagined = model.imagine(initial_state, actions, stochastic=False)
        rollout_mse_by_horizon = (
            (imagined["observations"] - observations[:, 1:]).square().mean(dim=(0, 2, 3, 4)).tolist()
        )
        rollout_predictions = imagined["goal_logits"].argmax(-1)
        rollout_targets = goal_class_targets(observations[:, 1:]).reshape(
            observations.shape[0], config.sequence_length
        )
        rollout_goal_accuracy_by_horizon = (
            (rollout_predictions == rollout_targets).float().mean(dim=0).tolist()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, observations.shape[1], figsize=(2 * observations.shape[1], 4))
    for time_index in range(observations.shape[1]):
        axes[0, time_index].imshow(_image(observations[0, time_index]))
        axes[0, time_index].set_title(f"true t={time_index}")
        axes[1, time_index].imshow(_image(posterior["reconstructions"][0, time_index]))
        axes[1, time_index].set_title(f"posterior t={time_index}")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "reconstruction.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, config.sequence_length + 1, figsize=(2 * (config.sequence_length + 1), 4))
    axes[0, 0].imshow(_image(observations[0, 0]))
    axes[1, 0].imshow(_image(observations[0, 0]))
    axes[0, 0].set_title("true t=0")
    axes[1, 0].set_title("posterior seed")
    for time_index in range(config.sequence_length):
        axes[0, time_index + 1].imshow(_image(observations[0, time_index + 1]))
        axes[1, time_index + 1].imshow(_image(imagined["observations"][0, time_index]))
        axes[0, time_index + 1].set_title(f"true t={time_index + 1}")
        axes[1, time_index + 1].set_title(f"prior t={time_index + 1}")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "latent_rollout.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, config.sequence_length + 1), rollout_mse_by_horizon, marker="o")
    axis.set(xlabel="prior rollout horizon", ylabel="pixel MSE", title="RSSM prior imagination error")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_error.png", dpi=170)
    plt.close(figure)

    metrics: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "evaluation_sequences": len(dataset),
        "posterior_reconstruction_mse": posterior_reconstruction_mse,
        "posterior_state_head_goal_class_accuracy": posterior_goal_accuracy,
        "one_step_prior_pixel_mse": one_step_prior_mse,
        "one_step_prior_state_head_goal_class_accuracy": one_step_prior_goal_accuracy,
        "prior_rollout_mean_pixel_mse": float(np.mean(rollout_mse_by_horizon)),
        "prior_rollout_pixel_mse_by_horizon": rollout_mse_by_horizon,
        "prior_rollout_mean_state_head_goal_class_accuracy": float(
            np.mean(rollout_goal_accuracy_by_horizon)
        ),
        "prior_rollout_state_head_goal_class_accuracy_by_horizon": rollout_goal_accuracy_by_horizon,
        "deterministic_state_shape": list(posterior["deterministic_states"].shape),
        "stochastic_state_shape": list(posterior["stochastic_states"].shape),
        "prior_parameter_shape": list(posterior["prior_means"].shape),
        "posterior_parameter_shape": list(posterior["posterior_means"].shape),
        "parameter_count": parameter_count(model),
        "evaluation_entry_point": "python 03_memory/03_rssm/evaluate.py",
    }
    save_json(output_dir / "evaluation_metrics.json", metrics)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir)
