from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import JEPAConfig
from dataset import NoisyRobotTransitionDataset
from train import build_model


ROOT = Path(__file__).resolve().parent


def fit_probe(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    augmented = torch.cat((features, torch.ones(features.shape[0], 1)), dim=-1)
    return torch.linalg.lstsq(augmented, targets).solution


def apply_probe(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.cat((features, torch.ones(features.shape[0], 1)), dim=-1) @ weights


def evaluate(output_dir: Path = ROOT / "outputs"):
    config = JEPAConfig(); checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=False); model = build_model(config); model.load_state_dict(checkpoint["model"]); model.eval()
    probe = NoisyRobotTransitionDataset(1024, config.seed + 20_000); test = NoisyRobotTransitionDataset(512, config.seed + 30_000)
    with torch.no_grad():
        probe_latent = model.encoder(probe.observation); test_latent = model.encoder(test.observation)
        predicted = model.predictor(torch.cat((test_latent, test.action), -1)); no_action = model.predictor(torch.cat((test_latent, torch.zeros_like(test.action)), -1))
        target = model.target_encoder(test.next_observation)
    weights = fit_probe(probe_latent, probe.true_state)
    predicted_state = apply_probe(predicted, weights); no_action_state = apply_probe(no_action, weights)
    predicted_rmse = torch.sqrt(torch.nn.functional.mse_loss(predicted_state, test.next_true_state)); no_action_rmse = torch.sqrt(torch.nn.functional.mse_loss(no_action_state, test.next_true_state))
    target_rmse = torch.sqrt(torch.nn.functional.mse_loss(apply_probe(target, weights), test.next_true_state)); std = predicted.std(dim=0).mean()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4)); index = torch.arange(40)
    axes[0].quiver(test.true_state[index, 0], test.true_state[index, 1], test.next_true_state[index, 0]-test.true_state[index, 0], test.next_true_state[index, 1]-test.true_state[index, 1], color="black", label="true")
    axes[0].scatter(predicted_state[index, 0], predicted_state[index, 1], marker="x", label="JEPA probe"); axes[0].set(title="Action-conditioned physical prediction", xlabel="x", ylabel="y"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].bar(["action", "zero action", "target enc"], [float(predicted_rmse), float(no_action_rmse), float(target_rmse)]); axes[1].set(title="Linear-probe next-state RMSE", ylabel="RMSE")
    figure.tight_layout(); figure.savefig(output_dir / "physical_prediction.png", dpi=170); plt.close(figure)
    metrics = {"dataset_version": config.dataset_version, "seed": config.seed, "action_conditioned_probe_rmse": float(predicted_rmse), "zero_action_probe_rmse": float(no_action_rmse), "target_encoder_probe_rmse": float(target_rmse), "mean_predicted_latent_std": float(std), "uses_pixel_decoder": False, "evaluation_entry_point": "python 12_physical_ai/01_action_conditioned_jepa/evaluate.py"}
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n"); print(json.dumps(metrics, indent=2)); return metrics


if __name__ == "__main__": evaluate()
