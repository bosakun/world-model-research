from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import JEPAConfig
from dataset import NoisyRobotTransitionDataset
from losses import jepa_loss
from model import ActionJEPA


ROOT = Path(__file__).resolve().parent


def build_model(config: JEPAConfig) -> ActionJEPA:
    return ActionJEPA(config.observation_dim, config.action_dim, config.latent_dim, config.hidden_dim)


def train(config: JEPAConfig, output_dir: Path):
    torch.manual_seed(config.seed)
    train_data = NoisyRobotTransitionDataset(config.train_samples, config.seed)
    validation = NoisyRobotTransitionDataset(config.validation_samples, config.seed + 10_000)
    loader = DataLoader(train_data, config.batch_size, shuffle=True)
    model = build_model(config)
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=config.learning_rate)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train(); totals = {key: 0.0 for key in ("total", "prediction", "variance", "covariance")}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            values = jepa_loss(model(batch["observation"], batch["action"], batch["next_observation"]), config.variance_weight, config.covariance_weight)
            values["total"].backward(); optimizer.step(); model.update_target(config.ema)
            for key in totals: totals[key] += float(values[key].detach()) * batch["action"].shape[0]
        model.eval()
        with torch.no_grad():
            values = jepa_loss(model(validation.observation, validation.action, validation.next_observation), config.variance_weight, config.covariance_weight)
        row = {"epoch": epoch, **{f"train_{key}": value / len(train_data) for key, value in totals.items()},
               **{f"validation_{key}": float(value) for key, value in values.items()}}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:03d} prediction={row['validation_prediction']:.5f} variance={row['validation_variance']:.5f} covariance={row['validation_covariance']:.5f}")
    output_dir.mkdir(parents=True, exist_ok=True); steps = config.epochs * len(loader)
    torch.save({"format_version": 1, "model": model.state_dict(), "config": config.to_dict(), "optimizer": "Adam", "training_steps": steps}, output_dir / "checkpoint.pt")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4)); epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["validation_prediction"] for row in history]); axes[0].set(title="JEPA prediction", xlabel="epoch", ylabel="smooth L1")
    axes[1].plot(epochs, [row["validation_variance"] for row in history], label="variance"); axes[1].plot(epochs, [row["validation_covariance"] for row in history], label="covariance"); axes[1].set(title="Anti-collapse regularizers", xlabel="epoch"); axes[1].legend()
    for axis in axes: axis.grid(alpha=.3)
    figure.tight_layout(); figure.savefig(output_dir / "loss_curve.png", dpi=170); plt.close(figure)
    summary = {**config.to_dict(), "optimizer": "Adam", "training_steps": steps,
               "parameter_count": sum(parameter.numel() for parameter in model.parameters()), **{key: value for key, value in history[-1].items() if key != "epoch"}}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=JEPAConfig.epochs); parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs"); args = parser.parse_args()
    _, result = train(JEPAConfig(epochs=args.epochs), args.output_dir); print(result)
