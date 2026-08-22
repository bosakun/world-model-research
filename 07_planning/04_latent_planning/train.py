from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import LatentPlanningConfig
from dataset import LatentPlanningSequenceDataset
from losses import latent_model_loss
from model import TaskOrientedLatentModel
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def build_model(config: LatentPlanningConfig) -> TaskOrientedLatentModel:
    return TaskOrientedLatentModel(config.observation_dim, config.action_dim, config.latent_dim, config.hidden_dim)


def compute_loss(model, batch, config):
    return latent_model_loss(
        model,
        batch["observations"],
        batch["actions"],
        batch["rewards"],
        batch["values"],
        config.consistency_weight,
        config.reward_weight,
        config.value_weight,
    )


def train(config: LatentPlanningConfig, output_dir: Path) -> tuple[TaskOrientedLatentModel, dict[str, object]]:
    seed_everything(config.seed)
    train_data = LatentPlanningSequenceDataset(config.train_sequences, config.horizon, config.seed, config.action_scale)
    validation_data = LatentPlanningSequenceDataset(
        config.validation_sequences, config.horizon, config.seed + 10_000, config.action_scale
    )
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("total", "consistency", "reward", "value")}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss(model, batch, config)
            losses["total"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch["observations"].shape[0]
        row = {"epoch": epoch, **{f"train_{key}": value / len(train_data) for key, value in totals.items()}}
        model.eval()
        with torch.no_grad():
            validation = compute_loss(
                model,
                {"observations": validation_data.observations, "actions": validation_data.actions,
                 "rewards": validation_data.rewards, "values": validation_data.values},
                config,
            )
        row.update({f"validation_{key}": float(value) for key, value in validation.items()})
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:03d} total={row['validation_total']:.5f} "
                  f"consistency={row['validation_consistency']:.5f} reward={row['validation_reward']:.5f} "
                  f"value={row['validation_value']:.5f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = config.epochs * len(loader)
    torch.save(
        {"format_version": 1, "model": model.state_dict(), "config": config.to_dict(),
         "optimizer": "Adam", "training_steps": steps},
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_total"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_total"] for row in history], label="validation")
    axes[0].set(title="Joint latent model objective", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    for key in ("consistency", "reward", "value"):
        axes[1].plot(epochs, [row[f"validation_{key}"] for row in history], label=key)
    axes[1].set(title="Validation components", xlabel="epoch", ylabel="MSE")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=170)
    plt.close(figure)
    summary = {
        **config.to_dict(), "optimizer": "Adam", "training_steps": steps,
        "parameter_count": parameter_count(model), "checkpoint_format_version": 1,
        **{key: value for key, value in history[-1].items() if key != "epoch"},
    }
    save_json(output_dir / "training_summary.json", summary)
    return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=LatentPlanningConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    _, result = train(LatentPlanningConfig(epochs=args.epochs), args.output_dir)
    print(result)
