from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import PredictionHeadsConfig
from navigation_dataset import GoalNavigationSequenceDataset
from prediction_heads import RewardValueContinuationHeads
from prediction_losses import prediction_head_loss
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def build_model(config: PredictionHeadsConfig) -> RewardValueContinuationHeads:
    return RewardValueContinuationHeads(config.state_dim, config.action_dim, config.hidden_dim)


def _loss(model, batch, config):
    return prediction_head_loss(
        model(batch["states"][:, :-1], batch["actions"]),
        batch["rewards"],
        batch["value_targets"],
        batch["continuations"],
        batch["valid"],
        config.reward_weight,
        config.value_weight,
        config.continuation_weight,
    )


def train(config: PredictionHeadsConfig, output_dir: Path) -> dict[str, object]:
    seed_everything(config.seed)
    train_data = GoalNavigationSequenceDataset(
        config.train_sequences, config.horizon, config.discount, config.seed
    )
    val_data = GoalNavigationSequenceDataset(
        config.val_sequences, config.horizon, config.discount, config.seed + 10_000
    )
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    keys = ("total", "reward", "value", "continuation")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in keys}
        valid_count = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = _loss(model, batch, config)
            losses["total"].backward()
            optimizer.step()
            count = float(batch["valid"].sum())
            valid_count += count
            for key in keys:
                totals[key] += float(losses[key].detach()) * count
        train_metrics = {key: value / valid_count for key, value in totals.items()}
        model.eval()
        with torch.no_grad():
            validation_batch = {
                "states": val_data.states,
                "actions": val_data.actions,
                "rewards": val_data.rewards,
                "value_targets": val_data.value_targets,
                "continuations": val_data.continuations,
                "valid": val_data.valid,
            }
            val_losses = _loss(
                model,
                validation_batch,
                config,
            )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": float(val_losses[key]) for key in keys})
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} total={train_metrics['total']:.5f} "
                f"reward={float(val_losses['reward']):.5f} value={float(val_losses['value']):.5f} "
                f"continue={float(val_losses['continuation']):.5f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = config.epochs * len(train_loader)
    torch.save(
        {"format_version": 1, "model": model.state_dict(), "config": config.to_dict(), "optimizer": "Adam", "training_steps": steps},
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_total"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_total"] for row in history], label="validation")
    axes[0].set(title="Joint prediction-head objective", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    for key in ("reward", "value", "continuation"):
        axes[1].plot(epochs, [row[f"val_{key}"] for row in history], label=key)
    axes[1].set(title="Validation loss components", xlabel="epoch")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=170)
    plt.close(figure)
    summary = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "discount": config.discount,
        "epochs": config.epochs,
        "training_steps": steps,
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "parameter_count": parameter_count(model),
        "checkpoint_format_version": 1,
        "final_train_total": history[-1]["train_total"],
        "final_val_total": history[-1]["val_total"],
        "final_val_reward": history[-1]["val_reward"],
        "final_val_value": history[-1]["val_value"],
        "final_val_continuation": history[-1]["val_continuation"],
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=PredictionHeadsConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    print(train(PredictionHeadsConfig(epochs=arguments.epochs), arguments.output_dir))
