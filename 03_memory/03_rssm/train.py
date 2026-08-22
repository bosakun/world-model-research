from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import RSSMConfig
from rssm import RecurrentStateSpaceModel
from rssm_dataset import build_rssm_dataset
from rssm_losses import rssm_loss
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def build_model(config: RSSMConfig) -> RecurrentStateSpaceModel:
    return RecurrentStateSpaceModel(
        config.action_dim,
        config.observation_embedding_dim,
        config.deterministic_dim,
        config.stochastic_dim,
        config.hidden_mlp_dim,
        config.min_std,
    )


def train(config: RSSMConfig, output_dir: Path) -> dict[str, float | int | str]:
    seed_everything(config.seed)
    train_data = build_rssm_dataset(config.train_sequences, config.sequence_length, config.seed)
    val_data = build_rssm_dataset(config.val_sequences, config.sequence_length, config.seed + 10_000)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, config.batch_size)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {
            key: 0.0
            for key in ("total", "reconstruction", "goal_classification", "kl", "kl_raw")
        }
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = rssm_loss(
                model.observe(batch["observations"], batch["actions"], stochastic=True),
                batch["observations"],
                config.kl_weight,
                config.free_nats,
                config.goal_classification_weight,
                config.green_channel_weight,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch["observations"].shape[0]
        train_metrics = {key: value / len(train_data) for key, value in totals.items()}

        model.eval()
        val_totals = {key: 0.0 for key in totals}
        with torch.no_grad():
            for batch in val_loader:
                losses = rssm_loss(
                    model.observe(batch["observations"], batch["actions"], stochastic=False),
                    batch["observations"],
                    config.kl_weight,
                    config.free_nats,
                    config.goal_classification_weight,
                    config.green_channel_weight,
                )
                for key in val_totals:
                    val_totals[key] += float(losses[key]) * batch["observations"].shape[0]
        val_metrics = {key: value / len(val_data) for key, value in val_totals.items()}
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} train={train_metrics['total']:.6f} "
                f"val={val_metrics['total']:.6f} kl_raw={val_metrics['kl_raw']:.4f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "config": config.to_dict(),
        "optimizer": "Adam",
        "training_steps": config.epochs * len(train_loader),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_total"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_total"] for row in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="total loss", title="RSSM objective")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_reconstruction"] for row in history], label="reconstruction")
    axes[1].plot(epochs, [row["val_kl_raw"] for row in history], label="raw KL")
    axes[1].set(xlabel="epoch", title="Validation loss components")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=170)
    plt.close(figure)

    summary: dict[str, float | int | str] = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "epochs": config.epochs,
        "training_steps": config.epochs * len(train_loader),
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "parameter_count": parameter_count(model),
        "checkpoint_format_version": 1,
        "initial_train_total": float(history[0]["train_total"]),
        "final_train_total": float(history[-1]["train_total"]),
        "final_val_total": float(history[-1]["val_total"]),
        "final_val_reconstruction": float(history[-1]["val_reconstruction"]),
        "final_val_goal_classification": float(history[-1]["val_goal_classification"]),
        "final_val_kl_raw": float(history[-1]["val_kl_raw"]),
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=RSSMConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    print(train(RSSMConfig(epochs=args.epochs), args.output_dir))
