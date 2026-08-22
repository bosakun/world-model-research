from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import TransformerMemoryConfig
from transformer_dataset import build_transformer_dataset
from transformer_losses import transformer_world_model_loss
from transformer_memory import TransformerMemoryWorldModel
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def build_model(config: TransformerMemoryConfig) -> TransformerMemoryWorldModel:
    return TransformerMemoryWorldModel(
        config.latent_dim,
        config.action_dim,
        config.model_dim,
        config.num_heads,
        config.num_layers,
        config.feedforward_dim,
        config.max_context,
        config.dropout,
    )


def _losses(
    model: TransformerMemoryWorldModel,
    batch: dict[str, torch.Tensor],
    config: TransformerMemoryConfig,
) -> dict[str, torch.Tensor]:
    return transformer_world_model_loss(
        model(batch["observations"], batch["actions"]),
        batch["observations"],
        config.latent_prediction_weight,
        config.goal_classification_weight,
        config.green_channel_weight,
    )


def train(config: TransformerMemoryConfig, output_dir: Path) -> dict[str, float | int | str]:
    seed_everything(config.seed)
    train_data = build_transformer_dataset(
        config.train_sequences, config.sequence_length, config.seed
    )
    val_data = build_transformer_dataset(
        config.val_sequences, config.sequence_length, config.seed + 10_000
    )
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, config.batch_size)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    keys = ("total", "reconstruction", "prediction_image", "latent_prediction", "goal_classification")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in keys}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = _losses(model, batch, config)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            for key in keys:
                totals[key] += float(losses[key].detach()) * batch["observations"].shape[0]
        train_metrics = {key: value / len(train_data) for key, value in totals.items()}

        model.eval()
        val_totals = {key: 0.0 for key in keys}
        with torch.no_grad():
            for batch in val_loader:
                losses = _losses(model, batch, config)
                for key in keys:
                    val_totals[key] += float(losses[key]) * batch["observations"].shape[0]
        val_metrics = {key: value / len(val_data) for key, value in val_totals.items()}
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} train={train_metrics['total']:.6f} "
                f"val={val_metrics['total']:.6f} goal={val_metrics['goal_classification']:.4f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "optimizer": "Adam",
            "training_steps": config.epochs * len(train_loader),
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epoch_numbers = [row["epoch"] for row in history]
    axes[0].plot(epoch_numbers, [row["train_total"] for row in history], label="train")
    axes[0].plot(epoch_numbers, [row["val_total"] for row in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Transformer memory objective")
    axes[0].legend()
    axes[1].plot(
        epoch_numbers, [row["val_prediction_image"] for row in history], label="future image"
    )
    axes[1].plot(
        epoch_numbers, [row["val_latent_prediction"] for row in history], label="future latent"
    )
    axes[1].set(xlabel="epoch", title="Validation prediction terms")
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
        "final_val_prediction_image": float(history[-1]["val_prediction_image"]),
        "final_val_latent_prediction": float(history[-1]["val_latent_prediction"]),
        "final_val_goal_classification": float(history[-1]["val_goal_classification"]),
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TransformerMemoryConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    print(train(TransformerMemoryConfig(epochs=arguments.epochs), arguments.output_dir))
