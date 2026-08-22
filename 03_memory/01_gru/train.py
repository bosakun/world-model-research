from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from dataset import GridSequenceDataset
from losses import world_model_loss
from model import GRUWorldModel
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def train(config: ExperimentConfig, output_dir: Path) -> dict[str, float | int]:
    seed_everything(config.seed)
    train_data = GridSequenceDataset(
        config.train_sequences, config.sequence_length, config.grid_size, config.cell_size, config.seed
    )
    val_data = GridSequenceDataset(
        config.val_sequences, config.sequence_length, config.grid_size, config.cell_size, config.seed + 10_000
    )
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    model = GRUWorldModel(config.latent_dim, config.action_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"total": 0.0, "reconstruction": 0.0, "dynamics": 0.0, "position": 0.0}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["observations"], batch["actions"])
            losses = world_model_loss(
                outputs,
                batch["observations"],
                config.reconstruction_weight,
                config.dynamics_weight,
                config.position_weight,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch["observations"].shape[0]
        train_metrics = {key: value / len(train_data) for key, value in totals.items()}

        model.eval()
        val_totals = {"total": 0.0, "reconstruction": 0.0, "dynamics": 0.0, "position": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                losses = world_model_loss(model(batch["observations"], batch["actions"]), batch["observations"])
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
                f"val={val_metrics['total']:.6f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, output_dir / "checkpoint.pt")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["epoch"] for row in history], [row["train_total"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_total"] for row in history], label="validation")
    axis.set(xlabel="epoch", ylabel="total loss", title="GRU world-model training")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(figure)

    summary: dict[str, float | int] = {
        "epochs": config.epochs,
        "parameter_count": parameter_count(model),
        "initial_train_total": float(history[0]["train_total"]),
        "final_train_total": float(history[-1]["train_total"]),
        "final_val_total": float(history[-1]["val_total"]),
        "final_val_reconstruction": float(history[-1]["val_reconstruction"]),
        "final_val_dynamics": float(history[-1]["val_dynamics"]),
        "final_val_position_cross_entropy": float(history[-1]["val_position"]),
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--train-sequences", type=int, default=ExperimentConfig.train_sequences)
    parser.add_argument("--val-sequences", type=int, default=ExperimentConfig.val_sequences)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    configuration = ExperimentConfig(
        epochs=args.epochs,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
    )
    print(train(configuration, args.output_dir))
