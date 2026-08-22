from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import OvershootingConfig
from latent_dynamics import LatentDynamics
from overshooting_losses import long_horizon_loss
from sequence_dataset import ControlledOscillatorSequenceDataset
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def build_model(config: OvershootingConfig) -> LatentDynamics:
    return LatentDynamics(config.state_dim, config.action_dim, config.hidden_dim)


def train(config: OvershootingConfig, output_dir: Path) -> dict[str, object]:
    seed_everything(config.seed)
    train_data = ControlledOscillatorSequenceDataset(
        config.train_sequences, config.sequence_length, config.seed
    )
    val_data = ControlledOscillatorSequenceDataset(
        config.val_sequences, config.sequence_length, config.seed + 10_000
    )
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, config.batch_size)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_totals = {"total": 0.0, "one_step": 0.0, "overshooting": 0.0}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = long_horizon_loss(
                model,
                batch["states"],
                batch["actions"],
                config.overshooting_distance,
                config.overshooting_weight,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            for key in train_totals:
                train_totals[key] += float(losses[key].detach()) * batch["states"].shape[0]
        train_metrics = {key: value / len(train_data) for key, value in train_totals.items()}

        model.eval()
        val_totals = {key: 0.0 for key in train_totals}
        with torch.no_grad():
            for batch in val_loader:
                losses = long_horizon_loss(
                    model,
                    batch["states"],
                    batch["actions"],
                    config.overshooting_distance,
                    config.overshooting_weight,
                )
                for key in val_totals:
                    val_totals[key] += float(losses[key]) * batch["states"].shape[0]
        val_metrics = {key: value / len(val_data) for key, value in val_totals.items()}
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} train={train_metrics['total']:.7f} "
                f"val={val_metrics['total']:.7f} rollout5={val_metrics['overshooting']:.7f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = config.epochs * len(train_loader)
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "optimizer": "Adam",
            "training_steps": steps,
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = [row["epoch"] for row in history]
    axis.semilogy(epochs, [row["train_total"] for row in history], label="train total")
    axis.semilogy(epochs, [row["val_total"] for row in history], label="validation total")
    axis.semilogy(epochs, [row["val_one_step"] for row in history], label="validation one-step")
    axis.set(xlabel="epoch", ylabel="MSE (log scale)", title="Latent overshooting objective")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=170)
    plt.close(figure)

    summary: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "epochs": config.epochs,
        "training_steps": steps,
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "overshooting_distance": config.overshooting_distance,
        "overshooting_weight": config.overshooting_weight,
        "parameter_count": parameter_count(model),
        "checkpoint_format_version": 1,
        "initial_train_total": history[0]["train_total"],
        "final_train_total": history[-1]["train_total"],
        "final_val_total": history[-1]["val_total"],
        "final_val_one_step": history[-1]["val_one_step"],
        "final_val_overshooting": history[-1]["val_overshooting"],
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=OvershootingConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    print(train(OvershootingConfig(epochs=arguments.epochs), arguments.output_dir))
