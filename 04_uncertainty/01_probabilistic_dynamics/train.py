from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import ProbabilisticDynamicsConfig
from probabilistic_dynamics import ProbabilisticDynamics
from probabilistic_losses import probabilistic_dynamics_loss
from stochastic_dataset import HeteroscedasticTransitionDataset
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def build_model(config: ProbabilisticDynamicsConfig) -> ProbabilisticDynamics:
    return ProbabilisticDynamics(config.state_dim, config.action_dim, config.hidden_dim)


def train(config: ProbabilisticDynamicsConfig, output_dir: Path) -> dict[str, object]:
    seed_everything(config.seed)
    train_data = HeteroscedasticTransitionDataset(config.train_transitions, config.seed)
    val_data = HeteroscedasticTransitionDataset(config.val_transitions, config.seed + 10_000)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, config.batch_size)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_totals = {"total": 0.0, "negative_log_likelihood": 0.0}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = probabilistic_dynamics_loss(
                model, model(batch["states"], batch["actions"]), batch["next_states"]
            )
            losses["total"].backward()
            optimizer.step()
            for key in train_totals:
                train_totals[key] += float(losses[key].detach()) * batch["states"].shape[0]
        train_metrics = {key: value / len(train_data) for key, value in train_totals.items()}

        model.eval()
        val_totals = {"total": 0.0, "negative_log_likelihood": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                losses = probabilistic_dynamics_loss(
                    model, model(batch["states"], batch["actions"]), batch["next_states"]
                )
                for key in val_totals:
                    val_totals[key] += float(losses[key]) * batch["states"].shape[0]
        val_metrics = {key: value / len(val_data) for key, value in val_totals.items()}
        history.append(
            {
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "train_nll": train_metrics["negative_log_likelihood"],
                "val_total": val_metrics["total"],
                "val_nll": val_metrics["negative_log_likelihood"],
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} train_nll={train_metrics['negative_log_likelihood']:.5f} "
                f"val_nll={val_metrics['negative_log_likelihood']:.5f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_steps = config.epochs * len(train_loader)
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "optimizer": "Adam",
            "training_steps": training_steps,
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["train_nll"] for row in history], label="train NLL")
    axis.plot(epochs, [row["val_nll"] for row in history], label="validation NLL")
    axis.set(xlabel="epoch", ylabel="Gaussian NLL", title="Probabilistic dynamics training")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=170)
    plt.close(figure)

    summary: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "epochs": config.epochs,
        "training_steps": training_steps,
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "parameter_count": parameter_count(model),
        "checkpoint_format_version": 1,
        "initial_train_nll": history[0]["train_nll"],
        "final_train_nll": history[-1]["train_nll"],
        "final_val_nll": history[-1]["val_nll"],
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=ProbabilisticDynamicsConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    print(train(ProbabilisticDynamicsConfig(epochs=arguments.epochs), arguments.output_dir))
