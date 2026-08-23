from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import IntegratedConfig
from dataset import IntegratedNavigationDataset
from losses import integrated_loss
from model import IntegratedWorldModel

ROOT = Path(__file__).resolve().parent
LOSS_NAMES = (
    "total",
    "reconstruction",
    "reward",
    "value",
    "continuation",
    "goal",
    "kl",
    "overshooting",
)


def build_model(config: IntegratedConfig) -> IntegratedWorldModel:
    return IntegratedWorldModel(
        config.embedding_dim,
        config.deterministic_dim,
        config.stochastic_dim,
        config.ensemble_size,
    )


def validation_batch(dataset: IntegratedNavigationDataset) -> dict[str, torch.Tensor]:
    return {
        "observations": dataset.observations,
        "actions": dataset.actions,
        "true_states": dataset.true_states,
        "rewards": dataset.rewards,
        "values": dataset.values,
        "continuations": dataset.continuations,
    }


def train(config: IntegratedConfig, output_directory: Path):
    torch.manual_seed(config.seed)
    training_data = IntegratedNavigationDataset(
        config.train_sequences, config.sequence_length, config.seed
    )
    validation_data = IntegratedNavigationDataset(
        config.validation_sequences, config.sequence_length, config.seed + 10_000
    )
    loader = DataLoader(training_data, config.batch_size, shuffle=True)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in LOSS_NAMES}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses, _ = integrated_loss(model, batch, config)
            losses["total"].backward()
            optimizer.step()
            batch_size = batch["actions"].shape[0]
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach()) * batch_size

        model.eval()
        with torch.no_grad():
            validation_losses, _ = integrated_loss(
                model, validation_batch(validation_data), config
            )
        row = {
            "epoch": epoch,
            **{
                f"train_{name}": value / len(training_data)
                for name, value in totals.items()
            },
            **{
                f"validation_{name}": float(value)
                for name, value in validation_losses.items()
            },
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} total={row['validation_total']:.4f} "
                f"reward={row['validation_reward']:.4f} "
                f"goal={row['validation_goal']:.4f} kl={row['validation_kl']:.4f}"
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    training_steps = config.epochs * len(loader)
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "optimizer": "Adam",
            "training_steps": training_steps,
        },
        output_directory / "checkpoint.pt",
    )
    with (output_directory / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["validation_total"] for row in history])
    axes[0].set(title="Integrated objective", xlabel="epoch")
    for name in ("reward", "value", "goal", "kl", "overshooting"):
        axes[1].plot(
            epochs,
            [row[f"validation_{name}"] for row in history],
            label=name,
        )
    axes[1].set(title="Validation components", xlabel="epoch")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_directory / "loss_curve.png", dpi=170)
    plt.close(figure)

    summary = {
        **config.to_dict(),
        "optimizer": "Adam",
        "training_steps": training_steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **{name: value for name, value in history[-1].items() if name != "epoch"},
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=IntegratedConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    _, training_summary = train(
        IntegratedConfig(epochs=arguments.epochs), arguments.output_dir
    )
    print(training_summary)
