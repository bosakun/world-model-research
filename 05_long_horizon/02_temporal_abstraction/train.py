from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import TemporalAbstractionConfig
from macro_dataset import MacroSequenceDataset
from macro_dynamics import MacroDynamics
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def build_model(config: TemporalAbstractionConfig) -> MacroDynamics:
    return MacroDynamics(
        config.state_dim,
        config.action_dim,
        config.action_embedding_dim,
        config.hidden_dim,
    )


def train(config: TemporalAbstractionConfig, output_dir: Path) -> dict[str, object]:
    seed_everything(config.seed)
    train_data = MacroSequenceDataset(
        config.train_sequences, config.sequence_length, config.chunk_size, config.seed
    )
    val_data = MacroSequenceDataset(
        config.val_sequences,
        config.sequence_length,
        config.chunk_size,
        config.seed + 10_000,
    )
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["boundary_states"][:, :-1], batch["action_chunks"])
            loss = F.mse_loss(prediction, batch["boundary_states"][:, 1:])
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * batch["states"].shape[0]
        train_mse = train_total / len(train_data)
        model.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(
                model(val_data.boundary_states[:, :-1], val_data.action_chunks),
                val_data.boundary_states[:, 1:],
            ).item()
        history.append({"epoch": epoch, "train_macro_mse": train_mse, "val_macro_mse": val_mse})
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:03d} train_macro={train_mse:.7f} val_macro={val_mse:.7f}")

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
    axis.semilogy(epochs, [row["train_macro_mse"] for row in history], label="train")
    axis.semilogy(epochs, [row["val_macro_mse"] for row in history], label="validation")
    axis.set(xlabel="epoch", ylabel="five-step macro MSE", title="Temporal abstraction training")
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
        "chunk_size": config.chunk_size,
        "macro_steps_per_rollout": config.sequence_length // config.chunk_size,
        "parameter_count": parameter_count(model),
        "checkpoint_format_version": 1,
        "initial_train_macro_mse": history[0]["train_macro_mse"],
        "final_train_macro_mse": history[-1]["train_macro_mse"],
        "final_val_macro_mse": history[-1]["val_macro_mse"],
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TemporalAbstractionConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    print(train(TemporalAbstractionConfig(epochs=arguments.epochs), arguments.output_dir))
