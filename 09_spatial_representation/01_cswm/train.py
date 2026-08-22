from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import CSWMConfig
from dataset import TwoObjectTransitionDataset
from losses import contrastive_world_model_loss
from model import ContrastiveStructuredWorldModel


ROOT = Path(__file__).resolve().parent


def build_model(config: CSWMConfig) -> ContrastiveStructuredWorldModel:
    return ContrastiveStructuredWorldModel(
        config.num_objects, config.action_dim, config.slot_dim, config.hidden_dim, config.image_size
    )


def loss_for(model, batch, config):
    return contrastive_world_model_loss(model, batch["image"], batch["action"], batch["next_image"], config.margin)


def train(config: CSWMConfig, output_dir: Path):
    torch.manual_seed(config.seed)
    train_data = TwoObjectTransitionDataset(config.train_samples, config.seed, config.image_size)
    validation_data = TwoObjectTransitionDataset(config.validation_samples, config.seed + 10_000, config.image_size)
    loader = DataLoader(train_data, config.batch_size, shuffle=True)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(1, config.epochs + 1):
        totals = {key: 0.0 for key in ("total", "positive_energy", "negative_energy", "hinge")}
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for(model, batch, config)
            losses["total"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch["image"].shape[0]
        model.eval()
        with torch.no_grad():
            validation = loss_for(model, {
                "image": validation_data.images, "action": validation_data.actions,
                "next_image": validation_data.next_images,
            }, config)
        row = {"epoch": epoch, **{f"train_{key}": value / len(train_data) for key, value in totals.items()},
               **{f"validation_{key}": float(value) for key, value in validation.items()}}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:03d} pos={row['validation_positive_energy']:.4f} "
                  f"neg={row['validation_negative_energy']:.4f} hinge={row['validation_hinge']:.4f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model": model.state_dict(), "config": config.to_dict(),
                "optimizer": "Adam", "training_steps": config.epochs * len(loader)}, output_dir / "checkpoint.pt")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(history)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["epoch"] for row in history], [row["validation_positive_energy"] for row in history], label="positive")
    axis.plot([row["epoch"] for row in history], [row["validation_negative_energy"] for row in history], label="negative")
    axis.set(title="Contrastive transition energies", xlabel="epoch", ylabel="energy")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(output_dir / "energy_curve.png", dpi=170); plt.close(figure)
    summary = {**config.to_dict(), "optimizer": "Adam", "training_steps": config.epochs * len(loader),
               "parameter_count": sum(p.numel() for p in model.parameters()),
               **{key: value for key, value in history[-1].items() if key != "epoch"}}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=CSWMConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs"); args = parser.parse_args()
    _, summary = train(CSWMConfig(epochs=args.epochs), args.output_dir); print(summary)
