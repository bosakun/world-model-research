from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import SlotAttentionConfig
from dataset import TwoObjectImagesDataset
from model import SlotAttentionAutoencoder

ROOT = Path(__file__).resolve().parent


def build_model(config): return SlotAttentionAutoencoder(config.image_size, config.num_slots, config.slot_dim, config.iterations)


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, foreground_weight: float) -> torch.Tensor:
    weights = 1.0 + foreground_weight * target.mean(dim=1, keepdim=True)
    return (((prediction - target) ** 2) * weights).mean()


def train(config: SlotAttentionConfig, output_dir: Path):
    torch.manual_seed(config.seed)
    train_data = TwoObjectImagesDataset(config.train_samples, config.seed, config.image_size)
    validation = TwoObjectImagesDataset(config.validation_samples, config.seed + 10_000, config.image_size)
    loader = DataLoader(train_data, config.batch_size, shuffle=True); model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate); history = []
    for epoch in range(1, config.epochs + 1):
        model.train(); total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); output = model(batch["image"])
            loss = reconstruction_loss(output["reconstruction"], batch["image"], config.foreground_weight); loss.backward(); optimizer.step()
            total += float(loss.detach()) * batch["image"].shape[0]
        model.eval()
        torch.manual_seed(config.seed + epoch)
        with torch.no_grad(): val_loss = reconstruction_loss(model(validation.images, True)["reconstruction"], validation.images, config.foreground_weight)
        row = {"epoch": epoch, "train_reconstruction": total / len(train_data), "validation_reconstruction": float(val_loss)}; history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs: print(f"epoch={epoch:03d} val_reconstruction={float(val_loss):.6f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model": model.state_dict(), "config": config.to_dict(), "optimizer": "Adam",
                "training_steps": config.epochs * len(loader)}, output_dir / "checkpoint.pt")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(history)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([r["epoch"] for r in history], [r["train_reconstruction"] for r in history], label="train")
    axis.plot([r["epoch"] for r in history], [r["validation_reconstruction"] for r in history], label="validation")
    axis.set(title="Slot Attention reconstruction", xlabel="epoch", ylabel="MSE"); axis.legend(); axis.grid(alpha=.3)
    figure.tight_layout(); figure.savefig(output_dir / "loss_curve.png", dpi=170); plt.close(figure)
    summary = {**config.to_dict(), "optimizer": "Adam", "training_steps": config.epochs * len(loader),
               "parameter_count": sum(p.numel() for p in model.parameters()), **history[-1]}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=SlotAttentionConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs"); args = parser.parse_args()
    _, summary = train(SlotAttentionConfig(epochs=args.epochs), args.output_dir); print(summary)
