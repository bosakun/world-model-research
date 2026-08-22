from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import PredictionHeadsConfig
from navigation_dataset import GoalNavigationSequenceDataset
from train import build_model
from utils import parameter_count, save_json, seed_everything


ROOT = Path(__file__).resolve().parent


def _masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum()


def evaluate(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PredictionHeadsConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = GoalNavigationSequenceDataset(
        config.val_sequences, config.horizon, config.discount, config.seed + 20_000
    )
    with torch.no_grad():
        outputs = model(dataset.states[:, :-1], dataset.actions)
        mask = dataset.valid
        reward_rmse = _masked((outputs["reward"] - dataset.rewards).square(), mask).sqrt().item()
        value_rmse = _masked((outputs["value"] - dataset.value_targets).square(), mask).sqrt().item()
        continuation_probability = outputs["continuation_logit"].sigmoid()
        continuation_accuracy = _masked(
            ((continuation_probability >= 0.5) == dataset.continuations.bool()).float(), mask
        ).item()
        continuation_brier = _masked(
            (continuation_probability - dataset.continuations).square(), mask
        ).item()
        terminal_mask = (dataset.continuations == 0) * mask
        nonterminal_mask = (dataset.continuations == 1) * mask
        terminal_probability = _masked(continuation_probability, terminal_mask).item()
        nonterminal_probability = _masked(continuation_probability, nonterminal_mask).item()

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_steps = int(dataset.valid[0].sum().item())
    time = torch.arange(valid_steps).numpy()
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(time, dataset.rewards[0, :valid_steps], marker="o", label="target")
    axes[0].plot(time, outputs["reward"][0, :valid_steps], marker="x", label="prediction")
    axes[0].set(ylabel="reward")
    axes[1].plot(time, dataset.value_targets[0, :valid_steps], marker="o", label="target")
    axes[1].plot(time, outputs["value"][0, :valid_steps], marker="x", label="prediction")
    axes[1].set(ylabel="discounted value")
    axes[2].plot(time, dataset.continuations[0, :valid_steps], marker="o", label="target")
    axes[2].plot(time, continuation_probability[0, :valid_steps], marker="x", label="probability")
    axes[2].set(xlabel="transition", ylabel="continuation")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    axes[0].set_title("Reward, value, and continuation predictions")
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_sequence.png", dpi=170)
    plt.close(figure)

    metrics = {
        "dataset_version": config.dataset_version,
        "evaluation_sequences": len(dataset),
        "valid_transitions": int(mask.sum()),
        "terminal_transitions": int(terminal_mask.sum()),
        "reward_rmse": reward_rmse,
        "value_rmse": value_rmse,
        "continuation_accuracy": continuation_accuracy,
        "continuation_brier": continuation_brier,
        "mean_continuation_probability_on_terminal": terminal_probability,
        "mean_continuation_probability_on_nonterminal": nonterminal_probability,
        "discount": config.discount,
        "parameter_count": parameter_count(model),
        "evaluation_entry_point": "python 06_reward_value/01_prediction_heads/evaluate.py",
    }
    save_json(output_dir / "evaluation_metrics.json", metrics)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    evaluate(arguments.checkpoint, arguments.output_dir)
