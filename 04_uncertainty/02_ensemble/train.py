from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from config import EnsembleConfig
from ensemble_dataset import HeteroscedasticTransitionDataset, bootstrap_indices
from probabilistic_ensemble import PROBABILISTIC_EXPERIMENT, ProbabilisticEnsemble
from utils import parameter_count, save_json, seed_everything


if str(PROBABILISTIC_EXPERIMENT) not in sys.path:
    sys.path.append(str(PROBABILISTIC_EXPERIMENT))
from probabilistic_losses import probabilistic_dynamics_loss  # noqa: E402


ROOT = Path(__file__).resolve().parent


def build_model(config: EnsembleConfig) -> ProbabilisticEnsemble:
    return ProbabilisticEnsemble(
        config.ensemble_size, config.state_dim, config.action_dim, config.hidden_dim
    )


def train(config: EnsembleConfig, output_dir: Path) -> dict[str, object]:
    seed_everything(config.seed)
    training_data = HeteroscedasticTransitionDataset(config.train_transitions, config.seed)
    validation_data = HeteroscedasticTransitionDataset(
        config.val_transitions, config.seed + 10_000
    )
    indices = bootstrap_indices(len(training_data), config.ensemble_size, config.seed + 1)
    loaders = [
        DataLoader(
            Subset(training_data, member_indices.tolist()),
            config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + member_index + 100),
        )
        for member_index, member_indices in enumerate(indices)
    ]
    ensemble = build_model(config)
    optimizers = [
        torch.optim.Adam(member.parameters(), lr=config.learning_rate)
        for member in ensemble.members
    ]
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        ensemble.train()
        train_nlls = []
        for member, optimizer, loader in zip(ensemble.members, optimizers, loaders, strict=True):
            total_nll = 0.0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                losses = probabilistic_dynamics_loss(
                    member, member(batch["states"], batch["actions"]), batch["next_states"]
                )
                losses["total"].backward()
                optimizer.step()
                total_nll += float(losses["negative_log_likelihood"].detach()) * batch["states"].shape[0]
            train_nlls.append(total_nll / len(training_data))

        ensemble.eval()
        validation_nlls = []
        with torch.no_grad():
            for member in ensemble.members:
                losses = probabilistic_dynamics_loss(
                    member,
                    member(validation_data.states, validation_data.actions),
                    validation_data.next_states,
                )
                validation_nlls.append(float(losses["negative_log_likelihood"]))
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_member_mean_nll": sum(train_nlls) / len(train_nlls),
            "val_member_mean_nll": sum(validation_nlls) / len(validation_nlls),
        }
        row.update({f"train_member_{index}_nll": value for index, value in enumerate(train_nlls)})
        row.update({f"val_member_{index}_nll": value for index, value in enumerate(validation_nlls)})
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} train_nll={row['train_member_mean_nll']:.5f} "
                f"val_nll={row['val_member_mean_nll']:.5f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_member = config.epochs * len(loaders[0])
    torch.save(
        {
            "format_version": 1,
            "model": ensemble.state_dict(),
            "config": config.to_dict(),
            "optimizer": "independent Adam per member",
            "training_steps_per_member": steps_per_member,
            "bootstrap_seed": config.seed + 1,
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["train_member_mean_nll"] for row in history], label="train")
    axis.plot(epochs, [row["val_member_mean_nll"] for row in history], label="validation")
    axis.set(xlabel="epoch", ylabel="member-mean NLL", title="Bootstrap ensemble training")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=170)
    plt.close(figure)

    summary: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "bootstrap_seed": config.seed + 1,
        "ensemble_size": config.ensemble_size,
        "epochs": config.epochs,
        "training_steps_per_member": steps_per_member,
        "optimizer": "independent Adam per member",
        "learning_rate": config.learning_rate,
        "parameter_count": parameter_count(ensemble),
        "checkpoint_format_version": 1,
        "initial_train_member_mean_nll": history[0]["train_member_mean_nll"],
        "final_train_member_mean_nll": history[-1]["train_member_mean_nll"],
        "final_val_member_mean_nll": history[-1]["val_member_mean_nll"],
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EnsembleConfig.epochs)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    print(train(EnsembleConfig(epochs=arguments.epochs), arguments.output_dir))
