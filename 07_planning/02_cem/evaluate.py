from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from cem import CEMPlanner
from config import CEMConfig


ROOT = Path(__file__).resolve().parent
PLANNING_ROOT = ROOT.parent
if str(PLANNING_ROOT) not in sys.path:
    sys.path.append(str(PLANNING_ROOT))
from planning_core import PointWorldModel  # noqa: E402


def evaluate(output_dir: Path = ROOT / "outputs") -> dict[str, object]:
    config = CEMConfig()
    initial = torch.tensor([-0.9, -0.8, 0.8, 0.7])
    result = CEMPlanner(
        PointWorldModel(),
        config.horizon,
        config.candidates,
        config.elites,
        config.iterations,
        config.discount,
        config.momentum,
        config.seed,
    ).plan(initial)
    path = torch.cat((initial[None, :2], result.predicted_states[:, :2]), dim=0)
    final_distance = torch.linalg.vector_norm(path[-1] - initial[2:])
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(path[:, 0], path[:, 1], marker="o")
    axes[0].scatter(initial[2], initial[3], marker="*", s=180, label="goal")
    axes[0].set(xlabel="x", ylabel="y", title="CEM optimized trajectory")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(range(1, config.iterations + 1), result.iteration_best_scores, marker="o")
    axes[1].set(xlabel="CEM iteration", ylabel="best predicted return", title="Distribution refitting")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "cem_plan.png", dpi=170)
    plt.close(figure)
    initial_distance = torch.linalg.vector_norm(initial[:2] - initial[2:])
    metrics = {
        **config.to_dict(),
        "dataset_version": "exact-point-world-v1",
        "initial_distance": float(initial_distance),
        "final_predicted_distance": float(final_distance),
        "distance_reduction_fraction": float(1.0 - final_distance / initial_distance),
        "selected_predicted_score": float(result.score),
        "final_mean_action_std": float(result.std.mean()),
        "iteration_best_scores": result.iteration_best_scores.tolist(),
        "evaluation_entry_point": "python 07_planning/02_cem/evaluate.py",
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
