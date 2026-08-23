from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import RandomShootingConfig
from random_shooting import RandomShootingPlanner


ROOT = Path(__file__).resolve().parent
PLANNING_ROOT = ROOT.parent
if str(PLANNING_ROOT) not in sys.path:
    sys.path.append(str(PLANNING_ROOT))
from planning_core import PointWorldModel  # noqa: E402


def evaluate(output_dir: Path = ROOT / "outputs") -> dict[str, object]:
    config = RandomShootingConfig()
    model = PointWorldModel()
    initial = torch.tensor([-0.9, -0.8, 0.8, 0.7])
    result = RandomShootingPlanner(
        model, config.horizon, config.candidates, config.discount, config.seed
    ).plan(initial)
    distances = torch.linalg.vector_norm(result.predicted_states[:, :2] - initial[2:], dim=-1)
    path = torch.cat((initial[None, :2], result.predicted_states[:, :2]), dim=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(path[:, 0], path[:, 1], marker="o")
    axes[0].scatter(initial[2], initial[3], marker="*", s=180, label="goal")
    axes[0].set(xlabel="x", ylabel="y", title="Best random-shooting trajectory")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].hist(result.candidate_scores.numpy(), bins=50)
    axes[1].axvline(float(result.score), color="red", label="selected")
    axes[1].set(xlabel="predicted return", title="Candidate score distribution")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "random_shooting_plan.png", dpi=170)
    plt.close(figure)
    metrics = {
        **config.to_dict(),
        "dataset_version": "exact-point-world-v1",
        "initial_distance": float(torch.linalg.vector_norm(initial[:2] - initial[2:])),
        "final_predicted_distance": float(distances[-1]),
        "distance_reduction_fraction": float(1.0 - distances[-1] / torch.linalg.vector_norm(initial[:2] - initial[2:])),
        "selected_predicted_score": float(result.score),
        "mean_candidate_score": float(result.candidate_scores.mean()),
        "evaluation_entry_point": "python 07_planning/01_random_shooting/evaluate.py",
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
