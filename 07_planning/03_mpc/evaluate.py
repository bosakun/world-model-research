from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import MPCConfig
from mpc import RecedingHorizonMPC


ROOT = Path(__file__).resolve().parent
PLANNING_ROOT = ROOT.parent
CEM_ROOT = PLANNING_ROOT / "02_cem"
for path in (PLANNING_ROOT, CEM_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from cem import CEMPlanner  # noqa: E402
from planning_core import PointWorldEnvironment, PointWorldModel  # noqa: E402


def evaluate(output_dir: Path = ROOT / "outputs") -> dict[str, object]:
    config = MPCConfig()
    planner = CEMPlanner(
        PointWorldModel(),
        config.planning_horizon,
        config.candidates,
        config.elites,
        config.cem_iterations,
        config.discount,
        seed=config.seed,
    )
    result = RecedingHorizonMPC(planner, config.max_environment_steps).run(
        PointWorldEnvironment(max_steps=config.max_environment_steps)
    )
    distances = torch.linalg.vector_norm(result.states[:, :2] - result.states[:, 2:], dim=-1)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(result.states[:, 0], result.states[:, 1], marker="o")
    axes[0].scatter(result.states[0, 2], result.states[0, 3], marker="*", s=180, label="goal")
    axes[0].set(xlabel="x", ylabel="y", title="Receding-horizon executed path")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(range(len(distances)), distances, marker="o")
    axes[1].set(xlabel="environment step / replanning call", ylabel="goal distance", title="MPC correction")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "mpc_rollout.png", dpi=170)
    plt.close(figure)
    metrics = {
        **config.to_dict(),
        "dataset_version": "exact-point-world-v1",
        "success": result.success,
        "executed_steps": result.actions.shape[0],
        "replanning_calls": result.actions.shape[0],
        "initial_distance": float(distances[0]),
        "final_distance": float(distances[-1]),
        "total_environment_reward": float(result.rewards.sum()),
        "evaluation_entry_point": "python 07_planning/03_mpc/evaluate.py",
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
