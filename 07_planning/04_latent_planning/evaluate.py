from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import LatentPlanningConfig
from planner import LatentCEMPlanner
from train import build_model
from utils import save_json


ROOT = Path(__file__).resolve().parent
PLANNING_ROOT = ROOT.parent
if str(PLANNING_ROOT) not in sys.path:
    sys.path.append(str(PLANNING_ROOT))
from planning_core import PointWorldModel  # noqa: E402


def evaluate(output_dir: Path = ROOT / "outputs") -> dict[str, object]:
    config = LatentPlanningConfig()
    checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    planner = LatentCEMPlanner(
        model, config.horizon, config.candidates, config.elites, config.cem_iterations, config.discount, config.seed + 1
    )
    initial = torch.tensor([-0.9, -0.8, 0.8, 0.7])
    plan = planner.plan(initial)
    exact_model = PointWorldModel(action_scale=config.action_scale)
    exact = exact_model.evaluate_action_sequences(initial, plan.actions.unsqueeze(0), config.discount)
    true_states = torch.cat((initial[None], exact["states"][0]), dim=0)
    distances = torch.linalg.vector_norm(true_states[:, :2] - true_states[:, 2:], dim=-1)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(true_states[:, 0], true_states[:, 1], marker="o", label="exact executed states")
    axes[0].scatter(initial[2], initial[3], marker="*", s=180, label="goal")
    axes[0].set(title="Decoder-free latent CEM plan", xlabel="x", ylabel="y")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(distances, marker="o")
    axes[1].set(title="Exact-world distance after latent planning", xlabel="step", ylabel="goal distance")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "latent_plan.png", dpi=170)
    plt.close(figure)
    metrics = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "planning_horizon": config.horizon,
        "candidates": config.candidates,
        "elites": config.elites,
        "cem_iterations": config.cem_iterations,
        "learned_latent_score": plan.score,
        "exact_world_score": float(exact["scores"][0]),
        "initial_distance": float(distances[0]),
        "final_distance": float(distances[-1]),
        "distance_reduction_fraction": float(1.0 - distances[-1] / distances[0]),
        "uses_observation_decoder": False,
        "evaluation_entry_point": "python 07_planning/04_latent_planning/evaluate.py",
    }
    save_json(output_dir / "evaluation_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
