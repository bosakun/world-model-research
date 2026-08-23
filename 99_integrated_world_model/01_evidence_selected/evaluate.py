from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import IntegratedConfig
from model import State
from planner import DiscreteActionGuard, RiskAwarePlanner
from train import build_model

PARTIAL_ROOT = Path(__file__).resolve().parents[2] / "03_memory" / "02_partial_observation"
if str(PARTIAL_ROOT) not in sys.path:
    sys.path.append(str(PARTIAL_ROOT))

from partial_env import LEFT, PartialObservationGridWorld  # noqa: E402

ROOT = Path(__file__).resolve().parent


def initial_filter(model, observations: torch.Tensor, actions: torch.Tensor) -> State:
    with torch.no_grad():
        outputs = model.observe(observations, actions)
    return State(outputs["h"][:, -1], outputs["z"][:, -1])


def run_episode(model, config, goal, seed: int, risk: bool = True):
    environment = PartialObservationGridWorld()
    first_observation = environment.reset((2, 2), goal)
    observations = [torch.from_numpy(first_observation).float()]
    prefix_actions = []

    # After this prefix the goal is hidden, so the planner must use filtered memory.
    for _ in range(2):
        observation, _, _, _ = environment.step(LEFT)
        prefix_actions.append(LEFT)
        observations.append(torch.from_numpy(observation).float())

    observation_tensor = torch.stack(observations)[None]
    action_tensor = torch.nn.functional.one_hot(
        torch.tensor(prefix_actions)[None], 4
    ).float()
    state = initial_filter(model, observation_tensor, action_tensor)
    planner = RiskAwarePlanner(
        model,
        config.planning_horizon,
        config.planning_candidates,
        config.discount,
        config.uncertainty_penalty if risk else 0.0,
        seed,
    )
    guard = DiscreteActionGuard()
    states = [environment.true_state_array()]
    uncertainties = []
    success = False

    for _ in range(config.max_control_steps):
        plan = planner.plan(state)
        action = guard.filter(int(plan.actions[0]), enabled=True)
        observation, _, done, info = environment.step(action)
        uncertainties.append(plan.epistemic_std)
        states.append(info["true_state"])
        one_hot_action = torch.nn.functional.one_hot(torch.tensor([action]), 4).float()
        with torch.no_grad():
            state = model.posterior_step(
                state,
                one_hot_action,
                torch.from_numpy(observation).float()[None],
            )
        if done:
            success = True
            break

    return success, torch.from_numpy(np.stack(states)), uncertainties


def _success_rate(rows: list[dict]) -> float:
    return sum(row["success"] for row in rows) / len(rows)


def evaluate(output_directory: Path = ROOT / "outputs") -> dict[str, object]:
    config = IntegratedConfig()
    checkpoint = torch.load(output_directory / "checkpoint.pt", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    results = []

    for risk in (True, False):
        for seed in range(20):
            for goal in ((2, 3), (3, 2)):
                success, states, uncertainties = run_episode(
                    model, config, goal, config.seed + seed, risk
                )
                results.append(
                    {
                        "risk": risk,
                        "goal": goal,
                        "success": success,
                        "steps": len(states) - 1,
                        "mean_uncertainty": sum(uncertainties) / len(uncertainties),
                        "states": states,
                    }
                )

    risk_rows = [row for row in results if row["risk"]]
    mean_only_rows = [row for row in results if not row["risk"]]
    right_path = next(row["states"] for row in risk_rows if row["goal"] == (2, 3))
    down_path = next(row["states"] for row in risk_rows if row["goal"] == (3, 2))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, path, title in (
        (axes[0], right_path, "Remembered goal: right"),
        (axes[1], down_path, "Remembered goal: down"),
    ):
        axis.plot(path[:, 1], path[:, 0], marker="o")
        axis.scatter(path[-1, 3], path[-1, 2], marker="*", s=180, label="goal")
        axis.invert_yaxis()
        axis.set(title=title, xlabel="column", ylabel="row", xlim=(-0.25, 4.25), ylim=(4.25, -0.25))
        axis.grid(alpha=0.3)
        axis.legend()
    axes[2].bar(
        ["risk-aware", "mean-only"],
        [_success_rate(risk_rows), _success_rate(mean_only_rows)],
    )
    axes[2].set(title="Planning success", ylim=(0, 1.05))
    figure.tight_layout()
    figure.savefig(output_directory / "integrated_rollout.png", dpi=170)
    plt.close(figure)

    metrics = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "episodes_per_variant": len(risk_rows),
        "risk_aware_success_rate": _success_rate(risk_rows),
        "mean_only_success_rate": _success_rate(mean_only_rows),
        "risk_aware_mean_steps": sum(row["steps"] for row in risk_rows) / len(risk_rows),
        "risk_aware_mean_epistemic_std": sum(
            row["mean_uncertainty"] for row in risk_rows
        )
        / len(risk_rows),
        "external_actions_sent": False,
        "evaluation_entry_point": (
            "python 99_integrated_world_model/01_evidence_selected/evaluate.py"
        ),
    }
    metrics_path = output_directory / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
