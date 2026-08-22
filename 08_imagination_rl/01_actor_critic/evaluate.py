from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from behavior import Critic, GaussianActor
from config import ImaginationConfig
from world_model import FrozenLatentWorldModel


ROOT = Path(__file__).resolve().parent
PLANNING_ROOT = ROOT.parents[1] / "07_planning"
if str(PLANNING_ROOT) not in sys.path:
    sys.path.append(str(PLANNING_ROOT))
from planning_core import PointWorldEnvironment  # noqa: E402


def evaluate(output_dir: Path = ROOT / "outputs") -> dict[str, object]:
    config = ImaginationConfig()
    checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=False)
    world = FrozenLatentWorldModel(config.observation_dim, config.action_dim, config.latent_dim, config.world_hidden_dim)
    actor = GaussianActor(config.latent_dim, config.action_dim, config.behavior_hidden_dim)
    critic = Critic(config.latent_dim, config.behavior_hidden_dim)
    world.load_state_dict(checkpoint["world_model"])
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    world.freeze()
    actor.eval()
    critic.eval()
    environment = PointWorldEnvironment(max_steps=config.evaluation_steps)
    observation = environment.reset()
    states = [observation]
    actions, rewards, predicted_values = [], [], []
    for _ in range(config.evaluation_steps):
        with torch.no_grad():
            latent = world.encode(observation)
            action = actor.mode(latent)
            predicted_values.append(critic(latent))
        observation, reward, done = environment.step(action)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        if done:
            break
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards)
    distances = torch.linalg.vector_norm(states[:, :2] - states[:, 2:], dim=-1)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(states[:, 0], states[:, 1], marker="o", label="actor in exact world")
    axes[0].scatter(states[0, 2], states[0, 3], marker="*", s=180, label="goal")
    axes[0].set(title="Policy learned only from latent imagination", xlabel="x", ylabel="y")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(distances, marker="o", label="exact goal distance")
    axes[1].plot(range(len(predicted_values)), torch.stack(predicted_values), label="critic value")
    axes[1].set(title="Reality/critic diagnostic", xlabel="environment step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "actor_rollout.png", dpi=170)
    plt.close(figure)
    success = bool(distances[-1] <= environment.model.success_radius)
    metrics = {
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "success": success,
        "executed_steps": actions.shape[0],
        "initial_distance": float(distances[0]),
        "final_distance": float(distances[-1]),
        "distance_reduction_fraction": float(1.0 - distances[-1] / distances[0]),
        "exact_total_reward": float(rewards.sum()),
        "deterministic_actor_evaluation": True,
        "real_environment_used_for_behavior_training": False,
        "evaluation_entry_point": "python 08_imagination_rl/01_actor_critic/evaluate.py",
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
