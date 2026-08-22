from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from behavior import Critic, GaussianActor
from config import ImaginationConfig
from imagination import imagine, lambda_returns
from world_model import FrozenLatentWorldModel


ROOT = Path(__file__).resolve().parent
DEFAULT_WORLD_CHECKPOINT = ROOT.parents[1] / "07_planning" / "04_latent_planning" / "outputs" / "checkpoint.pt"


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)


def parameter_count(*modules: torch.nn.Module) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def sample_start_observations(batch_size: int) -> torch.Tensor:
    agents = torch.empty(batch_size, 2).uniform_(-0.9, 0.9)
    goals = torch.empty(batch_size, 2).uniform_(-0.9, 0.9)
    return torch.cat((agents, goals), dim=-1)


def load_world_model(config: ImaginationConfig, checkpoint_path: Path) -> FrozenLatentWorldModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run 07_planning/04_latent_planning/train.py first."
        )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = FrozenLatentWorldModel(config.observation_dim, config.action_dim, config.latent_dim, config.world_hidden_dim)
    model.load_state_dict(checkpoint["model"])
    return model.freeze()


def update_target(target: Critic, critic: Critic, ema: float) -> None:
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), critic.parameters()):
            target_parameter.mul_(ema).add_(parameter, alpha=1.0 - ema)


def train(
    config: ImaginationConfig,
    output_dir: Path,
    world_checkpoint: Path = DEFAULT_WORLD_CHECKPOINT,
) -> tuple[FrozenLatentWorldModel, GaussianActor, Critic, dict[str, object]]:
    seed_everything(config.seed)
    world_model = load_world_model(config, world_checkpoint)
    actor = GaussianActor(config.latent_dim, config.action_dim, config.behavior_hidden_dim)
    critic = Critic(config.latent_dim, config.behavior_hidden_dim)
    target_critic = copy.deepcopy(critic)
    for parameter in target_critic.parameters():
        parameter.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
    history = []
    for update in range(1, config.updates + 1):
        observations = sample_start_observations(config.batch_size)
        initial_latent = world_model.encode(observations)

        actor_optimizer.zero_grad(set_to_none=True)
        imagined = imagine(world_model, actor, initial_latent, config.imagination_horizon)
        next_values = target_critic(imagined["latents"][:, 1:])
        returns = lambda_returns(imagined["rewards"], next_values, config.discount, config.lambda_)
        actor_loss = -returns.mean() - config.entropy_weight * imagined["entropies"].mean()
        actor_loss.backward()
        actor_optimizer.step()

        with torch.no_grad():
            critic_imagination = imagine(world_model, actor, initial_latent, config.imagination_horizon)
            targets = lambda_returns(
                critic_imagination["rewards"],
                target_critic(critic_imagination["latents"][:, 1:]),
                config.discount,
                config.lambda_,
            )
        critic_optimizer.zero_grad(set_to_none=True)
        predictions = critic(critic_imagination["latents"][:, :-1].detach())
        critic_loss = F.mse_loss(predictions, targets.detach())
        critic_loss.backward()
        critic_optimizer.step()
        update_target(target_critic, critic, config.target_ema)

        row = {
            "update": update,
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "imagined_reward_mean": float(imagined["rewards"].mean().detach()),
            "imagined_return_mean": float(returns.mean().detach()),
            "action_entropy": float(imagined["entropies"].mean().detach()),
        }
        history.append(row)
        if update == 1 or update % 100 == 0 or update == config.updates:
            print(f"update={update:04d} actor={row['actor_loss']:.4f} critic={row['critic_loss']:.4f} "
                  f"reward={row['imagined_reward_mean']:.4f} return={row['imagined_return_mean']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "world_model": world_model.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "config": config.to_dict(),
            "training_updates": config.updates,
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    updates = [row["update"] for row in history]
    axes[0].plot(updates, [row["imagined_return_mean"] for row in history])
    axes[0].set(title="Imagined lambda return", xlabel="behavior update", ylabel="return")
    axes[1].plot(updates, [row["critic_loss"] for row in history], label="critic loss")
    axes[1].plot(updates, [row["action_entropy"] for row in history], label="actor entropy")
    axes[1].set(title="Behavior learning diagnostics", xlabel="behavior update")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "imagination_training.png", dpi=170)
    plt.close(figure)
    summary = {
        **config.to_dict(),
        "world_checkpoint": str(world_checkpoint.relative_to(ROOT.parents[1])),
        "actor_parameter_count": parameter_count(actor),
        "critic_parameter_count": parameter_count(critic),
        "world_parameter_count": parameter_count(world_model),
        "optimizer": "Adam",
        "checkpoint_format_version": 1,
        **{key: value for key, value in history[-1].items() if key != "update"},
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return world_model, actor, critic, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=ImaginationConfig.updates)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--world-checkpoint", type=Path, default=DEFAULT_WORLD_CHECKPOINT)
    arguments = parser.parse_args()
    _, _, _, metrics = train(
        ImaginationConfig(updates=arguments.updates), arguments.output_dir, arguments.world_checkpoint
    )
    print(metrics)
