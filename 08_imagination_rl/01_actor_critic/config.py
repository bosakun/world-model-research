from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ImaginationConfig:
    seed: int = 131
    dataset_version: str = "point-world-imagination-starts-v1"
    observation_dim: int = 4
    action_dim: int = 2
    latent_dim: int = 16
    world_hidden_dim: int = 64
    behavior_hidden_dim: int = 64
    imagination_horizon: int = 12
    batch_size: int = 256
    updates: int = 500
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    discount: float = 0.97
    lambda_: float = 0.95
    entropy_weight: float = 1e-3
    target_ema: float = 0.98
    action_scale: float = 0.2
    evaluation_steps: int = 20

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
