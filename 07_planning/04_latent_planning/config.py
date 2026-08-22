from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LatentPlanningConfig:
    seed: int = 101
    dataset_version: str = "point-world-latent-planning-v1"
    observation_dim: int = 4
    action_dim: int = 2
    latent_dim: int = 16
    hidden_dim: int = 64
    action_scale: float = 0.2
    horizon: int = 10
    train_sequences: int = 512
    validation_sequences: int = 128
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-3
    discount: float = 0.97
    consistency_weight: float = 1.0
    reward_weight: float = 2.0
    value_weight: float = 1.0
    candidates: int = 512
    elites: int = 64
    cem_iterations: int = 6

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
