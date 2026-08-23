from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class IntegratedConfig:
    seed: int = 331
    dataset_version: str = "integrated-partial-navigation-v1"
    sequence_length: int = 12
    train_sequences: int = 768
    validation_sequences: int = 192
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 8e-4

    embedding_dim: int = 64
    deterministic_dim: int = 64
    stochastic_dim: int = 16
    ensemble_size: int = 3

    free_nats: float = 0.5
    kl_weight: float = 0.05
    reconstruction_weight: float = 3.0
    reward_weight: float = 2.0
    value_weight: float = 1.0
    continuation_weight: float = 0.5
    goal_weight: float = 0.5
    overshooting_weight: float = 0.1
    overshooting_distance: int = 3

    discount: float = 0.97
    planning_horizon: int = 6
    planning_candidates: int = 512
    uncertainty_penalty: float = 0.5
    max_control_steps: int = 12

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
