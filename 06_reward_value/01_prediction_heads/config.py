from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PredictionHeadsConfig:
    dataset_version: str = "goal-navigation-v1"
    state_dim: int = 4
    action_dim: int = 4
    hidden_dim: int = 64
    horizon: int = 20
    discount: float = 0.95
    train_sequences: int = 512
    val_sequences: int = 128
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    reward_weight: float = 1.0
    value_weight: float = 1.0
    continuation_weight: float = 1.0
    seed: int = 59

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
