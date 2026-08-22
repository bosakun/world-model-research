from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OvershootingConfig:
    dataset_version: str = "controlled-oscillator-v1"
    state_dim: int = 2
    action_dim: int = 4
    hidden_dim: int = 64
    sequence_length: int = 30
    overshooting_distance: int = 5
    overshooting_weight: float = 0.5
    train_sequences: int = 256
    val_sequences: int = 64
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    seed: int = 47

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
