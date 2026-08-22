from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RSSMConfig:
    dataset_version: str = "partial-observation-v1"
    sequence_length: int = 6
    train_sequences: int = 128
    val_sequences: int = 32
    image_channels: int = 3
    image_size: int = 20
    action_dim: int = 4
    observation_embedding_dim: int = 64
    deterministic_dim: int = 64
    stochastic_dim: int = 16
    hidden_mlp_dim: int = 64
    min_std: float = 0.1
    kl_weight: float = 1e-3
    free_nats: float = 1.0
    goal_classification_weight: float = 0.1
    green_channel_weight: float = 20.0
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 3e-3
    seed: int = 23

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
