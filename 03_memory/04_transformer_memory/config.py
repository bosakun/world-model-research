from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TransformerMemoryConfig:
    dataset_version: str = "partial-observation-v1"
    sequence_length: int = 6
    train_sequences: int = 128
    val_sequences: int = 32
    image_channels: int = 3
    image_size: int = 20
    action_dim: int = 4
    latent_dim: int = 16
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 128
    max_context: int = 16
    dropout: float = 0.0
    latent_prediction_weight: float = 0.5
    goal_classification_weight: float = 0.1
    green_channel_weight: float = 20.0
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 3e-3
    seed: int = 29

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
