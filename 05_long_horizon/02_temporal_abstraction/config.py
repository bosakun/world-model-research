from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TemporalAbstractionConfig:
    dataset_version: str = "controlled-oscillator-v1-macro5"
    state_dim: int = 2
    action_dim: int = 4
    action_embedding_dim: int = 32
    hidden_dim: int = 64
    sequence_length: int = 30
    chunk_size: int = 5
    train_sequences: int = 256
    val_sequences: int = 64
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    seed: int = 53

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
