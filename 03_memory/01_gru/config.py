from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    grid_size: int = 5
    cell_size: int = 4
    sequence_length: int = 8
    train_sequences: int = 256
    val_sequences: int = 64
    latent_dim: int = 16
    hidden_dim: int = 64
    action_dim: int = 4
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 3e-3
    reconstruction_weight: float = 1.0
    dynamics_weight: float = 2.0
    position_weight: float = 0.2
    seed: int = 7

    @property
    def image_size(self) -> int:
        return self.grid_size * self.cell_size

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
