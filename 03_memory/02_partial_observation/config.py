from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PartialObservationConfig:
    grid_size: int = 5
    cell_size: int = 4
    view_radius: int = 1
    sequence_length: int = 6
    action_dim: int = 4
    latent_dim: int = 16
    hidden_dim: int = 64
    seed: int = 17

    @property
    def image_size(self) -> int:
        return self.grid_size * self.cell_size

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

