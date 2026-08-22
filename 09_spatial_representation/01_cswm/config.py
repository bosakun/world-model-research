from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CSWMConfig:
    seed: int = 151
    dataset_version: str = "two-object-relational-v1"
    image_size: int = 16
    num_objects: int = 2
    action_dim: int = 2
    slot_dim: int = 8
    hidden_dim: int = 64
    train_samples: int = 768
    validation_samples: int = 192
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    margin: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
