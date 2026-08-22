from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SlotAttentionConfig:
    seed: int = 163
    dataset_version: str = "colored-two-object-images-v2"
    image_size: int = 16
    num_slots: int = 3
    slot_dim: int = 32
    iterations: int = 3
    train_samples: int = 768
    validation_samples: int = 192
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 4e-4
    foreground_weight: float = 8.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
