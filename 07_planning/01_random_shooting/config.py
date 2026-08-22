from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RandomShootingConfig:
    model_version: str = "exact-point-world-v1"
    horizon: int = 10
    candidates: int = 4096
    discount: float = 0.97
    seed: int = 67

    def to_dict(self):
        return asdict(self)
