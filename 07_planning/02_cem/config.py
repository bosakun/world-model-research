from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CEMConfig:
    model_version: str = "exact-point-world-v1"
    horizon: int = 10
    candidates: int = 512
    elites: int = 64
    iterations: int = 5
    discount: float = 0.97
    momentum: float = 0.1
    seed: int = 71

    def to_dict(self):
        return asdict(self)
