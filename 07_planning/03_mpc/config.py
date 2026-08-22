from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MPCConfig:
    model_version: str = "exact-point-world-v1"
    planning_horizon: int = 8
    candidates: int = 256
    elites: int = 32
    cem_iterations: int = 4
    discount: float = 0.97
    max_environment_steps: int = 20
    seed: int = 73

    def to_dict(self):
        return asdict(self)
