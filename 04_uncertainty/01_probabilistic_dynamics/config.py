from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProbabilisticDynamicsConfig:
    dataset_version: str = "heteroscedastic-point-v1"
    state_dim: int = 2
    action_dim: int = 4
    hidden_dim: int = 64
    train_transitions: int = 1024
    val_transitions: int = 256
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    seed: int = 37
    rollout_horizon: int = 12

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
