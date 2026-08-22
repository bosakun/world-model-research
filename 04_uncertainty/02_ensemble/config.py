from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnsembleConfig:
    dataset_version: str = "heteroscedastic-point-v1"
    state_dim: int = 2
    action_dim: int = 4
    hidden_dim: int = 64
    ensemble_size: int = 5
    train_transitions: int = 1024
    val_transitions: int = 256
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-3
    seed: int = 41
    rollout_horizon: int = 12
    rollout_particles: int = 128

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
