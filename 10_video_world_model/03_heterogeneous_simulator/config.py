from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class HeterogeneousConfig:
    seed:int=239;dataset_version:str="heterogeneous-square-controls-v1";image_size:int=16;condition_dim:int=16;latent_dim:int=32
    train_samples:int=1536;validation_samples:int=384;batch_size:int=64;epochs:int=50;learning_rate:float=8e-4;foreground_weight:float=6.
    def to_dict(self):return asdict(self)
