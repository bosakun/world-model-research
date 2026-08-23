from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class JEPAConfig:
 seed:int=271;dataset_version:str="noisy-robot-transitions-v1";observation_dim:int=6;action_dim:int=2;latent_dim:int=32;hidden_dim:int=64;train_samples:int=2048;validation_samples:int=512;batch_size:int=128;epochs:int=60;learning_rate:float=8e-4;ema:float=.99;variance_weight:float=.1;covariance_weight:float=.01
 def to_dict(self):return asdict(self)
