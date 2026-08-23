from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class LatentActionConfig:
    seed:int=223;dataset_version:str="vq-moving-square-pairs-v2";frames:int=6;codebook_size:int=32;token_rows:int=4;token_cols:int=4
    token_dim:int=32;latent_actions:int=5;model_dim:int=64;heads:int=4;layers:int=2;train_videos:int=512;validation_videos:int=128
    batch_size:int=128;epochs:int=50;learning_rate:float=7e-4;gumbel_temperature:float=.5;confidence_weight:float=.03;balance_weight:float=.1;changed_token_weight:float=5.
    def to_dict(self):return asdict(self)
