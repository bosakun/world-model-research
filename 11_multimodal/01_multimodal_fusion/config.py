from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class FusionConfig:
    seed:int=251;dataset_version:str="multimodal-navigation-v1";model_dim:int=48;heads:int=4;layers:int=2;train_samples:int=1536;validation_samples:int=384;batch_size:int=64;epochs:int=50;learning_rate:float=8e-4;missing_probability:float=.2;image_weight:float=.5
    def to_dict(self):return asdict(self)
