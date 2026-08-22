from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class VQVideoConfig:
    seed:int=211;dataset_version:str="moving-square-video-v1";image_size:int=16;frames:int=6;channels:int=3
    embedding_dim:int=16;codebook_size:int=32;train_videos:int=512;validation_videos:int=128;batch_size:int=64
    epochs:int=50;learning_rate:float=1e-3;commitment_weight:float=.25
    def to_dict(self):return asdict(self)
