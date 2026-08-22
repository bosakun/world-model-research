from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class OccupancyConfig:
    seed:int=191; dataset_version:str="moving-voxel-sphere-v1"; grid_size:int=8; action_dim:int=3; latent_dim:int=32
    horizon:int=6; train_sequences:int=512; validation_sequences:int=128; batch_size:int=32; epochs:int=80
    learning_rate:float=8e-4; positive_weight:float=12.0; bce_weight:float=1.0; dice_weight:float=1.0
    consistency_weight:float=1.0; center_weight:float=.25
    def to_dict(self): return asdict(self)
