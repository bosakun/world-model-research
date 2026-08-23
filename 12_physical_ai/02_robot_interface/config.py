from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class RobotConfig:
 seed:int=283;dataset_version:str="safe-mobile-demonstrations-v1";observation_dim:int=6;action_dim:int=2;hidden_dim:int=64;demonstrations:int=256;horizon:int=20;batch_size:int=64;epochs:int=50;learning_rate:float=1e-3;max_speed:float=.2;workspace_limit:float=1.;success_radius:float=.08
 def to_dict(self):return asdict(self)
