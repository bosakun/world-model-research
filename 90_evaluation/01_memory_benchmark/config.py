from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass
class BenchmarkConfig:
 dataset_version:str="partial-observation-memory-benchmark-v1";seeds:tuple[int,...]=(301,302,303);train_sequences:int=128;test_sequences:int=64;sequence_length:int=12;context_steps:int=2;latent_dim:int=16;hidden_dim:int=64;batch_size:int=32;epochs:int=35;learning_rate:float=1e-3;goal_weight:float=1.;prediction_weight:float=5.;reconstruction_weight:float=1.;latent_weight:float=1.;kl_weight:float=.05
 def to_dict(self):return asdict(self)
