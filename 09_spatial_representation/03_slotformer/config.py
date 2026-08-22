from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass
class SlotFormerConfig:
    seed:int=179; dataset_version:str="relational-slot-sequences-v1"; num_slots:int=2; slot_dim:int=2
    model_dim:int=64; heads:int=4; layers:int=2; max_frames:int=16; sequence_length:int=12
    context_frames:int=4; rollout_frames:int=8; train_sequences:int=768; validation_sequences:int=192
    batch_size:int=64; epochs:int=60; learning_rate:float=5e-4
    def to_dict(self): return asdict(self)
