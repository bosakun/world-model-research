from __future__ import annotations
import torch
from torch.utils.data import Dataset

def step_objects(positions:torch.Tensor, velocities:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    difference=positions[:,0]-positions[:,1]; distance=torch.linalg.vector_norm(difference,dim=-1,keepdim=True).clamp_min(1e-4)
    impulse=0.018*(distance<0.45)*difference/distance
    velocities=velocities+torch.stack((impulse,-impulse),dim=1)
    proposed=positions+velocities
    hit=proposed.abs()>0.9; velocities=torch.where(hit,-velocities,velocities); positions=(positions+velocities).clamp(-0.9,0.9)
    return positions,velocities

class RelationalSlotSequenceDataset(Dataset):
    def __init__(self,sequences:int,length:int,seed:int):
        generator=torch.Generator().manual_seed(seed)
        positions=torch.empty(sequences,2,2).uniform_(-.7,.7,generator=generator)
        velocities=torch.empty(sequences,2,2).uniform_(-.09,.09,generator=generator)
        frames=[positions]
        for _ in range(length-1): positions,velocities=step_objects(positions,velocities); frames.append(positions)
        self.slots=torch.stack(frames,dim=1)
    def __len__(self): return self.slots.shape[0]
    def __getitem__(self,index): return {"slots":self.slots[index]}
