from __future__ import annotations
import torch
from torch.utils.data import Dataset

def render_voxels(centers:torch.Tensor,grid_size:int=8)->torch.Tensor:
    coordinates=torch.arange(grid_size,dtype=torch.float32)
    zz,yy,xx=torch.meshgrid(coordinates,coordinates,coordinates,indexing="ij"); grid=torch.stack((xx,yy,zz),dim=-1)
    distance=((grid[None]-centers[:,None,None,None])**2).sum(dim=-1)
    return (distance<=1.5**2).float()[:,None]

class MovingOccupancyDataset(Dataset):
    def __init__(self,sequences:int,horizon:int,seed:int,grid_size:int=8):
        g=torch.Generator().manual_seed(seed); centers=torch.empty(sequences,3).uniform_(1.5,grid_size-2.5,generator=g)
        occupancies=[render_voxels(centers,grid_size)]; center_history=[centers]; actions=[]
        for _ in range(horizon):
            action=torch.empty(sequences,3).uniform_(-1,1,generator=g); centers=(centers+.75*torch.tanh(action)).clamp(1.,grid_size-2.)
            actions.append(action); occupancies.append(render_voxels(centers,grid_size)); center_history.append(centers)
        self.occupancies=torch.stack(occupancies,dim=1); self.actions=torch.stack(actions,dim=1); self.centers=torch.stack(center_history,dim=1)
    def __len__(self): return self.occupancies.shape[0]
    def __getitem__(self,index): return {"occupancies":self.occupancies[index],"actions":self.actions[index],"centers":self.centers[index]}
