from __future__ import annotations
import torch
from torch.utils.data import Dataset
ACTION_DELTAS=torch.tensor([[0,0],[0,-1],[0,1],[-1,0],[1,0]])
def render_squares(positions:torch.Tensor,size:int=16)->torch.Tensor:
    batch=positions.shape[0];image=torch.zeros(batch,3,size,size)
    for i,(x,y) in enumerate(positions.tolist()):image[i,0,max(0,y-1):min(size,y+2),max(0,x-1):min(size,x+2)]=1.;image[i,1,max(0,y):min(size,y+1),max(0,x-1):min(size,x+2)]=.5
    return image
class MovingSquareVideoDataset(Dataset):
    def __init__(self,videos:int,frames:int,seed:int,image_size:int=16):
        g=torch.Generator().manual_seed(seed);positions=torch.randint(2,image_size-2,(videos,2),generator=g);images=[render_squares(positions,image_size)];actions=[];states=[positions]
        for _ in range(frames-1):
            action=torch.randint(0,len(ACTION_DELTAS),(videos,),generator=g);positions=(positions+ACTION_DELTAS[action]).clamp(1,image_size-2);actions.append(action);states.append(positions);images.append(render_squares(positions,image_size))
        self.videos=torch.stack(images,dim=1);self.actions=torch.stack(actions,dim=1);self.positions=torch.stack(states,dim=1)
    def __len__(self):return self.videos.shape[0]
    def __getitem__(self,index):return {"video":self.videos[index],"actions":self.actions[index],"positions":self.positions[index]}
