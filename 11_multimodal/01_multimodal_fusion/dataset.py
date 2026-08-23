from __future__ import annotations
import torch
from torch.utils.data import Dataset
DELTAS=torch.tensor([[0.,0.],[0.,-.18],[0.,.18],[-.18,0.],[.18,0.]])
def render(position,size=16):
    pixel=((position+1.)*.5*(size-1)).round().long().clamp(1,size-2);image=torch.zeros(position.shape[0],3,size,size)
    for i,(x,y) in enumerate(pixel.tolist()):image[i,0,y-1:y+2,x-1:x+2]=1.;image[i,1,y,x-1:x+2]=.5
    return image
class MultimodalNavigationDataset(Dataset):
    def __init__(self,samples:int,seed:int,missing_probability:float=.2):
        g=torch.Generator().manual_seed(seed);position=torch.empty(samples,2).uniform_(-.8,.8,generator=g);language=torch.randint(0,5,(samples,),generator=g);next_position=(position+DELTAS[language]).clamp(-1,1)
        touch=torch.stack((position[:,0]<-.72,position[:,0]>.72,position[:,1]<-.72,position[:,1]>.72),dim=-1).float();mask=torch.rand(samples,4,generator=g)>missing_probability;mask[:,1]=True
        self.vision=render(position);self.proprio=position;self.language=language;self.touch=touch;self.mask=mask;self.next_position=next_position;self.next_vision=render(next_position)
    def __len__(self):return self.vision.shape[0]
    def __getitem__(self,i):return {"vision":self.vision[i],"proprio":self.proprio[i],"language":self.language[i],"touch":self.touch[i],"mask":self.mask[i],"next_position":self.next_position[i],"next_vision":self.next_vision[i]}
