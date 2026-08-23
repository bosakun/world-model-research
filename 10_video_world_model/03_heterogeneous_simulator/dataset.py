from __future__ import annotations
import torch
from torch.utils.data import Dataset
DELTAS=torch.tensor([[0.,0.],[0.,-2.],[0.,2.],[-2.,0.],[2.,0.]])
def render(position,size=16):
    image=torch.zeros(position.shape[0],3,size,size)
    for i,(x,y) in enumerate(position.round().long().tolist()):image[i,0,max(0,y-1):min(size,y+2),max(0,x-1):min(size,x+2)]=1.;image[i,1,max(0,y):min(size,y+1),max(0,x-1):min(size,x+2)]=.5
    return image
class HeterogeneousControlDataset(Dataset):
    """type 0=motor vector, 1=language ID, 2=goal coordinate."""
    def __init__(self,samples:int,seed:int,size:int=16):
        g=torch.Generator().manual_seed(seed);position=torch.randint(2,size-2,(samples,2),generator=g).float();kind=torch.arange(samples)%3;command=torch.randint(0,5,(samples,),generator=g);motor=DELTAS[command]/2.;goal=(position+DELTAS[command]).clamp(1,size-2)
        delta=torch.where((kind==2)[:,None],(goal-position).sign()*2.,DELTAS[command]);next_position=(position+delta).clamp(1,size-2)
        self.current=render(position,size);self.next=render(next_position,size);self.kind=kind.long();self.motor=motor;self.language=command;self.goal=goal/(size-1)*2.-1.;self.position=position;self.next_position=next_position
    def __len__(self):return self.current.shape[0]
    def __getitem__(self,i):return {"current":self.current[i],"next":self.next[i],"kind":self.kind[i],"motor":self.motor[i],"language":self.language[i],"goal":self.goal[i]}
