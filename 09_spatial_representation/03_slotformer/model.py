from __future__ import annotations
import torch
from torch import nn

def frame_causal_mask(frames:int,slots:int,device:torch.device|None=None)->torch.Tensor:
    frame_index=torch.arange(frames,device=device).repeat_interleave(slots)
    return frame_index[None,:] > frame_index[:,None]

class SlotFormer(nn.Module):
    def __init__(self,num_slots:int=2,slot_dim:int=2,model_dim:int=64,heads:int=4,layers:int=2,max_frames:int=16):
        super().__init__(); self.num_slots=num_slots; self.max_frames=max_frames
        self.input_projection=nn.Linear(slot_dim,model_dim); self.time_embedding=nn.Embedding(max_frames,model_dim)
        self.slot_embedding=nn.Embedding(num_slots,model_dim)
        layer=nn.TransformerEncoderLayer(model_dim,heads,4*model_dim,batch_first=True,norm_first=True,dropout=0.0,activation="gelu")
        self.transformer=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False); self.output=nn.Sequential(nn.LayerNorm(model_dim),nn.Linear(model_dim,slot_dim))
    def forward(self,slots:torch.Tensor)->torch.Tensor:
        batch,frames,num_slots,_=slots.shape
        if frames>self.max_frames or num_slots!=self.num_slots: raise ValueError("invalid frame or slot count")
        token=self.input_projection(slots)
        token=token+self.time_embedding(torch.arange(frames,device=slots.device))[None,:,None,:]
        token=token+self.slot_embedding(torch.arange(num_slots,device=slots.device))[None,None,:,:]
        token=token.reshape(batch,frames*num_slots,-1)
        hidden=self.transformer(token,mask=frame_causal_mask(frames,num_slots,slots.device)).reshape(batch,frames,num_slots,-1)
        return slots+self.output(hidden)
    def rollout(self,context:torch.Tensor,steps:int)->torch.Tensor:
        history=context; predictions=[]
        for _ in range(steps):
            next_slots=self(history[:,-self.max_frames:])[:,-1]
            predictions.append(next_slots); history=torch.cat((history,next_slots[:,None]),dim=1)
        return torch.stack(predictions,dim=1)
