from __future__ import annotations
import torch
from torch import nn
class MultimodalFusionWorldModel(nn.Module):
    def __init__(self,d=48,heads=4,layers=2):
        super().__init__();self.vision=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.ReLU(),nn.Conv2d(16,24,4,2,1),nn.ReLU(),nn.Flatten(),nn.Linear(24*4*4,d));self.proprio=nn.Linear(2,d);self.language=nn.Embedding(5,d);self.touch=nn.Linear(4,d);self.types=nn.Embedding(4,d);self.missing=nn.Parameter(torch.randn(4,d)*.02)
        layer=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,dropout=0.,activation="gelu",norm_first=True);self.fusion=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False);self.position_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2));self.image_head=nn.Sequential(nn.Linear(d,24*4*4),nn.ReLU(),nn.Unflatten(1,(24,4,4)),nn.ConvTranspose2d(24,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,3,4,2,1))
    def forward(self,vision,proprio,language,touch,mask):
        encoded=torch.stack((self.vision(vision),self.proprio(proprio),self.language(language),self.touch(touch)),dim=1);encoded=encoded+self.types(torch.arange(4,device=vision.device))[None];encoded=torch.where(mask[...,None],encoded,self.missing[None]);fused=self.fusion(encoded).mean(dim=1);return {"tokens":encoded,"fused":fused,"next_position":torch.tanh(self.position_head(fused)),"next_vision":torch.sigmoid(self.image_head(fused))}
