from __future__ import annotations
import copy,torch
from torch import nn
class ActionJEPA(nn.Module):
 def __init__(self,obs=6,action=2,latent=32,hidden=64):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(obs,hidden),nn.SiLU(),nn.Linear(hidden,latent));self.predictor=nn.Sequential(nn.Linear(latent+action,hidden),nn.SiLU(),nn.Linear(hidden,latent));self.target_encoder=copy.deepcopy(self.encoder)
  for p in self.target_encoder.parameters():p.requires_grad_(False)
 def forward(self,o,a,no):return {"online":self.encoder(o),"predicted":self.predictor(torch.cat((self.encoder(o),a),-1)),"target":self.target_encoder(no).detach()}
 @torch.no_grad()
 def update_target(self,ema):
  for t,o in zip(self.target_encoder.parameters(),self.encoder.parameters()):t.mul_(ema).add_(o,alpha=1-ema)
