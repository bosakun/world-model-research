from __future__ import annotations
import torch
from torch import nn
class ImitationPolicy(nn.Module):
 def __init__(self,obs=6,action=2,hidden=64,max_speed=.2):super().__init__();self.max_speed=max_speed;self.network=nn.Sequential(nn.Linear(obs,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,action))
 def forward(self,observation):return self.max_speed*torch.tanh(self.network(observation))
