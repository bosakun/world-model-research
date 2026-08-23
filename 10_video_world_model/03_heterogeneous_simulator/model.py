from __future__ import annotations
import torch
from torch import nn
class HeterogeneousSimulator(nn.Module):
    def __init__(self,condition_dim=16,latent_dim=32):
        super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.ReLU(),nn.Conv2d(16,32,4,2,1),nn.ReLU(),nn.Flatten(),nn.Linear(32*4*4,latent_dim),nn.Tanh())
        self.motor=nn.Linear(2,condition_dim);self.language=nn.Embedding(5,condition_dim);self.goal=nn.Linear(2,condition_dim);self.source=nn.Embedding(3,condition_dim)
        self.transition=nn.Sequential(nn.Linear(latent_dim+condition_dim,64),nn.SiLU(),nn.Linear(64,latent_dim));self.decoder=nn.Sequential(nn.Linear(latent_dim,32*4*4),nn.ReLU(),nn.Unflatten(1,(32,4,4)),nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,3,4,2,1))
    def condition(self,kind,motor,language,goal):
        candidates=torch.stack((self.motor(motor),self.language(language),self.goal(goal)),dim=1);selected=candidates[torch.arange(kind.shape[0],device=kind.device),kind];return selected+self.source(kind)
    def forward(self,current,kind,motor,language,goal):
        latent=self.encoder(current);condition=self.condition(kind,motor,language,goal);next_latent=torch.tanh(latent+self.transition(torch.cat((latent,condition),dim=-1)));return {"latent":latent,"condition":condition,"next_latent":next_latent,"prediction":torch.sigmoid(self.decoder(next_latent))}
