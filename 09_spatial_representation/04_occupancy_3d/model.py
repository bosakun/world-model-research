from __future__ import annotations
import torch
from torch import nn
class OccupancyWorldModel(nn.Module):
    def __init__(self,action_dim:int=3,latent_dim:int=32):
        super().__init__(); self.encoder_conv=nn.Sequential(nn.Conv3d(1,8,3,stride=2,padding=1),nn.ReLU(),nn.Conv3d(8,16,3,stride=2,padding=1),nn.ReLU())
        self.encoder_linear=nn.Sequential(nn.Flatten(),nn.Linear(16*2*2*2,latent_dim),nn.Tanh())
        self.transition=nn.Sequential(nn.Linear(latent_dim+action_dim,64),nn.SiLU(),nn.Linear(64,latent_dim))
        self.decoder_linear=nn.Sequential(nn.Linear(latent_dim,16*2*2*2),nn.ReLU())
        self.decoder_conv=nn.Sequential(nn.ConvTranspose3d(16,8,4,stride=2,padding=1),nn.ReLU(),nn.ConvTranspose3d(8,1,4,stride=2,padding=1))
    def encode(self,occupancy): return self.encoder_linear(self.encoder_conv(occupancy))
    def next_latent(self,latent,action): return torch.tanh(latent+self.transition(torch.cat((latent,action),dim=-1)))
    def decode_logits(self,latent): return self.decoder_conv(self.decoder_linear(latent).view(-1,16,2,2,2))
    def forward(self,occupancy,action):
        latent=self.encode(occupancy); next_latent=self.next_latent(latent,action); return {"latent":latent,"next_latent":next_latent,"logits":self.decode_logits(next_latent)}
    def rollout(self,initial_occupancy,actions):
        latent=self.encode(initial_occupancy); logits=[]; latents=[]
        for t in range(actions.shape[1]): latent=self.next_latent(latent,actions[:,t]); latents.append(latent); logits.append(self.decode_logits(latent))
        return {"latents":torch.stack(latents,dim=1),"logits":torch.stack(logits,dim=1)}
