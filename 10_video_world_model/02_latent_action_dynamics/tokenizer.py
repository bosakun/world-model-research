from __future__ import annotations
import torch
from torch import nn
class FrozenVQTokenizer(nn.Module):
    """Checkpoint-compatible architecture from Phase 10/01."""
    def __init__(self,codes:int=32,dim:int=16):
        super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.ReLU(),nn.Conv2d(32,dim,4,2,1));self.codebook=nn.Embedding(codes,dim);self.decoder=nn.Sequential(nn.ConvTranspose2d(dim,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,3,4,2,1))
    def load_vq_state(self,state):
        translated={k.replace("quantizer.codebook.","codebook."):v for k,v in state.items() if not k.startswith("quantizer.") or k.startswith("quantizer.codebook.")};self.load_state_dict(translated);self.eval()
        for p in self.parameters():p.requires_grad_(False)
        return self
    def tokens(self,video):
        b,t,c,h,w=video.shape;continuous=self.encoder(video.reshape(b*t,c,h,w));vectors=continuous.permute(0,2,3,1).reshape(-1,continuous.shape[1]);distance=vectors.square().sum(1,keepdim=True)-2*vectors@self.codebook.weight.T+self.codebook.weight.square().sum(1);return distance.argmin(1).view(b,t,4,4)
    def decode_tokens(self,tokens):
        embedding=self.codebook(tokens.flatten()).view(tokens.shape[0],4,4,-1).permute(0,3,1,2);return torch.sigmoid(self.decoder(embedding))
