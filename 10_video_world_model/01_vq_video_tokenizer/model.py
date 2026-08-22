from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
class VectorQuantizer(nn.Module):
    def __init__(self,codes:int=32,dim:int=16,beta:float=.25):super().__init__();self.codebook=nn.Embedding(codes,dim);nn.init.uniform_(self.codebook.weight,-1/codes,1/codes);self.beta=beta
    def forward(self,continuous:torch.Tensor):
        vectors=continuous.permute(0,2,3,1).reshape(-1,continuous.shape[1]);distance=(vectors.square().sum(1,keepdim=True)-2*vectors@self.codebook.weight.T+self.codebook.weight.square().sum(1))
        indices=distance.argmin(1);quantized=self.codebook(indices).view(continuous.shape[0],continuous.shape[2],continuous.shape[3],-1).permute(0,3,1,2)
        codebook_loss=F.mse_loss(quantized,continuous.detach());commitment=F.mse_loss(continuous,quantized.detach());straight=continuous+(quantized-continuous).detach()
        return {"quantized":straight,"indices":indices.view(continuous.shape[0],continuous.shape[2],continuous.shape[3]),"codebook_loss":codebook_loss,"commitment":commitment,"vq_loss":codebook_loss+self.beta*commitment}
class VQFrameTokenizer(nn.Module):
    def __init__(self,codes:int=32,dim:int=16,beta:float=.25):
        super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.ReLU(),nn.Conv2d(32,dim,4,2,1));self.quantizer=VectorQuantizer(codes,dim,beta)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(dim,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,3,4,2,1))
    def forward(self,frames):
        q=self.quantizer(self.encoder(frames));return {**q,"reconstruction":torch.sigmoid(self.decoder(q["quantized"]))}
    def tokenize_video(self,video):
        b,t,c,h,w=video.shape;return self.forward(video.reshape(b*t,c,h,w))["indices"].reshape(b,t,4,4)
