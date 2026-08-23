from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
class LatentActionModel(nn.Module):
    def __init__(self,codes=32,token_dim=32,latent_actions=5,model_dim=64,heads=4,layers=2,spatial_tokens=16):
        super().__init__();self.latent_actions=latent_actions;self.embedding=nn.Embedding(codes,token_dim)
        self.action_inference=nn.Sequential(nn.Linear(2*spatial_tokens*token_dim,128),nn.ReLU(),nn.Linear(128,latent_actions))
        self.token_projection=nn.Linear(token_dim,model_dim);self.action_projection=nn.Linear(latent_actions,model_dim,bias=False);self.position=nn.Embedding(spatial_tokens,model_dim)
        layer=nn.TransformerEncoderLayer(model_dim,heads,4*model_dim,batch_first=True,dropout=0.,activation="gelu",norm_first=True)
        self.dynamics=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False);self.output=nn.Sequential(nn.LayerNorm(model_dim),nn.Linear(model_dim,codes))
    def infer_action(self,current,next_tokens,temperature=.5,hard=True):
        pair=torch.cat((self.embedding(current).flatten(1),self.embedding(next_tokens).flatten(1)),dim=-1);logits=self.action_inference(pair);one_hot=F.gumbel_softmax(logits,tau=temperature,hard=hard,dim=-1);return logits,one_hot
    def predict(self,current,action_one_hot):
        flat=current.flatten(1);tokens=self.token_projection(self.embedding(flat));tokens=tokens+self.position(torch.arange(flat.shape[1],device=flat.device))[None]+self.action_projection(action_one_hot)[:,None]
        return self.output(self.dynamics(tokens)).view(current.shape[0],current.shape[1],current.shape[2],-1)
    def forward(self,current,next_tokens,temperature=.5):
        action_logits,action=self.infer_action(current,next_tokens,temperature,True);return {"action_logits":action_logits,"action":action,"token_logits":self.predict(current,action)}
    def rollout(self,initial,latent_actions):
        current=initial;frames=[]
        for t in range(latent_actions.shape[1]):current=self.predict(current,latent_actions[:,t]).argmax(dim=-1);frames.append(current)
        return torch.stack(frames,dim=1)
