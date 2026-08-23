from __future__ import annotations
import math,torch
from torch import nn
import torch.nn.functional as F
class Encoder(nn.Module):
 def __init__(self,d=16):super().__init__();self.net=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.ELU(),nn.Conv2d(16,32,4,2,1),nn.ELU(),nn.Flatten(),nn.Linear(32*5*5,d),nn.Tanh())
 def forward(self,x):shape=x.shape[:-3];return self.net(x.reshape(-1,3,20,20)).reshape(*shape,-1)
class Decoder(nn.Module):
 def __init__(self,d=16):super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.ELU(),nn.Linear(256,1200),nn.Sigmoid())
 def forward(self,z):return self.net(z.reshape(-1,z.shape[-1])).reshape(*z.shape[:-1],3,20,20)
class NoMemory(nn.Module):
 def __init__(self,d=16,h=64):super().__init__();self.encoder=Encoder(d);self.decoder=Decoder(d);self.transition=nn.Sequential(nn.Linear(d+4,h),nn.ELU(),nn.Linear(h,d),nn.Tanh());self.goal=nn.Linear(d,2)
 def teacher(self,o,a):z=self.encoder(o);p=self.transition(torch.cat((z[:,:-1],a),-1));return {"latents":z,"pred":p,"images":self.decoder(p),"recon":self.decoder(z),"goal":self.goal(p),"extra":z.new_zeros(())}
 def rollout(self,o,a,context=2,ablate=False):
  z=self.encoder(o[:,context]);pred=[];goals=[]
  for t in range(context,a.shape[1]):z=self.transition(torch.cat((z,a[:,t]),-1));pred.append(self.decoder(z));goals.append(self.goal(z))
  return torch.stack(pred,1),torch.stack(goals,1)
class GRUMemory(NoMemory):
 def __init__(self,d=16,h=64):super().__init__(d,h);self.cell=nn.GRUCell(d+4,h);self.head=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Linear(h,d),nn.Tanh());self.memory_goal=nn.Linear(h,2);del self.transition;del self.goal
 def teacher(self,o,a):
  z=self.encoder(o);hidden=z.new_zeros(z.shape[0],self.cell.hidden_size);pred=[];memories=[]
  for t in range(a.shape[1]):hidden=self.cell(torch.cat((z[:,t],a[:,t]),-1),hidden);pred.append(self.head(hidden));memories.append(hidden)
  p=torch.stack(pred,1);return {"latents":z,"pred":p,"images":self.decoder(p),"recon":self.decoder(z),"goal":self.memory_goal(torch.stack(memories,1)),"extra":z.new_zeros(())}
 def rollout(self,o,a,context=2,ablate=False):
  zall=self.encoder(o[:,:context+1]);hidden=zall.new_zeros(zall.shape[0],self.cell.hidden_size)
  for t in range(context):hidden=self.cell(torch.cat((zall[:,t],a[:,t]),-1),hidden)
  z=zall[:,context];pred=[];goals=[]
  for t in range(context,a.shape[1]):
   if ablate:hidden=torch.zeros_like(hidden)
   hidden=self.cell(torch.cat((z,a[:,t]),-1),hidden);z=self.head(hidden);pred.append(self.decoder(z));goals.append(self.memory_goal(hidden))
  return torch.stack(pred,1),torch.stack(goals,1)
class TransformerMemory(NoMemory):
 def __init__(self,d=16,h=64):
  super().__init__(d,h);self.token=nn.Linear(d+4,h);self.pos=nn.Embedding(16,h);layer=nn.TransformerEncoderLayer(h,4,2*h,batch_first=True,dropout=0.,norm_first=True);self.transformer=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,d),nn.Tanh());self.memory_goal=nn.Linear(h,2);del self.transition;del self.goal
 def dynamics(self,z,a):
  x=self.token(torch.cat((z,a),-1))+self.pos(torch.arange(z.shape[1],device=z.device))[None];mask=torch.triu(torch.ones(z.shape[1],z.shape[1],dtype=torch.bool,device=z.device),1);context=self.transformer(x,mask=mask);return self.head(context),context
 def teacher(self,o,a):
  z=self.encoder(o);p,context=self.dynamics(z[:,:-1],a);return {"latents":z,"pred":p,"images":self.decoder(p),"recon":self.decoder(z),"goal":self.memory_goal(context),"extra":z.new_zeros(())}
 def rollout(self,o,a,context=2,ablate=False):
  zhist=list(self.encoder(o[:,:context+1]).unbind(1));ahist=list(a[:,:context].unbind(1));pred=[];goals=[]
  for t in range(context,a.shape[1]):
   ahist.append(a[:,t]);use_z=zhist[-1:] if ablate else zhist;use_a=ahist[-1:] if ablate else ahist;prediction,context_tokens=self.dynamics(torch.stack(use_z,1),torch.stack(use_a,1));z=prediction[:,-1];zhist.append(z);pred.append(self.decoder(z));goals.append(self.memory_goal(context_tokens[:,-1]))
  return torch.stack(pred,1),torch.stack(goals,1)
class RSSMMemory(nn.Module):
 def __init__(self,d=16,h=64):
  super().__init__();self.encoder=Encoder(64);self.decoder=Decoder(d);self.cell=nn.GRUCell(d+4,h);self.prior=nn.Linear(h,2*d);self.posterior=nn.Linear(h+64,2*d);self.to_decode=nn.Linear(h+d,d);self.goal=nn.Linear(h+d,2);self.d=d
 def gaussian(self,x):mean,raw=x.chunk(2,-1);return mean,F.softplus(raw)+.1
 def teacher(self,o,a):
  e=self.encoder(o);h=e.new_zeros(e.shape[0],self.cell.hidden_size);z=e.new_zeros(e.shape[0],self.d);features=[];priors=[];posts=[]
  for t in range(o.shape[1]):
   if t:h=self.cell(torch.cat((z,a[:,t-1]),-1),h)
   pm,ps=self.gaussian(self.prior(h));qm,qs=self.gaussian(self.posterior(torch.cat((h,e[:,t]),-1)));z=qm;features.append(torch.cat((h,z),-1));priors.append((pm,ps));posts.append((qm,qs))
  feature=torch.stack(features,1);decoded=self.to_decode(feature);pm=torch.stack([x[0] for x in priors],1);ps=torch.stack([x[1] for x in priors],1);qm=torch.stack([x[0] for x in posts],1);qs=torch.stack([x[1] for x in posts],1);kl=(torch.log(ps/qs)+(qs.square()+(qm-pm).square())/(2*ps.square())-.5).mean()
  return {"latents":decoded,"pred":decoded[:,1:],"images":self.decoder(decoded[:,1:]),"recon":self.decoder(decoded),"goal":self.goal(feature[:,1:]),"extra":kl}
 def rollout(self,o,a,context=2,ablate=False):
  e=self.encoder(o[:,:context+1]);h=e.new_zeros(e.shape[0],self.cell.hidden_size);z=e.new_zeros(e.shape[0],self.d)
  for t in range(context+1):
   if t:h=self.cell(torch.cat((z,a[:,t-1]),-1),h)
   z=self.gaussian(self.posterior(torch.cat((h,e[:,t]),-1)))[0]
  if ablate:h=torch.zeros_like(h);z=torch.zeros_like(z)
  images=[];goals=[]
  for t in range(context,a.shape[1]):
   if ablate:h=torch.zeros_like(h);z=torch.zeros_like(z)
   h=self.cell(torch.cat((z,a[:,t]),-1),h);z=self.gaussian(self.prior(h))[0];f=torch.cat((h,z),-1);images.append(self.decoder(self.to_decode(f)));goals.append(self.goal(f))
  return torch.stack(images,1),torch.stack(goals,1)
def build_models(d=16,h=64):return {"no_memory":NoMemory(d,h),"gru":GRUMemory(d,h),"rssm":RSSMMemory(d,h),"transformer":TransformerMemory(d,h)}
