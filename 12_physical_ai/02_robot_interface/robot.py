from __future__ import annotations
from dataclasses import dataclass
import torch
@dataclass
class SafetyResult: action:torch.Tensor;clipped:bool;stopped:bool;reason:str
class SafetyEnvelope:
 def __init__(self,max_speed=.2,workspace_limit=1.):self.max_speed=max_speed;self.workspace_limit=workspace_limit
 def filter(self,observation,requested,enabled=True):
  if not enabled:return SafetyResult(torch.zeros_like(requested),False,True,"deadman_disabled")
  if observation[:2].abs().max()>self.workspace_limit:return SafetyResult(torch.zeros_like(requested),False,True,"workspace_violation")
  action=requested.clamp(-self.max_speed,self.max_speed);return SafetyResult(action,bool(not torch.equal(action,requested)),False,"clipped" if not torch.equal(action,requested) else "ok")
class SimulatedMobileRobot:
 def __init__(self,start=(-.8,-.8),goal=(.8,.8),max_steps=30,success_radius=.08):self.initial=torch.tensor((*start,0.,0.,*goal));self.max_steps=max_steps;self.success_radius=success_radius;self.reset()
 def reset(self):self.state=self.initial.clone();self.steps=0;return self.state.clone()
 def step(self,action):
  velocity=.5*self.state[2:4]+.5*action;position=(self.state[:2]+velocity).clamp(-1,1);self.state=torch.cat((position,velocity,self.state[4:]));self.steps+=1;distance=torch.linalg.vector_norm(position-self.state[4:]);done=bool(distance<=self.success_radius or self.steps>=self.max_steps);return self.state.clone(),float(-distance),done,{"success":bool(distance<=self.success_radius)}
