from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for n in ("config","robot","dataset","model"):sys.modules.pop(n,None)
from robot import SafetyEnvelope,SimulatedMobileRobot
from dataset import DemonstrationDataset
from model import ImitationPolicy
def test_safety_clip_deadman_and_workspace_stop():
 s=SafetyEnvelope(.2);o=torch.zeros(6);assert s.filter(o,torch.tensor([1.,-1.])).clipped;assert s.filter(o,torch.ones(2),False).stopped;o[0]=2;assert s.filter(o,torch.ones(2)).reason=="workspace_violation"
def test_robot_transition_and_terminal_schema():
 r=SimulatedMobileRobot();o=r.reset();n,reward,done,info=r.step(torch.tensor([.1,.1]));assert n.shape==(6,) and not torch.equal(o,n);assert isinstance(reward,float) and isinstance(done,bool) and "success" in info
def test_demonstration_alignment():
 d=DemonstrationDataset(8,10,2);assert d.observations.shape==d.next_observations.shape;assert d.actions.shape[1]==2;torch.testing.assert_close(d.next_observations[:-1][d.episode_ids[:-1]==d.episode_ids[1:]],d.observations[1:][d.episode_ids[:-1]==d.episode_ids[1:]])
def test_policy_bounds_and_gradients():
 m=ImitationPolicy();o=torch.randn(8,6);a=m(o);assert a.abs().max()<=.2;a.square().mean().backward();assert all(p.grad is not None for p in m.parameters())
