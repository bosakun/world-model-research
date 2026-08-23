from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for n in ("config","models","benchmark_dataset"):sys.modules.pop(n,None)
from benchmark_dataset import alias_mask,goal_labels,make_dataset  # noqa:E402
from models import build_models  # noqa:E402
def test_benchmark_dataset_alias_and_labels():
 d=make_dataset(8,12,1);assert d.observations.shape==(8,13,3,20,20);assert alias_mask(d.observations)[:,2].all();assert set(goal_labels(d.true_states).unique().tolist())=={0,1}
def test_all_models_share_teacher_interface():
 d=make_dataset(4,4,2)
 for model in build_models().values():
  out=model.teacher(d.observations,d.actions);assert out["images"].shape==(4,4,3,20,20);assert out["goal"].shape==(4,4,2);assert torch.isfinite(out["images"]).all()
def test_all_models_roll_ten_steps():
 d=make_dataset(4,12,3)
 for model in build_models().values():
  images,goals=model.rollout(d.observations,d.actions,2);assert images.shape==(4,10,3,20,20);assert goals.shape==(4,10,2)
def test_all_models_receive_gradients():
 d=make_dataset(4,4,4)
 for model in build_models().values():
  out=model.teacher(d.observations,d.actions);value=torch.nn.functional.mse_loss(out["images"],d.observations[:,1:])+torch.nn.functional.cross_entropy(out["goal"].reshape(-1,2),goal_labels(d.true_states[:,1:]).reshape(-1))+out["extra"]*.01;value.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
