from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for n in ("config","dataset","model"):sys.modules.pop(n,None)
from dataset import HeterogeneousControlDataset  # noqa:E402
from model import HeterogeneousSimulator  # noqa:E402
def test_all_sources_and_shapes():
    d=HeterogeneousControlDataset(9,1);assert d.current.shape==(9,3,16,16);assert set(d.kind.tolist())=={0,1,2};assert d.motor.shape==d.goal.shape==(9,2)
def test_condition_adapters_select_one_shared_shape():
    d=HeterogeneousControlDataset(6,2);m=HeterogeneousSimulator();c=m.condition(d.kind,d.motor,d.language,d.goal);assert c.shape==(6,16);assert torch.isfinite(c).all()
def test_forward_shapes_and_all_adapter_gradients():
    d=HeterogeneousControlDataset(9,3);m=HeterogeneousSimulator();o=m(d.current,d.kind,d.motor,d.language,d.goal);assert o["prediction"].shape==d.next.shape;loss=torch.nn.functional.mse_loss(o["prediction"],d.next);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
def test_unused_field_does_not_change_selected_condition():
    d=HeterogeneousControlDataset(6,4);m=HeterogeneousSimulator();first=m.condition(d.kind,d.motor,d.language,d.goal);changed=m.condition(d.kind,d.motor+100,d.language,d.goal);torch.testing.assert_close(first[d.kind!=0],changed[d.kind!=0])
