from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for n in ("config","dataset","model"):sys.modules.pop(n,None)
from dataset import MultimodalNavigationDataset  # noqa:E402
from model import MultimodalFusionWorldModel  # noqa:E402
def test_dataset_shapes_and_transition_targets():
    d=MultimodalNavigationDataset(8,1);assert d.vision.shape==(8,3,16,16);assert d.proprio.shape==(8,2);assert d.touch.shape==(8,4);assert d.mask.shape==(8,4);assert d.next_position.abs().max()<=1
def test_fusion_shapes_with_missing_modalities():
    d=MultimodalNavigationDataset(6,2);m=MultimodalFusionWorldModel();o=m(d.vision,d.proprio,d.language,d.touch,d.mask);assert o["tokens"].shape==(6,4,48);assert o["next_position"].shape==(6,2);assert o["next_vision"].shape==(6,3,16,16)
def test_all_components_receive_finite_gradients():
    d=MultimodalNavigationDataset(64,3,.5);m=MultimodalFusionWorldModel();o=m(d.vision,d.proprio,d.language,d.touch,d.mask);loss=torch.nn.functional.mse_loss(o["next_position"],d.next_position)+torch.nn.functional.mse_loss(o["next_vision"],d.next_vision);loss.backward();assert torch.isfinite(loss);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
def test_masked_modality_value_cannot_leak():
    d=MultimodalNavigationDataset(5,4,0.);m=MultimodalFusionWorldModel();mask=d.mask.clone();mask[:,3]=False;m.eval()
    with torch.no_grad():first=m(d.vision,d.proprio,d.language,d.touch,mask)["fused"];second=m(d.vision,d.proprio,d.language,d.touch+100,mask)["fused"]
    torch.testing.assert_close(first,second)
