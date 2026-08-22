from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
for name in ("config","dataset","model"): sys.modules.pop(name,None)
from dataset import RelationalSlotSequenceDataset,step_objects  # noqa:E402
from model import SlotFormer,frame_causal_mask  # noqa:E402

def test_dataset_shape_and_transition():
    data=RelationalSlotSequenceDataset(5,7,1); assert data.slots.shape==(5,7,2,2); assert data.slots.abs().max()<=.9
def test_frame_causal_mask_allows_same_frame_and_blocks_future():
    mask=frame_causal_mask(3,2); assert mask.shape==(6,6); assert not mask[0,1] and not mask[1,0]; assert mask[0,2] and not mask[2,0]
def test_forward_rollout_shapes_and_finite_values():
    model=SlotFormer(); slots=torch.randn(4,5,2,2); prediction=model(slots); rollout=model.rollout(slots[:,:3],4)
    assert prediction.shape==(4,5,2,2); assert rollout.shape==(4,4,2,2); assert torch.isfinite(rollout).all()
def test_causal_prediction_unchanged_by_future_frame_edit():
    torch.manual_seed(2); model=SlotFormer(); model.eval(); slots=torch.randn(2,5,2,2); changed=slots.clone(); changed[:,4]+=100
    with torch.no_grad(): first=model(slots); second=model(changed)
    torch.testing.assert_close(first[:,:4],second[:,:4],atol=1e-5,rtol=1e-5)
def test_gradient_reaches_attention_and_output():
    model=SlotFormer(); slots=RelationalSlotSequenceDataset(4,6,3).slots; loss=torch.nn.functional.mse_loss(model(slots[:,:-1]),slots[:,1:]); loss.backward()
    assert torch.isfinite(loss); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
