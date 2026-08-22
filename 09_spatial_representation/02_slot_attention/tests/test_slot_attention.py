from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
for name in ("config","dataset","model","train"): sys.modules.pop(name,None)
from dataset import TwoObjectImagesDataset  # noqa:E402
from model import SlotAttention, SlotAttentionAutoencoder  # noqa:E402
from train import reconstruction_loss  # noqa:E402

def test_dataset_shapes_and_two_objects_share_one_image():
    data=TwoObjectImagesDataset(6,1); assert data.images.shape==(6,3,16,16); assert data.masks.shape==(6,3,16,16)
    assert data.images[:,0].sum()>0 and data.images[:,1].sum()>0; assert data.images[:,2].sum()==0
    torch.testing.assert_close(data.masks.sum(dim=1),torch.ones(6,16,16))

def test_attention_competes_over_slots_for_every_token():
    module=SlotAttention(); module.eval(); slots,attention=module(torch.randn(4,256,32),False)
    assert slots.shape==(4,3,32); assert attention.shape==(4,256,3)
    torch.testing.assert_close(attention.sum(dim=-1),torch.ones(4,256),atol=1e-5,rtol=1e-5)

def test_autoencoder_shapes_masks_and_gradients():
    model=SlotAttentionAutoencoder(); images=TwoObjectImagesDataset(4,2).images; output=model(images)
    assert output["reconstruction"].shape==images.shape; assert output["masks"].shape==(4,3,1,16,16)
    torch.testing.assert_close(output["masks"].sum(dim=1),torch.ones(4,1,16,16),atol=1e-5,rtol=1e-5)
    loss=reconstruction_loss(output["reconstruction"],images,8.0); loss.backward(); assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_deterministic_evaluation_initialization():
    model=SlotAttentionAutoencoder(); model.eval(); images=TwoObjectImagesDataset(2,3).images
    torch.manual_seed(99)
    with torch.no_grad(): first=model(images,True)["slots"]
    torch.manual_seed(99)
    with torch.no_grad(): second=model(images,True)["slots"]
    torch.testing.assert_close(first,second)
