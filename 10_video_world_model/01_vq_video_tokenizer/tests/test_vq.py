from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for n in ("config","dataset","model"):sys.modules.pop(n,None)
from dataset import ACTION_DELTAS,MovingSquareVideoDataset  # noqa:E402
from model import VQFrameTokenizer,VectorQuantizer  # noqa:E402
def test_video_action_alignment_and_shapes():
    d=MovingSquareVideoDataset(5,6,1);assert d.videos.shape==(5,6,3,16,16);assert d.actions.shape==(5,5);expected=(d.positions[:,:-1]+ACTION_DELTAS[d.actions]).clamp(1,14);torch.testing.assert_close(d.positions[:,1:],expected)
def test_quantizer_indices_bounds_and_straight_through():
    q=VectorQuantizer();x=torch.randn(3,16,4,4,requires_grad=True);o=q(x);assert o["indices"].shape==(3,4,4);assert o["indices"].min()>=0 and o["indices"].max()<32;o["quantized"].mean().backward();assert x.grad is not None
def test_tokenizer_shapes_and_gradients():
    m=VQFrameTokenizer();frames=torch.rand(4,3,16,16);o=m(frames);assert o["reconstruction"].shape==frames.shape;assert o["indices"].shape==(4,4,4);loss=torch.nn.functional.mse_loss(o["reconstruction"],frames)+o["vq_loss"];loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
def test_video_token_shape():
    m=VQFrameTokenizer();video=torch.rand(2,6,3,16,16);assert m.tokenize_video(video).shape==(2,6,4,4)
