from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for n in ("config","dataset","model","tokenizer"):sys.modules.pop(n,None)
from dataset import MovingSquareVideoDataset,TokenTransitionDataset  # noqa:E402
from model import LatentActionModel  # noqa:E402
from tokenizer import FrozenVQTokenizer  # noqa:E402
def test_token_pair_dataset_shapes_and_hidden_labels():
    tokenizer=FrozenVQTokenizer();data=TokenTransitionDataset(tokenizer,4,6,1);assert data.current.shape==(20,4,4);assert data.next.shape==(20,4,4);assert data.true_actions.shape==(20,)
def test_latent_action_and_token_logits_shapes():
    m=LatentActionModel();current=torch.randint(0,32,(7,4,4));future=torch.randint(0,32,(7,4,4));out=m(current,future)
    assert out["action_logits"].shape==(7,5);assert out["action"].shape==(7,5);assert out["token_logits"].shape==(7,4,4,32);torch.testing.assert_close(out["action"].sum(-1),torch.ones(7))
def test_joint_objective_gradients_reach_inference_and_dynamics():
    m=LatentActionModel();current=torch.randint(0,32,(8,4,4));future=torch.randint(0,32,(8,4,4));out=m(current,future);loss=torch.nn.functional.cross_entropy(out["token_logits"].permute(0,3,1,2),future);loss.backward();assert torch.isfinite(loss);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
def test_interactive_rollout_uses_supplied_latent_actions():
    m=LatentActionModel();initial=torch.randint(0,32,(3,4,4));actions=torch.nn.functional.one_hot(torch.randint(0,5,(3,4)),5).float();rollout=m.rollout(initial,actions);assert rollout.shape==(3,4,4,4);assert rollout.min()>=0 and rollout.max()<32
def test_tokenizer_checkpoint_translation_keys():
    tokenizer=FrozenVQTokenizer();state={}
    for k,v in tokenizer.state_dict().items():state[("quantizer.codebook."+k.split(".",1)[1]) if k.startswith("codebook.") else k]=v.clone()
    loaded=FrozenVQTokenizer().load_vq_state(state);assert all(not p.requires_grad for p in loaded.parameters())
