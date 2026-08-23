from __future__ import annotations
import itertools,json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import LatentActionConfig
from dataset import MovingSquareVideoDataset,TokenTransitionDataset
from model import LatentActionModel
from tokenizer import FrozenVQTokenizer
ROOT=Path(__file__).resolve().parent
def best_mapping(predicted,true,k=5):
    best_acc=-1.;best=None
    for permutation in itertools.permutations(range(k)):
        mapped=torch.tensor(permutation)[predicted];acc=float((mapped==true).float().mean())
        if acc>best_acc:best_acc,best=acc,permutation
    return best,best_acc
def evaluate(output_dir:Path=ROOT/"outputs"):
    c=LatentActionConfig();cp=torch.load(output_dir/"checkpoint.pt",weights_only=False);tokenizer=FrozenVQTokenizer();tokenizer.load_state_dict(cp["tokenizer"]);tokenizer.eval();model=LatentActionModel(c.codebook_size,c.token_dim,c.latent_actions,c.model_dim,c.heads,c.layers);model.load_state_dict(cp["model"]);model.eval();pairs=TokenTransitionDataset(tokenizer,256,c.frames,c.seed+20_000)
    with torch.no_grad():
        logits,_=model.infer_action(pairs.current,pairs.next,c.gumbel_temperature,False);pred_action=logits.argmax(-1)
        onehot=torch.nn.functional.one_hot(pred_action,c.latent_actions).float();token_logits=model.predict(pairs.current,onehot);pred_tokens=token_logits.argmax(-1)
    mapping,action_accuracy=best_mapping(pred_action,pairs.true_actions,c.latent_actions);token_accuracy=float((pred_tokens==pairs.next).float().mean());changed=pairs.current!=pairs.next;changed_accuracy=float((pred_tokens[changed]==pairs.next[changed]).float().mean());copy_accuracy=float((pairs.current==pairs.next).float().mean());usage=torch.bincount(pred_action,minlength=c.latent_actions)
    video=MovingSquareVideoDataset(1,c.frames,c.seed+30_000);true_tokens=tokenizer.tokens(video.videos);inverse=torch.empty(c.latent_actions,dtype=torch.long)
    for latent,true_action in enumerate(mapping):inverse[true_action]=latent
    action_onehot=torch.nn.functional.one_hot(inverse[video.actions],c.latent_actions).float()
    with torch.no_grad():rollout=model.rollout(true_tokens[:,0],action_onehot)
    all_tokens=torch.cat((true_tokens[:,0:1],rollout),dim=1)
    with torch.no_grad():reconstructed=torch.stack([tokenizer.decode_tokens(all_tokens[:,i]) for i in range(c.frames)],dim=1)
    fig,axes=plt.subplots(2,c.frames,figsize=(2*c.frames,4));
    for i in range(c.frames):axes[0,i].imshow(video.videos[0,i].permute(1,2,0));axes[0,i].set_title(f"true {i}");axes[1,i].imshow(reconstructed[0,i].permute(1,2,0));axes[1,i].set_title(f"token rollout {i}")
    for ax in axes.flat:ax.axis("off")
    fig.tight_layout();fig.savefig(output_dir/"latent_action_rollout.png",dpi=170);plt.close(fig);metrics={"dataset_version":c.dataset_version,"seed":c.seed,"latent_to_true_action_permutation":list(mapping),"permutation_action_accuracy":action_accuracy,"next_token_accuracy":token_accuracy,"changed_token_accuracy":changed_accuracy,"copy_current_token_accuracy":copy_accuracy,"changed_token_fraction":float(changed.float().mean()),"latent_action_usage":usage.tolist(),"rollout_token_accuracy_by_frame":[float((all_tokens[:,i]==true_tokens[:,i]).float().mean()) for i in range(c.frames)],"evaluation_entry_point":"python 10_video_world_model/02_latent_action_dynamics/evaluate.py"};(output_dir/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2));return metrics
if __name__=="__main__":evaluate()
