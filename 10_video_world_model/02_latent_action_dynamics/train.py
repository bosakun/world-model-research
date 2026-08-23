from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from config import LatentActionConfig
from dataset import TokenTransitionDataset
from model import LatentActionModel
from tokenizer import FrozenVQTokenizer
ROOT=Path(__file__).resolve().parent;DEFAULT_TOKENIZER=ROOT.parents[1]/"10_video_world_model"/"01_vq_video_tokenizer"/"outputs"/"checkpoint.pt"
def load_tokenizer(path):
    if not path.exists():raise FileNotFoundError(f"Run 10_video_world_model/01_vq_video_tokenizer/train.py first: {path}")
    cp=torch.load(path,weights_only=False);return FrozenVQTokenizer().load_vq_state(cp["model"])
def objective(model,batch,c):
    out=model(batch["current"],batch["next"],c.gumbel_temperature);per_token=torch.nn.functional.cross_entropy(out["token_logits"].permute(0,3,1,2),batch["next"],reduction="none");weights=1.+c.changed_token_weight*(batch["current"]!=batch["next"]).float();ce=(per_token*weights).sum()/weights.sum()
    probability=out["action_logits"].softmax(-1);confidence=-(probability*probability.clamp_min(1e-8).log()).sum(-1).mean();marginal=probability.mean(0);balance=(marginal*marginal.clamp_min(1e-8).log()).sum()
    return {"total":ce+c.confidence_weight*confidence+c.balance_weight*balance,"token_ce":ce,"action_entropy":confidence,"balance":balance},out
def train(c:LatentActionConfig,output_dir:Path,tokenizer_checkpoint:Path=DEFAULT_TOKENIZER):
    torch.manual_seed(c.seed);tokenizer=load_tokenizer(tokenizer_checkpoint);train_data=TokenTransitionDataset(tokenizer,c.train_videos,c.frames,c.seed);val=TokenTransitionDataset(tokenizer,c.validation_videos,c.frames,c.seed+10_000);loader=DataLoader(train_data,c.batch_size,shuffle=True);model=LatentActionModel(c.codebook_size,c.token_dim,c.latent_actions,c.model_dim,c.heads,c.layers);opt=torch.optim.Adam(model.parameters(),lr=c.learning_rate);history=[]
    for epoch in range(1,c.epochs+1):
        model.train();tot={k:0. for k in ("total","token_ce","action_entropy","balance")}
        for batch in loader:
            opt.zero_grad(set_to_none=True);ls,_=objective(model,batch,c);ls["total"].backward();opt.step()
            for k in tot:tot[k]+=float(ls[k].detach())*batch["current"].shape[0]
        model.eval();torch.manual_seed(c.seed+epoch)
        with torch.no_grad():v,_=objective(model,{"current":val.current,"next":val.next},c)
        row={"epoch":epoch,**{f"train_{k}":x/len(train_data) for k,x in tot.items()},**{f"validation_{k}":float(x) for k,x in v.items()}};history.append(row)
        if epoch==1 or epoch%10==0 or epoch==c.epochs:print(f"epoch={epoch:03d} token_ce={row['validation_token_ce']:.4f} action_H={row['validation_action_entropy']:.4f}")
    output_dir.mkdir(parents=True,exist_ok=True);steps=c.epochs*len(loader);torch.save({"format_version":1,"model":model.state_dict(),"tokenizer":tokenizer.state_dict(),"config":c.to_dict(),"optimizer":"Adam","training_steps":steps},output_dir/"checkpoint.pt")
    with (output_dir/"training_history.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=history[0].keys(),lineterminator="\n");w.writeheader();w.writerows(history)
    fig,axes=plt.subplots(1,2,figsize=(10,4));epochs=[r["epoch"] for r in history];axes[0].plot(epochs,[r["validation_token_ce"] for r in history]);axes[0].set(title="Next-token cross entropy",xlabel="epoch");axes[1].plot(epochs,[r["validation_action_entropy"] for r in history],label="per-example entropy");axes[1].plot(epochs,[-r["validation_balance"] for r in history],label="marginal entropy");axes[1].set(title="Latent-action usage",xlabel="epoch");axes[1].legend()
    for ax in axes:ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(output_dir/"training_diagnostics.png",dpi=170);plt.close(fig);summary={**c.to_dict(),"optimizer":"Adam","training_steps":steps,"parameter_count":sum(p.numel() for p in model.parameters()),**{k:v for k,v in history[-1].items() if k!="epoch"}};(output_dir/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return tokenizer,model,summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=LatentActionConfig.epochs);p.add_argument("--output-dir",type=Path,default=ROOT/"outputs");p.add_argument("--tokenizer-checkpoint",type=Path,default=DEFAULT_TOKENIZER);a=p.parse_args();*_,s=train(LatentActionConfig(epochs=a.epochs),a.output_dir,a.tokenizer_checkpoint);print(s)
