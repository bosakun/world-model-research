from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from config import HeterogeneousConfig
from dataset import HeterogeneousControlDataset
from model import HeterogeneousSimulator
ROOT=Path(__file__).resolve().parent
def reconstruction(pred,target,w):return (((pred-target)**2)*(1.+w*target.mean(1,keepdim=True))).mean()
def train(c:HeterogeneousConfig,out:Path):
    torch.manual_seed(c.seed);data=HeterogeneousControlDataset(c.train_samples,c.seed,c.image_size);val=HeterogeneousControlDataset(c.validation_samples,c.seed+10_000,c.image_size);loader=DataLoader(data,c.batch_size,shuffle=True);m=HeterogeneousSimulator(c.condition_dim,c.latent_dim);opt=torch.optim.Adam(m.parameters(),lr=c.learning_rate);history=[]
    for epoch in range(1,c.epochs+1):
        m.train();total=0.
        for b in loader:
            opt.zero_grad(set_to_none=True);p=m(b["current"],b["kind"],b["motor"],b["language"],b["goal"])["prediction"];loss=reconstruction(p,b["next"],c.foreground_weight);loss.backward();opt.step();total+=float(loss.detach())*p.shape[0]
        m.eval();
        with torch.no_grad():p=m(val.current,val.kind,val.motor,val.language,val.goal)["prediction"];per=[float(reconstruction(p[val.kind==k],val.next[val.kind==k],c.foreground_weight)) for k in range(3)]
        row={"epoch":epoch,"train_loss":total/len(data),"validation_motor":per[0],"validation_language":per[1],"validation_goal":per[2]};history.append(row)
        if epoch==1 or epoch%10==0 or epoch==c.epochs:print(f"epoch={epoch:03d} motor={per[0]:.4f} language={per[1]:.4f} goal={per[2]:.4f}")
    out.mkdir(parents=True,exist_ok=True);steps=c.epochs*len(loader);torch.save({"format_version":1,"model":m.state_dict(),"config":c.to_dict(),"optimizer":"Adam","training_steps":steps},out/"checkpoint.pt")
    with (out/"training_history.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=history[0].keys(),lineterminator="\n");w.writeheader();w.writerows(history)
    fig,ax=plt.subplots(figsize=(7,4));
    for key in ("motor","language","goal"):ax.plot([r["epoch"] for r in history],[r[f"validation_{key}"] for r in history],label=key)
    ax.set(title="Heterogeneous-condition validation",xlabel="epoch",ylabel="weighted MSE");ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/"loss_curve.png",dpi=170);plt.close(fig);summary={**c.to_dict(),"optimizer":"Adam","training_steps":steps,"parameter_count":sum(p.numel() for p in m.parameters()),**history[-1]};(out/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return m,summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=HeterogeneousConfig.epochs);p.add_argument("--output-dir",type=Path,default=ROOT/"outputs");a=p.parse_args();_,s=train(HeterogeneousConfig(epochs=a.epochs),a.output_dir);print(s)
