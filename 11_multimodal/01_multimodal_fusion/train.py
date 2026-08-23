from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from config import FusionConfig
from dataset import MultimodalNavigationDataset
from model import MultimodalFusionWorldModel
ROOT=Path(__file__).resolve().parent
def loss(m,b,c):
    o=m(b["vision"],b["proprio"],b["language"],b["touch"],b["mask"]);position=torch.nn.functional.mse_loss(o["next_position"],b["next_position"]);image=torch.nn.functional.mse_loss(o["next_vision"],b["next_vision"]);return {"total":position+c.image_weight*image,"position":position,"image":image},o
def train(c:FusionConfig,out:Path):
    torch.manual_seed(c.seed);data=MultimodalNavigationDataset(c.train_samples,c.seed,c.missing_probability);val=MultimodalNavigationDataset(c.validation_samples,c.seed+10_000,c.missing_probability);loader=DataLoader(data,c.batch_size,shuffle=True);m=MultimodalFusionWorldModel(c.model_dim,c.heads,c.layers);opt=torch.optim.Adam(m.parameters(),lr=c.learning_rate);history=[]
    for epoch in range(1,c.epochs+1):
        m.train();tot={k:0. for k in ("total","position","image")}
        for b in loader:
            opt.zero_grad(set_to_none=True);ls,_=loss(m,b,c);ls["total"].backward();opt.step()
            for k in tot:tot[k]+=float(ls[k].detach())*b["vision"].shape[0]
        m.eval();
        with torch.no_grad():v,_=loss(m,{"vision":val.vision,"proprio":val.proprio,"language":val.language,"touch":val.touch,"mask":val.mask,"next_position":val.next_position,"next_vision":val.next_vision},c)
        row={"epoch":epoch,**{f"train_{k}":x/len(data) for k,x in tot.items()},**{f"validation_{k}":float(x) for k,x in v.items()}};history.append(row)
        if epoch==1 or epoch%10==0 or epoch==c.epochs:print(f"epoch={epoch:03d} position={row['validation_position']:.5f} image={row['validation_image']:.5f}")
    out.mkdir(parents=True,exist_ok=True);steps=c.epochs*len(loader);torch.save({"format_version":1,"model":m.state_dict(),"config":c.to_dict(),"optimizer":"Adam","training_steps":steps},out/"checkpoint.pt")
    with (out/"training_history.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=history[0].keys(),lineterminator="\n");w.writeheader();w.writerows(history)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([r["epoch"] for r in history],[r["validation_position"] for r in history],label="position");ax.plot([r["epoch"] for r in history],[r["validation_image"] for r in history],label="image");ax.set(title="Multimodal fusion validation",xlabel="epoch");ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/"loss_curve.png",dpi=170);plt.close(fig);summary={**c.to_dict(),"optimizer":"Adam","training_steps":steps,"parameter_count":sum(p.numel() for p in m.parameters()),**{k:v for k,v in history[-1].items() if k!="epoch"}};(out/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return m,summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=FusionConfig.epochs);p.add_argument("--output-dir",type=Path,default=ROOT/"outputs");a=p.parse_args();_,s=train(FusionConfig(epochs=a.epochs),a.output_dir);print(s)
