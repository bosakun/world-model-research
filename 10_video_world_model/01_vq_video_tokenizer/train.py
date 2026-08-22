from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from config import VQVideoConfig
from dataset import MovingSquareVideoDataset
from model import VQFrameTokenizer
ROOT=Path(__file__).resolve().parent
def build_model(c):return VQFrameTokenizer(c.codebook_size,c.embedding_dim,c.commitment_weight)
def losses(model,video):
    b,t,c,h,w=video.shape;out=model(video.reshape(b*t,c,h,w));recon=torch.nn.functional.mse_loss(out["reconstruction"],video.reshape(b*t,c,h,w));return {"total":recon+out["vq_loss"],"reconstruction":recon,"codebook":out["codebook_loss"],"commitment":out["commitment"]},out
def train(c:VQVideoConfig,output_dir:Path):
    torch.manual_seed(c.seed);train_data=MovingSquareVideoDataset(c.train_videos,c.frames,c.seed,c.image_size);val=MovingSquareVideoDataset(c.validation_videos,c.frames,c.seed+10_000,c.image_size);loader=DataLoader(train_data,c.batch_size,shuffle=True);model=build_model(c);opt=torch.optim.Adam(model.parameters(),lr=c.learning_rate);history=[]
    for epoch in range(1,c.epochs+1):
        model.train();tot={k:0. for k in ("total","reconstruction","codebook","commitment")}
        for batch in loader:
            opt.zero_grad(set_to_none=True);ls,_=losses(model,batch["video"]);ls["total"].backward();opt.step()
            for k in tot:tot[k]+=float(ls[k].detach())*batch["video"].shape[0]
        model.eval();
        with torch.no_grad():v,_=losses(model,val.videos)
        row={"epoch":epoch,**{f"train_{k}":x/len(train_data) for k,x in tot.items()},**{f"validation_{k}":float(x) for k,x in v.items()}};history.append(row)
        if epoch==1 or epoch%10==0 or epoch==c.epochs:print(f"epoch={epoch:03d} recon={row['validation_reconstruction']:.5f} vq={row['validation_codebook']+c.commitment_weight*row['validation_commitment']:.5f}")
    output_dir.mkdir(parents=True,exist_ok=True);steps=c.epochs*len(loader);torch.save({"format_version":1,"model":model.state_dict(),"config":c.to_dict(),"optimizer":"Adam","training_steps":steps},output_dir/"checkpoint.pt")
    with (output_dir/"training_history.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=history[0].keys(),lineterminator="\n");w.writeheader();w.writerows(history)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([r["epoch"] for r in history],[r["validation_reconstruction"] for r in history],label="reconstruction");ax.plot([r["epoch"] for r in history],[r["validation_codebook"] for r in history],label="codebook");ax.set(title="VQ video-frame tokenizer",xlabel="epoch",ylabel="loss");ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(output_dir/"loss_curve.png",dpi=170);plt.close(fig)
    summary={**c.to_dict(),"optimizer":"Adam","training_steps":steps,"parameter_count":sum(p.numel() for p in model.parameters()),**{k:v for k,v in history[-1].items() if k!="epoch"}};(output_dir/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return model,summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=VQVideoConfig.epochs);p.add_argument("--output-dir",type=Path,default=ROOT/"outputs");a=p.parse_args();_,s=train(VQVideoConfig(epochs=a.epochs),a.output_dir);print(s)
