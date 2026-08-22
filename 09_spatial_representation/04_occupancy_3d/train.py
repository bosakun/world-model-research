from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from config import OccupancyConfig
from dataset import MovingOccupancyDataset
from losses import occupancy_loss
from model import OccupancyWorldModel
ROOT=Path(__file__).resolve().parent
def build_model(c): return OccupancyWorldModel(c.action_dim,c.latent_dim)
def occupancy_center(probability:torch.Tensor)->torch.Tensor:
    size=probability.shape[-1]; coordinates=torch.arange(size,dtype=probability.dtype,device=probability.device)
    zz,yy,xx=torch.meshgrid(coordinates,coordinates,coordinates,indexing="ij"); grid=torch.stack((xx,yy,zz),dim=-1)
    weights=probability.squeeze(-4); mass=weights.sum(dim=(-3,-2,-1),keepdim=True).clamp_min(1e-6)
    return (weights[...,None]*grid).sum(dim=(-4,-3,-2))/mass.reshape(mass.shape[0],mass.shape[1],1)
def batch_loss(model,occupancies,actions,centers,c):
    rollout=model.rollout(occupancies[:,0],actions); voxel=occupancy_loss(rollout["logits"],occupancies[:,1:],c.positive_weight,c.bce_weight,c.dice_weight)
    shape=occupancies[:,1:].shape; target_latents=model.encode(occupancies[:,1:].reshape(-1,*shape[2:])).reshape(shape[0],shape[1],-1).detach()
    consistency=torch.nn.functional.mse_loss(rollout["latents"],target_latents)
    center=torch.nn.functional.mse_loss(occupancy_center(rollout["logits"].sigmoid()),centers[:,1:])
    return {**voxel,"consistency":consistency,"center":center,"total":voxel["total"]+c.consistency_weight*consistency+c.center_weight*center}
def train(c:OccupancyConfig,output_dir:Path):
    torch.manual_seed(c.seed); train_data=MovingOccupancyDataset(c.train_sequences,c.horizon,c.seed,c.grid_size); val=MovingOccupancyDataset(c.validation_sequences,c.horizon,c.seed+10_000,c.grid_size)
    loader=DataLoader(train_data,c.batch_size,shuffle=True); model=build_model(c); optimizer=torch.optim.Adam(model.parameters(),lr=c.learning_rate); history=[]
    for epoch in range(1,c.epochs+1):
        model.train(); totals={k:0. for k in ("total","bce","dice","consistency","center")}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); losses=batch_loss(model,batch["occupancies"],batch["actions"],batch["centers"],c); losses["total"].backward(); optimizer.step()
            for k in totals: totals[k]+=float(losses[k].detach())*batch["actions"].shape[0]
        model.eval()
        with torch.no_grad(): validation=batch_loss(model,val.occupancies,val.actions,val.centers,c)
        row={"epoch":epoch,**{f"train_{k}":v/len(train_data) for k,v in totals.items()},**{f"validation_{k}":float(v) for k,v in validation.items()}}; history.append(row)
        if epoch==1 or epoch%10==0 or epoch==c.epochs: print(f"epoch={epoch:03d} bce={row['validation_bce']:.4f} dice={row['validation_dice']:.4f}")
    output_dir.mkdir(parents=True,exist_ok=True); steps=c.epochs*len(loader)
    torch.save({"format_version":1,"model":model.state_dict(),"config":c.to_dict(),"optimizer":"Adam","training_steps":steps},output_dir/"checkpoint.pt")
    with (output_dir/"training_history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys(),lineterminator="\n"); w.writeheader(); w.writerows(history)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([r["epoch"] for r in history],[r["validation_bce"] for r in history],label="BCE"); ax.plot([r["epoch"] for r in history],[r["validation_dice"] for r in history],label="Dice")
    ax.set(title="3D occupancy rollout losses",xlabel="epoch"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(output_dir/"loss_curve.png",dpi=170); plt.close(fig)
    summary={**c.to_dict(),"optimizer":"Adam","training_steps":steps,"parameter_count":sum(p.numel() for p in model.parameters()),**{k:v for k,v in history[-1].items() if k!="epoch"}}
    (output_dir/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return model,summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=OccupancyConfig.epochs);p.add_argument("--output-dir",type=Path,default=ROOT/"outputs");a=p.parse_args();_,s=train(OccupancyConfig(epochs=a.epochs),a.output_dir);print(s)
