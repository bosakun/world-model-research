from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from config import SlotFormerConfig
from dataset import RelationalSlotSequenceDataset
from model import SlotFormer

ROOT=Path(__file__).resolve().parent
def build_model(c): return SlotFormer(c.num_slots,c.slot_dim,c.model_dim,c.heads,c.layers,c.max_frames)
def train(config:SlotFormerConfig,output_dir:Path):
    torch.manual_seed(config.seed); train_data=RelationalSlotSequenceDataset(config.train_sequences,config.sequence_length,config.seed)
    validation=RelationalSlotSequenceDataset(config.validation_sequences,config.sequence_length,config.seed+10_000)
    loader=DataLoader(train_data,config.batch_size,shuffle=True); model=build_model(config); optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    history=[]
    for epoch in range(1,config.epochs+1):
        model.train(); total=0.
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); prediction=model(batch["slots"][:,:-1]); loss=torch.nn.functional.mse_loss(prediction,batch["slots"][:,1:])
            loss.backward(); optimizer.step(); total+=float(loss.detach())*batch["slots"].shape[0]
        model.eval()
        with torch.no_grad(): val=torch.nn.functional.mse_loss(model(validation.slots[:,:-1]),validation.slots[:,1:])
        row={"epoch":epoch,"train_teacher_forced_mse":total/len(train_data),"validation_teacher_forced_mse":float(val)}; history.append(row)
        if epoch==1 or epoch%10==0 or epoch==config.epochs: print(f"epoch={epoch:03d} validation_mse={float(val):.6f}")
    output_dir.mkdir(parents=True,exist_ok=True); steps=config.epochs*len(loader)
    torch.save({"format_version":1,"model":model.state_dict(),"config":config.to_dict(),"optimizer":"Adam","training_steps":steps},output_dir/"checkpoint.pt")
    with (output_dir/"training_history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys(),lineterminator="\n"); w.writeheader(); w.writerows(history)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([r["epoch"] for r in history],[r["train_teacher_forced_mse"] for r in history],label="train")
    ax.plot([r["epoch"] for r in history],[r["validation_teacher_forced_mse"] for r in history],label="validation")
    ax.set(title="SlotFormer teacher-forced objective",xlabel="epoch",ylabel="MSE"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(output_dir/"loss_curve.png",dpi=170); plt.close(fig)
    summary={**config.to_dict(),"optimizer":"Adam","training_steps":steps,"parameter_count":sum(p.numel() for p in model.parameters()),**history[-1]}
    (output_dir/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return model,summary
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=SlotFormerConfig.epochs); p.add_argument("--output-dir",type=Path,default=ROOT/"outputs"); a=p.parse_args()
    _,s=train(SlotFormerConfig(epochs=a.epochs),a.output_dir); print(s)
