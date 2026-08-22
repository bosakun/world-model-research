from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import SlotFormerConfig
from dataset import RelationalSlotSequenceDataset
from train import build_model

ROOT=Path(__file__).resolve().parent
def evaluate(output_dir:Path=ROOT/"outputs"):
    c=SlotFormerConfig(); cp=torch.load(output_dir/"checkpoint.pt",weights_only=False); model=build_model(c); model.load_state_dict(cp["model"]); model.eval()
    data=RelationalSlotSequenceDataset(192,c.sequence_length,c.seed+20_000); context=data.slots[:,:c.context_frames]; target=data.slots[:,c.context_frames:c.context_frames+c.rollout_frames]
    with torch.no_grad(): predicted=model.rollout(context,c.rollout_frames)
    per_horizon=((predicted-target)**2).mean(dim=(0,2,3)); rmse=per_horizon.sqrt()
    index=0; fig,axes=plt.subplots(1,2,figsize=(11,4))
    for obj,color in ((0,"tab:red"),(1,"tab:green")):
        axes[0].plot(context[index,:,obj,0],context[index,:,obj,1],"--o",color=color,label=f"object {obj} context")
        axes[0].plot(predicted[index,:,obj,0],predicted[index,:,obj,1],"-x",color=color,label=f"object {obj} prediction")
        axes[0].plot(target[index,:,obj,0],target[index,:,obj,1],":s",color=color,label=f"object {obj} true")
    axes[0].set(xlim=(-1,1),ylim=(-1,1),title="Autoregressive object-slot rollout",xlabel="x",ylabel="y"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
    axes[1].plot(torch.arange(1,c.rollout_frames+1),rmse,marker="o"); axes[1].set(title="Compounding slot error",xlabel="future frame",ylabel="position RMSE"); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output_dir/"slot_rollout.png",dpi=170); plt.close(fig)
    metrics={"dataset_version":c.dataset_version,"seed":c.seed,"context_frames":c.context_frames,"rollout_frames":c.rollout_frames,
             "rollout_rmse_by_horizon":[float(x) for x in rmse],"final_rollout_rmse":float(rmse[-1]),
             "evaluation_entry_point":"python 09_spatial_representation/03_slotformer/evaluate.py"}
    (output_dir/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); print(json.dumps(metrics,indent=2)); return metrics
if __name__=="__main__": evaluate()
