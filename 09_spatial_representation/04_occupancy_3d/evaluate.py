from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import OccupancyConfig
from dataset import MovingOccupancyDataset
from train import build_model
ROOT=Path(__file__).resolve().parent
def evaluate(output_dir:Path=ROOT/"outputs"):
    c=OccupancyConfig();cp=torch.load(output_dir/"checkpoint.pt",weights_only=False);model=build_model(c);model.load_state_dict(cp["model"]);model.eval();data=MovingOccupancyDataset(128,c.horizon,c.seed+20_000,c.grid_size)
    with torch.no_grad(): probability=model.rollout(data.occupancies[:,0],data.actions)["logits"].sigmoid()
    predicted=probability>=.5; target=data.occupancies[:,1:].bool(); intersection=(predicted&target).sum(dim=(0,2,3,4,5)); union=(predicted|target).sum(dim=(0,2,3,4,5)).clamp_min(1); iou=intersection.float()/union
    index=0; fig,axes=plt.subplots(2,c.horizon,figsize=(2*c.horizon,4))
    for t in range(c.horizon):
        axes[0,t].imshow(target[index,t,0].any(dim=0),cmap="gray");axes[0,t].set_title(f"true t+{t+1}")
        axes[1,t].imshow(predicted[index,t,0].any(dim=0),cmap="gray");axes[1,t].set_title(f"pred t+{t+1}")
    for ax in axes.flat: ax.axis("off")
    fig.tight_layout();fig.savefig(output_dir/"occupancy_rollout.png",dpi=170);plt.close(fig)
    metrics={"dataset_version":c.dataset_version,"seed":c.seed,"voxel_iou_by_horizon":[float(x) for x in iou],"final_voxel_iou":float(iou[-1]),"threshold":.5,"evaluation_entry_point":"python 09_spatial_representation/04_occupancy_3d/evaluate.py"}
    (output_dir/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2));return metrics
if __name__=="__main__":evaluate()
