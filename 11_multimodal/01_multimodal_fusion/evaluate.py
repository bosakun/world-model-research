from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import FusionConfig
from dataset import MultimodalNavigationDataset
from model import MultimodalFusionWorldModel
ROOT=Path(__file__).resolve().parent
def evaluate(out:Path=ROOT/"outputs"):
    c=FusionConfig();cp=torch.load(out/"checkpoint.pt",weights_only=False);m=MultimodalFusionWorldModel(c.model_dim,c.heads,c.layers);m.load_state_dict(cp["model"]);m.eval();d=MultimodalNavigationDataset(384,c.seed+20_000,0.);conditions={"all":torch.ones_like(d.mask),"no_vision":torch.ones_like(d.mask),"no_proprio":torch.ones_like(d.mask),"no_language":torch.ones_like(d.mask),"no_touch":torch.ones_like(d.mask)}
    for index,name in enumerate(("no_vision","no_proprio","no_language","no_touch")):conditions[name][:,index]=False
    metrics={}
    with torch.no_grad():
        outputs={name:m(d.vision,d.proprio,d.language,d.touch,mask) for name,mask in conditions.items()}
    for name,o in outputs.items():metrics[name+"_position_rmse"]=float(torch.sqrt(torch.nn.functional.mse_loss(o["next_position"],d.next_position)))
    fig,axes=plt.subplots(2,3,figsize=(8,5));axes[0,0].imshow(d.vision[0].permute(1,2,0));axes[0,0].set_title("current");axes[0,1].imshow(d.next_vision[0].permute(1,2,0));axes[0,1].set_title("true next");axes[0,2].imshow(outputs["all"]["next_vision"][0].permute(1,2,0));axes[0,2].set_title("all modalities")
    for ax,(name,o) in zip(axes[1],list(outputs.items())[1:4]):ax.imshow(o["next_vision"][0].permute(1,2,0));ax.set_title(name)
    for ax in axes.flat:ax.axis("off")
    fig.tight_layout();fig.savefig(out/"modality_ablation.png",dpi=170);plt.close(fig);metrics.update({"dataset_version":c.dataset_version,"seed":c.seed,"evaluation_entry_point":"python 11_multimodal/01_multimodal_fusion/evaluate.py"});(out/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2));return metrics
if __name__=="__main__":evaluate()
