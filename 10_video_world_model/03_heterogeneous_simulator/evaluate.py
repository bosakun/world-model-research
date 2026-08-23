from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import HeterogeneousConfig
from dataset import HeterogeneousControlDataset
from model import HeterogeneousSimulator
ROOT=Path(__file__).resolve().parent
def evaluate(out:Path=ROOT/"outputs"):
    c=HeterogeneousConfig();cp=torch.load(out/"checkpoint.pt",weights_only=False);m=HeterogeneousSimulator(c.condition_dim,c.latent_dim);m.load_state_dict(cp["model"]);m.eval();d=HeterogeneousControlDataset(384,c.seed+20_000,c.image_size)
    with torch.no_grad():p=m(d.current,d.kind,d.motor,d.language,d.goal)["prediction"]
    mse=[float(torch.nn.functional.mse_loss(p[d.kind==k],d.next[d.kind==k])) for k in range(3)];names=["motor","language","goal"];fig,axes=plt.subplots(3,3,figsize=(7,7))
    for k in range(3):
        i=int((d.kind==k).nonzero()[0]);axes[k,0].imshow(d.current[i].permute(1,2,0));axes[k,1].imshow(d.next[i].permute(1,2,0));axes[k,2].imshow(p[i].permute(1,2,0));axes[k,0].set_ylabel(names[k]);
    for ax in axes.flat:ax.axis("off")
    axes[0,0].set_title("current");axes[0,1].set_title("true next");axes[0,2].set_title("predicted");fig.tight_layout();fig.savefig(out/"heterogeneous_predictions.png",dpi=170);plt.close(fig);metrics={"dataset_version":c.dataset_version,"seed":c.seed,"mse_by_source":dict(zip(names,mse)),"max_source_mse":max(mse),"evaluation_entry_point":"python 10_video_world_model/03_heterogeneous_simulator/evaluate.py"};(out/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2));return metrics
if __name__=="__main__":evaluate()
