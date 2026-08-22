from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import VQVideoConfig
from dataset import MovingSquareVideoDataset
from train import build_model
ROOT=Path(__file__).resolve().parent
def evaluate(output_dir:Path=ROOT/"outputs"):
    c=VQVideoConfig();cp=torch.load(output_dir/"checkpoint.pt",weights_only=False);m=build_model(c);m.load_state_dict(cp["model"]);m.eval();d=MovingSquareVideoDataset(128,c.frames,c.seed+20_000,c.image_size);b,t,ch,h,w=d.videos.shape
    with torch.no_grad():out=m(d.videos.reshape(b*t,ch,h,w));tokens=out["indices"].reshape(b,t,4,4);recon=out["reconstruction"].reshape_as(d.videos)
    counts=torch.bincount(tokens.flatten(),minlength=c.codebook_size).float();prob=counts/counts.sum();perplexity=torch.exp(-(prob[prob>0]*prob[prob>0].log()).sum());mse=torch.nn.functional.mse_loss(recon,d.videos)
    fig,axes=plt.subplots(2,c.frames,figsize=(2*c.frames,4));
    for i in range(c.frames):axes[0,i].imshow(d.videos[0,i].permute(1,2,0));axes[0,i].set_title(f"true {i}");axes[1,i].imshow(recon[0,i].permute(1,2,0));axes[1,i].set_title(f"recon {i}")
    for ax in axes.flat:ax.axis("off")
    fig.tight_layout();fig.savefig(output_dir/"video_reconstruction.png",dpi=170);plt.close(fig)
    metrics={"dataset_version":c.dataset_version,"seed":c.seed,"reconstruction_mse":float(mse),"codebook_perplexity":float(perplexity),"active_codes":int((counts>0).sum()),"tokens_per_frame":16,"evaluation_entry_point":"python 10_video_world_model/01_vq_video_tokenizer/evaluate.py"};(output_dir/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2));return metrics
if __name__=="__main__":evaluate()
