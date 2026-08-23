from __future__ import annotations
import csv,json,time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from benchmark_dataset import alias_mask,goal_labels,make_dataset
from config import BenchmarkConfig
from models import build_models
ROOT=Path(__file__).resolve().parent
def count(model):return sum(p.numel() for p in model.parameters())
def loss(model,b,c):
 out=model.teacher(b["observations"],b["actions"]);target=b["observations"][:,1:];labels=goal_labels(b["true_states"][:,1:]);prediction=torch.nn.functional.mse_loss(out["images"],target);recon=torch.nn.functional.mse_loss(out["recon"],b["observations"]);latent=torch.nn.functional.mse_loss(out["pred"],out["latents"][:,1:].detach());goal=torch.nn.functional.cross_entropy(out["goal"].reshape(-1,2),labels.reshape(-1));total=c.prediction_weight*prediction+c.reconstruction_weight*recon+c.latent_weight*latent+c.goal_weight*goal+c.kl_weight*out["extra"];return total
def evaluate(model,data,c,ablate=False):
 model.eval();o=data.observations;a=data.actions
 with torch.no_grad():images,logits=model.rollout(o,a,c.context_steps,ablate)
 target=o[:,c.context_steps+1:];labels=goal_labels(data.true_states[:,c.context_steps+1:]);aliases=alias_mask(o)[:,c.context_steps+1:];per=((images-target)**2).mean(dim=(0,2,3,4));correct=logits.argmax(-1)==labels;alias_accuracy=float(correct[aliases].float().mean()) if aliases.any() else float("nan")
 for _ in range(3):model.rollout(o[:16],a[:16],c.context_steps,ablate)
 start=time.perf_counter()
 with torch.no_grad():
  for _ in range(20):model.rollout(o[:16],a[:16],c.context_steps,ablate)
 latency=(time.perf_counter()-start)/20*1000
 return {"mse_h1":float(per[0]),"mse_h5":float(per[4]),"mse_h10":float(per[9]),"goal_accuracy":float(correct.float().mean()),"aliased_goal_accuracy":alias_accuracy,"latency_ms_batch16":latency}
def run(c:BenchmarkConfig,out:Path=ROOT/"outputs"):
 out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True);rows=[]
 for seed in c.seeds:
  train_data=make_dataset(c.train_sequences,c.sequence_length,seed);test=make_dataset(c.test_sequences,c.sequence_length,seed+10_000);loader=DataLoader(train_data,c.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
  for name,model in build_models(c.latent_dim,c.hidden_dim).items():
   torch.manual_seed(seed);model=build_models(c.latent_dim,c.hidden_dim)[name];optimizer=torch.optim.Adam(model.parameters(),lr=c.learning_rate)
   for epoch in range(c.epochs):
    model.train()
    for batch in loader:optimizer.zero_grad(set_to_none=True);value=loss(model,batch,c);value.backward();optimizer.step()
   metrics=evaluate(model,test,c);ablation=evaluate(model,test,c,True)
   row={"seed":seed,"model":name,"parameters":count(model),"parameter_bytes":count(model)*4,"training_steps":c.epochs*len(loader),**metrics,"ablated_aliased_goal_accuracy":ablation["aliased_goal_accuracy"]};rows.append(row);print(row)
   torch.save({"format_version":1,"model":model.state_dict(),"model_name":name,"seed":seed,"config":c.to_dict(),"optimizer":"Adam","training_steps":c.epochs*len(loader)},out/"checkpoints"/f"{name}_seed{seed}.pt")
 with (out/"per_seed_results.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n");w.writeheader();w.writerows(rows)
 aggregate={}
 for name in build_models():
  selected=[r for r in rows if r["model"]==name];aggregate[name]={key:{"mean":sum(r[key] for r in selected)/len(selected),"std":float(torch.tensor([r[key] for r in selected]).std(unbiased=False))} for key in ("mse_h1","mse_h5","mse_h10","goal_accuracy","aliased_goal_accuracy","ablated_aliased_goal_accuracy","latency_ms_batch16")};aggregate[name]["parameters"]=selected[0]["parameters"]
 payload={"dataset_version":c.dataset_version,"config":c.to_dict(),"aggregate":aggregate,"rows":rows,"evaluation_entry_point":"python 90_evaluation/01_memory_benchmark/run_benchmark.py"};(out/"benchmark_results.json").write_text(json.dumps(payload,indent=2)+"\n")
 names=list(aggregate);fig,axes=plt.subplots(1,3,figsize=(14,4));
 for name in names:axes[0].plot([1,5,10],[aggregate[name][f"mse_h{h}"]["mean"] for h in (1,5,10)],marker="o",label=name)
 axes[0].set(title="Autoregressive image error",xlabel="horizon",ylabel="MSE");axes[0].legend();axes[1].bar(names,[aggregate[n]["aliased_goal_accuracy"]["mean"] for n in names]);axes[1].set(title="Aliased hidden-Goal accuracy",ylim=(0,1));x=torch.arange(len(names));axes[2].bar(x-.18,[aggregate[n]["aliased_goal_accuracy"]["mean"] for n in names],.36,label="normal");axes[2].bar(x+.18,[aggregate[n]["ablated_aliased_goal_accuracy"]["mean"] for n in names],.36,label="memory reset/context 1");axes[2].set_xticks(x,names);axes[2].set(title="Memory ablation",ylim=(0,1));axes[2].legend()
 for ax in axes:ax.grid(alpha=.3)
 fig.tight_layout();fig.savefig(out/"memory_comparison.png",dpi=170);plt.close(fig);return payload
if __name__=="__main__":run(BenchmarkConfig())
