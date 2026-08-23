from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from config import RobotConfig
from model import ImitationPolicy
from robot import SafetyEnvelope,SimulatedMobileRobot
ROOT=Path(__file__).resolve().parent
def evaluate(out:Path=ROOT/"outputs"):
 c=RobotConfig();cp=torch.load(out/"checkpoint.pt",weights_only=False);m=ImitationPolicy(c.observation_dim,c.action_dim,c.hidden_dim,c.max_speed);m.load_state_dict(cp["model"]);m.eval();g=torch.Generator().manual_seed(c.seed+20_000);successes=[];final_distances=[];clip_count=0;example=None
 for episode in range(64):
  start=tuple(torch.empty(2).uniform_(-.9,0,generator=g).tolist());goal=tuple(torch.empty(2).uniform_(.1,.9,generator=g).tolist());robot=SimulatedMobileRobot(start,goal,c.horizon,c.success_radius);safety=SafetyEnvelope(c.max_speed,c.workspace_limit);o=robot.reset();states=[o]
  for _ in range(c.horizon):
   with torch.no_grad():requested=m(o)
   filtered=safety.filter(o,requested,True);clip_count+=int(filtered.clipped);o,_,done,info=robot.step(filtered.action);states.append(o)
   if done:break
  distance=float(torch.linalg.vector_norm(o[:2]-o[4:]));successes.append(info["success"]);final_distances.append(distance)
  if example is None:example=torch.stack(states)
 fig,ax=plt.subplots(figsize=(5,5));ax.plot(example[:,0],example[:,1],marker="o",label="policy path");ax.scatter(example[0,4],example[0,5],marker="*",s=180,label="goal");ax.set(xlim=(-1,1),ylim=(-1,1),title="Safe adapter closed-loop rollout",xlabel="x",ylabel="y");ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/"robot_rollout.png",dpi=170);plt.close(fig)
 metrics={"dataset_version":c.dataset_version,"seed":c.seed,"episodes":64,"success_rate":sum(successes)/len(successes),"mean_final_distance":sum(final_distances)/len(final_distances),"safety_clip_count":clip_count,"external_hardware_commands_sent":False,"evaluation_entry_point":"python 12_physical_ai/02_robot_interface/evaluate.py"};(out/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2));return metrics
if __name__=="__main__":evaluate()
