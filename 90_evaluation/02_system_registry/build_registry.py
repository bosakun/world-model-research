from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def discover(root:Path=ROOT):
 rows=[]
 for path in sorted(root.rglob("outputs/evaluation_metrics.json")):
  if HERE in path.parents:continue
  payload=json.loads(path.read_text());experiment=str(path.parent.parent.relative_to(root));phase=experiment.split("/")[0];rows.append({"phase":phase,"experiment":experiment,"metrics_path":str(path.relative_to(root)),"dataset_version":payload.get("dataset_version","unrecorded"),"seed":payload.get("seed","unrecorded"),"evaluation_entry_point":payload.get("evaluation_entry_point","unrecorded"),"metrics":payload})
 return rows
def build(out:Path=HERE/"outputs"):
 rows=discover();out.mkdir(parents=True,exist_ok=True);flat=[]
 for row in rows:
  numeric={k:v for k,v in row["metrics"].items() if isinstance(v,(int,float,bool))};flat.append({k:v for k,v in row.items() if k!="metrics"}|{"numeric_metrics":json.dumps(numeric,sort_keys=True)})
 with (out/"experiment_registry.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=flat[0].keys(),lineterminator="\n");w.writeheader();w.writerows(flat)
 phases={}
 for row in rows:phases[row["phase"]]=phases.get(row["phase"],0)+1
 payload={"registry_version":1,"experiment_count":len(rows),"phase_counts":phases,"experiments":rows,"limitations":["Metrics are heterogeneous and are not ranked across tasks.","Missing multiple-seed evidence remains explicit; only the Phase 90 memory benchmark is matched multi-seed.","Parameter bytes are a lower bound, not measured peak process memory."]};(out/"experiment_registry.json").write_text(json.dumps(payload,indent=2)+"\n")
 fig,ax=plt.subplots(figsize=(10,4));ax.bar(phases.keys(),phases.values());ax.set(title="Experiments with executable evaluation evidence",xlabel="phase",ylabel="count");ax.tick_params(axis="x",rotation=45);ax.grid(axis="y",alpha=.3);fig.tight_layout();fig.savefig(out/"evaluation_coverage.png",dpi=170);plt.close(fig);print(json.dumps({"experiment_count":len(rows),"phase_counts":phases},indent=2));return payload
if __name__=="__main__":build()
