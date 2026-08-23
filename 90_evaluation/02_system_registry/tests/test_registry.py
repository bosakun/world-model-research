from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from build_registry import discover  # noqa:E402
def test_registry_finds_cross_phase_evidence():
 rows=discover();experiments={r["experiment"] for r in rows};assert "04_uncertainty/01_probabilistic_dynamics" in experiments;assert "07_planning/03_mpc" in experiments;assert "12_physical_ai/02_robot_interface" in experiments;assert "99_integrated_world_model/01_evidence_selected" in experiments
def test_every_registered_experiment_has_entry_point_and_dataset():
 rows=discover();assert rows;assert all(r["evaluation_entry_point"]!="unrecorded" for r in rows);assert all(r["dataset_version"]!="unrecorded" for r in rows)
