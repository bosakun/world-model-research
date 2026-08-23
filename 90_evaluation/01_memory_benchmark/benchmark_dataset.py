from __future__ import annotations
import sys
from pathlib import Path
import torch
PARTIAL_ROOT=Path(__file__).resolve().parents[2]/"03_memory"/"02_partial_observation"
if str(PARTIAL_ROOT) not in sys.path:sys.path.append(str(PARTIAL_ROOT))
from partial_dataset import PartialObservationSequenceDataset  # noqa:E402
def make_dataset(sequences,length,seed):return PartialObservationSequenceDataset(sequences,length,seed)
def goal_labels(true_states):return (true_states[...,2]==3).long()
def alias_mask(observations):
 pairs=observations.reshape(observations.shape[0]//2,2,*observations.shape[1:]);same=(pairs[:,0]==pairs[:,1]).flatten(2).all(-1);return same[:,None].expand(-1,2,-1).reshape(observations.shape[0],observations.shape[1])
