from __future__ import annotations

import sys
from pathlib import Path

import torch


OVERSHOOTING_EXPERIMENT = Path(__file__).resolve().parents[1] / "01_latent_overshooting"
if str(OVERSHOOTING_EXPERIMENT) not in sys.path:
    sys.path.append(str(OVERSHOOTING_EXPERIMENT))

from sequence_dataset import ControlledOscillatorSequenceDataset  # noqa: E402


def chunk_sequences(
    states: torch.Tensor, actions: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.shape[1] % chunk_size:
        raise ValueError("action horizon must be divisible by chunk_size")
    macro_steps = actions.shape[1] // chunk_size
    boundary_states = states[:, ::chunk_size]
    action_chunks = actions.reshape(actions.shape[0], macro_steps, chunk_size, actions.shape[-1])
    if boundary_states.shape[1] != macro_steps + 1:
        raise ValueError("states must contain one more primitive step than actions")
    return boundary_states, action_chunks


class MacroSequenceDataset(ControlledOscillatorSequenceDataset):
    def __init__(self, num_sequences: int, sequence_length: int, chunk_size: int, seed: int):
        super().__init__(num_sequences, sequence_length, seed)
        self.boundary_states, self.action_chunks = chunk_sequences(
            self.states, self.actions, chunk_size
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        item["boundary_states"] = self.boundary_states[index]
        item["action_chunks"] = self.action_chunks[index]
        return item
