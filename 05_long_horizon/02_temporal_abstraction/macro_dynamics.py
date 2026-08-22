from __future__ import annotations

import torch
from torch import nn


class ActionChunkEncoder(nn.Module):
    def __init__(self, action_dim: int = 4, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gru = nn.GRU(action_dim, embedding_dim, batch_first=True)

    def forward(self, action_chunks: torch.Tensor) -> torch.Tensor:
        if action_chunks.ndim < 3:
            raise ValueError("action chunks must be [...,K,A]")
        leading = action_chunks.shape[:-2]
        flat = action_chunks.reshape(-1, action_chunks.shape[-2], action_chunks.shape[-1])
        _, hidden = self.gru(flat)
        return hidden[-1].reshape(*leading, self.embedding_dim)


class MacroDynamics(nn.Module):
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 4,
        action_embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.action_encoder = ActionChunkEncoder(action_dim, action_embedding_dim)
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, states: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        chunk_embeddings = self.action_encoder(action_chunks)
        return states + self.transition(torch.cat((states, chunk_embeddings), dim=-1))

    def rollout(self, initial_states: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        states = initial_states
        predictions = []
        for macro_index in range(action_chunks.shape[1]):
            states = self(states, action_chunks[:, macro_index])
            predictions.append(states)
        return torch.stack(predictions, dim=1)
