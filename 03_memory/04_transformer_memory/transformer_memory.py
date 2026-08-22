from __future__ import annotations

import torch
from torch import nn


class VisualEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, latent_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-3:] != (3, 20, 20):
            raise ValueError("encoder expects [...,3,20,20]")
        leading = observations.shape[:-3]
        latents = self.network(observations.reshape(-1, 3, 20, 20))
        return latents.reshape(*leading, -1)


class VisualDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 3 * 20 * 20),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        leading = latents.shape[:-1]
        images = self.network(latents.reshape(-1, latents.shape[-1]))
        return images.reshape(*leading, 3, 20, 20)


class CausalTransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, tokens: torch.Tensor, causal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.attention_norm(tokens)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feedforward(self.feedforward_norm(tokens))
        return tokens, weights


class TransformerMemoryDynamics(nn.Module):
    """Causal latent/action sequence model for next-latent prediction."""

    def __init__(
        self,
        latent_dim: int = 16,
        action_dim: int = 4,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        max_context: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.max_context = max_context
        self.token_projection = nn.Linear(latent_dim + action_dim, model_dim)
        self.position_embedding = nn.Embedding(max_context, model_dim)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(model_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.prediction_head = nn.Linear(model_dim, latent_dim)

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def tokenize(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or actions.ndim != 3 or latents.shape[:2] != actions.shape[:2]:
            raise ValueError("tokenize expects matching latents [B,T,Dz] and actions [B,T,Da]")
        steps = latents.shape[1]
        if steps > self.max_context:
            raise ValueError(f"sequence length {steps} exceeds max_context {self.max_context}")
        positions = torch.arange(steps, device=latents.device)
        return self.token_projection(torch.cat((latents, actions), dim=-1)) + self.position_embedding(
            positions
        ).unsqueeze(0)

    def forward(
        self, latents: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        tokens = self.tokenize(latents, actions)
        mask = self.causal_mask(tokens.shape[1], tokens.device)
        attention_maps = []
        for block in self.blocks:
            tokens, attention = block(tokens, mask)
            attention_maps.append(attention)
        context = self.output_norm(tokens)
        return {
            "predicted_next_latents": self.prediction_head(context),
            "context_tokens": context,
            "attention_maps": torch.stack(attention_maps, dim=0),
        }

    def rollout(
        self, initial_latent: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Autoregressively append predicted latents while retaining causal context."""
        latent_history: list[torch.Tensor] = []
        action_history: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        last_attention: torch.Tensor | None = None
        current_latent = initial_latent
        for time_index in range(actions.shape[1]):
            latent_history.append(current_latent)
            action_history.append(actions[:, time_index])
            context_latents = torch.stack(latent_history[-self.max_context :], dim=1)
            context_actions = torch.stack(action_history[-self.max_context :], dim=1)
            outputs = self(context_latents, context_actions)
            current_latent = outputs["predicted_next_latents"][:, -1]
            predictions.append(current_latent)
            last_attention = outputs["attention_maps"]
        if not predictions:
            empty = initial_latent[:, None, :][:, :0]
            return {"predicted_next_latents": empty, "last_attention_maps": None}
        return {
            "predicted_next_latents": torch.stack(predictions, dim=1),
            "last_attention_maps": last_attention,
        }


class TransformerMemoryWorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        action_dim: int = 4,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        max_context: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim)
        self.dynamics = TransformerMemoryDynamics(
            latent_dim,
            action_dim,
            model_dim,
            num_heads,
            num_layers,
            feedforward_dim,
            max_context,
            dropout,
        )
        self.goal_head = nn.Sequential(
            nn.Linear(latent_dim, model_dim), nn.ELU(), nn.Linear(model_dim, 10)
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("observations must contain one more step than actions")
        latents = self.encoder(observations)
        dynamics = self.dynamics(latents[:, :-1], actions)
        predicted_latents = dynamics["predicted_next_latents"]
        return {
            "latents": latents,
            "reconstructions": self.decoder(latents),
            "predicted_next_latents": predicted_latents,
            "predicted_next_observations": self.decoder(predicted_latents),
            "goal_logits": self.goal_head(predicted_latents),
            "context_tokens": dynamics["context_tokens"],
            "attention_maps": dynamics["attention_maps"],
        }

    def rollout(self, initial_observation: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        initial_latent = self.encoder(initial_observation)
        dynamics = self.dynamics.rollout(initial_latent, actions)
        predicted_latents = dynamics["predicted_next_latents"]
        return {
            "predicted_next_latents": predicted_latents,
            "predicted_next_observations": self.decoder(predicted_latents),
            "goal_logits": self.goal_head(predicted_latents),
            "last_attention_maps": dynamics["last_attention_maps"],
        }
