from __future__ import annotations

import torch
from torch import nn


class VisualEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.to_latent = nn.Linear(16 * 5 * 5, latent_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-3:] != (3, 20, 20):
            raise ValueError("encoder expects observations shaped [..., 3, 20, 20]")
        leading_shape = observations.shape[:-3]
        flat = observations.reshape(-1, 3, 20, 20)
        # Bound scale while encoder and detached-target dynamics are jointly learned.
        latent = torch.tanh(self.to_latent(self.features(flat)))
        return latent.reshape(*leading_shape, -1)


class VisualDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.image = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 20 * 20),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        leading_shape = latents.shape[:-1]
        flat = latents.reshape(-1, latents.shape[-1])
        image = self.image(flat).reshape(-1, 3, 20, 20)
        return image.reshape(*leading_shape, 3, 20, 20)


class GRUDynamics(nn.Module):
    """Action-conditioned deterministic recurrent latent transition."""

    def __init__(self, latent_dim: int = 16, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )

    def initial_hidden(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self, latent: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2 or action.ndim != 2 or hidden.ndim != 2:
            raise ValueError("step expects [B,D] latent, action, and hidden tensors")
        next_hidden = self.cell(torch.cat((latent, action), dim=-1), hidden)
        predicted_next_latent = self.prediction_head(next_hidden)
        return predicted_next_latent, next_hidden

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced sequence transition.

        latents: [B,T,Dz], actions: [B,T,Da], hidden: [B,Dh]
        returns predictions [B,T,Dz], hidden sequence [B,T,Dh], final hidden [B,Dh]
        """
        batch_size, steps, _ = latents.shape
        if actions.shape[:2] != (batch_size, steps):
            raise ValueError("latents and actions must share [B,T]")
        if hidden is None:
            hidden = self.initial_hidden(batch_size, device=latents.device)
        predictions, hidden_states = [], []
        for step in range(steps):
            prediction, hidden = self.step(latents[:, step], actions[:, step], hidden)
            predictions.append(prediction)
            hidden_states.append(hidden)
        return torch.stack(predictions, 1), torch.stack(hidden_states, 1), hidden

    def rollout(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive rollout using each prediction as the next input."""
        if hidden is None:
            hidden = self.initial_hidden(initial_latent.shape[0], device=initial_latent.device)
        latent = initial_latent
        predictions, hidden_states = [], []
        for step in range(actions.shape[1]):
            latent, hidden = self.step(latent, actions[:, step], hidden)
            predictions.append(latent)
            hidden_states.append(hidden)
        return torch.stack(predictions, 1), torch.stack(hidden_states, 1), hidden


class GRUWorldModel(nn.Module):
    def __init__(self, latent_dim: int = 16, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim)
        self.dynamics = GRUDynamics(latent_dim, action_dim, hidden_dim)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self.encoder(observations)
        reconstructions = self.decoder(latents)
        predicted, hidden_states, final_hidden = self.dynamics(latents[:, :-1], actions)
        return {
            "latents": latents,
            "reconstructions": reconstructions,
            "predicted_next_latents": predicted,
            "hidden_states": hidden_states,
            "final_hidden": final_hidden,
        }
