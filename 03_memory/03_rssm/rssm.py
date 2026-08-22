from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DiagonalGaussian:
    mean: torch.Tensor
    std: torch.Tensor

    def sample(self, stochastic: bool = True) -> torch.Tensor:
        if not stochastic:
            return self.mean
        noise = torch.randn_like(self.std)
        return self.mean + self.std * noise


@dataclass
class RSSMState:
    deterministic: torch.Tensor
    stochastic: torch.Tensor


class ObservationEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, embedding_dim),
            nn.ELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-3:] != (3, 20, 20):
            raise ValueError("encoder expects [...,3,20,20]")
        leading = observations.shape[:-3]
        embeddings = self.network(observations.reshape(-1, 3, 20, 20))
        return embeddings.reshape(*leading, -1)


class ObservationDecoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ELU(),
            nn.Linear(256, 3 * 20 * 20),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        leading = features.shape[:-1]
        images = self.network(features.reshape(-1, features.shape[-1]))
        return images.reshape(*leading, 3, 20, 20)


class GaussianHead(nn.Module):
    def __init__(self, input_dim: int, stochastic_dim: int, hidden_dim: int, min_std: float):
        super().__init__()
        self.stochastic_dim = stochastic_dim
        self.min_std = min_std
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * stochastic_dim)
        )

    def forward(self, inputs: torch.Tensor) -> DiagonalGaussian:
        mean, raw_std = self.network(inputs).chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return DiagonalGaussian(mean, std)


class RecurrentStateSpaceModel(nn.Module):
    """Small continuous Gaussian RSSM following the PlaNet state factorization."""

    def __init__(
        self,
        action_dim: int = 4,
        observation_embedding_dim: int = 64,
        deterministic_dim: int = 64,
        stochastic_dim: int = 16,
        hidden_mlp_dim: int = 64,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.encoder = ObservationEncoder(observation_embedding_dim)
        self.recurrent_transition = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)
        self.prior = GaussianHead(deterministic_dim, stochastic_dim, hidden_mlp_dim, min_std)
        self.posterior = GaussianHead(
            deterministic_dim + observation_embedding_dim,
            stochastic_dim,
            hidden_mlp_dim,
            min_std,
        )
        feature_dim = deterministic_dim + stochastic_dim
        self.decoder = ObservationDecoder(feature_dim)
        self.goal_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_mlp_dim), nn.ELU(), nn.Linear(hidden_mlp_dim, 10)
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deterministic=torch.zeros(batch_size, self.deterministic_dim, device=device),
            stochastic=torch.zeros(batch_size, self.stochastic_dim, device=device),
        )

    def transition(
        self, previous: RSSMState, action: torch.Tensor
    ) -> tuple[torch.Tensor, DiagonalGaussian]:
        deterministic = self.recurrent_transition(
            torch.cat((previous.stochastic, action), dim=-1), previous.deterministic
        )
        return deterministic, self.prior(deterministic)

    def infer_posterior(
        self, deterministic: torch.Tensor, embedding: torch.Tensor
    ) -> DiagonalGaussian:
        return self.posterior(torch.cat((deterministic, embedding), dim=-1))

    def observe(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        stochastic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Filter a sequence with posterior samples whenever observations exist.

        observations [B,T+1,C,H,W], actions [B,T,A]. The t=0 posterior uses
        zero deterministic state. For t>0, h_t is updated from z_{t-1},a_{t-1}.
        """
        if observations.ndim != 5 or actions.ndim != 3:
            raise ValueError("observe expects observations [B,T+1,C,H,W] and actions [B,T,A]")
        batch_size, observation_steps = observations.shape[:2]
        if actions.shape[:2] != (batch_size, observation_steps - 1):
            raise ValueError("actions must contain one fewer time step than observations")
        embeddings = self.encoder(observations)
        state = self.initial_state(batch_size, observations.device)
        deterministic_states, stochastic_states = [], []
        prior_means, prior_stds, posterior_means, posterior_stds = [], [], [], []

        for time_index in range(observation_steps):
            if time_index == 0:
                deterministic = state.deterministic
                prior = self.prior(deterministic)
            else:
                deterministic, prior = self.transition(state, actions[:, time_index - 1])
            posterior = self.infer_posterior(deterministic, embeddings[:, time_index])
            stochastic_state = posterior.sample(stochastic)
            state = RSSMState(deterministic, stochastic_state)
            deterministic_states.append(deterministic)
            stochastic_states.append(stochastic_state)
            prior_means.append(prior.mean)
            prior_stds.append(prior.std)
            posterior_means.append(posterior.mean)
            posterior_stds.append(posterior.std)

        outputs = {
            "embeddings": embeddings,
            "deterministic_states": torch.stack(deterministic_states, dim=1),
            "stochastic_states": torch.stack(stochastic_states, dim=1),
            "prior_means": torch.stack(prior_means, dim=1),
            "prior_stds": torch.stack(prior_stds, dim=1),
            "posterior_means": torch.stack(posterior_means, dim=1),
            "posterior_stds": torch.stack(posterior_stds, dim=1),
        }
        outputs["reconstructions"] = self.decode(
            outputs["deterministic_states"], outputs["stochastic_states"]
        )
        outputs["goal_logits"] = self.predict_goal(
            outputs["deterministic_states"], outputs["stochastic_states"]
        )
        return outputs

    def imagine(
        self,
        initial_state: RSSMState,
        actions: torch.Tensor,
        stochastic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Roll forward from the prior only; no future observation is consumed."""
        state = initial_state
        deterministic_states, stochastic_states, prior_means, prior_stds = [], [], [], []
        for time_index in range(actions.shape[1]):
            deterministic, prior = self.transition(state, actions[:, time_index])
            stochastic_state = prior.sample(stochastic)
            state = RSSMState(deterministic, stochastic_state)
            deterministic_states.append(deterministic)
            stochastic_states.append(stochastic_state)
            prior_means.append(prior.mean)
            prior_stds.append(prior.std)
        outputs = {
            "deterministic_states": torch.stack(deterministic_states, dim=1),
            "stochastic_states": torch.stack(stochastic_states, dim=1),
            "prior_means": torch.stack(prior_means, dim=1),
            "prior_stds": torch.stack(prior_stds, dim=1),
        }
        outputs["observations"] = self.decode(
            outputs["deterministic_states"], outputs["stochastic_states"]
        )
        outputs["goal_logits"] = self.predict_goal(
            outputs["deterministic_states"], outputs["stochastic_states"]
        )
        return outputs

    def decode(self, deterministic: torch.Tensor, stochastic: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((deterministic, stochastic), dim=-1))

    def predict_goal(self, deterministic: torch.Tensor, stochastic: torch.Tensor) -> torch.Tensor:
        return self.goal_head(torch.cat((deterministic, stochastic), dim=-1))
