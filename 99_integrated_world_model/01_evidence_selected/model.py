from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class State:
    deterministic: torch.Tensor
    stochastic: torch.Tensor


class GaussianHead(nn.Module):
    """Predict a diagonal Gaussian distribution over stochastic state z."""

    def __init__(self, input_dim: int, stochastic_dim: int):
        super().__init__()
        # Keep the stable ``net`` name because it is part of the checkpoint schema.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2 * stochastic_dim),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, unconstrained_std = self.net(inputs).chunk(2, dim=-1)
        return mean, F.softplus(unconstrained_std) + 0.1


class IntegratedWorldModel(nn.Module):
    """RSSM with correlated prior ensemble and task prediction heads."""

    def __init__(
        self,
        embedding_dim: int = 64,
        deterministic_dim: int = 64,
        stochastic_dim: int = 16,
        ensemble_size: int = 3,
    ):
        super().__init__()
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.h = deterministic_dim
        self.z = stochastic_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ELU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(800, embedding_dim),
            nn.ELU(),
        )
        self.cell = nn.GRUCell(stochastic_dim + 4, deterministic_dim)
        self.priors = nn.ModuleList(
            GaussianHead(deterministic_dim, stochastic_dim)
            for _ in range(ensemble_size)
        )
        self.posterior = GaussianHead(deterministic_dim + embedding_dim, stochastic_dim)

        feature_dim = deterministic_dim + stochastic_dim
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ELU(),
            nn.Linear(256, 3 * 20 * 20),
            nn.Sigmoid(),
        )
        self.reward = self._task_head(feature_dim, 1)
        self.value = self._task_head(feature_dim, 1)
        self.continuation = self._task_head(feature_dim, 1)
        self.goal = self._task_head(feature_dim, 2)

    @staticmethod
    def _task_head(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, output_dim),
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        leading_shape = observations.shape[:-3]
        flat = observations.reshape(-1, 3, 20, 20)
        return self.encoder(flat).reshape(*leading_shape, -1)

    def initial(self, batch_size: int, device: torch.device) -> State:
        return State(
            deterministic=torch.zeros(batch_size, self.deterministic_dim, device=device),
            stochastic=torch.zeros(batch_size, self.stochastic_dim, device=device),
        )

    @staticmethod
    def feature(state: State) -> torch.Tensor:
        return torch.cat((state.deterministic, state.stochastic), dim=-1)

    def observe(self, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Filter observations using posterior states over a complete sequence."""
        embeddings = self.encode(observations)
        state = self.initial(observations.shape[0], observations.device)
        deterministic_states = []
        stochastic_states = []
        prior_means = []
        prior_stds = []
        posterior_means = []
        posterior_stds = []

        for time in range(observations.shape[1]):
            if time > 0:
                recurrent_input = torch.cat((state.stochastic, actions[:, time - 1]), dim=-1)
                deterministic = self.cell(recurrent_input, state.deterministic)
                state = State(deterministic, state.stochastic)

            prior_parameters = [head(state.deterministic) for head in self.priors]
            posterior_input = torch.cat((state.deterministic, embeddings[:, time]), dim=-1)
            posterior_mean, posterior_std = self.posterior(posterior_input)

            # The mean path is deterministic for stable small-scale planning.
            state = State(state.deterministic, posterior_mean)
            deterministic_states.append(state.deterministic)
            stochastic_states.append(state.stochastic)
            prior_means.append(torch.stack([parameters[0] for parameters in prior_parameters]))
            prior_stds.append(torch.stack([parameters[1] for parameters in prior_parameters]))
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)

        deterministic = torch.stack(deterministic_states, dim=1)
        stochastic = torch.stack(stochastic_states, dim=1)
        feature = torch.cat((deterministic, stochastic), dim=-1)
        transition_feature = feature[:, 1:]

        return {
            "feature": feature,
            "h": deterministic,
            "z": stochastic,
            "prior_mean": torch.stack(prior_means, dim=2),
            "prior_std": torch.stack(prior_stds, dim=2),
            "post_mean": torch.stack(posterior_means, dim=1),
            "post_std": torch.stack(posterior_stds, dim=1),
            "reconstruction": self.decoder(feature).reshape(*feature.shape[:-1], 3, 20, 20),
            "reward": self.reward(transition_feature).squeeze(-1),
            "value": self.value(transition_feature).squeeze(-1),
            "continuation_logits": self.continuation(transition_feature).squeeze(-1),
            "goal_logits": self.goal(transition_feature),
        }

    def posterior_step(
        self,
        state: State,
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> State:
        recurrent_input = torch.cat((state.stochastic, action), dim=-1)
        deterministic = self.cell(recurrent_input, state.deterministic)
        posterior_input = torch.cat((deterministic, self.encode(observation)), dim=-1)
        posterior_mean, _ = self.posterior(posterior_input)
        return State(deterministic, posterior_mean)

    def prior_step(
        self,
        state: State,
        action: torch.Tensor,
        member: int,
    ) -> tuple[State, torch.Tensor]:
        recurrent_input = torch.cat((state.stochastic, action), dim=-1)
        deterministic = self.cell(recurrent_input, state.deterministic)
        prior_mean, prior_std = self.priors[member](deterministic)
        return State(deterministic, prior_mean), prior_std
