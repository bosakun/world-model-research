from __future__ import annotations

import torch
import torch.nn.functional as F

from model import State


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.log(prior_std / posterior_std)
        + (
            posterior_std.square()
            + (posterior_mean - prior_mean).square()
        )
        / (2.0 * prior_std.square())
        - 0.5
    ).sum(dim=-1)


def integrated_loss(model, batch, config):
    outputs = model.observe(batch["observations"], batch["actions"])
    ensemble_size = outputs["prior_mean"].shape[0]

    posterior_mean = outputs["post_mean"][None]
    posterior_std = outputs["post_std"][None]
    kl_per_state = diagonal_gaussian_kl(
        posterior_mean,
        posterior_std,
        outputs["prior_mean"],
        outputs["prior_std"],
    )
    kl = torch.clamp(kl_per_state, min=config.free_nats).mean()
    reconstruction = F.mse_loss(outputs["reconstruction"], batch["observations"])

    goal_labels = (batch["true_states"][:, 1:, 2] == 3).long()
    posterior_reward = F.mse_loss(outputs["reward"], batch["rewards"])
    posterior_value = F.mse_loss(outputs["value"], batch["values"])
    posterior_continuation = F.binary_cross_entropy_with_logits(
        outputs["continuation_logits"], batch["continuations"]
    )
    posterior_goal = F.cross_entropy(
        outputs["goal_logits"].reshape(-1, 2), goal_labels.reshape(-1)
    )

    # Planning never has future observations, so supervise task heads on prior states too.
    prior_h = outputs["h"][:, 1:][None].expand(ensemble_size, -1, -1, -1)
    prior_feature = torch.cat((prior_h, outputs["prior_mean"][:, :, 1:]), dim=-1)
    prior_reward = F.mse_loss(
        model.reward(prior_feature).squeeze(-1),
        batch["rewards"][None].expand(ensemble_size, -1, -1),
    )
    prior_value = F.mse_loss(
        model.value(prior_feature).squeeze(-1),
        batch["values"][None].expand(ensemble_size, -1, -1),
    )
    prior_continuation = F.binary_cross_entropy_with_logits(
        model.continuation(prior_feature).squeeze(-1),
        batch["continuations"][None].expand(ensemble_size, -1, -1),
    )
    expanded_goals = goal_labels[None].expand(ensemble_size, -1, -1)
    prior_goal = F.cross_entropy(
        model.goal(prior_feature).reshape(-1, 2), expanded_goals.reshape(-1)
    )

    reward = (posterior_reward + prior_reward) / 2.0
    value = (posterior_value + prior_value) / 2.0
    continuation = (posterior_continuation + prior_continuation) / 2.0
    goal = (posterior_goal + prior_goal) / 2.0

    overshooting_terms = []
    sequence_length = batch["actions"].shape[1]
    for start in range(sequence_length):
        state = State(outputs["h"][:, start], outputs["z"][:, start])
        max_distance = min(config.overshooting_distance, sequence_length - start)
        for distance in range(1, max_distance + 1):
            action = batch["actions"][:, start + distance - 1]
            state, _ = model.prior_step(state, action, member=0)
            target = outputs["z"][:, start + distance].detach()
            overshooting_terms.append(F.mse_loss(state.stochastic, target))
    overshooting = torch.stack(overshooting_terms).mean()

    total = (
        config.reconstruction_weight * reconstruction
        + config.reward_weight * reward
        + config.value_weight * value
        + config.continuation_weight * continuation
        + config.goal_weight * goal
        + config.kl_weight * kl
        + config.overshooting_weight * overshooting
    )
    losses = {
        "total": total,
        "reconstruction": reconstruction,
        "reward": reward,
        "value": value,
        "continuation": continuation,
        "goal": goal,
        "kl": kl,
        "overshooting": overshooting,
    }
    return losses, outputs
