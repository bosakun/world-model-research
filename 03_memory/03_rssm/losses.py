"""Conventional loss entry point."""

from rssm_losses import (
    diagonal_gaussian_kl,
    goal_class_targets,
    observation_goal_evidence,
    rssm_loss,
)

__all__ = [
    "diagonal_gaussian_kl",
    "goal_class_targets",
    "observation_goal_evidence",
    "rssm_loss",
]
