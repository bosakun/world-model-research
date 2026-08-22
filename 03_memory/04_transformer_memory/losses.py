"""Conventional loss entry point."""

from transformer_losses import (
    goal_class_targets,
    transformer_world_model_loss,
    weighted_image_mse,
)

__all__ = ["goal_class_targets", "transformer_world_model_loss", "weighted_image_mse"]
