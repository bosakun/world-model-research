from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


class ColorObjectEncoder(nn.Module):
    """Known color-channel binding isolates C-SWM relational dynamics from discovery."""

    def __init__(self, num_objects: int = 2, slot_dim: int = 8, image_size: int = 16):
        super().__init__()
        self.num_objects = num_objects
        self.shared = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(16 * image_size * image_size, 64), nn.ReLU(), nn.Linear(64, slot_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        slots = []
        for object_index in range(self.num_objects):
            slots.append(self.project(self.shared(images[:, object_index : object_index + 1])))
        return torch.stack(slots, dim=1)


class RelationalTransition(nn.Module):
    def __init__(self, slot_dim: int = 8, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.edge = mlp(2 * slot_dim, hidden_dim, hidden_dim)
        self.node = mlp(slot_dim + action_dim + hidden_dim, hidden_dim, slot_dim)

    def forward(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        num_objects = slots.shape[1]
        effects = []
        for receiver in range(num_objects):
            aggregate = torch.zeros(slots.shape[0], self.edge[-1].out_features, device=slots.device)
            for sender in range(num_objects):
                if sender != receiver:
                    aggregate = aggregate + self.edge(torch.cat((slots[:, receiver], slots[:, sender]), dim=-1))
            delta = self.node(torch.cat((slots[:, receiver], actions[:, receiver], aggregate), dim=-1))
            effects.append(slots[:, receiver] + delta)
        return torch.stack(effects, dim=1)


class ContrastiveStructuredWorldModel(nn.Module):
    def __init__(self, num_objects: int = 2, action_dim: int = 2, slot_dim: int = 8, hidden_dim: int = 64, image_size: int = 16):
        super().__init__()
        self.encoder = ColorObjectEncoder(num_objects, slot_dim, image_size)
        self.transition = RelationalTransition(slot_dim, action_dim, hidden_dim)

    def predict(self, image: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.encoder(image)
        return slots, self.transition(slots, action)
