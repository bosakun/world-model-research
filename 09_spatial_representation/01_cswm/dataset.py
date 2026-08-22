from __future__ import annotations

import torch
from torch.utils.data import Dataset


def relational_transition(positions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    displacement = 0.14 * torch.tanh(actions)
    difference = positions[:, 0] - positions[:, 1]
    distance = torch.linalg.vector_norm(difference, dim=-1, keepdim=True).clamp_min(1e-4)
    close = (distance < 0.65).to(positions.dtype)
    repulsion = 0.035 * close * difference / distance
    interaction = torch.stack((repulsion, -repulsion), dim=1)
    return (positions + displacement + interaction).clamp(-0.9, 0.9)


def render_objects(positions: torch.Tensor, image_size: int = 16) -> torch.Tensor:
    coordinates = torch.linspace(-1.0, 1.0, image_size, device=positions.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    squared_distance = ((grid[None, None] - positions[:, :, None, None]) ** 2).sum(dim=-1)
    blobs = torch.exp(-squared_distance / (2.0 * 0.11**2))
    background = torch.zeros(positions.shape[0], 1, image_size, image_size, device=positions.device)
    return torch.cat((blobs, background), dim=1).clamp(0.0, 1.0)


class TwoObjectTransitionDataset(Dataset):
    def __init__(self, samples: int, seed: int, image_size: int = 16):
        generator = torch.Generator().manual_seed(seed)
        positions = torch.empty(samples, 2, 2).uniform_(-0.75, 0.75, generator=generator)
        actions = torch.empty(samples, 2, 2).uniform_(-1.0, 1.0, generator=generator)
        next_positions = relational_transition(positions, actions)
        self.images = render_objects(positions, image_size)
        self.next_images = render_objects(next_positions, image_size)
        self.actions = actions
        self.positions = positions
        self.next_positions = next_positions

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index], "action": self.actions[index], "next_image": self.next_images[index],
            "position": self.positions[index], "next_position": self.next_positions[index],
        }
