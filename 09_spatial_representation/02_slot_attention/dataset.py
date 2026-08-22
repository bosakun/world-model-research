from __future__ import annotations

import torch
from torch.utils.data import Dataset


def render_objects(positions: torch.Tensor, image_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    distance = ((grid[None, None] - positions[:, :, None, None]) ** 2).sum(dim=-1)
    objects = torch.exp(-distance / (2.0 * 0.18**2))
    background = torch.zeros(positions.shape[0], 1, image_size, image_size)
    images = torch.cat((objects[:, 0:1], objects[:, 1:2], background), dim=1).clamp(0.0, 1.0)
    background_score = (0.20 - objects.max(dim=1).values).clamp_min(0.0)
    labels = torch.stack((background_score, objects[:, 0], objects[:, 1]), dim=1).argmax(dim=1)
    masks = torch.nn.functional.one_hot(labels, num_classes=3).permute(0, 3, 1, 2).float()
    return images, masks


class TwoObjectImagesDataset(Dataset):
    def __init__(self, samples: int, seed: int, image_size: int = 16):
        generator = torch.Generator().manual_seed(seed)
        first = torch.empty(samples, 2).uniform_(-0.75, 0.75, generator=generator)
        angle = torch.empty(samples).uniform_(0.0, 2.0 * torch.pi, generator=generator)
        radius = torch.empty(samples).uniform_(0.45, 0.9, generator=generator)
        second = (first + radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)).clamp(-0.8, 0.8)
        self.positions = torch.stack((first, second), dim=1)
        self.images, self.masks = render_objects(self.positions, image_size)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "mask": self.masks[index], "position": self.positions[index]}
