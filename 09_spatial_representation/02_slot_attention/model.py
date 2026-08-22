from __future__ import annotations

import torch
from torch import nn


def coordinate_grid(size: int) -> torch.Tensor:
    values = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx, yy, 1.0 - xx, 1.0 - yy), dim=-1)


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int = 3, input_dim: int = 32, slot_dim: int = 32, iterations: int = 3):
        super().__init__()
        self.num_slots, self.slot_dim, self.iterations = num_slots, slot_dim, iterations
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.norm_inputs = nn.LayerNorm(input_dim); self.norm_slots = nn.LayerNorm(slot_dim); self.norm_mlp = nn.LayerNorm(slot_dim)
        self.key = nn.Linear(input_dim, slot_dim, bias=False); self.value = nn.Linear(input_dim, slot_dim, bias=False)
        self.query = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, 64), nn.ReLU(), nn.Linear(64, slot_dim))

    def forward(self, inputs: torch.Tensor, stochastic_initialization: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = inputs.shape[0]
        stochastic = self.training if stochastic_initialization is None else stochastic_initialization
        slots = self.slot_mu.expand(batch, self.num_slots, -1)
        if stochastic:
            slots = slots + self.slot_log_sigma.exp().expand_as(slots) * torch.randn_like(slots)
        inputs = self.norm_inputs(inputs); keys = self.key(inputs); values = self.value(inputs)
        attention = None
        for _ in range(self.iterations):
            previous = slots
            queries = self.query(self.norm_slots(slots)) * self.slot_dim**-0.5
            logits = torch.einsum("bnd,bkd->bnk", keys, queries)
            attention = logits.softmax(dim=-1) + 1e-8
            weights = attention / attention.sum(dim=1, keepdim=True)
            updates = torch.einsum("bnk,bnd->bkd", weights, values)
            slots = self.gru(updates.reshape(-1, self.slot_dim), previous.reshape(-1, self.slot_dim)).view(batch, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attention


class SlotAttentionAutoencoder(nn.Module):
    def __init__(self, image_size: int = 16, num_slots: int = 3, slot_dim: int = 32, iterations: int = 3):
        super().__init__(); self.image_size = image_size; self.num_slots = num_slots
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.Conv2d(32, 32, 5, padding=2), nn.ReLU())
        self.encoder_position = nn.Linear(4, 32)
        self.encoder_mlp = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        self.slot_attention = SlotAttention(num_slots, 32, slot_dim, iterations)
        self.decoder_position = nn.Linear(4, slot_dim)
        self.decoder = nn.Sequential(nn.Linear(slot_dim, 64), nn.ReLU(), nn.Linear(64, 4))
        self.register_buffer("grid", coordinate_grid(image_size), persistent=False)

    def forward(self, images: torch.Tensor, stochastic_initialization: bool | None = None) -> dict[str, torch.Tensor]:
        features = self.encoder(images).permute(0, 2, 3, 1)
        features = features + self.encoder_position(self.grid)
        tokens = self.encoder_mlp(features.reshape(images.shape[0], -1, 32))
        slots, attention = self.slot_attention(tokens, stochastic_initialization)
        spatial = slots[:, :, None, None, :] + self.decoder_position(self.grid)[None, None]
        decoded = self.decoder(spatial)
        slot_rgb = torch.sigmoid(decoded[..., :3]).permute(0, 1, 4, 2, 3)
        mask_logits = decoded[..., 3].unsqueeze(2)
        masks = mask_logits.softmax(dim=1)
        reconstruction = (slot_rgb * masks).sum(dim=1)
        return {"reconstruction": reconstruction, "slots": slots, "masks": masks,
                "slot_rgb": slot_rgb, "attention": attention}
