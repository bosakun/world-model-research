from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
