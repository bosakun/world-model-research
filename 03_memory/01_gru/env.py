from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_DELTAS = np.asarray(
    [(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64
)
ACTION_NAMES = ("up", "down", "left", "right")


@dataclass
class GridState:
    row: int
    col: int


class FullyObservableGridWorld:
    """Deterministic square grid rendered as a full RGB observation."""

    def __init__(self, grid_size: int = 5, cell_size: int = 4, seed: int = 0):
        if grid_size < 2 or cell_size < 1:
            raise ValueError("grid_size must be >=2 and cell_size must be >=1")
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.goal = GridState(grid_size - 1, grid_size - 1)
        self.rng = np.random.default_rng(seed)
        self.state = GridState(0, 0)

    def reset(self, state: tuple[int, int] | None = None) -> np.ndarray:
        if state is None:
            index = int(self.rng.integers(0, self.grid_size**2 - 1))
            row, col = divmod(index, self.grid_size)
        else:
            row, col = state
            if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                raise ValueError("initial state is outside the grid")
        self.state = GridState(int(row), int(col))
        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, tuple[int, int]]]:
        if not 0 <= action < len(ACTION_DELTAS):
            raise ValueError(f"action must be in [0, {len(ACTION_DELTAS) - 1}]")
        delta_row, delta_col = ACTION_DELTAS[action]
        row = int(np.clip(self.state.row + delta_row, 0, self.grid_size - 1))
        col = int(np.clip(self.state.col + delta_col, 0, self.grid_size - 1))
        self.state = GridState(row, col)
        done = self.state == self.goal
        reward = 1.0 if done else 0.0
        return self.render(), reward, done, {"state": (row, col)}

    def render(self) -> np.ndarray:
        size = self.grid_size * self.cell_size
        image = np.full((3, size, size), 0.05, dtype=np.float32)
        for boundary in range(0, size, self.cell_size):
            image[:, boundary, :] = 0.18
            image[:, :, boundary] = 0.18
        self._paint_cell(image, self.goal, (0.1, 0.85, 0.2))
        self._paint_cell(image, self.state, (0.95, 0.15, 0.1))
        return image

    def _paint_cell(
        self, image: np.ndarray, state: GridState, color: tuple[float, float, float]
    ) -> None:
        row = state.row * self.cell_size
        col = state.col * self.cell_size
        inset = 1 if self.cell_size > 2 else 0
        image[
            :,
            row + inset : row + self.cell_size,
            col + inset : col + self.cell_size,
        ] = np.asarray(color, dtype=np.float32)[:, None, None]

