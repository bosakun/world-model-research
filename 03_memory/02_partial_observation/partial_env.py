from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_DELTAS = np.asarray([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)
ACTION_NAMES = ("up", "down", "left", "right")
UP, DOWN, LEFT, RIGHT = range(4)

UNKNOWN_COLOR = (0.12, 0.12, 0.35)
EMPTY_COLOR = (0.05, 0.05, 0.05)
GRID_COLOR = (0.18, 0.18, 0.18)
AGENT_COLOR = (0.95, 0.15, 0.10)
GOAL_COLOR = (0.10, 0.85, 0.20)


@dataclass(frozen=True)
class WorldState:
    agent_row: int
    agent_col: int
    goal_row: int
    goal_col: int

    @property
    def agent(self) -> tuple[int, int]:
        return self.agent_row, self.agent_col

    @property
    def goal(self) -> tuple[int, int]:
        return self.goal_row, self.goal_col


class PartialObservationGridWorld:
    """A deterministic grid world with an agent-centred local observation function.

    The environment owns the complete state. The model receives only a 3x3 local
    view, drawn at the centre of a fixed 5x5 image canvas; all other cells are
    explicitly unknown. Full world renders are evaluation-only information.
    """

    def __init__(self, grid_size: int = 5, cell_size: int = 4, view_radius: int = 1):
        if grid_size != 5 or cell_size != 4:
            raise ValueError("this educational adapter fixes a 5x5 world rendered at 20x20")
        if view_radius != 1:
            raise ValueError("this experiment fixes an agent-centred 3x3 view (view_radius=1)")
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.view_radius = view_radius
        self.state = WorldState(2, 2, 2, 3)

    def reset(
        self,
        agent: tuple[int, int] = (2, 2),
        goal: tuple[int, int] = (2, 3),
    ) -> np.ndarray:
        self._validate_coordinate(agent, "agent")
        self._validate_coordinate(goal, "goal")
        self.state = WorldState(*agent, *goal)
        return self.render_partial_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if not 0 <= action < len(ACTION_DELTAS):
            raise ValueError(f"action must be in [0, {len(ACTION_DELTAS) - 1}]")
        delta_row, delta_col = ACTION_DELTAS[action]
        next_agent = (
            int(np.clip(self.state.agent_row + delta_row, 0, self.grid_size - 1)),
            int(np.clip(self.state.agent_col + delta_col, 0, self.grid_size - 1)),
        )
        self.state = WorldState(*next_agent, self.state.goal_row, self.state.goal_col)
        done = self.state.agent == self.state.goal
        return self.render_partial_observation(), float(done), done, self.evaluation_info()

    def visible_world_coordinates(self) -> set[tuple[int, int]]:
        return {
            (row, col)
            for row in range(self.state.agent_row - self.view_radius, self.state.agent_row + self.view_radius + 1)
            for col in range(self.state.agent_col - self.view_radius, self.state.agent_col + self.view_radius + 1)
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size
        }

    def goal_is_visible(self) -> bool:
        return self.state.goal in self.visible_world_coordinates()

    def evaluation_info(self) -> dict[str, object]:
        """Full state is intentionally supplied only through this evaluation API."""
        return {
            "true_state": self.true_state_array(),
            "goal_visible": self.goal_is_visible(),
            "visible_world_coordinates": tuple(sorted(self.visible_world_coordinates())),
        }

    def true_state_array(self) -> np.ndarray:
        return np.asarray(
            [self.state.agent_row, self.state.agent_col, self.state.goal_row, self.state.goal_col],
            dtype=np.int64,
        )

    def render_full_world(self) -> np.ndarray:
        image = self._empty_grid_canvas(EMPTY_COLOR)
        self._paint_world_cell(image, self.state.goal, GOAL_COLOR)
        self._paint_world_cell(image, self.state.agent, AGENT_COLOR)
        return image

    def render_partial_observation(self) -> np.ndarray:
        """Render only local contents in the central 3x3 canvas region.

        A displayed cell `(display_row, display_col)` maps to world coordinate
        `(agent_row + display_row - 2, agent_col + display_col - 2)` for display
        rows/columns 1..3. Thus the agent is always shown in display cell (2,2).
        """
        image = self._empty_grid_canvas(UNKNOWN_COLOR)
        centre = self.grid_size // 2
        for local_row in range(-self.view_radius, self.view_radius + 1):
            for local_col in range(-self.view_radius, self.view_radius + 1):
                world = self.state.agent_row + local_row, self.state.agent_col + local_col
                display = centre + local_row, centre + local_col
                self._paint_display_cell(image, display, EMPTY_COLOR)
                if not self._in_bounds(world):
                    continue
                if world == self.state.goal:
                    self._paint_display_cell(image, display, GOAL_COLOR)
                if world == self.state.agent:
                    self._paint_display_cell(image, display, AGENT_COLOR)
        return image

    def _empty_grid_canvas(self, cell_color: tuple[float, float, float]) -> np.ndarray:
        image = np.empty(
            (3, self.grid_size * self.cell_size, self.grid_size * self.cell_size), dtype=np.float32
        )
        image[:] = np.asarray(cell_color, dtype=np.float32)[:, None, None]
        for boundary in range(0, self.grid_size * self.cell_size, self.cell_size):
            image[:, boundary, :] = np.asarray(GRID_COLOR, dtype=np.float32)[:, None]
            image[:, :, boundary] = np.asarray(GRID_COLOR, dtype=np.float32)[:, None]
        return image

    def _paint_world_cell(
        self, image: np.ndarray, world: tuple[int, int], color: tuple[float, float, float]
    ) -> None:
        self._paint_display_cell(image, world, color)

    def _paint_display_cell(
        self, image: np.ndarray, display: tuple[int, int], color: tuple[float, float, float]
    ) -> None:
        row, col = display
        start_row, start_col = row * self.cell_size, col * self.cell_size
        image[:, start_row + 1 : start_row + self.cell_size, start_col + 1 : start_col + self.cell_size] = np.asarray(
            color, dtype=np.float32
        )[:, None, None]

    def _validate_coordinate(self, coordinate: tuple[int, int], name: str) -> None:
        if not self._in_bounds(coordinate):
            raise ValueError(f"{name} must be inside the {self.grid_size}x{self.grid_size} grid")

    def _in_bounds(self, coordinate: tuple[int, int]) -> bool:
        row, col = coordinate
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
