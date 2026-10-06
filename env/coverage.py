"""Coverage-grid foundation.

The coverage grid tracks which parts of the disaster zone have been
explored. It is defined over the horizontal ``x``/``z`` plane of the
y-up world: columns run along ``x``, rows along ``z``. Cells are
identified by a flat integer index ``row * cols + col``; this is the
representation used by the WebSocket ``newly_visited_cells`` field.

Week 1 provides the grid, visited-cell bookkeeping, explored
percentage, and local-coverage queries. Heatmap rendering and
curriculum-driven coverage goals are later work.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class CoverageGrid:
    """Boolean visited-cell grid over the horizontal plane."""

    def __init__(self, world_size: Sequence[float], cell_size: float = 5.0) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        self.cell_size = float(cell_size)
        self.cols = max(1, int(math.ceil(float(world_size[0]) / self.cell_size)))
        self.rows = max(1, int(math.ceil(float(world_size[2]) / self.cell_size)))
        self.visited = np.zeros((self.rows, self.cols), dtype=bool)

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.rows * self.cols

    @property
    def visited_count(self) -> int:
        """Number of visited cells."""
        return int(np.count_nonzero(self.visited))

    def reset(self) -> None:
        """Clear all visited cells."""
        self.visited.fill(False)

    def cell_coords(self, x: float, z: float) -> tuple[int, int]:
        """Map a world ``(x, z)`` position to ``(row, col)``, clamped in-bounds."""
        col = int(np.clip(math.floor(x / self.cell_size), 0, self.cols - 1))
        row = int(np.clip(math.floor(z / self.cell_size), 0, self.rows - 1))
        return row, col

    def cell_index(self, x: float, z: float) -> int:
        """Flat index of the cell containing ``(x, z)``."""
        row, col = self.cell_coords(x, z)
        return row * self.cols + col

    def mark(self, x: float, z: float) -> list[int]:
        """Mark the cell containing ``(x, z)``.

        Returns:
            Flat indices that changed from unvisited to visited
            (empty when the cell was already visited).
        """
        row, col = self.cell_coords(x, z)
        if self.visited[row, col]:
            return []
        self.visited[row, col] = True
        return [row * self.cols + col]

    def mark_segment(
        self, x0: float, z0: float, x1: float, z1: float, samples_per_cell: int = 2
    ) -> list[int]:
        """Mark every cell touched by a straight move, without duplicates.

        Points are sampled along the segment at half-cell intervals so
        fast movement cannot skip cells.
        """
        distance = math.hypot(x1 - x0, z1 - z0)
        step_length = self.cell_size / max(1, samples_per_cell)
        sample_count = max(1, int(math.ceil(distance / step_length)))
        newly_visited: list[int] = []
        seen: set[int] = set()
        for i in range(sample_count + 1):
            t = i / sample_count
            x = x0 + (x1 - x0) * t
            z = z0 + (z1 - z0) * t
            for index in self.mark(x, z):
                if index not in seen:
                    seen.add(index)
                    newly_visited.append(index)
        return newly_visited

    def explored_pct(self) -> float:
        """Percentage of the grid that has been visited, in ``[0, 100]``."""
        return 100.0 * self.visited_count / float(self.total_cells)

    def local_fraction(self, x: float, z: float, radius: float) -> float:
        """Fraction of cells whose centers lie within ``radius`` of ``(x, z)``.

        Returns 1.0 when no cell centers fall inside the radius so the
        value stays within the observation bounds for edge positions.
        """
        row_near = int((z - radius) // self.cell_size)
        row_far = int((z + radius) // self.cell_size)
        col_near = int((x - radius) // self.cell_size)
        col_far = int((x + radius) // self.cell_size)
        radius_squared = float(radius) * float(radius)
        total = 0
        visited = 0
        for row in range(max(0, row_near), min(self.rows - 1, row_far) + 1):
            for col in range(max(0, col_near), min(self.cols - 1, col_far) + 1):
                center_x = (col + 0.5) * self.cell_size
                center_z = (row + 0.5) * self.cell_size
                dx = center_x - x
                dz = center_z - z
                if dx * dx + dz * dz <= radius_squared:
                    total += 1
                    if self.visited[row, col]:
                        visited += 1
        if total == 0:
            return 1.0
        return visited / float(total)
