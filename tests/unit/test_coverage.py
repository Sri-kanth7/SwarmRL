"""Unit tests for the coverage grid and mock-stream coverage behavior.

Two groups are covered:

* ``CoverageGrid`` geometry, the ``explored_pct`` formula, duplicate-cell
  handling, ``mark_segment`` sweeping, cumulative bookkeeping, reset, and
  the clamp-in-bounds mapping for out-of-range positions.
* The mock inference engine's coverage over a long run: it must be
  cumulative (non-decreasing) and legitimately saturate once the fixed
  circular orbits stop touching new cells, while drone positions keep
  changing.

These tests use the existing dependencies only (numpy via the engine);
no network, Ray runtime, or checkpoint is involved.
"""

from itertools import pairwise

import pytest

from env.coverage import CoverageGrid
from server.config import ServerConfig
from server.inference import MockInferenceEngine

WORLD_SIZE = (100.0, 50.0, 100.0)
CELL_SIZE = 5.0

# The mock generator's slowest orbit is 2*pi/(0.1 rad/s) = 62.8 s, i.e.
# 2*pi/(0.1 * 0.1) ~= 629 steps at the 100 ms tick, so every drone has
# completed at least one full revolution well before this horizon. The
# tail window therefore lies entirely in the saturated regime.
MOCK_HORIZON_STEPS = 700
MOCK_SATURATION_TAIL = 40


# ----------------------------------------------------------------------
# CoverageGrid
# ----------------------------------------------------------------------
def test_grid_dimensions_and_total_cells():
    grid = CoverageGrid(WORLD_SIZE, CELL_SIZE)
    assert (grid.rows, grid.cols) == (20, 20)
    assert grid.total_cells == 400
    assert grid.visited_count == 0
    assert grid.explored_pct() == 0.0


def test_explored_pct_is_visited_over_total():
    grid = CoverageGrid(WORLD_SIZE, CELL_SIZE)
    grid.mark(2.5, 2.5)  # row 0, col 0
    assert grid.explored_pct() == pytest.approx(100.0 * 1 / 400)
    grid.mark(7.5, 2.5)  # row 0, col 1
    assert grid.visited_count == 2
    assert grid.explored_pct() == pytest.approx(100.0 * 2 / 400)


def test_cell_coords_and_index_map_and_clamp_positions():
    grid = CoverageGrid(WORLD_SIZE, CELL_SIZE)
    assert grid.cell_coords(2.5, 2.5) == (0, 0)
    assert grid.cell_coords(97.5, 97.5) == (19, 19)
    assert grid.cell_index(7.5, 12.5) == 2 * 20 + 1
    # Out-of-bounds positions clamp to the nearest edge cell (not rejected).
    assert grid.cell_coords(-10.0, -10.0) == (0, 0)
    assert grid.cell_coords(250.0, 250.0) == (19, 19)


def test_mark_reports_only_newly_visited_cells():
    grid = CoverageGrid(WORLD_SIZE, CELL_SIZE)
    first = grid.mark(12.5, 7.5)  # row 1, col 2 -> index 22
    assert first == [22]
    assert grid.mark(12.5, 7.5) == []
    assert grid.visited_count == 1


def test_mark_segment_is_deduplicated_and_cumulative():
    grid = CoverageGrid(WORLD_SIZE, CELL_SIZE)
    cells = grid.mark_segment(2.5, 2.5, 97.5, 2.5)
    # A straight sweep across row 0 touches every column exactly once.
    assert sorted(cells) == list(range(20))
    assert len(cells) == len(set(cells))
    assert grid.visited_count == 20
    # Retracing the same segment discovers nothing new.
    assert grid.mark_segment(2.5, 2.5, 97.5, 2.5) == []
    assert grid.visited_count == 20


def test_positions_outside_the_grid_clamp_to_edge_cells():
    grid = CoverageGrid(WORLD_SIZE, CELL_SIZE)
    assert grid.mark(-5.0, -5.0) == [0]
    assert grid.mark(1000.0, 1000.0) == [19 * 20 + 19]
    assert grid.visited_count == 2


def test_reset_clears_all_visited_cells():
    grid = CoverageGrid(WORLD_SIZE, CELL_SIZE)
    grid.mark_segment(2.5, 2.5, 97.5, 2.5)
    assert grid.visited_count == 20
    grid.reset()
    assert grid.visited_count == 0
    assert grid.explored_pct() == 0.0
    assert not grid.visited.any()


# ----------------------------------------------------------------------
# Mock engine coverage (cumulative, then saturating)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_stream():
    """One long deterministic mock run shared read-only by the tests below."""
    engine = MockInferenceEngine(ServerConfig(agent_count=50, seed=42))
    return [engine.step() for _ in range(MOCK_HORIZON_STEPS)]


def test_mock_newly_visited_cells_are_valid_grid_indices(mock_stream):
    total = CoverageGrid(WORLD_SIZE, CELL_SIZE).total_cells
    for message in mock_stream:
        for index in message.newly_visited_cells:
            assert 0 <= index < total


def test_mock_coverage_never_decreases_and_is_cumulative(mock_stream):
    pcts = [message.explored_pct for message in mock_stream]
    assert all(later >= earlier for earlier, later in pairwise(pcts))
    assert 0.0 <= pcts[0] <= pcts[-1] <= 100.0
    # The swarm discovers new cells before the orbits close.
    assert pcts[-1] > pcts[0]


def test_mock_coverage_saturates_while_agents_keep_moving(mock_stream):
    tail = mock_stream[-MOCK_SATURATION_TAIL:]
    # Every orbit has closed by the start of this window, so no further
    # cells are discovered and the percentage is stable.
    assert all(message.newly_visited_cells == [] for message in tail)
    assert tail[-1].explored_pct == pytest.approx(tail[0].explored_pct)
    # Positions still change: the plateau is saturation, not a frozen stream.
    assert len(tail[-1].agents) == 50
    last_positions = [(a.x, a.y, a.z) for a in tail[-1].agents]
    previous_positions = [(a.x, a.y, a.z) for a in tail[-2].agents]
    assert last_positions != previous_positions
