"""Tests for pantr.grid.overlay (coarsest common refinement of two grids)."""

from __future__ import annotations

import numpy as np
import numpy.testing as nptest
import pytest

from pantr.grid import TensorProductGrid, overlay, uniform_grid


def test_overlay_union_1d() -> None:
    """Overlay merges per-axis breakpoints into their sorted union."""
    result = overlay(uniform_grid([[0.0, 1.0]], 2), uniform_grid([[0.0, 1.0]], 3))
    nptest.assert_allclose(
        result.breakpoints[0],
        [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0],
    )


def test_overlay_returns_tensor_product_grid() -> None:
    """Overlay returns a TensorProductGrid of the shared dimension."""
    result = overlay(uniform_grid([[0.0, 1.0]], 2), uniform_grid([[0.0, 1.0]], 3))
    assert isinstance(result, TensorProductGrid)
    assert result.ndim == 1


def test_overlay_is_symmetric() -> None:
    """overlay(a, b) and overlay(b, a) yield the same breakpoints."""
    a = uniform_grid([[0.0, 2.0], [0.0, 1.0]], [3, 2])
    b = uniform_grid([[0.0, 2.0], [0.0, 1.0]], [2, 5])
    ab = overlay(a, b)
    ba = overlay(b, a)
    for d in range(a.ndim):
        nptest.assert_allclose(ab.breakpoints[d], ba.breakpoints[d])


def test_overlay_union_2d() -> None:
    """Each axis of a 2-D overlay is the union of the two inputs' breakpoints."""
    a = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [2, 2])
    b = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [4, 3])
    result = overlay(a, b)
    nptest.assert_allclose(result.breakpoints[0], [0.0, 0.25, 0.5, 0.75, 1.0])
    nptest.assert_allclose(result.breakpoints[1], [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0])


def test_overlay_refines_both_inputs() -> None:
    """Every overlay cell lies inside exactly one cell of each input."""
    a = uniform_grid([[0.0, 2.0], [0.0, 2.0]], [2, 3])
    b = uniform_grid([[0.0, 2.0], [0.0, 2.0]], [3, 2])
    result = overlay(a, b)
    for cid in range(result.num_cells):
        lo, hi = result.cell_bounds(cid)
        center = 0.5 * (lo + hi)
        for parent in (a, b):
            pid = parent.locate(center)
            assert pid is not None
            plo, phi = parent.cell_bounds(pid)
            assert np.all(plo <= lo + 1e-12)
            assert np.all(hi <= phi + 1e-12)


def test_overlay_high_dimension() -> None:
    """Overlay is defined for ndim > 3 (generalized beyond ocelat's 2/3-D cap)."""
    a = uniform_grid([[0.0, 1.0]] * 4, 2)
    b = uniform_grid([[0.0, 1.0]] * 4, 3)
    result = overlay(a, b)
    assert result.ndim == 4
    for d in range(4):
        nptest.assert_allclose(result.breakpoints[d], [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0])


def test_overlay_restricts_to_domain_intersection() -> None:
    """Overlay is taken over the intersection of the two domains."""
    a = uniform_grid([[0.0, 2.0]], 4)  # breakpoints 0, 0.5, 1, 1.5, 2
    b = uniform_grid([[1.0, 3.0]], 4)  # breakpoints 1, 1.5, 2, 2.5, 3
    result = overlay(a, b)
    nptest.assert_allclose(result.breakpoints[0], [1.0, 1.5, 2.0])


def test_overlay_merges_near_coincident_breakpoints() -> None:
    """Breakpoints closer than the float64 tolerance collapse into one."""
    a = TensorProductGrid([[0.0, 0.5, 1.0]])
    b = TensorProductGrid([[0.0, 0.5 + 1e-13, 1.0]])
    result = overlay(a, b)
    nptest.assert_allclose(result.breakpoints[0], [0.0, 0.5, 1.0])


def test_overlay_keeps_distinct_close_breakpoints() -> None:
    """Breakpoints farther apart than the tolerance are both retained."""
    a = TensorProductGrid([[0.0, 0.5, 1.0]])
    b = TensorProductGrid([[0.0, 0.5 + 1e-3, 1.0]])
    result = overlay(a, b)
    assert result.breakpoints[0].shape[0] == 4


def test_overlay_ndim_mismatch_raises() -> None:
    """Mismatched ndim is a ValueError."""
    with pytest.raises(ValueError, match="share ndim"):
        overlay(uniform_grid([[0.0, 1.0]], 2), uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2))


def test_overlay_disjoint_domains_raises() -> None:
    """Non-overlapping domains are a ValueError."""
    with pytest.raises(ValueError, match="do not overlap"):
        overlay(uniform_grid([[0.0, 1.0]], 2), uniform_grid([[2.0, 3.0]], 2))


def test_overlay_non_grid_input_raises() -> None:
    """Non-TensorProductGrid inputs are a TypeError."""
    grid = uniform_grid([[0.0, 1.0]], 2)
    with pytest.raises(TypeError, match="TensorProductGrid"):
        overlay(grid, object())  # type: ignore[arg-type]
