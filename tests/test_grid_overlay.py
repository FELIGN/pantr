"""Tests for pantr.grid.overlay (coarsest common refinement of two grids)."""

from __future__ import annotations

import numpy as np
import numpy.testing as nptest
import pytest

from pantr._backend import Backend, use_backend
from pantr.grid import TensorProductGrid, overlay, uniform_grid
from tests._parity_harness import demand_cpp_backend


@pytest.fixture
def cpp_backend() -> None:
    """Require the compiled extension for the test that switches to the C++ backend.

    Routed through the parity harness rather than a bare ``skipif``: a bare skip is
    silent, and a suite that skips its way to green has let real failures through in
    this repository.
    """
    demand_cpp_backend()


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
    """Overlay is defined for ndim > 3 (generalized beyond the originating suite's 2/3-D cap)."""
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
    """Breakpoints closer than the tolerance collapse into one.

    The merge tolerance is ``get_default(float64) * window = 1.42e-14`` here, so
    the separation below is well inside it (9 ulp at 0.5) while remaining a
    perfectly representable pair.
    """
    a = TensorProductGrid([[0.0, 0.5, 1.0]])
    b = TensorProductGrid([[0.0, 0.5 + 1e-15, 1.0]])
    result = overlay(a, b)
    nptest.assert_allclose(result.breakpoints[0], [0.0, 0.5, 1.0])


def test_overlay_keeps_distinct_close_breakpoints() -> None:
    """Breakpoints farther apart than the tolerance are both retained."""
    a = TensorProductGrid([[0.0, 0.5, 1.0]])
    b = TensorProductGrid([[0.0, 0.5 + 1e-3, 1.0]])
    result = overlay(a, b)
    assert result.breakpoints[0].shape[0] == 4


@pytest.mark.parametrize("lam", [1e-6, 1.0, 1e6])
def test_overlay_merge_verdict_is_scale_covariant(lam: float) -> None:
    """The same breakpoints in units of ``lam`` get the same verdict at every ``lam``.

    The tolerance is relative to the axis's own magnitude, so a separation of
    ``1e-15 * lam`` merges and one of ``1e-3 * lam`` does not, whatever ``lam`` is.
    An absolute tolerance merged everything at ``lam = 1e-6`` and nothing at
    ``lam = 1e6``.
    """
    near = overlay(
        TensorProductGrid([[0.0, 0.5 * lam, 1.0 * lam]]),
        TensorProductGrid([[0.0, 0.5 * lam + 1e-15 * lam, 1.0 * lam]]),
    )
    assert near.breakpoints[0].shape[0] == 3

    far = overlay(
        TensorProductGrid([[0.0, 0.5 * lam, 1.0 * lam]]),
        TensorProductGrid([[0.0, 0.5 * lam + 1e-3 * lam, 1.0 * lam]]),
    )
    assert far.breakpoints[0].shape[0] == 4


def test_overlay_tolerance_follows_the_offset_not_only_the_window() -> None:
    """A short window far from the origin is graded on the coordinates, not its length.

    The other covariance test rescales both the window and its position together, so
    the window always dominates ``max(hi - lo, |lo|, |hi|)``. This is the other branch:
    a window of 400 ulp sitting at 1e10, where the *offset* sets the tolerance
    (``get_default(float64) * 1e10 = 1.42e-4``, about 75 ulp there). An interior
    breakpoint further than that from both ends survives; one closer to an end than
    that is folded into it, which is what merging means and not a loss -- the end is
    emitted regardless.
    """
    lo = 1e10
    ulp = float(np.spacing(lo))
    hi = lo + 400.0 * ulp
    plain = TensorProductGrid([[lo, hi]])

    kept = overlay(TensorProductGrid([[lo, lo + 200.0 * ulp, hi]]), plain)
    assert kept.breakpoints[0].shape[0] == 3, (
        "a breakpoint 200 ulp from both ends is outside the ~75 ulp tolerance and must survive"
    )

    folded = overlay(TensorProductGrid([[lo, lo + 30.0 * ulp, hi]]), plain)
    assert folded.breakpoints[0].shape[0] == 2, (
        "a breakpoint 30 ulp from the lower end is inside the tolerance and must fold into it"
    )
    nptest.assert_array_equal(folded.breakpoints[0], [lo, hi])


def test_overlay_ndim_mismatch_raises() -> None:
    """Mismatched ndim is a ValueError."""
    with pytest.raises(ValueError, match="share ndim"):
        overlay(uniform_grid([[0.0, 1.0]], 2), uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2))


def test_overlay_disjoint_domains_raises() -> None:
    """Non-overlapping domains are a ValueError."""
    with pytest.raises(ValueError, match="do not overlap"):
        overlay(uniform_grid([[0.0, 1.0]], 2), uniform_grid([[2.0, 3.0]], 2))


def test_overlay_touching_domains_raises() -> None:
    """Domains that share only an endpoint (zero-width intersection) are a ValueError."""
    with pytest.raises(ValueError, match="do not overlap"):
        overlay(uniform_grid([[0.0, 1.0]], 2), uniform_grid([[1.0, 2.0]], 2))


def test_overlay_non_grid_input_raises() -> None:
    """Non-TensorProductGrid inputs are a TypeError."""
    grid = uniform_grid([[0.0, 1.0]], 2)
    with pytest.raises(TypeError, match="TensorProductGrid"):
        overlay(grid, object())  # type: ignore[arg-type]


def test_overlay_runs_on_cpp_backed_grids(cpp_backend: None) -> None:
    """Overlay's ``isinstance`` gate admits a grid whose implementation is the C++ one.

    ``overlay`` opens with ``isinstance(grid, TensorProductGrid)`` on both arguments and
    refuses anything else, so under the C++ backend that gate is the whole of the port's
    exposure here. Asserting the call returned is not enough on its own: a grid that had
    quietly stayed on the Python oracle would pass the same gate and say nothing about
    the port, which is why the implementation class is asserted on the inputs *and* on
    the result. ``test_overlay_non_grid_input_raises`` is the other half -- it pins that
    the gate still rejects, so this one cannot pass by the gate having been deleted.
    """
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415  (absent without the compiled extension)

    with use_backend(Backend.CPP):
        grid_a = uniform_grid([[0.0, 1.0]], 2)
        grid_b = uniform_grid([[0.0, 1.0]], 3)
        assert type(grid_a._impl) is _pantr_cpp.TensorProductGrid
        assert type(grid_b._impl) is _pantr_cpp.TensorProductGrid

        result = overlay(grid_a, grid_b)

        assert type(result._impl) is _pantr_cpp.TensorProductGrid
        nptest.assert_allclose(result.breakpoints[0], [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0])
