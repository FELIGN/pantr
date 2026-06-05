"""Tests for pantr.bspline.MultiLevelExtraction (multi-level Bézier extraction)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import tabulate_bernstein
from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    MultiLevelExtraction,
    SpanwiseElementExtraction,
    THBSplineSpace,
)
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_KNOTS_DEG2_4 = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


def _root_1d() -> BsplineSpace:
    return BsplineSpace([BsplineSpace1D(_KNOTS_DEG2_4, 2)])


def _root_2d() -> BsplineSpace:
    sp = BsplineSpace1D(_KNOTS_DEG2_4, 2)
    return BsplineSpace([sp, sp])


def _grid_1d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)


def _grid_2d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)


def _interior_points(thb: THBSplineSpace) -> npt.NDArray[np.float64]:
    """Reference points ξ ∈ (0,1)^d in the cell interior (reference space)."""
    u = np.linspace(0.0, 1.0, thb.degrees[0] + 3)[1:-1]
    mesh = np.meshgrid(*[u] * thb.dim, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def _reproduction_error(thb: THBSplineSpace) -> float:
    """Max |operator·Bernstein - tabulate_basis| over all cells; also checks C == M·E."""
    mle = MultiLevelExtraction(thb)
    degrees = list(thb.degrees)
    err = 0.0
    for cid in range(thb.grid.num_cells):
        lo, hi = thb.grid.cell_bounds(cid)
        xi = _interior_points(thb)
        x = lo + (hi - lo) * xi
        c_op = mle.operator(cid)
        bernstein = tabulate_bernstein(degrees, xi)
        from_extraction = (c_op @ bernstein.T).T
        direct, _ = thb.tabulate_basis(cid, x)
        err = max(err, float(np.abs(from_extraction - direct).max()))
        # C == M @ E by construction.
        m_op = mle.multilevel_operator(cid)
        e_op = mle._level_extraction(thb.grid.cell_level(cid)).operator(
            thb.grid.cell_multi_index(cid)
        )
        err = max(err, float(np.abs(m_op @ e_op - c_op).max()))
    return err


def _pou_error(thb: THBSplineSpace) -> float:
    """Max |column-sum - 1| of the Cᵉ and Mᵉ operators over all cells (THB only)."""
    mle = MultiLevelExtraction(thb)
    err = 0.0
    for cid in range(thb.grid.num_cells):
        for op in (mle.operator(cid), mle.multilevel_operator(cid)):
            err = max(err, float(np.abs(op.sum(axis=0) - 1.0).max()))
    return err


# ──────────────────────────────────────────────────────────────────────────────
# Construction / properties
# ──────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    """Constructor validation and properties (mirroring SpanwiseElementExtraction)."""

    def test_non_thb_space_raises(self) -> None:
        with pytest.raises(TypeError, match="THBSplineSpace"):
            MultiLevelExtraction(_root_1d())  # type: ignore[arg-type]

    def test_bad_target_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="target"):
            MultiLevelExtraction(thb, target="nope")  # type: ignore[arg-type]

    def test_properties(self) -> None:
        thb = THBSplineSpace(_root_2d(), _grid_2d())
        mle = MultiLevelExtraction(thb)
        assert mle.space is thb
        assert mle.target == "bezier"
        assert mle.dim == 2
        assert mle.dtype == np.float64
        assert mle.num_elements == thb.grid.num_cells
        assert "MultiLevelExtraction" in repr(mle)


# ──────────────────────────────────────────────────────────────────────────────
# Reproduction (Cᵉ·B == tabulate_basis)
# ──────────────────────────────────────────────────────────────────────────────


class TestReproduction:
    """The extraction reproduces the hierarchical functions on every cell."""

    def test_unrefined(self) -> None:
        assert _reproduction_error(THBSplineSpace(_root_1d(), _grid_1d())) < 1e-12

    def test_1d_two_levels(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        assert _reproduction_error(THBSplineSpace(_root_1d(), grid)) < 1e-12

    def test_1d_three_levels(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(1, [0], [2])
        assert _reproduction_error(THBSplineSpace(_root_1d(), grid)) < 1e-12

    def test_2d_corner(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        assert _reproduction_error(THBSplineSpace(_root_2d(), grid)) < 1e-12

    def test_2d_three_levels(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [1, 1], [3, 3])
        grid.refine(1, [2, 2], [6, 6])
        assert _reproduction_error(THBSplineSpace(_root_2d(), grid)) < 1e-12

    def test_hb(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(1, [0], [2])
        assert _reproduction_error(THBSplineSpace(_root_1d(), grid, truncate=False)) < 1e-12

    def test_unrefined_equals_single_level_bezier(self) -> None:
        # On an unrefined grid every cell has all (p+1)^d level-0 functions active, so
        # Mᵉ is the identity and the operator equals the single-level Bézier extraction.
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        ext = SpanwiseElementExtraction(thb.root_space, "bezier")
        mle = MultiLevelExtraction(thb)
        n = thb.degrees[0] + 1
        for cid in range(thb.grid.num_cells):
            np.testing.assert_allclose(mle.multilevel_operator(cid), np.eye(n), atol=1e-12)
            np.testing.assert_allclose(
                mle.operator(cid), ext.operator(thb.grid.cell_multi_index(cid)), atol=1e-12
            )


# ──────────────────────────────────────────────────────────────────────────────
# Partition of unity through the operator
# ──────────────────────────────────────────────────────────────────────────────


class TestPartitionOfUnity:
    """Column sums of Cᵉ and Mᵉ are 1 for THB (the operator-level PoU check)."""

    def test_pou_1d(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(1, [0], [2])
        assert _pou_error(THBSplineSpace(_root_1d(), grid)) < 1e-10

    def test_pou_2d(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        assert _pou_error(THBSplineSpace(_root_2d(), grid)) < 1e-10


class TestMayerNarrowBand:
    """Narrow single-element-wide multi-level bands (the D'Angella §3.6.1 bug trigger).

    A passive level-l function straddling the band refines into the deeper region; the
    buggy local truncation would drop it and break partition of unity. pantr keeps it,
    so the extraction reproduces the basis and the column sums stay 1.
    """

    def test_narrow_band_1d(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [1], [2])  # single root cell -> level 1
        grid.refine(1, [2], [3])  # single level-1 cell -> level 2
        thb = THBSplineSpace(_root_1d(), grid)
        assert thb.num_levels == 3
        assert _reproduction_error(thb) < 1e-12
        assert _pou_error(thb) < 1e-10

    def test_narrow_band_2d(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [1, 1], [2, 2])  # single root cell -> level 1
        grid.refine(1, [2, 2], [3, 3])  # single level-1 cell -> level 2
        thb = THBSplineSpace(_root_2d(), grid)
        assert thb.num_levels == 3
        assert _reproduction_error(thb) < 1e-12
        assert _pou_error(thb) < 1e-10

    def test_narrow_band_no_zero_rows(self) -> None:
        # If the §3.6.1 bug were present, the straddling passive function would be
        # spuriously truncated to zero in Mᵉ.  Guard that no row is all zeros.
        grid = _grid_1d()
        grid.refine(0, [1], [2])
        grid.refine(1, [2], [3])
        thb = THBSplineSpace(_root_1d(), grid)
        mle = MultiLevelExtraction(thb)
        for cid in range(thb.grid.num_cells):
            m_op = mle.multilevel_operator(cid)
            assert not np.any(np.all(m_op == 0.0, axis=1)), (
                f"cell {cid}: multilevel_operator has an all-zero row "
                "(passive straddling function was spuriously dropped)"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Operator shape / API / validation
# ──────────────────────────────────────────────────────────────────────────────


class TestOperatorApi:
    """Shapes, row labelling, target plumbing, and ``out=`` validation."""

    def test_shapes_and_active_basis(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        mle = MultiLevelExtraction(thb)
        n_bernstein = thb.degrees[0] + 1
        for cid in range(thb.grid.num_cells):
            rows = mle.active_basis(cid)
            assert mle.operator(cid).shape == (rows.shape[0], n_bernstein)
            assert mle.multilevel_operator(cid).shape == (rows.shape[0], n_bernstein)
            np.testing.assert_array_equal(rows, thb.active_basis(cid))

    def test_out_argument(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        k = mle.active_basis(0).shape[0]
        out = np.empty((k, thb.degrees[0] + 1), dtype=np.float64)
        ret = mle.operator(0, out=out)
        assert ret is out
        np.testing.assert_allclose(out, mle.operator(0))

    def test_out_bad_shape_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        with pytest.raises(ValueError, match="shape"):
            mle.operator(0, out=np.empty((1, 99)))

    def test_out_bad_dtype_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        k = mle.active_basis(0).shape[0]
        n = thb.degrees[0] + 1
        with pytest.raises(ValueError, match="dtype"):
            mle.operator(0, out=np.empty((k, n), dtype=np.float32))
        with pytest.raises(ValueError, match="dtype"):
            mle.multilevel_operator(0, out=np.empty((k, n), dtype=np.float32))

    def test_out_not_writeable_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        k = mle.active_basis(0).shape[0]
        n = thb.degrees[0] + 1
        buf = np.empty((k, n), dtype=np.float64)
        buf.flags.writeable = False
        with pytest.raises(ValueError, match="writeable"):
            mle.operator(0, out=buf)
        with pytest.raises(ValueError, match="writeable"):
            mle.multilevel_operator(0, out=buf)

    def test_multilevel_operator_out_argument(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        k = mle.active_basis(0).shape[0]
        out = np.empty((k, thb.degrees[0] + 1), dtype=np.float64)
        ret = mle.multilevel_operator(0, out=out)
        assert ret is out
        np.testing.assert_allclose(out, mle.multilevel_operator(0))

    def test_stale_grid_raises(self) -> None:
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid)
        mle = MultiLevelExtraction(thb)
        grid.refine(0, [0], [2])
        with pytest.raises(RuntimeError, match="stale"):
            mle.operator(0)
        with pytest.raises(RuntimeError, match="stale"):
            mle.multilevel_operator(0)

    def test_negative_cid_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        with pytest.raises(IndexError):
            mle.operator(-1)
        with pytest.raises(IndexError):
            mle.multilevel_operator(-1)

    def test_cardinal_target(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb, "cardinal")
        assert mle.target == "cardinal"
        for cid in range(thb.grid.num_cells):
            mle.operator(cid)
            mle.multilevel_operator(cid)

    def test_len(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        assert len(mle) == mle.num_elements == thb.grid.num_cells

    def test_out_of_range_cid_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        mle = MultiLevelExtraction(thb)
        with pytest.raises(IndexError):
            mle.operator(999)

    def test_lagrange_target(self) -> None:
        # Mᵉ is target-independent; only Eᵉ (hence Cᵉ) changes with the target.
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        bezier = MultiLevelExtraction(thb, "bezier")
        lagrange = MultiLevelExtraction(thb, "lagrange")
        assert lagrange.target == "lagrange"
        for cid in range(thb.grid.num_cells):
            np.testing.assert_allclose(
                lagrange.multilevel_operator(cid), bezier.multilevel_operator(cid), atol=1e-12
            )
        # Cᵉ must differ between targets (target is actually threaded through to Eᵉ).
        assert any(
            not np.allclose(lagrange.operator(cid), bezier.operator(cid))
            for cid in range(thb.grid.num_cells)
        ), "lagrange and bezier operators should differ on at least one cell"
