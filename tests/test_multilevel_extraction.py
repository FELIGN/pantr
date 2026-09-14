"""Tests for pantr.bspline.MultiLevelExtraction (multi-level Bézier extraction)."""

from __future__ import annotations

import tracemalloc
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import tabulate_bernstein
from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    ExtractionTarget,
    MultiLevelExtraction,
    SpanwiseElementExtraction,
    THBSplineSpace,
    create_uniform_space,
)
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid
from tests._parity_harness import unit_roundoff

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
        assert mle.target is ExtractionTarget.BEZIER
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
        grid = grid.refine(0, [0], [2])
        assert _reproduction_error(THBSplineSpace(_root_1d(), grid)) < 1e-12

    def test_1d_three_levels(self) -> None:
        grid = _grid_1d()
        grid = grid.refine(0, [0], [2])
        grid = grid.refine(1, [0], [2])
        assert _reproduction_error(THBSplineSpace(_root_1d(), grid)) < 1e-12

    def test_2d_corner(self) -> None:
        grid = _grid_2d()
        grid = grid.refine(0, [0, 0], [2, 2])
        assert _reproduction_error(THBSplineSpace(_root_2d(), grid)) < 1e-12

    def test_2d_three_levels(self) -> None:
        grid = _grid_2d()
        grid = grid.refine(0, [1, 1], [3, 3])
        grid = grid.refine(1, [2, 2], [6, 6])
        assert _reproduction_error(THBSplineSpace(_root_2d(), grid)) < 1e-12

    def test_hb(self) -> None:
        grid = _grid_1d()
        grid = grid.refine(0, [0], [2])
        grid = grid.refine(1, [0], [2])
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
        grid = grid.refine(0, [0], [2])
        grid = grid.refine(1, [0], [2])
        assert _pou_error(THBSplineSpace(_root_1d(), grid)) < 1e-10

    def test_pou_2d(self) -> None:
        grid = _grid_2d()
        grid = grid.refine(0, [0, 0], [2, 2])
        assert _pou_error(THBSplineSpace(_root_2d(), grid)) < 1e-10


class TestMayerNarrowBand:
    """Narrow single-element-wide multi-level bands (the D'Angella §3.6.1 bug trigger).

    A passive level-l function straddling the band refines into the deeper region; the
    buggy local truncation would drop it and break partition of unity. pantr keeps it,
    so the extraction reproduces the basis and the column sums stay 1.
    """

    def test_narrow_band_1d(self) -> None:
        grid = _grid_1d()
        grid = grid.refine(0, [1], [2])  # single root cell -> level 1
        grid = grid.refine(1, [2], [3])  # single level-1 cell -> level 2
        thb = THBSplineSpace(_root_1d(), grid)
        assert thb.num_levels == 3
        assert _reproduction_error(thb) < 1e-12
        assert _pou_error(thb) < 1e-10

    def test_narrow_band_2d(self) -> None:
        grid = _grid_2d()
        grid = grid.refine(0, [1, 1], [2, 2])  # single root cell -> level 1
        grid = grid.refine(1, [2, 2], [3, 3])  # single level-1 cell -> level 2
        thb = THBSplineSpace(_root_2d(), grid)
        assert thb.num_levels == 3
        assert _reproduction_error(thb) < 1e-12
        assert _pou_error(thb) < 1e-10

    def test_narrow_band_no_zero_rows(self) -> None:
        # If the §3.6.1 bug were present, the straddling passive function would be
        # spuriously truncated to zero in Mᵉ.  Guard that no row is all zeros.
        grid = _grid_1d()
        grid = grid.refine(0, [1], [2])
        grid = grid.refine(1, [2], [3])
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
        grid = grid.refine(0, [0], [2])
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

    def test_refine_returns_new_grid_and_leaves_extraction_unaffected(self) -> None:
        """No staleness RuntimeError any more: HierarchicalGrid.refine returns a new grid."""
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid)
        mle = MultiLevelExtraction(thb)
        op0_before = mle.operator(0)
        mop0_before = mle.multilevel_operator(0)
        refined = grid.refine(0, [0], [2])
        assert refined is not grid
        assert thb.grid is grid
        assert mle.space is thb
        np.testing.assert_array_equal(mle.operator(0), op0_before)
        np.testing.assert_array_equal(mle.multilevel_operator(0), mop0_before)

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
        assert mle.target is ExtractionTarget.CARDINAL
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
        grid = grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        bezier = MultiLevelExtraction(thb, "bezier")
        lagrange = MultiLevelExtraction(thb, "lagrange")
        assert lagrange.target is ExtractionTarget.LAGRANGE
        for cid in range(thb.grid.num_cells):
            np.testing.assert_allclose(
                lagrange.multilevel_operator(cid), bezier.multilevel_operator(cid), atol=1e-12
            )
        # Cᵉ must differ between targets (target is actually threaded through to Eᵉ).
        assert any(
            not np.allclose(lagrange.operator(cid), bezier.operator(cid))
            for cid in range(thb.grid.num_cells)
        ), "lagrange and bezier operators should differ on at least one cell"


# ──────────────────────────────────────────────────────────────────────────────
# Windowed kernel state and row bookkeeping (#336)
# ──────────────────────────────────────────────────────────────────────────────


class TestWindowedKernelBookkeeping:
    """The kernel's row labels and zero flags, and the extraction's frozen tables."""

    @staticmethod
    def _thb() -> THBSplineSpace:
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
        grid = grid.refine(0, [0, 0], [2, 2])
        grid = grid.refine(1, [0, 0], [2, 2])
        grid = grid.refine(2, [0, 0], [2, 2])
        return THBSplineSpace(_root_2d(), grid)

    def test_tables_are_frozen(self) -> None:
        ext = MultiLevelExtraction(self._thb())
        for name, array in ext._tables._asdict().items():
            assert not array.flags.writeable, f"table {name} is writeable"

    def test_rows_are_labelled_by_active_basis(self) -> None:
        thb = self._thb()
        ext = MultiLevelExtraction(thb)
        for cid in range(thb.grid.num_cells):
            rows, dofs, nonzero = ext._windowed_rows(cid)
            np.testing.assert_array_equal(dofs, thb.active_basis(cid))
            assert rows.shape == (dofs.size, 9)
            assert nonzero.shape == dofs.shape

    def test_zero_flags_match_direct_evaluation(self) -> None:
        # A nonnegative combination of B-splines that has a positive coefficient on one of
        # the cell's window functions is strictly positive in the cell's interior, and a
        # truncated function with no such coefficient evaluates to exactly 0.0.  So the
        # flag is decided exactly by sampling interior points.
        thb = self._thb()
        ext = MultiLevelExtraction(thb)
        xi = _interior_points(thb)
        flagged_zero = 0
        for cid in range(thb.grid.num_cells):
            rows, _, nonzero = ext._windowed_rows(cid)
            lo, hi = thb.grid.cell_bounds(cid)
            vals, _ = thb.tabulate_basis(cid, lo + (hi - lo) * xi)
            np.testing.assert_array_equal(nonzero, np.any(vals != 0.0, axis=0))
            np.testing.assert_array_equal(nonzero, np.any(rows != 0.0, axis=1))
            flagged_zero += int((~nonzero).sum())
        assert flagged_zero > 0, "the hierarchy should contain vanishing truncated functions"

    def test_repeated_calls_identical(self) -> None:
        """Two operator calls on the same cell are bitwise identical."""
        ext = MultiLevelExtraction(self._thb())
        cid = ext.num_elements - 1
        np.testing.assert_array_equal(ext.operator(cid), ext.operator(cid))


# ──────────────────────────────────────────────────────────────────────────────
# Windowed per-element recursion (#336): direct-evaluation oracle, partition of
# unity and depth
# ──────────────────────────────────────────────────────────────────────────────

_UNIT_ROUNDOFF = unit_roundoff(np.float64)
"""Unit roundoff ``u`` of binary64, from the parity harness rather than spelled out again."""

_OSLO_STAGE_OPS = 5
"""Rounded operations per stage of a nonnegative knot-insertion recurrence.

One stage is ``w * a + w' * a'`` with ``w = (x - t_j) / (t_{j+k} - t_j)``: two knot
differences and a division for the weight, a product, and the sum -- five operations,
each carrying a relative error of at most ``u`` against the exact result on the stored
knots.  Counting ``1 - w`` as one more subtraction (Bézier extraction, Bernstein
recurrence) is covered by the same count only while ``w`` stays bounded away from ``1``,
which holds on the uniform knots used here; that part is an admitted assumption.
"""


def _gamma(n: int) -> float:
    """Return Higham's ``gamma_n = n u / (1 - n u)``.

    With nonnegative operands and no cancellation, ``n`` rounded operations perturb a
    result by at most a relative ``gamma_n`` (Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., Lemma 3.1 and Theorem 3.1 in relative form).

    Args:
        n (int): Number of rounded operations.

    Returns:
        float: The relative bound.
    """
    return n * _UNIT_ROUNDOFF / (1.0 - n * _UNIT_ROUNDOFF)


def _chain_ops(degrees: tuple[int, ...], level: int) -> int:
    """Upper count of rounded operations behind one entry of ``M^e`` on a level-``level`` cell.

    Every two-scale coefficient is nonnegative and truncation only zeroes entries, so an
    entry of ``M^e`` is a cancellation-free sum of products.  Each of the ``level``
    transitions contracts direction ``k`` against ``p_k + 1`` coefficients
    (``level * sum(p_k + 1)`` operations), and the product it sums has ``level * dim``
    two-scale factors, each out of a ``p_k``-stage recurrence (``_OSLO_STAGE_OPS * level
    * sum(p_k)``).

    Args:
        degrees (tuple[int, ...]): Per-direction degrees.
        level (int): The cell's level (number of transitions).

    Returns:
        int: The operation count.
    """
    return level * sum(p + 1 for p in degrees) + _OSLO_STAGE_OPS * level * sum(degrees)


def _extraction_ops(degrees: tuple[int, ...], level: int) -> int:
    """Upper count of rounded operations behind one entry of ``C^e = M^e E^e``.

    Adds the Bézier extraction entries (a ``p_k``-stage recurrence per direction) and the
    length-``prod(p_k + 1)`` inner product of the matrix product to :func:`_chain_ops`.

    Args:
        degrees (tuple[int, ...]): Per-direction degrees.
        level (int): The cell's level.

    Returns:
        int: The operation count.
    """
    n_single = int(np.prod([p + 1 for p in degrees]))
    return _chain_ops(degrees, level) + _OSLO_STAGE_OPS * sum(degrees) + n_single


def _oracle_tolerance(
    thb: THBSplineSpace, cid: int, lo: npt.NDArray[np.float64], hi: npt.NDArray[np.float64]
) -> float:
    """Bound ``|C^e B(xi) - tabulate_basis(cid, x)|`` on cell ``cid``, per entry.

    Both sides are cancellation-free sums of nonnegative terms of value at most ``1``
    (the truncated basis is a partition of unity; the untruncated functions are each at
    most ``1``), so each side's rounding is a relative bound read as an absolute one.

    - Extraction side: :func:`_extraction_ops`, the Bernstein recurrence
      (``_OSLO_STAGE_OPS * sum(p_k)``) and the length-``n`` inner product with ``B``.
    - Direct side: the Cox-de Boor recurrence (``_OSLO_STAGE_OPS * sum(p_k)``), the
      truncated coefficients' own chain (:func:`_chain_ops`), ``dim`` tensor-product
      factors and a length-``n`` contraction.
    - The mapping ``x = lo + xi * (hi - lo)``: three operations, so
      ``|dx_k| <= gamma_3 (|lo_k| + h_k)``, and every function on the cell has
      ``|d/dx_k| <= 2 p_k / h_k`` (a B-spline derivative is ``p`` over a knot span of at
      least the cell width times a difference of two nonnegative lower-degree terms;
      nonnegative combinations of weight at most ``1`` keep the bound).

    Args:
        thb (THBSplineSpace): The space.
        cid (int): Active cell id.
        lo (npt.NDArray[np.float64]): Cell lower bounds.
        hi (npt.NDArray[np.float64]): Cell upper bounds.

    Returns:
        float: The absolute tolerance.
    """
    degrees = thb.degrees
    level = thb.grid.cell_level(cid)
    n_single = int(np.prod([p + 1 for p in degrees]))
    recurrence = _OSLO_STAGE_OPS * sum(degrees)
    extraction_side = _gamma(_extraction_ops(degrees, level) + recurrence + n_single)
    direct_side = _gamma(recurrence + _chain_ops(degrees, level) + thb.dim + n_single)
    width = hi - lo
    mapping = float(np.sum(2.0 * np.asarray(degrees) / width * _gamma(3) * (np.abs(lo) + width)))
    return extraction_side + direct_side + mapping


class _CornerCase(NamedTuple):
    """A corner-refined hierarchy: the ticket's reproduction family, generalised."""

    degrees: tuple[int, ...]
    """Per-direction degrees."""
    root_cells: int
    """Root cells per direction."""
    refinements: int
    """Number of refinement steps (levels minus one)."""
    truncate: bool = True
    """THB (``True``) or HB (``False``)."""
    factor: tuple[int, ...] | None = None
    """Per-direction subdivision factor; ``None`` means dyadic."""
    regularity: tuple[int, ...] | None = None
    """Per-direction regularity at inserted knots; ``None`` means maximal."""


def _corner_space(case: _CornerCase) -> THBSplineSpace:
    """Build the hierarchy that refines the ``[0, 2]^d`` corner block once per level.

    Args:
        case (_CornerCase): The configuration.

    Returns:
        THBSplineSpace: The space.
    """
    dim = len(case.degrees)
    factor = case.factor if case.factor is not None else (2,) * dim
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, [case.root_cells] * dim), factor)
    for level in range(case.refinements):
        grid = grid.refine(level, [0] * dim, [2] * dim)
    root = create_uniform_space(list(case.degrees), [case.root_cells] * dim)
    return THBSplineSpace(root, grid, truncate=case.truncate, regularity=case.regularity)


def _check_against_direct_evaluation(
    thb: THBSplineSpace,
    ext: MultiLevelExtraction,
    cid: int,
    xi: npt.NDArray[np.float64],
) -> None:
    """Assert ``C^e B(xi)`` equals ``tabulate_basis`` on cell ``cid``, dof by dof.

    The extraction's rows are matched to ``tabulate_basis`` columns by global dof, so the
    check holds whether or not the extraction lists the functions that vanish on the
    cell; any column the extraction does not list must evaluate to zero there.

    Args:
        thb (THBSplineSpace): The space.
        ext (MultiLevelExtraction): Its extraction.
        cid (int): Active cell id.
        xi (npt.NDArray[np.float64]): Reference points in ``[0, 1]^d``, ``(n_pts, d)``.
    """
    lo, hi = (np.asarray(a, dtype=np.float64) for a in thb.grid.cell_bounds(cid))
    tol = _oracle_tolerance(thb, cid, lo, hi)
    vals, dofs = thb.tabulate_basis(cid, lo + xi * (hi - lo))
    ext_dofs = ext.active_basis(cid)
    pos = np.searchsorted(dofs, ext_dofs)
    assert np.all(pos < dofs.size), f"cell {cid}: extraction lists a dof the space does not"
    np.testing.assert_array_equal(dofs[pos], ext_dofs)
    c_op = ext.operator(cid)
    assert c_op.shape == (ext_dofs.size, int(np.prod([p + 1 for p in thb.degrees])))
    from_extraction = tabulate_bernstein(list(thb.degrees), xi) @ c_op.T
    residual = np.abs(from_extraction - vals[:, pos])
    assert float(residual.max()) <= tol, (
        f"cell {cid}: extraction residual {residual.max():.3e} exceeds {tol:.3e}"
    )
    dropped = np.setdiff1d(np.arange(dofs.size), pos)
    if dropped.size:
        assert float(np.abs(vals[:, dropped]).max()) <= tol, (
            f"cell {cid}: a dof omitted by the extraction is non-zero on the cell"
        )


_ORACLE_CASES = [
    _CornerCase((3,), 8, 2),
    _CornerCase((3, 3), 4, 2),
    _CornerCase((3, 3, 3), 4, 2),
    _CornerCase((3, 3, 3), 4, 2, truncate=False),
    _CornerCase((2, 2), 4, 3),
    _CornerCase((3, 2), 4, 3, factor=(3, 2), regularity=(1, 0)),
    _CornerCase((2,), 4, 4, truncate=False, factor=(3,)),
]


def _case_id(case: _CornerCase) -> str:
    """Return a readable pytest id for a corner case.

    Args:
        case (_CornerCase): The configuration.

    Returns:
        str: The id.
    """
    kind = "thb" if case.truncate else "hb"
    extra = f"-f{case.factor}" if case.factor else ""
    extra += f"-r{case.regularity}" if case.regularity else ""
    return f"p{case.degrees}-n{case.root_cells}-L{case.refinements + 1}-{kind}{extra}"


class TestDirectEvaluationOracle:
    """``C^e B(x)`` equals direct evaluation of the hierarchical basis (#336 AC3)."""

    @pytest.mark.parametrize("case", _ORACLE_CASES, ids=_case_id)
    def test_matches_tabulate_basis(self, case: _CornerCase) -> None:
        thb = _corner_space(case)
        ext = MultiLevelExtraction(thb, "bezier")
        rng = np.random.default_rng(0)
        num_cells = thb.grid.num_cells
        cells = rng.choice(num_cells, size=min(25, num_cells), replace=False)
        deepest = [c for c in range(num_cells) if thb.grid.cell_level(c) == thb.grid.max_level]
        for cid in {int(c) for c in cells} | set(deepest[:4]):
            _check_against_direct_evaluation(thb, ext, cid, rng.random((7, thb.dim)))


class TestWindowedPartitionOfUnity:
    """Column sums of ``M^e`` and ``C^e`` are 1 on every cell, every level (#336 AC4)."""

    @pytest.mark.parametrize("case", [c for c in _ORACLE_CASES if c.truncate], ids=_case_id)
    def test_column_sums(self, case: _CornerCase) -> None:
        thb = _corner_space(case)
        ext = MultiLevelExtraction(thb)
        levels_seen: set[int] = set()
        for cid in range(thb.grid.num_cells):
            level = thb.grid.cell_level(cid)
            levels_seen.add(level)
            m_op = ext.multilevel_operator(cid)
            k = m_op.shape[0]
            m_tol = _gamma(_chain_ops(thb.degrees, level) + k)
            c_tol = _gamma(_extraction_ops(thb.degrees, level) + k)
            m_defect = float(np.abs(m_op.sum(axis=0) - 1.0).max())
            c_defect = float(np.abs(ext.operator(cid).sum(axis=0) - 1.0).max())
            assert m_defect <= m_tol, f"cell {cid}: M^e column-sum defect {m_defect:.3e}"
            assert c_defect <= c_tol, f"cell {cid}: C^e column-sum defect {c_defect:.3e}"
        assert levels_seen == set(range(thb.num_levels))


class TestDeepHierarchy:
    """A 6-level 3D hierarchy stays within the element window (#336 AC5)."""

    _CASE = _CornerCase((3, 3, 3), 4, 5)

    @staticmethod
    def _deep_cells(thb: THBSplineSpace, count: int) -> list[int]:
        grid = thb.grid
        return [c for c in range(grid.num_cells) if grid.cell_level(c) == grid.max_level][:count]

    def test_peak_memory_is_bounded_by_the_element_window(self) -> None:
        thb = _corner_space(self._CASE)
        ext = MultiLevelExtraction(thb)
        # First call outside the trace: JIT dispatch and lazily built state are not the
        # per-cell cost under test.
        ext.multilevel_operator(0)
        cells = self._deep_cells(thb, 4)
        level = thb.grid.max_level
        n_single = int(np.prod([p + 1 for p in self._CASE.degrees]))
        # The windowed recursion holds at most one row per candidate function, i.e.
        # (level + 1) * n_single rows of n_single doubles, once as scratch and once as
        # the returned operator.  The global-box recursion instead materialises a
        # coefficient box of width 2^level (p + 1) - p per direction, which at this depth
        # is two orders of magnitude above that.  The fixed MiB covers interpreter and
        # array-header overhead, which no derivation reaches: a heuristic allowance.
        # tracemalloc sees only NumPy/Python allocations, not the Numba kernel's own
        # scratch; that scratch is O(level * n_single) by construction, and a box built
        # inside the kernel would escape this test.
        window_bytes = 2 * (level + 1) * n_single * n_single * 8
        cap = window_bytes + 2**20
        tracemalloc.start()
        try:
            for cid in cells:
                ext.multilevel_operator(cid)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak <= cap, f"peak {peak / 2**20:.2f} MiB exceeds the window cap {cap / 2**20:.2f}"

    def test_deep_cells_match_direct_evaluation(self) -> None:
        thb = _corner_space(self._CASE)
        ext = MultiLevelExtraction(thb)
        rng = np.random.default_rng(1)
        for cid in self._deep_cells(thb, 4):
            _check_against_direct_evaluation(thb, ext, cid, rng.random((5, 3)))
