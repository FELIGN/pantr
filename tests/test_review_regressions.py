"""Regression and property tests from the June 2026 full-codebase review.

Two kinds of tests live here:

- **Known-bug regressions** -- each ``xfail(strict=True)`` test reproduces a confirmed
  bug. When a bug is fixed the corresponding test starts passing, pytest reports a
  strict XPASS failure, and the marker should be removed (promoting the test to a
  permanent regression guard).
- **Property tests** -- randomized/adversarial checks that currently pass, covering
  gaps found during the review (THB-spline basis identities, admissible refinement,
  multi-level extraction, and the distributed local-space machinery).
"""

from __future__ import annotations

import os
from math import pi
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pantr.basis import tabulate_bernstein_1d
from pantr.bezier import Bezier, find_roots
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    MultiLevelExtraction,
    SpanwiseElementExtraction,
    THBSpline,
    THBSplineSpace,
    build_local,
    create_uniform_space,
    l2_project_bspline,
)
from pantr.bspline._local_space import _thb_dof_owner, dof_owner
from pantr.change_basis import compute_monomial_to_bernstein_1d
from pantr.grid import HierarchicalGrid, Partition, tensor_product_grid, uniform_grid
from pantr.quad import PointsLattice, get_chebyshev_gauss_2nd_kind_1d
from pantr.viz._vtk_ordering import vtk_ordering_hex, vtk_ordering_quad

if TYPE_CHECKING:
    import numpy.typing as npt

_JIT_DISABLED = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _open_uniform_1d(degree: int, n_intervals: int, length: float = 1.0) -> BsplineSpace1D:
    """Open-uniform 1D space with ``n_intervals`` equal knot spans on ``[0, length]``."""
    knots = np.concatenate(
        [
            np.zeros(degree + 1),
            np.linspace(0.0, length, n_intervals + 1)[1:-1],
            np.full(degree + 1, length),
        ]
    )
    return BsplineSpace1D(knots, degree)


def _make_thb(  # noqa: PLR0913
    dim: int,
    degree: int,
    n0: int,
    refines: int,
    seed: int,
    *,
    regularity: int | None = None,
    admissible: int | None = None,
) -> THBSplineSpace:
    """Build a THB space on an open-uniform root, randomly refined ``refines`` times."""
    root = BsplineSpace([_open_uniform_1d(degree, n0) for _ in range(dim)])
    grid = HierarchicalGrid(uniform_grid([[0.0, 1.0]] * dim, [n0] * dim), 2)
    space = THBSplineSpace(root, grid, regularity=regularity)
    rng = np.random.default_rng(seed)
    for _ in range(refines):
        n_cells = space.grid.num_cells
        ids = rng.choice(n_cells, size=max(1, n_cells // 6), replace=False)
        space = space.refine(ids, admissible_class=admissible)
    return space


def _cell_points(
    space: THBSplineSpace, cid: int, n_pts: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Random points strictly inside cell ``cid``."""
    lo, hi = space.grid.cell_bounds(cid)
    pts: npt.NDArray[np.float64] = lo + (hi - lo) * rng.uniform(0.05, 0.95, size=(n_pts, space.dim))
    return pts


def _bernstein_degree6_from_roots(
    roots: list[float], extra_monomial: list[float] | None = None
) -> npt.NDArray[np.float64]:
    """Bernstein coefficients of the monic polynomial with the given roots."""
    mono = np.polynomial.polynomial.polyfromroots(roots)
    if extra_monomial is not None:
        mono = np.polynomial.polynomial.polymul(mono, extra_monomial)
    matrix = np.asarray(compute_monomial_to_bernstein_1d(len(mono) - 1), dtype=np.float64)
    coeffs: npt.NDArray[np.float64] = matrix @ np.asarray(mono, dtype=np.float64)
    return coeffs


# ---------------------------------------------------------------------------
# Known-bug regressions: bezier root finding
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="dedup merge radius zero_tol/|f'| is unbounded at a multiple root: an exact "
    "double root at t=0 swallows every other root on the clipping path (degree >= 6)",
)
def test_find_roots_double_root_at_zero_keeps_other_roots() -> None:
    # f(t) = t^2 (t - 0.5) (t + 1) (t^2 + 1): roots in [0, 1] are 0 (double) and 0.5.
    coeffs = _bernstein_degree6_from_roots([0.0, 0.0, 0.5, -1.0], [1.0, 0.0, 1.0])
    roots = np.asarray(find_roots(Bezier(coeffs)), dtype=np.float64)
    assert np.any(np.abs(roots - 0.5) < 1e-6), f"root at 0.5 missing: {roots!r}"
    assert np.any(np.abs(roots) < 1e-6), f"root at 0 missing: {roots!r}"


@pytest.mark.skipif(
    not _JIT_DISABLED,
    reason="the unclamped count overflows out_roots: an out-of-bounds write is "
    "memory-unsafe under JIT (nopython kernels have no bounds checks)",
)
@pytest.mark.xfail(
    strict=True,
    reason="batch dedup does not clamp the root count to the polynomial degree: "
    "candidates around several even-multiplicity roots overflow the (degree,)-wide "
    "output row (IndexError with JIT disabled, memory corruption with JIT)",
)
def test_find_roots_batch_multiple_double_roots_within_degree() -> None:
    # f(t) = t^2 (t - 0.3)^2 (t - 0.7)^2, degree 6, three double roots.
    coeffs = _bernstein_degree6_from_roots([0.0, 0.0, 0.3, 0.3, 0.7, 0.7])
    result = find_roots([Bezier(coeffs), Bezier(coeffs)])
    assert isinstance(result, tuple)
    roots, counts = result
    degree = coeffs.shape[0] - 1
    for row in range(2):
        count = int(counts[row])
        assert count <= degree
        row_roots = np.asarray(roots[row][:count], dtype=np.float64)
        assert np.all((row_roots >= 0.0) & (row_roots <= 1.0))


# ---------------------------------------------------------------------------
# Known-bug regressions: hierarchical grid
# ---------------------------------------------------------------------------


def _two_level_jump_grid() -> tuple[HierarchicalGrid, int, int]:
    """1D grid with a level-0 cell facing a level-2 cell across one facet.

    Returns the grid, the level-0 cell id ([0, 0.25]), and the level-2 cell id
    ([0.25, 0.3125]) that share the facet at x = 0.25.
    """
    grid = HierarchicalGrid(uniform_grid([[0.0, 1.0]], [4]), 2)
    grid.refine(0, [1], [3])
    grid.refine(1, [2], [4])
    cid_coarse = next(
        c
        for c in range(grid.num_cells)
        if grid.cell_level(c) == 0 and abs(float(grid.cell_bounds(c)[0][0])) < 1e-12
    )
    cid_fine = next(
        c
        for c in range(grid.num_cells)
        if grid.cell_level(c) == 2 and abs(float(grid.cell_bounds(c)[0][0]) - 0.25) < 1e-12
    )
    return grid, cid_coarse, cid_fine


@pytest.mark.xfail(
    strict=True,
    reason="neighbor_across_facet only probes +-1 level: a facet between level-0 and "
    "level-2 cells (which plain refine() permits) reports no neighbor and the interior "
    "facet is misclassified as a mesh boundary",
)
def test_hierarchical_neighbor_across_two_level_jump() -> None:
    grid, cid_coarse, cid_fine = _two_level_jump_grid()
    assert grid.neighbor_across_facet(cid_coarse, 1) == cid_fine
    assert grid.neighbor_across_facet(cid_fine, 0) == cid_coarse
    assert not grid.is_mesh_boundary_facet(cid_coarse, 1)
    assert grid.hanging_neighbors(cid_coarse, 1) == (cid_fine,)


@pytest.mark.xfail(
    strict=True,
    reason="the (max_level, num_cells) staleness snapshot is fooled by a compensating "
    "refine+coarsen pair: the stale THB space then silently returns wrong dofs/values "
    "instead of raising",
)
def test_thb_space_detects_compensating_refine_coarsen() -> None:
    root = BsplineSpace([_open_uniform_1d(2, 8)])
    grid = HierarchicalGrid(uniform_grid([[0.0, 1.0]], [8]), 2)
    grid.refine(0, [0], [2])
    grid.refine(0, [4], [6])
    space = THBSplineSpace(root, grid)
    snapshot = (grid.max_level, grid.num_cells)

    grid.coarsen(0, [4], [6])
    grid.refine(0, [5], [7])
    assert (grid.max_level, grid.num_cells) == snapshot  # the snapshot cannot tell

    # Cell id 2 now decodes to a different cell; a stale evaluation must raise, not
    # silently mix the old active set with the new cell layout.
    lo, hi = grid.cell_bounds(2)
    pt = 0.5 * (lo + hi).reshape(1, 1)
    with pytest.raises(RuntimeError):
        space.tabulate_basis(2, pt)


@pytest.mark.xfail(
    strict=True,
    reason="locate/locate_many accept NaN coordinates and return a valid cell id "
    "(last cell on the scalar path, first cell on the batch path) instead of a miss",
)
def test_locate_nan_is_a_miss() -> None:
    grid = uniform_grid([[0.0, 1.0]], [4])
    assert grid.locate([float("nan")]) is None
    batch = grid.locate_many(np.array([[float("nan")]]))
    assert int(np.asarray(batch)[0]) == -1


# ---------------------------------------------------------------------------
# Known-bug regressions: B-spline core
# ---------------------------------------------------------------------------


def test_snap_knots_does_not_merge_distant_knots() -> None:
    space = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 0.500001, 1.0, 1.0, 1.0]), 2)
    _, mults = space.get_unique_knots_and_multiplicity()
    assert np.asarray(mults).tolist() == [3, 1, 1, 3]


def test_periodic_reverse_is_pointwise_mirror() -> None:
    space = create_uniform_space(2, [4], periodic=True)
    n_basis = space.spaces[0].num_basis
    rng = np.random.default_rng(1)
    spline = Bspline(space, rng.uniform(-1.0, 1.0, (n_basis, 1)))
    reversed_spline = spline.reverse()
    lo, hi = np.asarray(space.domain, dtype=np.float64).ravel()
    t = np.linspace(lo + 1e-9, hi - 1e-9, 7)
    values = np.asarray(spline.evaluate(t), dtype=np.float64)
    mirrored = np.asarray(reversed_spline.evaluate(lo + hi - t), dtype=np.float64)
    np.testing.assert_allclose(values, mirrored, atol=1e-12)


def test_bspline_evaluate_1d_rejects_out_of_domain() -> None:
    space = create_uniform_space(2, [4])
    n_basis = space.spaces[0].num_basis
    spline = Bspline(space, np.ones((n_basis, 1)))
    with pytest.raises(ValueError, match="domain|outside"):
        spline.evaluate(np.array([-0.5]))


def test_to_open_preserves_asymmetric_unclamped_spline() -> None:
    space = BsplineSpace([BsplineSpace1D(np.array([-0.5, 0.0, 0.5, 1.0, 1.0]), 1)])
    n_basis = space.spaces[0].num_basis
    rng = np.random.default_rng(2)
    spline = Bspline(space, rng.uniform(-1.0, 1.0, (n_basis, 1)))
    opened = spline.to_open_bspline()
    np.testing.assert_allclose(
        np.asarray(opened.space.domain, dtype=np.float64),
        np.asarray(space.domain, dtype=np.float64),
    )
    t = np.linspace(1e-9, 1.0 - 1e-9, 9)
    np.testing.assert_allclose(
        np.asarray(opened.evaluate(t), dtype=np.float64),
        np.asarray(spline.evaluate(t), dtype=np.float64),
        atol=1e-12,
    )


def test_l2_project_boundary_interpolation_2d_reproduces_space_function() -> None:
    space = create_uniform_space(2, [5, 5])

    def func(lattice: PointsLattice) -> npt.NDArray[np.float64]:
        pts = np.asarray(lattice.get_all_points(), dtype=np.float64)
        result: npt.NDArray[np.float64] = pts[:, 0] ** 2 + 0.5 * pts[:, 1]
        return result

    projected = l2_project_bspline(func, space, boundary_interpolation=True)
    pts = np.random.default_rng(3).uniform(0.0, 1.0, (60, 2))
    exact = pts[:, 0] ** 2 + 0.5 * pts[:, 1]
    values = np.asarray(projected.evaluate(pts), dtype=np.float64).ravel()
    np.testing.assert_allclose(values, exact, atol=1e-8)


def test_unique_knots_cache_is_not_corruptible() -> None:
    space = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]), 2)
    unique, _ = space.get_unique_knots_and_multiplicity()
    unique_arr = np.asarray(unique)
    if unique_arr.flags.writeable:
        unique_arr[0] = 99.0
    fresh, _ = space.get_unique_knots_and_multiplicity()
    assert float(np.asarray(fresh)[0]) == 0.0


def test_snap_knots_false_does_not_freeze_caller_array() -> None:
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    BsplineSpace1D(knots, 1, snap_knots=False)
    assert knots.flags.writeable


def test_degree0_extraction_operator() -> None:
    space = create_uniform_space(0, [4])
    extraction = SpanwiseElementExtraction(space, "bezier")
    operator = np.asarray(extraction.operator((0,)), dtype=np.float64)
    np.testing.assert_allclose(operator, np.ones((1, 1)))


def test_cm1_interior_knot_derivative_emits_no_warning() -> None:
    knots = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    n_basis = space.spaces[0].num_basis
    spline = Bspline(space, np.arange(n_basis, dtype=np.float64)[:, None])
    derivative = spline.derivative(0)
    values = np.asarray(derivative.evaluate(np.array([0.25, 0.75])), dtype=np.float64)
    assert np.all(np.isfinite(values))


# ---------------------------------------------------------------------------
# Known-bug regressions: quadrature / lattice / VTK ordering
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="get_chebyshev_gauss_2nd_kind_1d pairs Chebyshev-Lobatto nodes with "
    "Gauss-Chebyshev-U interior-node weights: the rule has no polynomial accuracy",
)
def test_chebyshev_gauss_2nd_kind_integrates_quadratic() -> None:
    pts, wts = get_chebyshev_gauss_2nd_kind_1d(5)
    x = np.asarray(pts, dtype=np.float64)
    w = np.asarray(wts, dtype=np.float64)
    mapped = 2.0 * x - 1.0  # rule lives on [0, 1]; weight fn is sqrt(1 - (2x-1)^2)
    assert abs(float(np.sum(w)) - pi / 4.0) < 1e-12  # weight-function mass
    assert abs(float(np.sum(w * mapped**2)) - pi / 16.0) < 1e-10


@pytest.mark.xfail(
    strict=True,
    reason="get_all_points(order='F') uses meshgrid(indexing='xy'), which only swaps "
    "the first two axes: for dim >= 3 the first index does not vary fastest",
)
def test_points_lattice_f_order_first_axis_fastest_3d() -> None:
    lattice = PointsLattice(
        [np.array([0.0, 1.0]), np.array([10.0, 11.0]), np.array([20.0, 21.0, 22.0])]
    )
    pts = np.asarray(lattice.get_all_points(order="F"), dtype=np.float64)
    assert pts[0].tolist() == [0.0, 10.0, 20.0]
    assert pts[1].tolist() == [1.0, 10.0, 20.0]  # axis 0 must vary fastest
    assert pts[2].tolist() == [0.0, 11.0, 20.0]


@pytest.mark.xfail(
    strict=True,
    reason="VTK high-order quad ordering: edge 2 must run in increasing i and the "
    "interior block i-fastest (vtkHigherOrderQuadrilateral); pantr reverses edge 2 and "
    "emits the interior j-fastest, so cubic-and-higher quads render wrong",
)
def test_vtk_quad_ordering_matches_vtk_convention_cubic() -> None:
    perm = np.asarray(vtk_ordering_quad(3, 3), dtype=np.int64)
    ij = [divmod(int(flat), 4) for flat in perm]  # tp flat index is i-major
    expected = [
        (0, 0),
        (3, 0),
        (3, 3),
        (0, 3),  # corners
        (1, 0),
        (2, 0),  # edge 0 (+i)
        (3, 1),
        (3, 2),  # edge 1 (+j)
        (1, 3),
        (2, 3),  # edge 2 (+i, NOT reversed)
        (0, 1),
        (0, 2),  # edge 3 (+j)
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),  # interior, i fastest
    ]
    assert ij == expected


@pytest.mark.xfail(
    strict=True,
    reason="VTK high-order hex ordering: the six face blocks must come in the order "
    "(i=0, i=pu, j=0, j=pv, k=0, k=pw) per vtkHigherOrderHexahedron; pantr emits "
    "(k=0, k=pw, j=0, i=pu, j=pv, i=0), so every quadratic-and-higher hex renders wrong",
)
def test_vtk_hex_face_block_order_quadratic() -> None:
    perm = np.asarray(vtk_ordering_hex(2, 2, 2), dtype=np.int64)
    # Degree (2,2,2): 8 corners + 12 edge nodes, then the 6 face centres at
    # positions 20..25. tp flat index is i-major: flat = i*9 + j*3 + k.
    face_centres = perm[20:26].tolist()
    expected = [
        0 * 9 + 1 * 3 + 1,  # i = 0  -> (0,1,1) = 4
        2 * 9 + 1 * 3 + 1,  # i = pu -> (2,1,1) = 22
        1 * 9 + 0 * 3 + 1,  # j = 0  -> (1,0,1) = 10
        1 * 9 + 2 * 3 + 1,  # j = pv -> (1,2,1) = 16
        1 * 9 + 1 * 3 + 0,  # k = 0  -> (1,1,0) = 12
        1 * 9 + 1 * 3 + 2,  # k = pw -> (1,1,2) = 14
    ]
    assert face_centres == expected


# ---------------------------------------------------------------------------
# Property tests: THB-spline basis identities (review coverage gaps)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("regularity", "admissible", "seed"),
    [(0, None, 11), (None, 2, 12)],
    ids=["c0-ungraded", "smooth-admissible2"],
)
def test_thb_partition_of_unity_randomized(
    regularity: int | None, admissible: int | None, seed: int
) -> None:
    """THB basis sums to one on random points of every active cell."""
    space = _make_thb(2, 2, 5, 2, seed, regularity=regularity, admissible=admissible)
    rng = np.random.default_rng(seed + 100)
    for cid in range(0, space.grid.num_cells, 3):
        pts = _cell_points(space, cid, 4, rng)
        values, _ = space.tabulate_basis(cid, pts)
        np.testing.assert_allclose(np.asarray(values).sum(axis=-1), 1.0, atol=1e-10)
        assert float(np.asarray(values).min()) > -1e-12


def test_thb_derivatives_match_finite_differences() -> None:
    """tabulate_basis_derivatives agrees with central differences of tabulate_basis."""
    space = _make_thb(2, 2, 5, 2, seed=21)
    rng = np.random.default_rng(22)
    step = 1e-6
    for cid in range(0, space.grid.num_cells, max(1, space.grid.num_cells // 5)):
        lo, hi = space.grid.cell_bounds(cid)
        pts = lo + (hi - lo) * rng.uniform(0.3, 0.7, size=(3, 2))
        for axis in range(2):
            orders = tuple(1 if k == axis else 0 for k in range(2))
            derivs, _ = space.tabulate_basis_derivatives(cid, pts, orders)
            plus = pts.copy()
            plus[:, axis] += step
            minus = pts.copy()
            minus[:, axis] -= step
            v_plus, _ = space.tabulate_basis(cid, plus)
            v_minus, _ = space.tabulate_basis(cid, minus)
            fd = (np.asarray(v_plus) - np.asarray(v_minus)) / (2.0 * step)
            np.testing.assert_allclose(np.asarray(derivs), fd, atol=5e-5, rtol=1e-4)


def test_multilevel_extraction_matches_tabulate() -> None:
    """C^eps applied to Bernstein values reproduces tabulate_basis on every cell."""
    space = _make_thb(2, 2, 5, 2, seed=31)
    extraction = MultiLevelExtraction(space, "bezier")
    rng = np.random.default_rng(32)
    for cid in range(0, space.grid.num_cells, max(1, space.grid.num_cells // 6)):
        lo, hi = space.grid.cell_bounds(cid)
        ref = rng.uniform(0.05, 0.95, size=(4, 2))
        pts = lo + (hi - lo) * ref
        operator = np.asarray(extraction.operator(cid), dtype=np.float64)
        bern_u = np.asarray(tabulate_bernstein_1d(2, ref[:, 0]), dtype=np.float64)
        bern_v = np.asarray(tabulate_bernstein_1d(2, ref[:, 1]), dtype=np.float64)
        bern = (bern_u[:, :, None] * bern_v[:, None, :]).reshape(4, -1)
        via_extraction = bern @ operator.T
        direct, _ = space.tabulate_basis(cid, pts)
        np.testing.assert_allclose(via_extraction, np.asarray(direct), atol=1e-10)


def test_thb_admissible_refine_bounds_function_levels() -> None:
    """After class-2 graded refinement, functions on any cell span <= 2 levels.

    Per Carraturo et al. (2019), Definition 3.2, admissibility counts the truncated
    basis functions taking *non-zero values* on an element -- `tabulate_basis` also
    lists functions whose untruncated support touches the cell but that truncation
    zeroes there, so those are filtered out by value.
    """
    space = _make_thb(2, 2, 6, 0, seed=41)
    rng = np.random.default_rng(42)
    for _ in range(3):
        n_cells = space.grid.num_cells
        ids = rng.choice(n_cells, size=max(1, n_cells // 8), replace=False)
        space = space.refine(ids, admissible_class=2)
    assert space.num_levels >= 3  # the chain actually went deep
    for cid in range(space.grid.num_cells):
        values, dofs = space.tabulate_basis(cid, _cell_points(space, cid, 6, rng))
        values_arr = np.asarray(values, dtype=np.float64)
        nonzero = [
            int(d)
            for col, d in enumerate(np.asarray(dofs))
            if float(np.abs(values_arr[:, col]).max()) > 1e-12
        ]
        levels = sorted(
            {int(np.searchsorted(space._func_offset, d, side="right")) - 1 for d in nonzero}
        )
        assert levels[-1] - levels[0] + 1 <= 2


def test_thb_prolongation_preserves_function() -> None:
    """P @ u represents the same function after (ungraded) refinement."""
    space = _make_thb(1, 2, 8, 1, seed=51)
    rng = np.random.default_rng(52)
    ids = rng.choice(space.grid.num_cells, size=2, replace=False)
    fine = space.refine(ids, admissible_class=None)
    prolongation = space.prolongation_to(fine)
    coarse_coeffs = rng.uniform(-1.0, 1.0, space.num_total_basis)
    coarse = THBSpline(space, coarse_coeffs)
    refined = THBSpline(fine, prolongation @ coarse_coeffs)
    pts = rng.uniform(0.01, 0.99, size=(40, 1))
    np.testing.assert_allclose(
        np.asarray(coarse.evaluate(pts), dtype=np.float64),
        np.asarray(refined.evaluate(pts), dtype=np.float64),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Property tests: distributed local spaces under adversarial partitions
# ---------------------------------------------------------------------------


def _adversarial_partitions(n_cells: int, n_parts: int, seed: int) -> list[Partition]:
    """Scatter, stripe, and single-cell-rank ownership patterns."""
    rng = np.random.default_rng(seed)
    owners = [
        rng.integers(0, n_parts, n_cells).astype(np.int32),
        (np.arange(n_cells) * n_parts // n_cells).astype(np.int32),
    ]
    lone = np.zeros(n_cells, dtype=np.int32)
    lone[int(rng.integers(0, n_cells))] = 1
    owners.append(lone)
    return [Partition(owner, int(owner.max()) + 1) for owner in owners]


def _assert_local_matches_global_thb(
    space: THBSplineSpace, partition: Partition, rng: np.random.Generator
) -> None:
    """Local basis values on owned cells must reproduce the global ones exactly."""
    for rank in range(partition.n_parts):
        if partition.owned_cells(rank).size == 0:
            continue
        local = build_local(space, partition, rank)
        local_space = local.space
        assert isinstance(local_space, THBSplineSpace)
        for local_cell in np.flatnonzero(local.owned_cell_mask):
            global_cell = int(local.local_to_global_cell[local_cell])
            pts = _cell_points(space, global_cell, 3, rng)
            g_values, g_dofs = space.tabulate_basis(global_cell, pts)
            l_values, l_dofs = local_space.tabulate_basis(int(local_cell), pts)
            global_cols = {
                int(d): np.asarray(g_values)[:, j] for j, d in enumerate(np.asarray(g_dofs))
            }
            local_cols: dict[int, npt.NDArray[np.float64]] = {}
            for j, local_dof in enumerate(np.asarray(l_dofs)):
                mapped = int(local.local_to_global_dof[int(local_dof)])
                column = np.asarray(l_values, dtype=np.float64)[:, j]
                if mapped >= 0:
                    local_cols[mapped] = column
                else:  # unmapped boundary dofs must vanish on owned cells
                    np.testing.assert_allclose(column, 0.0, atol=1e-12)
            for dof, column in global_cols.items():
                local_column = local_cols.get(dof)
                if local_column is None:
                    np.testing.assert_allclose(column, 0.0, atol=1e-12)
                else:
                    np.testing.assert_allclose(local_column, column, atol=1e-10)


def test_build_local_thb_matches_global_adversarial() -> None:
    space = _make_thb(2, 2, 5, 2, seed=61)
    rng = np.random.default_rng(62)
    for partition in _adversarial_partitions(space.grid.num_cells, 3, seed=63):
        _assert_local_matches_global_thb(space, partition, rng)


def test_thb_dof_ownership_exclusive_and_complete() -> None:
    space = _make_thb(2, 2, 5, 2, seed=71)
    for partition in _adversarial_partitions(space.grid.num_cells, 3, seed=72):
        owners = np.asarray(_thb_dof_owner(space, partition))
        assert int((owners < 0).sum()) == 0  # no dead dofs when every cell is active
        counts = np.zeros(space.num_total_basis, dtype=np.int64)
        for rank in range(partition.n_parts):
            if partition.owned_cells(rank).size == 0:
                continue
            local = build_local(space, partition, rank)
            owned = local.local_to_global_dof[local.owned_dof_mask]
            counts += np.bincount(owned, minlength=space.num_total_basis)
        assert np.all(counts == 1)


def test_build_local_tp_matches_global_with_multiplicities() -> None:
    """TP local spaces match globally even with random interior knot multiplicities."""
    rng = np.random.default_rng(81)
    spaces_1d = []
    for _ in range(2):
        knots: list[float] = [0.0] * 3
        for u in np.linspace(0.0, 1.0, 6)[1:-1]:
            knots.extend([float(u)] * int(rng.integers(1, 3)))
        knots.extend([1.0] * 3)
        spaces_1d.append(BsplineSpace1D(np.array(knots), 2))
    space = BsplineSpace(spaces_1d)
    n_cells = space.num_total_intervals
    num_basis = space.num_basis
    degrees = space.degrees

    grid = tensor_product_grid(space)
    for partition in _adversarial_partitions(n_cells, 3, seed=82):
        global_owner = np.asarray(dof_owner(space, partition))
        assert int((global_owner < 0).sum()) == 0
        counts = np.zeros(space.num_total_basis, dtype=np.int64)
        for rank in range(partition.n_parts):
            if partition.owned_cells(rank).size == 0:
                continue
            local = build_local(space, partition, rank)
            owned = local.local_to_global_dof[local.owned_dof_mask]
            counts += np.bincount(owned, minlength=space.num_total_basis)
            local_space = local.space
            assert isinstance(local_space, BsplineSpace)
            local_num_basis = local_space.num_basis
            for local_cell in np.flatnonzero(local.owned_cell_mask):
                global_cell = int(local.local_to_global_cell[local_cell])
                lo, hi = grid.cell_bounds(global_cell)
                pts = lo + (hi - lo) * rng.uniform(0.05, 0.95, size=(2, 2))
                g_values, g_first = space.tabulate_basis(pts)
                l_values, l_first = local_space.tabulate_basis(pts)
                for pt_idx in range(pts.shape[0]):
                    for off_u in range(degrees[0] + 1):
                        for off_v in range(degrees[1] + 1):
                            g_multi = (
                                int(np.asarray(g_first)[pt_idx, 0]) + off_u,
                                int(np.asarray(g_first)[pt_idx, 1]) + off_v,
                            )
                            g_dof = int(np.ravel_multi_index(g_multi, num_basis))
                            l_multi = (
                                int(np.asarray(l_first)[pt_idx, 0]) + off_u,
                                int(np.asarray(l_first)[pt_idx, 1]) + off_v,
                            )
                            l_dof = int(np.ravel_multi_index(l_multi, local_num_basis))
                            assert int(local.local_to_global_dof[l_dof]) == g_dof
                            np.testing.assert_allclose(
                                float(np.asarray(l_values)[pt_idx, off_u, off_v]),
                                float(np.asarray(g_values)[pt_idx, off_u, off_v]),
                                atol=1e-12,
                            )
        assert np.all(counts == 1)
