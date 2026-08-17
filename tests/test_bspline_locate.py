"""Tests for physical-to-parametric point inversion (``Bspline.locate``)."""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._bspline_locate import (
    _cell_midpoints,
    _cell_parametric_bounds,
    _cell_physical_bounds,
    _geometric_scale,
    _locate_context,
    _LocateContext,
    _nearest_corner_starts,
    _newton_refine,
)
from pantr.geometry import AABB
from pantr.tolerance import get_default
from pantr.transform import AffineTransform

_XI_REL_ATOL: float = 1e-10
"""
Absolute tolerance on a recovered parametric coordinate, as a multiple of the geometry's
scale (the larger of its control-point bounding-box diagonal and its largest coordinate
magnitude).

The inversion stops when ``||F(xi) - x||_2 <= 1e-12 * scale``, so the coordinate error is
that residual divided by the smallest singular value of the Jacobian:
``1e-12 * scale / sigma_min``. Every geometry below has ``sigma_min`` of order one, and
the measured worst case over them is ``2.8e-12`` on a geometry of scale ``2.83``, i.e.
``1e-12 * scale`` exactly as predicted. The ``1e-10`` factor leaves two decades of margin
for a less favourable Jacobian.
"""

_XI_ATOL_FLOAT32: float = 1e-4
"""
Absolute tolerance on a recovered parametric coordinate for a ``float32`` B-spline.

The convergence threshold is :func:`pantr.tolerance.get_default` for ``float32``
(``1e-6``) times the geometric scale, so the coordinate error is four orders of magnitude
larger than in the ``float64`` case. Measured worst case over the cases below:
``5.2e-6``; the ``1e-4`` leaves a factor of 19.
"""

_CYCLING_PARAMETER: npt.NDArray[np.float64] = np.array([[0.05411231, 0.64513102]])
"""
The parameter whose image an undamped Newton iteration could not recover.

An ordinary interior point of :func:`_warped_patch`'s default geometry, kept as the exact
triggering datum rather than re-searched for. Its image has one candidate cell, 3, which
is the cell that contains it, so there is no second start to fall back on. From that
cell's midpoint the second undamped step is 67 times longer than the residual it removes;
the iterate leaves the basin and the clamp to the parametric domain then pins it in a
period-2 cycle between ``(1, 1)`` and ``(0.403985, 0.995004)``, whose residual is
``0.77``. Being periodic rather than slow is what made the iteration budget useless, and
a residual seven decades above any plausible threshold is what made the tolerance useless.
"""


def _quarter_annulus(
    r_in: float = 1.0,
    r_out: float = 2.0,
    shift: tuple[float, float] = (0.0, 0.0),
    dtype: npt.DTypeLike = np.float64,
) -> Bspline:
    """Return the exact quarter annulus: a degree-2 rational arc times a linear radius.

    The angular direction is the standard degree-2 NURBS quarter circle (three control
    points, middle weight ``sqrt(2) / 2``), so ``|F(u, v) - shift| == r_in + v *
    (r_out - r_in)`` holds exactly, not approximately.
    """
    weight = np.sqrt(2.0) / 2.0
    arc = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    weights = np.array([1.0, weight, 1.0])
    radii = np.array([r_in, r_out])
    cp = np.empty((3, 2, 3), dtype=np.float64)
    for i in range(3):
        for j in range(2):
            cp[i, j, :2] = weights[i] * (radii[j] * arc[i] + np.asarray(shift))
            cp[i, j, 2] = weights[i]
    angular = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=dtype), 2)
    radial = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype), 1)
    return Bspline(BsplineSpace([angular, radial]), cp.astype(dtype), is_rational=True)


def _greville_abscissae(space: BsplineSpace1D) -> npt.NDArray[np.float64]:
    """Return the Greville abscissae of a 1-D space, one per basis function."""
    knots = np.asarray(space.knots, dtype=np.float64)
    degree = space.degree
    return np.array(
        [knots[i + 1 : i + degree + 1].mean() for i in range(space.num_basis)], dtype=np.float64
    )


def _identity_map(
    knots: npt.ArrayLike, degree: int, dim: int, dtype: npt.DTypeLike = np.float64
) -> Bspline:
    """Return the identity mapping of the knot vector's box, from Greville abscissae."""
    spaces = [BsplineSpace1D(np.asarray(knots, dtype=dtype), degree) for _ in range(dim)]
    mesh = np.meshgrid(*[_greville_abscissae(sub) for sub in spaces], indexing="ij")
    cp = np.stack(mesh, axis=-1) if dim > 1 else mesh[0][:, np.newaxis]
    return Bspline(BsplineSpace(spaces), np.ascontiguousarray(cp.astype(dtype)))


def _perturbed_patch(
    degree: int, n_cells: int, dim: int, seed: int, dtype: npt.DTypeLike = np.float64
) -> Bspline:
    """Return an injective patch: the identity map plus a bounded random perturbation.

    The perturbation (``0.12`` of a unit cell) is small enough to keep the Jacobian
    determinant positive, which the tests that compare recovered coordinates assert
    directly rather than assume.
    """
    rng = np.random.default_rng(seed)
    knots = np.concatenate(
        [np.zeros(degree + 1), np.arange(1.0, n_cells), np.full(degree + 1, float(n_cells))]
    )
    spaces = [BsplineSpace1D(knots.astype(dtype), degree) for _ in range(dim)]
    mesh = np.meshgrid(*[_greville_abscissae(sub) for sub in spaces], indexing="ij")
    cp = np.stack(mesh, axis=-1) if dim > 1 else mesh[0][:, np.newaxis]
    cp = cp + 0.12 * rng.uniform(-1.0, 1.0, size=cp.shape)
    return Bspline(BsplineSpace(spaces), np.ascontiguousarray(cp.astype(dtype)))


def _warped_patch(dim: int = 2, degree: int = 1, n_elem: int = 6, offset: float = 0.0) -> Bspline:
    """Return the identity map of the unit box plus a smooth 15 % sinusoidal warp.

    Control points sit on a uniform lattice of the unit box and are displaced in every
    coordinate by ``0.15 * sin(2 * pi * sum(coords))``, then translated by ``offset``.

    The defaults are the exact geometry the undamped Newton iteration diverged on: a
    degree-1 bivariate map on 6 uniform elements at unit scale.

    The warp is smooth and the Jacobian is well conditioned along it, but the family is
    **not** injective for ``dim >= 2``, which is worth knowing before reading a result off
    it. The displacement field ``y + a * sin(2 * pi * sum(y))`` has Jacobian
    ``I + 2 * pi * a * cos(2 * pi * sum(y)) * ones``, whose eigenvalues are 1 and
    ``1 + 2 * pi * a * dim * cos(...)``; at ``a == 0.15`` the second turns negative once
    ``dim >= 2``. Measured over the family: ``det J`` reaches ``-0.56`` at ``dim == 2`` and
    ``-1.34`` at ``dim == 3``. So a query built from this family has a preimage by
    construction, but not a unique one, and only ``F(ref_coords) == points`` may be asserted
    of what comes back.
    """
    knots = np.concatenate([np.zeros(degree), np.linspace(0.0, 1.0, n_elem + 1), np.ones(degree)])
    space = BsplineSpace([BsplineSpace1D(knots, degree) for _ in range(dim)])
    axes = [np.linspace(0.0, 1.0, n) for n in space.num_basis]
    cp = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    cp = cp + 0.15 * np.sin(2.0 * np.pi * cp.sum(axis=-1))[..., None] + offset
    return Bspline(space, np.ascontiguousarray(cp))


def _folded_patch() -> Bspline:
    """Return a patch whose Jacobian determinant changes sign, so it is not injective."""
    knots = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    sub = BsplineSpace1D(knots, 2)
    space = BsplineSpace([sub, sub])
    u_grev, v_grev = (np.linspace(0.0, 1.0, n) for n in space.num_basis)
    u_mesh, v_mesh = np.meshgrid(u_grev, v_grev, indexing="ij")
    cp = np.stack([u_mesh + 0.9 * v_mesh, v_mesh + 0.9 * np.sin(3.0 * u_mesh)], axis=-1)
    return Bspline(space, np.ascontiguousarray(cp))


def _stretched_patch() -> Bspline:
    """Return an injective sheared patch whose per-cell physical boxes overlap heavily."""
    knots = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    sub = BsplineSpace1D(knots, 2)
    space = BsplineSpace([sub, sub])
    u_grev, v_grev = (np.linspace(0.0, 1.0, n) for n in space.num_basis)
    u_mesh, v_mesh = np.meshgrid(u_grev, v_grev, indexing="ij")
    cp = np.stack([u_mesh + 0.9 * v_mesh, v_mesh + 0.25 * np.sin(3.0 * u_mesh)], axis=-1)
    return Bspline(space, np.ascontiguousarray(cp))


def _collapsed_edge_patch() -> Bspline:
    """Return the bilinear patch ``F(u, v) = (u * v, v)``, whose ``v == 0`` edge collapses.

    Its Jacobian ``[[v, u], [0, 1]]`` is singular along that edge, and its image is the
    triangle ``0 <= x <= y <= 1`` rather than the full control-point box.
    """
    cp = np.array([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]], dtype=np.float64)
    sub = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)
    return Bspline(BsplineSpace([sub, sub]), cp)


def _scale_of(spline: Bspline) -> float:
    """Return the geometric scale the default tolerance is expressed in."""
    return _geometric_scale(*_cell_physical_bounds(spline))


def _cache_of(spline: Bspline) -> _LocateContext | None:
    """Return the point-inversion cache attribute of ``spline``.

    Read through a function rather than inline: an inline ``assert spline._locate_cache is
    None`` narrows the attribute's type for the rest of the test, and mypy then calls the
    later assertions unreachable. A call expression is not narrowed.
    """
    return spline._locate_cache


def _sample_parametric(
    spline: Bspline, n_pts: int, seed: int, margin: float = 0.02
) -> npt.NDArray[np.float64]:
    """Return ``n_pts`` random parametric points, kept ``margin`` away from the boundary."""
    domain = np.asarray(spline.space.domain, dtype=np.float64)
    lo, hi = domain[:, 0], domain[:, 1]
    rng = np.random.default_rng(seed)
    unit = rng.uniform(margin, 1.0 - margin, size=(n_pts, spline.dim))
    return lo + (hi - lo) * unit


def _evaluate_at(spline: Bspline, xi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate ``spline`` at ``(n, dim)`` parametric points, always returning ``(n, rank)``."""
    pts = np.ascontiguousarray((xi[:, 0] if spline.dim == 1 else xi).astype(spline.dtype))
    values = spline.evaluate(pts)
    return np.asarray(values, dtype=np.float64).reshape(xi.shape[0], spline.rank)


def _jacobian_determinants(spline: Bspline, xi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``det J`` at each parametric point, for an injectivity guard."""
    pts = np.ascontiguousarray((xi[:, 0] if spline.dim == 1 else xi).astype(spline.dtype))
    columns = []
    for axis in range(spline.dim):
        orders = [0] * spline.dim
        orders[axis] = 1
        column = spline.evaluate_derivatives(pts, orders)
        columns.append(np.asarray(column, dtype=np.float64).reshape(xi.shape[0], spline.rank))
    return np.asarray(np.linalg.det(np.stack(columns, axis=-1)), dtype=np.float64)


def _assert_cell_contains(
    spline: Bspline,
    cell_ids: npt.NDArray[np.int64],
    ref_coords: npt.NDArray[np.float64],
) -> None:
    """Assert every reported cell's parametric box contains its reported coordinates.

    An independent check of the ``cell_ids`` half of the result: it reads the per-axis
    breakpoints straight off the knot vectors instead of asking the grid again, which is
    what produced the ids.
    """
    multi = np.unravel_index(cell_ids, spline.space.num_intervals)
    for axis, sub in enumerate(spline.space.spaces):
        breakpoints = np.asarray(
            sub.get_unique_knots_and_multiplicity(in_domain=True)[0], dtype=np.float64
        )
        index = multi[axis]
        assert np.all(breakpoints[index] <= ref_coords[:, axis])
        assert np.all(ref_coords[:, axis] <= breakpoints[index + 1])


class TestRoundtrip:
    """Inverting ``evaluate`` recovers the parametric coordinates it was called with."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    @pytest.mark.parametrize("dim", [1, 2])
    def test_perturbed_identity(self, degree: int, dim: int) -> None:
        """A perturbed identity patch inverts to the generating coordinates."""
        spline = _perturbed_patch(degree, 4, dim, seed=100 * degree + dim)
        xi_true = _sample_parametric(spline, 200, seed=1)
        assert np.all(_jacobian_determinants(spline, xi_true) > 0.0), "patch must be injective"
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points)

        atol = _XI_REL_ATOL * _scale_of(spline)
        assert np.all(cell_ids >= 0)
        np.testing.assert_allclose(ref_coords, xi_true, atol=atol, rtol=0.0)
        np.testing.assert_allclose(_evaluate_at(spline, ref_coords), points, atol=atol, rtol=0.0)
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_three_dimensional(self) -> None:
        """A 3-D perturbed identity volume inverts to the generating coordinates."""
        spline = _perturbed_patch(2, 3, 3, seed=77)
        xi_true = _sample_parametric(spline, 200, seed=2)
        assert np.all(_jacobian_determinants(spline, xi_true) > 0.0)
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points)

        assert np.all(cell_ids >= 0)
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_repeated_interior_knots(self) -> None:
        """A knot vector with a repeated interior knot (a C0 line) inverts correctly."""
        knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        spline = _identity_map(knots, 3, 2)
        xi_true = _sample_parametric(spline, 200, seed=3)
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points)

        assert np.all(cell_ids >= 0)
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_one_dimensional_input_is_a_list_of_scalars(self) -> None:
        """For ``rank == 1`` a 1-D input is read as ``n`` points, matching ``evaluate``."""
        spline = _perturbed_patch(3, 4, 1, seed=5)
        xi_true = _sample_parametric(spline, 20, seed=4)
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points[:, 0])

        assert cell_ids.shape == (20,)
        assert ref_coords.shape == (20, 1)
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )

    def test_single_point_as_a_flat_array(self) -> None:
        """A flat array of ``rank`` coordinates is read as one point."""
        spline = _perturbed_patch(2, 3, 2, seed=6)
        xi_true = _sample_parametric(spline, 1, seed=7)
        point = _evaluate_at(spline, xi_true)[0]

        cell_ids, ref_coords = spline.locate(point)

        assert cell_ids.shape == (1,)
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )

    def test_empty_input(self) -> None:
        """An empty batch returns empty results rather than raising."""
        spline = _perturbed_patch(2, 2, 2, seed=8)
        cell_ids, ref_coords = spline.locate(np.zeros((0, 2)))
        assert cell_ids.shape == (0,)
        assert ref_coords.shape == (0, 2)

    def test_outputs_are_read_only(self) -> None:
        """Both returned arrays are frozen, like the rest of pantr's exposed arrays."""
        spline = _perturbed_patch(2, 2, 2, seed=9)
        points = _evaluate_at(spline, _sample_parametric(spline, 5, seed=10))
        cell_ids, ref_coords = spline.locate(points)
        assert not cell_ids.flags.writeable
        assert not ref_coords.flags.writeable


class TestRationalAnnulus:
    """Inversion of an exact NURBS quarter annulus, interior and boundary."""

    def test_geometry_is_the_exact_annulus(self) -> None:
        """Guard on the fixture: the radius is exactly affine in ``v``."""
        annulus = _quarter_annulus()
        xi = _sample_parametric(annulus, 100, seed=11)
        radii = np.linalg.norm(_evaluate_at(annulus, xi), axis=1)
        np.testing.assert_allclose(radii, 1.0 + xi[:, 1], atol=1e-15, rtol=0.0)

    def test_interior_points(self) -> None:
        """Interior points invert to the generating coordinates."""
        annulus = _quarter_annulus()
        xi_true = _sample_parametric(annulus, 200, seed=11)
        points = _evaluate_at(annulus, xi_true)

        cell_ids, ref_coords = annulus.locate(points)

        assert np.all(cell_ids == 0), "a single-cell patch has only cell 0"
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(annulus), rtol=0.0
        )

    def test_boundary_and_corner_points(self) -> None:
        """Points on the domain boundary, including corners, are found."""
        annulus = _quarter_annulus()
        xi_true = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
            ]
        )
        points = _evaluate_at(annulus, xi_true)

        cell_ids, ref_coords = annulus.locate(points)

        assert np.all(cell_ids == 0)
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(annulus), rtol=0.0
        )

    def test_float32_spline(self) -> None:
        """A ``float32`` annulus inverts to ``float32``-limited accuracy."""
        annulus32 = _quarter_annulus(dtype=np.float32)
        xi_true = _sample_parametric(annulus32, 200, seed=12)
        points = _evaluate_at(annulus32, xi_true)

        cell_ids, ref_coords = annulus32.locate(points)

        assert ref_coords.dtype == np.float64
        assert np.all(cell_ids == 0)
        np.testing.assert_allclose(ref_coords, xi_true, atol=_XI_ATOL_FLOAT32, rtol=0.0)


class TestOutsidePoints:
    """Points that are not on the mapping's image report ``-1`` and ``nan``."""

    def test_outside_the_bounding_box(self) -> None:
        """A point far from the geometry is reported as not found."""
        annulus = _quarter_annulus()
        cell_ids, ref_coords = annulus.locate(np.array([[10.0, 10.0], [-3.0, 0.5]]))
        assert cell_ids.tolist() == [-1, -1]
        assert bool(np.all(np.isnan(ref_coords)))

    def test_inside_the_bounding_box_but_off_the_domain(self) -> None:
        """The annulus hole is inside the control-point box yet off the geometry."""
        annulus = _quarter_annulus()
        holes = np.array([[0.0, 0.0], [0.2, 0.2], [0.5, 0.5], [1.9, 1.9]])
        cell_ids, ref_coords = annulus.locate(holes)
        assert cell_ids.tolist() == [-1, -1, -1, -1]
        assert bool(np.all(np.isnan(ref_coords)))

    def test_mixed_batch_keeps_the_found_points(self) -> None:
        """A batch mixing found and unfound points resolves each independently."""
        annulus = _quarter_annulus()
        xi_true = np.array([[0.3, 0.7], [0.8, 0.1]])
        points = np.vstack([_evaluate_at(annulus, xi_true), np.array([[0.0, 0.0], [9.0, 9.0]])])

        cell_ids, ref_coords = annulus.locate(points)

        assert cell_ids.tolist() == [0, 0, -1, -1]
        np.testing.assert_allclose(
            ref_coords[:2], xi_true, atol=_XI_REL_ATOL * _scale_of(annulus), rtol=0.0
        )
        assert bool(np.all(np.isnan(ref_coords[2:])))


class TestCandidateCells:
    """Behaviour when several cells' physical boxes contain the query point."""

    def test_stretched_patch_with_overlapping_boxes(self) -> None:
        """A sheared patch whose cell boxes overlap still resolves every point."""
        from pantr.bspline._bspline_locate import _locate_context  # noqa: PLC0415
        from pantr.geometry import AABB  # noqa: PLC0415

        spline = _stretched_patch()
        xi_true = _sample_parametric(spline, 300, seed=13, margin=0.01)
        assert np.all(_jacobian_determinants(spline, xi_true) > 0.0)
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points)

        # The premise of the test: the AABB test really is ambiguous here.
        context = _locate_context(spline)
        tol_phys = 1e-12 * context.scale
        counts = np.array([context.bvh.query_aabb(AABB(x, x).pad(tol_phys)).size for x in points])
        assert counts.mean() > 2.0, "expected genuinely ambiguous candidate boxes"

        assert np.all(cell_ids >= 0)
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_folded_patch_returns_a_valid_preimage(self) -> None:
        """A non-injective mapping is inverted to *some* preimage, not a specified one.

        This pins the documented contract: what holds is ``F(ref_coords) == points``, not
        ``ref_coords == the coordinates evaluate was called with``.
        """
        spline = _folded_patch()
        xi_true = _sample_parametric(spline, 300, seed=14, margin=0.01)
        determinants = _jacobian_determinants(spline, xi_true)
        assert determinants.min() < 0.0 < determinants.max(), "fixture must actually fold"
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points)

        atol = _XI_REL_ATOL * _scale_of(spline)
        assert np.all(cell_ids >= 0)
        np.testing.assert_allclose(_evaluate_at(spline, ref_coords), points, atol=atol, rtol=0.0)
        assert np.abs(ref_coords - xi_true).max() > atol, (
            "a folded mapping is expected to return a different preimage for some points"
        )
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_collapsed_edge_patch(self) -> None:
        """A patch with a collapsed edge (singular Jacobian there) still inverts."""
        spline = _collapsed_edge_patch()
        # F(u, v) = (u * v, v): the whole v == 0 edge collapses onto the origin, so the
        # last query has a whole family of preimages and any of them is correct.
        points = np.array([[0.25, 0.5], [0.5, 0.75], [0.0, 0.0]])

        cell_ids, ref_coords = spline.locate(points)

        assert np.all(cell_ids == 0)
        np.testing.assert_allclose(
            _evaluate_at(spline, ref_coords), points, atol=_XI_REL_ATOL, rtol=0.0
        )

    def test_singular_jacobian_abandons_the_candidate(self) -> None:
        """A query that drives the iterate onto a singular Jacobian is reported unfound.

        The image of ``F(u, v) = (u * v, v)`` is the triangle ``0 <= x <= y <= 1``, so
        ``(0.5, 0)`` lies outside it while still inside the control-point box. Newton is
        pushed onto the collapsed ``v == 0`` edge, where the smallest singular value of
        the Jacobian is exactly zero (measured), and the candidate is abandoned instead of
        an arbitrary step being taken from a rank-deficient solve.
        """
        spline = _collapsed_edge_patch()

        cell_ids, ref_coords = spline.locate(np.array([[0.5, 0.0]]))

        assert cell_ids.tolist() == [-1]
        assert bool(np.all(np.isnan(ref_coords)))

    def test_off_image_query_ends_by_stalling_not_by_the_rank_guard(self) -> None:
        """The other way a candidate fails: a healthy Jacobian that never converges.

        ``(0.75, 0.25)`` is also outside the triangle, but the iterate settles near
        ``(1, 0.25)`` with a well-conditioned Jacobian, so what ends the attempt is not the
        rank guard. It used to be the iteration budget, burnt one full step at a time; it is
        now the line search, which cannot reduce a residual whose minimum over the domain is
        the distance from the query to the image, and abandons the candidate instead
        (measured: the midpoint start exits through the line search, the corner start
        through the rank guard).
        """
        spline = _collapsed_edge_patch()

        cell_ids, ref_coords = spline.locate(np.array([[0.75, 0.25]]))

        assert cell_ids.tolist() == [-1]
        assert bool(np.all(np.isnan(ref_coords)))

    def test_fully_degenerate_patch_reports_unfound_instead_of_raising(self) -> None:
        """A patch that collapses onto a line is handled, not crashed on.

        ``F(u, v) = (u + v, u + v)`` has ``J = [[1, 1], [1, 1]]`` everywhere, and
        ``numpy.linalg.solve`` raises ``LinAlgError`` on it (verified). The rank guard is
        what turns that into a plain "not found". A query that happens to lie on the
        image is still answered, since the residual test runs before the Jacobian.
        """
        sub = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)
        cp = np.array([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]], dtype=np.float64)
        flat = Bspline(BsplineSpace([sub, sub]), cp)

        cell_ids, ref_coords = flat.locate(np.array([[0.3, 0.9], [1.0, 1.0]]))

        assert cell_ids.tolist() == [-1, 0]
        assert bool(np.all(np.isnan(ref_coords[0])))
        np.testing.assert_allclose(
            _evaluate_at(flat, ref_coords[1:]), np.array([[1.0, 1.0]]), atol=1e-15, rtol=0.0
        )


class TestIdentityMap:
    """The identity mapping inverts in one Newton step, to machine precision."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_machine_precision_but_not_bitwise(self, degree: int) -> None:
        """Recovery is exact to a few ulp of the domain, which is all that is available.

        The control points are the Greville abscissae, so the mapping is the identity
        mathematically but not in floating point: ``F(xi)`` already differs from ``xi`` by
        up to ``1.3e-15`` on a domain of length 3 (measured, degrees 1-4). A bitwise
        round trip is therefore unattainable by any implementation, and the reachable
        statement is a few ulp.
        """
        knots = np.concatenate([np.zeros(degree + 1), [1.0, 2.0], np.full(degree + 1, 3.0)])
        spline = _identity_map(knots, degree, 2)
        xi_true = _sample_parametric(spline, 200, seed=15)

        forward_error = np.abs(_evaluate_at(spline, xi_true) - xi_true).max()
        cell_ids, ref_coords = spline.locate(xi_true)

        assert np.all(cell_ids >= 0)
        # 8 ulp of the domain length: measured worst case is 1.8e-15, i.e. under 3 ulp.
        atol = 8.0 * float(np.finfo(np.float64).eps) * 3.0
        np.testing.assert_allclose(ref_coords, xi_true, atol=atol, rtol=0.0)
        assert forward_error > 0.0, "the discrete identity is not exact in floating point"

    def test_converges_in_a_single_newton_step(self) -> None:
        """One step suffices: the mapping is affine, so the Jacobian is constant."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
        spline = _identity_map(knots, 2, 2)
        xi_true = _sample_parametric(spline, 100, seed=16)

        cell_ids, _ = spline.locate(xi_true, max_iter=1)

        assert np.all(cell_ids >= 0)


class TestNewtonGlobalization:
    """A Newton step that does not reduce the residual is rejected, so no iterate cycles."""

    def test_the_cycling_point_is_found_at_every_tolerance_and_budget(self) -> None:
        """The undamped iteration lost this point at every setting; the damped one must not.

        Neither knob could recover it before: not a tolerance seven decades above the
        default tier, and not ten times the iteration budget. That is the signature of a
        cycle rather than of slow convergence, and it is why both axes are swept here.
        """
        spline = _warped_patch()
        target = _evaluate_at(spline, _CYCLING_PARAMETER)
        scale = _scale_of(spline)

        for factor in (64.0, 4096.0, 1.0e9):
            for max_iter in (20, 200):
                tol = factor * float(np.finfo(np.float64).eps) * scale
                cell_ids, ref_coords = spline.locate(target, tol=tol, max_iter=max_iter)
                assert cell_ids.tolist() == [3], f"lost at tol={factor}*eps*scale, {max_iter=}"
                residual = np.linalg.norm(_evaluate_at(spline, ref_coords) - target)
                assert residual <= tol

    def test_the_recovered_parameter_is_the_preimage(self) -> None:
        """The answer is the parameter the target was built from, not merely some root."""
        spline = _warped_patch()
        target = _evaluate_at(spline, _CYCLING_PARAMETER)
        assert np.all(_jacobian_determinants(spline, _CYCLING_PARAMETER) > 0.0)

        cell_ids, ref_coords = spline.locate(target)

        np.testing.assert_allclose(
            ref_coords, _CYCLING_PARAMETER, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_the_residual_never_increases_over_the_accepted_iterates(self) -> None:
        """Monotonicity, read off the iteration itself rather than assumed.

        Budget ``k`` returns the iterate after at most ``k`` accepted steps, so sweeping
        ``k`` and evaluating the residual at what comes back walks the accepted sequence
        without needing a hook into the solver. Undamped, the sequence rises from
        ``1.05e-2`` to ``7.3e-1`` at the third step and then oscillates.
        """
        spline = _warped_patch()
        target = _evaluate_at(spline, _CYCLING_PARAMETER)
        context = _locate_context(spline)
        tol = get_default(spline.dtype) * context.scale
        candidates = np.sort(context.bvh.query_aabb(AABB(target[0], target[0]).pad(tol)))
        assert candidates.tolist() == [3], "the premise: one candidate, so there is no fallback"
        start = _cell_midpoints(spline.space, candidates)

        norms = [
            float(
                np.linalg.norm(
                    _evaluate_at(spline, _newton_refine(spline, target, start, tol, k)[1]) - target
                )
            )
            for k in range(1, 13)
        ]

        assert np.all(np.diff(norms) <= 0.0), f"residual increased over the iterates: {norms}"
        assert norms[-1] <= tol, f"did not converge within 12 steps: {norms}"

    @pytest.mark.parametrize("offset", [0.0, 1.0e6])
    @pytest.mark.parametrize("n_elem", [3, 6])
    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_the_warped_family_loses_no_point_that_has_a_preimage(
        self, dim: int, degree: int, n_elem: int, offset: float
    ) -> None:
        """Every point built by evaluating the map is recovered, across the whole family.

        Sampling the parameter first and mapping it forward is what makes "has a preimage"
        a fact about each query rather than a hope: the preimage is the parameter the
        target was made from. It is *a* preimage that is required back, not that one --
        this family folds for every ``dim >= 2`` member (measured: ``det J`` reaches
        ``-0.56`` at ``dim == 2``), and a folded mapping sends several parameters to one
        point. The offset column is there because the residual floor follows the coordinate
        magnitude, so a family that converges at unit scale can still lose points at
        ``1e6``.
        """
        spline = _warped_patch(dim, degree, n_elem, offset)
        xi_true = _sample_parametric(spline, 40, seed=1000 * dim + 10 * degree + n_elem)
        points = _evaluate_at(spline, xi_true)
        tol = get_default(spline.dtype) * _scale_of(spline)

        cell_ids, ref_coords = spline.locate(points)

        lost = np.flatnonzero(cell_ids < 0)
        assert lost.size == 0, f"lost {lost.size} of 40 points: {xi_true[lost]}"
        residuals = np.linalg.norm(_evaluate_at(spline, ref_coords) - points, axis=1)
        assert residuals.max() <= tol, f"worst residual {residuals.max():.3e} over tol {tol:.3e}"
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_a_second_start_is_what_recovers_a_boundary_minimum(self) -> None:
        """The one point in the sweep the line search alone cannot reach, and why.

        A monotone iteration from cell 32's midpoint descends to ``(1, 0.4059)``, which is
        the minimum of the residual along the ``u == 1`` face of the domain: the only
        descent direction there points out of the box, and moving inward raises the
        residual, so the root at ``u == 0.9485`` sits behind a ridge in a different basin.
        Damping cannot cross that ridge -- no monotone method can -- and the corner start
        does, which is what this pins.
        """
        spline = _warped_patch()
        xi_true = np.array([[0.9485013, 0.34608885]])
        target = _evaluate_at(spline, xi_true)
        context = _locate_context(spline)
        tol = get_default(spline.dtype) * context.scale
        candidates = np.sort(context.bvh.query_aabb(AABB(target[0], target[0]).pad(tol)))
        assert candidates.tolist() == [32], "the premise: one candidate, so there is no fallback"

        from_midpoint = _newton_refine(
            spline, target, _cell_midpoints(spline.space, candidates), tol, 40
        )
        from_corner = _newton_refine(
            spline, target, _nearest_corner_starts(spline, candidates, target), tol, 40
        )

        assert not bool(from_midpoint[0][0]), "the midpoint start is expected to jam"
        np.testing.assert_allclose(from_midpoint[1][0], [1.0, 0.405897], atol=1e-6, rtol=0.0)
        assert bool(from_corner[0][0]), "the corner start must recover the root"
        np.testing.assert_allclose(
            from_corner[1], xi_true, atol=_XI_REL_ATOL * context.scale, rtol=0.0
        )
        assert spline.locate(target)[0].tolist() == [32]

    def test_the_second_start_is_always_a_corner_of_its_own_cell(self) -> None:
        """The postcondition holds even when no corner can be ranked.

        The nearest-image search keeps the best corner seen so far, and a mapping that
        evaluates to ``nan`` -- nothing rejects a non-finite control point at construction --
        makes every distance ``nan`` and every "is this closer" comparison False. Starting
        the search from a real corner rather than an empty buffer is what keeps the answer a
        point of the cell instead of whatever the allocator returned.
        """
        spline = _warped_patch()
        cells = np.array([0, 17, 35], dtype=np.int64)
        lo, hi = _cell_parametric_bounds(spline.space, cells)

        finite = _nearest_corner_starts(spline, cells, _evaluate_at(spline, 0.5 * (lo + hi)))
        unrankable = _nearest_corner_starts(spline, cells, np.full((3, 2), np.nan))

        for starts in (finite, unrankable):
            assert np.all(np.isfinite(starts))
            assert np.all((starts == lo) | (starts == hi)), "a start must be a cell corner"
            assert np.all(lo <= starts) and np.all(starts <= hi)

    def test_a_second_start_cannot_invent_a_solution(self) -> None:
        """Retrying widens what is reached, never what counts as reached.

        Both queries are off the mapping's image, so no start can drive the residual under
        the threshold; the extra solve must leave them reported as not found. This is the
        guard on the one way a second start could do harm.
        """
        annulus = _quarter_annulus()
        holes = np.array([[0.2, 0.2], [0.5, 0.5]])

        cell_ids, ref_coords = annulus.locate(holes)

        assert cell_ids.tolist() == [-1, -1]
        assert bool(np.all(np.isnan(ref_coords)))


class TestToleranceScaling:
    """The default tolerance follows the geometry's own size, offset included."""

    def test_scale_uses_the_coordinate_magnitude_not_only_the_diagonal(self) -> None:
        """Translating a geometry leaves its diagonal alone but raises its scale."""
        centered = _quarter_annulus()
        shifted = _quarter_annulus(shift=(1.0e6, 1.0e6))
        assert _scale_of(centered) == pytest.approx(2.8284271247461903)
        # The diagonal is translation invariant; the scale must not be.
        assert _scale_of(shifted) > 1.0e6

    @pytest.mark.parametrize("shift", [1.0e3, 1.0e6, 1.0e9])
    def test_offset_geometry_still_inverts(self, shift: float) -> None:
        """A geometry far from the origin is fully recovered with the default tolerance.

        A tolerance taken from the bounding-box diagonal alone would be below the
        residual's own arithmetic floor here (measured: 113 of 200 points lost at a
        ``1e6`` offset), because the floor scales with the coordinate magnitude and the
        diagonal does not.
        """
        spline = _quarter_annulus(shift=(shift, shift))
        xi_true = _sample_parametric(spline, 200, seed=17)
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points)

        assert np.all(cell_ids == 0)
        np.testing.assert_allclose(
            ref_coords, xi_true, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )

    def test_explicit_tolerance_is_an_absolute_distance(self) -> None:
        """A slack ``tol`` accepts a point that is merely near the geometry.

        The image of a square mapping is a full-dimensional region, so "near but not on"
        only exists outside it: the point below sits ``1e-3`` inside the annulus hole,
        just off the ``v == 0`` boundary.
        """
        annulus = _quarter_annulus()
        direction = np.array([[np.cos(0.7), np.sin(0.7)]])
        nearby = (1.0 - 1.0e-3) * direction

        assert annulus.locate(nearby)[0].tolist() == [-1]
        assert annulus.locate(nearby, tol=1.0e-2)[0].tolist() == [0]

    def test_max_iter_is_honoured(self) -> None:
        """Too small an iteration budget reports "not found" instead of a wrong answer.

        The annulus needs 5 Newton steps from a cell midpoint (measured); with a budget
        of 1 the residual test simply never passes.
        """
        annulus = _quarter_annulus()
        points = _evaluate_at(annulus, np.array([[0.05, 0.95], [0.95, 0.05]]))

        assert annulus.locate(points, max_iter=1)[0].tolist() == [-1, -1]
        assert annulus.locate(points, max_iter=30)[0].tolist() == [0, 0]

    @pytest.mark.parametrize("lam", [1.0e-9, 1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6])
    def test_the_scale_is_proportional_to_the_geometry_at_every_size(self, lam: float) -> None:
        """No floor at one: a model in metres and in kilometres get the same relative bar.

        ``max(diagonal, magnitude, 1.0)`` clamped the scale for any geometry smaller than
        a unit, so a millimetre-scale part was held to a thousand times the relative
        accuracy of a metre-scale one, and the tolerance stopped being covariant exactly
        where the rest of this overhaul made it so.
        """
        scaled = _quarter_annulus(r_in=lam, r_out=2.0 * lam)
        assert _scale_of(scaled) == pytest.approx(lam * _scale_of(_quarter_annulus()), rel=1e-12)

    @pytest.mark.parametrize("lam", [1.0e-9, 1.0e-6, 1.0, 1.0e6])
    def test_a_small_geometry_still_inverts(self, lam: float) -> None:
        """Removing the floor tightens the bar on a sub-unit model; it must still be met.

        This is the direction worth guarding. The old floor made the threshold *looser*
        than the geometry warranted below unit size, so nothing down there had ever been
        required to converge to its own scale.
        """
        spline = _quarter_annulus(r_in=lam, r_out=2.0 * lam)
        xi_true = _sample_parametric(spline, 100, seed=23)
        points = _evaluate_at(spline, xi_true)

        cell_ids, ref_coords = spline.locate(points)

        assert np.all(cell_ids == 0)
        np.testing.assert_allclose(ref_coords, xi_true, atol=_XI_REL_ATOL, rtol=0.0)

    def test_a_geometry_with_no_extent_at_the_origin_still_gets_a_positive_scale(self) -> None:
        """The one case with nothing to read a scale off keeps the fallback of one.

        Every control point identical and at the origin leaves both the diagonal and the
        coordinate magnitude at zero, and a tolerance of zero would be unusable.
        """
        sub = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)
        point_map = Bspline(BsplineSpace([sub, sub]), np.zeros((2, 2, 2), dtype=np.float64))
        assert _scale_of(point_map) == 1.0

    def test_a_geometry_with_no_extent_away_from_the_origin_uses_its_magnitude(self) -> None:
        """A degenerate geometry that is somewhere still has a somewhere to be graded at.

        The old floor rounded this up to one whenever the magnitude was smaller, which is
        the same non-covariance in a corner case.
        """
        sub = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)
        cp = np.full((2, 2, 2), 1.0e-6, dtype=np.float64)
        point_map = Bspline(BsplineSpace([sub, sub]), cp)
        assert _scale_of(point_map) == pytest.approx(1.0e-6, rel=1e-12)


class TestNumbaWarmup:
    """``locate`` waits for the import-time JIT warmup, as its sibling entry points do."""

    def test_the_barrier_runs_before_any_kernel_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``locate`` calls the warmup barrier, and calls it before it evaluates anything.

        ``pantr/__init__.py`` compiles its kernels on a background thread, and Numba's
        default workqueue threading layer is not safe against a concurrent ``parallel=True``
        call from another thread: the process *aborts* rather than raising, which takes the
        whole session with it. Every other Layer 2 entry point over parallel kernels calls
        :func:`pantr._numba_compat.wait_for_jit_warmup` first; ``locate`` did not, and
        inverting a batch evaluates often enough to land in the window. Measured before the
        barrier: 4 of 4 runs of this module's 60-case sweep aborted with ``Fatal Python
        error: Aborted`` when it was the first thing a process did, both serially and under
        ``pytest -n 4``; after it, 0 of 4.

        Asserting the *call* rather than the absence of the crash is deliberate. The barrier
        is a once-per-process event, so by the time any in-process test runs the warmup is
        long finished and no in-process check can observe the race; and a subprocess that
        merely inverts a batch does not reproduce it either once the Numba cache is warm
        (measured: 8 of 8 clean without the barrier). What is worth pinning is therefore the
        contract, which this does deterministically: delete the call and this test fails.
        """
        calls: list[str] = []
        module = sys.modules["pantr.bspline._bspline_locate"]
        real_eval_map = module._eval_map

        def _record_barrier() -> None:
            calls.append("barrier")

        def _record_eval(spline: Bspline, xi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            calls.append("evaluate")
            return np.asarray(real_eval_map(spline, xi), dtype=np.float64)

        monkeypatch.setattr(module, "wait_for_jit_warmup", _record_barrier)
        monkeypatch.setattr(module, "_eval_map", _record_eval)

        spline = _warped_patch()
        spline.locate(_evaluate_at(spline, _CYCLING_PARAMETER))

        assert "barrier" in calls, "locate must wait for the JIT warmup"
        assert calls.index("barrier") < calls.index("evaluate"), (
            f"the barrier must precede the first kernel call; got {calls[:3]}"
        )


class TestCellPhysicalBounds:
    """The per-cell physical boxes match an independent brute-force oracle."""

    @pytest.mark.parametrize(
        "spline_factory",
        [
            lambda: _quarter_annulus(),
            lambda: _quarter_annulus().subdivide(3),
            lambda: _perturbed_patch(3, 4, 2, seed=18),
            lambda: _identity_map(
                np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]), 3, 2
            ),
            lambda: _perturbed_patch(2, 2, 3, seed=19),
        ],
    )
    def test_matches_cell_supports_oracle(self, spline_factory: Callable[[], Bspline]) -> None:
        """A running min/max per axis equals a gather over the full per-cell support.

        Bitwise equality is the right assertion here: both routes take the minimum and
        maximum of the same finite set of floats, with no arithmetic in between.
        """
        spline = spline_factory()
        space = spline.space
        lo, hi = _cell_physical_bounds(spline)

        supports = space.cell_supports(np.arange(space.num_total_intervals))
        cp = np.asarray(spline.control_points, dtype=np.float64).reshape(space.num_total_basis, -1)
        physical = cp[:, :-1] / cp[:, -1:] if spline.is_rational else cp
        boxes = physical[supports]

        assert np.array_equal(lo, boxes.min(axis=1))
        assert np.array_equal(hi, boxes.max(axis=1))


class TestCaching:
    """The inversion state is cached, and the in-place mutators invalidate it."""

    def test_second_call_reuses_the_context(self) -> None:
        """The hierarchy is built once per B-spline."""
        spline = _quarter_annulus()
        assert _cache_of(spline) is None
        spline.locate(np.array([[1.2, 0.3]]))
        first = _cache_of(spline)
        assert first is not None
        spline.locate(np.array([[0.3, 1.2]]))
        second = _cache_of(spline)
        assert second is first
        assert second is not None
        assert second.bvh is first.bvh

    @pytest.mark.parametrize("mutate", ["reverse", "permute_directions", "transform"])
    def test_in_place_mutation_invalidates_the_context(self, mutate: str) -> None:
        """Every in-place mutator drops the cache, as it does the Bézier cache.

        Without this a stale hierarchy would survive a mutation that moves the geometry,
        and the next call would silently report points as not found.
        """
        spline = _perturbed_patch(2, 2, 2, seed=20)
        xi_true = _sample_parametric(spline, 20, seed=21)
        spline.locate(_evaluate_at(spline, xi_true))
        assert _cache_of(spline) is not None

        if mutate == "reverse":
            spline.reverse(0, in_place=True)
        elif mutate == "permute_directions":
            spline.permute_directions([1, 0], in_place=True)
        else:
            spline.transform(AffineTransform.translation([100.0, 100.0]), in_place=True)

        assert _cache_of(spline) is None
        # And the mutated geometry inverts correctly, which a stale cache would break.
        moved = _sample_parametric(spline, 20, seed=22)
        cell_ids, ref_coords = spline.locate(_evaluate_at(spline, moved))
        assert np.all(cell_ids >= 0)
        np.testing.assert_allclose(
            ref_coords, moved, atol=_XI_REL_ATOL * _scale_of(spline), rtol=0.0
        )


class TestValidation:
    """Rejected inputs, each with the exception the contract promises."""

    def test_embedded_geometry_is_not_implemented(self) -> None:
        """A curve in the plane (``rank > dim``) needs a closest-point solve."""
        sub = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2)
        curve = Bspline(BsplineSpace([sub]), np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]]))
        with pytest.raises(NotImplementedError, match="rank 2 > dim 1"):
            curve.locate(np.array([[1.0, 1.0]]))

    def test_rank_below_dim_is_rejected(self) -> None:
        """A scalar field over a surface cannot be inverted at all."""
        sub = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)
        space = BsplineSpace([sub, sub])
        field = Bspline(space, np.arange(4.0).reshape(2, 2, 1))
        with pytest.raises(ValueError, match="rank 1 < dim 2"):
            field.locate(np.array([[1.0]]))

    def test_periodic_space_is_rejected(self) -> None:
        """Periodic directions have no bounded knot-span grid to report cells from."""
        knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        periodic = BsplineSpace1D(knots, 2, periodic=True)
        space = BsplineSpace([periodic, periodic])
        n_0, n_1 = space.num_basis
        rng = np.random.default_rng(23)
        spline = Bspline(space, rng.uniform(size=(n_0, n_1, 2)))
        with pytest.raises(ValueError, match="periodic"):
            spline.locate(np.array([[0.5, 0.5]]))

    def test_non_positive_weight_is_rejected(self) -> None:
        """The convex-hull property behind the candidate search needs positive weights."""
        annulus = _quarter_annulus()
        cp = np.array(annulus.control_points, dtype=np.float64)
        cp[1, 0, 2] = -cp[1, 0, 2]
        broken = Bspline(annulus.space, cp, is_rational=True)
        with pytest.raises(ValueError, match="weight must be strictly positive"):
            broken.locate(np.array([[1.2, 0.3]]))

    @pytest.mark.parametrize(
        "bad_points",
        [
            np.zeros((3, 3)),
            np.zeros((3, 1)),
            np.zeros((2, 2, 2)),
            np.array([np.nan, 0.5]),
            np.array([[0.5, np.inf]]),
        ],
    )
    def test_bad_points_are_rejected(self, bad_points: npt.NDArray[np.float64]) -> None:
        """Wrong trailing dimension, wrong rank, or a non-finite coordinate."""
        annulus = _quarter_annulus()
        with pytest.raises(ValueError):
            annulus.locate(bad_points)

    @pytest.mark.parametrize("max_iter", [0, -1])
    def test_max_iter_must_be_positive(self, max_iter: int) -> None:
        """A budget of zero steps is a caller mistake, not a silent empty result."""
        annulus = _quarter_annulus()
        with pytest.raises(ValueError, match="max_iter"):
            annulus.locate(np.array([[1.2, 0.3]]), max_iter=max_iter)

    @pytest.mark.parametrize("tol", [0.0, -1e-6, np.nan, np.inf])
    def test_tol_must_be_positive_and_finite(self, tol: float) -> None:
        """An explicit tolerance is an absolute distance, so it must be usable as one."""
        annulus = _quarter_annulus()
        with pytest.raises(ValueError, match="tol"):
            annulus.locate(np.array([[1.2, 0.3]]), tol=tol)
