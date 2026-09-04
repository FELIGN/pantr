"""Tests for physical-to-parametric point inversion (``Bspline.locate``)."""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._bspline_locate import (
    _cell_midpoints,
    _cell_parametric_bounds,
    _cell_physical_bounds,
    _default_tolerance,
    _geometric_scale,
    _locate_context,
    _LocateContext,
    _LocateThresholds,
    _nearest_corner_starts,
    _newton_refine,
    _parametric_reach,
    _parametric_scale,
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

_PARAMETRIC_ULPS_ALLOWED: float = 4.0
"""
How many parametric ulps' worth of residual a returned coordinate may carry.

One ulp of a parametric coordinate is ``eps * |xi_k|``, which the map turns into a physical
displacement of ``eps * |xi_k| * ||J e_k||``; :func:`_parametric_scale` is exactly that
product with the control net's span standing in for the local stretch, so
``eps * _parametric_scale(spline)`` is the physical length the parametrization quantizes at
and no returned coordinate can do better than a fraction of it. The bound is therefore
``_PARAMETRIC_ULPS_ALLOWED * eps * _parametric_scale(spline)``, which is the acceptance
threshold divided by ``64 / _PARAMETRIC_ULPS_ALLOWED``.

**Why four.** Two ulps for where the last Newton step lands relative to the true root, and a
factor of two for the fixture's local stretch against the net-span average the substitution
uses. Measured over seven seeds on :func:`_parametrically_offset_patch`, the worst residual
is ``0.33`` of a single ulp's image for targets that are exact images and ``0.85`` for
targets a half ulp off one, so the bound carries ``4.7`` times its measured requirement
here; on sibling fixtures (degree 3 at reach ``1e4``, degree 2 on 8 elements at reach
``1e10``) the half-ulp worst rises to ``1.33`` and the margin falls to ``3.0``. The
single-threshold policy it replaces exceeds it on 9 of 60 queries in both families with a
worst case of 52 ulps, so the two policies are separated by any factor from 3 to 16 -- but
not by every factor below that, since the good policy itself passes 1.33 on a sibling.

**What it is not an oracle for.** The bound calls :func:`_parametric_scale`, the same
function that builds the acceptance threshold, so it cannot detect an error in *that*
length -- inflating it threefold leaves every test using this bound green. What it detects
is the stopping rule, because the residual it grades is produced by the Newton iteration's
own arithmetic and not by any threshold. :func:`_parametric_scale`'s numeric value is
pinned independently by :class:`TestParametricStretch`, against hand-derived literals.
"""

_PARAMETRIC_OFFSETS: list[float] = [0.0, 1.0e2, 1.0e3, 1.0e4, 1.0e6]
"""
Parametric offsets of a unit-width knot span, spanning the resolution frontier.

On the fixture below the attainable residual floor is ``0.008`` of the physical-only
threshold at ``0``, half of it at ``1e2``, and four times it at ``1e3`` -- the first row
that is impossible to satisfy. At ``1e6`` it is 4096 times the threshold.
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
    return Bspline(space, np.ascontiguousarray(_warped_lattice(space) + offset))


def _warped_lattice(space: BsplineSpace) -> npt.NDArray[np.float64]:
    """Return the unit-box control lattice of ``space`` under the 15 % sinusoidal warp.

    Shared by :func:`_warped_patch` and :func:`_parametrically_offset_patch` so that the
    two differ in their *knot vectors* alone and any difference between them is
    attributable to the parametrization.
    """
    axes = [np.linspace(0.0, 1.0, n) for n in space.num_basis]
    cp = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return np.asarray(cp + 0.15 * np.sin(2.0 * np.pi * cp.sum(axis=-1))[..., None])


def _parametrically_offset_patch(
    degree: int = 3, n_elem: int = 4, offset: float = 1.0e6
) -> Bspline:
    """Return the warped unit patch with direction 1 parametrized on ``[offset, offset + 1]``.

    The geometry is :func:`_warped_patch`'s, so the physical half of the threshold is the
    one of a unit-sized patch at the origin; only direction 1's knot vector moves. That
    separates the two halves as far as they go: direction 0 has reach zero and direction 1
    reach ``offset``, so the acceptance threshold is six decades above the stopping one
    while the geometry it grades is unchanged.

    A bivariate map rather than a univariate one, and the offset in one direction only,
    because that is the configuration in which the two thresholds are least alike.
    """
    spaces = []
    for direction_offset in (0.0, offset):
        knots = np.concatenate(
            [
                np.full(degree, direction_offset),
                np.linspace(direction_offset, direction_offset + 1.0, n_elem + 1),
                np.full(degree, direction_offset + 1.0),
            ]
        )
        spaces.append(BsplineSpace1D(knots, degree))
    space = BsplineSpace(spaces)
    return Bspline(space, np.ascontiguousarray(_warped_lattice(space)))


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


def _offset_identity_patch(
    offset: float, degree: int = 2, n_elem: int = 4, dim: int = 1, size: float = 1.0
) -> Bspline:
    """Return ``F(xi) = size * (xi - offset)`` on the parametric box ``[offset, offset + 1]``.

    The map is *exactly* affine: the control points are the Greville abscissae less the
    offset, scaled by ``size``, and linear precision reproduces an affine map exactly. So
    every singular value of the Jacobian is ``size`` at every point, which is the best
    conditioned a non-degenerate map can be. Whatever such a patch loses is therefore lost
    to the parametrization's own resolution and not to conditioning, which is what makes it
    the fixture for moving the parametric offset while holding the geometry fixed.
    """
    knots = np.concatenate(
        [
            np.full(degree, offset),
            np.linspace(offset, offset + 1.0, n_elem + 1),
            np.full(degree, offset + 1.0),
        ]
    )
    spaces = [BsplineSpace1D(knots, degree) for _ in range(dim)]
    axes = [size * (_greville_abscissae(sub) - offset) for sub in spaces]
    mesh = np.meshgrid(*axes, indexing="ij")
    cp = np.stack(mesh, axis=-1) if dim > 1 else mesh[0][:, np.newaxis]
    return Bspline(BsplineSpace(spaces), np.ascontiguousarray(cp))


def _half_ulp_targets(spline: Bspline, xi_true: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return targets midway between the images of two consecutive representable parameters.

    The hardest request the parametric grid can carry: the target sits halfway between
    ``F(xi)`` and ``F(nextafter(xi))``, so no representable parameter maps nearer than half
    the local image spacing, however good the solver is. Asking for the image of ``xi``
    itself would instead make the exact preimage a solution and hide the resolution limit
    entirely.
    """
    domain = np.asarray(spline.space.domain, dtype=np.float64)
    upper = np.nextafter(xi_true, domain[:, 1])
    return 0.5 * (_evaluate_at(spline, xi_true) + _evaluate_at(spline, upper))


def _attainable_floor_1d(
    spline: Bspline, xi_true: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
) -> float:
    """Return the smallest residual any representable parameter can reach, over the queries.

    Scans each parameter and its two neighbours. On a strictly increasing one-dimensional
    map that is exhaustive rather than a sample: the residual is unimodal in the parameter,
    so its minimum over *all* representable parameters is attained at one of the two
    bracketing the target, and widening the scan cannot lower it.
    """
    lo, hi = np.asarray(spline.space.domain, dtype=np.float64)[0]
    worst = 0.0
    for xi, target in zip(xi_true[:, 0].tolist(), targets[:, 0].tolist(), strict=True):
        neighbours = np.array([[np.nextafter(xi, lo)], [xi], [np.nextafter(xi, hi)]])
        residuals = np.abs(_evaluate_at(spline, neighbours)[:, 0] - target)
        worst = max(worst, float(residuals.min()))
    return worst


def _steep_span_patch(offset: float, n_elem: int, flat_fraction: float = 0.5) -> Bspline:
    """Return a degree-1 monotone map of ``[offset, offset + 1]`` onto ``[0, 1]``, with a kink.

    A fraction ``flat_fraction`` of the rise is spread uniformly over the whole span and the
    rest is concentrated in one interior span, so ``sup|F'|`` is ``flat_fraction + (1 -
    flat_fraction) * n_elem`` while the image stays exactly ``[0, 1]``. The amplification the
    parametric term has to absorb is therefore set by ``n_elem`` alone, and is ``n_elem / 2 +
    1 / 2`` at the default fraction.

    This is the family the ``sigma_max`` substitution in :func:`_geometric_scale` is argued
    against: a map whose *local* stretch exceeds the average over its own direction is
    exactly what a net span underestimates.
    """
    knots = np.concatenate(
        [[offset], np.linspace(offset, offset + 1.0, n_elem + 1), [offset + 1.0]]
    )
    space = BsplineSpace1D(knots, 1)
    uniform = np.linspace(0.0, 1.0, n_elem + 1)
    step = np.zeros(n_elem + 1)
    step[n_elem // 2 + 1 :] = 1.0
    cp = flat_fraction * uniform + (1.0 - flat_fraction) * step
    return Bspline(BsplineSpace([space]), np.ascontiguousarray(cp[:, np.newaxis]))


def _steep_span_queries(
    spline: Bspline, n_elem: int, seed: int = 3
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return half-ulp queries placed inside the steep span, where the floor is highest."""
    lo = np.asarray(spline.space.domain, dtype=np.float64)[0, 0]
    rng = np.random.default_rng(seed)
    xi_true = (lo + (n_elem // 2 + rng.uniform(0.05, 0.95, size=40)) / n_elem)[:, np.newaxis]
    return xi_true, _half_ulp_targets(spline, xi_true)


def _anisotropic_patch(offset1: float, extent1: float, degree: int = 2, n_elem: int = 4) -> Bspline:
    """Return an affine 2-D map whose two directions differ in offset and in image size.

    Direction 0 spans ``[0, 1]`` onto a unit image; direction 1 spans ``[offset1, offset1 +
    1]`` onto an image of length ``extent1``. The two parametric resolutions are therefore
    unrelated, which is what a term reducing over directions too early gets wrong.
    """
    axes, spaces = [], []
    for offset, extent in ((0.0, 1.0), (offset1, extent1)):
        knots = np.concatenate(
            [
                np.full(degree, offset),
                np.linspace(offset, offset + 1.0, n_elem + 1),
                np.full(degree, offset + 1.0),
            ]
        )
        sub = BsplineSpace1D(knots, degree)
        spaces.append(sub)
        axes.append((_greville_abscissae(sub) - offset) * extent)
    cp = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return Bspline(BsplineSpace(spaces), np.ascontiguousarray(cp))


def _parametric_ulp_image(spline: Bspline) -> float:
    """Return the resolution floor a returned coordinate is graded against, per ulp.

    The acceptance threshold divided by the tolerance tier, so it moves with the geometry
    and with the knot vector exactly as the threshold does and carries no constant of its
    own; :data:`_PARAMETRIC_ULPS_ALLOWED` counts it.

    Taken over *both* lengths and not over the parametric one alone. On a fixture whose
    parametrization is offset the parametric length wins and the two readings agree, which
    is every current caller; on one where the physical length wins, the parametric reading
    would put the bound below the residual floor the coordinates themselves impose and the
    test would fail for a reason having nothing to do with the parametrization.
    """
    scale = _geometric_scale(_physical_only_scale(spline), _parametric_scale(spline))
    return float(np.finfo(np.float64).eps) * scale


def _first_candidates(
    context: _LocateContext, targets: npt.NDArray[np.float64], thresholds: _LocateThresholds
) -> npt.NDArray[np.int64]:
    """Return the first candidate cell of each target, as ``_locate_impl``'s first round does.

    Lets a test drive :func:`_newton_refine` from the same starting guesses the solver uses
    without reaching into it, so a comparison between two threshold policies varies the
    policy alone.
    """
    return np.asarray(
        [np.sort(context.bvh.query_aabb(AABB(x, x).pad(thresholds.accept)))[0] for x in targets],
        dtype=np.int64,
    )


def _physical_only_scale(spline: Bspline) -> float:
    """Return the scale the rule gave before it had a parametric term.

    Written out here rather than imported, so the "did this loosen anything" comparisons
    have an oracle that cannot move with the code they grade.
    """
    lo, hi = _cell_physical_bounds(spline)
    box_lo, box_hi = lo.min(axis=0), hi.max(axis=0)
    diagonal = float(np.linalg.norm(box_hi - box_lo))
    magnitude = float(max(np.abs(box_lo).max(), np.abs(box_hi).max()))
    scale = max(diagonal, magnitude)
    return scale if scale > 0.0 else 1.0


def _scale_of(spline: Bspline) -> float:
    """Return the geometric scale the default acceptance threshold is expressed in."""
    return _geometric_scale(_physical_only_scale(spline), _parametric_scale(spline))


def _cache_of(spline: Bspline) -> _LocateContext | None:
    """Return the point-inversion memo of ``spline``.

    Read through a function rather than inline: an inline ``assert spline._derived.locate
    is None`` narrows the attribute's type for the rest of the test, and mypy then calls
    the later assertions unreachable. A call expression is not narrowed.

    The memo lives in the derived block rather than in a slot of its own, so that an
    in-place mutator cannot discard it without also discarding the Bézier
    decomposition; see ``pantr.bspline._bspline._Derived``.
    """
    return spline._derived.locate


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
    """Evaluate ``spline`` at ``(n, dim)`` parametric points, always returning ``(n, rank)``.

    Waits for the import-time JIT warmup first, for the reason
    :class:`TestNumbaWarmup` states: :meth:`Bspline.evaluate` is not itself behind that
    barrier, so a test that evaluates before it locates is a bare ``parallel=True`` call
    racing the background compilation thread, and the process *aborts* rather than raising.
    Measured on this file: a selection whose first act is a burst of evaluations aborted 5
    of 5 runs without this, and 0 of 5 with it. The barrier is once per process, so every
    call after the first costs nothing.
    """
    wait_for_jit_warmup()
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
                    _evaluate_at(
                        spline,
                        _newton_refine(
                            spline, target, start, _LocateThresholds(stop=tol, accept=tol), k
                        )[1],
                    )
                    - target
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

        thresholds = _LocateThresholds(stop=tol, accept=tol)
        from_midpoint = _newton_refine(
            spline, target, _cell_midpoints(spline.space, candidates), thresholds, 40
        )
        from_corner = _newton_refine(
            spline, target, _nearest_corner_starts(spline, candidates, target), thresholds, 40
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


class TestParametricResolution:
    """The default threshold also follows what the *parametrization* can resolve.

    One ulp of a parametric coordinate is ``eps * |xi|``, which a map of stretch ``sigma``
    turns into a physical residual of ``eps * |xi| * sigma``. A threshold read off the
    physical side alone is therefore unreachable for a knot span sitting far from the origin
    relative to its own width, and every query on such a patch is reported not found although
    it has a preimage.
    """

    def test_the_frontier_parametric_offset_is_recovered(self) -> None:
        """The exact case where the attainable floor first passes the old threshold.

        At an offset of ``1e3`` on a unit-width knot span the smallest residual any
        representable parameter can reach is four times the threshold the old rule demanded,
        so all 25 queries were reported not found. Kept as its own test, separate from the
        sweep, because it is the frontier: a threshold that grew with the offset but too
        slowly would still pass the ``0`` and ``1e2`` rows and fail here.
        """
        spline = _offset_identity_patch(1.0e3)
        xi_true = _sample_parametric(spline, 25, seed=11, margin=0.05)
        targets = _half_ulp_targets(spline, xi_true)
        floor = _attainable_floor_1d(spline, xi_true, targets)

        cell_ids, ref_coords = spline.locate(targets)

        tol = get_default(spline.dtype) * _scale_of(spline)
        assert floor <= tol, f"the request is impossible: floor {floor:.3e} over tol {tol:.3e}"
        assert np.all(cell_ids >= 0), f"lost {int(np.sum(cell_ids < 0))} of 25 queries"
        residuals = np.abs(_evaluate_at(spline, ref_coords)[:, 0] - targets[:, 0])
        assert residuals.max() <= tol

    @pytest.mark.parametrize("offset", _PARAMETRIC_OFFSETS)
    def test_a_shifted_knot_span_loses_no_query_that_has_a_preimage(self, offset: float) -> None:
        """Every offset of the ticket's table, including the ``1e6`` end.

        The geometry is identical in all five rows -- the same unit segment, the same
        Jacobian of exactly one -- and only the knot vector moves, so a row that loses points
        can only have lost them to the parametrization.
        """
        spline = _offset_identity_patch(offset)
        xi_true = _sample_parametric(spline, 25, seed=11, margin=0.05)
        targets = _half_ulp_targets(spline, xi_true)

        cell_ids, ref_coords = spline.locate(targets)

        assert np.all(cell_ids >= 0), f"lost {int(np.sum(cell_ids < 0))} of 25 at offset {offset}"
        residuals = np.abs(_evaluate_at(spline, ref_coords)[:, 0] - targets[:, 0])
        assert residuals.max() <= get_default(spline.dtype) * _scale_of(spline)

    @pytest.mark.parametrize("offset", _PARAMETRIC_OFFSETS)
    def test_the_threshold_stays_above_the_attainable_floor(self, offset: float) -> None:
        """The derivation, checked directly rather than through the solver.

        What went wrong was not a solver that gave up but a bar set below anything a float64
        parameter can reach. This measures the bar and the floor separately and asserts the
        order between them, so it still fails if a future change makes the queries pass for
        some unrelated reason.
        """
        spline = _offset_identity_patch(offset)
        xi_true = _sample_parametric(spline, 25, seed=11, margin=0.05)
        targets = _half_ulp_targets(spline, xi_true)

        floor = _attainable_floor_1d(spline, xi_true, targets)
        tol = get_default(spline.dtype) * _scale_of(spline)

        assert floor <= tol, f"floor {floor:.3e} over tol {tol:.3e} at offset {offset}"

    @pytest.mark.parametrize(
        ("name", "spline", "expected"),
        [
            ("quarter annulus", _quarter_annulus(), 2.8284271247461903),
            ("warped 2-D degree 1", _warped_patch(), 1.781637023790572),
            ("warped 3-D degree 3", _warped_patch(dim=3, degree=3, n_elem=3), 2.226234269696677),
            ("perturbed degree 3 on [0, 4]", _perturbed_patch(3, 4, 2, seed=7), 5.910822971201255),
            ("folded", _folded_patch(), 2.6704143675085485),
            ("stretched", _stretched_patch(), 2.27072620893615),
            (
                "identity 2-D degree 2",
                _identity_map([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2, 2),
                1.4142135623730951,
            ),
        ],
    )
    def test_a_knot_vector_starting_at_the_origin_keeps_todays_scale(
        self, name: str, spline: Bspline, expected: float
    ) -> None:
        """Ordinary geometry is held to exactly the bar it was held to before.

        The threshold is an accuracy contract rather than a safety margin, so loosening it
        degrades every returned coordinate one-for-one. These are the values measured before
        the parametric term existed, and equality is asserted *bitwise* rather than to a
        relative tolerance because it is exact and not merely close: a knot vector spanning
        ``[0, L]`` gives ``reach == L / L``, which IEEE division makes exactly ``1.0``, and
        ``1.0 * diagonal`` is exactly ``diagonal``, so the three-way maximum returns the
        two-way one bit for bit. A term that perturbed it at all would fail here.
        """
        assert _scale_of(spline) == expected, name

    @pytest.mark.parametrize("half_width", [1.0, 1.0e-3, 1.0e6])
    def test_a_knot_vector_centred_on_the_origin_keeps_todays_scale(
        self, half_width: float
    ) -> None:
        """A symmetric parametrization resolves its extent better than a one-sided one.

        On ``[-a, a]`` the largest coordinate is half the extent, so the parametric term is
        half the physical one and can never be what decides the threshold. Pinning this
        keeps a future form of the term from reading the *extent* where it must read the
        distance from the origin.
        """
        degree, n_elem = 2, 4
        knots = np.concatenate(
            [
                np.full(degree, -half_width),
                np.linspace(-half_width, half_width, n_elem + 1),
                np.full(degree, half_width),
            ]
        )
        space = BsplineSpace1D(knots, degree)
        cp = np.ascontiguousarray(_greville_abscissae(space)[:, np.newaxis])
        spline = Bspline(BsplineSpace([space]), cp)

        # ``a / (2a)``, which IEEE makes exactly one half since ``2a`` is exact.
        reach = _parametric_reach(np.asarray(spline.space.domain, dtype=np.float64))
        assert reach.tolist() == [0.5]
        # Diagonal and magnitude of the image ``[-a, a]``: ``2a`` and ``a``.
        assert _scale_of(spline) == pytest.approx(2.0 * half_width, rel=1e-15)

    def test_a_moderately_offset_knot_vector_does_loosen_and_is_meant_to(self) -> None:
        """The unchanged-at-the-origin pins must not be read as unchanged everywhere.

        A knot vector spanning ``[1, 2]`` has reach two, so its threshold is exactly twice
        the one the same geometry gets on ``[0, 1]`` -- because its resolution floor is twice
        as high, one ulp of a coordinate near 2 being twice one near 1. This is the ordinary
        intermediate case, and it is pinned so that "offset zero is unaffected" is not
        mistaken for "no ordinary domain is affected".
        """
        at_origin = _offset_identity_patch(0.0)
        offset_by_one = _offset_identity_patch(1.0)

        assert _scale_of(offset_by_one) == pytest.approx(2.0 * _scale_of(at_origin), rel=1e-15)

    @pytest.mark.parametrize("size", [1.0e-3, 1.0, 1.0e3])
    @pytest.mark.parametrize("offset", _PARAMETRIC_OFFSETS)
    @pytest.mark.parametrize("dim", [1, 2])
    def test_the_sweep_over_offsets_and_physical_scales_loses_nothing(
        self, dim: int, offset: float, size: float
    ) -> None:
        """Parametric offset and physical size vary independently; neither may lose a query.

        Two decades of geometry either side of unit size crossed with four parametric
        offsets. Sampling the parameter first and mapping it forward is what makes "has a
        preimage" a fact about each query, and the map is injective here, so a ``-1`` would
        be unambiguously wrong.
        """
        spline = _offset_identity_patch(offset, degree=2, n_elem=3, dim=dim, size=size)
        xi_true = _sample_parametric(spline, 40, seed=int(dim * 977 + np.log10(size + 1.0)))
        targets = _half_ulp_targets(spline, xi_true)

        cell_ids, ref_coords = spline.locate(targets)

        lost = np.flatnonzero(cell_ids < 0)
        assert lost.size == 0, f"lost {lost.size} of 40 at {dim=}, {offset=}, {size=}"
        residuals = np.linalg.norm(_evaluate_at(spline, ref_coords) - targets, axis=1)
        tol = get_default(spline.dtype) * _scale_of(spline)
        assert residuals.max() <= tol, f"worst residual {residuals.max():.3e} over tol {tol:.3e}"
        _assert_cell_contains(spline, cell_ids, ref_coords)

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0e-3, 1.0e3, 1.0e6])
    def test_the_scale_is_invariant_under_scaling_the_parametrization(self, lam: float) -> None:
        """Rescaling the knot vector alone changes no physical quantity, so it changes no bar.

        Under ``xi -> lam * xi`` the coordinate magnitude grows by ``lam`` and the Jacobian
        shrinks by it, so the physical residual a one-ulp parametric step produces is
        untouched. A term reading the parametric magnitude *without* dividing by the extent
        would scale the threshold here and break covariance.
        """
        base = _offset_identity_patch(1.0e3, degree=2, n_elem=4)
        knots = lam * np.asarray(base.space.spaces[0].knots, dtype=np.float64)
        space = BsplineSpace1D(knots, base.space.spaces[0].degree)
        rescaled = Bspline(BsplineSpace([space]), np.asarray(base.control_points).copy())

        assert _scale_of(rescaled) == pytest.approx(_scale_of(base), rel=1e-12)

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0e-3, 1.0e3, 1.0e6])
    def test_the_scale_is_proportional_to_the_geometry_at_a_parametric_offset(
        self, lam: float
    ) -> None:
        """Scaling the geometry scales the threshold, offset parametrization included.

        The parametric term is a physical length like the other two, so it must carry the
        geometry's own factor rather than sit at a fixed size beside it.
        """
        unit = _offset_identity_patch(1.0e3, degree=2, n_elem=4, size=1.0)
        scaled = _offset_identity_patch(1.0e3, degree=2, n_elem=4, size=lam)

        assert _scale_of(scaled) == pytest.approx(lam * _scale_of(unit), rel=1e-12)

    def test_a_parametric_direction_with_no_extent_keeps_the_scale_finite(self) -> None:
        """A direction with no interval leaves nothing to divide by, and must not divide.

        ``BsplineSpace1D`` accepts ``[a, a, a, a]``: a legal space with zero intervals, whose
        domain has zero extent. The parametric term reads an extent as a denominator, so this
        is the one input that could turn the threshold into ``inf`` or ``nan`` -- and an
        infinite threshold reports every query "found", which is worse than the "not found"
        this ticket exists to remove.

        Exercised on :func:`_parametric_reach` directly rather than through
        :meth:`Bspline.locate`, which raises on such a space for an unrelated reason: with no
        knot-span cell there is no per-cell box to reduce over. Guarding the division here
        keeps the term total on its own inputs whatever that separate question is settled to.
        """
        domain = np.array([[5.0, 5.0], [0.0, 1.0]])

        reach = _parametric_reach(domain)

        assert np.all(np.isfinite(reach))
        # The zero-extent direction contributes nothing; the ordinary one gives reach 1.
        assert reach.tolist() == [0.0, 1.0]


class TestParametricStretch:
    """The one substitution in the derivation, on the family built to break it.

    ``_net_span_per_direction`` stands in for ``sup||J e_k||``. That is exact for an affine
    map and an underestimate for one whose local stretch exceeds the average over its own
    direction, by the amplification ``A = L_k * sup||J e_k|| / net_span_k``. The floor is a
    half ulp against a tier of ``64 * eps``, so the gap is absorbed to ``A`` of order 100.
    """

    @pytest.mark.parametrize(("n_elem", "amplification"), [(4, 2.5), (64, 32.5), (256, 128.5)])
    def test_a_locally_steep_map_is_recovered_while_the_tier_absorbs_its_stretch(
        self, n_elem: int, amplification: float
    ) -> None:
        """Below the absorbed limit, a kinked map at a far parametric offset loses nothing.

        Every one of these is lost outright without the parametric term, so the row is a
        statement about the cure and not only about the substitution inside it.
        """
        spline = _steep_span_patch(1.0e6, n_elem)
        xi_true, targets = _steep_span_queries(spline, n_elem)
        slope = float(
            np.abs(
                np.asarray(
                    spline.evaluate_derivatives(np.ascontiguousarray(xi_true[:, 0]), [1]),
                    dtype=np.float64,
                )
            ).max()
        )
        assert slope == pytest.approx(amplification, rel=1e-9), "fixture must carry the stretch"

        cell_ids, ref_coords = spline.locate(targets)

        tol = get_default(spline.dtype) * _scale_of(spline)
        assert _attainable_floor_1d(spline, xi_true, targets) <= tol
        assert np.all(cell_ids >= 0), f"lost {int(np.sum(cell_ids < 0))} of 40 at A = {slope}"
        assert np.abs(_evaluate_at(spline, ref_coords)[:, 0] - targets[:, 0]).max() <= tol

    def test_past_the_absorbed_stretch_the_request_is_impossible_either_way(self) -> None:
        """Where the substitution stops covering the floor, and why that is not a regression.

        At ``A == 256`` the attainable floor passes the threshold and the queries go back to
        being unreachable. What makes that a residue rather than a regression is the second
        assertion: the physical-only threshold cannot reach the floor either, and by decades
        more, so nothing here was working before and stopped.
        """
        spline = _steep_span_patch(1.0e6, 512)
        xi_true, targets = _steep_span_queries(spline, 512)
        floor = _attainable_floor_1d(spline, xi_true, targets)

        tol_new = get_default(spline.dtype) * _scale_of(spline)
        tol_old = get_default(spline.dtype) * _physical_only_scale(spline)

        assert floor > tol_new, "the frontier is meant to sit between A = 128 and A = 256"
        assert floor > tol_old * 1.0e3, "and the physical-only threshold is nowhere near it"
        assert tol_new > tol_old, "the term only ever loosened this case"

    @pytest.mark.parametrize("offset", _PARAMETRIC_OFFSETS)
    @pytest.mark.parametrize("n_elem", [4, 256])
    def test_the_parametric_term_never_tightens_the_threshold(
        self, n_elem: int, offset: float
    ) -> None:
        """The term may loosen a threshold or leave it alone; it may never raise the bar.

        With the bitwise-unchanged pin for a knot vector spanning ``[0, L]``, this is the
        whole characterization of what the term is allowed to do, and it is what makes "no
        geometry is held to a stricter bar than before" checkable rather than argued.
        """
        spline = _steep_span_patch(offset, n_elem)

        assert _scale_of(spline) >= _physical_only_scale(spline)

    def test_an_anisotropic_domain_pairs_each_offset_with_its_own_direction(self) -> None:
        """The reach of one direction may not be multiplied by another direction's stretch.

        Both patches put direction 0 on ``[0, 1]`` onto a unit image and direction 1 at a
        parametric offset of ``1e6``; they differ only in direction 1's image extent. The
        parametric length must follow *that* extent, because direction 1's resolution floor
        is what its own reach and its own stretch produce.

        Reducing the reach over directions before pairing it with a physical length gives
        both patches ``1e6`` instead, handing direction 1's offset to direction 0's unit
        stretch: a factor of ``1e6`` too loose on the thin patch, on geometry the
        physical-only threshold had inverted without losing anything. The assertion is on
        the two lengths rather than on a ratio to the attainable floor, because at this
        extent that floor is below one ulp of unity and moves by a factor of a few with
        which representable point is sampled.
        """
        thick = _anisotropic_patch(1.0e6, 1.0)
        thin = _anisotropic_patch(1.0e6, 1.0e-6)

        assert _parametric_scale(thick) == pytest.approx(1.0e6, rel=1.0e-5)
        assert _parametric_scale(thin) == pytest.approx(1.0, rel=1.0e-5)

    @pytest.mark.parametrize("window", [(984375.0, 1.0e6), (999984.375, 1.0e6)])
    def test_restricting_a_patch_leaves_its_absolute_threshold_alone(
        self, window: tuple[float, float]
    ) -> None:
        """A sub-patch is held to the same absolute bar as the parent it was cut from.

        This is the property the reach form has and a diagonal alone cannot. Cutting a
        window out of an affine map divides the geometry's diagonal by the same factor it
        multiplies the reach by -- the sub-patch is smaller, but it sits proportionally
        further from the parametric origin -- so the product, and with it the threshold, is
        untouched. That is right: the physical resolution attainable at a point of the map
        does not depend on how much of the map around it was kept.

        Both windows are dyadic, so the restriction lands on exact breakpoints and the only
        thing under test is the tolerance rule.
        """
        degree, n_cells = 2, 64
        knots = np.concatenate(
            [
                np.zeros(degree),
                np.linspace(0.0, 1.0e6, n_cells + 1),
                np.full(degree, 1.0e6),
            ]
        )
        space = BsplineSpace1D(knots, degree)
        cp = np.ascontiguousarray((_greville_abscissae(space) / 1.0e6)[:, np.newaxis])
        parent = Bspline(BsplineSpace([space]), cp)

        child = parent.restrict(window)

        assert _scale_of(child) == pytest.approx(_scale_of(parent), rel=1e-12)

    def test_a_parametrization_that_cannot_carry_a_threshold_is_refused(self) -> None:
        """Past a reach of ``1 / (64 * eps)`` the default threshold exceeds the geometry.

        A "found" verdict admitting more than a whole diameter of error says only that the
        query was somewhere near the model, so it is refused rather than served. Clamping
        instead would demand an accuracy the parametrization cannot deliver, which is the
        very defect the term exists to remove, relocated to the far end of the range.
        """
        spline = _offset_identity_patch(1.0e14, degree=1, n_elem=1)

        with pytest.raises(ValueError, match="parametric origin"):
            spline.locate(np.array([0.5]))

        # An explicit tol is never refused: naming a distance is taking responsibility.
        assert spline.locate(np.array([0.5]), tol=1.0e-3)[0].tolist() == [0]

    def test_an_ordinary_far_offset_is_not_refused(self) -> None:
        """The refusal must sit far above anything a real model reaches.

        Seven decades of parametric offset below the line, all still served. The boundary is
        ``1 / (64 * eps)``, about ``7e13``, which is within a factor of eight of what the
        knot layer refuses outright.
        """
        for offset in (1.0e2, 1.0e6, 1.0e10, 1.0e12):
            spline = _offset_identity_patch(offset, degree=1, n_elem=1)
            assert spline.locate(np.array([0.5]))[0].tolist() == [0], f"refused at {offset}"


class TestStoppingVersusAcceptance:
    """One threshold cannot be both the accuracy contract and the attainability bound.

    The physical half of the scale says what accuracy a returned coordinate is entitled to;
    the parametric half says how far below the residual no *representable* parameter can go.
    Stopping at the second delivers only the second, and on a knot vector far from the
    parametric origin the two are six decades apart. These pin that the iteration stops at
    the tighter one while acceptance stays at the looser one, on the geometry where the gap
    is widest.
    """

    def test_the_answer_is_as_accurate_as_the_parametrization_allows(self) -> None:
        """Targets that are exact images are inverted to the parametric grid, not to the bar.

        Sampling the parameter first and mapping it forward makes the preimage a fact about
        each query, and it is *representable*, so the attainable residual is zero and
        anything the solver leaves on the table is the stopping rule's doing. The bound is
        the parametrization's own quantization; see :data:`_PARAMETRIC_ULPS_ALLOWED`.
        """
        spline = _parametrically_offset_patch()
        xi_true = _sample_parametric(spline, 60, seed=321)
        targets = _evaluate_at(spline, xi_true)
        bound = _PARAMETRIC_ULPS_ALLOWED * _parametric_ulp_image(spline)

        cell_ids, ref_coords = spline.locate(targets)

        assert np.all(cell_ids >= 0), f"lost {int(np.sum(cell_ids < 0))} of 60 queries"
        residuals = np.linalg.norm(_evaluate_at(spline, ref_coords) - targets, axis=1)
        over = int(np.sum(residuals > bound))
        assert over == 0, (
            f"{over} of 60 coordinates stopped short: worst {residuals.max():.3e} over a "
            f"parametric-resolution bound of {bound:.3e}"
        )
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_a_target_no_representable_parameter_reaches_is_still_found(self) -> None:
        """The looser bar keeps its coverage while the tighter one sets the accuracy.

        Half-ulp targets have no exact preimage at all, so the tight threshold alone loses
        every one of them (measured: 60 of 60). They must still come back found, and with
        the residual the parametric grid allows rather than the one the acceptance bar does.
        """
        spline = _parametrically_offset_patch()
        xi_true = _sample_parametric(spline, 60, seed=321)
        targets = _half_ulp_targets(spline, xi_true)
        bound = _PARAMETRIC_ULPS_ALLOWED * _parametric_ulp_image(spline)

        cell_ids, ref_coords = spline.locate(targets)

        assert np.all(cell_ids >= 0), f"lost {int(np.sum(cell_ids < 0))} of 60 queries"
        residuals = np.linalg.norm(_evaluate_at(spline, ref_coords) - targets, axis=1)
        over = int(np.sum(residuals > bound))
        assert over == 0, (
            f"{over} of 60 coordinates stopped short: worst {residuals.max():.3e} over a "
            f"parametric-resolution bound of {bound:.3e}"
        )
        _assert_cell_contains(spline, cell_ids, ref_coords)

    def test_a_start_between_the_two_bars_is_refined_instead_of_returned(self) -> None:
        """The stopping rule is the tight one: an iterate inside the loose bar keeps going.

        A starting guess one ulp off the true preimage already satisfies the acceptance
        threshold, so under a single threshold it is returned untouched -- that is exactly
        how the accuracy was lost. Under the split it is refined until it meets the stopping
        one, which the same call with both thresholds set to the loose value shows it does
        not do of its own accord.

        **One ulp, not a number picked to work.** The image of one ulp of a parametric
        coordinate is what :func:`_parametric_scale` measures, and the acceptance threshold
        is the tolerance tier times that length, so a one-ulp start sits below the loose bar
        by about the tier itself and far above the tight one. Measured over 20 seeds of 20
        queries: at worst ``0.025`` of the acceptance threshold (a margin of 40, against the
        derived 64, the shortfall being this fixture's local stretch over the net-span
        average) and at least 2178 times the stopping one. Both premises therefore hold by
        derivation rather than by a constant chosen to make them hold, which is what a
        premise assertion has to do if its failure is to mean anything.
        """
        spline = _parametrically_offset_patch()
        context = _locate_context(spline)
        thresholds = _default_tolerance(spline, context)
        xi_true = _sample_parametric(spline, 20, seed=17)
        targets = _evaluate_at(spline, xi_true)
        start = xi_true.copy()
        start[:, 1] = np.nextafter(start[:, 1], np.inf)
        start_residual = np.linalg.norm(_evaluate_at(spline, start) - targets, axis=1)
        assert np.all(start_residual > thresholds.stop), "the premise: above the tight bar"
        assert np.all(start_residual <= thresholds.accept), "the premise: inside the loose one"

        loose_only = _newton_refine(
            spline, targets, start, _LocateThresholds(thresholds.accept, thresholds.accept), 30
        )
        split = _newton_refine(spline, targets, start, thresholds, 30)

        assert np.all(loose_only[0]) and np.all(split[0]), "both policies report found"
        assert np.array_equal(loose_only[1], start), "one threshold returns the guess as given"
        refined = np.linalg.norm(_evaluate_at(spline, split[1]) - targets, axis=1)
        assert np.all(refined < start_residual), f"not refined: {refined} vs {start_residual}"
        assert np.all(refined <= thresholds.stop), f"did not reach the tight bar: {refined.max()}"

    def test_a_residual_that_cannot_reach_the_tight_bar_is_still_accepted(self) -> None:
        """The acceptance rule is the loose one, and it is what keeps the coverage.

        Half-ulp targets have no representable preimage, so *no* iterate reaches the
        stopping threshold; the verdict has to come from the acceptance one, and it does,
        for a residual the parametric grid genuinely cannot beat. Driven from a single
        candidate cell and a single start, which is less than the solver gives itself, so
        the rows this leaves unconverged are the ones a second start recovers rather than a
        loss.
        """
        spline = _parametrically_offset_patch()
        context = _locate_context(spline)
        thresholds = _default_tolerance(spline, context)
        tight_only = _LocateThresholds(stop=thresholds.stop, accept=thresholds.stop)
        xi_true = _sample_parametric(spline, 20, seed=17)
        targets = _half_ulp_targets(spline, xi_true)
        starts = _cell_midpoints(spline.space, _first_candidates(context, targets, thresholds))

        by_the_tight_bar, _ = _newton_refine(spline, targets, starts, tight_only, 30)
        converged, xi = _newton_refine(spline, targets, starts, thresholds, 30)

        residuals = np.linalg.norm(_evaluate_at(spline, xi) - targets, axis=1)
        assert not np.any(by_the_tight_bar), "the premise: no iterate reaches the tight bar"
        assert np.any(converged), "acceptance at the loose bar recovered nothing"
        assert np.all(residuals[converged] > thresholds.stop)
        assert np.all(residuals[converged] <= thresholds.accept)
        # What one candidate cell from one start leaves, locate's own fallbacks recover.
        assert np.all(spline.locate(targets)[0] >= 0)

    def test_the_split_never_returns_a_worse_coordinate_than_one_threshold_did(self) -> None:
        """Coverage may not shrink, checked on the geometry with the widest gap.

        The argument is in :class:`_LocateThresholds`: accepted steps never raise the
        residual, so lowering the *stopping* test only extends trajectories. This runs both
        policies from the same starts and pins the two consequences -- residuals no larger,
        and therefore verdicts no fewer -- on queries on and off the mapping's image.
        """
        spline = _parametrically_offset_patch()
        context = _locate_context(spline)
        thresholds = _default_tolerance(spline, context)
        loose = _LocateThresholds(stop=thresholds.accept, accept=thresholds.accept)
        # Without this the test goes quiet rather than red: were the two bars ever equal,
        # `loose` would be `thresholds` and every comparison below a run against itself.
        assert thresholds.stop < thresholds.accept, "the premise: the two policies differ"
        xi_true = _sample_parametric(spline, 60, seed=321)
        for targets in (_evaluate_at(spline, xi_true), _half_ulp_targets(spline, xi_true)):
            starts = _cell_midpoints(spline.space, _first_candidates(context, targets, thresholds))

            single, xi_single = _newton_refine(spline, targets, starts, loose, 30)
            split, xi_split = _newton_refine(spline, targets, starts, thresholds, 30)

            res_single = np.linalg.norm(_evaluate_at(spline, xi_single) - targets, axis=1)
            res_split = np.linalg.norm(_evaluate_at(spline, xi_split) - targets, axis=1)
            assert np.any(single), "the premise: the single threshold finds something here"
            assert np.all(res_split <= res_single), "the split returned a worse coordinate"
            assert np.all(split | ~single), "the split lost a query the single threshold found"

    @pytest.mark.parametrize(
        "factory",
        [
            _warped_patch,
            _quarter_annulus,
            lambda: _offset_identity_patch(0.0),
            lambda: _perturbed_patch(2, 3, 2, seed=4),
        ],
        ids=["warped", "annulus", "from-origin-affine", "perturbed"],
    )
    def test_a_knot_vector_from_the_origin_gets_one_threshold_bit_for_bit(
        self, factory: Callable[[], Bspline]
    ) -> None:
        """The split changes nothing for ordinary geometry, in accuracy or in cost.

        The reach is exactly ``1.0`` for a knot vector spanning ``[0, L]`` and no net span
        exceeds the diagonal, so the parametric length cannot win the maximum and the two
        thresholds are the same float. Equal thresholds make the iteration the one that ran
        before, step for step, which is why no cost measurement is needed here.
        """
        spline = factory()
        thresholds = _default_tolerance(spline, _locate_context(spline))

        assert thresholds.stop == thresholds.accept
        assert thresholds.stop == get_default(spline.dtype) * _physical_only_scale(spline)

    def test_a_parametric_offset_is_what_separates_them(self) -> None:
        """The complement of the test above: the gap is real where the reach is.

        Without this the bitwise-equality test could be passing because the split does
        nothing anywhere.
        """
        spline = _parametrically_offset_patch()
        thresholds = _default_tolerance(spline, _locate_context(spline))

        assert thresholds.stop < thresholds.accept
        assert thresholds.stop == get_default(spline.dtype) * _physical_only_scale(spline)
        assert thresholds.accept / thresholds.stop > 1.0e5, "the fixture's gap is decades wide"

    @pytest.mark.parametrize("offset", [0.0, 1.0e2, 1.0e4, 1.0e5])
    def test_a_float32_spline_never_opens_the_split(self, offset: float) -> None:
        """A float32 caller gets one threshold at every offset its knot vector can carry.

        Not an accident of these four rows. The acceptance threshold's parametric term is
        format-independent by construction -- :func:`_parametric_scale` multiplies by
        ``eps64 / eps(dtype)``, so ``get_default(dtype)`` times it is ``64 * eps64 * reach *
        span`` in every format -- while the stopping threshold is ``64 * eps(dtype) *
        physical``. Opening the split therefore needs ``reach * span > (eps(dtype) / eps64)
        * physical``, and no net span exceeds the bounding-box diagonal, so it needs
        ``reach > eps32 / eps64 == 5.4e8``. A float32 knot vector of width ``L`` at offset
        ``c`` needs its two endpoints to be distinct float32s at all: for ``c`` in ``[2**e,
        2**(e+1))`` the smallest representable gap is ``2**(e-24)`` while ``c < 2**(e+1)``,
        so ``reach < 2**25 == 3.4e7``. The provable margin is therefore exactly ``2**4 ==
        16``, a format identity rather than a measured quantity. The knot layer is stricter
        still -- it snaps at ``8 * eps``, which drops the ceiling by another ``8`` -- and the
        next test confirms the cut-off is its and not this module's.

        So the split is a float64 phenomenon, and deliberately: for a float32 caller the
        format the *data* carries is already coarser than anything the parametrization
        quantizes at, which is the same reason the inversion promotes to float64 to iterate.
        """
        spline = _offset_identity_patch(offset, degree=2, n_elem=4)
        knots = np.asarray(spline.space.spaces[0].knots, dtype=np.float32)
        spline32 = Bspline(
            BsplineSpace([BsplineSpace1D(knots, 2)]),
            np.ascontiguousarray(np.asarray(spline.control_points, dtype=np.float32)),
        )
        thresholds = _default_tolerance(spline32, _locate_context(spline32))

        assert thresholds.stop == thresholds.accept
        assert thresholds.stop == get_default(np.float32) * _physical_only_scale(spline32)

    def test_the_float32_offset_that_would_open_the_split_is_refused_first(self) -> None:
        """The knot layer stops a float32 knot vector before this module has to.

        The complement of the test above, and what makes its argument a statement about
        every float32 spline rather than about the four offsets sampled: one decade further
        out the knot vector cannot be built at all, because in float32 at ``1e6`` the mesh
        collapses onto a single knot.
        """
        offset = 1.0e6
        knots = np.concatenate(
            [np.full(2, offset), np.linspace(offset, offset + 1.0, 5), np.full(2, offset + 1.0)]
        ).astype(np.float32)

        with pytest.raises(ValueError, match="collapsed every knot"):
            BsplineSpace1D(knots, 2)

    def test_an_explicit_tolerance_governs_stopping_as_well(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A named threshold is not silently refined past, so a proximity query stays cheap.

        The caller who passes ``tol`` has said what distance they mean; answering to eight
        decades better would charge them for accuracy they did not ask for, and the split is
        exactly what makes that newly possible.

        Read off the thresholds :func:`_locate_impl` hands the solver rather than off the
        residual it happens to produce. A residual assertion cannot pin this: on this
        geometry the internal stopping bar is eleven decades below a coarse ``tol``, so a
        partial leak of it into ``stop`` still leaves every residual far above anything a
        residual test could distinguish, and passes. The thresholds themselves are exact.
        """
        spline = _warped_patch()
        targets = _evaluate_at(spline, _sample_parametric(spline, 30, seed=5))
        seen: list[_LocateThresholds] = []
        module = sys.modules["pantr.bspline._bspline_locate"]

        def _record(
            sp: Bspline,
            pts: npt.NDArray[np.float64],
            start: npt.NDArray[np.float64],
            thresholds: _LocateThresholds,
            max_iter: int,
        ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
            # The module-level import still names the real function; only the module
            # attribute the solver looks up is redirected.
            seen.append(thresholds)
            return _newton_refine(sp, pts, start, thresholds, max_iter)

        monkeypatch.setattr(module, "_newton_refine", _record)

        cell_ids, ref_coords = spline.locate(targets, tol=1.0e-2)

        assert seen, "the premise: the solver was reached at all"
        assert all(t == _LocateThresholds(stop=1.0e-2, accept=1.0e-2) for t in seen), (
            f"an explicit tol must govern both thresholds; got {set(seen)}"
        )
        assert np.all(cell_ids >= 0)
        residuals = np.linalg.norm(_evaluate_at(spline, ref_coords) - targets, axis=1)
        assert residuals.max() <= 1.0e-2, "the threshold asked for must still hold"


class TestNearCriticalJacobian:
    """A long Newton step is capped, not walked back one halving at a time."""

    def test_a_monotone_map_with_a_near_critical_jacobian_is_found(self) -> None:
        """The step-length cap earns its place: without it this query is reported not found.

        A strictly monotone degree-5 map, so the preimage is unique and ``-1`` is unambiguously
        wrong. Its derivative ``F'(xi) = eta + a * xi**2 * (xi - 1/2)**2`` with ``eta = 1e-4``
        and ``a = 100`` is near-critical at ``xi == 0`` and ``xi == 1/2``, which are exactly the
        two points the solver starts from -- the cell corner and the cell midpoint -- so both
        starts begin where ``||J^-1 r||`` is enormous. ``sigma_min`` is ``1e-4`` there, fifteen
        orders clear of the rank guard, so nothing about this is degenerate.

        Uncapped, the line search spends its budget shortening a step 300 domain extents long:
        11 halvings were needed where 8 were allowed, and ``locate`` returned ``-1`` at library
        defaults. Capping the first trial at one domain extent drops the requirement to 2.
        """
        knots = np.array([0.0] * 6 + [1.0] * 6)
        control_points = np.array(
            [
                [0.0],
                [2.0e-5],
                [4.0e-5],
                [0.8333933333333333],
                [-1.6665866666666664],
                [3.333433333333334],
            ]
        )
        spline = Bspline(BsplineSpace([BsplineSpace1D(knots, 5)]), control_points)
        query = np.array([0.03308666666666668])  # F(0.2), to the last bit

        cell_ids, ref_coords = spline.locate(query)

        assert cell_ids.tolist() == [0], "a strictly monotone map must not lose its own image"
        np.testing.assert_allclose(ref_coords, [[0.2]], atol=1e-12, rtol=0.0)
        derivative = np.asarray(spline.evaluate_derivatives(np.linspace(0.0, 1.0, 1001), [1]))
        assert derivative.min() > 0.0, "fixture must be strictly monotone"


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
