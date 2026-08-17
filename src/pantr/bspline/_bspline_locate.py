"""Physical-to-parametric point inversion for :class:`~pantr.bspline.Bspline`.

Layer 2 implementation behind :meth:`pantr.bspline.Bspline.locate`: given physical
points, recover the parametric coordinates whose image they are, together with the
knot-span cell that contains them.

The inversion has two stages:

1. **Candidate cells.** The image of a knot-span cell lies inside the axis-aligned box
   of the control points supporting that cell -- the convex-hull property of the
   B-spline basis, which for a NURBS holds for the *projected* control points provided
   every weight is positive. Those per-cell boxes are indexed by a
   :class:`pantr.grid.BVH`, so the cells that can contain a query point are found by a
   tree descent instead of by trying Newton on every cell.
2. **Newton.** Per candidate cell, Newton's method on ``F(xi) - x = 0`` starting from
   the cell's parametric midpoint, with the Jacobian assembled column by column from
   :meth:`~pantr.bspline.Bspline.evaluate_derivatives` (already rational-aware through
   the generalized quotient rule). Iterates are clamped to the parametric domain box,
   so an iterate may migrate out of its candidate cell: the candidate box is a superset
   test, never a constraint on the solution.

Newton runs on the whole batch of points at once: one candidate slot per round, and
within a round one evaluator call per Jacobian column for all still-active points
together. The number of Python-level calls into the evaluators is therefore
``O(rounds * max_iter * dim)``, independent of the number of query points.

What is guaranteed is ``F(ref_coords[i]) == points[i]`` within the tolerance, and *not*
that ``ref_coords[i]`` is any particular preimage. A mapping whose Jacobian determinant
changes sign folds, and a folded mapping sends several parametric points to the same
physical point; every one of them is a correct answer, and which one comes back depends on
the candidate order and on the Newton path. Only for an injective mapping does inverting
``evaluate`` return the coordinates it was called with.

Precision contract: the inversion always runs in ``float64``, promoting a ``float32``
spline to an exact ``float64`` copy first -- casting float32 coefficients to float64 is
exact, so the promoted spline is the same mapping. The reason is margin, not
impossibility: the residual of a ``float32`` evaluation cannot be resolved below
``1 - 3 * eps32 * scale`` (measured over supports from 6 to 125 control points), while
:func:`pantr.tolerance.get_default` is ``64 * eps`` in every format. Iterating in
float32 would therefore leave between one and two decades of headroom, and would
additionally quantize the parametric iterate at ``eps32``. Promotion removes both for
the price of one exact cast: the arithmetic floor becomes ``1 - 3 * eps64 * scale``,
which for a float32 caller is eight orders below the threshold it asks for, so what
limits the answer is the precision the caller's data carries and not the solver. The
returned coordinates invert the promoted mapping to float64 accuracy; evaluating them
on the original ``float32`` spline reproduces the query point to float32 accuracy only.

v1 inverts square maps only (``rank == dim``): planar and volumetric geometry maps. An
embedded curve or surface (``rank > dim``) needs a Gauss-Newton closest-point solve,
which is a documented follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..geometry import AABB
from ..grid import BVH, tensor_product_grid
from ..tolerance import get_default

if TYPE_CHECKING:
    from numpy import typing as npt

    from ._bspline import Bspline
    from ._bspline_space_nd import BsplineSpace

_JACOBIAN_RANK_REL_TOL: float = float(np.finfo(np.float64).eps)
"""
Relative threshold at which a Newton Jacobian counts as rank-deficient.

A candidate is abandoned when ``sigma_min <= dim * _JACOBIAN_RANK_REL_TOL * sigma_max``,
which is the rule and the magnitude :func:`numpy.linalg.matrix_rank` uses by default
(``max(M, N) * eps * sigma_max``): below it the null direction is indistinguishable from
rounding noise and the solve returns an arbitrary step. The test is a ratio, so it is
invariant under scaling the geometry or the parametrization. It is deliberately
permissive -- it only rejects a Jacobian whose condition number reaches ``1 / (dim *
eps)`` -- because a merely ill-conditioned Jacobian still gives a usable step, and the
clamp to the parametric domain box already bounds an overlong one.
"""


class _LocateContext(NamedTuple):
    """Per-:class:`~pantr.bspline.Bspline` state reused across :meth:`locate` calls.

    Cached on the B-spline instance and invalidated by the in-place mutators, exactly
    like the Bézier decomposition cache: none of these three depends on the query
    points, and the BVH costs ``O(num_cells)`` to build.

    Attributes:
        spline (Bspline): The spline to invert, in ``float64``. The original instance
            when it is already ``float64``, otherwise an exact promoted copy.
        bvh (BVH): Hierarchy over the per-cell physical control-point boxes, with cell
            ids flat in C-order over ``space.num_intervals``.
        scale (float): The geometric scale the default tolerance is expressed in; see
            :func:`_geometric_scale`.
    """

    spline: Bspline
    bvh: BVH
    scale: float


def _physical_control_points(spline: Bspline) -> npt.NDArray[np.float64]:
    """Return the projected control points of ``spline`` as ``float64``.

    Args:
        spline (Bspline): The spline whose control points are wanted. A rational spline
            must have strictly positive weights, which the caller has checked.

    Returns:
        npt.NDArray[np.float64]: Array of shape ``(*num_basis, rank)``: the control
        points a non-rational spline stores directly, or ``P_i / w_i`` for a NURBS.
    """
    cp = np.asarray(spline.control_points, dtype=np.float64)
    if not spline.is_rational:
        return cp
    return np.asarray(cp[..., :-1] / cp[..., -1:], dtype=np.float64)


def _promote_to_float64(spline: Bspline) -> Bspline:
    """Return an exact ``float64`` copy of ``spline``, or ``spline`` if already so.

    Casting ``float32`` knots and control points to ``float64`` is exact, so the copy
    represents the same mapping. Knots that a ``float32`` construction snapped together
    stay bitwise equal under the cast, so the promoted knot vector has the same
    multiplicity structure.

    Args:
        spline (Bspline): The spline to promote.

    Returns:
        Bspline: ``spline`` when its dtype is already ``float64``, else a new
        :class:`~pantr.bspline.Bspline` over ``float64`` knots and control points.
    """
    if np.dtype(spline.dtype) == np.float64:
        return spline

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415 -- breaks an import cycle
    from ._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415 -- breaks an import cycle
    from ._bspline_space_nd import BsplineSpace  # noqa: PLC0415 -- breaks an import cycle

    spaces_64 = [
        BsplineSpace1D(np.asarray(sub.knots, dtype=np.float64), sub.degree, periodic=sub.periodic)
        for sub in spline.space.spaces
    ]
    return BsplineCls(
        BsplineSpace(spaces_64),
        np.asarray(spline.control_points, dtype=np.float64),
        is_rational=spline.is_rational,
    )


def _cell_physical_bounds(
    spline: Bspline,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the physical AABB of every knot-span cell's supporting control points.

    Reduces one parametric axis at a time rather than gathering the full
    ``prod(degree + 1)`` support of each cell: the support is a contiguous window per
    axis, so a running min/max over ``degree + 1`` shifted takes gives the same result
    in ``O(num_cells * sum(degree + 1))`` work and without ever materializing the
    per-cell support table.

    Args:
        spline (Bspline): The spline whose cell boxes are wanted, in ``float64``, with
            no periodic direction and (if rational) positive weights.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ``(lo, hi)``, both of
        shape ``(num_total_intervals, rank)``, indexed by flat cell id in C-order over
        ``space.num_intervals``.
    """
    lo = _physical_control_points(spline)
    hi = lo
    for axis, sub in enumerate(spline.space.spaces):
        first = sub.first_basis_per_interval()
        acc_lo = np.take(lo, first, axis=axis)
        acc_hi = np.take(hi, first, axis=axis)
        for offset in range(1, sub.degree + 1):
            np.minimum(acc_lo, np.take(lo, first + offset, axis=axis), out=acc_lo)
            np.maximum(acc_hi, np.take(hi, first + offset, axis=axis), out=acc_hi)
        lo, hi = acc_lo, acc_hi
    n_cells = spline.space.num_total_intervals
    rank = spline.rank
    return lo.reshape(n_cells, rank), hi.reshape(n_cells, rank)


def _geometric_scale(lo: npt.NDArray[np.float64], hi: npt.NDArray[np.float64]) -> float:
    """Return the length the default convergence tolerance is a multiple of.

    Two lengths matter and the larger one has to win:

    - the **diagonal** of the geometry's bounding box, which is what makes a residual
      threshold a *relative* accuracy statement about the mapping; and
    - the largest coordinate **magnitude** present, because the computed residual
      ``F(xi) - x`` cannot be resolved below ``~C * eps * max|coordinate|`` (the de Boor
      sum accumulates ``C`` roundings, of the order of ``prod(degree + 1)``), whatever
      the box's extent is.

    Using the diagonal alone silently makes the tolerance unreachable for a geometry far
    from the origin -- a patch of diameter 1 sitting at ``x = 1e6`` has a residual floor
    around ``1e-10`` while ``eps * 1`` would be demanded -- so the scale is the maximum
    of the two.

    **What the default tier buys, measured.** :func:`pantr.tolerance.get_default` is
    ``64 * eps``, and an earlier reading of this called that a six percent margin over an
    arithmetic floor: the worst residual observed came out at ``60 * eps * scale`` against
    a threshold of 64. That reading is wrong, and the way it is wrong matters, because a
    thin margin invites loosening the tier and loosening the tier here is the one thing
    that costs accuracy outright.

    The iteration *stops* at the threshold, so the residuals of the points it returns are
    censored by it and ``60`` was the ceiling, not a floor. Re-measured at six thresholds
    from ``2`` to ``4096`` epsilons, on 3960 mildly warped and 5520 strongly warped query
    points (degrees 1 to 5 in 1 to 3 directions, at scales 1 and ``1e6``, with the targets
    pushed a few ulp off the machine-exact image so the exact preimage is not a solution):
    the worst achieved residual tracks whatever threshold is set, to within one part in a
    hundred, at every one of them. Nothing was lost even at ``4 * eps * scale``.

    Two consequences. The threshold is an **accuracy contract**, not a safety margin, so
    moving to a looser tier would degrade the returned coordinates one-for-one and buy no
    robustness. And ``C ~ prod(degree + 1)`` is not the operative floor: the residual shows
    no dependence on ``prod(degree + 1)`` across 3 to 125. The default tier stays, now for
    a stated reason rather than by a margin that was an artifact of the measurement.

    A geometry with no length at all -- every control point identical, *at the origin* --
    falls back to ``1.0``: there is nothing to read a scale off, and the tolerance must
    stay positive. That fallback is deliberately not a floor of one. A floor would make the
    tolerance stop shrinking below unit size, so a model in metres and the same model in
    kilometres would be held to different relative accuracies, which is exactly the
    non-covariance the rest of this derivation exists to remove. A degenerate geometry
    sitting *away* from the origin has a magnitude and uses it.

    Args:
        lo (npt.NDArray[np.float64]): Per-cell box lower corners, shape
            ``(n_cells, rank)``.
        hi (npt.NDArray[np.float64]): Per-cell box upper corners, same shape.

    Returns:
        float: The geometric scale, strictly positive.
    """
    box_lo = lo.min(axis=0)
    box_hi = hi.max(axis=0)
    diagonal = float(np.linalg.norm(box_hi - box_lo))
    magnitude = float(max(np.abs(box_lo).max(), np.abs(box_hi).max()))
    scale = max(diagonal, magnitude)
    return scale if scale > 0.0 else 1.0


def _build_context(spline: Bspline) -> _LocateContext:
    """Build the cached inversion state of ``spline``.

    Args:
        spline (Bspline): The spline to invert; already validated by
            :func:`_locate_impl`.

    Returns:
        _LocateContext: The promoted spline, the BVH over its per-cell physical boxes,
        and the geometric scale.
    """
    spline_64 = _promote_to_float64(spline)
    lo, hi = _cell_physical_bounds(spline_64)
    return _LocateContext(
        spline=spline_64, bvh=BVH.from_cell_bounds(lo, hi), scale=_geometric_scale(lo, hi)
    )


def _cell_midpoints(
    space: BsplineSpace, cell_ids: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Return the parametric midpoint of each of the given knot-span cells.

    Args:
        space (BsplineSpace): The space the cells belong to, with no periodic direction.
        cell_ids (npt.NDArray[np.int64]): Flat cell ids in C-order over
            ``space.num_intervals``, shape ``(n,)``.

    Returns:
        npt.NDArray[np.float64]: Shape ``(n, space.dim)`` parametric midpoints.
    """
    multi = np.unravel_index(cell_ids, space.num_intervals)
    out = np.empty((cell_ids.shape[0], space.dim), dtype=np.float64)
    for axis, sub in enumerate(space.spaces):
        breakpoints = np.asarray(
            sub.get_unique_knots_and_multiplicity(in_domain=True)[0], dtype=np.float64
        )
        index = multi[axis]
        out[:, axis] = 0.5 * (breakpoints[index] + breakpoints[index + 1])
    return out


def _eval_map(spline: Bspline, xi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate the mapping at parametric points, in the batched shape Newton wants.

    Args:
        spline (Bspline): A ``float64`` spline with ``rank == dim``.
        xi (npt.NDArray[np.float64]): Parametric points, shape ``(n, dim)``, inside the
            domain.

    Returns:
        npt.NDArray[np.float64]: Physical points, shape ``(n, rank)``. The reshape also
        undoes the ``squeeze()`` the 1-D evaluation path applies to scalar output.
    """
    pts = np.ascontiguousarray(xi[:, 0] if spline.dim == 1 else xi)
    values = spline.evaluate(pts)
    return np.asarray(values, dtype=np.float64).reshape(xi.shape[0], spline.rank)


def _eval_jacobian(spline: Bspline, xi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate the Jacobian of the mapping at parametric points.

    Column ``d`` is the first partial derivative in parametric direction ``d``, taken
    from :meth:`~pantr.bspline.Bspline.evaluate_derivatives`, which returns derivatives
    of the projected mapping for a NURBS.

    Args:
        spline (Bspline): A ``float64`` spline with ``rank == dim``.
        xi (npt.NDArray[np.float64]): Parametric points, shape ``(n, dim)``, inside the
            domain.

    Returns:
        npt.NDArray[np.float64]: Shape ``(n, rank, dim)`` Jacobians.
    """
    n_pts, dim = xi.shape
    rank = spline.rank
    pts = np.ascontiguousarray(xi[:, 0] if dim == 1 else xi)
    out = np.empty((n_pts, rank, dim), dtype=np.float64)
    for axis in range(dim):
        orders = [0] * dim
        orders[axis] = 1
        column = spline.evaluate_derivatives(pts, orders)
        out[:, :, axis] = np.asarray(column, dtype=np.float64).reshape(n_pts, rank)
    return out


def _newton_refine(
    spline: Bspline,
    targets: npt.NDArray[np.float64],
    xi_start: npt.NDArray[np.float64],
    tol_phys: float,
    max_iter: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Run a batched Newton inversion, one starting guess per target point.

    Every point is iterated independently but evaluated collectively: each round drops
    the points that have converged and the points whose Jacobian has become
    rank-deficient, then takes one Newton step for the rest. Iterates are clamped to the
    parametric domain box, which both keeps them inside the evaluators' legal input
    range and bounds an overlong step from an ill-conditioned Jacobian.

    Args:
        spline (Bspline): A ``float64`` spline with ``rank == dim``, non-periodic.
        targets (npt.NDArray[np.float64]): Physical query points, shape ``(n, rank)``.
        xi_start (npt.NDArray[np.float64]): Starting parametric guesses, shape
            ``(n, dim)``.
        tol_phys (float): Convergence threshold on ``||F(xi) - x||_2``, a distance in
            physical units.
        max_iter (int): Maximum number of Newton steps. The residual is tested once more
            after the last step, so a point converging on step ``max_iter`` is reported.

    Returns:
        tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]: ``(converged, xi)``.
        ``converged`` has shape ``(n,)``; ``xi`` has shape ``(n, dim)`` and is only
        meaningful where ``converged`` is True.
    """
    dim = xi_start.shape[1]
    domain = np.asarray(spline.space.domain, dtype=np.float64)
    lo, hi = domain[:, 0], domain[:, 1]

    xi = np.clip(xi_start, lo, hi)
    converged = np.zeros(xi.shape[0], dtype=np.bool_)
    active = np.arange(xi.shape[0], dtype=np.int64)

    for step in range(max_iter + 1):
        xi_active = xi[active]
        residual = _eval_map(spline, xi_active) - targets[active]
        done = np.linalg.norm(residual, axis=1) <= tol_phys
        converged[active[done]] = True
        active, xi_active, residual = active[~done], xi_active[~done], residual[~done]
        if step == max_iter or active.size == 0:
            break

        jacobian = _eval_jacobian(spline, xi_active)
        singular_values = np.asarray(np.linalg.svd(jacobian, compute_uv=False), dtype=np.float64)
        healthy = singular_values[:, -1] > (dim * _JACOBIAN_RANK_REL_TOL * singular_values[:, 0])
        active = active[healthy]
        if active.size == 0:
            break
        delta = np.asarray(
            np.linalg.solve(jacobian[healthy], residual[healthy][..., np.newaxis]),
            dtype=np.float64,
        )[..., 0]
        xi[active] = np.clip(xi_active[healthy] - delta, lo, hi)

    return converged, xi


def _validate_points(points: npt.ArrayLike, rank: int) -> npt.NDArray[np.float64]:
    """Normalize the query points to a finite ``(n, rank)`` ``float64`` array.

    A 1-D input is read as a single point of ``rank`` coordinates, except for ``rank ==
    1`` where it is read as ``n`` scalar coordinates -- the convention
    :meth:`~pantr.bspline.Bspline.evaluate` already uses for a 1-D spline.

    Args:
        points (npt.ArrayLike): The query points.
        rank (int): The spline's physical rank.

    Returns:
        npt.NDArray[np.float64]: Shape ``(n, rank)`` array of query points.

    Raises:
        ValueError: If the shape is not ``(n, rank)`` (after the 1-D readings above) or
            any coordinate is not finite.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1) if rank == 1 else pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != rank:  # noqa: PLR2004 -- (n, rank) is 2-D
        raise ValueError(
            f"locate: points must have shape (n, {rank}) for a rank-{rank} B-spline; "
            f"got shape {np.asarray(points).shape}."
        )
    if not bool(np.all(np.isfinite(pts))):
        raise ValueError("locate: points must be finite; got a NaN or infinite coordinate.")
    return np.ascontiguousarray(pts)


def _locate_impl(
    spline: Bspline,
    points: npt.ArrayLike,
    tol: float | None,
    max_iter: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Invert the mapping of ``spline`` at the given physical points.

    Args:
        spline (Bspline): The spline to invert. Must be square (``rank == dim``) and
            non-periodic, and, if rational, have strictly positive weights.
        points (npt.ArrayLike): Physical query points; see :func:`_validate_points`.
        tol (float | None): Convergence threshold on ``||F(xi) - x||_2`` as an absolute
            distance in physical units, or ``None`` for
            ``pantr.tolerance.get_default(spline.dtype) * scale``, with ``scale`` from
            :func:`_geometric_scale`.
        max_iter (int): Maximum number of Newton steps per candidate cell.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ``(cell_ids,
        ref_coords)``, both read-only. ``cell_ids`` has shape ``(n,)`` and holds flat
        knot-span cell ids in C-order over ``space.num_intervals``, or ``-1`` where the
        point was not found; ``ref_coords`` has shape ``(n, dim)`` and holds parametric
        coordinates, or ``nan`` where the point was not found.

    Raises:
        NotImplementedError: If ``rank > dim`` (an embedded curve or surface).
        ValueError: If ``rank < dim``, any direction is periodic, a rational spline has
            a non-positive weight, ``points`` has a bad shape or a non-finite entry,
            ``max_iter < 1``, or ``tol`` is given and is not positive and finite.
    """
    dim, rank = spline.dim, spline.rank
    if rank > dim:
        raise NotImplementedError(
            f"locate: embedded geometries (rank {rank} > dim {dim}) need a Gauss-Newton "
            "closest-point solve, which is not implemented; only square maps "
            "(rank == dim) can be inverted."
        )
    if rank < dim:
        raise ValueError(
            f"locate: a B-spline mapping cannot be inverted with fewer physical "
            f"coordinates than parametric directions; got rank {rank} < dim {dim}."
        )
    for axis, sub in enumerate(spline.space.spaces):
        if sub.periodic:
            raise ValueError(f"locate: periodic B-spline spaces are not supported (axis {axis}).")
    if spline.is_rational and not bool(
        np.all(np.asarray(spline.control_points[..., -1], dtype=np.float64) > 0.0)
    ):
        raise ValueError(
            "locate: every NURBS weight must be strictly positive; the convex-hull "
            "property the candidate search relies on does not hold otherwise."
        )
    if max_iter < 1:
        raise ValueError(f"locate: max_iter must be >= 1; got {max_iter}.")
    if tol is not None and not (float(tol) > 0.0 and np.isfinite(tol)):
        raise ValueError(f"locate: tol must be positive and finite; got {tol}.")

    pts = _validate_points(points, rank)
    context = _locate_context(spline)
    tol_phys = float(tol) if tol is not None else get_default(spline.dtype) * context.scale

    n_pts = pts.shape[0]
    cell_ids = np.full(n_pts, -1, dtype=np.int64)
    ref_coords = np.full((n_pts, dim), np.nan, dtype=np.float64)

    # Candidates per point, in ascending cell id: the order the rounds below try them in.
    candidates = [np.sort(context.bvh.query_aabb(AABB(x, x).pad(tol_phys))) for x in pts]
    pending = [i for i, cand in enumerate(candidates) if cand.size > 0]
    found: list[int] = []
    slot = 0
    while pending:
        batch = [i for i in pending if candidates[i].size > slot]
        if not batch:
            break
        index = np.asarray(batch, dtype=np.int64)
        starts = _cell_midpoints(
            context.spline.space, np.asarray([candidates[i][slot] for i in batch], dtype=np.int64)
        )
        converged, xi = _newton_refine(context.spline, pts[index], starts, tol_phys, max_iter)
        hits = index[converged]
        ref_coords[hits] = xi[converged]
        found.extend(int(i) for i in hits)
        resolved = set(found)
        pending = [i for i in pending if i not in resolved]
        slot += 1

    if found:
        rows = np.asarray(sorted(found), dtype=np.int64)
        grid = tensor_product_grid(context.spline.space)
        cell_ids[rows] = grid.locate_many(ref_coords[rows])

    cell_ids.flags.writeable = False
    ref_coords.flags.writeable = False
    return cell_ids, ref_coords


def _locate_context(spline: Bspline) -> _LocateContext:
    """Return the cached inversion state of ``spline``, building it on first use.

    Args:
        spline (Bspline): The spline to invert; already validated by
            :func:`_locate_impl`.

    Returns:
        _LocateContext: The cached context. Invalidated by the in-place mutators of
        :class:`~pantr.bspline.Bspline`, which reset the cache attribute to ``None``.
    """
    # Layer 2 owns this cache slot; Layer 1 only declares and invalidates it.
    if spline._locate_cache is None:
        spline._locate_cache = _build_context(spline)
    return spline._locate_cache


__all__ = ["_locate_impl"]
