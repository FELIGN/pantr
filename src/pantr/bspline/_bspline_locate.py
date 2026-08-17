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
2. **Newton.** Per candidate cell, a damped Newton's method on ``F(xi) - x = 0`` starting
   from the cell's parametric midpoint and, if that does not converge, from the cell
   corner whose image is nearest the query. The Jacobian is assembled column by column
   from :meth:`~pantr.bspline.Bspline.evaluate_derivatives` (already rational-aware
   through the generalized quotient rule). Iterates are clamped to the parametric domain
   box, so an iterate may migrate out of its candidate cell: the candidate box is a
   superset test, never a constraint on the solution.

   The damping is a backtracking line search, and it is what makes the iteration usable
   away from the basin of attraction: an undamped Newton step is only guaranteed to
   reduce the residual near a root, and out there it can be arbitrarily longer than the
   residual it is trying to remove. Accepting such a step and then clamping the result to
   the domain box does not merely waste an iteration, it can be *periodic* -- the clamp
   maps two off-basin iterates onto each other -- and then no iteration budget and no
   tolerance recovers the point. Requiring each accepted step to reduce the residual
   removes both failure modes at once, and makes the residual of the accepted iterates a
   monotonically non-increasing sequence.

   What damping cannot do is change which root the iteration finds. A monotone method
   descends into whichever basin it starts in, and on a mapping that folds, the residual
   has minima on the domain boundary that are not roots; a start that descends into one of
   those is stuck there legitimately. That is what the second start is for, and why it is a
   *different point of the same cell* rather than a longer iteration.

Newton runs on the whole batch of points at once: one candidate slot per round, and
within a round one evaluator call per Jacobian column for all still-active points
together, plus one per backtracking level for the points still searching at that level.
The number of Python-level calls into the evaluators is therefore
``O(rounds * starts * max_iter * (dim + halvings))``, independent of the number of query
points, with ``starts`` at most two, ``halvings`` at most ``1 + _MAX_STEP_HALVINGS`` and
equal to one wherever the full Newton step is accepted, and the second start costing a
further ``2 ** dim`` evaluations for the points that need it.

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

from .._numba_compat import wait_for_jit_warmup
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
line search below rejects an overlong one.
"""

_ARMIJO_DECREASE: float = 1.0e-4
"""
Fraction of its own size the residual norm must lose per unit step length, to be accepted.

A damped step ``lam * delta`` passes when ``||F(xi - lam * delta) - x|| <= (1 -
_ARMIJO_DECREASE * lam) * ||F(xi) - x||``: the Armijo sufficient-decrease rule, applied to
the merit function ``||F - x||`` whose descent direction the exact Newton step is (with
``delta = J^-1 r`` one has ``d/dlam ||F(xi - lam * delta) - x||`` at ``lam = 0`` equal to
``-||r||``, so a decrease of ``lam * ||r||`` is what the linear model promises and this
constant is the fraction of it that must actually materialize).

**Why any value in ``(0, 1)`` terminates.** Expanding to second order,
``F(xi - lam * delta) - x = (1 - lam) * r + O(lam^2 * M * ||delta||^2)`` with ``M`` a bound
on the second derivatives, so the test holds as soon as
``lam <= 2 * (1 - _ARMIJO_DECREASE) * ||r|| / (M * ||delta||^2)``; the halving loop
therefore ends after finitely many trials wherever the Jacobian is nonsingular and the
iterate is interior. See :data:`_MAX_STEP_HALVINGS` for how many that is.

**Why this magnitude.** The constant trades how much of Newton's own step is admitted
against how much progress an accepted step must show, and both ends are one-sided:

- It must be small enough never to reject the asymptotic step. Newton converges
  quadratically near a root, ``||r_next|| ~ K * ||r||^2``, which beats
  ``(1 - _ARMIJO_DECREASE) * ||r||`` by orders of magnitude the moment ``||r||`` is small,
  so ``lam == 1`` is accepted throughout the endgame and the observed convergence rate is
  Newton's own. A constant near one would instead veto perfectly good steps.
- It must stay far above the noise of the comparison. The two norms being compared differ
  by the relative amount ``_ARMIJO_DECREASE * lam``, and each carries a rounding error of
  order ``eps``, so a constant approaching ``eps`` would let rounding decide the test.
  ``1e-4`` sits twelve decades above ``eps`` at ``lam == 1``, and nine decades above it at
  the smallest step length :data:`_MAX_STEP_HALVINGS` admits.

``1e-4`` is also the value the line-search literature settles on for exactly this rule, so
nothing here is unconventional; what matters for this library is that both bounds above are
satisfied with decades to spare, which makes the choice insensitive.
"""

_MAX_STEP_HALVINGS: int = 8
"""
Number of times a Newton step may be halved before its candidate cell is abandoned.

The smallest step length tried is ``2 ** -_MAX_STEP_HALVINGS``, about ``3.9e-3``.

**What a sufficient condition would ask for.** By the expansion in
:data:`_ARMIJO_DECREASE`, the test accepts once ``lam <= 2 * (1 - _ARMIJO_DECREASE) *
||r|| / (M * ||delta||^2)``, and with ``||delta|| <= ||J^-1|| * ||r||`` a sufficient
condition is ``lam <= 2 / (M * ||J^-1||^2 * ||r||)``. Reading the factors off a spline map
of geometric scale ``S`` over a parametric domain of unit extent -- ``||r|| <= S``,
``||J^-1|| ~ cond(J) / S``, and ``M ~ C * S`` with ``C`` the map's parametric curvature
factor -- gives ``lam <= 2 / (C * cond(J)^2)``, in which the scale cancels. That is the
right shape (the halvings needed are a property of the map, not of its units) but a very
loose bound, since it takes the worst case of three independent factors at once: it would
demand 26 halvings for a map with ``C ~ 100`` and ``cond(J) ~ 1e3``, and such a step is
never what decides whether a query is found.

**So the magnitude comes from measurement, with the margin stated.** Over the warped-map
sweep in ``tests/test_bspline_locate.py`` -- 2400 queries, 60 configurations, dimensions 1
to 3, degrees 1 to 5, at unit scale and at an offset of ``1e6``, on mappings that fold --
the smallest budget that loses no point is **2**: at 1 the sweep loses 2 queries and at 0 it
loses 6. This budget is four doublings of step length beyond that, and covers every point
where ``C * cond(J)^2 <= 2 ** (_MAX_STEP_HALVINGS + 1)``.

**The cost is why it is not larger.** A candidate that will never converge spends its whole
budget before being abandoned, so the budget sets the price of a doomed solve, and doomed
solves are the common case for a query that is off the mapping's image. Against the
undamped iteration this fix replaces, measured on the sweep above: at this budget, queries
on the image cost 5 % more and queries off it 31 % *less* (the early abandonment more than
pays for the line search); at a budget of 30 both cost about three times more.

Exhausting the budget abandons that candidate cell rather than stepping anyway. It is not a
verdict on the query: the second start of :func:`_nearest_corner_starts` and the remaining
candidate cells are tried afterwards, and only a query that fails all of them is reported
not found.
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


def _cell_parametric_bounds(
    space: BsplineSpace, cell_ids: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the parametric box of each of the given knot-span cells.

    Args:
        space (BsplineSpace): The space the cells belong to, with no periodic direction.
        cell_ids (npt.NDArray[np.int64]): Flat cell ids in C-order over
            ``space.num_intervals``, shape ``(n,)``.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ``(lo, hi)``, both of
        shape ``(n, space.dim)``: the consecutive breakpoints bracketing each cell in each
        parametric direction.
    """
    multi = np.unravel_index(cell_ids, space.num_intervals)
    lo = np.empty((cell_ids.shape[0], space.dim), dtype=np.float64)
    hi = np.empty_like(lo)
    for axis, sub in enumerate(space.spaces):
        breakpoints = np.asarray(
            sub.get_unique_knots_and_multiplicity(in_domain=True)[0], dtype=np.float64
        )
        index = multi[axis]
        lo[:, axis] = breakpoints[index]
        hi[:, axis] = breakpoints[index + 1]
    return lo, hi


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
    lo, hi = _cell_parametric_bounds(space, cell_ids)
    return 0.5 * (lo + hi)


def _nearest_corner_starts(
    spline: Bspline,
    cell_ids: npt.NDArray[np.int64],
    targets: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the corner of each cell whose image is nearest that cell's query point.

    The second starting guess for a candidate cell whose midpoint start did not converge.
    A monotone iteration can only reach the root whose basin it starts in, and a cell
    midpoint is one guess out of many: on a mapping that folds, the residual has minima on
    the domain boundary that a midpoint start can descend into and then never leave, while
    another point of the *same* cell descends into the root instead (measured). Retrying
    matters most exactly where the candidate loop cannot help -- a query with a single
    candidate cell otherwise gets one Newton solve and no fallback at all.

    Among the cell's ``2 ** dim`` corners, the one with the nearest image is the one that
    starts the iteration at the smallest residual, which is the only ordering the merit
    function gives. The corners are visited one at a time rather than materialized
    together, so the cost is ``2 ** dim`` map evaluations and ``O(n * dim)`` memory.

    Args:
        spline (Bspline): A ``float64`` spline with ``rank == dim``, non-periodic.
        cell_ids (npt.NDArray[np.int64]): One candidate cell per query point, as flat ids
            in C-order over ``space.num_intervals``, shape ``(n,)``.
        targets (npt.NDArray[np.float64]): The query points, shape ``(n, rank)``.

    Returns:
        npt.NDArray[np.float64]: Shape ``(n, dim)`` parametric starting guesses.
    """
    lo, hi = _cell_parametric_bounds(spline.space, cell_ids)
    dim = lo.shape[1]
    axes = np.arange(dim)
    best = np.empty_like(lo)
    best_distance = np.full(cell_ids.shape[0], np.inf, dtype=np.float64)

    for corner in range(1 << dim):
        take_hi = ((corner >> axes) & 1).astype(np.bool_)
        candidate = np.where(take_hi, hi, lo)
        distance = np.asarray(
            np.linalg.norm(_eval_map(spline, candidate) - targets, axis=1), dtype=np.float64
        )
        closer = distance < best_distance
        best[closer] = candidate[closer]
        best_distance[closer] = distance[closer]

    return best


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


class _NewtonState(NamedTuple):
    """The state of the batched Newton iteration, one row per query point.

    Three invariants hold on entry to and exit from every step, for every row ``i``:
    ``lo <= xi[i] <= hi``, ``residual[i] == F(xi[i]) - target[i]``, and ``res_norm[i] ==
    ||residual[i]||``. Carrying the residual next to the iterate is what lets one map
    evaluation serve both the line search's acceptance test and the next round's Newton
    direction, so the damping costs no evaluation at all wherever the full step is
    accepted; carrying the box is what makes staying inside it the state's own business.

    :func:`_accept_damped_step` writes into the three arrays in place, at the rows it
    advanced, in the same ``out``-argument style the rest of Layer 2 uses. ``lo`` and ``hi``
    are fixed for the whole solve.

    Attributes:
        xi (npt.NDArray[np.float64]): Current parametric iterates, shape ``(n, dim)``.
        residual (npt.NDArray[np.float64]): ``F(xi) - targets``, shape ``(n, rank)``.
        res_norm (npt.NDArray[np.float64]): Row 2-norms of ``residual``, shape ``(n,)``.
        lo (npt.NDArray[np.float64]): Lower corner of the parametric domain box, shape
            ``(dim,)``.
        hi (npt.NDArray[np.float64]): Upper corner of the same box, shape ``(dim,)``.
    """

    xi: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]
    res_norm: npt.NDArray[np.float64]
    lo: npt.NDArray[np.float64]
    hi: npt.NDArray[np.float64]


def _accept_damped_step(
    spline: Bspline,
    targets: npt.NDArray[np.float64],
    state: _NewtonState,
    active: npt.NDArray[np.int64],
    delta: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Take the longest damped Newton step that reduces the residual, per active point.

    A backtracking line search on the merit function ``||F(xi) - x||``: the full step is
    tried first and then halved until it satisfies the Armijo test ``||F(xi - lam * delta)
    - x|| <= (1 - _ARMIJO_DECREASE * lam) * ||F(xi) - x||``, up to
    :data:`_MAX_STEP_HALVINGS` halvings.

    The trial iterate is clipped to the parametric domain box, so the search runs along the
    projected path ``lam -> clip(xi - lam * delta)`` rather than along the ray. That path
    still tends to ``xi`` as ``lam`` does, so shortening the step remains meaningful on the
    boundary; a direction that leaves the box along the whole path is rejected at every
    ``lam``, which is the correct answer for a point whose Newton direction only points out
    of the domain.

    One map evaluation per halving level serves every point still searching at that level,
    so a round in which every point accepts its full step costs a single evaluation.

    Args:
        spline (Bspline): A ``float64`` spline with ``rank == dim``, non-periodic.
        targets (npt.NDArray[np.float64]): Physical query points, shape ``(n, rank)``.
        state (_NewtonState): The iteration state; its three arrays are updated in place at
            the rows of ``active`` that accept a step, and left untouched at the others.
        active (npt.NDArray[np.int64]): Row indices to step, shape ``(m,)``.
        delta (npt.NDArray[np.float64]): Undamped Newton steps ``J^-1 * residual`` for
            those rows, shape ``(m, dim)``, in the order of ``active``.

    Returns:
        npt.NDArray[np.bool_]: Shape ``(m,)``, True where a step was accepted. A False
        entry means the line search was exhausted, and its point is at a residual minimum
        no Newton step from it improves on.
    """
    accepted = np.zeros(active.size, dtype=np.bool_)
    searching = np.arange(active.size, dtype=np.int64)
    step_length = 1.0

    for _ in range(_MAX_STEP_HALVINGS + 1):
        rows = active[searching]
        trial_xi = np.clip(state.xi[rows] - step_length * delta[searching], state.lo, state.hi)
        trial_residual = _eval_map(spline, trial_xi) - targets[rows]
        trial_norm = np.asarray(np.linalg.norm(trial_residual, axis=1), dtype=np.float64)

        passes = trial_norm <= (1.0 - _ARMIJO_DECREASE * step_length) * state.res_norm[rows]
        advanced = rows[passes]
        state.xi[advanced] = trial_xi[passes]
        state.residual[advanced] = trial_residual[passes]
        state.res_norm[advanced] = trial_norm[passes]
        accepted[searching[passes]] = True

        searching = searching[~passes]
        if searching.size == 0:
            break
        step_length *= 0.5

    return accepted


def _newton_refine(
    spline: Bspline,
    targets: npt.NDArray[np.float64],
    xi_start: npt.NDArray[np.float64],
    tol_phys: float,
    max_iter: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Run a batched damped Newton inversion, one starting guess per target point.

    Every point is iterated independently but evaluated collectively: each round drops the
    points that have converged, the points whose Jacobian has become rank-deficient, and
    the points whose line search was exhausted, then takes one damped Newton step for the
    rest. Iterates are clamped to the parametric domain box, which keeps them inside the
    evaluators' legal input range; it is :func:`_accept_damped_step`, not the clamp, that
    bounds an overlong step from an ill-conditioned Jacobian.

    Because every accepted step reduces the residual, the residuals of the iterates form a
    monotonically non-increasing sequence. It is in fact strictly decreasing, hence free of
    cycles, for every residual in the normal range: the accepted step satisfies ``||r_next||
    <= (1 - _ARMIJO_DECREASE * lam) * ||r||`` and that bound is below ``||r||`` whenever
    ``_ARMIJO_DECREASE * lam`` exceeds the rounding of the product, which it does by nine
    decades at the smallest step length admitted. The one gap is a *subnormal* residual,
    where the relative spacing is large enough that the product can round back to ``||r||``;
    that is 300 decades below any threshold except a subnormal ``tol``. So a point that is
    not converging exhausts its line search and is abandoned, instead of cycling until the
    iteration budget runs out.

    Args:
        spline (Bspline): A ``float64`` spline with ``rank == dim``, non-periodic.
        targets (npt.NDArray[np.float64]): Physical query points, shape ``(n, rank)``.
        xi_start (npt.NDArray[np.float64]): Starting parametric guesses, shape
            ``(n, dim)``.
        tol_phys (float): Convergence threshold on ``||F(xi) - x||_2``, a distance in
            physical units.
        max_iter (int): Maximum number of accepted Newton steps. The residual is tested
            once more after the last step, so a point converging on step ``max_iter`` is
            reported. The extra map evaluations a line search makes are not steps.

    Returns:
        tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]: ``(converged, xi)``.
        ``converged`` has shape ``(n,)``; ``xi`` has shape ``(n, dim)`` and holds each
        point's last accepted iterate, which is a solution only where ``converged`` is
        True.
    """
    dim = xi_start.shape[1]
    domain = np.asarray(spline.space.domain, dtype=np.float64)
    lo, hi = domain[:, 0], domain[:, 1]

    xi = np.clip(xi_start, lo, hi)
    residual = _eval_map(spline, xi) - targets
    state = _NewtonState(
        xi=xi,
        residual=residual,
        res_norm=np.asarray(np.linalg.norm(residual, axis=1), dtype=np.float64),
        lo=lo,
        hi=hi,
    )
    converged = state.res_norm <= tol_phys
    active = np.arange(xi.shape[0], dtype=np.int64)[~converged]

    for _ in range(max_iter):
        if active.size == 0:
            break

        jacobian = _eval_jacobian(spline, state.xi[active])
        singular_values = np.asarray(np.linalg.svd(jacobian, compute_uv=False), dtype=np.float64)
        healthy = singular_values[:, -1] > (dim * _JACOBIAN_RANK_REL_TOL * singular_values[:, 0])
        active = active[healthy]
        if active.size == 0:
            break
        delta = np.asarray(
            np.linalg.solve(jacobian[healthy], state.residual[active][..., np.newaxis]),
            dtype=np.float64,
        )[..., 0]

        active = active[_accept_damped_step(spline, targets, state, active, delta)]
        done = state.res_norm[active] <= tol_phys
        converged[active[done]] = True
        active = active[~done]

    return converged, state.xi


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


def _validate_invertible(spline: Bspline, tol: float | None, max_iter: int) -> None:
    """Check everything about an inversion request that does not depend on the query points.

    Args:
        spline (Bspline): The spline to invert.
        tol (float | None): The caller's convergence threshold, or ``None`` for the default.
        max_iter (int): The caller's iteration budget.

    Raises:
        NotImplementedError: If ``rank > dim`` (an embedded curve or surface).
        ValueError: If ``rank < dim``, any direction is periodic, a rational spline has a
            non-positive weight, ``max_iter < 1``, or ``tol`` is given and is not positive
            and finite.
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
            ``max_iter < 1``, or ``tol`` is given and is not positive and finite. See
            :func:`_validate_invertible` and :func:`_validate_points`.
    """
    dim, rank = spline.dim, spline.rank
    _validate_invertible(spline, tol, max_iter)

    # Ensure background JIT compilation is complete before calling Numba kernels that use
    # parallel=True (avoids concurrent-compilation crash), as the other Layer 2 entry points
    # over parallel kernels do. The inversion evaluates the mapping many times in quick
    # succession, so it hits the window the import-time warmup thread leaves open.
    wait_for_jit_warmup()

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
        cells = np.asarray([candidates[i][slot] for i in batch], dtype=np.int64)

        starts = _cell_midpoints(context.spline.space, cells)
        converged, xi = _newton_refine(context.spline, pts[index], starts, tol_phys, max_iter)
        ref_coords[index[converged]] = xi[converged]

        # Second start for the candidates the midpoint could not resolve; see
        # _nearest_corner_starts. It can only turn a "not found" into a solution that
        # passes the same residual test, never loosen what "found" means.
        retry = ~converged
        if bool(np.any(retry)):
            retry_index, retry_cells = index[retry], cells[retry]
            corner_starts = _nearest_corner_starts(context.spline, retry_cells, pts[retry_index])
            retried, xi = _newton_refine(
                context.spline, pts[retry_index], corner_starts, tol_phys, max_iter
            )
            ref_coords[retry_index[retried]] = xi[retried]
            converged[retry] = retried

        found.extend(int(i) for i in index[converged])
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
