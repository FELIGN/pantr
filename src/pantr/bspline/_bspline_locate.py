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

   The damping is a backtracking line search over a step first capped to one parametric
   domain extent, and it is what makes the iteration usable away from the basin of
   attraction: an undamped Newton step is only guaranteed to reduce the residual near a
   root, and out there it can be arbitrarily longer than the residual it is trying to
   remove -- longer, in fact, than the domain the answer has to lie in. Accepting such a
   step and then clamping the result to the domain box does not merely waste an iteration,
   it can be *periodic* -- the clamp maps two off-basin iterates onto each other -- and
   then no iteration budget and no tolerance recovers the point. Requiring each accepted
   step to reduce the residual removes both failure modes at once, and makes the residual
   of the accepted iterates a monotonically non-increasing sequence.

   Stopping and acceptance are two different thresholds: the iteration runs until the
   residual is below what the *image* alone entitles the caller to, and a result counts as
   found once it is below what the *parametrization* can attain. On a knot vector far from
   the parametric origin those are decades apart, and serving both from one number would
   hand back the looser of them. :class:`_LocateThresholds` derives the split, and
   :func:`_default_tolerance` builds the two numbers.

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
returned coordinates invert the promoted mapping to the accuracy the caller's own
tolerance tier names, ``64 * eps(dtype) * physical`` -- *not* to float64 accuracy, which
the solver could reach but is never asked for: :func:`_default_tolerance` takes the tier
from the caller's dtype. Measured on a float32 unit patch at the origin, against targets
that are exact images of the promoted map: worst residual ``1.07e-05`` against a stopping
threshold of ``1.08e-05``, where the float64 tier would have demanded ``2.01e-14``. So a
float32 caller's answer is capped at their declared precision, 5.3e8 above what the
iteration is capable of, and evaluating it on the original ``float32`` spline reproduces
the query point to float32 accuracy only.

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
rounding noise and the solve returns an arbitrary step. It is deliberately permissive -- it only
rejects a Jacobian whose condition number reaches ``1 / (dim * eps)`` -- because a merely
ill-conditioned Jacobian still gives a usable step, and :data:`_MAX_STEP_EXTENTS` bounds an
overlong one.

The test is a ratio, so it is invariant under scaling the geometry, and under scaling the
parametrization **isotropically**. It is *not* invariant under scaling the parametric directions
independently, which is no more exotic than a different knot-vector span per direction: the
Jacobian is right-multiplied by a diagonal matrix, and only a scalar multiple of a signed
permutation leaves singular values alone. Exactly, ``J = diag(1, 100 * eps)`` passes against a
threshold of ``2 * eps``, while under ``T = diag(1, eps / 100)`` the same map has ratio ``eps**2``
and is rejected as rank-deficient. This bites only where the Jacobian already sits near the
guard, so it changes no verdict this library has been observed to reach, but the invariance is
narrower than it looks.
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

**When a step is accepted at all.** Let ``dtilde`` be ``delta`` with every coordinate zeroed
that the clip holds fixed (coordinate ``k`` is held when ``xi[k]`` lies on the domain face
that ``-delta[k]`` points out of), and let ``Jv`` be the Jacobian of the polynomial piece the
segment ``xi - t * dtilde`` enters for small ``t > 0``. Some small enough ``lam`` passes if and
only if

    <r, Jv @ dtilde>  >  _ARMIJO_DECREASE * ||r||^2,

and under the reverse strict inequality every sufficiently small ``lam`` *fails*, so an
exhausted search is a statement about the point and not about the budget. Only the product
``Jv @ dtilde`` is well defined, not ``Jv`` itself: a step along a cell face has two pieces to
choose between and they agree on that product, not on the transverse columns.

**An iterate in the open interior of a knot-span cell, with no coordinate held, always
passes.** There ``Jv == J``, the left-hand side is ``||r||^2``, and the test holds with
``_ARMIJO_DECREASE < 1`` to spare. This needs only differentiability of the map at that one
point together with a nonsingular Jacobian -- no second derivative and no bound on one, which
is why a degree-1 map having no global bound on its second derivatives is not a threat to
termination. A second-derivative bound buys a step *length* rather than existence, and only
for ``lam`` short enough to keep the trial point inside the cell it was derived on; see
:data:`_MAX_STEP_EXTENTS`, which bounds that length by construction instead.

**An iterate on a knot line where the map is only C0** -- a simple interior knot at degree 1,
or any knot of multiplicity at least the degree -- can fail at every ``lam``. The span search
is right-closed, so ``J`` is the right-hand derivative; when the step goes left, ``Jv`` is the
left-hand one and the criterion becomes ``rho > _ARMIJO_DECREASE`` with ``rho`` the ratio of
the two directional derivatives along the step. A kink across which that derivative shrinks by
more than ``1 / _ARMIJO_DECREASE == 1e4`` therefore has every step rejected *while the residual
is still strictly decreasing along all of them*, and the candidate is abandoned. Measured at
degree 1 with slopes ``2e-5`` and ``1.99998``, and at degree 3 across a triple interior knot: no
budget up to 40 converges, and moving the iterate one ulp off the knot converges at 8. The
configuration is reachable rather than hypothetical, since :func:`_nearest_corner_starts`
returns cell corners and those are knot values. It is a known limitation of a monotone line
search on a C0 join, not something a larger budget fixes.

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

``1e-4`` is also the established value for exactly this rule -- sufficient decrease tested on
``||F||`` itself, for Newton on ``F(x) = 0`` -- in C. T. Kelley, *Iterative Methods for Linear
and Nonlinear Equations*, SIAM, 1995, Eq. (8.1) and Algorithm 8.2.1. That attribution is worth
being precise about, because the far more widely cited ``1e-4`` belongs to a *different* test:
sufficient decrease on the squared merit function with a gradient inner product (Nocedal and
Wright, *Numerical Optimization*, 2nd ed., Eq. (3.4)), which is the form PETSc's
``SNESLINESEARCHBT`` and SciPy's ``scalar_search_armijo`` default to, and the form Kelley says
he took the number from. So the choice is conventional, but checking it against the Armijo
literature at large finds the squared form rather than this one.

What matters for this library either way is that both bounds above are satisfied with decades to
spare, which makes the choice insensitive.
"""

_MAX_STEP_EXTENTS: float = 1.0
"""
Longest first trial step of the line search, in parametric domain extents per direction.

The Newton step is not tried at full length when it is longer than this; the search starts at
``_MAX_STEP_EXTENTS / max(reach, _MAX_STEP_EXTENTS)`` instead, where ``reach`` is
``max_k |delta_k| / (hi_k - lo_k)``. It is a trust region, and it is what keeps the halvings
below :data:`_MAX_STEP_HALVINGS` a bounded requirement rather than a hope.

**Why a cap is needed at all.** The step ``delta = J^-1 r`` grows like ``1 / sigma_min(J)``, so a
merely ill-conditioned Jacobian -- one nowhere near the rank guard -- produces a step orders of
magnitude longer than the domain it has to land in. The line search then spends its whole budget
walking that step back inside, and the halvings needed grow like ``log2(||delta|| / L)`` with
``L`` the domain extent (measured across ~30 witnesses spanning degrees 1-5, dimensions 1-3 and
five decades of ``sigma_min``: ``log2(||delta|| / L) + 1.5`` to ``3``). Since ``||delta||`` is
unbounded over legal inputs, no fixed budget covers that, and exhausting it abandons a candidate
that was converging. Measured before this cap existed: a strictly monotone degree-5 map whose
``sigma_min`` is ``1e-4`` -- fifteen orders clear of the rank guard -- was reported *not found*
with library defaults, needing 11 halvings where the budget allowed 8.

**Why one extent.** A step longer than the domain cannot be a step toward a solution *inside* the
domain, whatever the linear model says: the clip truncates it anyway, so the only thing its extra
length changes is how many halvings are needed to discover that. Capping at one extent removes
exactly the ``log2(||delta|| / L)`` term and leaves the 1.5-to-3 halvings the curvature itself
demands. On the witness above it cut the requirement from 11 halvings to 2. Nothing is lost near
the root, where Newton's step is short and the cap is inactive, so the asymptotic rate is
untouched.

**Why per direction, normalized by each direction's own extent.** ``max_k |delta_k| / (hi_k -
lo_k)`` is invariant under scaling the parametric directions independently, which is nothing more
exotic than a different knot-vector span per direction; a cap on ``||delta||_2`` against a single
length would not be, and would tighten or loosen with the aspect ratio of the domain box.
"""

_MAX_STEP_HALVINGS: int = 8
"""
Number of times a Newton step may be halved before its candidate cell is abandoned.

The shortest step length tried is ``2 ** -_MAX_STEP_HALVINGS`` of the first one, and the first
one is itself bounded by :data:`_MAX_STEP_EXTENTS`. That pairing is the whole point: without the
cap the halvings needed grow like ``log2(||delta|| / L)`` -- ``||delta|| = ||J^-1 r||`` and ``L``
the domain extent -- which is unbounded over legal inputs, so **no** fixed budget is correct and
a converging solve gets abandoned. With the cap that term is gone and what remains is set by the
map's curvature, which is bounded on any given geometry.

**Measured requirement, with the cap in place.** Every query converges at a budget of **2**:
across the warped-map sweep in ``tests/test_bspline_locate.py`` (2400 queries, 60
configurations, dimensions 1 to 3, degrees 1 to 5, unit scale and an offset of ``1e6``, on
mappings that fold), and across 27 near-critical configurations spanning ``sigma_min`` from
``1e-3`` to ``1e-5`` and raw Newton steps up to ``7e5`` domain extents. At a budget of 1 the
sweep loses 2 queries and at 0 it loses 6. This budget is four doublings beyond that.

**What the budget does *not* cover, stated plainly.** A single candidate solve can need more:
on the folding sweep family, solves converge using up to **15** halvings, and that number is
real rather than an artifact of the ceiling -- the depth histogram is identical at budgets 16
and 24, so it has genuinely stopped. Those deep solves are not what decides a query, because
another candidate cell or the second start reaches the root first, which is why the sweep loses
nothing from a budget of 2 upward. But that redundancy is a property of the candidate list, not
headroom in the line search, and it should not be read as margin: **the honest statement is that
a solve needing more than 8 halvings is abandoned, and a query all of whose candidates and both
starts need more than 8 would be lost.** Nothing measured exhibits one.

**Cost is what stops it being larger.** A candidate that will never converge spends its whole
budget before being abandoned, and doomed candidates are the common case for a query off the
mapping's image. Measured against the undamped iteration this replaces: at this budget, queries
on the image cost 10 % more and queries off it 31 % *less*, the early abandonment more than
paying for the search. At 12 the on-image case costs about twice the undamped baseline, to cover
solves whose queries are already found.

Exhausting the budget abandons that candidate cell rather than stepping anyway. It is not a
verdict on the query: the second start of :func:`_nearest_corner_starts` and the remaining
candidate cells are tried afterwards, and only a query that fails all of them is reported not
found.
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
        scale (float): The length the default *acceptance* threshold is expressed in,
            covering the geometry's size, its distance from the origin, and what its
            parametrization can resolve; see :func:`_geometric_scale`. It is
            ``max(physical, parametric)``.
        physical (float): The image-only half of that scale, which the default *stopping*
            threshold is expressed in; see :func:`_physical_scale` and
            :class:`_LocateThresholds`.
        diagonal (float): The geometry's own bounding-box diagonal, kept alongside the
            scale it feeds because :func:`_locate_impl` grades the parametric term against
            it before accepting a default threshold.
        parametric (float): The parametric length from :func:`_parametric_scale`, kept for
            the same comparison.
    """

    spline: Bspline
    bvh: BVH
    scale: float
    physical: float
    diagonal: float
    parametric: float


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


def _parametric_reach(domain: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return, per direction, how far the parametric box sits from the origin in its own widths.

    ``max(|lo_k|, |hi_k|) / (hi_k - lo_k)``: the dimensionless factor by which the
    representable parameters around the box are coarser, *relative to the extent they have
    to resolve*, than they would be on a box touching the origin. It is one for a knot
    vector spanning ``[0, L]``, one half for one spanning ``[-a, a]``, and grows linearly
    once the box is further from the origin than it is wide. :func:`_geometric_scale` is
    where it is used and where the argument for it is written down.

    Returned per direction rather than reduced here, because it is only meaningful paired
    with *that same direction's* physical span: reducing first would pair the worst-offset
    direction with another direction's stretch, which on an anisotropic domain over-loosens
    the threshold by exactly the ratio between them -- a factor of ``1e6`` on a map whose
    second direction has a ``1e-6`` image extent, on geometry the physical-only threshold
    inverted without losing anything. See :func:`_parametric_scale`.

    A direction with no extent contributes nothing rather than dividing by zero. Such a
    direction carries no knot span, hence no cell for the inversion to place anything in,
    and an infinite scale would report every query *found* -- the opposite failure to the
    one the reach exists to remove, and the worse of the two. The test is for an exact zero,
    which is narrower than the B-spline layer's own notion of a vanishing extent
    (``8 * eps * scale``): a positive extent below that is possible only through
    ``snap_knots=False``, and is a knot vector whose own layer would not vouch for it.

    Args:
        domain (npt.NDArray[np.float64]): The parametric box, shape ``(dim, 2)``, as
            :attr:`pantr.bspline.BsplineSpace.domain` returns it.

    Returns:
        npt.NDArray[np.float64]: Shape ``(dim,)``, non-negative and finite.
    """
    lo, hi = domain[:, 0], domain[:, 1]
    extent = hi - lo
    magnitude = np.maximum(np.abs(lo), np.abs(hi))
    return np.divide(magnitude, extent, out=np.zeros_like(extent), where=extent > 0.0)


def _net_span_per_direction(spline: Bspline) -> npt.NDArray[np.float64]:
    """Return the physical length each parametric direction sweeps, one entry per direction.

    For direction ``k``, the diagonal of the bounding box of one line of the control net
    along ``k``, maximised over the transverse indices. This is the factor that supplies
    ``sigma_max`` in :func:`_geometric_scale`'s derivation: it stands for ``L_k *
    ||J e_k||``, the physical length a full traverse of direction ``k`` covers.

    **Exact for an affine map.** The net line is then a straight segment, whose coordinate
    box has sides ``|B_i - A_i|`` and therefore diagonal exactly ``||B - A||``; and ``B - A``
    is ``L_k * J e_k``, because an open knot vector's Greville abscissae span the full
    parametric extent and linear precision reproduces the affine map on them.

    **Why the box of the line and not its chord.** The chord ``||P_last - P_first||`` is also
    exact for an affine map and cheaper, but it vanishes on a direction that doubles back --
    a folded map whose net returns to where it started -- and a zero there would delete the
    parametric term exactly where the map is hardest, reintroducing the defect this exists to
    remove. The bounding box measures the excursion instead of the displacement, so it is
    never smaller than the chord and never larger than the geometry's own diagonal (the line
    lies inside the control-point box).

    Args:
        spline (Bspline): A ``float64`` spline; rational splines are read through their
            projected control points, as the rest of this module does.

    Returns:
        npt.NDArray[np.float64]: Shape ``(dim,)``, non-negative.
    """
    cp = _physical_control_points(spline)
    spans = np.empty(spline.dim, dtype=np.float64)
    for axis in range(spline.dim):
        line_extent = cp.max(axis=axis) - cp.min(axis=axis)
        spans[axis] = float(np.linalg.norm(line_extent, axis=-1).max())
    return spans


def _parametric_scale(spline: Bspline) -> float:
    """Return the physical length the parametrization's own resolution puts a floor at.

    ``max_k [ reach_k * net_span_k ]``, pulled into the format the *iterate* carries. The
    pairing is per direction and the maximum is taken over the products, never over the two
    factors separately; :func:`_parametric_reach` says what goes wrong otherwise.

    **The precision ratio.** The inversion always iterates in ``float64``, whatever the
    caller's spline carries -- see this module's precision contract. So one ulp of a
    parametric coordinate is ``eps64 * |xi|`` even for a ``float32`` caller, while
    :func:`_locate_impl` multiplies this length by ``get_default(spline.dtype)``, which is
    ``64 * eps32`` there. Handing that tier a ``float64`` quantization would overshoot by
    ``eps32 / eps64``, a factor of ``5.4e8``, and it shows: an otherwise ordinary
    ``float32`` patch of unit size at a parametric offset of ``1e5`` would be given a
    threshold of ``0.76`` of its own diameter, so a query half a diameter outside its image
    would be reported found. Multiplying by ``eps64 / eps(dtype)`` cancels the caller's
    format back out, leaving ``64 * eps64 * reach * span`` in every format. The ratio is
    exactly ``1.0`` for a ``float64`` spline, which is the common path.

    The two physical lengths in :func:`_geometric_scale` are deliberately *not* treated this
    way. They grade the caller's own data, whose precision really is the caller's format;
    this one grades the solver's iterate, whose precision is not.

    Args:
        spline (Bspline): The spline being inverted, before promotion, so that its dtype is
            the caller's.

    Returns:
        float: A physical length, non-negative and finite.
    """
    domain = np.asarray(spline.space.domain, dtype=np.float64)
    products = _parametric_reach(domain) * _net_span_per_direction(spline)
    precision_ratio = float(np.finfo(np.float64).eps) / float(np.finfo(spline.dtype).eps)
    return precision_ratio * float(products.max())


def _physical_scale(lo: npt.NDArray[np.float64], hi: npt.NDArray[np.float64]) -> float:
    """Return the length the *accuracy* a returned coordinate is entitled to is a multiple of.

    Two lengths matter and the larger one has to win:

    - the **diagonal** of the geometry's bounding box, which is what makes a residual
      threshold a *relative* accuracy statement about the mapping;
    - the largest coordinate **magnitude** present, because the computed residual
      ``F(xi) - x`` cannot be resolved below ``~C * eps * max|coordinate|`` (the de Boor
      sum accumulates ``C`` roundings, of the order of ``prod(degree + 1)``), whatever
      the box's extent is.

    Using the diagonal alone silently makes the tolerance unreachable for a geometry far
    from the origin -- a patch of diameter 1 sitting at ``x = 1e6`` has a residual floor
    around ``1e-10`` while ``eps * 1`` would be demanded -- so the scale is the maximum
    of the two.

    Both are properties of the **image** alone. Nothing about the parametrization enters,
    which is exactly what makes this the length the accuracy contract is read off:
    reparametrizing a geometry does not change what a caller may expect of a coordinate it
    gets back. :func:`_geometric_scale` adds the length the parametrization *can resolve*,
    which is a different question and gets a different threshold; :class:`_LocateThresholds`
    is where the two are separated.

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
        float: The physical scale, strictly positive.
    """
    box_lo = lo.min(axis=0)
    box_hi = hi.max(axis=0)
    diagonal = float(np.linalg.norm(box_hi - box_lo))
    magnitude = float(max(np.abs(box_lo).max(), np.abs(box_hi).max()))
    scale = max(diagonal, magnitude)
    return scale if scale > 0.0 else 1.0


def _geometric_scale(physical: float, parametric: float) -> float:
    """Return the length the default *acceptance* threshold is a multiple of.

    The larger of two lengths: the image-only one of :func:`_physical_scale`, and the
    **parametric** length of :func:`_parametric_scale`, because a parametric coordinate is
    a float64 too and the residual cannot be resolved below what one ulp of it produces.
    Which of the two thresholds this length serves, and which one
    :func:`_physical_scale` serves alone, is settled in :class:`_LocateThresholds`.

    **The parametric term.** The two lengths behind :func:`_physical_scale` are properties
    of the image alone, and a threshold built from them is equally unreachable whenever the
    *parametrization* cannot resolve the geometry that finely. One ulp of a parametric
    coordinate is ``eps * |xi_k|``, which the map turns into a physical displacement of
    ``eps * |xi_k| * ||J e_k||``, so no representable parameter drives the residual below
    about half the largest such step, however good the solver is. That is a floor of exactly
    the kind the ``magnitude`` term already covers, and it is invisible to both of the other
    lengths:
    translating the knot vector changes it and changes neither of them. Measured on a
    unit-width span carrying the exactly affine map ``F(xi) = xi - offset``, against the
    threshold the two physical lengths alone give: the floor is ``0.008`` of it at offset
    zero, ``0.504`` at ``1e2``, ``4.0`` at ``1e3`` and ``4096`` at ``1e6``, and every
    query is lost from ``1e3`` upward.

    Splitting that floor at the parametric extent ``L_k = hi_k - lo_k`` separates what can
    be read off the space from what cannot::

        eps * |xi_k| * ||J e_k||  ==  eps * (|xi_k| / L_k) * (L_k * ||J e_k||)

    The first factor is the reach of :func:`_parametric_reach`. The second is the physical
    length a full traverse of direction ``k`` sweeps, and **the control net's span along
    that direction supplies it** -- that is where ``sigma_max`` enters, through
    :func:`_net_span_per_direction`, exactly for an affine map. The product is formed **per
    direction and maximised afterwards**. Maximising the two factors separately, as an
    earlier form of this did, pairs the worst-offset direction with a different direction's
    stretch: measured on an affine map whose second direction sits at offset ``1e6`` while
    its image extent shrinks, that inflates the threshold to ``2.4e6`` times the attainable
    floor once the extent reaches ``1e-4``, on geometry the two physical lengths alone had
    inverted without losing anything. With the product formed per direction the ratio stays
    at ``244`` across all four decades of extent. Shrinking the extent further keeps
    widening the gap, but stops being worth quoting a number for: the floor falls below one
    ulp of unity, where which representable point is sampled moves it by a factor of a few.
    The *term's* own inflation there is exact and needs no measurement -- it is
    ``diagonal / net_span_1``, so ``1e6`` at an extent of ``1e-6``.

    **What the substitution assumes, and what the tier absorbs.** A map whose local stretch
    exceeds the average over its own direction has a floor above this length, by the
    amplification ``A = max_k L_k * sup||J e_k|| / net_span_k``. The floor is a half ulp and
    the tier is ``64 * eps``, so the gap is absorbed to ``A`` of order ``100``: measured on a
    degree-1 monotone family carrying half its rise in a single span, floor over threshold
    is ``0.13`` at ``A == 32`` and ``0.53`` at ``A == 128``, passing one between ``A == 128``
    and ``A == 256``.

    Three things about that limit, so it is not read as more than it is. It is **not new**:
    the physical ``diagonal`` has always stood in for stretch the same way, and the identical
    sweep at reach one -- where this term is inactive -- also loses queries once ``A``
    reaches a few hundred. ``A`` is **unbounded** over legal inputs, rationals included, so
    this term is a mitigation and not a guarantee. And past the limit the queries are lost
    against the *physical-only* threshold too, by decades more, so what remains is a residue
    of the same defect rather than anything the cure introduced.

    **Why the reach and not the parametric magnitude alone.** Dividing by the extent is
    what keeps the threshold covariant. Under ``xi -> lam * xi`` the coordinate magnitude
    grows by ``lam`` while the Jacobian shrinks by it, so the physical floor is unchanged
    and the threshold must be too; a term reading ``max|xi_k|`` without the extent would
    scale with ``lam``, and a pure reparametrization would change the accuracy demanded of
    an answer that did not move.

    **Why it multiplies a span and not the magnitude.** The parametric floor is a statement
    about stretch, and only a span measures stretch: a constant map has ``J == 0`` and no
    parametric floor at all whatever its coordinates are, and reading the magnitude there
    would invent one. It is also what holds the term at its floor for ordinary geometry --
    reach is one for a knot vector spanning ``[0, L]`` and one half for ``[-a, a]``, and no
    net span exceeds the diagonal, so the term cannot be what decides the maximum.
    **Bitwise** for the common case, not merely to within a rounding: on a map whose net
    span along the widest direction *is* the diagonal, ``[0, L]`` gives ``reach == L / L``,
    exactly ``1.0`` in IEEE arithmetic, and ``1.0 * diagonal`` is exactly ``diagonal``, so
    the three-way maximum is the two-way one bit for bit. Where the net span is shorter the
    term is strictly smaller and the maximum returns the physical value unchanged anyway.
    Nothing parametrized from the origin is held to a different bar than it was.

    An offset that is neither zero nor large does loosen the threshold, and is meant to: a
    knot vector spanning ``[1, 2]`` has reach two, so its bar doubles, because its floor
    doubles too.

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

    That the achieved residual *tracks* the threshold is also why this length may not be
    what the iteration stops at. A threshold this one raises to cover what the
    parametrization can resolve would, as a stopping rule, hand back exactly that much error
    on geometry whose answer was far more accurate one iteration on. Which of the two
    thresholds governs stopping and which governs acceptance is settled in
    :class:`_LocateThresholds`.

    The degenerate fallback lives in :func:`_physical_scale` and covers this function too:
    the physical scale vanishes only when every control point is identical and at the
    origin, and such a geometry has no net span in any direction, so ``parametric`` is zero
    there as well and the maximum is the fallback.

    Args:
        physical (float): The image-only length from :func:`_physical_scale`, which already
            carries the strictly-positive fallback.
        parametric (float): The parametric length from :func:`_parametric_scale`.

    Returns:
        float: The geometric scale, strictly positive and never below ``physical``.
    """
    return max(physical, parametric)


def _build_context(spline: Bspline) -> _LocateContext:
    """Build the cached inversion state of ``spline``.

    Args:
        spline (Bspline): The spline to invert; already validated by
            :func:`_locate_impl`.

    Returns:
        _LocateContext: The promoted spline, the BVH over its per-cell physical boxes,
        and the lengths the two default thresholds are expressed in.
    """
    spline_64 = _promote_to_float64(spline)
    lo, hi = _cell_physical_bounds(spline_64)
    # Read from the caller's spline, not the promoted one: the precision ratio it applies
    # has to see the format the caller's tier will be taken from.
    parametric = _parametric_scale(spline)
    physical = _physical_scale(lo, hi)
    box_lo, box_hi = lo.min(axis=0), hi.max(axis=0)
    return _LocateContext(
        spline=spline_64,
        bvh=BVH.from_cell_bounds(lo, hi),
        scale=_geometric_scale(physical, parametric),
        physical=physical,
        diagonal=float(np.linalg.norm(box_hi - box_lo)),
        parametric=parametric,
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
        npt.NDArray[np.float64]: Shape ``(n, dim)`` parametric starting guesses, each a
        corner of its own cell. That postcondition is why the search starts from a corner
        rather than from an empty buffer: a mapping whose evaluation is not finite (nothing
        rejects a non-finite control point at construction) makes every distance ``nan`` and
        every comparison below False, and an empty buffer would then be returned unwritten.
    """
    lo, hi = _cell_parametric_bounds(spline.space, cell_ids)
    dim = lo.shape[1]
    axes = np.arange(dim)
    best = lo.copy()
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

    The first trial is shortened to at most :data:`_MAX_STEP_EXTENTS` domain extents in every
    parametric direction, so an overlong step from an ill-conditioned Jacobian is capped rather
    than halved back down one level at a time.

    The trial iterate is clipped to the parametric domain box, so the search runs along the
    projected path ``lam -> clip(xi - lam * delta)`` rather than along the ray. That path still
    tends to ``xi`` as ``lam`` does (``clip`` is 1-Lipschitz and fixes ``xi``, so
    ``||p(lam) - xi|| <= lam * ||delta||``), which is what keeps shortening the step meaningful
    on the boundary. What is rejected at *every* ``lam`` is a **constant** projected path -- every
    coordinate either held by the clip or already zero in ``delta`` -- and not merely a direction
    that leaves the box, which with only some coordinates held still moves and can be accepted.

    One map evaluation per halving level serves every point still searching at that level, so a
    round in which every point accepts its first step costs a single evaluation.

    Args:
        spline (Bspline): A ``float64`` spline with ``rank == dim``, non-periodic.
        targets (npt.NDArray[np.float64]): Physical query points, shape ``(n, rank)``.
        state (_NewtonState): The iteration state; its three arrays are updated in place at
            the rows of ``active`` that accept a step, and left untouched at the others.
        active (npt.NDArray[np.int64]): Row indices to step, shape ``(m,)``.
        delta (npt.NDArray[np.float64]): Undamped Newton steps ``J^-1 * residual`` for
            those rows, shape ``(m, dim)``, in the order of ``active``.

    Returns:
        npt.NDArray[np.bool_]: Shape ``(m,)``, True where a step was accepted. A False entry
        means the line search was exhausted: either no step length short enough was reachable
        within :data:`_MAX_STEP_HALVINGS`, or none exists at all because the criterion in
        :data:`_ARMIJO_DECREASE` fails. In one dimension an exhausted point on the boundary
        really is the constrained minimum -- the merit gradient ``J * r == J^2 * delta`` carries
        ``sign(delta)``, which points out of the box. In two or more it need not be, because
        ``sign(delta[k])`` need not agree with ``sign((J.T @ J @ delta)[k])``: a bilinear map
        with ``J.T @ J == [[5, -2], [-2, 1]]`` and ``cond(J) == 5.83`` is abandoned at a corner
        while its residual still decreases into the box (measured). So a False entry means this
        candidate made no progress, not that the point is a minimum of the residual.
    """
    reach = np.max(np.abs(delta) / (state.hi - state.lo), axis=1)
    first = _MAX_STEP_EXTENTS / np.maximum(reach, _MAX_STEP_EXTENTS)

    accepted = np.zeros(active.size, dtype=np.bool_)
    searching = np.arange(active.size, dtype=np.int64)
    shrink = 1.0

    for _ in range(_MAX_STEP_HALVINGS + 1):
        rows = active[searching]
        step_length = shrink * first[searching]
        trial_xi = np.clip(
            state.xi[rows] - step_length[:, np.newaxis] * delta[searching], state.lo, state.hi
        )
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
        shrink *= 0.5

    return accepted


class _LocateThresholds(NamedTuple):
    """The two residual thresholds a solve is governed by, with ``stop <= accept``.

    One number cannot do both jobs. ``stop`` is the **accuracy contract** -- what a caller
    is entitled to expect of a coordinate that comes back -- and ``accept`` is an
    **attainability** statement: the residual below which no representable parameter can
    go, so that a query whose exact preimage is unrepresentable is still reported found.
    Serving both from one number forces the accuracy contract up to the attainability
    bound, and it forces it all the way: the re-measurement recorded in
    :func:`_geometric_scale` found the achieved residual tracking whatever threshold is
    set, to within one part in a hundred, at every tier from ``2`` to ``4096`` epsilons. So
    :func:`_newton_refine` keeps iterating while the residual is above ``stop``, and takes
    the verdict at ``accept``.

    **Why only ``accept`` grows with the parametrization.** ``stop`` is
    ``get_default(dtype) * physical`` with the length of :func:`_physical_scale`, read off
    the **image** alone; ``accept`` is the same tier times that length maximised against
    :func:`_parametric_scale`. How finely a float64 parametric coordinate resolves the
    geometry decides whether a solution *exists* among representable parameters. It says
    nothing about the accuracy a caller is owed, which is a property of the image and of
    nothing else -- reparametrizing a geometry may not change what an answer is worth. For
    a knot vector spanning ``[0, L]`` the two are equal **bit for bit** (the reach is
    exactly ``1.0`` and no net span exceeds the diagonal), so nothing parametrized from the
    origin sees any of this, in accuracy or in cost.

    **Coverage cannot shrink, by construction rather than by measurement.** Every accepted
    step strictly reduces the residual (:data:`_ARMIJO_DECREASE`), so the residuals of the
    accepted iterates are monotonically non-increasing. Lowering the *stopping* test from
    ``accept`` to ``stop`` therefore only extends trajectories the single-threshold
    iteration would have cut short, and a row whose residual once passed ``accept`` still
    passes it at exit, however many further steps it takes and whether it ends by
    converging, by exhausting the line search, by meeting the rank guard or by running out
    of budget. The queries reported found under the split are a superset of those reported
    found under ``accept`` alone, on every map. No appeal to the redundancy of candidate
    cells or of the second start is needed.

    **What each verdict certifies.** ``converged`` is a test on the residual the solve
    actually achieved, taken once at the end, so no coordinate is returned that fails the
    threshold it is graded against -- including one whose starting guess was already inside
    ``accept`` and which the iteration then moved.

    **What it buys.** Measured on a warped unit patch whose second direction is
    parametrized on ``[1e6, 1e6 + 1]`` (degree 3, 4 elements per direction, 60 sampled
    parameters), where ``stop`` is ``2.53e-14`` and ``accept`` is ``1.47e-08``:

    ========================  ================  =======  ==============
    targets                   policy            lost     worst residual
    ========================  ================  =======  ==============
    exact images              ``accept`` alone   0 / 60   ``1.20e-08``
    exact images              ``stop`` alone     4 / 60   ``2.46e-14``
    exact images              split              0 / 60   ``4.99e-11``
    a half ulp off the image  ``accept`` alone   0 / 60   ``1.19e-08``
    a half ulp off the image  ``stop`` alone    60 / 60   --
    a half ulp off the image  split              0 / 60   ``1.88e-10``
    ========================  ================  =======  ==============

    The split has the coverage of the looser threshold and 240 times the accuracy of it,
    with the median residual on exact images falling from ``3.8e-11`` to ``2.3e-16``. It
    does **not** reach ``stop`` on every query, and cannot: at this reach one ulp of the
    parametric coordinate already moves the image by ``2.3e-10``. What it reaches is the
    parametric grid's own floor -- the half-ulp worst case of ``1.88e-10`` equals, to five
    digits, the best residual any parameter within two ulps of the true preimage attains.

    **What it costs.** Where the two coincide the code path is identical, and the
    ``2400``-query warped-map sweep of ``tests/test_bspline_locate.py`` (60 configurations,
    dimensions 1 to 3, degrees 1 to 5, physical offsets ``0`` and ``1e6``) measures
    ``1.000x`` map evaluations and ``1.000x`` Jacobians, because a physical offset moves
    both thresholds together. On the fixture above at parametric offset ``1e6``, where the
    split is at its widest: ``1.12x`` map evaluations for targets on the image and
    ``1.61x`` for half-ulp targets, and ``1.15x`` and ``1.25x`` Jacobians. Counting instead
    the *accepted Newton steps* -- one step being one the line search actually took, and one
    **solve** being one :func:`_newton_refine` row, so one query against one candidate cell
    from one start, of which those 60 queries make 69 -- the totals go from 350 to 406 and
    to 398, the median over the solves that step at all from 5 to 6, and the **maximum per
    solve from 10 to 11**, against a default budget of 30. Queries **off** the mapping's
    image, which dominate the cost of a bad batch, are unchanged at ``1.00x``: they never
    reach either threshold, so their trajectories are identical. A worst case of ``1.6x`` on
    the evaluation count, confined to geometry parametrized a million widths from the
    origin, is what the accuracy above is bought with.

    An explicit ``tol`` sets both. A caller who names a threshold has taken responsibility
    for what it means and may well be asking a proximity question rather than an inversion
    one, and refining past it would charge them for accuracy they did not ask for.

    **The ordering is relied on, not enforced.** It holds by construction --
    :func:`_geometric_scale` is a maximum over :func:`_physical_scale`, and an explicit
    ``tol`` sets both members to itself -- and :func:`_newton_refine` uses it: it returns
    ``res_norm <= accept`` alone, with no separate record of which rows stopped, because a
    row that stopped at ``stop`` necessarily passes ``accept``. Were ``accept < stop`` ever
    constructed, the loop would end against the larger number and the verdict be taken
    against the smaller, so nearly every query would be reported not found -- a silent
    collapse of coverage indistinguishable from genuine non-invertibility. No guard is
    added, for the same reason :class:`_NewtonState`'s three invariants carry none: both
    construction sites are in this module and a few lines from the definition. What stands
    in for one is the suite, which pins the relation in both regimes -- equality for a knot
    vector spanning ``[0, L]``, strict inequality at a parametric offset.

    Attributes:
        stop (float): The residual at or below which the iteration stops, in physical
            units. The accuracy contract.
        accept (float): The residual at or below which the result counts as found, in
            physical units. Never below ``stop``.
    """

    stop: float
    accept: float


def _newton_refine(
    spline: Bspline,
    targets: npt.NDArray[np.float64],
    xi_start: npt.NDArray[np.float64],
    thresholds: _LocateThresholds,
    max_iter: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Run a batched damped Newton inversion, one starting guess per target point.

    Every point is iterated independently but evaluated collectively: each round drops the
    points that have reached ``thresholds.stop``, the points whose Jacobian has become
    rank-deficient, and the points whose line search was exhausted, then takes one damped
    Newton step for the rest. Iterates are clamped to the parametric domain box, which
    keeps them inside the evaluators' legal input range; it is :func:`_accept_damped_step`,
    not the clamp, that bounds an overlong step from an ill-conditioned Jacobian.

    Stopping and acceptance are two different thresholds and the reason is in
    :class:`_LocateThresholds`. Only ``thresholds.stop`` ends a trajectory; the verdict is
    a single test of the residual each point actually achieved against
    ``thresholds.accept``, applied once at the end and therefore covering the points the
    loop dropped as well as the ones that stopped.

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
        thresholds (_LocateThresholds): The stopping and acceptance thresholds on
            ``||F(xi) - x||_2``, both distances in physical units.
        max_iter (int): Maximum number of accepted Newton steps. Exhausting it is not a
            verdict: the acceptance test is taken afterwards either way, so a point that
            reached ``thresholds.accept`` on the way is still reported. The extra map
            evaluations a line search makes are not steps.

    Returns:
        tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]: ``(converged, xi)``.
        ``converged`` has shape ``(n,)`` and is ``||F(xi) - x||_2 <= thresholds.accept``
        row by row; ``xi`` has shape ``(n, dim)`` and holds each point's last accepted
        iterate, which is a solution only where ``converged`` is True.
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
    # Tested before the first step, so a starting guess already inside the stopping
    # threshold is returned untouched. Under the acceptance one it would not be: on a
    # geometry far from either origin a cell midpoint can already sit inside that, and
    # returning it would deliver the bar rather than what the map can resolve.
    active = np.arange(xi.shape[0], dtype=np.int64)[state.res_norm > thresholds.stop]

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
        active = active[state.res_norm[active] > thresholds.stop]

    return state.res_norm <= thresholds.accept, state.xi


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


def _default_tolerance(spline: Bspline, context: _LocateContext) -> _LocateThresholds:
    """Return the default thresholds, or refuse a parametrization that can carry none.

    ``get_default(dtype)`` times each of the two lengths the context carries: the
    image-only one of :func:`_physical_scale` to stop at, and the one of
    :func:`_geometric_scale` to accept at. :class:`_LocateThresholds` is where the split
    itself is derived; this function only assembles the numbers, and refuses a
    parametrization whose own resolution would demand a threshold larger than the geometry
    it grades instead of serving it.

    Both are the same tier applied to a length the geometry supplies, so neither introduces
    a constant, and ``stop <= accept`` holds because :func:`_geometric_scale` is a maximum
    over :func:`_physical_scale`.

    **Why refuse rather than clamp.** Past that point the answer carries no information: a
    "found" verdict admitting more than a whole diameter of error says only that the query
    was somewhere near the model. Clamping the scale back would restore an accuracy that
    *cannot be attained*, which is precisely the defect the parametric term exists to
    remove -- the same bug, relocated to the far end of the range rather than fixed. So the
    only honest answers are to serve a meaningless threshold or to say so, and this library
    already says so elsewhere: :func:`pantr.bspline._bspline_knots._check_snapping_kept_an_interval`
    refuses a knot vector whose mesh the format cannot resolve, and names the same remedy.

    **Where the line is.** ``get_default(dtype) * parametric > diagonal``: the one place no
    constant has to be invented, since it compares two lengths the geometry itself supplies
    and is therefore covariant under scaling the geometry and invariant under scaling the
    parametrization. It fires only above a reach of ``1 / (64 * eps)``, about ``7e13`` --
    a knot span sitting seventy trillion times further from the parametric origin than it
    is wide, and within a factor of eight of what the knot layer refuses outright.

    **What it deliberately does not cover.** Below that line the threshold can still be a
    sizeable fraction of the geometry: a reach of ``1e13`` buys ``0.14`` of a diameter, at
    which a query ten percent outside the image is reported found. That is a real loss of
    meaning and it is stated in :meth:`pantr.bspline.Bspline.locate`'s docstring rather
    than legislated here, because every candidate line below ``1`` is a number someone
    picked. The physical half of the scale is left alone entirely: a small geometry far
    from the origin has always been graded this way, and that behaviour is correct.

    An explicit ``tol`` is never refused. A caller who names a threshold has taken
    responsibility for what it means, and may well be asking a proximity question rather
    than an inversion one.

    Args:
        spline (Bspline): The spline being inverted, whose dtype sets the tolerance tier.
        context (_LocateContext): Its cached inversion state.

    Returns:
        _LocateThresholds: The default thresholds on ``||F(xi) - x||_2``, both strictly
        positive.

    Raises:
        ValueError: If the parametric term alone would put the acceptance threshold above
            the geometry's own bounding-box diagonal.
    """
    tier = get_default(spline.dtype)
    if tier * context.parametric > context.diagonal:
        raise ValueError(
            "locate: this knot vector sits so far from the parametric origin that one ulp "
            "of a parametric coordinate moves the image by more than the geometry's own "
            f"size ({tier * context.parametric:.3e} against a bounding-box diagonal of "
            f"{context.diagonal:.3e}), so no residual threshold can mean anything. Move "
            "the knot vector nearer the parametric origin, or pass an explicit tol to say "
            "what distance you want."
        )
    return _LocateThresholds(stop=tier * context.physical, accept=tier * context.scale)


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
            distance in physical units, governing stopping and acceptance alike, or
            ``None`` for the pair :func:`_default_tolerance` builds.
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
            :func:`_validate_invertible` and :func:`_validate_points`. Also if ``tol`` is
            ``None`` and the parametrization cannot support any meaningful threshold; see
            :func:`_default_tolerance`.
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
    # An explicit tol governs stopping as well as acceptance; see :class:`_LocateThresholds`.
    thresholds = (
        _LocateThresholds(stop=float(tol), accept=float(tol))
        if tol is not None
        else _default_tolerance(spline, context)
    )

    n_pts = pts.shape[0]
    cell_ids = np.full(n_pts, -1, dtype=np.int64)
    ref_coords = np.full((n_pts, dim), np.nan, dtype=np.float64)

    # Candidates per point, in ascending cell id: the order the rounds below try them in.
    # The pad is the *acceptance* threshold, which is the one a returned coordinate is
    # graded against, so no cell that could hold a solution is missed. A geometry whose
    # threshold is large -- far from the physical origin, or far from the parametric one --
    # therefore returns more candidate cells per query. That costs solves, never
    # correctness: every extra candidate still has to pass the same residual test. Measured
    # on a 16x16-cell patch, mean candidates per query: 4.0 (of 256) at parametric offsets
    # up to 1e9, 7.0 at 1e11, 57.1 at 1e13 -- the blow-up starting only where the verdict is
    # already near-meaningless and _default_tolerance is about to refuse outright.
    candidates = [np.sort(context.bvh.query_aabb(AABB(x, x).pad(thresholds.accept))) for x in pts]
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
        converged, xi = _newton_refine(context.spline, pts[index], starts, thresholds, max_iter)
        ref_coords[index[converged]] = xi[converged]

        # Second start for the candidates the midpoint could not resolve; see
        # _nearest_corner_starts. It can only turn a "not found" into a solution that
        # passes the same residual test, never loosen what "found" means.
        retry = ~converged
        if bool(np.any(retry)):
            retry_index, retry_cells = index[retry], cells[retry]
            corner_starts = _nearest_corner_starts(context.spline, retry_cells, pts[retry_index])
            retried, xi = _newton_refine(
                context.spline, pts[retry_index], corner_starts, thresholds, max_iter
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
