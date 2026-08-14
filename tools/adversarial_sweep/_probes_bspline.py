"""Probes for the B-spline space, spline, extraction and THB-spline surface.

This is the group the sweep weights heaviest, for one reason: **an interior knot at
multiplicity ``degree + 1``** -- a discontinuous, C^-1 spline -- is a documented public
construction (``BsplineSpace1D.subdivide(n, regularity=-1)``) that several algorithms in
this package silently assume away. So the shape of this module is deliberately not
"one crossing per entry point": it builds a compact set of *spline fixtures* spanning
the multiplicity ladder, the degree corners, both dtypes and the domain-magnitude axis,
and then pushes **every** operation that takes a spline through each of them. An
operation that only breaks on a C^-1 operand is invisible to a per-operation sweep and
obvious to this one.

The other axes are folded in where they bite:

* **degree** 0 (a repeated defect source here), 1, 2, 3, 15, and 62 -- the exact point
  where ``_bincoeff``'s integer recurrence wraps int64.
* **knot magnitude**, through the domain axis: ``BsplineSpace1D``'s knot-snapping
  tolerance is *absolute*, so from ``|knot| ~ 5`` upward it merges nothing and the space
  can silently carry the wrong continuity.
* **periodic and non-clamped** knot vectors, which have already produced a genuinely
  negative index and a span search running off the front of the array.
* **dtype**, because several kernels hardwire a double epsilon while accepting float32.

Invariants asserted here are only ones the code or its docstrings actually claim:
degree elevation and knot insertion leave the map pointwise unchanged, splitting and
restriction reproduce the original on their subdomain, ``reduce_degree`` reproduces the
endpoints *bit for bit* (``_bspline.py:433-436``), a value at an in-domain point lies in
the convex hull of the control points, the local basis sums to one, ``first_basis`` is a
valid index, and ``first_basis_per_interval`` is non-decreasing with successive
differences equal to the interior knot multiplicities (``_bspline_space_1d.py:348-350``).

One trap worth naming: a **non-truncated** hierarchical basis is *not* a partition of
unity -- only the truncated one is (``_thb_spline_space.py:196-202``) -- so the THB
probes assert it for ``truncate=True`` alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np

from pantr.bezier import Bezier
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    SpanwiseElementExtraction,
    THBSplineSpace,
    create_cardinal_knots,
    create_thb_space,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
    create_uniform_space,
    find_roots,
    get_greville_abscissae,
    interpolate_bspline,
    quasi_interpolate_bspline,
)
from pantr.tolerance import get_machine_epsilon

from ._axes import (
    Profile,
    coeff_specs,
    degrees,
    dims,
    domains,
    dtypes,
    interior_multiplicities,
    knot_specs,
    point_specs,
    rng,
)
from ._core import Case, custom, expected_shape

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

    Predicate = Callable[[object], str | None]
    """A bare invariant predicate, before :func:`adversarial_sweep._core.custom` wraps it."""

GROUP = "bspline"
"""Registry name of this probe group."""

_ERROR_SAFETY_FACTOR: Final = 8.0
"""Explicit safety factor absorbing the unmodelled constants of de Boor's recurrence.

Every bound below is derived from first principles (see :func:`_eval_tolerance`) and then
multiplied by this one stated factor. It is fixed and acknowledged rather than tuned per
call site, so a bound that fails is failing by orders of magnitude, not by a constant.
"""

_GRAM_CONDITIONED_DEGREE: Final = 6
"""Highest degree at which the exactly-reducible recovery check is not vacuous.

The Bernstein Gram matrix's condition number grows like ``4 ** degree``, so the honest
bound on a degree-reduction round trip grows with it; past this degree the bound exceeds
the values being compared and the check would assert nothing.
"""

_PAIR: Final = 2
"""Length of the ``(basis, first_basis)`` and ``(left, right)`` two-returns checked here."""

_BINCOEFF_SAFE_DEGREE: Final = 61
"""Highest combined degree at which ``_bincoeff``'s exact-integer recurrence is sound.

``_bspline_degree_core.py:22-67`` wraps int64 from ``n = 62``. That overflow is already
confirmed and logged for ``Bspline.elevate_degree``, ``Bezier.elevate_degree`` and
``_compose_bezier``, so the degree-elevation probes stay at or below this bound: their
job is to find something *else*, and a case that only rediscovered the known overflow
would drown it.
"""


class _Fixture(NamedTuple):
    """One 1D B-spline space plus the axis values that produced it.

    Attributes:
        label (str): Identity of the fixture, embedded in every case label built on it.
        space (BsplineSpace1D): The space itself.
        degree (int): Polynomial degree.
        mult (int): Interior knot multiplicity, or ``0`` when the family has no single
            interior multiplicity (uniform, periodic, minimal).
        domain (tuple[float, float]): The ``(lo, hi)`` the knots were built on.
        dtype (np.dtype[np.float32 | np.float64]): Knot precision.
        params (dict[str, Any]): The axis values, recorded in the journal.
    """

    label: str
    space: BsplineSpace1D
    degree: int
    mult: int
    domain: tuple[float, float]
    dtype: np.dtype[np.float32 | np.float64]
    params: dict[str, Any]


# ---------------------------------------------------------------------------
# Derived tolerances
# ---------------------------------------------------------------------------


def _min_nonzero_span(space: BsplineSpace1D) -> float:
    """Measure the shortest non-empty knot span of a space.

    Args:
        space (BsplineSpace1D): The space.

    Returns:
        float: The shortest positive difference between consecutive knots, or the whole
            knot range when every span is empty.
    """
    knots = np.asarray(space.knots, dtype=np.float64)
    gaps = np.diff(knots)
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        return float(abs(knots[-1] - knots[0])) or 1.0
    return float(np.min(positive))


def _eval_tolerance(
    degree: int,
    dtype: npt.DTypeLike,
    value_scale: float,
    *,
    coord_scale: float = 1.0,
    span: float = 1.0,
) -> float:
    """Derive the bound on the difference between two evaluations of the same map.

    De Boor's recurrence forms the value as a convex combination through ``degree``
    stages. Each stage rounds its two products and their sum, so the value carries at
    most ``3 * degree * eps * value_scale`` of rounding error, and comparing *two*
    representations of the same map doubles that.

    On top of that sits the conditioning of the parametrization. The stage weights are
    ``(x - t_i) / (t_j - t_i)``, and two representations of one map (before and after
    degree elevation, say) hold knots that differ by the rounding of their own
    construction, about ``eps * coord_scale``. That perturbs each weight by
    ``eps * coord_scale / span``, which is why a domain translated to ``1e6`` with spans
    of order one has six orders of magnitude less accuracy available than the unit
    domain -- an honest statement about the input, not slack for the algorithm.

    Args:
        degree (int): Polynomial degree, which sets the number of recurrence stages.
        dtype (npt.DTypeLike): Working precision, which sets machine epsilon. Passing
            the *actual* dtype matters: a kernel that hardwires a double epsilon while
            accepting float32 is exactly what this sweep is looking for.
        value_scale (float): Magnitude of the control points, which the error scales
            with.
        coord_scale (float): Magnitude of the parametric coordinates. Defaults to 1.
        span (float): Shortest non-empty knot span. Defaults to 1.

    Returns:
        float: The bound.
    """
    eps = get_machine_epsilon(dtype)
    conditioning = max(1.0, abs(coord_scale) / span) if span > 0.0 else 1.0
    return _ERROR_SAFETY_FACTOR * (degree + 1) * eps * max(value_scale, 1.0) * conditioning


def _fixture_tolerance(fixture: _Fixture, value_scale: float) -> float:
    """Derive the evaluation tolerance for one fixture.

    Args:
        fixture (_Fixture): The fixture, supplying degree, dtype, domain and spans.
        value_scale (float): Magnitude of the control points.

    Returns:
        float: The bound, per :func:`_eval_tolerance`.
    """
    lo, hi = fixture.domain
    return _eval_tolerance(
        fixture.degree,
        fixture.dtype,
        value_scale,
        coord_scale=max(abs(lo), abs(hi)),
        span=_min_nonzero_span(fixture.space),
    )


# ---------------------------------------------------------------------------
# Invariant predicates
# ---------------------------------------------------------------------------


def _basis_of(result: object) -> npt.NDArray[np.floating[Any]] | None:
    """Extract the basis-value array from a ``(basis, first_basis)`` two-return.

    Args:
        result (object): Whatever the entry point returned.

    Returns:
        npt.NDArray[np.floating[Any]] | None: The first element when the result is a
            pair, else ``None``.
    """
    if isinstance(result, tuple) and len(result) == _PAIR:
        return np.asarray(result[0], dtype=np.float64)
    return None


def _local_partition_of_unity(degree: int, dtype: npt.DTypeLike) -> Predicate:
    """Build a predicate requiring the local basis values to sum to one.

    Cox-de Boor forms each value as a convex combination, so the ``degree + 1`` locally
    supported functions sum to one exactly in real arithmetic wherever the point lies in
    the domain -- for *any* valid knot vector, clamped or not, since the property is
    local to the span. Summing ``degree + 1`` non-negative values whose exact total is
    one carries at most ``degree * eps``; each value comes out of a recurrence of depth
    ``degree``, contributing ``O(degree) * eps`` more, so the product bounds the total by
    ``(degree + 1) ** 2 * eps``.

    Args:
        degree (int): Polynomial degree, which sets the term count.
        dtype (npt.DTypeLike): Working precision, which sets machine epsilon.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    tol = _ERROR_SAFETY_FACTOR * (degree + 1) ** 2 * get_machine_epsilon(dtype)

    def predicate(result: object) -> str | None:
        basis = _basis_of(result)
        if basis is None:
            return f"expected a (basis, first_basis) pair, got {type(result).__name__}"
        if basis.size == 0:
            return None
        sums = np.sum(basis, axis=-1)
        worst = float(np.max(np.abs(sums - 1.0)))
        if not np.isfinite(worst) or worst > tol:
            return f"max|sum(N) - 1| = {worst:.3e} > {tol:.3e}"
        return None

    return predicate


def _first_basis_in_range(num_basis: int, degree: int, *, periodic: bool) -> Predicate:
    """Build a predicate requiring every returned ``first_basis`` index to be usable.

    ``first_basis`` is what a caller scatters the local block with, so an index that
    leaves no room for the block means the caller writes out of bounds -- silently, in a
    Numba kernel. Checking it here is cheaper than discovering it in the port.

    The admissible range differs by knot family. On a clamped or unclamped space the
    block ``first_basis .. first_basis + degree`` must fit, so the index stops at
    ``num_basis - degree - 1``. On a **periodic** space the block wraps modulo
    ``num_basis`` by construction, so any index in ``[0, num_basis - 1]`` is valid and
    the tighter bound would flag correct behavior.

    Args:
        num_basis (int): Number of basis functions in the space.
        degree (int): Polynomial degree, which sets the local block width.
        periodic (bool): Whether the space wraps.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    limit = num_basis - 1 if periodic else num_basis - degree - 1

    def predicate(result: object) -> str | None:
        if not (isinstance(result, tuple) and len(result) == _PAIR):
            return f"expected a (basis, first_basis) pair, got {type(result).__name__}"
        first = np.asarray(result[1])
        if first.size == 0:
            return None
        lowest = int(np.min(first))
        highest = int(np.max(first))
        if lowest < 0 or highest > limit:
            return f"first_basis in [{lowest}, {highest}], must be within [0, {limit}]"
        return None

    return predicate


def _within_convex_hull(
    control_points: npt.NDArray[np.floating[Any]], degree: int, dtype: npt.DTypeLike
) -> Predicate:
    """Build a predicate requiring in-domain values to lie in the control hull.

    A B-spline value is a convex combination of at most ``degree + 1`` control points, so
    it cannot leave their componentwise range. Rounding lets it out by at most
    ``3 * degree * eps * max|control point|``: a perturbed stage weight that rounds
    slightly outside ``[0, 1]`` is the only escape, and it is bounded by one epsilon per
    stage. This is the cheapest available detector of silent garbage -- a wrong index or
    a wrong span gives a value far outside the hull, not a slightly wrong one.

    Args:
        control_points (npt.NDArray[np.floating[Any]]): The control points.
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): Working precision.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    flat = np.asarray(control_points, dtype=np.float64).reshape(-1, control_points.shape[-1])
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    scale = float(np.max(np.abs(flat))) if flat.size else 1.0
    slack = _ERROR_SAFETY_FACTOR * (degree + 1) * get_machine_epsilon(dtype) * max(scale, 1.0)

    def predicate(result: object) -> str | None:
        values = np.asarray(result, dtype=np.float64)
        if values.size == 0:
            return None
        flat_values = values.reshape(-1, lo.size)
        below = float(np.max(lo - flat_values.min(axis=0)))
        above = float(np.max(flat_values.max(axis=0) - hi))
        worst = max(below, above)
        if not np.isfinite(worst) or worst > slack:
            return f"value escapes the control hull by {worst:.3e} > {slack:.3e}"
        return None

    return predicate


def _reproduces(
    reference: Bspline,
    pts: npt.NDArray[np.floating[Any]],
    tol: float,
) -> Predicate:
    """Build a predicate requiring a derived spline to reproduce a reference pointwise.

    Args:
        reference (Bspline): The spline whose values the result must reproduce.
        pts (npt.NDArray[np.floating[Any]]): Parameters at which to compare.
        tol (float): Bound on the difference, derived by :func:`_eval_tolerance`.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    expected = np.asarray(reference.evaluate(pts), dtype=np.float64)

    def predicate(result: object) -> str | None:
        if not isinstance(result, Bspline | Bezier):
            return f"expected a spline, got {type(result).__name__}"
        got = np.asarray(result.evaluate(pts.astype(result.dtype)), dtype=np.float64)
        if got.shape != expected.shape:
            return f"shape {got.shape} != reference {expected.shape}"
        worst = float(np.max(np.abs(got - expected))) if got.size else 0.0
        if not np.isfinite(worst) or worst > tol:
            return f"max|f - f_ref| = {worst:.3e} > {tol:.3e}"
        return None

    return predicate


def _has_degrees(expected: tuple[int, ...]) -> Predicate:
    """Build a predicate requiring a returned spline to carry the expected degrees.

    This is deliberately weak, and the reason is worth recording. ``Bspline.reduce_degree``
    states outright that it is **not exact** (``_bspline.py:429-430``), and that the knot
    removal following the per-segment reduction "is a forced removal with no deviation
    bound" (``:437-440``) -- so no accuracy invariant is available on an arbitrary curve.
    Its bit-exactness sentence ("adjacent segments therefore agree at every breakpoint bit
    for bit") is about the *intermediate* Bézier segments before that removal, not about
    reproducing the original's endpoints, and asserting the latter would be inventing a
    claim. The exactly-reducible case is probed separately, where the claim does bite.

    Args:
        expected (tuple[int, ...]): Degree per parametric direction.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """

    def predicate(result: object) -> str | None:
        if not isinstance(result, Bspline):
            return f"expected a Bspline, got {type(result).__name__}"
        got = tuple(result.degree)
        if got != expected:
            return f"degrees {got} != expected {expected}"
        return None

    return predicate


def _nd_partition_of_unity(degrees_per_dir: tuple[int, ...], dtype: npt.DTypeLike) -> Predicate:
    """Build a predicate for the tensor-product local partition of unity.

    ``BsplineSpace.tabulate_basis`` returns basis values of shape
    ``(num_pts, order[0], ..., order[d-1])`` -- a tensor per point, not a flat vector -- so
    the sum runs over every trailing axis. The term count is the product of the orders, and
    each value is a product of ``d`` univariate values each from a recurrence of depth
    ``degree``, giving a bound of ``(prod(order) + sum(degree)) * eps``.

    Args:
        degrees_per_dir (tuple[int, ...]): Degree in each parametric direction.
        dtype (npt.DTypeLike): Working precision.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    terms = int(np.prod([d + 1 for d in degrees_per_dir]))
    depth = int(sum(degrees_per_dir))
    tol = _ERROR_SAFETY_FACTOR * (terms + depth) * get_machine_epsilon(dtype)

    def predicate(result: object) -> str | None:
        basis = _basis_of(result)
        if basis is None:
            return f"expected a (basis, first_basis) pair, got {type(result).__name__}"
        if basis.size == 0:
            return None
        axes = tuple(range(basis.ndim - len(degrees_per_dir), basis.ndim))
        sums = np.sum(basis, axis=axes)
        worst = float(np.max(np.abs(sums - 1.0)))
        if not np.isfinite(worst) or worst > tol:
            return f"max|sum(N) - 1| = {worst:.3e} > {tol:.3e}"
        return None

    return predicate


def _nd_first_basis_in_range(
    num_basis_per_dir: tuple[int, ...], degrees_per_dir: tuple[int, ...]
) -> Predicate:
    """Build a predicate requiring every per-direction ``first_basis`` index to be usable.

    The nD two-return carries ``first_basis`` of shape ``(..., dim)``, one index per
    direction, and each must leave room for that direction's local block.

    Args:
        num_basis_per_dir (tuple[int, ...]): Basis count per direction.
        degrees_per_dir (tuple[int, ...]): Degree per direction.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    limits = tuple(n - p - 1 for n, p in zip(num_basis_per_dir, degrees_per_dir, strict=True))

    def predicate(result: object) -> str | None:
        if not (isinstance(result, tuple) and len(result) == _PAIR):
            return f"expected a (basis, first_basis) pair, got {type(result).__name__}"
        first = np.asarray(result[1]).reshape(-1, len(limits))
        if first.size == 0:
            return None
        for direction, limit in enumerate(limits):
            column = first[:, direction]
            lowest = int(np.min(column))
            highest = int(np.max(column))
            if lowest < 0 or highest > limit:
                return (
                    f"direction {direction}: first_basis in [{lowest}, {highest}], "
                    f"must be within [0, {limit}]"
                )
        return None

    return predicate


def _monotone_index_ladder(space: BsplineSpace1D) -> Predicate:
    """Build a predicate for the ``first_basis_per_interval`` contract.

    ``_bspline_space_1d.py:348-350`` states the result is non-decreasing and that its
    successive differences are the interior knot multiplicities. Both halves are exact
    integer claims, so both are asserted exactly.

    Args:
        space (BsplineSpace1D): The space whose interior multiplicities are the oracle.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    unique, mult = space.get_unique_knots_and_multiplicity(in_domain=True)
    interior = (
        np.asarray(mult, dtype=np.int64)[1:-1]
        if unique.size >= _PAIR
        else np.zeros(0, dtype=np.int64)
    )

    def predicate(result: object) -> str | None:
        first = np.asarray(result, dtype=np.int64)
        if first.size == 0:
            return None
        diffs = np.diff(first)
        if np.any(diffs < 0):
            return f"not non-decreasing: {first.tolist()}"
        if diffs.size != interior.size:
            return f"{diffs.size} successive differences for {interior.size} interior knots"
        if not np.array_equal(diffs, interior):
            return f"differences {diffs.tolist()} != interior multiplicities {interior.tolist()}"
        return None

    return predicate


def _root_quality(spline: Bspline, degree: int, dtype: npt.DTypeLike) -> Predicate:
    """Build a predicate for the three things a root list must satisfy.

    A degree-``n`` polynomial has at most ``n`` roots per interval, and a spline with
    ``k`` intervals therefore at most ``degree * k``; more than that is a defect no
    conditioning argument excuses. Each root must lie in the domain, and each must be an
    actual root: de Boor evaluation of a degree-``n`` spline with coefficients of
    magnitude ``cs`` carries about ``n * eps * cs`` of noise, so the residual is checked
    against that and nothing tighter.

    Args:
        spline (Bspline): The spline whose roots were requested.
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): Working precision.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    space = spline.space.spaces[0]
    knots = np.asarray(space.knots, dtype=np.float64)
    lo = float(knots[degree])
    hi = float(knots[knots.size - degree - 1])
    limit = degree * space.num_intervals
    scale = float(np.max(np.abs(np.asarray(spline.control_points, dtype=np.float64))))
    residual_tol = (
        _ERROR_SAFETY_FACTOR * (degree + 1) * get_machine_epsilon(dtype) * max(scale, 1.0)
    )

    def predicate(result: object) -> str | None:
        roots = np.asarray(result, dtype=np.float64).ravel()
        if roots.size > limit:
            return f"{roots.size} roots for degree {degree} on {space.num_intervals} intervals"
        if roots.size == 0:
            return None
        if float(np.min(roots)) < lo or float(np.max(roots)) > hi:
            return f"roots outside the domain [{lo:g}, {hi:g}]: {roots.tolist()}"
        values = np.asarray(spline.evaluate(roots.astype(spline.dtype)), dtype=np.float64)
        worst = float(np.max(np.abs(values)))
        if worst > residual_tol:
            return f"max|f(root)| = {worst:.3e} > {residual_tol:.3e}"
        return None

    return predicate


def _derivative_agreement(
    curve: Bspline,
    pts: npt.NDArray[np.floating[Any]],
    fixture: _Fixture,
    value_scale: float,
) -> Predicate:
    """Build a predicate comparing an analytic derivative to a central difference.

    The central difference has truncation error ``O(h ** 2)`` and rounding error
    ``O(eps / h)``; balancing them gives ``h = eps ** (1/3)`` and a floor of
    ``eps ** (2/3)`` on the *relative* accuracy. Scaling by the value magnitude over the
    parametric span converts that into the derivative's own units. Anything tighter would
    report the finite difference's own noise as a defect.

    Args:
        curve (Bspline): The spline whose derivative is under test.
        pts (npt.NDArray[np.floating[Any]]): Parameters strictly inside knot intervals,
            since a C^-1 spline has no derivative at a knot.
        fixture (_Fixture): Supplies dtype, degree and the parametric scale.
        value_scale (float): Magnitude of the control points.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    eps = get_machine_epsilon(fixture.dtype)
    span = _min_nonzero_span(fixture.space)
    step = float(eps ** (1.0 / 3.0)) * span
    tol = (
        _ERROR_SAFETY_FACTOR
        * (fixture.degree + 1)
        * eps ** (2.0 / 3.0)
        * max(value_scale, 1.0)
        / span
    )

    def predicate(result: object) -> str | None:
        if not isinstance(result, Bspline):
            return f"expected a Bspline, got {type(result).__name__}"
        if pts.size == 0:
            return None
        forward = np.asarray(curve.evaluate((pts + step).astype(curve.dtype)), dtype=np.float64)
        backward = np.asarray(curve.evaluate((pts - step).astype(curve.dtype)), dtype=np.float64)
        approx = (forward - backward) / (2.0 * step)
        exact = np.asarray(result.evaluate(pts.astype(result.dtype)), dtype=np.float64)
        if exact.shape != approx.shape:
            return f"derivative shape {exact.shape} != finite difference {approx.shape}"
        worst = float(np.max(np.abs(exact - approx))) if exact.size else 0.0
        if not np.isfinite(worst) or worst > tol:
            return f"max|f' - FD| = {worst:.3e} > {tol:.3e}"
        return None

    return predicate


def _split_agreement(curve: Bspline, fixture: _Fixture, tol: float) -> Predicate:
    """Build a predicate requiring both halves of a split to reproduce the original.

    The samples exclude both ends of each half. The split point is a knot, and a spline
    whose interior multiplicity is ``degree + 1`` is genuinely discontinuous there: the
    original and the two halves each pick a one-sided value by their own span convention,
    and neither is wrong. Comparing *at* the breakpoint would report the discontinuity as
    a defect, which is the one thing this probe must not do.

    Args:
        curve (Bspline): The spline before splitting.
        fixture (_Fixture): Supplies the domain and dtype.
        tol (float): Bound from :func:`_eval_tolerance`.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    lo, hi = _domain_of(fixture)
    mid = lo + 0.5 * (hi - lo)
    inner = np.array([0.13, 0.37, 0.61, 0.83])
    left_pts = (lo + (mid - lo) * inner).astype(fixture.dtype)
    right_pts = (mid + (hi - mid) * inner).astype(fixture.dtype)

    def predicate(result: object) -> str | None:
        if not (isinstance(result, tuple) and len(result) == _PAIR):
            return f"expected two halves, got {type(result).__name__}"
        for name, half, sample in (("left", result[0], left_pts), ("right", result[1], right_pts)):
            expected = np.asarray(curve.evaluate(sample), dtype=np.float64)
            got = np.asarray(half.evaluate(sample), dtype=np.float64)
            worst = float(np.max(np.abs(got - expected))) if got.size else 0.0
            if not np.isfinite(worst) or worst > tol:
                return f"{name} half: max|f - f_ref| = {worst:.3e} > {tol:.3e}"
        return None

    return predicate


def _bezier_agreement(curve: Bspline, fixture: _Fixture, tol: float) -> Predicate:
    """Build a predicate requiring each extracted Bézier to reproduce its own interval.

    The local parameters stay strictly inside ``(0, 1)`` for the same reason
    :func:`_split_agreement` avoids the split point: a patch's endpoint is a knot, and at
    an interior knot of multiplicity ``degree + 1`` the spline has two values, so the
    patch and the spline legitimately disagree there.

    Args:
        curve (Bspline): The spline that was decomposed.
        fixture (_Fixture): Supplies the degree and dtype.
        tol (float): Bound from :func:`_eval_tolerance`.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    knots = np.asarray(curve.space.spaces[0].knots, dtype=np.float64)
    degree = fixture.degree
    breaks = np.unique(knots[degree : knots.size - degree])
    local = np.array([0.13, 0.37, 0.61, 0.83], dtype=fixture.dtype)

    def predicate(result: object) -> str | None:
        patches = np.asarray(result, dtype=object).ravel()
        if patches.size == 0:
            return "no Bézier patches returned"
        if patches.size != breaks.size - 1:
            return f"{patches.size} patches for {breaks.size - 1} intervals"
        for index, patch in enumerate(patches):
            left = float(breaks[index])
            width = float(breaks[index + 1]) - left
            if width <= 0.0:
                continue
            global_pts = (left + width * local.astype(np.float64)).astype(fixture.dtype)
            expected = np.asarray(curve.evaluate(global_pts), dtype=np.float64)
            got = np.asarray(patch.evaluate(local), dtype=np.float64)
            worst = float(np.max(np.abs(got - expected))) if got.size else 0.0
            if not np.isfinite(worst) or worst > tol:
                return f"patch {index}: max|B - f| = {worst:.3e} > {tol:.3e}"
        return None

    return predicate


def _product_agreement(
    scalar: Bspline, pts: npt.NDArray[np.floating[Any]], fixture: _Fixture
) -> Predicate:
    """Build a predicate requiring a spline product to equal the pointwise product.

    ``_bspline_product.py``'s module docstring claims the product is *exact* in the
    product space of degree ``p + q``. The comparison's own error is each factor's
    evaluation error times the other factor's magnitude, so the bound is the evaluation
    tolerance scaled by the value magnitude once more.

    Args:
        scalar (Bspline): The scalar spline being squared.
        pts (npt.NDArray[np.floating[Any]]): Comparison parameters.
        fixture (_Fixture): Supplies degree and dtype.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """
    values = np.asarray(scalar.evaluate(pts), dtype=np.float64)
    expected = values * values
    scale = max(float(np.max(np.abs(values))) if values.size else 1.0, 1.0)
    tol = _fixture_tolerance(fixture, scale) * scale

    def predicate(result: object) -> str | None:
        if not isinstance(result, Bspline):
            return f"expected a Bspline, got {type(result).__name__}"
        got = np.asarray(result.evaluate(pts.astype(result.dtype)), dtype=np.float64)
        if got.shape != expected.shape:
            return f"shape {got.shape} != pointwise product {expected.shape}"
        worst = float(np.max(np.abs(got - expected))) if got.size else 0.0
        reference = max(float(np.max(np.abs(expected))), 1.0) if expected.size else 1.0
        if not np.isfinite(worst) or worst > tol:
            return f"max|fg - f*g| = {worst:.3e} > {tol:.3e} (relative {worst / reference:.3e})"
        return None

    return predicate


def _bitwise_control_points(expected: npt.NDArray[np.floating[Any]]) -> Predicate:
    """Build a predicate requiring a spline's control points to match bit for bit.

    Reversal is a pure permutation of the control points, so applying it twice must
    restore them exactly. No tolerance applies to a permutation.

    Args:
        expected (npt.NDArray[np.floating[Any]]): The original control points.

    Returns:
        Predicate: Check for :func:`adversarial_sweep._core.custom`.
    """

    def predicate(result: object) -> str | None:
        if not isinstance(result, Bspline):
            return f"expected a Bspline, got {type(result).__name__}"
        got = np.asarray(result.control_points)
        if got.shape != expected.shape:
            return f"shape {got.shape} != original {expected.shape}"
        if not np.array_equal(got, expected):
            worst = float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))
            return f"control points not restored bitwise, max|diff| = {worst:.3e}"
        return None

    return predicate


def _non_decreasing(result: object) -> str | None:
    """Check that a returned knot vector is non-decreasing.

    Every knot-vector factory documents a valid knot vector, and non-decreasing is the
    defining property of one.

    Args:
        result (object): The returned knot vector.

    Returns:
        str | None: ``None`` when it holds, otherwise the violation.
    """
    knots = np.asarray(result, dtype=np.float64).ravel()
    if knots.size < _PAIR:
        return None
    gaps = np.diff(knots)
    if np.any(gaps < 0.0):
        worst = int(np.argmin(gaps))
        return f"knots decrease at index {worst}: {knots[worst]:.17g} > {knots[worst + 1]:.17g}"
    return None


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def _interior_mult_of(name: str) -> int:
    """Read the interior multiplicity out of a knot-family name.

    Args:
        name (str): Family name from :func:`adversarial_sweep._axes.knot_specs`.

    Returns:
        int: The multiplicity, or ``0`` when the family does not encode one.
    """
    prefix = "interior_mult"
    return int(name[len(prefix) :]) if name.startswith(prefix) else 0


def _try_space(
    knots: npt.NDArray[np.floating[Any]], degree: int, *, periodic: bool
) -> BsplineSpace1D | None:
    """Build a space, returning ``None`` when the knot family is not a legal one.

    Fixture enumeration must not raise: an illegal knot vector is covered as a
    *construction* case elsewhere, and here it simply yields no fixture.

    Args:
        knots (npt.NDArray[np.floating[Any]]): The knot vector.
        degree (int): Polynomial degree.
        periodic (bool): Whether to build a periodic space.

    Returns:
        BsplineSpace1D | None: The space, or ``None`` if it could not be built.
    """
    try:
        return BsplineSpace1D(knots, degree, periodic=periodic)
    except (ValueError, TypeError):
        return None


def _mult_knots(
    degree: int,
    mult: int,
    domain: tuple[float, float],
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.floating[Any]]:
    """Build a clamped knot vector with one interior knot at a chosen multiplicity.

    Two interior breakpoints are used, one carrying the swept multiplicity and one
    simple, so that operations special-casing the first or the last interval still see a
    neighbour.

    Args:
        degree (int): Polynomial degree.
        mult (int): Multiplicity of the first interior knot, in ``[1, degree + 1]``.
        domain (tuple[float, float]): ``(lo, hi)`` parametric interval.
        dtype (np.dtype[np.float32 | np.float64]): Knot precision.

    Returns:
        npt.NDArray[np.floating[Any]]: The knot vector.
    """
    lo, hi = domain
    span = hi - lo
    return np.concatenate(
        [
            np.full(degree + 1, lo, dtype=dtype),
            np.full(mult, lo + 0.5 * span, dtype=dtype),
            np.array([lo + 0.75 * span], dtype=dtype),
            np.full(degree + 1, hi, dtype=dtype),
        ]
    )


def _fixtures(profile: Profile) -> Iterator[_Fixture]:
    """Yield the 1D space fixtures every operation probe is run against.

    The crossing is factorized rather than exhaustive, on the sweep's own principle of
    corners over middles: the full multiplicity ladder is crossed with degree and dtype
    on the unit domain, and the ``degree + 1`` corner -- the discontinuous case -- is
    then repeated on every other domain, together with the uniform and periodic families
    that give a continuity baseline to compare it against.

    Args:
        profile (Profile): Sweep width.

    Yields:
        _Fixture: One usable space plus its axis values.
    """
    unit = domains(Profile.SMOKE)[0]
    for degree in degrees(profile):
        for dtype in dtypes(profile):
            for mult in interior_multiplicities(degree, profile):
                space = _try_space(_mult_knots(degree, mult, unit, dtype), degree, periodic=False)
                if space is not None:
                    yield _Fixture(
                        f"d{degree}_m{mult}_{dtype.name}",
                        space,
                        degree,
                        mult,
                        unit,
                        dtype,
                        {"degree": degree, "mult": mult, "dtype": dtype, "domain": "unit"},
                    )
            if profile is not Profile.FULL:
                continue
            wanted = ("open_uniform", "periodic_uniform", f"interior_mult{degree + 1}")
            for domain in domains(profile)[1:]:
                for spec in knot_specs(degree, domain, dtype, profile):
                    if spec.name not in wanted:
                        continue
                    space = _try_space(spec.knots, degree, periodic=spec.periodic)
                    if space is None:
                        continue
                    yield _Fixture(
                        f"d{degree}_{spec.name}_{dtype.name}_[{domain[0]:g},{domain[1]:g}]",
                        space,
                        degree,
                        _interior_mult_of(spec.name),
                        domain,
                        dtype,
                        {
                            "degree": degree,
                            "knots": spec.name,
                            "dtype": dtype,
                            "domain": list(domain),
                        },
                    )


def _domain_of(fixture: _Fixture) -> tuple[float, float]:
    """Read a fixture's actual parametric domain from its knot vector.

    The nominal ``(lo, hi)`` of the axis is not the domain for an unclamped or periodic
    knot family, where the domain is the inner ``[knots[degree], knots[-degree-1]]``.
    Evaluating outside it is a different probe, so operations must use this.

    Args:
        fixture (_Fixture): The fixture.

    Returns:
        tuple[float, float]: The domain endpoints.
    """
    knots = np.asarray(fixture.space.knots, dtype=np.float64)
    return float(knots[fixture.degree]), float(knots[knots.size - fixture.degree - 1])


def _sample_points(
    fixture: _Fixture, *, avoid_knots: bool = False
) -> npt.NDArray[np.floating[Any]]:
    """Build in-domain sample parameters for a fixture.

    Args:
        fixture (_Fixture): The fixture whose domain to sample.
        avoid_knots (bool): When ``True``, keep the samples strictly inside the knot
            intervals. A C^-1 spline has no derivative *at* a knot, so a
            finite-difference comparison must not straddle one.

    Returns:
        npt.NDArray[np.floating[Any]]: Sample parameters.
    """
    lo, hi = _domain_of(fixture)
    if not avoid_knots:
        fractions = np.array([0.0, 0.13, 0.37, 0.5, 0.61, 0.87, 1.0])
        return (lo + (hi - lo) * fractions).astype(fixture.dtype)
    knots = np.asarray(fixture.space.knots, dtype=np.float64)
    breaks = np.unique(knots[(knots >= lo) & (knots <= hi)])
    mids = 0.5 * (breaks[:-1] + breaks[1:]) if breaks.size >= _PAIR else np.array([0.5 * (lo + hi)])
    return mids.astype(fixture.dtype)


def _curve(fixture: _Fixture, dim: int, *, kind: str = "random", offset: int = 0) -> Bspline:
    """Build a spline on a fixture's space with control points from a chosen family.

    Args:
        fixture (_Fixture): The space to build on.
        dim (int): Number of physical components.
        kind (str): Control-point family name from
            :func:`adversarial_sweep._axes.coeff_specs`. Defaults to ``"random"``.
        offset (int): Random-stream offset. Defaults to 0.

    Returns:
        Bspline: The spline.
    """
    shape = (fixture.space.num_basis, dim)
    specs = {
        spec.name: spec for spec in coeff_specs(shape, fixture.dtype, Profile.FULL, offset=offset)
    }
    values = specs[kind].values if kind in specs else specs["random"].values
    return Bspline(BsplineSpace([fixture.space]), values)


def _try_elevate(curve: Bspline) -> Bspline | None:
    """Elevate a curve's degree by one, returning ``None`` when the call refuses.

    A generator body must not raise -- an exception here aborts the whole sweep instead of
    being classified -- and degree elevation is currently *unable* to handle a
    discontinuous spline. The refusal is itself covered as a ``must_succeed`` case, so
    swallowing it here loses nothing.

    Args:
        curve (Bspline): The curve to elevate.

    Returns:
        Bspline | None: The elevated curve, or ``None``.
    """
    try:
        return curve.elevate_degree(1)
    except (ValueError, TypeError, IndexError):
        return None


def _sign_alternating_curve(fixture: _Fixture) -> Bspline:
    """Build a scalar spline whose control points alternate in sign.

    Alternating signs guarantee sign changes for the root finder to find, and spread the
    roots roughly one per interval rather than clustering them -- a *clustered* root set
    is already known to defeat the deduplication radius, so a count violation there would
    only rediscover that.

    Args:
        fixture (_Fixture): The space to build on.

    Returns:
        Bspline: The scalar spline.
    """
    n = fixture.space.num_basis
    values = np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(fixture.dtype).reshape(n, 1)
    return Bspline(BsplineSpace([fixture.space]), values)


def _locate_round_trip(fixture: _Fixture) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Invert a monotone scalar spline at its own sampled values.

    Args:
        fixture (_Fixture): The space to build the monotone spline on.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: Whatever
            :meth:`pantr.bspline.Bspline.locate` returns for the sampled images.
    """
    n = fixture.space.num_basis
    values = np.linspace(0.0, 1.0, n, dtype=fixture.dtype).reshape(n, 1)
    curve = Bspline(BsplineSpace([fixture.space]), values)
    images = np.asarray(curve.evaluate(_sample_points(fixture, avoid_knots=True)))
    return curve.locate(images)


# ---------------------------------------------------------------------------
# Case groups
# ---------------------------------------------------------------------------


def _construction_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`pantr.bspline.BsplineSpace1D` construction cases.

    This is the one exhaustive crossing in the module -- every degree, dtype, domain and
    knot family -- because a constructor call is cheap and because construction is where
    a malformed knot vector must be *rejected*, which the verdict rule can only judge
    against the docstring's own ``Raises:`` list.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One construction attempt.
    """
    for degree in degrees(profile):
        for dtype in dtypes(profile):
            for domain in domains(profile):
                tag = f"d{degree}_{dtype.name}_[{domain[0]:g},{domain[1]:g}]"
                for spec in knot_specs(degree, domain, dtype, profile):
                    params = {
                        "degree": degree,
                        "knots": spec.name,
                        "periodic": spec.periodic,
                        "dtype": dtype,
                        "domain": list(domain),
                    }
                    yield Case(
                        GROUP,
                        f"space1d_{spec.name}_{tag}",
                        BsplineSpace1D,
                        lambda spec=spec, degree=degree: BsplineSpace1D(
                            spec.knots, degree, periodic=spec.periodic
                        ),
                        params,
                        arrays={"knots": np.asarray(spec.knots)},
                    )
                    if profile is not Profile.FULL:
                        continue
                    # snap_knots=False is the documented escape hatch from the absolute
                    # snapping tolerance; the space it builds must still be usable.
                    yield Case(
                        GROUP,
                        f"space1d_nosnap_{spec.name}_{tag}",
                        BsplineSpace1D,
                        lambda spec=spec, degree=degree: BsplineSpace1D(
                            spec.knots, degree, periodic=spec.periodic, snap_knots=False
                        ),
                        {**params, "snap_knots": False},
                    )


def _tabulation_cases(profile: Profile) -> Iterator[Case]:
    """Yield basis-tabulation cases over every fixture and point family.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One tabulation call.
    """
    in_domain_families = ("interior", "left_endpoint", "right_endpoint", "at_knots", "single")
    for fixture in _fixtures(profile):
        space = fixture.space
        knots = np.asarray(space.knots, dtype=np.float64)
        interior = knots[fixture.degree + 1 : knots.size - fixture.degree - 1]
        pou = custom(
            "local-partition-of-unity", _local_partition_of_unity(fixture.degree, fixture.dtype)
        )
        index = custom(
            "first-basis-in-range",
            _first_basis_in_range(space.num_basis, fixture.degree, periodic=space.periodic),
        )
        for pts in point_specs(_domain_of(fixture), fixture.dtype, profile, breakpoints=interior):
            in_domain = pts.name in in_domain_families
            yield Case(
                GROUP,
                f"tabulate_basis_{fixture.label}_{pts.name}",
                BsplineSpace1D.tabulate_basis,
                lambda space=space, pts=pts: space.tabulate_basis(pts.pts),
                {**fixture.params, "pts": pts.name},
                invariants=(pou, index) if in_domain else (index,),
                must_succeed=in_domain,
                finite_inputs=pts.finite,
                arrays={"knots": np.asarray(space.knots), "pts": np.asarray(pts.pts)},
            )
            if profile is not Profile.FULL:
                continue
            n_deriv = min(fixture.degree + 1, 3)
            yield Case(
                GROUP,
                f"tabulate_derivs_{fixture.label}_{pts.name}",
                BsplineSpace1D.tabulate_basis_derivatives,
                lambda space=space, pts=pts, n_deriv=n_deriv: space.tabulate_basis_derivatives(
                    pts.pts, n_deriv
                ),
                {**fixture.params, "pts": pts.name, "n_deriv": n_deriv},
                invariants=(index,),
                must_succeed=in_domain,
                finite_inputs=pts.finite,
            )
            # validate=False is the documented escape hatch: out-of-domain points are
            # then undefined behavior by contract, which is exactly what the bounds check
            # is here to catch.
            yield Case(
                GROUP,
                f"tabulate_novalidate_{fixture.label}_{pts.name}",
                BsplineSpace1D.tabulate_basis,
                lambda space=space, pts=pts: space.tabulate_basis(pts.pts, validate=False),
                {**fixture.params, "pts": pts.name, "validate": False},
                invariants=(index,),
                finite_inputs=pts.finite,
            )


def _space_operation_cases(profile: Profile) -> Iterator[Case]:
    """Yield cases for the operations :class:`BsplineSpace1D` offers on itself.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One space-level operation.
    """
    for fixture in _fixtures(profile):
        space = fixture.space
        degree = fixture.degree
        legal = not space.periodic
        lo, hi = _domain_of(fixture)
        yield Case(
            GROUP,
            f"first_basis_per_interval_{fixture.label}",
            BsplineSpace1D.first_basis_per_interval,
            space.first_basis_per_interval,
            dict(fixture.params),
            invariants=(custom("index-ladder", _monotone_index_ladder(space)),),
            must_succeed=legal,
        )
        yield Case(
            GROUP,
            f"cardinal_intervals_{fixture.label}",
            BsplineSpace1D.get_cardinal_intervals,
            space.get_cardinal_intervals,
            dict(fixture.params),
            invariants=(expected_shape((space.num_intervals,)),),
            must_succeed=legal,
        )
        yield Case(
            GROUP,
            f"greville_{fixture.label}",
            get_greville_abscissae,
            lambda space=space: get_greville_abscissae(space),
            dict(fixture.params),
            must_succeed=legal,
        )
        for regularity in sorted({-1, 0, degree - 1}):
            if not -1 <= regularity <= degree - 1:
                continue
            yield Case(
                GROUP,
                f"subdivide_space_{fixture.label}_reg{regularity}",
                BsplineSpace1D.subdivide,
                lambda space=space, regularity=regularity: space.subdivide(
                    3, regularity=regularity
                ),
                {**fixture.params, "regularity": regularity},
                must_succeed=legal,
            )
        if profile is not Profile.FULL:
            continue
        yield Case(
            GROUP,
            f"bezier_extraction_{fixture.label}",
            BsplineSpace1D.tabulate_Bezier_extraction_operators,
            space.tabulate_Bezier_extraction_operators,
            dict(fixture.params),
            invariants=(expected_shape((space.num_intervals, degree + 1, degree + 1)),),
            must_succeed=legal,
        )
        yield Case(
            GROUP,
            f"cardinal_extraction_{fixture.label}",
            BsplineSpace1D.tabulate_cardinal_extraction_operators,
            space.tabulate_cardinal_extraction_operators,
            dict(fixture.params),
            must_succeed=legal,
        )
        yield Case(
            GROUP,
            f"lagrange_extraction_{fixture.label}",
            BsplineSpace1D.tabulate_Lagrange_extraction_operators,
            space.tabulate_Lagrange_extraction_operators,
            dict(fixture.params),
            must_succeed=legal,
        )
        # Insert at a coordinate that is not a midpoint of the existing knots, so
        # snapping cannot hide a mis-merge.
        inserted = np.array([lo + (hi - lo) * 0.3125], dtype=fixture.dtype)
        yield Case(
            GROUP,
            f"insert_knots_space_{fixture.label}",
            BsplineSpace1D.insert_knots,
            lambda space=space, inserted=inserted: space.insert_knots(inserted),
            dict(fixture.params),
        )
        yield Case(
            GROUP,
            f"restrict_space_{fixture.label}",
            BsplineSpace1D.restrict,
            lambda space=space: space.restrict(0, max(1, space.num_intervals - 1)),
            dict(fixture.params),
        )
        yield Case(
            GROUP,
            f"has_open_knots_{fixture.label}",
            BsplineSpace1D.has_open_knots,
            space.has_open_knots,
            dict(fixture.params),
        )


def _curve_operation_cases(profile: Profile) -> Iterator[Case]:
    """Yield the operation matrix every fixture's spline is pushed through.

    This is the core of the group: one spline per fixture and control-point family, then
    every operation that takes a spline, each with the strongest invariant its own
    contract supports.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One spline operation.
    """
    kinds = ("random", "identical") if profile is Profile.FULL else ("random",)
    for fixture in _fixtures(profile):
        for kind in kinds:
            yield from _one_curve_cases(fixture, kind, profile)


def _one_curve_cases(fixture: _Fixture, kind: str, profile: Profile) -> Iterator[Case]:
    """Yield every operation case for one fixture and one control-point family.

    Args:
        fixture (_Fixture): The space to build the spline on.
        kind (str): Control-point family name.
        profile (Profile): Sweep width.

    Yields:
        Case: One spline operation.
    """
    degree = fixture.degree
    dtype = fixture.dtype
    # Every operation below is legal on a clamped space by construction, so any exception
    # is a finding. A periodic space is different: most of these document a ValueError for
    # a periodic direction, and that rejection is correct.
    legal = not fixture.space.periodic
    lo, hi = _domain_of(fixture)
    mid = lo + 0.5 * (hi - lo)
    curve = _curve(fixture, 2, kind=kind)
    scalar = _curve(fixture, 1, kind=kind, offset=3)
    tag = f"{fixture.label}_{kind}"
    pts = _sample_points(fixture)
    smooth_pts = _sample_points(fixture, avoid_knots=True)
    value_scale = float(np.max(np.abs(np.asarray(curve.control_points, dtype=np.float64))))
    tol = _fixture_tolerance(fixture, value_scale)
    inserted = np.array([lo + (hi - lo) * 0.3125], dtype=dtype)
    in_domain_families = ("interior", "left_endpoint", "right_endpoint", "single")

    for family in point_specs((lo, hi), dtype, profile):
        yield Case(
            GROUP,
            f"evaluate_{tag}_{family.name}",
            Bspline.evaluate,
            lambda curve=curve, family=family: curve.evaluate(family.pts),
            {**fixture.params, "cp": kind, "pts": family.name},
            invariants=(
                (
                    custom(
                        "in-control-hull",
                        _within_convex_hull(curve.control_points, degree, dtype),
                    ),
                )
                if family.name in in_domain_families
                else ()
            ),
            must_succeed=legal and family.name in in_domain_families,
            finite_inputs=family.finite,
            arrays={
                "control_points": np.asarray(curve.control_points),
                "pts": np.asarray(family.pts),
            },
        )

    orders = min(degree, 2)
    yield Case(
        GROUP,
        f"evaluate_derivatives_{tag}",
        Bspline.evaluate_derivatives,
        lambda curve=curve, pts=pts, orders=orders: curve.evaluate_derivatives(pts, orders),
        {**fixture.params, "cp": kind, "orders": orders},
        must_succeed=legal,
    )

    if degree >= 1:
        yield Case(
            GROUP,
            f"derivative_{tag}",
            Bspline.derivative,
            lambda curve=curve: curve.derivative(0),
            {**fixture.params, "cp": kind},
            invariants=(
                custom(
                    "derivative-matches-central-difference",
                    _derivative_agreement(curve, smooth_pts, fixture, value_scale),
                ),
            ),
            must_succeed=legal,
        )
        yield Case(
            GROUP,
            f"reduce_degree_{tag}",
            Bspline.reduce_degree,
            lambda curve=curve: curve.reduce_degree(1),
            {**fixture.params, "cp": kind, "decrement": 1},
            invariants=(custom("reduced-degree", _has_degrees((degree - 1,))),),
            must_succeed=legal,
        )
        lifted = _try_elevate(_curve(fixture, 2, kind=kind, offset=7))
        if degree <= _GRAM_CONDITIONED_DEGREE and lifted is not None:
            # An exactly-reducible spline is the one case where reduction *must* be
            # accurate: the L2-optimal lower-degree approximation of a curve that already
            # lies in the lower-degree space is the curve itself. The attainable accuracy
            # is set by the Bernstein Gram matrix, whose condition number grows like
            # 4 ** degree, which is also why the case stops at a modest degree: past it
            # the honest bound exceeds the values being compared and the check is vacuous.
            recovery_tol = tol * 4.0**degree
            yield Case(
                GROUP,
                f"reduce_degree_exact_recovery_{tag}",
                Bspline.reduce_degree,
                lambda lifted=lifted: lifted.reduce_degree(1),
                {**fixture.params, "cp": kind, "kind": "exactly-reducible"},
                invariants=(
                    custom(
                        "recovers-the-reducible-curve",
                        _reproduces(lifted, pts, recovery_tol),
                    ),
                ),
                must_succeed=legal,
            )

    if degree + 1 <= _BINCOEFF_SAFE_DEGREE:
        yield Case(
            GROUP,
            f"elevate_degree_{tag}",
            Bspline.elevate_degree,
            lambda curve=curve: curve.elevate_degree(1),
            {**fixture.params, "cp": kind, "increment": 1},
            invariants=(custom("elevation-is-exact", _reproduces(curve, pts, tol)),),
            must_succeed=legal,
        )

    yield Case(
        GROUP,
        f"insert_knots_{tag}",
        Bspline.insert_knots,
        lambda curve=curve, inserted=inserted: curve.insert_knots(inserted),
        {**fixture.params, "cp": kind},
        invariants=(custom("insertion-is-exact", _reproduces(curve, pts, tol)),),
        must_succeed=legal,
    )

    yield Case(
        GROUP,
        f"subdivide_{tag}",
        Bspline.subdivide,
        lambda curve=curve: curve.subdivide(2),
        {**fixture.params, "cp": kind},
        invariants=(custom("subdivision-is-exact", _reproduces(curve, pts, tol)),),
        must_succeed=legal,
    )

    yield Case(
        GROUP,
        f"split_{tag}",
        Bspline.split,
        lambda curve=curve, mid=mid: curve.split(0, mid),
        {**fixture.params, "cp": kind},
        invariants=(custom("halves-reproduce-original", _split_agreement(curve, fixture, tol)),),
        must_succeed=legal,
    )

    yield Case(
        GROUP,
        f"to_beziers_{tag}",
        Bspline.to_beziers,
        curve.to_beziers,
        {**fixture.params, "cp": kind},
        invariants=(custom("beziers-reproduce-original", _bezier_agreement(curve, fixture, tol)),),
        must_succeed=legal,
    )

    yield Case(
        GROUP,
        f"reverse_roundtrip_{tag}",
        Bspline.reverse,
        lambda curve=curve: curve.reverse(0).reverse(0),
        {**fixture.params, "cp": kind},
        invariants=(
            custom(
                "reverse-twice-is-identity",
                _bitwise_control_points(np.asarray(curve.control_points)),
            ),
        ),
        must_succeed=legal,
    )

    if profile is not Profile.FULL:
        return

    yield Case(
        GROUP,
        f"multiply_{tag}",
        Bspline.multiply,
        lambda scalar=scalar: scalar.multiply(scalar),
        {**fixture.params, "cp": kind},
        invariants=(custom("product-is-pointwise", _product_agreement(scalar, pts, fixture)),),
        must_succeed=legal,
    )

    window = (lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo))
    yield Case(
        GROUP,
        f"restrict_{tag}",
        Bspline.restrict,
        lambda curve=curve, window=window: curve.restrict(window),
        {**fixture.params, "cp": kind},
        must_succeed=legal,
    )

    yield Case(
        GROUP,
        f"slice_{tag}",
        Bspline.slice,
        lambda curve=curve, mid=mid: curve.slice(0, mid),
        {**fixture.params, "cp": kind},
        must_succeed=legal,
    )

    yield Case(
        GROUP,
        f"boundary_{tag}",
        Bspline.boundary,
        lambda curve=curve: curve.boundary(0, 1),
        {**fixture.params, "cp": kind},
        must_succeed=legal,
    )

    # No must_succeed here: `to_open_bspline` documents a ValueError when the spline is
    # already open in every direction, which is the case for every clamped fixture.
    yield Case(
        GROUP,
        f"to_open_{tag}",
        Bspline.to_open_bspline,
        lambda curve=curve: curve.to_open_bspline(),
        {**fixture.params, "cp": kind},
        invariants=(custom("open-form-is-exact", _reproduces(curve, pts, tol)),),
    )

    yield Case(
        GROUP,
        f"remove_inserted_knot_{tag}",
        Bspline.remove_knots,
        lambda curve=curve, inserted=inserted: curve.insert_knots(inserted).remove_knots(inserted),
        {**fixture.params, "cp": kind},
        invariants=(custom("removal-restores-original", _reproduces(curve, pts, tol)),),
        must_succeed=legal,
    )

    if degree >= 1:
        signed = _sign_alternating_curve(fixture)
        yield Case(
            GROUP,
            f"find_roots_{tag}",
            find_roots,
            lambda signed=signed: find_roots(signed),
            {**fixture.params, "cp": "sign-alternating"},
            invariants=(custom("root-quality", _root_quality(signed, degree, dtype)),),
            must_succeed=legal,
            arrays={"control_points": np.asarray(signed.control_points)},
        )
        yield Case(
            GROUP,
            f"locate_{tag}",
            Bspline.locate,
            lambda fixture=fixture: _locate_round_trip(fixture),
            {**fixture.params, "cp": "monotone"},
            must_succeed=legal,
        )


def _try_nd_space(
    dim: int, degree: int, dtype: np.dtype[np.float32 | np.float64]
) -> BsplineSpace | None:
    """Build a small nD space, returning ``None`` if the combination is rejected.

    Args:
        dim (int): Parametric dimension.
        degree (int): Degree in every direction.
        dtype (np.dtype[np.float32 | np.float64]): Knot precision.

    Returns:
        BsplineSpace | None: The space, or ``None``.
    """
    try:
        return create_uniform_space([degree] * dim, [2] * dim, dtype=dtype)
    except (ValueError, TypeError):
        return None


def _nd_cases(profile: Profile) -> Iterator[Case]:
    """Yield multi-dimensional space and spline cases.

    Dimension 4 is included deliberately: the space and grid layers enforce only
    ``dim >= 1``, so ``n ** dim`` growth is reachable, and the repo has four different
    answers to "what is the maximum dimension?" in four modules.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One multi-dimensional case.
    """
    generator = rng(21)
    if profile is Profile.FULL:
        # The nD point convention is (n_pts, dim), which on a *1D* space means (n, 1).
        # That shape is already logged as leaking a raw Numba TypingError out of
        # `_evaluate_Bspline_1D`, so it gets exactly one labelled case here rather than
        # firing on every 1-D crossing: the port needs the case, the findings list does
        # not need it eleven times.
        space_1d = create_uniform_space([2], [3])
        curve_1d = Bspline(
            space_1d, np.asarray(generator.uniform(-1.0, 1.0, (space_1d.num_total_basis, 1)))
        )
        column_pts = np.asarray(generator.uniform(0.1, 0.9, (4, 1)))
        yield Case(
            GROUP,
            "evaluate_1d_with_column_points",
            Bspline.evaluate,
            lambda curve_1d=curve_1d, column_pts=column_pts: curve_1d.evaluate(column_pts),
            {"dim": 1, "degree": 2, "pts_shape": "(n, 1)"},
            arrays={"pts": column_pts},
        )
        # The same 1D/nD asymmetry in `restrict`: a 1D spline takes the bare
        # `(lower, upper)` pair, the nD convention is one pair per direction. Writing the
        # nD form on a 1D spline gives a bare `IndexError: list index out of range` from
        # `_validate_restrict_bounds` instead of the documented `ValueError`. One case, for
        # the same reason as above.
        yield Case(
            GROUP,
            "restrict_1d_with_nested_bounds",
            Bspline.restrict,
            lambda curve_1d=curve_1d: curve_1d.restrict([(0.25, 0.75)]),
            {"dim": 1, "degree": 2, "bounds_shape": "nD form on a 1D spline"},
        )
    for dim in dims(profile, max_dim=4)[1:]:
        for degree in (0, 1, 3) if profile is Profile.FULL else (1,):
            for dtype in dtypes(profile):
                tag = f"dim{dim}_d{degree}_{dtype.name}"
                params: dict[str, Any] = {"dim": dim, "degree": degree, "dtype": dtype}
                yield Case(
                    GROUP,
                    f"create_uniform_space_{tag}",
                    create_uniform_space,
                    lambda dim=dim, degree=degree, dtype=dtype: create_uniform_space(
                        [degree] * dim, [2] * dim, dtype=dtype
                    ),
                    params,
                )
                space = _try_nd_space(dim, degree, dtype)
                if space is None:
                    continue
                pts = np.asarray(generator.uniform(0.0, 1.0, (5, dim)), dtype=dtype)
                yield Case(
                    GROUP,
                    f"nd_tabulate_basis_{tag}",
                    BsplineSpace.tabulate_basis,
                    lambda space=space, pts=pts: space.tabulate_basis(pts),
                    params,
                    invariants=(
                        custom(
                            "local-partition-of-unity",
                            _nd_partition_of_unity(tuple(space.degrees), dtype),
                        ),
                        custom(
                            "first-basis-in-range",
                            _nd_first_basis_in_range(tuple(space.num_basis), tuple(space.degrees)),
                        ),
                    ),
                )
                control = np.asarray(
                    generator.uniform(-1.0, 1.0, (space.num_total_basis, dim)), dtype=dtype
                )
                curve = Bspline(space, control)
                yield Case(
                    GROUP,
                    f"nd_evaluate_{tag}",
                    Bspline.evaluate,
                    lambda curve=curve, pts=pts: curve.evaluate(pts),
                    params,
                    invariants=(
                        custom(
                            "in-control-hull",
                            _within_convex_hull(curve.control_points, degree, dtype),
                        ),
                    ),
                    arrays={"control_points": control, "pts": pts},
                )
                if profile is not Profile.FULL:
                    continue
                yield Case(
                    GROUP,
                    f"nd_cell_supports_{tag}",
                    BsplineSpace.cell_supports,
                    lambda space=space: space.cell_supports(np.arange(space.num_total_intervals)),
                    params,
                )
                yield Case(
                    GROUP,
                    f"nd_boundary_dofs_{tag}",
                    BsplineSpace.boundary_dofs,
                    lambda space=space, dim=dim: space.boundary_dofs(dim - 1, 1),
                    params,
                )
                orders = [min(degree, 1)] * dim
                yield Case(
                    GROUP,
                    f"nd_evaluate_derivatives_{tag}",
                    Bspline.evaluate_derivatives,
                    lambda curve=curve, pts=pts, orders=orders: (
                        curve.evaluate_derivatives(pts, orders)
                    ),
                    params,
                )
                if dim >= _PAIR:
                    yield Case(
                        GROUP,
                        f"nd_permute_{tag}",
                        Bspline.permute_directions,
                        lambda curve=curve, dim=dim: curve.permute_directions([*range(1, dim), 0]),
                        params,
                    )


def _try_extraction(fixture: _Fixture, target: str) -> SpanwiseElementExtraction | None:
    """Build an extraction, returning ``None`` when the combination is rejected.

    Args:
        fixture (_Fixture): The space to build on.
        target (str): Extraction target name.

    Returns:
        SpanwiseElementExtraction | None: The extraction, or ``None``.
    """
    try:
        return SpanwiseElementExtraction(BsplineSpace([fixture.space]), target)
    except (ValueError, TypeError, NotImplementedError, IndexError):
        return None


def _extraction_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`SpanwiseElementExtraction` cases across targets and fixtures.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One extraction case.
    """
    generator = rng(31)
    for fixture in _fixtures(profile):
        if fixture.space.periodic:
            continue
        for target in ("bezier", "cardinal", "lagrange"):
            tag = f"{fixture.label}_{target}"
            params = {**fixture.params, "target": target}
            yield Case(
                GROUP,
                f"extraction_build_{tag}",
                SpanwiseElementExtraction,
                lambda fixture=fixture, target=target: SpanwiseElementExtraction(
                    BsplineSpace([fixture.space]), target
                ),
                params,
            )
            extraction = _try_extraction(fixture, target)
            if extraction is None:
                continue
            n_in = int(np.prod(extraction.input_shape_per_dir))
            n_out = int(np.prod(extraction.output_shape_per_dir))
            operand = np.asarray(generator.uniform(-1.0, 1.0, n_in), dtype=fixture.dtype)
            yield Case(
                GROUP,
                f"extraction_apply_{tag}",
                SpanwiseElementExtraction.apply,
                lambda extraction=extraction, operand=operand: extraction.apply(operand, 0),
                params,
                invariants=(expected_shape((n_out,)),),
                arrays={"operand": operand},
            )
            if profile is not Profile.FULL:
                continue
            yield Case(
                GROUP,
                f"extraction_tabulate_{tag}",
                SpanwiseElementExtraction.tabulate,
                extraction.tabulate,
                params,
            )
            yield Case(
                GROUP,
                f"extraction_operator_last_{tag}",
                SpanwiseElementExtraction.operator,
                lambda extraction=extraction: extraction.operator(
                    extraction.num_total_intervals - 1
                ),
                params,
            )
            yield Case(
                GROUP,
                f"extraction_apply_many_{tag}",
                SpanwiseElementExtraction.apply_many,
                lambda extraction=extraction, operand=operand: extraction.apply_many(
                    np.broadcast_to(operand, (extraction.num_total_intervals, operand.size)).copy(),
                    np.arange(extraction.num_total_intervals),
                ),
                params,
            )


def _factory_cases(profile: Profile) -> Iterator[Case]:
    """Yield knot-vector factory, interpolation and quasi-interpolation cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One factory or fitting case.
    """
    swept_domains = domains(profile) if profile is Profile.FULL else (domains(profile)[0],)
    for degree in degrees(profile):
        for dtype in dtypes(profile):
            for domain in swept_domains:
                tag = f"d{degree}_{dtype.name}_[{domain[0]:g},{domain[1]:g}]"
                params: dict[str, Any] = {
                    "degree": degree,
                    "dtype": dtype,
                    "domain": list(domain),
                }
                for continuity in sorted({-1, 0, degree - 1}):
                    if not -1 <= continuity <= degree - 1:
                        continue
                    yield Case(
                        GROUP,
                        f"uniform_open_knots_{tag}_c{continuity}",
                        create_uniform_open_knots,
                        lambda degree=degree, continuity=continuity, domain=domain, dtype=dtype: (
                            create_uniform_open_knots(
                                3, degree, continuity, domain=domain, dtype=dtype
                            )
                        ),
                        {**params, "continuity": continuity},
                        invariants=(custom("knots-non-decreasing", _non_decreasing),),
                    )
                yield Case(
                    GROUP,
                    f"uniform_periodic_knots_{tag}",
                    create_uniform_periodic_knots,
                    lambda degree=degree, domain=domain, dtype=dtype: (
                        create_uniform_periodic_knots(3, degree, domain=domain, dtype=dtype)
                    ),
                    params,
                    invariants=(custom("knots-non-decreasing", _non_decreasing),),
                )
                yield Case(
                    GROUP,
                    f"cardinal_knots_{tag}",
                    create_cardinal_knots,
                    lambda degree=degree, dtype=dtype: create_cardinal_knots(
                        3, degree, dtype=dtype
                    ),
                    params,
                    invariants=(custom("knots-non-decreasing", _non_decreasing),),
                )
                # Zero and one interval: the count corners the factories must either
                # accept cleanly or reject.
                for n_intervals in (0, 1):
                    yield Case(
                        GROUP,
                        f"uniform_open_knots_{tag}_n{n_intervals}",
                        create_uniform_open_knots,
                        lambda degree=degree, n=n_intervals, domain=domain, dtype=dtype: (
                            create_uniform_open_knots(n, degree, domain=domain, dtype=dtype)
                        ),
                        {**params, "n_intervals": n_intervals},
                    )

    if profile is not Profile.FULL:
        return

    for degree in (1, 2, 3):
        for dtype in dtypes(profile):
            space = create_uniform_space([degree], [4], dtype=dtype)
            params = {"degree": degree, "dtype": dtype}
            yield Case(
                GROUP,
                f"interpolate_bspline_d{degree}_{dtype.name}",
                interpolate_bspline,
                lambda space=space: interpolate_bspline(np.sin, space),
                params,
            )
            yield Case(
                GROUP,
                f"quasi_interpolate_bspline_d{degree}_{dtype.name}",
                quasi_interpolate_bspline,
                lambda space=space: quasi_interpolate_bspline(np.cos, space),
                params,
            )


def _try_thb(root: BsplineSpace, truncate: bool) -> THBSplineSpace | None:
    """Build a THB space, returning ``None`` when the combination is rejected.

    Args:
        root (BsplineSpace): The level-0 space.
        truncate (bool): Whether to truncate.

    Returns:
        THBSplineSpace | None: The space, or ``None``.
    """
    try:
        return create_thb_space(root, 2, truncate=truncate)
    except (ValueError, TypeError, NotImplementedError):
        return None


def _thb_cell_points(
    space: THBSplineSpace, cid: int, generator: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Sample points strictly inside one cell of a THB space's grid.

    ``THBSplineSpace.tabulate_basis`` rejects points outside the named cell, so the
    samples must come from that cell's own bounds.

    Args:
        space (THBSplineSpace): The THB space.
        cid (int): Flat cell id.
        generator (np.random.Generator): Seeded generator.

    Returns:
        npt.NDArray[np.float64]: Points of shape ``(3, dim)``.
    """
    lo, hi = space.grid.cell_bounds(cid)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    unit = np.asarray(generator.uniform(0.2, 0.8, (3, lo.size)))
    return lo + unit * (hi - lo)


def _thb_cases(profile: Profile) -> Iterator[Case]:
    """Yield THB-spline space cases, truncated and not.

    Only the *truncated* basis is a partition of unity: the non-truncated hierarchical
    basis genuinely is not (``_thb_spline_space.py:196-202``), so asserting it there
    would manufacture findings. The refinement depth is kept shallow on purpose -- the
    ``factor ** level`` int64 overflow in the hierarchical grid past level 63 is already
    logged and is not this group's target.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One THB case.
    """
    generator = rng(41)
    for dim in dims(profile, max_dim=2):
        for degree in (1, 2, 3) if profile is Profile.FULL else (2,):
            for truncate in (True, False):
                tag = f"dim{dim}_d{degree}_trunc{int(truncate)}"
                params = {"dim": dim, "degree": degree, "truncate": truncate}
                root = create_uniform_space([degree] * dim, [4] * dim)
                yield Case(
                    GROUP,
                    f"thb_create_{tag}",
                    create_thb_space,
                    lambda root=root, truncate=truncate: create_thb_space(
                        root, 2, truncate=truncate
                    ),
                    params,
                )
                space = _try_thb(root, truncate)
                if space is None:
                    continue
                refined = space.refine([0])
                for level_name, current in (("level0", space), ("level1", refined)):
                    cid = current.grid.num_cells - 1
                    pts = _thb_cell_points(current, cid, generator)
                    yield Case(
                        GROUP,
                        f"thb_tabulate_{tag}_{level_name}",
                        type(current).tabulate_basis,
                        lambda current=current, cid=cid, pts=pts: current.tabulate_basis(cid, pts),
                        {**params, "level": level_name, "cid": cid},
                        invariants=(
                            (
                                custom(
                                    "thb-partition-of-unity",
                                    _local_partition_of_unity(degree * dim, np.float64),
                                ),
                            )
                            if truncate
                            else ()
                        ),
                    )
                if profile is not Profile.FULL:
                    continue
                yield Case(
                    GROUP,
                    f"thb_prolongation_{tag}",
                    type(space).prolongation_to,
                    lambda space=space, refined=refined: space.prolongation_to(refined),
                    params,
                )
                yield Case(
                    GROUP,
                    f"thb_refine_out_of_range_{tag}",
                    type(space).refine,
                    lambda space=space: space.refine([space.grid.num_cells]),
                    params,
                )


def cases(profile: Profile) -> Iterator[Case]:
    """Yield every case in this group.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: The group's cases.
    """
    yield from _construction_cases(profile)
    yield from _tabulation_cases(profile)
    yield from _space_operation_cases(profile)
    yield from _curve_operation_cases(profile)
    yield from _nd_cases(profile)
    yield from _extraction_cases(profile)
    yield from _factory_cases(profile)
    yield from _thb_cases(profile)
