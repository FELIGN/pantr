"""B-spline knot vector utilities and analysis.

This module provides functions for validating, analyzing, and querying
B-spline knot vectors including multiplicity computation, domain checks,
cardinal interval identification, and knot vector generation utilities.

Every ``tol`` argument in this module is an **absolute** parametric length, in the
units of the knot vector itself. :func:`_knot_tolerance` is what turns the
dimensionless presets of :mod:`pantr.tolerance` into one, by multiplying by the
knot vector's own magnitude; :class:`~pantr.bspline.BsplineSpace1D` calls it once
at construction and hands the result to everything below.
"""

from __future__ import annotations

from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit
from ..basis._basis_utils import _validate_float_dtype, _validate_out_array
from ..tolerance import get_strict

_KNOT_MERGE_SAFETY: Final[float] = 2.0
"""Extra factor on the strict tier for deciding that two knots are the same knot.

The quantity to cover is how far apart two knots a caller *means* to be equal can
land when they are obtained by different routes -- a uniform ``a + i * (b - a) / n``
against a barycentric ``(a * (n - i) + b * i) / n``, a refinement, a
reparametrization, a Bezier split. Write ``u = eps / 2`` for the unit roundoff and
``scale`` for what :func:`_knot_scale` returns.

Each route is four floating-point operations and neither amplifies. The uniform
one: ``b - a`` carries an absolute error at most ``u * |b - a| <= u * scale``; the
division and the multiplication each add ``u`` relative on a quantity bounded by
``scale``; the final sum adds ``u * scale``. Four terms of ``u * scale``, so the
computed knot is within ``4u * scale = 2 * eps * scale`` of the exact one. The
barycentric one is the same count and comes out slightly smaller, because the two
products are divided by ``n`` after being added. By the triangle inequality two
routes differ by at most ``4 * eps * scale``.

Eight is that bound doubled, and the doubling buys the things the count does not
see: an FMA contraction (which removes a rounding but changes the value), a
different vector width, and one further rounding upstream of the two routes.
:func:`~pantr.tolerance.get_strict` is four epsilons, so the factor here is two.

The bound is not tight, which is the intent. Measured over 200000 random
``(a, b, n, i)`` with ``|a|`` and the span drawn across 1e-3 to 1e7, the largest
observed ``|route_A - route_B| / (eps * scale)`` is 1.74, against the derived 4 and
the applied 8.
"""


def _knot_scale(knots: npt.NDArray[np.float32 | np.float64]) -> float:
    """Get the magnitude a parametric comparison on a knot vector is relative to.

    The span alone is not enough: on a knot vector based at 1e6 a knot difference
    is formed from two coordinates of that size, so it carries an absolute error
    of ``eps * 1e6`` however short the span is. The coordinate magnitudes are
    therefore taken alongside the span, and the largest wins.

    No floor is applied. A floor of 1.0 would be a physical choice this module is
    not entitled to make, and it would destroy scale covariance on a domain
    smaller than one unit -- exactly the case (``[0, 1e-6]``) where the previous
    absolute tolerance merged every knot in the vector.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Non-decreasing knot vector.

    Returns:
        float: ``max(span, |knots[0]|, |knots[-1]|)``, zero only for a knot vector
        that is identically zero.
    """
    lo, hi = float(knots[0]), float(knots[-1])
    return max(hi - lo, abs(lo), abs(hi))


def _knot_tolerance(knots: npt.NDArray[np.float32 | np.float64]) -> float:
    """Get the absolute tolerance for comparing parametric coordinates of a knot vector.

    ``_KNOT_MERGE_SAFETY * get_strict(dtype) * _knot_scale(knots)``, i.e.
    ``8 * eps * scale``. Every parametric comparison the B-spline layer makes --
    are these two knots one knot, is this point inside the domain, are these two
    knot intervals the same length -- is a comparison of quantities of that
    magnitude, so they all share this one number.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Non-decreasing knot vector.

    Returns:
        float: Absolute parametric tolerance, in the units of ``knots``.

    Raises:
        ValueError: If the knot dtype is not a supported floating-point type.
    """
    return _KNOT_MERGE_SAFETY * get_strict(knots.dtype) * _knot_scale(knots)


def _check_snapping_kept_an_interval(
    raw: npt.NDArray[np.float32 | np.float64],
    snapped: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> None:
    """Reject a knot vector that snapping collapsed onto a single point.

    This function refuses one specific cause -- the knots *were* distinct, and the
    merge rule found none of them distinguishable at their own magnitude -- so that
    the caller is told the thing they could not see coming. A vector that arrived
    already flat is not this function's case and passes through it untouched; it is
    refused a step later by :func:`_check_space_has_an_interval`, which owns the
    *consequence* both causes share. Keeping the two apart is what lets each message
    be true: "this mesh is finer than float32 resolves here" is actionable, and false
    of a vector the caller supplied flat.

    This is reachable, and the arithmetic that makes it so is not a defect. Two knots
    obtained by different routes differ by up to ``4 * eps * scale``, so telling them
    apart needs a tolerance at least that wide; a mesh of ``n`` intervals over a span
    ``s`` at offset ``x`` survives only while ``s / n > 8 * eps * max(s, |x|)``. When
    ``|x|`` dominates, that fails once ``|x| / s`` reaches ``1 / (8 * eps * n)`` --
    about ``5.2e5 / n`` in float32 and ``5.6e14 / n`` in float64. Past it no threshold
    satisfies both requirements at once, because an interior knot is then uncertain by
    a sizeable fraction of the span. Saying so is the only honest answer.

    Args:
        raw (npt.NDArray[np.float32 | np.float64]): The knot vector as supplied,
            non-decreasing, before snapping.
        snapped (npt.NDArray[np.float32 | np.float64]): The same vector afterwards.
        tol (float): The absolute tolerance snapping used, from :func:`_knot_tolerance`.

    Raises:
        ValueError: If ``raw`` held more than one distinct knot while ``snapped``
            holds only one.
    """
    # Both vectors are non-decreasing, so "all knots identical" is exactly
    # "first equals last", tested bitwise and needing no tolerance of its own.
    if raw[0] == raw[-1] or snapped[0] != snapped[-1]:
        return

    name = raw.dtype.name
    scale = _knot_scale(raw)
    ulp = float(np.spacing(raw.dtype.type(scale)))
    gaps = np.diff(np.asarray(raw, dtype=np.float64))
    closest = float(gaps[gaps > 0.0].min())
    # Widening the format is only a remedy if there is a wider one to move to.
    remedy = (
        "Use float64, move the domain nearer the origin, or coarsen the mesh."
        if raw.dtype.type is np.float32
        else "Move the domain nearer the origin, or coarsen the mesh."
    )
    raise ValueError(
        f"knot snapping collapsed every knot onto {float(snapped[0])!r}: in {name} at "
        f"|coordinate| ~ {scale:.3g} two knots are the same knot unless they differ by "
        f"more than {tol:.3g} ({tol / ulp:.0f} ulp there), and the closest pair in "
        f"[{float(raw[0])!r}, {float(raw[-1])!r}] is {closest:.3g} apart. This mesh is "
        f"finer than {name} resolves at that magnitude. {remedy}"
    )


def _check_space_has_an_interval(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    num_intervals: int,
    tol: float,
) -> None:
    """Reject a knot vector whose domain is a single point.

    A space with no interval has no cell, and every operation defined over its cells
    is therefore undefined on it. It was accepted until issue #320, and each consumer
    then met the degeneracy on its own terms: ``tensor_product_grid`` named the
    breakpoints, ``Bspline.locate`` surfaced ``numpy``'s zero-size ``min``,
    ``tabulate_basis_derivatives`` divided by zero, and -- worse than any error --
    ``Bspline.evaluate`` returned ``0.0`` while ``tabulate_basis`` and
    ``evaluate_derivatives`` returned ``NaN``. Those three returned a value, so a
    survey of what *raised* would not have found them, which is why the rule lives
    here at the one place a space comes into being rather than at each consumer. It
    is also the rule ``create_cardinal_knots`` has always applied, as
    ``num_intervals must be at least 1``.

    The predicate is the interval count, not "every knot is the same value". The
    family is wider than it looks: ``[0, 1, 1, 1, 2]`` at degree 1 holds three
    distinct values and still has a domain of ``(1.0, 1.0)``, because the domain runs
    from ``knots[degree]`` to ``knots[-degree-1]`` and an interior knot of high enough
    multiplicity swallows it whole. Counting intervals catches that case and the flat
    one together, and it introduces no threshold of its own -- the count is decided by
    the same knot grouping the space uses everywhere else.

    That grouping splits on ``knots[i] - knots[i-1] > tol``, so it chains: the two
    domain ends can be further apart than ``tol`` and still fall in one class, through
    knots between them. The message therefore reports what is actually true of every
    case it fires on -- each consecutive step across the domain is at most ``tol`` --
    rather than claiming the two ends are within ``tol`` of each other, which they need
    not be.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector, non-decreasing
            and already snapped if snapping was requested.
        degree (int): Polynomial degree, which fixes where the domain starts and ends.
        num_intervals (int): The space's interval count, one less than the number of
            distinct in-domain knots.
        tol (float): The absolute tolerance the count was taken at, from
            :func:`_knot_tolerance`.

    Raises:
        ValueError: If ``num_intervals`` is zero.
    """
    if num_intervals > 0:
        return

    last = knots.size - degree - 1
    lo, hi = float(knots[degree]), float(knots[last])
    raise ValueError(
        f"knot vector spans no interval: at degree {degree} the domain runs from "
        f"knots[{degree}] = {lo!r} to knots[{last}] = {hi!r}, and every step between "
        f"them is at most this vector's tolerance of {tol:.3g}, so its in-domain knots "
        f"are all one knot, the space has no cell, and nothing can be evaluated, "
        f"tabulated or located on it. The domain needs two consecutive knots more than "
        f"{tol:.3g} apart."
    )


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _check_spline_info(knots: npt.NDArray[np.float32 | np.float64], degree: int) -> None:
    """Validate basic constraints on B-spline knot vector and degree.

    Performs core checks on a B-spline knot vector and polynomial degree,
    raising an ValueError if any validation fails.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): 1D array representing the B-spline
            knot vector to check.
        degree (int): Non-negative polynomial degree for the B-spline.

    Raises:
        TypeError: If `knots` is not 1-dimensional.
        ValueError: If `degree` is negative,
            if there are fewer than `2*degree+2` knots, or if the knot vector
            is not non-decreasing.
    """
    if knots.ndim != 1:
        raise TypeError("knots must be a 1D array")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if knots.size < (2 * degree + 2):
        raise ValueError("knots must have at least 2*degree+2 elements")
    if not np.all(np.diff(knots) >= knots.dtype.type(0.0)):
        raise ValueError("knots must be non-decreasing")


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_multiplicity_of_first_knot_in_domain_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
) -> int:
    """Get the multiplicity of the first knot in the domain (i.e., the `degree`-th knot).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        int: Multiplicity of the first knot in the domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    first_knot = knots[degree]
    return int(np.sum(np.abs(knots[: degree + 1] - first_knot) <= tol))


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_unique_knots_and_multiplicity_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    in_domain: bool = False,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Get unique knots and their multiplicities.

    Knots are grouped by *gap*: the vector is non-decreasing, so one pass suffices,
    and a class ends where the step to the next knot exceeds ``tol``. Every member
    of a class is represented by the class's **first** knot, chosen rather than
    averaged, so the values returned are knots the vector actually contains and
    the grouping is idempotent -- regrouping the represented vector reproduces it.
    An average would be a fresh rounding, and could place two adjacent classes'
    representatives a single ulp apart.

    Grouping by gap rather than by a rounding grid is what makes the answer depend
    on the knots and not on where they sit relative to an origin: two values one
    ulp apart can straddle a grid line and never merge, for any grid spacing.

    The cost of the gap rule is chaining -- ``n`` knots each within ``tol`` of the
    next collapse into one class however far apart the ends are. That is accepted
    rather than capped: a class-width cap would break the idempotence above, since
    a class split by the cap leaves two representatives that may be within ``tol``
    of each other and would merge on the next pass. Chaining needs every step of
    the chain to sit at the format's noise floor, which is a mesh that has already
    stopped being resolvable.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): **Absolute** parametric tolerance: consecutive knots closer
            than this are one knot. See :func:`_knot_tolerance` for how it is
            derived from the knot vector's own magnitude.
        in_domain (bool): If True, only consider knots in the domain.
            Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts. Both arrays have
            the same length. With ``in_domain=False`` the multiplicities sum to
            ``knots.size``, so ``np.repeat(unique_knots, multiplicities)`` rebuilds
            the whole vector with each class collapsed onto its representative.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = knots.size
    # Index of the first knot of each class, and how many knots the class holds.
    first_ids = np.empty(n, dtype=np.int_)
    mult = np.zeros(n, dtype=np.int_)

    # The class containing the first and the last knot of the domain. Their whole
    # class is reported, clamped copies outside the domain included, which is what
    # the multiplicity of a boundary knot means.
    lo_index, hi_index = degree, n - degree - 1
    lo_class, hi_class = 0, 0

    j = 0
    first_ids[0] = 0
    mult[0] = 1
    for i in range(1, n):
        if knots[i] - knots[i - 1] > tol:
            j += 1
            first_ids[j] = i
        mult[j] += 1
        if i == lo_index:
            lo_class = j
        if i == hi_index:
            hi_class = j

    lo, hi = (lo_class, hi_class) if in_domain else (0, j)

    unique_knots = knots[first_ids[lo : hi + 1]]
    mults = mult[lo : hi + 1]
    return unique_knots, mults


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _is_in_domain_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    pts: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> npt.NDArray[np.bool_]:
    """Check if points are within the B-spline domain (up to tolerance).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        pts (npt.NDArray[np.float32 | np.float64]): Points to check.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates points
            are within the domain. It has the same length as the number of points.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    knot_begin, knot_end = knots[degree], knots[-degree - 1]
    # ``tol`` is an absolute tolerance, so the admitted band is exactly ``tol`` wide at
    # each end.  ``np.isclose`` would carry its default ``rtol=1e-5`` and, since the
    # rtol leg attaches to the second operand, accept an overshoot of
    # ``1e-5 * |knot_end|`` at the right end while rejecting ``tol``-sized undershoot
    # at a left end sitting at zero -- an asymmetry that is an accident of argument
    # order.
    #
    # The difference is formed first and only then compared against ``tol``.  Shifting
    # the bound instead (``pts <= knot_end + tol``) reads more directly but is not the
    # same predicate in floating point: ``knot_end + tol`` rounds to the nearest
    # representable value, so once ``tol`` drops below ``ulp(knot_end)`` the effective
    # tolerance becomes ``ulp(knot_end)`` rather than ``tol`` -- 1.16e-10 instead of a
    # requested 1e-10 on a domain ending at 1e6.  Subtracting first is exact here
    # (Sterbenz, for the nearby points that decide the boundary), so the band stays
    # ``tol`` wide at every scale.
    return np.logical_and(  # type: ignore[no-any-return]
        (knot_begin < pts) | (np.abs(knot_begin - pts) <= tol),
        (pts < knot_end) | (np.abs(pts - knot_end) <= tol),
    )


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_last_knot_smaller_equal_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    pts: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.int_]:
    """Get the index of the last knot which is less than or equal to each point in pts.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector (must be non-decreasing).
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to find knot indices for.

    Returns:
        npt.NDArray[np.int_]: Array of computed indices, one for each point in pts.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return np.searchsorted(knots, pts, side="right") - 1


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_Bspline_num_basis_1D_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
) -> int:
    """Compute the number of basis functions.

    In the non-periodic case, the number of basis functions is given by
    the number of knots minus the degree minus 1.

    In the periodic case, the number of basis functions is computed as
    before, minus the regularity at the domain's beginning/end minus 1.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (np.float32 | np.float64): Tolerance for numerical comparisons.

    Returns:
        int: Number of basis functions.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    num_basis = int(len(knots) - degree - 1)

    if periodic:
        # Determining the number of extra basis required in the periodic case.
        # This depends on the regularity of the knot vector at domain's
        # begining.
        regularity = degree - _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        num_basis -= regularity + 1

    return num_basis


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_Bspline_cardinal_intervals_1D_core(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.bool_],
) -> None:
    """Core implementation to compute cardinal intervals, writing to output array.

    An interval is cardinal if it has the same length as the degree-1
    previous and the degree-1 next intervals.

    In the case of open knot vectors, this definition automatically
    discards the first degree-1 and the last degree-1 intervals.

    At ``degree == 0`` there are no neighbouring intervals to compare against, so the
    equal-length condition holds vacuously and only the multiplicity gate below
    decides the answer; see the in-body comment for why that is consistent with the
    geometry.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.bool_]): Output array where results will be written.
            Must have the correct shape (no validation performed inside this
            numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-validated inputs and assumes the output array has the
        correct shape. Inputs are assumed to be correct (no validation performed).
        For general use, call _get_Bspline_cardinal_intervals_1D_impl instead.
    """
    _, mult = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    out.fill(np.False_)

    if np.all(mult > 1):
        return

    knot_id = degree

    # Note: this loop could be shortened by only looking at those
    # intervals for which the multiplicity of the first knot is 1.
    # This would require to compute knot_id differently.
    for elem_id in range(num_intervals):
        if mult[elem_id] == 1 and mult[elem_id + 1] == 1:
            if degree == 0:
                # The comparison window below spans ``2 * degree - 1`` knot intervals
                # and its reference is the centre entry ``degree - 1``; at degree 0 the
                # window is empty and that index addressed a zero-length array.  There
                # is nothing to compare, so the length condition holds vacuously and
                # this interval is cardinal -- it already passed the multiplicity gate
                # above, which still applies at degree 0.  Consistent with the geometry
                # either way: a degree-0 space has one basis function per interval, so
                # its cardinal extraction operator is the 1x1 identity on every
                # interval whatever this flag says.
                out[elem_id] = np.True_
            else:
                local_knots = knots[knot_id - degree + 1 : knot_id + degree + 1]
                lengths = np.diff(local_knots)
                if np.all(np.abs(lengths - lengths[degree - 1]) <= tol):
                    out[elem_id] = np.True_

        knot_id += mult[elem_id + 1]


def _get_Bspline_cardinal_intervals_1D_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    """Get boolean array indicating whether intervals are cardinal.

    An interval is cardinal if it has the same length as the degree-1
    previous and the degree-1 next intervals.

    In the case of open knot vectors, this definition automatically
    discards the first degree-1 and the last degree-1 intervals.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.bool_] | None): Optional output array where the result will be
            stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates cardinal intervals.
            It has length equal to the number of intervals. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    _, mult = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    if out is None:
        out = np.empty(num_intervals, dtype=np.bool_)
    else:
        _validate_out_array(out, (num_intervals,), np.bool_)

    _get_Bspline_cardinal_intervals_1D_core(knots, degree, tol, out)

    return out


def _validate_knot_input(
    num_intervals: int,
    degree: int,
    continuity: int,
    domain: tuple[np.float32 | np.float64, np.float32 | np.float64],
    dtype: npt.DTypeLike,
) -> None:
    """Validate input parameters for knot vector generation.

    Args:
        num_intervals (int_): Number of intervals in the domain.
        degree (int): B-spline degree.
        continuity (int): Continuity level at interior knots.
        domain (tuple[np.float32 | np.float64, np.float32 | np.float64]):
            Domain boundaries as (start, end).
        dtype (np.dtype): Data type for the knot vector.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if domain[0] >= domain[1]:
        raise ValueError("domain[0] must be less than domain[1]")

    if num_intervals < 0:
        raise ValueError("num_intervals must be non-negative")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    if continuity < -1 or continuity >= degree:
        raise ValueError(f"Continuity must be between -1 and {degree - 1} for degree {degree}.")

    _validate_float_dtype(dtype)


def _ensure_scalar_arrays(
    values: dict[str, float | int | np.floating[Any] | None],
) -> dict[str, npt.NDArray[np.generic]]:
    """Convert scalar inputs to zero-dimensional arrays.

    Args:
        values (dict[str, Optional[float | int | np.floating]]): Mapping of parameter
            names to scalar values.

    Returns:
        dict[str, npt.NDArray[np.generic]]: Mapping of provided names to 0-D arrays.

    Raises:
        ValueError: If any value is not scalar.
    """
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    for name, value in values.items():
        if value is None:
            continue
        array_value = np.array(value)
        if array_value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value")
        arrays[name] = array_value
    return arrays


def _resolve_dtype_from_arrays(
    arrays: dict[str, npt.NDArray[np.generic]],
    requested_dtype: npt.DTypeLike | None,
) -> np.dtype[np.floating[Any]]:
    """Resolve the floating dtype to use for knot endpoints.

    Args:
        arrays (dict[str, npt.NDArray[np.generic]]): Scalar arrays for each value.
        requested_dtype (Optional[npt.DTypeLike]): Explicit dtype request.

    Returns:
        np.dtype[np.floating[Any]]: Resolved floating-point dtype.

    Raises:
        ValueError: If the dtype is invalid or inconsistent across values.
    """
    if requested_dtype is not None:
        dtype_obj = np.dtype(requested_dtype)
        if dtype_obj.kind != "f":
            raise ValueError("dtype must be a floating-point type")
        for name, array_value in arrays.items():
            if array_value.dtype != dtype_obj:
                raise ValueError(f"{name} must be of type dtype {dtype_obj}")
        return cast(np.dtype[np.floating[Any]], dtype_obj)

    inferred_dtype: np.dtype[np.floating[Any]] | None = None
    for array_value in arrays.values():
        candidate = (
            cast(np.dtype[np.floating[Any]], np.dtype(array_value.dtype))
            if array_value.dtype.kind == "f"
            else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
        )
        if inferred_dtype is None:
            inferred_dtype = candidate
        elif candidate != inferred_dtype:
            raise ValueError("start and end must have the same dtype")

    return (
        inferred_dtype
        if inferred_dtype is not None
        else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
    )


def _coerce_scalar(
    array_value: npt.NDArray[np.generic] | None,
    dtype_obj: np.dtype[np.floating[Any]],
    default: float,
) -> np.floating[Any]:
    """Convert a scalar array to the target dtype or use the default value.

    Args:
        array_value (Optional[npt.NDArray[np.generic]]): Scalar array to convert.
        dtype_obj (np.dtype[np.floating[Any]]): Target floating dtype.
        default (float): Default value when the array is None.

    Returns:
        np.floating[Any]: Value converted to the requested dtype.
    """
    if array_value is None:
        return dtype_obj.type(default)
    return dtype_obj.type(array_value.astype(dtype_obj, copy=False).item())


def _get_knots_ends_and_dtype(
    start: float | int | np.floating[Any] | None = None,
    end: float | int | np.floating[Any] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> tuple[np.floating[Any], np.floating[Any], np.dtype[np.floating[Any]]]:
    """Get the start, end, and dtype for a knot vector.

    Args:
        start (Optional[float | int | np.floating]): Start value of the domain.
            Defaults to 0.0 if not provided.
        end (Optional[float | int | np.floating]): End value of the domain.
            Defaults to 1.0 if not provided.
        dtype (Optional[npt.DTypeLike]): Data type for the knot vector.
            If ``None``, inferred from start/end or defaults to ``np.float64``.

    Returns:
        tuple[np.floating, np.floating, np.dtype]: Tuple of (start, end, dtype).

    Raises:
        ValueError: If inputs are non-scalar, have incompatible dtypes, or if end <= start.
    """
    arrays = _ensure_scalar_arrays({"start": start, "end": end})
    dtype_obj = _resolve_dtype_from_arrays(arrays, dtype)
    start_value = _coerce_scalar(arrays.get("start"), dtype_obj, 0.0)
    end_value = _coerce_scalar(arrays.get("end"), dtype_obj, 1.0)

    if end_value <= start_value:
        raise ValueError("end must be greater than start")

    return start_value, end_value, dtype_obj


def _find_span(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    n_basis: int,
    value: float,
    tol: float,
) -> tuple[int, float]:
    """Find the knot span index for a parameter value (Piegl-Tiller convention).

    Returns the span index ``k`` such that ``knots[k] <= u < knots[k+1]``,
    with the convention that at the right boundary ``k = n_basis - 1``.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): Polynomial degree.
        n_basis (int): Number of basis functions.
        value (float): Parameter value (must be in the domain).
        tol (float): Tolerance for boundary clamping.

    Returns:
        tuple[int, float]: ``(k, u)`` where ``k`` is the span index and
        ``u`` is the (possibly clamped) parameter value.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    u = float(value)
    u_left = float(knots[degree])
    u_right = float(knots[n_basis])

    # Clamp to domain boundaries.
    if u <= u_left + tol:
        return degree, u_left
    if u >= u_right - tol:
        return n_basis - 1, u_right

    # Interior: binary search via searchsorted.
    k = int(np.searchsorted(knots, u, side="right")) - 1
    return k, u


def _count_multiplicity(
    knots: npt.NDArray[np.float32 | np.float64],
    k: int,
    degree: int,
    u: float,
    tol: float,
) -> int:
    """Count the multiplicity of ``u`` among the local knots around span ``k``.

    Scans knots ``k``, ``k-1``, ... going left while they match ``u``
    (within tolerance).  This gives the number of consecutive knots equal
    to ``u`` ending at position ``k``.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        k (int): Span index from :func:`_find_span`.
        degree (int): Polynomial degree.
        u (float): Parameter value.
        tol (float): Tolerance for knot coincidence tests.

    Returns:
        int: Multiplicity of ``u`` in the local knot neighborhood.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    s = 0
    for i in range(k, max(k - degree - 1, -1), -1):
        if abs(float(knots[i]) - u) <= tol:
            s += 1
        else:
            break
    return s


def _find_knot_index_and_multiplicity(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    knot_value: float,
    tol: float,
) -> tuple[int, int]:
    """Find the last index and multiplicity of a knot value in the knot vector.

    Searches the knot vector for the last occurrence of *knot_value* (within
    tolerance) and returns its index *r* together with the multiplicity *s*.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): Polynomial degree.
        knot_value (float): The knot value to locate.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        tuple[int, int]: ``(r, s)`` where *r* is the index of the last
        occurrence and *s* is the multiplicity.

    Raises:
        ValueError: If *knot_value* is not found in the knot vector.
    """
    unique_knots, mults = _get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=False
    )

    # Find the matching unique knot.  The comparison is absolute: ``np.isclose`` would
    # keep its default ``rtol=1e-5`` and match a knot up to ``1e-5 * |knot_value|``
    # away, so a removal request would silently target a different knot (10.0 away at
    # a knot value of 1e6).
    matches = np.where(np.abs(unique_knots - knot_value) <= tol)[0]
    if len(matches) == 0:
        raise ValueError(f"Knot value {knot_value} not found in knot vector (tolerance={tol}).")

    idx = int(matches[0])
    s = int(mults[idx])

    # Compute the last index r of this knot value in the full knot vector.
    # Sum multiplicities up to and including this knot.
    r = int(np.sum(mults[: idx + 1])) - 1

    return r, s


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2

    # Warmup each numba function with float64
    _check_spline_info(knots_dummy, degree_dummy)
    _get_multiplicity_of_first_knot_in_domain_impl(knots_dummy, degree_dummy, tol_dummy)
    _get_unique_knots_and_multiplicity_impl(knots_dummy, degree_dummy, tol_dummy, False)
    _is_in_domain_impl(knots_dummy, degree_dummy, pts_dummy, tol_dummy)
    _get_Bspline_num_basis_1D_impl(knots_dummy, degree_dummy, False, tol_dummy)
    _get_last_knot_smaller_equal_impl(knots_dummy, pts_dummy)
    # For cardinal intervals: with knots [0,0,0,1,1,1] and degree 2, we have 1 interval
    out_cardinal_dummy = np.empty(1, dtype=np.bool_)
    _get_Bspline_cardinal_intervals_1D_core(
        knots_dummy, degree_dummy, tol_dummy, out_cardinal_dummy
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)


__all__ = [
    "_check_snapping_kept_an_interval",
    "_check_space_has_an_interval",
    "_check_spline_info",
    "_count_multiplicity",
    "_find_knot_index_and_multiplicity",
    "_find_span",
    "_get_Bspline_cardinal_intervals_1D_impl",
    "_get_Bspline_num_basis_1D_impl",
    "_get_knots_ends_and_dtype",
    "_get_last_knot_smaller_equal_impl",
    "_get_multiplicity_of_first_knot_in_domain_impl",
    "_get_unique_knots_and_multiplicity_impl",
    "_is_in_domain_impl",
    "_knot_scale",
    "_knot_tolerance",
    "_validate_knot_input",
]
