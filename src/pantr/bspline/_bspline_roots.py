"""Root finding for scalar B-splines by the Mørken-Reimers method.

Holds the public :func:`find_roots` (Layer 1) and its validation and tolerance
layer (Layer 2). All computation is delegated to the Numba kernels in
:mod:`_bspline_roots_core`.

The method computes the zeros of a scalar spline on the spline's own knot vector
by repeatedly inserting the zero of the control polygon as a new knot
:cite:p:`morken2007zeros`. Compared with extracting the Bézier segments and
solving each of them (:func:`pantr.bezier.find_roots`), it needs no extraction
and no stitching of roots at segment boundaries, zeros sitting exactly on a knot
need no special case, and convergence is unconditional rather than heuristically
controlled.

**Precision contract.** Knots and coefficients are promoted to float64 before
the iteration, which is exact for a float32 spline, so the arithmetic is float64
throughout regardless of the input dtype. The tolerances, however, are resolved
from the *input* dtype: a float32 spline carries float32-level information, and
a residual below that level cannot be distinguished from zero no matter how the
iteration is carried out.

**What is found.** The control polygon says where to look: a spline whose
coefficients have no sign change has no zeros at all, so every zero where the
spline changes sign is bracketed by a sign change and is searched for. What it
does not do is certify. The tracking of a sign change stops for three different
reasons, and none of them implies that the point it stopped at is a zero, so a
value reaches the result only when its residual ``|f(x)|`` is within what a zero
may legitimately leave behind: the error of evaluating the spline, plus the value
the spline reaches at the parametric resolution the iteration stops at,
``|f'| * 2 * tol * scale``. That residual test is the only way a zero of even
multiplicity can be reported at all, since the sign changes bracketing it
disappear under refinement. The two domain endpoints are decided on the
evaluation error alone, being read straight off a coefficient with no location
error to allow for.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bspline._bspline_roots_core import _merge_roots, _morken_reimers_roots
from pantr.tolerance import get_machine_epsilon, get_strict

if TYPE_CHECKING:
    from pantr.bspline._bspline import Bspline

_INSERTIONS_BASE: int = 64
"""Insertion budget for the linear phase of a single zero.

A zero of multiplicity greater than one is approached linearly, and the observed
ratio is one half (measured on an exact double zero), so exhausting the 53-bit
significand takes ``log2(1 / eps) = 53`` insertions. Rounded up to 64, matching
``_MAX_NEWTON_ITER`` in ``pantr.bezier._yuksel_core``, which bounds the same
bisection-grade phase for the Bernstein solver.
"""

_INSERTIONS_PER_DEGREE: int = 8
"""Extra insertions granted per unit of degree.

The error is quadratic once per ``degree - 1`` insertions and not once per
insertion (Mørken-Reimers, Theorem 22), so a zero the theorem covers needs
roughly six quadratic steps of ``degree - 1`` insertions each to reach machine
precision. Eight per degree leaves a margin over the worst case measured on
random splines (173 insertions at degree 25, against a budget of 256).

The theorem's hypotheses are ``f'(z) != 0`` **and** ``f''(z) != 0``, not simplicity
alone: a simple zero with a vanishing second derivative is outside it, and so is
every multiple zero. Nothing rests on the hypotheses holding, since the budget is
soft -- exhausting it stops the tracking with ``_STATUS_BUDGET`` and leaves the
residual to decide -- but the quadratic-step count that sizes the budget is only
justified where they do.
"""

_CURVATURE_DEGREE: int = 2
"""Lowest degree whose splines have a second derivative, hence a curvature merge radius."""

_ZERO_TOL_SAFETY: float = 4.0
"""Safety factor on the evaluation-error bound used as the residual threshold.

Matches the factor applied to the de Casteljau bound in
``pantr.bezier._clipping_core``; de Boor's algorithm combines the same
``degree + 1`` coefficients through the same number of convex-combination
levels, so the bound has the same shape.
"""

_STOP_SLACK: float = 2.0
"""How far a reported zero may sit from the true one, in units of the stopping spread.

The tracking stops when the last ``degree`` iterates span less than ``tol * scale``.
Mørken and Reimers bound the *convergence rate* but not the distance from the last
iterate to the limit at an arbitrary stopping point, so this is **observed, not
proved**: the spread bounds how far the iterates still move, and one further step of
the same size covers the tail. Two is what
``test_residual_at_every_root_is_within_the_evaluation_bound`` measures the residual
against over 240 random splines at four parametric scales, and what
``test_no_returned_value_escapes_the_residual_certificate`` clears by a factor of five
on the families that break this method. Being an estimate, it is used with the safety
factors those tests carry rather than as a bound in its own right.
"""


def _max_insertions(degree: int) -> int:
    """Compute the insertion budget for one zero at a given degree.

    Args:
        degree (int): Polynomial degree of the spline.

    Returns:
        int: Maximum number of knot insertions spent on a single zero.
    """
    return _INSERTIONS_BASE + _INSERTIONS_PER_DEGREE * (degree - 1)


def _validate_bspline_for_roots(bspline: object) -> Bspline:
    """Validate that an object is a scalar univariate B-spline with positive weights.

    Args:
        bspline (object): Input to validate.

    Returns:
        Bspline: The validated B-spline, converted to an open (clamped),
        non-periodic representation when it is not already one.

    Raises:
        TypeError: If ``bspline`` is not a :class:`~pantr.bspline.Bspline`.
        ValueError: If it is not univariate (``dim != 1``), not scalar-valued
            (``rank != 1``), of degree zero, or rational with a non-positive
            weight.
    """
    from pantr.bspline._bspline import Bspline as BsplineCls  # noqa: PLC0415

    if not isinstance(bspline, BsplineCls):
        msg = f"Expected a Bspline instance, got {type(bspline).__name__}"
        raise TypeError(msg)
    if bspline.dim != 1:
        msg = f"Bspline must be univariate (dim == 1), got dim={bspline.dim}"
        raise ValueError(msg)
    if bspline.rank != 1:
        msg = f"Bspline must be scalar valued (rank == 1), got rank={bspline.rank}"
        raise ValueError(msg)
    if bspline.degree[0] < 1:
        msg = (
            "Root finding needs degree >= 1: a degree-zero spline is piecewise "
            "constant, so its zero set is a union of knot intervals, not a set of points"
        )
        raise ValueError(msg)
    if bspline.is_rational:
        weights = np.asarray(bspline.control_points[..., -1], dtype=np.float64)
        if not bool(np.all(weights > 0.0)):
            msg = (
                "Rational B-splines must have strictly positive weights, so that the "
                f"zeros of the numerator are the zeros of the mapping. Got min weight "
                f"{float(weights.min())}"
            )
            raise ValueError(msg)

    if not bspline.space.spaces[0].has_open_knots():
        return bspline.to_open_bspline()
    return bspline


def _check_connected(
    coeffs: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    degree: int,
    zero_tol: float,
) -> None:
    """Reject a spline that vanishes identically on a knot interval.

    Such a spline is *disconnected* in the sense of Mørken and Reimers: on that
    interval every zero is a zero of the spline, so there is no set of isolated
    roots to report and the variation-diminishing bound that the method rests on
    no longer holds. The paper assumes connectedness throughout and notes that
    the degeneracy is easy to detect, which is what this does.

    Args:
        coeffs (npt.NDArray[np.float64]): Scalar B-spline coefficients.
        knots (npt.NDArray[np.float64]): Open knot vector.
        degree (int): Polynomial degree.
        zero_tol (float): Residual below which a coefficient counts as zero.

    Raises:
        ValueError: If every coefficient active on some non-empty knot interval
            is zero to within ``zero_tol``.
    """
    num_coeffs = coeffs.shape[0]
    windows = np.lib.stride_tricks.sliding_window_view(np.abs(coeffs), degree + 1)
    window_max = np.asarray(windows.max(axis=1), dtype=np.float64)

    spans = np.arange(degree, num_coeffs)
    non_empty = knots[spans] < knots[spans + 1]
    vanishing = non_empty & (window_max[spans - degree] <= zero_tol)
    if bool(np.any(vanishing)):
        first = int(spans[vanishing][0])
        msg = (
            f"The spline vanishes identically on the knot interval "
            f"[{knots[first]}, {knots[first + 1]}], so its zero set is not a set of "
            "isolated points. Split the spline to exclude that interval."
        )
        raise ValueError(msg)


def _slope_bound(
    coeffs: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    degree: int,
) -> float:
    """Bound ``|f'|`` over the whole parametric domain, from the hodograph.

    The derivative of a spline is the spline of degree ``degree - 1`` whose
    coefficients are ``degree * (c[i] - c[i-1]) / (t[i+degree] - t[i])``, and a spline
    never leaves the range of its own coefficients, so the largest of those difference
    quotients bounds ``|f'|`` everywhere.

    A zero-length denominator is skipped rather than read as an infinite slope. It occurs
    only at a knot of multiplicity ``degree + 1``, where the spline jumps and no finite
    slope describes it; an infinite bound there would turn the residual test into an
    unconditional accept, which is the defect this whole threshold exists to prevent, and
    the jump itself is rejected on its own residual. At least one denominator is always
    positive: an open knot vector carries exactly ``degree + 1`` copies of the start, so
    ``knots[degree + 1] > knots[1]``.

    The bound is over the whole domain, where the threshold it feeds is applied at one
    zero at a time. A spline with one steep region therefore gets a threshold as loose
    everywhere as that region needs, which errs towards accepting rather than towards
    rejecting a genuine zero. Sharpening it to the knot span holding the zero is a
    tolerance-policy question and is *not* a matter of computing the same quantity on the
    working window: after a hundred insertions the local knot gaps there are degenerate
    and the same formula returns a bound larger than the spline's own range. A correct
    local version would have to index the *original* knot vector by parametric position,
    and ``root_tol`` reaches the kernel as one scalar decided before any zero is located,
    so it would also have to become a per-span quantity the kernel indexes.

    That work is deliberately not done, and the reason is that the looseness has not been
    shown to cost anything. A search for a live false positive -- a value this threshold
    accepts that the local bound would reject -- found none across 948 random splines and
    376 sweep cases. It is therefore a tightening rather than a bug fix, and the direction
    that would be fatal here is the opposite one: for a root finder, a ``root_tol`` that
    is too *large* returns values that are not roots, and no local sharpening can make
    the threshold larger than this bound.

    Args:
        coeffs (npt.NDArray[np.float64]): Scalar B-spline coefficients.
        knots (npt.NDArray[np.float64]): Open knot vector.
        degree (int): Polynomial degree, at least 1.

    Returns:
        float: An upper bound on ``|f'|`` over the domain. Zero for a spline whose
        coefficients are all equal, which is the correct bound for a constant.
    """
    num_coeffs = coeffs.shape[0]
    gaps = knots[degree + 1 : num_coeffs + degree] - knots[1:num_coeffs]
    usable = gaps > 0.0
    quotients = np.abs(np.diff(coeffs))[usable] / gaps[usable]
    return float(degree * quotients.max())


def _merge_radii(  # noqa: PLR0913
    numerator: Bspline,
    roots: npt.NDArray[np.float64],
    *,
    zero_tol: float,
    coeff_scale: float,
    domain_length: float,
    tol: float,
) -> npt.NDArray[np.float64]:
    """Compute the per-root radius within which two reports are the same zero.

    Around a zero of multiplicity ``m`` the set where ``|f| <= zero_tol`` has
    half-width ``(m! * zero_tol / |f^(m)|) ** (1 / m)``, from the first
    non-vanishing term of the Taylor expansion. The two cases that are computed
    are ``m = 1`` and ``m = 2``; every one of them is a length, so all of them
    transform correctly under a change of parametric scale.

    The result is floored at the parametric resolution
    ``tol * max(|x|, domain_length)`` and capped at
    ``domain_length * (degree! * zero_tol / coeff_scale) ** (1 / degree)``, which
    is that same half-width for the highest multiplicity the spline space admits,
    evaluated where ``f^(degree)`` has its natural magnitude
    ``coeff_scale / domain_length ** degree``. The cap is what covers a zero of
    multiplicity three or more, where the two computed derivatives both vanish
    and their radii would otherwise be dominated by rounding; without a cap at
    all, such a zero would merge every later root into itself.

    Args:
        numerator (Bspline): Scalar, non-rational spline whose zeros are sought.
        roots (npt.NDArray[np.float64]): Ascending root candidates.
        zero_tol (float): Residual below which a value counts as zero.
        coeff_scale (float): Largest absolute coefficient of the spline.
        domain_length (float): Length of the parametric domain.
        tol (float): Relative parametric tolerance.

    Returns:
        npt.NDArray[np.float64]: Merge radius for each root.
    """
    degree = numerator.degree[0]
    infinite = np.full(roots.shape, np.inf, dtype=np.float64)

    first = np.abs(np.asarray(numerator.evaluate_derivatives(roots, [1]), dtype=np.float64))
    first = first.reshape(-1)
    radius = np.divide(zero_tol, first, out=infinite.copy(), where=first > 0.0)

    if degree >= _CURVATURE_DEGREE:
        second = np.abs(np.asarray(numerator.evaluate_derivatives(roots, [2]), dtype=np.float64))
        second = second.reshape(-1)
        curved = np.sqrt(np.divide(2.0 * zero_tol, second, out=infinite.copy(), where=second > 0.0))
        radius = np.minimum(radius, curved)

    cap = domain_length * (math.factorial(degree) * zero_tol / coeff_scale) ** (1.0 / degree)
    floor = tol * np.maximum(np.abs(roots), domain_length)
    return np.asarray(np.maximum(np.minimum(radius, cap), floor), dtype=np.float64)


def _merge_certified(
    numerator: Bspline,
    roots: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
    *,
    root_tol: float,
) -> npt.NDArray[np.float64]:
    """Collapse repeated reports of one zero, keeping only the merges that are zeros.

    The midpoint of a run is a value nobody tracked, so it inherits no certificate from
    the reports it replaces and takes the same residual test they did. Where it fails,
    the run was not one zero seen several times: the reports are kept as they were, each
    already certified on its own.

    Without that test a single over-wide radius is enough to return a value that is not a
    root, which is what happens at high degree: the cap ``domain_length * (degree! *
    zero_tol / coeff_scale) ** (1 / degree)`` grows with degree and passes the whole
    domain around degree 17, and runs are joined on the *larger* of two radii, so one
    such radius joins every later root into its own run.

    Args:
        numerator (Bspline): Scalar, non-rational spline whose zeros are sought.
        roots (npt.NDArray[np.float64]): Ascending root candidates.
        radii (npt.NDArray[np.float64]): Per-root merge radius, same length.
        root_tol (float): Residual below which a value counts as a zero.

    Returns:
        npt.NDArray[np.float64]: Ascending roots, with every certified run collapsed to
        its midpoint and every other run left as it was.
    """
    merged, labels, n_merged = _merge_roots(roots, radii)
    merged = merged[:n_merged]
    residuals = np.abs(np.asarray(numerator.evaluate(merged), dtype=np.float64).reshape(-1))
    certified = residuals <= root_tol
    if bool(np.all(certified)):
        return np.ascontiguousarray(merged)
    return np.sort(np.concatenate((merged[certified], roots[~certified[labels]])))


def _find_roots_impl(
    bspline: Bspline,
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """Layer 2 implementation for :func:`find_roots`.

    Args:
        bspline (Bspline): Scalar univariate B-spline.
        tol (float | None): Relative parametric tolerance, or ``None`` for the
            dtype default.

    Returns:
        npt.NDArray[np.float64]: Sorted, read-only array of roots.

    Raises:
        TypeError: If ``bspline`` is not a :class:`~pantr.bspline.Bspline`.
        ValueError: If the spline is not scalar and univariate, has degree zero,
            has a non-positive weight, vanishes identically on a knot interval,
            or if ``tol`` is not positive.
    """
    spline = _validate_bspline_for_roots(bspline)
    if tol is not None and tol <= 0.0:
        msg = f"tol must be positive, got {tol}"
        raise ValueError(msg)

    dtype = spline.dtype
    resolved_tol = get_strict(dtype) if tol is None else tol
    epsilon = get_machine_epsilon(dtype)

    space = spline.space.spaces[0]
    degree = int(space.degree)
    knots = np.ascontiguousarray(space.knots, dtype=np.float64)
    coeffs = np.ascontiguousarray(spline.control_points[..., 0], dtype=np.float64)

    coeff_scale = float(np.abs(coeffs).max())
    if coeff_scale == 0.0:
        msg = "The spline is identically zero, so every point of its domain is a root"
        raise ValueError(msg)
    zero_tol = coeff_scale * max((degree + 1) * _ZERO_TOL_SAFETY * epsilon, resolved_tol)
    _check_connected(coeffs, knots, degree, zero_tol)

    # What a *tracked* iterate may leave behind, as opposed to a value read straight off
    # a coefficient. `zero_tol` is the error of evaluating the spline, and on its own it
    # rejects the genuine zeros of a steep one: the iteration stops at a parametric
    # resolution of `_STOP_SLACK * tol * scale`, and a zero located to that much leaves
    # `|f'|` times it behind however exactly the spline is then evaluated. Both terms are
    # values of the spline, so the sum transforms correctly under a change of either
    # scale: the second is a slope times a length.
    domain_length = float(knots[coeffs.shape[0]] - knots[degree])
    parametric_scale = max(abs(float(knots[degree])), abs(float(knots[coeffs.shape[0]])))
    parametric_scale = max(parametric_scale, domain_length)
    root_tol = zero_tol + _STOP_SLACK * _slope_bound(coeffs, knots, degree) * (
        resolved_tol * parametric_scale
    )

    wait_for_jit_warmup()
    raw, count, truncated, abandoned = _morken_reimers_roots(
        knots, degree, coeffs, resolved_tol, zero_tol, root_tol, _max_insertions(degree)
    )
    budget = _max_insertions(degree)
    if truncated > 0:
        warnings.warn(
            f"{truncated} of {count} roots exhausted their insertion budget of "
            f"{budget} before the iterates stagnated; they are reported at the last "
            "iterate reached, so they may be less accurate than tol asks for",
            RuntimeWarning,
            stacklevel=3,
        )
    if abandoned > 0:
        warnings.warn(
            f"{abandoned} sign changes exhausted their insertion budget of {budget} "
            f"without the residual ever falling to {root_tol:.3e}, so they are not "
            "reported; the result may be missing a root",
            RuntimeWarning,
            stacklevel=3,
        )

    roots = np.ascontiguousarray(raw[:count])
    if count > 1:
        numerator = _scalar_numerator(spline, knots, coeffs)
        radii = _merge_radii(
            numerator,
            roots,
            zero_tol=zero_tol,
            coeff_scale=coeff_scale,
            domain_length=domain_length,
            tol=resolved_tol,
        )
        roots = _merge_certified(numerator, roots, radii, root_tol=root_tol)

    roots.flags.writeable = False
    return roots


def _scalar_numerator(
    spline: Bspline,
    knots: npt.NDArray[np.float64],
    coeffs: npt.NDArray[np.float64],
) -> Bspline:
    """Build the float64, non-rational spline whose zeros are being reported.

    For a rational spline the zeros of the mapping are the zeros of its
    numerator, the weights being strictly positive, so the numerator is what the
    derivatives used by :func:`_merge_radii` must be taken of.

    Args:
        spline (Bspline): The validated input spline.
        knots (npt.NDArray[np.float64]): Its knot vector, as float64.
        coeffs (npt.NDArray[np.float64]): Its scalar coefficients, as float64.

    Returns:
        Bspline: A non-rational float64 spline with the same zeros.
    """
    from pantr.bspline._bspline import Bspline as BsplineCls  # noqa: PLC0415
    from pantr.bspline._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415
    from pantr.bspline._bspline_space_nd import BsplineSpace  # noqa: PLC0415

    degree = int(spline.space.spaces[0].degree)
    space = BsplineSpace([BsplineSpace1D(knots, degree, snap_knots=False)])
    return BsplineCls(space, coeffs.reshape(-1, 1))


def find_roots(
    bspline: Bspline,
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """Find every zero of a scalar univariate B-spline.

    Uses the Mørken-Reimers method: the zero of the control polygon is inserted
    as a new knot, over and over, until the iterates stagnate. The iteration
    needs no starting value, converges for any spline, and reaches a simple zero
    quadratically -- once per ``degree - 1`` insertions, not once per insertion.

    Args:
        bspline (Bspline): A univariate (``dim == 1``), scalar-valued
            (``rank == 1``) B-spline. Periodic or unclamped splines are
            converted to an open representation first. For rational splines the
            zeros are those of the numerator, which are the zeros of the
            mapping because the weights are positive.
        tol (float | None): Relative parametric tolerance. The iteration stops
            when the spread of the last ``degree`` iterates falls to
            ``tol * max(|t_a|, |t_{a+degree}|, domain_length)``, so that it
            transforms correctly when the parametric domain is scaled or moved
            away from the origin. Defaults to
            ``tolerance.get_strict(bspline.dtype)``.

    Returns:
        npt.NDArray[np.float64]: Sorted, read-only array of roots, always
        float64. Empty if the spline has no zero.

    Raises:
        TypeError: If ``bspline`` is not a :class:`~pantr.bspline.Bspline`.
        ValueError: If ``bspline`` is not univariate and scalar valued, has
            degree zero, is rational with a non-positive weight, vanishes
            identically on a knot interval, or if ``tol`` is not positive.

    Warns:
        RuntimeWarning: If a zero exhausted its insertion budget before the
            iterates stagnated, in which case the last iterate reached is
            reported and may be less accurate than ``tol`` asks for; and,
            separately, if a sign change exhausted that budget without its
            residual ever certifying it, in which case it is not reported at all
            and the result may be missing a root.

    Note:
        A value is reported only when the residual there is within what a zero
        may legitimately leave behind: the de Boor evaluation-error bound of the
        spline, plus ``|f'| * 2 * tol * scale``, the value the spline reaches at
        the parametric resolution the iteration stops at. The control polygon
        says where to look and never certifies on its own: the tracking of a sign
        change stops for three different reasons, and none of them implies that
        the point it stopped at is a zero. The residual test is also the only way
        a zero of even multiplicity is reported, since it does not change the
        sign of the spline and the sign changes bracketing it vanish under
        refinement.

        At a knot of multiplicity ``degree + 1`` the spline is discontinuous, and
        a zero of the piece on either side is reported at the knot. Evaluating
        there returns the *right* limit, so such a root can show a residual as
        large as the jump when the zero belongs to the left piece. Extracting the
        Bézier segments and solving each of them
        (:func:`pantr.bezier.find_roots`) reports the same knot for the same
        reason.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import BsplineSpace, BsplineSpace1D, Bspline, find_roots
        >>> knots = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
        >>> space = BsplineSpace([BsplineSpace1D(knots, 1)])
        >>> spline = Bspline(space, np.array([[-1.0], [1.0], [3.0]]))
        >>> find_roots(spline)
        array([0.5])

    References:
        The unconditionally convergent knot-insertion method of Mørken and
        Reimers :cite:p:`morken2007zeros`.
    """
    return _find_roots_impl(bspline, tol=tol)


__all__ = ["find_roots"]
