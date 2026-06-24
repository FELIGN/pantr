"""Bézier degree elevation, reduction, and minimization.

This module provides :func:`_degree_elevate_bezier`, which raises the polynomial
degree of a Bézier in one or more parametric directions while preserving the
same geometric mapping, :func:`_degree_reduce_bezier`, which computes a
least-squares degree-reduced approximation, and :func:`_minimize_degree_bezier`,
which automatically finds the lowest degree that preserves accuracy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._array_utils import _flatten_along_axis, _unflatten_along_axis
from ._bezier_core import _degree_elevate_bezier_1d_core, _degree_reduce_bezier_1d_core

if TYPE_CHECKING:
    from . import Bezier

_AUTO_REDUCTION_TOL_FACTOR: float = 1.0e3
"""Default relative tolerance for automatic degree reduction, in units of eps.

When :func:`_minimize_degree_bezier` is called without an explicit ``tol``, a
degree-1 reduction is accepted whenever the round-trip (reduce then re-elevate)
relative :math:`L^2` error stays below ``_AUTO_REDUCTION_TOL_FACTOR * eps``,
where ``eps`` is the machine epsilon of the control-point dtype.

The factor of ``1e3`` gives roughly three decimal digits of headroom above the
unit-roundoff floor. The least-squares reduction and the subsequent
re-elevation each accumulate ``O(p)`` floating-point operations, so a curve that
is exactly reducible (for instance a degree-elevated lower-degree curve)
produces a round-trip error of a small multiple of ``eps`` rather than exactly
zero. A threshold at the bare ``eps`` level would spuriously reject such curves;
``1e3 * eps`` (``~2.2e-13`` for ``float64``) comfortably accepts genuine
reductions while still rejecting curves whose true degree cannot be lowered
without visible geometric error.
"""


def _degree_elevate_bezier(
    bezier: Bezier,
    increments: tuple[int, ...],
) -> Bezier:
    """Degree-elevate a Bézier in one or more parametric directions.

    For each direction with a positive increment, applies the Bézier degree
    elevation kernel via the shared flatten/unflatten helpers.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to elevate.
        increments (tuple[int, ...]): Degree increment per direction. All
            values must be non-negative; at least one must be positive.

    Returns:
        ~pantr.bezier.Bezier: New Bézier with elevated degrees and updated
        control points.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    degrees = bezier.degree

    for d in range(bezier.dim):
        inc = increments[d]
        if inc == 0:
            continue

        p = degrees[d]

        pts_2d, trailing_shape = _flatten_along_axis(ctrl, d)
        new_pts_2d = _degree_elevate_bezier_1d_core(p, pts_2d, inc)
        ctrl = _unflatten_along_axis(new_pts_2d, trailing_shape, d)

        # Update degrees for subsequent iterations.
        degrees = (*degrees[:d], p + inc, *degrees[d + 1 :])

    return BezierCls(ctrl, is_rational=bezier.is_rational)


def _degree_reduce_bezier(
    bezier: Bezier,
    decrements: tuple[int, ...],
) -> Bezier:
    """Degree-reduce a Bézier in one or more parametric directions.

    For each direction with a positive decrement, applies the Bézier degree
    reduction kernel via the shared flatten/unflatten helpers.  The reduction is a
    least-squares approximation (not exact in general).

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to reduce.
        decrements (tuple[int, ...]): Degree decrement per direction. All
            values must be non-negative; at least one must be positive.  No
            decrement may exceed the current degree in that direction.

    Returns:
        ~pantr.bezier.Bezier: New Bézier with reduced degrees and updated
        control points.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    degrees = bezier.degree

    for d in range(bezier.dim):
        dec = decrements[d]
        if dec == 0:
            continue

        p = degrees[d]

        pts_2d, trailing_shape = _flatten_along_axis(ctrl, d)
        new_pts_2d = _degree_reduce_bezier_1d_core(p, pts_2d, dec)
        ctrl = _unflatten_along_axis(new_pts_2d, trailing_shape, d)

        # Update degrees for subsequent iterations.
        degrees = (*degrees[:d], p - dec, *degrees[d + 1 :])

    return BezierCls(ctrl, is_rational=bezier.is_rational)


def _bernstein_gram_matrix_1d(degree: int) -> npt.NDArray[np.float64]:
    r"""Build the degree-``n`` univariate Bernstein mass (Gram) matrix on :math:`[0, 1]`.

    The entries are the exact inner products of the Bernstein basis functions,
    given by the closed form of Farouki & Rajan (*Computer Aided Geometric
    Design* 5, 1988):

    .. math::

        G_{ij} = \int_0^1 B_{i,n}(x) B_{j,n}(x)\,dx
        = \frac{1}{2n+1}\,
          \frac{\binom{n}{i}\binom{n}{j}}{\binom{2n}{i+j}},
        \qquad 0 \le i, j \le n .

    Args:
        degree (int): Polynomial degree ``n`` (``>= 0``).

    Returns:
        npt.NDArray[np.float64]: Symmetric ``(n + 1, n + 1)`` Gram matrix.
    """
    n = degree
    binom_n = np.array([math.comb(n, i) for i in range(n + 1)], dtype=np.float64)
    binom_2n = np.array([math.comb(2 * n, k) for k in range(2 * n + 1)], dtype=np.float64)

    idx = np.arange(n + 1)
    numerator = np.outer(binom_n, binom_n)
    denominator = binom_2n[idx[:, None] + idx[None, :]]
    gram: npt.NDArray[np.float64] = numerator / denominator / (2.0 * n + 1.0)
    return gram


def _squared_l2_norm(
    coeffs: npt.NDArray[np.floating[Any]],
) -> float:
    r"""Compute the squared :math:`L^2` norm of a Bernstein polynomial on :math:`[0, 1]^d`.

    A polynomial with Bernstein coefficients ``c`` has squared norm equal to the
    Bernstein-Gram quadratic form :math:`\lVert p\rVert_2^2 = c^\top G\, c`,
    where :math:`G` is the Bernstein mass matrix.  For a tensor-product basis the
    mass matrix factorises as a Kronecker product of the univariate Gram matrices
    :math:`G = G^{(0)} \otimes \cdots \otimes G^{(d-1)}`, so the quadratic form is
    evaluated by contracting each univariate :math:`G^{(k)}` against the
    coefficient tensor along its axis and taking the final inner product with the
    coefficients.

    The closed-form univariate Gram entries are due to Farouki & Rajan (*Computer
    Aided Geometric Design* 5, 1988); see :func:`_bernstein_gram_matrix_1d`.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients
            (any shape).

    Returns:
        float: The squared :math:`L^2` norm ``||p||_2^2``.
    """
    c = coeffs.astype(np.float64, copy=False)

    # Apply G = G^(0) (x) ... (x) G^(d-1) by contracting one axis at a time.
    g_c = c
    for axis in range(c.ndim):
        gram = _bernstein_gram_matrix_1d(c.shape[axis] - 1)
        g_c = np.moveaxis(np.tensordot(gram, g_c, axes=([1], [axis])), 0, axis)

    # Quadratic form c^T (G c); the analytic value is non-negative, so guard
    # against a small negative result from floating-point round-off.
    return abs(float(np.sum(c * g_c)))


def _minimize_degree_bezier(
    bezier: Bezier,
    tol: float | None = None,
) -> Bezier:
    """Automatically reduce the degree of a Bézier while maintaining accuracy.

    Greedy, direction-by-direction degree reduction.  For each parametric
    direction the degree is lowered by one as long as the candidate reduction is
    accurate enough: the trial curve is degree-reduced (least squares) and then
    re-elevated back to the current degree, and the round-trip relative
    :math:`L^2` error is compared against ``tol``.  The first rejected trial in a
    direction stops further reduction in that direction (the error is
    monotonically non-decreasing as the degree drops, so there is nothing to gain
    by continuing).  For vector-valued Bézier all rank components are combined
    into a single error measure, so a reduction is accepted only when every
    component is preserved.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to simplify.
        tol (float | None): Relative tolerance for accepting a degree
            reduction.  If *None*, uses
            :data:`_AUTO_REDUCTION_TOL_FACTOR` ``* eps``.

    Returns:
        ~pantr.bezier.Bezier: A new Bézier with the lowest degree that
        preserves accuracy within ``tol``.  If no reduction is possible,
        returns a copy of the input.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.floating[Any]] = bezier.control_points  # (*orders, rank)
    rank = ctrl.shape[-1]

    if tol is None:
        tol = _AUTO_REDUCTION_TOL_FACTOR * float(np.finfo(ctrl.dtype).eps)

    if tol <= 0.0:
        return BezierCls(ctrl.copy(), is_rational=bezier.is_rational)

    def total_squared_norm(arr: npt.NDArray[np.floating[Any]]) -> float:
        """Sum the squared L2 norms of every rank component of *arr*."""
        return float(sum(_squared_l2_norm(arr[..., r]) for r in range(rank)))

    result = ctrl
    changed = False

    for dim in range(bezier.dim):
        # Each direction shrinks until a reduction is rejected.
        while result.shape[dim] >= 2:  # noqa: PLR2004
            degree = result.shape[dim] - 1

            # Reduce by one along `dim` (all rank components together) ...
            flat, trailing = _flatten_along_axis(result, dim)
            reduced = _unflatten_along_axis(
                _degree_reduce_bezier_1d_core(degree, flat, 1), trailing, dim
            )

            # ... then re-elevate so the trial can be compared to `result`.
            flat_reduced, trailing_reduced = _flatten_along_axis(reduced, dim)
            elevated = _unflatten_along_axis(
                _degree_elevate_bezier_1d_core(degree - 1, flat_reduced, 1),
                trailing_reduced,
                dim,
            )

            # Relative round-trip L2 error across all components.
            diff_norm2 = total_squared_norm(elevated - result)
            orig_norm2 = total_squared_norm(result)
            rel_error = math.sqrt(diff_norm2)
            if orig_norm2 > 0.0:
                rel_error /= math.sqrt(orig_norm2)

            if rel_error >= tol:
                break
            result = reduced
            changed = True

    if not changed:
        return BezierCls(ctrl.copy(), is_rational=bezier.is_rational)
    return BezierCls(result, is_rational=bezier.is_rational)
