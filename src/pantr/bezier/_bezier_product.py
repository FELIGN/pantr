"""Bézier pointwise product.

This module provides :func:`_multiply_bezier`, which computes the pointwise
product of two Bézier objects via the Bernstein product formula.  The result is a
Bézier of degree ``p_d + q_d`` per direction.

That degree represents the product **exactly**: no approximation is made in
choosing it, unlike degree reduction.  The coefficients are another matter, and
this module used to call them exact too.  Each is a binomial-weighted sum formed
in floating point, so each carries the roundings of that sum; the claim of
exactness is about the representation and not about the computed numbers.
``cpp/include/pantr/bezier/product.hpp`` states the same distinction for the C++
port, and ``tests/parity/test_bezier_product.py`` measures how far the two
backends' coefficients agree.

The core Bernstein product formulas (:func:`_bernstein_product_coefficients`
and :func:`_bernstein_product_coefficients_nd`) are also used by the B-spline
product modules, and are **not** dispatched: they stay pure numpy on either
backend.  ``design/cross_backend_types.md`` records why, and the C++ port
reproduces them rather than routing through the scalar product kernel they
resemble.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._control_points_utils import _append_unit_weight_column
from ._bezier_backend import multiply_kernel

if TYPE_CHECKING:
    from . import Bezier


def _bernstein_product_coefficients(
    b_f: npt.NDArray[np.float32 | np.float64],
    b_g: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Compute Bézier control points of the product of two Bézier segments.

    Applies the Bernstein product formula element-wise over the rank axis:

    .. math::

        d_k = \frac{1}{\binom{p+q}{k}} \sum_{i=\max(0,k-q)}^{\min(p,k)}
              \binom{p}{i} \binom{q}{k-i}\, b_f[i] \cdot b_g[k-i]

    Args:
        b_f (npt.NDArray[np.float32 | np.float64]): Control points of the first
            Bézier segment, shape ``(p+1, rank)``.
        b_g (npt.NDArray[np.float32 | np.float64]): Control points of the second
            Bézier segment, shape ``(q+1, rank)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Product Bézier control points of
        shape ``(p+q+1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    p, q = b_f.shape[0] - 1, b_g.shape[0] - 1
    r = p + q
    dtype = b_f.dtype

    cp = np.array([math.comb(p, i) for i in range(p + 1)], dtype=dtype)
    cq = np.array([math.comb(q, j) for j in range(q + 1)], dtype=dtype)
    inv_cr = np.array([1.0 / math.comb(r, k) for k in range(r + 1)], dtype=dtype)

    i_idx = np.arange(p + 1)
    j_idx = np.arange(q + 1)
    k_mat = i_idx[:, None] + j_idx[None, :]  # (p+1, q+1)
    coeff = cp[:, None] * cq[None, :] * inv_cr[k_mat]  # (p+1, q+1)

    products = coeff[:, :, None] * b_f[:, None, :] * b_g[None, :, :]

    d = np.zeros((r + 1, b_f.shape[1]), dtype=dtype)
    np.add.at(d, k_mat.ravel(), products.reshape(-1, b_f.shape[1]))
    return d


def _bernstein_product_coefficients_nd(
    b_f: npt.NDArray[np.float32 | np.float64],
    b_g: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Compute nD Bézier control points of the product of two Bézier patches.

    Applies the tensor-product Bernstein product formula element-wise over the
    rank axis.  For each multi-index :math:`\gamma`:

    .. math::

        h_\gamma = \frac{1}{\prod_d \binom{r_d}{\gamma_d}}
                   \sum_{\alpha} \prod_d \binom{p_d}{\alpha_d}
                   \binom{q_d}{\gamma_d - \alpha_d}\,
                   b_f[\alpha]\, b_g[\gamma - \alpha]

    where :math:`r_d = p_d + q_d`.

    The nD convolution is computed via the accumulation strategy: for each
    multi-index :math:`\alpha` of ``b_f``, the weighted product is accumulated
    into the output at the offset :math:`[\alpha : \alpha + q + 1]`.

    Args:
        b_f (npt.NDArray[np.float32 | np.float64]): Control points of the first
            Bézier patch, shape ``(p_0+1, ..., p_{D-1}+1, rank)``.
        b_g (npt.NDArray[np.float32 | np.float64]): Control points of the second
            Bézier patch, shape ``(q_0+1, ..., q_{D-1}+1, rank)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Product Bézier control points
        of shape ``(p_0+q_0+1, ..., p_{D-1}+q_{D-1}+1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    ndim = b_f.ndim - 1
    p = tuple(s - 1 for s in b_f.shape[:ndim])
    q = tuple(s - 1 for s in b_g.shape[:ndim])
    r = tuple(pi + qi for pi, qi in zip(p, q, strict=True))
    dtype = b_f.dtype
    rank = b_f.shape[-1]

    # Precompute per-direction binomial coefficients.
    binom_p = [
        np.array([math.comb(p[d], i) for i in range(p[d] + 1)], dtype=dtype) for d in range(ndim)
    ]
    binom_q = [
        np.array([math.comb(q[d], j) for j in range(q[d] + 1)], dtype=dtype) for d in range(ndim)
    ]
    inv_binom_r = [
        np.array([1.0 / math.comb(r[d], k) for k in range(r[d] + 1)], dtype=dtype)
        for d in range(ndim)
    ]

    # Build weight arrays via outer product of per-direction binomial vectors.
    w_f = binom_p[0]
    for d in range(1, ndim):
        w_f = np.multiply.outer(w_f, binom_p[d])
    w_g = binom_q[0]
    for d in range(1, ndim):
        w_g = np.multiply.outer(w_g, binom_q[d])
    inv_w_r = inv_binom_r[0]
    for d in range(1, ndim):
        inv_w_r = np.multiply.outer(inv_w_r, inv_binom_r[d])

    # Weight the control points.
    weighted_f = w_f[..., np.newaxis] * b_f
    weighted_g = w_g[..., np.newaxis] * b_g

    # nD convolution via accumulation.
    result_shape = (*tuple(ri + 1 for ri in r), rank)
    h_conv = np.zeros(result_shape, dtype=dtype)

    for alpha in itertools.product(*(range(pi + 1) for pi in p)):
        slices = tuple(slice(a, a + qi + 1) for a, qi in zip(alpha, q, strict=True))
        h_conv[slices] = h_conv[slices] + weighted_f[alpha] * weighted_g

    # Normalize by inverse product binomial coefficients.
    return inv_w_r[..., np.newaxis] * h_conv


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _binomial_tables(
    order: int, dtype: npt.DTypeLike
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Tabulate ``C(n, k)`` and ``1 / C(n, k)`` for ``n <= order``, in one dtype.

    The two tables the C++ product and composition are handed instead of assembling
    their own; ``cpp/include/pantr/bezier/product.hpp`` argues why they cross as data
    rather than being computed natively.  The short of it: :func:`math.comb` is exact
    at any size and the C++ exact-integer recurrence stops at ``61``, so a native
    assembly would import an envelope into an operation that does not have one.

    Every entry is built by **exactly the expression the numpy oracle uses**, which is
    what makes the comparison a comparison of the product rather than of two table
    assemblies: ``np.array([math.comb(n, k) ...], dtype=dtype)`` for the coefficients
    and ``np.array([1.0 / math.comb(n, k) ...], dtype=dtype)`` for the reciprocals.

    :mod:`~pantr.bezier._bezier_compose` reaches its binomials through a third
    expression, ``float(math.comb(m, i))`` multiplied into an array, and that value is
    the same as this table's for every argument.  A ``float32`` table is reached there
    by rounding the exact integer to ``float64`` and then to ``float32``, and here in
    one step; the two agree because ``float64`` carries 53 significand bits against
    ``float32``'s 24 and double rounding through an intermediate of at least
    ``2 * 24 + 2`` bits is equal to single rounding (Figueroa, *Yet another proof of
    the double rounding theorem*, 1995).  ``float64`` storage is the trivial case,
    where both spellings are one correctly rounded conversion.
    ``test_the_two_spellings_of_a_binomial_agree`` in
    ``tests/parity/test_bezier_product.py`` checks it over the whole range these
    operations reach, because the argument is a theorem about a hypothesis this code
    has to satisfy rather than about this code.

    Entries with ``k > n`` are zero and are never read.

    Args:
        order (int): Largest upper index to tabulate, so the tables are
            ``(order + 1, order + 1)``.
        dtype (npt.DTypeLike): Storage format, matching the Bézier's.

    Returns:
        tuple: The coefficient table and the reciprocal table, both C-contiguous.
    """
    size = order + 1
    binomials = np.zeros((size, size), dtype=dtype)
    inverse_binomials = np.zeros((size, size), dtype=dtype)
    for n in range(size):
        exact = [math.comb(n, k) for k in range(n + 1)]
        binomials[n, : n + 1] = np.array(exact, dtype=dtype)
        inverse_binomials[n, : n + 1] = np.array([1.0 / c for c in exact], dtype=dtype)
    return binomials, inverse_binomials


def _multiply_bezier(a: Bezier, b: Bezier) -> Bezier:
    """Compute the pointwise product of two Bézier objects.

    Validates that both operands have matching dimension, dtype, and rank, then
    hands the pair to the active backend.  The result has degree ``p_d + q_d`` per
    direction, which represents the product exactly; its coefficients are a
    binomial-weighted sum in floating point and carry that sum's roundings.  See the
    module docstring.

    Args:
        a (~pantr.bezier.Bezier): First operand.
        b (~pantr.bezier.Bezier): Second operand.

    Returns:
        ~pantr.bezier.Bezier: Product Bézier of degree ``p_d + q_d`` per
        direction.

    Raises:
        ValueError: If the operands have different dimensions, dtypes, or ranks.
    """
    if a.dim != b.dim:
        raise ValueError(f"Operands must have the same dimension. Got {a.dim} and {b.dim}.")
    if a.dtype != b.dtype:
        raise ValueError(f"Operands must have the same dtype. Got {a.dtype} and {b.dtype}.")
    if a.rank != b.rank:
        raise ValueError(f"Operands must have the same rank. Got {a.rank} and {b.rank}.")

    return multiply_kernel()(a, b)


def _multiply_python(a: Bezier, b: Bezier) -> Bezier:
    """Multiply with NumPy: the oracle for the port.

    Args:
        a (~pantr.bezier.Bezier): First operand.
        b (~pantr.bezier.Bezier): Second operand.

    Returns:
        ~pantr.bezier.Bezier: The product.

    Note:
        No input validation is performed here; Layer 2 did it above the branch.
    """
    if a.is_rational or b.is_rational:
        return _multiply_rational(a, b)
    return _multiply_nonrational(a, b)


def _multiply_nonrational(a: Bezier, b: Bezier) -> Bezier:
    """Compute the product of two non-rational Bézier objects.

    Args:
        a (~pantr.bezier.Bezier): First operand (non-rational).
        b (~pantr.bezier.Bezier): Second operand (non-rational).

    Returns:
        ~pantr.bezier.Bezier: Non-rational product Bézier.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    if a.dim == 1:
        new_ctrl = _bernstein_product_coefficients(a.control_points, b.control_points)
    else:
        new_ctrl = _bernstein_product_coefficients_nd(a.control_points, b.control_points)
    return BezierCls(new_ctrl, is_rational=False)


def _multiply_rational(a: Bezier, b: Bezier) -> Bezier:
    """Compute the product of two Bézier objects where at least one is rational.

    Promotes non-rational operands to rational (unit weights) and multiplies
    numerators and denominators independently.

    Args:
        a (~pantr.bezier.Bezier): First operand.
        b (~pantr.bezier.Bezier): Second operand.

    Returns:
        ~pantr.bezier.Bezier: Rational product Bézier.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    # Promote non-rational to rational with unit weights.
    a_ctrl = _ensure_rational_ctrl(a)
    b_ctrl = _ensure_rational_ctrl(b)

    # Decompose: numerator (weighted coords) and denominator (weights).
    a_num = a_ctrl[..., :-1]
    a_den = a_ctrl[..., -1:]
    b_num = b_ctrl[..., :-1]
    b_den = b_ctrl[..., -1:]

    # Multiply numerators and denominators.
    if a.dim == 1:
        h_num = _bernstein_product_coefficients(a_num, b_num)
        h_den = _bernstein_product_coefficients(a_den, b_den)
    else:
        h_num = _bernstein_product_coefficients_nd(a_num, b_num)
        h_den = _bernstein_product_coefficients_nd(a_den, b_den)

    result_ctrl = np.concatenate([h_num, h_den], axis=-1)
    return BezierCls(result_ctrl, is_rational=True)


def _ensure_rational_ctrl(
    bezier: Bezier,
) -> npt.NDArray[np.float32 | np.float64]:
    """Return control points in homogeneous form, adding unit weights if needed.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier (rational or non-rational).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Control points with shape
        ``(..., rank+1)`` where the last column is the weight.
    """
    ctrl = bezier.control_points
    if bezier.is_rational:
        return ctrl
    return _append_unit_weight_column(ctrl)
