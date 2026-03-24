"""Bézier composition.

This module provides :func:`_compose_bezier`, which computes the exact
composition of two non-rational Bézier objects: ``outer(inner(t))``.

The algorithm decomposes each scalar component of the inner Bézier into
Bernstein basis evaluations of the outer's degree, then combines them
with the outer's control points via a multi-index weighted summation.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bezier_core import _scalar_bernstein_product_1d_core
from ._bezier_product import (
    _bernstein_product_coefficients,
    _bernstein_product_coefficients_nd,
)

if TYPE_CHECKING:
    from . import Bezier


def _compose_bezier(outer: Bezier, inner: Bezier) -> Bezier:
    """Compose two non-rational Bézier objects: ``outer(inner(t))``.

    Validates that both operands are non-rational, that the inner's rank
    matches the outer's parametric dimension, and that their dtypes match,
    then dispatches to the composition implementation.

    Args:
        outer (~pantr.bezier.Bezier): The outer Bézier (the mapping being
            composed). Must be non-rational.
        inner (~pantr.bezier.Bezier): The inner Bézier (the reparametrization).
            Must be non-rational and satisfy ``inner.rank == outer.dim``.

    Returns:
        ~pantr.bezier.Bezier: Composed Bézier with ``dim = inner.dim``,
        ``rank = outer.rank``, and degree ``sum(outer.degree) * inner.degree[s]``
        in each parametric direction ``s``.

    Raises:
        TypeError: If either operand is rational.
        ValueError: If ``inner.rank != outer.dim``.
        ValueError: If the operands have different dtypes.
    """
    if outer.is_rational:
        raise TypeError("Composition is not supported for rational Béziers (outer is rational).")
    if inner.is_rational:
        raise TypeError("Composition is not supported for rational Béziers (inner is rational).")
    if inner.rank != outer.dim:
        raise ValueError(
            f"Inner Bézier rank ({inner.rank}) must equal outer Bézier "
            f"parametric dimension ({outer.dim})."
        )
    if outer.dtype != inner.dtype:
        raise ValueError(f"Operands must have the same dtype. Got {outer.dtype} and {inner.dtype}.")

    return _compose_impl(outer, inner)


def _compose_impl(outer: Bezier, inner: Bezier) -> Bezier:
    """Compute the composition of two non-rational Bézier objects.

    Implements the Bernstein basis decomposition algorithm:

    1. For each parametric direction ``d`` of ``outer``, extract the ``d``-th
       scalar component of ``inner`` and compute the Bernstein basis evaluations
       ``B_i^{m_d}(g_d)`` for ``i = 0, ..., m_d``.
    2. Iterate over all multi-indices of the outer control points, compute the
       tensor product of Bernstein bases, and accumulate weighted by the
       control point values.

    Args:
        outer (~pantr.bezier.Bezier): The outer Bézier (non-rational, validated).
        inner (~pantr.bezier.Bezier): The inner Bézier (non-rational, validated).

    Returns:
        ~pantr.bezier.Bezier: The composed Bézier.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_compose_bezier` instead.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    dim_f = outer.dim
    dim_g = inner.dim
    deg_f = outer.degree
    rank = outer.rank
    dtype = outer.dtype

    # Choose product function based on inner dimensionality.
    use_1d_kernel = dim_g == 1

    # Step 1: Compute Bernstein bases for each direction of outer.
    all_bases: list[list[npt.NDArray[np.float32 | np.float64]]] = []
    for d in range(dim_f):
        g_d_ctrl = _extract_scalar_component(inner.control_points, d)
        bases_d = _compute_bernstein_bases(g_d_ctrl, deg_f[d], use_1d_kernel)
        all_bases.append(bases_d)

    # Step 2: Compute result degree and accumulate.
    sum_deg_f = sum(deg_f)
    result_shape: tuple[int, ...] = tuple(sum_deg_f * n + 1 for n in inner.degree)

    result_ctrl = np.zeros((*result_shape, rank), dtype=dtype)

    outer_ctrl = outer.control_points
    for multi_idx in itertools.product(*(range(m + 1) for m in deg_f)):
        coef = outer_ctrl[multi_idx]  # shape (rank,)

        # Compute tensor product of Bernstein bases across all directions.
        basis = all_bases[0][multi_idx[0]]
        for d in range(1, dim_f):
            basis = _product_fn(basis, all_bases[d][multi_idx[d]], use_1d_kernel)

        # Accumulate: broadcast coef (rank,) with basis (*result_shape, 1).
        result_ctrl += coef * basis

    return BezierCls(result_ctrl, is_rational=False)


def _extract_scalar_component(
    ctrl: npt.NDArray[np.float32 | np.float64],
    component: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Extract a scalar component from control points, keeping the rank axis.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points with shape
            ``(*degrees_plus_1, rank)``.
        component (int): Index of the component to extract.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Scalar control points with shape
        ``(*degrees_plus_1, 1)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return ctrl[..., component : component + 1]


def _compute_bernstein_bases(
    g_ctrl: npt.NDArray[np.float32 | np.float64],
    degree: int,
    use_1d_kernel: bool,
) -> list[npt.NDArray[np.float32 | np.float64]]:
    r"""Compute the Bernstein basis evaluations ``B_i^m(g)`` for ``i = 0, ..., m``.

    Given a scalar Bézier ``g`` and a target degree ``m``, computes
    ``B_i^m(g) = \binom{m}{i}\, g^i\, (1 - g)^{m - i}`` for each ``i``.

    The computation uses pre-computed powers of ``g`` and ``(1 - g)`` via
    iterative Bernstein products.

    Args:
        g_ctrl (npt.NDArray[np.float32 | np.float64]): Control points of a
            scalar Bézier with shape ``(*degrees_plus_1, 1)``.
        degree (int): Target Bernstein degree (``m >= 0``).
        use_1d_kernel (bool): Whether to use the 1D Numba kernel for products.

    Returns:
        list[npt.NDArray[np.float32 | np.float64]]: List of ``degree + 1``
        arrays, each with shape ``(*result_degrees_plus_1, 1)``, representing
        the Bernstein basis evaluations in Bézier form.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if degree == 0:
        # B_0^0 = 1: constant (degree-0) Bézier with coefficient 1.
        shape = (*tuple(1 for _ in g_ctrl.shape[:-1]), 1)
        ones = np.ones(shape, dtype=g_ctrl.dtype)
        return [ones]

    # Compute (1 - g) control points (partition of unity property).
    one_minus_g_ctrl = 1.0 - g_ctrl

    # Compute powers: g^1, ..., g^degree and (1-g)^1, ..., (1-g)^degree.
    g_powers = _compute_scalar_powers(g_ctrl, degree, use_1d_kernel)
    one_minus_g_powers = _compute_scalar_powers(one_minus_g_ctrl, degree, use_1d_kernel)

    # Build Bernstein bases.
    bases: list[npt.NDArray[np.float32 | np.float64]] = [None] * (degree + 1)  # type: ignore[list-item]
    bases[0] = one_minus_g_powers[degree - 1]  # (1-g)^m
    bases[degree] = g_powers[degree - 1]  # g^m

    for i in range(1, degree):
        binom_coeff = math.comb(degree, i)
        prod = _product_fn(g_powers[i - 1], one_minus_g_powers[degree - i - 1], use_1d_kernel)
        bases[i] = float(binom_coeff) * prod

    return bases


def _compute_scalar_powers(
    g_ctrl: npt.NDArray[np.float32 | np.float64],
    max_power: int,
    use_1d_kernel: bool,
) -> list[npt.NDArray[np.float32 | np.float64]]:
    """Compute successive powers of a scalar Bézier: ``g^1, g^2, ..., g^{max_power}``.

    Each power is computed iteratively via Bernstein product:
    ``g^k = g^{k-1} * g``.

    Args:
        g_ctrl (npt.NDArray[np.float32 | np.float64]): Control points of a
            scalar Bézier with shape ``(*degrees_plus_1, 1)``.
        max_power (int): Maximum power to compute (``>= 1``).
        use_1d_kernel (bool): Whether to use the 1D Numba kernel for products.

    Returns:
        list[npt.NDArray[np.float32 | np.float64]]: List of ``max_power``
        arrays, where entry ``k`` (0-indexed) is ``g^{k+1}`` in Bézier form.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    powers: list[npt.NDArray[np.float32 | np.float64]] = [g_ctrl]
    for _ in range(1, max_power):
        powers.append(_product_fn(powers[-1], g_ctrl, use_1d_kernel))
    return powers


def _product_fn(
    a: npt.NDArray[np.float32 | np.float64],
    b: npt.NDArray[np.float32 | np.float64],
    use_1d_kernel: bool,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the Bernstein product of two scalar Bézier control point arrays.

    Dispatches to the 1D Numba kernel or the general nD NumPy implementation
    based on the ``use_1d_kernel`` flag.

    Args:
        a (npt.NDArray[np.float32 | np.float64]): Control points of the first
            scalar Bézier.
        b (npt.NDArray[np.float32 | np.float64]): Control points of the second
            scalar Bézier.
        use_1d_kernel (bool): If True, use the 1D Numba kernel (expects shape
            ``(n, 1)``). If False, use the nD NumPy implementation.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Product Bézier control points.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if use_1d_kernel:
        result_1d = _scalar_bernstein_product_1d_core(a[:, 0], b[:, 0])
        return result_1d[:, np.newaxis]
    if a.ndim - 1 == 1:
        return _bernstein_product_coefficients(a, b)
    return _bernstein_product_coefficients_nd(a, b)
