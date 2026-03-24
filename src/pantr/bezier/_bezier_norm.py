r"""Bézier L2 norm computation.

This module provides :func:`_squared_l2_norm_bezier` and
:func:`_l2_norm_bezier`, which compute the squared L2 norm and L2 norm of a
non-rational Bézier over the unit hypercube :math:`[0, 1]^D`.

The computation uses the Bernstein mass matrix, which is separable as a
Kronecker product of 1D mass matrices.  Each 1D mass matrix is applied via
:func:`numpy.tensordot` along its corresponding spatial axis, avoiding
formation of the full nD mass matrix.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from . import Bezier


def _bernstein_mass_matrix_1d(
    degree: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Build the 1D Bernstein mass matrix for a given degree.

    The mass matrix ``M`` has entries:

    .. math::

        M_{ij} = \frac{\binom{n}{i}\,\binom{n}{j}}
                      {(2n + 1)\,\binom{2n}{i + j}}

    where :math:`n` = *degree*.  This matrix is symmetric and positive
    semi-definite.

    Args:
        degree (int): Polynomial degree (:math:`n \ge 0`).
        dtype (npt.DTypeLike): Floating-point dtype for the output.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Symmetric mass matrix of shape
        ``(degree + 1, degree + 1)``.
    """
    n = degree
    size = n + 1
    binom_n = np.array([math.comb(n, i) for i in range(size)], dtype=dtype)
    inv_binom_2n = np.array([1.0 / math.comb(2 * n, k) for k in range(2 * n + 1)], dtype=dtype)
    i_idx = np.arange(size)
    sum_idx = i_idx[:, None] + i_idx[None, :]  # (size, size)
    result: npt.NDArray[np.float32 | np.float64] = (
        binom_n[:, None] * binom_n[None, :] * inv_binom_2n[sum_idx]
    ) / (2 * n + 1)
    return result


def _squared_l2_norm_bezier(
    bezier: Bezier,
) -> np.floating[Any]:
    r"""Compute the squared L2 norm of a non-rational Bézier over :math:`[0, 1]^D`.

    .. math::

        \|p\|^2 = \int_{[0,1]^D} \|p(x)\|^2 \, dx

    For scalar Bézier this equals :math:`c^T M c` where :math:`M` is the
    Bernstein mass matrix.  For vector-valued Bézier (rank > 1) it equals
    :math:`\sum_r c_r^T M c_r`.

    The mass matrix is separable (Kronecker product of 1D matrices) and is
    applied via successive :func:`numpy.tensordot` contractions along each
    spatial axis.

    Args:
        bezier (~pantr.bezier.Bezier): A non-rational Bézier.

    Returns:
        np.floating[Any]: The squared L2 norm (non-negative up to
        floating-point round-off).

    Raises:
        ValueError: If the Bézier is rational.
    """
    if bezier.is_rational:
        raise ValueError("L2 norm is not supported for rational Bézier.")

    ctrl = bezier.control_points
    dim = bezier.dim
    degrees = bezier.degree
    dtype = bezier.dtype

    mass_1d = [_bernstein_mass_matrix_1d(degrees[d], dtype) for d in range(dim)]

    mc = ctrl
    for d in range(dim):
        mc = np.tensordot(mass_1d[d], mc, axes=([1], [d]))
        mc = np.moveaxis(mc, 0, d)

    result: np.floating[Any] = np.sum(ctrl * mc)
    return result


def _l2_norm_bezier(
    bezier: Bezier,
) -> np.floating[Any]:
    r"""Compute the L2 norm of a non-rational Bézier over :math:`[0, 1]^D`.

    .. math::

        \|p\| = \sqrt{\int_{[0,1]^D} \|p(x)\|^2 \, dx}

    Args:
        bezier (~pantr.bezier.Bezier): A non-rational Bézier.

    Returns:
        np.floating[Any]: The L2 norm (non-negative scalar).

    Raises:
        ValueError: If the Bézier is rational.
    """
    result: np.floating[Any] = np.sqrt(np.abs(_squared_l2_norm_bezier(bezier)))
    return result
