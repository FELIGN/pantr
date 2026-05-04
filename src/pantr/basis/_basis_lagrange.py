"""Lagrange basis function implementations.

This module provides functions for evaluating Lagrange basis polynomials
with various point distributions (equispaced, Gauss-Legendre, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.interpolate import BarycentricInterpolator

if TYPE_CHECKING:
    from . import LagrangeVariant
from ..quad import (
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_trapezoidal_1d,
)


def _get_lagrange_points(
    variant: LagrangeVariant, n_pts: int, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[np.float32 | np.float64]:
    """Get nodes for a Lagrange basis on [0, 1] for the given variant and size.

    Note that for Gauss-Lobatto-Legendre and Chebyshev second kind the
    number of points must be at least 2.

    Args:
        variant (LagrangeVariant): The variant of the Lagrange basis.
        n_pts (int): The number of points.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The nodes.

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    # Lazy import to avoid circular dependency

    variant_value = getattr(variant, "value", variant)
    if variant_value == "equispaces":
        return get_trapezoidal_1d(n_pts, dtype)[0]
    elif variant_value == "gauss_legendre":
        return get_gauss_legendre_1d(n_pts, dtype)[0]
    elif variant_value == "gauss_lobatto_legendre":
        return get_gauss_lobatto_legendre_1d(n_pts, dtype)[0]
    elif variant_value == "chebyshev_1st":
        return get_chebyshev_gauss_1st_kind_1d(n_pts, dtype)[0]
    else:  # "chebyshev_2nd"
        return get_chebyshev_gauss_2nd_kind_1d(n_pts, dtype)[0]


def _tabulate_lagrange_basis_1D_core(
    n: int,
    variant: LagrangeVariant,
    t: npt.NDArray[np.float32 | np.float64],
    B_normalized: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute Lagrange basis values into the provided normalized array.

    Args:
        n (int): Degree of the Lagrange basis. Must be non-negative.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        t (npt.NDArray[np.float32 | np.float64]): Normalized 1D array
            of evaluation points.
        B_normalized (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) where results will be written. Must have the correct
            shape and dtype matching t.
    """
    # Lazy import to avoid circular dependency

    n_pts = n + 1

    # Degree-zero: basis is constant 1 for all evaluation points and variants.
    if n == 0:
        B_normalized[:, 0] = 1.0
        return

    nodes = _get_lagrange_points(variant, n + 1, t.dtype)

    # Ensure nodes are strictly increasing for numerical robustness and consistency
    perm = np.argsort(nodes)
    nodes_sorted = nodes[perm]
    # Precompute inverse permutation: inv_perm[idx_in_sorted] = original_index
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n_pts, dtype=perm.dtype)

    for j in range(n_pts):
        # Set 1 at the position corresponding to node j in the sorted order
        y_sorted = np.zeros(n_pts, dtype=t.dtype)
        y_sorted[inv_perm[j]] = 1.0

        interpolator = BarycentricInterpolator(nodes_sorted, y_sorted)

        # SciPy may upcast internally; ensure we preserve input dtype.
        B_normalized[:, j] = np.asarray(interpolator(t), dtype=t.dtype)

    # Snap to exact Kronecker-delta at interpolation nodes to avoid tiny roundoff
    # off-diagonals (notably with float32 and Chebyshev nodes).
    eps: float
    if B_normalized.dtype == np.float32:
        eps = float(np.finfo(np.float32).eps) * 16.0
    else:
        eps = float(np.finfo(np.float64).eps) * 16.0
    diffs = np.abs(t[:, None] - nodes_sorted[None, :])
    matches = diffs <= eps
    if np.any(matches):
        row_has_match = np.any(matches, axis=1)
        if np.any(row_has_match):
            matched_k = np.argmax(matches, axis=1)
            rows = np.nonzero(row_has_match)[0]
            ks = matched_k[rows]
            cols = perm[ks]
            # Zero full matched rows then set the diagonal 1
            B_normalized[rows, :] = 0.0
            B_normalized[rows, cols] = 1.0


__all__ = ["_get_lagrange_points"]
