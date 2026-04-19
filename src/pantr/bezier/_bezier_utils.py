"""Shared Layer 2 helpers for the ``bezier`` package.

Utilities that allocate and populate the output arrays expected by the
Layer 3 Bernstein kernels. These helpers are private and not part of the
public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core

if TYPE_CHECKING:
    import numpy.typing as npt


def _tabulate_bernstein_1d_fast(
    degree: int,
    pts: npt.NDArray[np.float32 | np.float64],
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Allocate an output array and evaluate the Bernstein basis of given degree.

    Thin Layer 2 wrapper around the Layer 3 kernel
    :func:`_tabulate_Bernstein_basis_1D_core`. Inputs are assumed already
    validated (degree >= 0, ``pts`` 1-D and of the given ``dtype``).

    Args:
        degree (int): Polynomial degree.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points, shape
            ``(n_pts,)``.
        dtype (npt.DTypeLike): Target floating-point dtype for the output.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Basis values of shape
        ``(n_pts, degree + 1)``.
    """
    basis = np.empty((pts.shape[0], degree + 1), dtype=dtype)
    _tabulate_Bernstein_basis_1D_core(np.int32(degree), pts, basis)
    return basis
