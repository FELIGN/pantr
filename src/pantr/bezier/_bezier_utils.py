"""Shared Layer 2 helpers for the ``bezier`` package.

Utilities that allocate output arrays and invoke Layer 3 Bernstein kernels.
These helpers are private and not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._numba_compat import wait_for_jit_warmup
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
    validated: ``degree >= 0``, ``pts`` is 1-D, and ``pts.dtype`` matches
    ``dtype`` exactly. Passing a mismatched dtype produces undefined behaviour
    inside the Numba kernel.

    Args:
        degree (int): Polynomial degree.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points, shape
            ``(n_pts,)``.
        dtype (npt.DTypeLike): Target floating-point dtype for the output.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Basis values of shape
        ``(n_pts, degree + 1)``.
    """
    # `__init__.py` compiles the kernels on a background thread, and numba's default
    # workqueue layer is not safe against a concurrent `parallel=True` call from another
    # thread: the process *aborts* rather than raising. This helper is the funnel for the
    # Bernstein path in `pantr.bezier`, so the barrier here also covers `interpolate_bezier`
    # and `fit_bezier`, which reach no other `parallel=True` kernel. It does *not* cover the
    # barriers in `_bezier_eval`: their `dim == 1` branches reach `evaluate_kernel()` and
    # `evaluate_deriv_kernel()`, which never come through here, so all three are needed.
    wait_for_jit_warmup()

    basis = np.empty((pts.shape[0], degree + 1), dtype=dtype)
    _tabulate_Bernstein_basis_1D_core(np.int32(degree), pts, basis)
    return basis
