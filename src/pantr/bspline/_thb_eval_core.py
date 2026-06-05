"""Numba kernels for THB-spline evaluation (Layer 3).

Pure computation for the hierarchical-eval hot path, decorated with
:func:`~pantr._numba_compat.nb_jit`.  The 1D B-spline basis values are produced by the
existing jitted tabulation; this module only fuses the per-function tensor-product
gather-and-product for the untruncated (common) functions on a cell, which otherwise
incurs heavy NumPy temporary/dispatch overhead for the tiny per-cell arrays of FEM
assembly (many cells, few points each).

Main exports:

- :func:`_combine_tp_values`: per-cell tensor-product combine of pre-tabulated 1D values.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(nopython=True, cache=True)
def _combine_tp_values(
    vals: npt.NDArray[np.float64],
    first_basis: npt.NDArray[np.int64],
    multis: npt.NDArray[np.int64],
    degrees: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Combine pre-tabulated 1D values into untruncated tensor-product function values.

    For each function ``f`` (a tensor-product B-spline with per-direction indices
    ``multis[f]``) and point ``p``, computes the product over directions ``k`` of the
    1D value ``vals[k, p, multis[f, k] - first_basis[k, p]]``; a factor is ``0`` when the
    local index falls outside ``[0, degrees[k]]`` (the function is not supported at the
    point in that direction); the inner ``k``-loop exits early on the first such direction
    via ``break`` — remaining directions are not evaluated.  This is the value of the mixed
    partial when ``vals`` holds per-direction derivative values (the partial of a product
    factorizes per direction).

    Args:
        vals (npt.NDArray[np.float64]): Per-direction 1D values, shape
            ``(dim, n_pts, max_order)``, zero-padded to ``max_order = max(degrees) + 1``.
        first_basis (npt.NDArray[np.int64]): Per-direction first nonzero basis index per
            point, shape ``(dim, n_pts)``.
        multis (npt.NDArray[np.int64]): Per-function direction indices, shape
            ``(n_funcs, dim)``.
        degrees (npt.NDArray[np.int64]): Per-direction polynomial degrees, shape ``(dim,)``.

    Returns:
        npt.NDArray[np.float64]: Function values of shape ``(n_pts, n_funcs)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.THBSplineSpace.tabulate_basis` instead.
    """
    dim = vals.shape[0]
    n_pts = vals.shape[1]
    n_funcs = multis.shape[0]
    out = np.empty((n_pts, n_funcs), dtype=np.float64)
    for f in range(n_funcs):
        for p in range(n_pts):
            product = 1.0
            for k in range(dim):
                local = multis[f, k] - first_basis[k, p]
                if local < 0 or local > degrees[k]:
                    product = 0.0
                    break
                product *= vals[k, p, local]
            out[p, f] = product
    return out
