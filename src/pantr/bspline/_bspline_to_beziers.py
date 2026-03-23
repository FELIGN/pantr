"""Bezier decomposition of B-splines via extraction operators.

This module decomposes a B-spline into its constituent Bezier patches using
Bezier extraction operators applied direction by direction. The result is a
multidimensional array of :class:`~pantr.bezier.Bezier` objects following the
tensor-product interval structure.

The core Numba kernel :func:`_apply_bezier_extraction_1d_core` applies the
extraction operators for one parametric direction, transforming B-spline
control points into Bezier control points for all elements simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit, nb_prange

if TYPE_CHECKING:
    from . import Bspline


@nb_jit(nopython=True, cache=True, parallel=True)
def _apply_bezier_extraction_1d_core(
    extraction_ops: npt.NDArray[Any],
    ctrl: npt.NDArray[Any],
    out: npt.NDArray[Any],
) -> None:
    r"""Apply Bezier extraction operators to control points along one direction.

    For each element ``i``, computes the Bezier control points as
    ``C_i^T @ P_local`` where ``C_i`` is the extraction operator and
    ``P_local = ctrl[i : i + order, :]`` are the local B-spline control points.

    Args:
        extraction_ops (npt.NDArray[Any]): Bezier extraction operators of shape
            ``(n_elements, order, order)`` where ``order = degree + 1``.
        ctrl (npt.NDArray[Any]): B-spline control points of shape
            ``(n_basis, M)`` where ``M`` is the flattened trailing dimension.
            Must be C-contiguous.
        out (npt.NDArray[Any]): Pre-allocated output array of shape
            ``(n_elements * order, M)`` where Bezier control points are written.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_to_beziers_impl` instead.
    """
    n_elements = extraction_ops.shape[0]
    order = extraction_ops.shape[1]
    m = ctrl.shape[1]

    for elem in nb_prange(n_elements):
        c_t = extraction_ops[elem].T
        out_start = elem * order
        for i in range(order):
            for j in range(m):
                val = c_t.dtype.type(0.0)
                for k in range(order):
                    val += c_t[i, k] * ctrl[elem + k, j]
                out[out_start + i, j] = val


def _to_beziers_impl(bspline: Bspline) -> npt.NDArray[np.object_]:
    """Decompose a B-spline into Bezier patches via extraction operators.

    Converts periodic directions to open form, then applies Bezier extraction
    operators direction by direction using the standard moveaxis pattern. The
    resulting control point array is reshaped and split into individual
    :class:`~pantr.bezier.Bezier` objects.

    Args:
        bspline (Bspline): Input B-spline to decompose.

    Returns:
        npt.NDArray[np.object_]: Array of :class:`~pantr.bezier.Bezier` objects
        with shape ``(*num_intervals)`` following the tensor-product interval
        structure.
    """
    from ..bezier import Bezier  # noqa: PLC0415

    dim = bspline.dim

    # Convert periodic directions to open form.
    if any(s.periodic for s in bspline.space.spaces):
        bspline = bspline.to_open_bspline()

    ctrl = bspline.control_points
    num_intervals = bspline.space.num_intervals
    degrees = bspline.degree

    # Apply extraction operators direction by direction.
    for d in range(dim):
        space_1d = bspline.space.spaces[d]
        extraction_ops = space_1d.tabulate_Bezier_extraction_operators()
        n_el = num_intervals[d]
        order = degrees[d] + 1

        # Move direction d to axis 0, flatten trailing axes into a 2D matrix.
        moved_ctrl = np.moveaxis(ctrl, d, 0)
        orig_shape = moved_ctrl.shape
        pts_2d = moved_ctrl.reshape(orig_shape[0], -1)
        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        # Apply extraction: (n_basis, M) -> (n_el * order, M).
        out_2d = np.empty((n_el * order, pts_2d.shape[1]), dtype=pts_2d.dtype)
        _apply_bezier_extraction_1d_core(extraction_ops, pts_2d, out_2d)

        # Restore shape and move axis back.
        new_shape = (n_el * order, *orig_shape[1:])
        ctrl = np.moveaxis(out_2d.reshape(new_shape), 0, d)

    # ctrl now has shape (n_el_0*order_0, n_el_1*order_1, ..., rank).
    # Reshape to (n_el_0, order_0, n_el_1, order_1, ..., rank).
    intermediate_shape: list[int] = []
    for d in range(dim):
        intermediate_shape.append(num_intervals[d])
        intermediate_shape.append(degrees[d] + 1)
    intermediate_shape.append(ctrl.shape[-1])

    ctrl_reshaped = ctrl.reshape(intermediate_shape)

    # Transpose to (n_el_0, n_el_1, ..., order_0, order_1, ..., rank).
    perm = [2 * d for d in range(dim)] + [2 * d + 1 for d in range(dim)] + [2 * dim]
    ctrl_transposed = np.transpose(ctrl_reshaped, perm)

    # Build Bezier objects.
    result = np.empty(num_intervals, dtype=object)
    is_rational = bspline.is_rational

    for idx in np.ndindex(*num_intervals):
        bez_ctrl = np.ascontiguousarray(ctrl_transposed[idx])
        bez_ctrl.flags.writeable = False
        result[idx] = Bezier(bez_ctrl, is_rational=is_rational)

    return result


def _warmup_numba_functions() -> None:
    """Precompile Numba functions with float64 signatures for faster first call.

    This function triggers compilation of the Numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    extraction_ops = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
    ctrl = np.zeros((3, 2), dtype=np.float64)
    out = np.zeros((3, 2), dtype=np.float64)
    _apply_bezier_extraction_1d_core(extraction_ops, ctrl, out)


__all__ = [
    "_apply_bezier_extraction_1d_core",
    "_to_beziers_impl",
]
