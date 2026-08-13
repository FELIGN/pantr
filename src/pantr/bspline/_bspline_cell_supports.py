"""Per-cell control-point support of a tensor-product B-spline space.

Layer-2 helper behind :meth:`~pantr.bspline.BsplineSpace.cell_supports`: it does all
the validation and the vectorized index arithmetic, so the Layer-1 method stays a
thin delegation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._bspline_space_nd import BsplineSpace


def _cell_supports_impl(space: BsplineSpace, cell_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return the control points supported on each of the given cells.

    Args:
        space (BsplineSpace): The tensor-product space.
        cell_ids (npt.ArrayLike): Flat cell ids, C-order over
            :attr:`~pantr.bspline.BsplineSpace.num_intervals`.

    Returns:
        npt.NDArray[np.int64]: Array of shape ``(n, W)`` with
        ``W = prod(degree + 1)``. Column ``k`` corresponds to the local offset whose
        C-order index within ``degrees + 1`` is ``k``, and holds the flat C-order
        control-point id.

    Raises:
        ValueError: If any direction is periodic, ``cell_ids`` is not of integer
            dtype, is not one-dimensional, or holds an id outside
            ``[0, num_total_intervals)``.
    """
    periodic = [k for k, one_d in enumerate(space.spaces) if one_d.periodic]
    if periodic:
        raise ValueError(
            f"cell_supports: periodic B-spline spaces are not supported; "
            f"direction(s) {periodic} are periodic."
        )

    ids = np.asarray(cell_ids)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if ids.ndim != 1:
        raise ValueError(f"cell_ids must be one-dimensional; got shape {ids.shape}")
    if ids.size and not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"cell_ids must have an integer dtype; got {ids.dtype}")

    num_intervals = space.num_intervals
    num_basis = space.num_basis
    degrees = space.degrees
    width = int(np.prod([degree + 1 for degree in degrees]))

    if ids.size == 0:
        return np.empty((0, width), dtype=np.int64)

    total = space.num_total_intervals
    if int(ids.min()) < 0 or int(ids.max()) >= total:
        raise ValueError(
            f"cell_ids must lie in [0, {total}); got range [{int(ids.min())}, {int(ids.max())}]"
        )

    multi = np.unravel_index(ids.astype(np.int64), num_intervals)
    # One (width,) offset vector per direction, C-order over degrees + 1, so that
    # column k of the result always means the same local offset.
    offset_grids = np.meshgrid(*[np.arange(degree + 1) for degree in degrees], indexing="ij")
    per_direction = tuple(
        space.spaces[direction].first_basis_per_interval()[multi[direction]][:, None]
        + offset_grids[direction].reshape(1, width)
        for direction in range(space.dim)
    )
    return np.ravel_multi_index(per_direction, num_basis).astype(np.int64)
