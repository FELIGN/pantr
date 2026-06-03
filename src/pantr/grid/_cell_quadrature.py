"""Per-cell quadrature: map a reference rule on the unit cube onto grid cells.

A :class:`pantr.quad.QuadratureRule` lives on the reference unit cube
``[0, 1]^ndim``. :func:`cell_quadrature` pushes it forward onto a grid's cells
with the per-cell affine map ``T(u) = lo + (hi - lo) * u`` (see
:meth:`pantr.grid.Grid.reference_map`): points are mapped by ``T`` and weights
are scaled by the cell volume ``prod(hi - lo)`` (the Jacobian determinant of
``T``). The reference weights sum to ``1``, so the mapped weights of a cell sum
to that cell's volume and ``sum_i w_i f(x_i)`` approximates the integral of
``f`` over the cell.

This is the uncut/background-cell quadrature bridge: a consumer (for example,
ocelat) takes this rule for interior cells and substitutes its own cut-cell
rule on cells flagged as cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..quad import QuadratureRule
    from ._grid import Grid


def cell_quadrature(
    grid: Grid,
    rule: QuadratureRule,
    cells: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map a reference :class:`~pantr.quad.QuadratureRule` onto grid cells.

    For each selected cell, the rule's unit-cube points are affinely mapped into
    the cell box and its weights are scaled by the cell volume.

    Args:
        grid (Grid): The grid whose cells are integrated. ``grid.ndim`` must
            equal ``rule.ndim``.
        rule (QuadratureRule): Reference rule on ``[0, 1]^ndim``.
        cells (npt.ArrayLike | None): Cell ids to map the rule onto. ``None``
            (the default) selects every cell in id order. Otherwise a 1D
            integer array-like of ids, each in ``[0, num_cells)`` (a scalar is
            treated as a single id); order and duplicates are preserved.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: A pair
        ``(points, weights)`` where ``points`` has shape
        ``(num_selected, num_points, ndim)`` and ``weights`` has shape
        ``(num_selected, num_points)``, ordered to match ``cells``.

    Raises:
        ValueError: If ``rule.ndim != grid.ndim``, ``cells`` is not a 1D integer
            array-like, or any id is outside ``[0, num_cells)``.

    Note:
        Cell boxes for the whole grid are materialized once (an
        ``O(num_cells * ndim)`` temporary), then the requested subset is
        selected; this keeps the common all-cells path vectorized.
    """
    if rule.ndim != grid.ndim:
        raise ValueError(f"rule.ndim ({rule.ndim}) must match grid.ndim ({grid.ndim}).")

    cell_lo_all, cell_hi_all = grid._collect_cell_bounds()
    if cells is None:
        cell_lo, cell_hi = cell_lo_all, cell_hi_all
    else:
        ids = _resolve_cell_ids(cells, grid.num_cells)
        cell_lo, cell_hi = cell_lo_all[ids], cell_hi_all[ids]

    span = cell_hi - cell_lo
    points = cell_lo[:, None, :] + span[:, None, :] * rule.points[None, :, :]
    weights = rule.weights[None, :] * np.prod(span, axis=1)[:, None]
    return points, weights


def _resolve_cell_ids(cells: npt.ArrayLike, num_cells: int) -> npt.NDArray[np.intp]:
    """Validate and normalize a ``cells`` selector to a 1D index array.

    Args:
        cells (npt.ArrayLike): Cell-id selector (a scalar id or a 1D integer
            array-like).
        num_cells (int): Total number of cells; valid ids are ``[0, num_cells)``.

    Returns:
        npt.NDArray[np.intp]: A 1D array of cell ids suitable for fancy indexing.

    Raises:
        ValueError: If ``cells`` does not have an integer dtype, is not 1D, or
            contains an id outside ``[0, num_cells)``.
    """
    arr = np.atleast_1d(np.asarray(cells))
    if arr.ndim != 1:
        raise ValueError(f"cells must be 1D; got shape {np.asarray(cells).shape}.")
    if arr.size == 0:
        return np.empty(0, dtype=np.intp)
    if arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"cells must be an integer array of cell ids; got dtype {arr.dtype!r}.")
    ids = arr.astype(np.intp, copy=False)
    if int(ids.min()) < 0 or int(ids.max()) >= num_cells:
        raise ValueError(
            f"cells must lie in [0, {num_cells}); got range [{int(ids.min())}, {int(ids.max())}]."
        )
    return ids


__all__ = ["cell_quadrature"]
