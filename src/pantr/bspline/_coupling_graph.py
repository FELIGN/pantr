"""Cell-coupling (dual) graph of a B-spline or THB-spline space.

Builds the graph whose vertices are the cells of a space's grid and whose edges
join cells that share at least one basis function (DOF), weighted by the number
of shared functions. This is the input a graph partitioner (METIS / Scotch)
needs to minimize cross-rank DOF coupling; it is produced serially with no MPI
and no external dependency, and consumed later by the optional graph-partition
backends.

The graph is emitted in the standard CSR adjacency format used by METIS and
Scotch (``xadj`` / ``adjncy`` / ``edge_weights`` / ``vertex_weights``, corresponding
to METIS ``xadj`` / ``adjncy`` / ``adjwgt`` / ``vwgt``), so a backend can hand it
to those libraries with no reshaping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from scipy import sparse

from ._bspline_space_nd import BsplineSpace
from ._local_space import _reject_periodic
from ._thb_spline_space import THBSplineSpace, _func_support_1d

if TYPE_CHECKING:
    import numpy.typing as npt


class CouplingGraph(NamedTuple):
    """Cell-coupling graph of a space, in METIS / Scotch CSR adjacency format.

    A :class:`typing.NamedTuple` returned by :func:`coupling_graph`. The graph is
    undirected (symmetric adjacency) and has no self-loops. All array fields are
    read-only.

    Attributes:
        num_vertices (int): Number of cells (graph vertices).
        xadj (npt.NDArray[np.int64]): CSR row pointers, shape ``(num_vertices + 1,)``;
            the neighbors of cell ``c`` are ``adjncy[xadj[c]:xadj[c + 1]]``.
            (METIS ``xadj``.)
        adjncy (npt.NDArray[np.int64]): Concatenated neighbor cell ids, shape
            ``(xadj[-1],)``. (METIS ``adjncy``.)
        edge_weights (npt.NDArray[np.int64]): Per-adjacency-entry weight = number of
            basis functions the two cells share, aligned with ``adjncy``.
            (METIS ``adjwgt``.)
        vertex_weights (npt.NDArray[np.float64]): Per-cell weight (assembly cost),
            shape ``(num_vertices,)``; uniform ``1.0`` unless ``cell_weights`` was
            given. (METIS ``vwgt``.)
    """

    num_vertices: int
    xadj: npt.NDArray[np.int64]
    adjncy: npt.NDArray[np.int64]
    edge_weights: npt.NDArray[np.int64]
    vertex_weights: npt.NDArray[np.float64]


def coupling_graph(
    space: BsplineSpace | THBSplineSpace,
    *,
    cell_weights: npt.ArrayLike | None = None,
) -> CouplingGraph:
    """Build the cell-coupling graph of a B-spline or THB-spline space.

    Two cells are joined by an edge when they share at least one basis function;
    the edge weight is the number of shared functions, and each vertex (cell)
    carries an optional assembly-cost weight. The result is the dual graph a
    graph partitioner uses to minimize cross-rank DOF coupling.

    Args:
        space (BsplineSpace | THBSplineSpace): The space whose cells to couple. A
            :class:`BsplineSpace` must be non-periodic.
        cell_weights (npt.ArrayLike | None): Optional per-cell assembly cost, shape
            ``(num_cells,)``, finite and non-negative. ``None`` means uniform.

    Returns:
        CouplingGraph: The coupling graph in METIS / Scotch CSR adjacency format.

    Raises:
        TypeError: If ``space`` is neither a :class:`BsplineSpace` nor a
            :class:`THBSplineSpace`.
        ValueError: If ``space`` is a periodic :class:`BsplineSpace`, or if
            ``cell_weights`` has the wrong shape or invalid values.

    Example:
        >>> from pantr.bspline import coupling_graph, create_uniform_space
        >>> space = create_uniform_space(2, 4)
        >>> graph = coupling_graph(space)
        >>> graph.num_vertices
        4
        >>> int(graph.xadj[-1]) == graph.adjncy.size
        True
    """
    if isinstance(space, THBSplineSpace):
        n_cells = space.grid.num_cells
        n_dofs = space.num_total_basis
        rows, cols = _thb_incidence(space)
    elif isinstance(space, BsplineSpace):
        _reject_periodic(space)
        n_cells = space.num_total_intervals
        n_dofs = space.num_total_basis
        rows, cols = _tp_incidence(space)
    else:
        raise TypeError(
            f"coupling_graph expects a BsplineSpace or THBSplineSpace; got {type(space).__name__}."
        )
    vertex_weights = _validate_vertex_weights(cell_weights, n_cells)
    return _dual_graph(rows, cols, n_cells, n_dofs, vertex_weights)


def _tp_incidence(space: BsplineSpace) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Build the cell -> DOF incidence (row, col) pairs for a tensor-product space.

    Cell ``c`` (multi-index ``i``) is supported by the tensor product of the per-axis
    function ranges ``fb_d[i_d], fb_d[i_d] + 1, ..., fb_d[i_d] + degree_d`` (that is,
    ``degree_d + 1`` functions per axis), where ``fb_d`` is the first function non-zero
    on each cell along axis ``d``. Cells and DOFs are flat-indexed in C-order over
    ``num_intervals`` and ``num_basis`` respectively.

    Args:
        space (BsplineSpace): Non-periodic tensor-product B-spline space.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ``(rows, cols)`` cell ids
        and the global DOF ids they support, one entry per (cell, supported DOF).
    """
    num_intervals = space.num_intervals
    num_basis = space.num_basis
    n_cells = space.num_total_intervals
    fb_axes = [_func_support_1d(sp)[0].astype(np.int64) for sp in space.spaces]

    offset_grids = np.meshgrid(*[np.arange(sp.degree + 1) for sp in space.spaces], indexing="ij")
    local_offsets = [grid.ravel() for grid in offset_grids]
    cell_multi = np.unravel_index(np.arange(n_cells), num_intervals)
    func_per_axis = [
        fb_axes[d][cell_multi[d]][:, None] + local_offsets[d][None, :] for d in range(space.dim)
    ]
    dof_ids = np.ravel_multi_index(func_per_axis, num_basis)

    k = dof_ids.shape[1]
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    cols = dof_ids.ravel().astype(np.int64)
    return rows, cols


def _thb_incidence(space: THBSplineSpace) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Build the cell -> DOF incidence (row, col) pairs for a THB-spline space.

    Uses :meth:`THBSplineSpace.active_basis` for the active global DOF ids on each cell.
    ``active_basis`` returns all functions whose untruncated support intersects the cell,
    including truncated functions that may evaluate to zero there; edge weights therefore
    count support-intersecting functions, not just non-zero ones.

    Args:
        space (THBSplineSpace): The hierarchical space.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ``(rows, cols)`` cell ids
        and the global DOF ids they support, one entry per (cell, active DOF).
    """
    # grid.num_cells >= 1 is enforced by TensorProductGrid construction, so the loop
    # always executes at least once and np.concatenate never receives an empty list.
    rows_list: list[npt.NDArray[np.int64]] = []
    cols_list: list[npt.NDArray[np.int64]] = []
    for cid in range(space.grid.num_cells):
        dofs = np.asarray(space.active_basis(cid), dtype=np.int64)
        rows_list.append(np.full(dofs.size, cid, dtype=np.int64))
        cols_list.append(dofs)
    return np.concatenate(rows_list), np.concatenate(cols_list)


def _dual_graph(
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    n_cells: int,
    n_dofs: int,
    vertex_weights: npt.NDArray[np.float64],
) -> CouplingGraph:
    """Assemble the dual graph from a cell -> DOF incidence.

    Forms the integer cell-by-DOF incidence ``B`` (entries are 1 for active
    ``(cell, DOF)`` pairs) and the symmetric product ``B @ B.T``: its off-diagonal entry
    ``(i, j)`` counts the DOFs cells ``i`` and ``j`` share. The diagonal (self-coupling)
    is dropped, leaving the CSR adjacency.

    Note:
        ``rows`` / ``cols`` must be duplicate-free. ``scipy.sparse.csr_matrix`` sums
        repeated ``(row, col)`` entries, so any duplicate ``(cell, DOF)`` pair would
        silently inflate the edge weight for that cell pair.

    Args:
        rows (npt.NDArray[np.int64]): Cell ids of the incidence entries.
        cols (npt.NDArray[np.int64]): Supported global DOF ids, aligned with ``rows``.
        n_cells (int): Number of cells (graph vertices).
        n_dofs (int): Number of global DOFs.
        vertex_weights (npt.NDArray[np.float64]): Per-cell weights (already validated).

    Returns:
        CouplingGraph: The assembled coupling graph.
    """
    incidence = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.int64), (rows, cols)),
        shape=(n_cells, n_dofs),
    )
    coupling = (incidence @ incidence.T).tocsr()
    coupling.setdiag(0)
    coupling.eliminate_zeros()
    coupling.sort_indices()

    xadj = coupling.indptr.astype(np.int64)
    adjncy = coupling.indices.astype(np.int64)
    edge_weights = coupling.data.astype(np.int64)
    for arr in (xadj, adjncy, edge_weights):
        arr.flags.writeable = False
    return CouplingGraph(
        int(n_cells),
        cast("npt.NDArray[np.int64]", xadj),
        cast("npt.NDArray[np.int64]", adjncy),
        cast("npt.NDArray[np.int64]", edge_weights),
        vertex_weights,
    )


def _validate_vertex_weights(
    cell_weights: npt.ArrayLike | None, n_cells: int
) -> npt.NDArray[np.float64]:
    """Validate and coerce ``cell_weights`` to a read-only ``(n_cells,)`` array.

    Args:
        cell_weights (npt.ArrayLike | None): Candidate per-cell weights, or ``None``
            for uniform.
        n_cells (int): Expected length.

    Returns:
        npt.NDArray[np.float64]: Read-only weights; uniform ``1.0`` if input was
        ``None``.

    Raises:
        ValueError: If the shape is not ``(n_cells,)`` or any entry is negative or
            non-finite.
    """
    weights: npt.NDArray[np.float64]
    if cell_weights is None:
        weights = np.ones(n_cells, dtype=np.float64)
    else:
        weights = np.array(cell_weights, dtype=np.float64)
        if weights.shape != (n_cells,):
            raise ValueError(f"cell_weights must have shape ({n_cells},); got {weights.shape}.")
        if not bool(np.all(np.isfinite(weights))) or bool(np.any(weights < 0.0)):
            raise ValueError("cell_weights must be finite and non-negative.")
    weights.flags.writeable = False
    return weights


__all__ = ["CouplingGraph", "coupling_graph"]
