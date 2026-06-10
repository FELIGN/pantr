"""Spectral partitioning of a space's cell-coupling graph into rank subdomains.

:func:`partition_graph` turns a :class:`~pantr.bspline.CouplingGraph` (built by
:func:`~pantr.bspline.coupling_graph`) into a :class:`~pantr.grid.Partition` by
recursive spectral (Fiedler) bisection, minimizing the cross-rank DOF coupling
that geometric backends (``block`` / ``rcb`` in :func:`pantr.grid.partition_grid`)
cannot see. It is the dependency-free, weight- and activity-aware graph backend:
pure core, ``scipy`` only -- no MPI, no external graph library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import ArpackError, eigsh

from ..grid import Partition
from ._coupling_graph import CouplingGraph

if TYPE_CHECKING:
    import numpy.typing as npt

_DENSE_FIEDLER_MAX = 512
"""Subgraphs with at most this many vertices use a dense eigensolver."""


def partition_graph(
    coupling: CouplingGraph,
    n_parts: int,
    *,
    cell_active: npt.ArrayLike | None = None,
) -> Partition:
    """Partition a cell-coupling graph into ``n_parts`` rank subdomains.

    Recursively bisects the graph, ordering each subgraph's vertices by its Fiedler
    vector (the second eigenvector of the weighted graph Laplacian) and cutting at the
    weighted split point. Balances :attr:`CouplingGraph.vertex_weights` and clamps the
    cut so no rank is left empty. Disconnected subgraphs are split by connected
    component first. The result minimizes cross-rank DOF coupling far better than a
    geometric split for irregular or hierarchical meshes.

    Args:
        coupling (CouplingGraph): The cell-coupling graph (see
            :func:`~pantr.bspline.coupling_graph`); its ``vertex_weights`` drive the
            load balance.
        n_parts (int): Number of parts (ranks); must be ``>= 1``.
        cell_active (npt.ArrayLike | None): Optional boolean mask, shape
            ``(coupling.num_vertices,)``; inactive cells get owner ``-1`` and are
            excluded from the bisection. ``None`` means all active.

    Returns:
        Partition: A per-cell owner assignment with ``n_parts`` parts; ``-1`` for
        inactive cells, otherwise a rank in ``range(n_parts)`` with no rank empty.

    Raises:
        TypeError: If ``coupling`` is not a :class:`CouplingGraph`.
        ValueError: If ``n_parts < 1``; if ``cell_active`` has the wrong shape or marks
            no cell active; or if ``n_parts`` exceeds the number of active cells.
    """
    if not isinstance(coupling, CouplingGraph):
        raise TypeError(f"coupling must be a CouplingGraph; got {type(coupling).__name__}.")
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1; got {n_parts}.")

    n = coupling.num_vertices
    active = _validate_active(cell_active, n)
    active_idx = np.arange(n) if active is None else np.flatnonzero(active)
    n_active = int(active_idx.size)
    if n_parts > n_active:
        raise ValueError(
            f"n_parts={n_parts} exceeds the number of active cells ({n_active}); "
            f"cannot assign every rank a cell."
        )
    n_parts = int(n_parts)

    adjacency = sparse.csr_matrix(
        (coupling.edge_weights.astype(np.float64), coupling.adjncy, coupling.xadj),
        shape=(n, n),
    )
    if active is not None:
        adjacency = adjacency[active_idx][:, active_idx]
    weights = coupling.vertex_weights[active_idx]

    owner_active = np.empty(n_active, dtype=np.int32)

    def bisect(idx: npt.NDArray[np.intp], part_lo: int, part_hi: int) -> None:
        # Split parts [part_lo, part_hi) by ordering this subgraph's vertices via its
        # Fiedler vector, cutting at the (clamped) weighted split point so each side
        # keeps at least as many cells as parts (no rank is left empty).
        k = part_hi - part_lo
        if k == 1:
            owner_active[idx] = part_lo
            return
        ordered = idx[_spectral_order(adjacency[idx][:, idx])]
        cumw = np.cumsum(weights[ordered])
        k_left = k // 2
        target = float(cumw[-1]) * k_left / k
        split = int(np.searchsorted(cumw, target, side="left")) + 1
        split = max(k_left, min(split, int(ordered.size) - (k - k_left)))
        bisect(ordered[:split], part_lo, part_lo + k_left)
        bisect(ordered[split:], part_lo + k_left, part_hi)

    bisect(np.arange(n_active), 0, n_parts)

    owner = np.full(n, -1, dtype=np.int32)
    owner[active_idx] = owner_active
    return Partition(owner, n_parts)


def _spectral_order(sub_adjacency: object) -> npt.NDArray[np.intp]:
    """Order a subgraph's vertices by its Fiedler vector (or by component).

    For a connected subgraph, returns the permutation sorting vertices by the Fiedler
    vector (second-smallest Laplacian eigenvector); the sign is canonicalized so the
    ordering is deterministic. A disconnected subgraph is ordered by connected-component
    label instead, which both avoids a singular Laplacian and keeps each component whole.

    Args:
        sub_adjacency (object): The ``(m, m)`` weighted adjacency (``scipy`` sparse
            matrix) of the subgraph, with ``m >= 2``.

    Returns:
        npt.NDArray[np.intp]: A length-``m`` permutation of ``range(m)``.

    Note:
        Inputs are assumed valid (``m >= 2``); no validation is performed.
    """
    m = sub_adjacency.shape[0]  # type: ignore[attr-defined]
    n_components, labels = csgraph.connected_components(sub_adjacency, directed=False)
    if n_components > 1:
        return np.argsort(labels, kind="stable")

    laplacian = csgraph.laplacian(sub_adjacency, normed=False)
    if m <= _DENSE_FIEDLER_MAX:
        dense = laplacian.toarray() if sparse.issparse(laplacian) else np.asarray(laplacian)
        _, vectors = np.linalg.eigh(dense)
        fiedler = vectors[:, 1]
    else:
        v0 = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
        try:
            values, vectors = eigsh(laplacian.tocsr().astype(np.float64), k=2, which="SA", v0=v0)
            fiedler = vectors[:, int(np.argsort(values)[1])]
        except ArpackError:
            _, vectors = np.linalg.eigh(laplacian.toarray())
            fiedler = vectors[:, 1]

    # Canonicalize the sign (largest-magnitude entry positive) for determinism.
    fiedler = fiedler * np.sign(fiedler[int(np.argmax(np.abs(fiedler)))])
    return np.argsort(fiedler, kind="stable")


def _validate_active(cell_active: npt.ArrayLike | None, n: int) -> npt.NDArray[np.bool_] | None:
    """Validate and coerce ``cell_active`` to a ``(n,)`` boolean array.

    Args:
        cell_active (npt.ArrayLike | None): Candidate activity mask, or ``None``.
        n (int): Expected length (number of graph vertices).

    Returns:
        npt.NDArray[np.bool_] | None: The coerced mask, or ``None`` if input was ``None``.

    Raises:
        ValueError: If the shape is not ``(n,)`` or no cell is active.
    """
    if cell_active is None:
        return None
    active = np.asarray(cell_active)
    if active.shape != (n,):
        raise ValueError(f"cell_active must have shape ({n},); got {active.shape}.")
    active = active.astype(bool)
    if not bool(active.any()):
        raise ValueError("cell_active must mark at least one cell active.")
    return cast("npt.NDArray[np.bool_]", active)


__all__ = ["partition_graph"]
