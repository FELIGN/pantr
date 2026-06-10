"""Graph partitioning of a space's cell-coupling graph into rank subdomains.

:func:`partition_graph` turns a :class:`~pantr.bspline.CouplingGraph` (built by
:func:`~pantr.bspline.coupling_graph`) into a :class:`~pantr.grid.Partition`, minimizing
the cross-rank DOF coupling that geometric backends (``block`` / ``rcb`` in
:func:`pantr.grid.partition_grid`) cannot see. Two backends, both serial (the partition
is computed redundantly per rank -- no MPI):

- ``"spectral"`` (default) -- recursive Fiedler bisection; pure ``scipy``, no extra
  dependency; weight- and activity-aware.
- ``"metis"`` -- METIS via the optional ``pymetis`` package (``pip install
  'pantr[metis]'``); higher-quality min-cut. Raises a clear error if ``pymetis`` is
  absent, so the default install never requires it.

Exports:
    - :func:`partition_graph`
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

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

_BACKENDS = ("spectral", "metis")
"""Graph-partition backends recognized by :func:`partition_graph`."""


def partition_graph(
    coupling: CouplingGraph,
    n_parts: int,
    *,
    backend: str = "spectral",
    cell_active: npt.ArrayLike | None = None,
) -> Partition:
    """Partition a cell-coupling graph into ``n_parts`` rank subdomains.

    Balances :attr:`CouplingGraph.vertex_weights` across parts while minimizing the
    weight of cut edges (shared DOFs). Cells excluded by ``cell_active`` get owner
    ``-1`` and are dropped from the graph.

    Args:
        coupling (CouplingGraph): The cell-coupling graph (see
            :func:`~pantr.bspline.coupling_graph`); its ``vertex_weights`` drive the
            load balance and ``edge_weights`` the cut cost.
        n_parts (int): Number of parts (ranks); must be ``>= 1``.
        backend (str): ``"spectral"`` (default; recursive Fiedler bisection, no extra
            dependency, never leaves a rank empty) or ``"metis"`` (METIS via the optional
            ``pymetis`` package; higher-quality min-cut, but may leave a rank empty).
        cell_active (npt.ArrayLike | None): Optional boolean mask, shape
            ``(coupling.num_vertices,)``; inactive cells get owner ``-1`` and are
            excluded. ``None`` means all active.

    Returns:
        Partition: A per-cell owner assignment with ``n_parts`` parts; ``-1`` for
        inactive cells, otherwise a rank in ``range(n_parts)``.

    Raises:
        TypeError: If ``coupling`` is not a :class:`CouplingGraph`.
        ValueError: If ``backend`` is unknown; if ``n_parts < 1``; if ``cell_active`` has
            the wrong shape or marks no cell active; or if ``n_parts`` exceeds the number
            of active cells.
        ImportError: If ``backend="metis"`` but ``pymetis`` is not installed.
    """
    if not isinstance(coupling, CouplingGraph):
        raise TypeError(f"coupling must be a CouplingGraph; got {type(coupling).__name__}.")
    if backend not in _BACKENDS:
        valid = ", ".join(repr(b) for b in _BACKENDS)
        raise ValueError(f"unknown backend {backend!r}; valid backends: {valid}.")
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

    if backend == "spectral":
        owner_active = _spectral_partition(adjacency, weights, n_parts)
    else:  # "metis"
        adjacency.sort_indices()
        owner_active = _metis_partition(
            adjacency.indptr, adjacency.indices, adjacency.data, weights, n_parts
        )

    owner = np.full(n, -1, dtype=np.int32)
    owner[active_idx] = owner_active
    return Partition(owner, n_parts)


def _spectral_partition(
    adjacency: Any,  # noqa: ANN401 -- a scipy sparse matrix (scipy is untyped)
    weights: npt.NDArray[np.float64],
    n_parts: int,
) -> npt.NDArray[np.int32]:
    """Partition by recursive spectral (Fiedler) bisection.

    Splits the part count ``k -> (k // 2, k - k // 2)``, orders each subgraph's vertices
    by its Fiedler vector, and cuts at the weighted split point -- clamped so each half
    receives at least as many vertices as it has parts (no rank is left empty).

    Args:
        adjacency (Any): ``(n_active, n_active)`` weighted ``scipy`` sparse adjacency.
        weights (npt.NDArray[np.float64]): Per-vertex weights.
        n_parts (int): Number of parts (``>= 1``).

    Returns:
        npt.NDArray[np.int32]: Per-vertex owner in ``range(n_parts)``.
    """
    owner = np.empty(int(adjacency.shape[0]), dtype=np.int32)

    def bisect(idx: npt.NDArray[np.intp], part_lo: int, part_hi: int) -> None:
        k = part_hi - part_lo
        if k == 1:
            owner[idx] = part_lo
            return
        ordered = idx[_spectral_order(adjacency[idx][:, idx])]
        cumw = np.cumsum(weights[ordered])
        k_left = k // 2
        target = float(cumw[-1]) * k_left / k
        split = int(np.searchsorted(cumw, target, side="left")) + 1
        split = max(k_left, min(split, int(ordered.size) - (k - k_left)))
        bisect(ordered[:split], part_lo, part_lo + k_left)
        bisect(ordered[split:], part_lo + k_left, part_hi)

    bisect(np.arange(owner.size), 0, n_parts)
    return owner


def _metis_partition(
    xadj: Any,  # noqa: ANN401 -- scipy CSR indptr; dtype (int32/int64) is scipy-version-dependent
    adjncy: Any,  # noqa: ANN401 -- scipy CSR indices; dtype (int32/int64) is scipy-version-dependent
    edge_weights: npt.NDArray[np.float64],
    vertex_weights: npt.NDArray[np.float64],
    n_parts: int,
) -> npt.NDArray[np.int32]:
    """Partition by METIS (k-way min-cut) via the optional ``pymetis`` package.

    Vertex weights are rounded to integers (clamped to ``>= 1``, as METIS requires
    positive integer weights) and edge weights to integers (must be ``> 0``); METIS then
    minimizes the cut while balancing the integer vertex weights. Unlike
    :func:`_spectral_partition`, METIS does not guarantee every part is non-empty.

    Args:
        xadj (Any): CSR row pointers of the (active) subgraph (scipy-version-dependent
            dtype; coerced to ``pymetis.zero_copy_dtype()`` internally).
        adjncy (Any): CSR neighbour indices (same dtype note as ``xadj``).
        edge_weights (npt.NDArray[np.float64]): Per-edge weights aligned with ``adjncy``;
            must round to integers ``>= 1``.
        vertex_weights (npt.NDArray[np.float64]): Per-vertex weights.
        n_parts (int): Number of parts (``>= 1``).

    Returns:
        npt.NDArray[np.int32]: Per-vertex owner in ``range(n_parts)``.

    Raises:
        ImportError: If ``pymetis`` is not installed or fails to load, and
            ``n_parts > 1``.
        ValueError: If any edge weight rounds to ``<= 0`` (METIS requires positive
            integer edge weights).
        RuntimeError: If the METIS library reports an internal error.
    """
    n_vertices = int(xadj.shape[0]) - 1
    if n_parts == 1:
        return np.zeros(n_vertices, dtype=np.int32)

    pymetis: Any = _require_pymetis()
    idx_dtype = pymetis.zero_copy_dtype()
    adjacency = pymetis.CSRAdjacency(
        np.asarray(xadj, dtype=idx_dtype),
        np.asarray(adjncy, dtype=idx_dtype),
    )
    eweights = np.rint(edge_weights).astype(idx_dtype)
    zero_count = int(np.sum(eweights <= 0))
    if zero_count > 0:
        raise ValueError(
            f"METIS requires positive integer edge weights; after rounding, "
            f"{zero_count} edge(s) have weight <= 0. Scale up the graph edge weights "
            f"or use backend='spectral'."
        )
    vweights = np.maximum(1, np.rint(vertex_weights)).astype(idx_dtype)
    try:
        _, membership = pymetis.part_graph(n_parts, adjacency, vweights=vweights, eweights=eweights)
    except RuntimeError as exc:
        raise RuntimeError(
            f"METIS partitioning failed for graph with {n_vertices} vertices and "
            f"{int(adjncy.shape[0])} edges (n_parts={n_parts}). "
            f"Try backend='spectral' as a fallback. METIS error: {exc}"
        ) from exc
    return cast("npt.NDArray[np.int32]", np.asarray(membership, dtype=np.int32))


def _require_pymetis() -> ModuleType:
    """Import and return the ``pymetis`` module, or raise a clear error if absent.

    Returns:
        ModuleType: The imported ``pymetis`` module.

    Raises:
        ImportError: If ``pymetis`` is not installed or fails to load (e.g. missing
            native library), with guidance on how to obtain it.
    """
    try:
        return importlib.import_module("pymetis")
    except ImportError as exc:
        raise ImportError(
            "backend='metis' requires 'pymetis', which is not installed or failed to load. "
            "Install it with \"pip install 'pantr[metis]'\" (or 'pip install pymetis'); or use "
            f"backend='spectral', which needs no extra dependency. (Original error: {exc})"
        ) from exc


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
        ``m >= 2`` is assumed; the caller guarantees this because subgraphs of size 1
        are assigned directly without further bisection.
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
