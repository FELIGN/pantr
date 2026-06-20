"""Top-level MPI-distributed B-spline / THB-spline space.

:class:`DistributedSpace` is the per-rank handle to a space distributed across an MPI
communicator. Given the global space and a :class:`~pantr.grid.Partition` (identical on
every rank), it builds *this rank's* windowed :class:`~pantr.bspline.LocalSpace` via
:func:`~pantr.bspline.build_local`. Construction is purely local -- no MPI communication
and no DOF exchange -- consistent with pantr's redundant per-rank storage model
(cross-rank coupling is the consumer's job, e.g. via a PETSc ``PtAP``).

``mpi4py`` is not imported by this module: the communicator is treated as an opaque object
and only its ``rank`` and ``size`` attributes are read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..bspline import Bspline, BsplineSpace, THBSpline, THBSplineSpace, build_local
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..bspline import LocalSpace
    from ..grid import Partition


class DistributedSpace:
    """A B-spline or THB-spline space distributed across an MPI communicator.

    The per-rank, SPMD handle: every rank constructs its own ``DistributedSpace`` from
    the same global space and partition, and holds only its own
    :class:`~pantr.bspline.LocalSpace` (``local``). A rank that owns no cells (an
    over-provisioned run, or a partition that excludes some ranks) has ``local`` set to
    ``None`` and ``owns_cells`` ``False`` instead of failing.

    Construction performs no MPI communication: the partition is assumed identical on
    every rank (guaranteed by pantr's deterministic partitioners and by
    :func:`~pantr.mpi.from_dolfinx`), so each rank can window the global space locally.

    The public surface exposes:

    - :attr:`comm` -- the MPI communicator passed at construction.
    - :attr:`rank` -- this rank's id within the communicator.
    - :attr:`n_parts` -- number of ranks the space is distributed over.
    - :attr:`global_space` -- the undistributed global space.
    - :attr:`partition` -- the cell-ownership partition.
    - :attr:`local` -- this rank's windowed local space (``None`` if it owns no cells).
    - :attr:`owns_cells` -- whether this rank owns at least one cell.
    - :attr:`owned_cells` -- read-only sorted global cell ids owned by this rank.
    """

    __slots__ = ("_comm", "_global_space", "_local", "_owned_cells", "_partition", "_rank")

    def __init__(
        self,
        global_space: BsplineSpace | THBSplineSpace,
        partition: Partition,
        comm: Any,  # noqa: ANN401 -- an mpi4py.MPI.Comm; mpi4py is an optional, untyped dep
    ) -> None:
        """Build the distributed space, windowing the global space to this rank.

        Args:
            global_space (BsplineSpace | THBSplineSpace): The global (non-periodic)
                space, identical on every rank.
            partition (Partition): Owner of every cell of the space's grid, identical on
                every rank. Its ``n_parts`` must equal ``comm.size``.
            comm (Any): An MPI communicator (e.g. ``mpi4py.MPI.COMM_WORLD``). Only its
                ``rank`` and ``size`` attributes are read.

        Raises:
            ValueError: If ``global_space`` is a periodic
                :class:`~pantr.bspline.BsplineSpace`; if ``comm.size != partition.n_parts``;
                if ``partition`` does not match the global space's cell count; or if
                ``comm.rank`` is outside ``[0, comm.size)`` (delegated to
                :meth:`~pantr.grid.Partition.owned_cells`).

        Note:
            Unlike :func:`~pantr.bspline.build_local`, construction succeeds when this rank
            owns no cells -- :attr:`local` is set to ``None`` and :attr:`owns_cells` to
            ``False``.

        Note:
            Construction engages the default MPI thread policy: unless threads were
            explicitly configured, this process's Numba thread pool is limited to one
            thread per rank. See :func:`pantr.mpi.configure_threads`.
        """
        _ensure_default_thread_policy()
        if isinstance(global_space, BsplineSpace) and any(
            sp.periodic for sp in global_space.spaces
        ):
            raise ValueError("periodic B-spline spaces are not supported.")
        size = int(comm.size)  # int() guards against mpi4py returning numpy.intp
        if size != partition.n_parts:
            raise ValueError(
                f"comm.size ({size}) must equal partition.n_parts ({partition.n_parts})."
            )
        n_global_cells = _global_cell_count(global_space)
        if partition.n_cells != n_global_cells:
            raise ValueError(
                f"partition has {partition.n_cells} cells; "
                f"expected {n_global_cells} (the global space's cell count)."
            )

        rank = int(comm.rank)  # int() guards against mpi4py returning numpy.intp
        owned = partition.owned_cells(rank)  # validates rank in [0, n_parts)
        owned.flags.writeable = False

        self._comm = comm
        self._global_space = global_space
        self._partition = partition
        self._rank = rank
        self._owned_cells = owned
        # build_local raises if the rank owns no cells; guard here to support empty ranks.
        self._local: LocalSpace | None = (
            build_local(global_space, partition, rank) if owned.size else None
        )

    @property
    def comm(self) -> Any:  # noqa: ANN401 -- the caller-supplied MPI communicator
        """Get the MPI communicator this space is distributed over.

        Returns:
            Any: The communicator passed at construction.
        """
        return self._comm

    @property
    def rank(self) -> int:
        """Get this rank's id within the communicator.

        Returns:
            int: ``comm.rank``.
        """
        return self._rank

    @property
    def n_parts(self) -> int:
        """Get the number of ranks (parts) the space is distributed over.

        Returns:
            int: ``partition.n_parts`` (equal to ``comm.size`` at construction time).
        """
        return self._partition.n_parts

    @property
    def global_space(self) -> BsplineSpace | THBSplineSpace:
        """Get the undistributed global space.

        Returns:
            BsplineSpace | THBSplineSpace: The global space passed at construction.
        """
        return self._global_space

    @property
    def partition(self) -> Partition:
        """Get the cell-ownership partition.

        Returns:
            Partition: The partition passed at construction.
        """
        return self._partition

    @property
    def local(self) -> LocalSpace | None:
        """Get this rank's local windowed space, or ``None`` if it owns no cells.

        Returns:
            LocalSpace | None: The rank-local space (windowed space plus local-to-global
            cell/DOF maps and ownership masks), or ``None`` when ``owns_cells`` is
            ``False``. See :class:`~pantr.bspline.LocalSpace` for all fields.

        Note:
            Callers must narrow the type before use: ``assert ds.local is not None``
            or ``if ds.local is not None``. :attr:`owns_cells` does not statically
            narrow this property in mypy.
        """
        return self._local

    @property
    def owns_cells(self) -> bool:
        """Report whether this rank owns at least one cell.

        Returns:
            bool: ``True`` iff :attr:`local` is not ``None``.
        """
        return self._local is not None

    @property
    def owned_cells(self) -> npt.NDArray[np.int64]:
        """Get the global ids of the cells this rank owns.

        Returns:
            npt.NDArray[np.int64]: Read-only sorted global cell ids owned by this rank
            (empty if the rank owns none).
        """
        return self._owned_cells

    def localize(self, control_points: npt.ArrayLike) -> Bspline | THBSpline | None:
        """Restrict a global coefficient field to this rank's local function.

        Slices the *global* control points (identical on every rank, one per global DOF)
        down to this rank's local DOFs via :attr:`local`'s ``local_to_global_dof`` map,
        and wraps them on the windowed :attr:`local` space.  The returned function equals
        the global one pointwise over the rank's owned cells, so per-element evaluation
        and assembly are local.  Reuse a single distributed space to localize many fields
        (right-hand side, solution, residual, ...).

        Args:
            control_points (npt.ArrayLike): Global control points, shape
                ``(n_global_dofs,)`` for a scalar field or ``(n_global_dofs, rank)`` for a
                vector field, where ``n_global_dofs == global_space.num_total_basis``.

        Returns:
            Bspline | THBSpline | None: This rank's local function (a
            :class:`~pantr.bspline.Bspline` for a tensor-product space, a
            :class:`~pantr.bspline.THBSpline` for a hierarchical one), or ``None`` if the
            rank owns no cells.  Scalar vs. vector kind is preserved.

        Raises:
            ValueError: If ``control_points``'s leading dimension is not
                ``global_space.num_total_basis``.
        """
        local = self._local
        if local is None:
            return None
        cp = np.asarray(control_points, dtype=np.float64)
        n_global = self._global_space.num_total_basis
        if cp.shape[0] != n_global:
            raise ValueError(
                f"control_points leading dimension must be {n_global} "
                f"(global_space.num_total_basis); got {cp.shape[0]}."
            )
        local_cp = cp[local.local_to_global_dof]
        if isinstance(local.space, THBSplineSpace):
            return THBSpline(local.space, local_cp)
        return Bspline(local.space, local_cp)


def _global_cell_count(space: BsplineSpace | THBSplineSpace) -> int:
    """Return the number of cells in a space's grid.

    Args:
        space (BsplineSpace | THBSplineSpace): The global space.

    Returns:
        int: ``grid.num_cells`` for a THB space, else ``num_total_intervals``.
    """
    if isinstance(space, THBSplineSpace):
        return space.grid.num_cells
    return space.num_total_intervals


__all__ = ["DistributedSpace"]
