"""Top-level MPI-distributed B-spline / THB-spline space.

:class:`DistributedSpace` is the per-rank handle to a space distributed across an MPI
communicator. Given the global space and a :class:`~pantr.grid.Partition` (identical on
every rank), it builds *this rank's* windowed :class:`~pantr.bspline.LocalSpace` via
:func:`~pantr.bspline.build_local`. Construction is purely local -- no MPI communication
and no DOF exchange -- consistent with pantr's redundant per-rank storage model
(cross-rank coupling is the consumer's job, e.g. via a PETSc ``PtAP``).

``mpi4py`` is not imported: the communicator is supplied by the caller and only its
``rank`` and ``size`` are read, so ``import pantr.mpi`` still works without MPI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..bspline import THBSplineSpace, build_local

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ..bspline import BsplineSpace, LocalSpace
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

    Attributes are exposed through the :attr:`comm`, :attr:`rank`, :attr:`n_parts`,
    :attr:`global_space`, :attr:`partition`, :attr:`local`, :attr:`owns_cells`, and
    :attr:`owned_cells` properties.
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
            ValueError: If ``comm.size != partition.n_parts``; if ``partition`` does not
                match the global space's cell count; if ``comm.rank`` is out of range; or
                if ``global_space`` is a periodic :class:`~pantr.bspline.BsplineSpace`.
        """
        size = int(comm.size)
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

        rank = int(comm.rank)
        owned = partition.owned_cells(rank)  # validates rank in [0, n_parts)
        owned.flags.writeable = False

        self._comm = comm
        self._global_space = global_space
        self._partition = partition
        self._rank = rank
        self._owned_cells = owned
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
            int: ``comm.size`` (equal to ``partition.n_parts``).
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
            ``False``.
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
