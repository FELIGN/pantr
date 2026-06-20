"""Per-rank handle for an MPI-distributed spline function (space + control points).

Provides :class:`DistributedFunction`, the function-level counterpart of
:class:`~pantr.mpi.DistributedSpace`, and the :func:`create_distributed_function`
factory.  A distributed function pairs a :class:`~pantr.mpi.DistributedSpace` with a
global coefficient field and exposes this rank's local
:class:`~pantr.bspline.Bspline` / :class:`~pantr.bspline.THBSpline`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..bspline import Bspline, THBSpline
from ._create import create_distributed_space
from ._distributed_space import DistributedSpace
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    import numpy.typing as npt


def _dof_coeffs(function: Bspline | THBSpline) -> npt.NDArray[np.float64]:
    """Return a function's control points indexed by global DOF.

    A :class:`~pantr.bspline.THBSpline` already stores its coefficients per active DOF;
    a :class:`~pantr.bspline.Bspline` stores them tensor-product reshaped
    (``(*num_basis, rank)``), so they are flattened to ``(num_total_basis, rank)`` in the
    C-order DOF numbering used by the windowing maps.

    Args:
        function (Bspline | THBSpline): The global function.

    Returns:
        npt.NDArray[np.float64]: Control points shaped ``(num_total_basis,)`` or
        ``(num_total_basis, rank)``, indexed by global DOF.
    """
    cp = np.asarray(function.control_points, dtype=np.float64)
    if isinstance(function, Bspline):
        return cp.reshape(function.space.num_total_basis, -1)
    return cp


class DistributedFunction:
    """Per-rank handle to an MPI-distributed spline function.

    Pairs a :class:`~pantr.mpi.DistributedSpace` with a global control-point field and
    holds this rank's local function -- the windowed space's
    :class:`~pantr.bspline.Bspline` / :class:`~pantr.bspline.THBSpline` whose control
    points are the global field sliced to the rank's local DOFs.  The local function
    equals the global one pointwise over the rank's owned cells.  Construction performs
    no MPI communication (it reuses the distributed space's windowing).

    Attributes:
        _global_function (Bspline | THBSpline): The undistributed global function.
        _distributed_space (DistributedSpace): The distributed space it is defined on.
        _local (Bspline | THBSpline | None): This rank's local function, or ``None``.
    """

    __slots__ = ("_distributed_space", "_global_function", "_local")

    def __init__(
        self,
        global_function: Bspline | THBSpline,
        distributed_space: DistributedSpace,
    ) -> None:
        """Build the distributed function by localizing the global control points.

        Args:
            global_function (Bspline | THBSpline): The global function, identical on
                every rank.  Its space must be the one ``distributed_space`` distributes.
            distributed_space (DistributedSpace): The distributed space whose
                ``global_space`` is exactly ``global_function.space``.

        Raises:
            ValueError: If ``global_function.space`` is not the distributed space's
                ``global_space``.
        """
        if global_function.space is not distributed_space.global_space:
            raise ValueError(
                "global_function.space must be the distributed_space's global_space; "
                "build the DistributedSpace from this function's space."
            )
        self._global_function = global_function
        self._distributed_space = distributed_space
        self._local: Bspline | THBSpline | None = distributed_space.localize(
            _dof_coeffs(global_function)
        )

    @property
    def global_function(self) -> Bspline | THBSpline:
        """Get the undistributed global function.

        Returns:
            Bspline | THBSpline: The global function passed at construction.
        """
        return self._global_function

    @property
    def distributed_space(self) -> DistributedSpace:
        """Get the distributed space this function is defined on.

        Returns:
            DistributedSpace: The underlying distributed space.
        """
        return self._distributed_space

    @property
    def local(self) -> Bspline | THBSpline | None:
        """Get this rank's local function, or ``None`` if it owns no cells.

        Returns:
            Bspline | THBSpline | None: The rank-local function on the windowed space
            (control points sliced to local DOFs), or ``None`` when
            :attr:`owns_cells` is ``False``.  Scalar vs. vector kind is preserved.
        """
        return self._local

    @property
    def rank(self) -> int:
        """Get this rank's id within the communicator.

        Returns:
            int: The rank id (``== distributed_space.rank``).
        """
        return self._distributed_space.rank

    @property
    def n_parts(self) -> int:
        """Get the number of ranks.

        Returns:
            int: The number of parts (``== distributed_space.n_parts``).
        """
        return self._distributed_space.n_parts

    @property
    def owns_cells(self) -> bool:
        """Report whether this rank owns at least one cell.

        Returns:
            bool: ``True`` iff :attr:`local` is not ``None``.
        """
        return self._local is not None


def create_distributed_function(  # noqa: PLR0913 -- public factory mirrors the partitioner args
    global_function: Bspline | THBSpline,
    comm: Any,  # noqa: ANN401 -- an mpi4py.MPI.Comm; mpi4py is an optional, untyped dep
    *,
    method: str = "grid",
    backend: str | None = None,
    cell_weights: npt.ArrayLike | None = None,
    cell_active: npt.ArrayLike | None = None,
) -> DistributedFunction:
    """Build an MPI-distributed function directly from a global function.

    The function-level counterpart of :func:`~pantr.mpi.create_distributed_space`:
    distributes the function's space across ``comm`` and slices the control points to
    each rank, returning a :class:`DistributedFunction` whose :attr:`~DistributedFunction.local`
    is this rank's :class:`~pantr.bspline.Bspline` / :class:`~pantr.bspline.THBSpline`.

    Args:
        global_function (Bspline | THBSpline): The global function to distribute,
            identical on every rank.
        comm (Any): An MPI communicator (e.g. ``mpi4py.MPI.COMM_WORLD``); only its
            ``rank`` and ``size`` are read.
        method (str): Partitioning strategy, ``"grid"`` (default) or ``"graph"``.  See
            :func:`~pantr.mpi.create_distributed_space`.
        backend (str | None): Partitioner backend; ``None`` selects each method's
            default.  See :func:`~pantr.mpi.create_distributed_space`.
        cell_weights (npt.ArrayLike | None): Per-cell assembly-cost weights.  Defaults
            to ``None``.
        cell_active (npt.ArrayLike | None): Boolean per-cell activity mask.  Defaults to
            ``None``.

    Returns:
        DistributedFunction: The per-rank distributed-function handle.

    Raises:
        ValueError: If ``method``/``backend`` are invalid, or the derived partition is
            incompatible with ``comm`` (as raised downstream).

    Example:
        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> from pantr.bspline import Bspline, create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_function  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> f = Bspline(space, coeffs)  # doctest: +SKIP
        >>> df = create_distributed_function(f, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> local_f = df.local  # this rank's windowed function  # doctest: +SKIP
    """
    _ensure_default_thread_policy()
    distributed_space = create_distributed_space(
        global_function.space,
        comm,
        method=method,
        backend=backend,
        cell_weights=cell_weights,
        cell_active=cell_active,
    )
    return DistributedFunction(global_function, distributed_space)
