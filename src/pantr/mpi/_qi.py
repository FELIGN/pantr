"""Distributed quasi-interpolation onto tensor-product B-spline spaces.

Provides :func:`quasi_interpolate_bspline_distributed`, the MPI-parallel counterpart of
:func:`~pantr.bspline.quasi_interpolate_bspline`.  Each rank evaluates the function only
on the points required by its *owned* DOFs (no redundant evaluation of halo-DOF points),
then a single :pyfunc:`allgather` collective assembles the full global coefficient field.
The result is a :class:`~pantr.mpi.DistributedFunction` whose
:attr:`~pantr.mpi.DistributedFunction.local` reproduces the serial quasi-interpolant
exactly over the rank's owned cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np

from ..bspline import Bspline, BsplineSpace
from ..bspline._bspline_quasi_interpolation import QIKind, quasi_interpolate_bspline
from ._distributed_function import DistributedFunction
from ._distributed_space import DistributedSpace
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


def quasi_interpolate_bspline_distributed(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    distributed_space: DistributedSpace,
    *,
    kind: QIKind = "llm",
) -> DistributedFunction:
    """Quasi-interpolate a callable onto a distributed tensor-product B-spline space.

    The MPI-parallel counterpart of :func:`~pantr.bspline.quasi_interpolate_bspline`.
    Each rank evaluates ``func`` only on the interior points required by its *owned*
    basis functions (Lee-Lyche-Mørken functionals); a single ``allgather`` at the end
    assembles the global coefficient field.  The returned
    :class:`~pantr.mpi.DistributedFunction` agrees with the serial quasi-interpolant
    pointwise over every owned cell.

    Construction requires one MPI collective (``comm.allgather``) after the local
    computation.  Per-rank ``func`` evaluation is purely local: no rank ever evaluates
    ``func`` outside its windowed parametric sub-domain.

    Args:
        func (Callable): Function to quasi-interpolate.  Called on a flat
            ``(M, dim)`` point array; must return ``(M,)`` (scalar) or ``(M, rank)``
            (vector-valued).
        distributed_space (DistributedSpace): The distributed space to interpolate onto.
            Its ``global_space`` must be a :class:`~pantr.bspline.BsplineSpace`.
        kind (QIKind): Quasi-interpolant kind.  Only ``"llm"`` (Lee-Lyche-Mørken) is
            currently supported.  Defaults to ``"llm"``.

    Returns:
        DistributedFunction: A distributed function whose
        :attr:`~pantr.mpi.DistributedFunction.local` quasi-interpolates ``func`` over
        this rank's owned cells, and whose
        :attr:`~pantr.mpi.DistributedFunction.global_function` holds the full assembled
        global coefficient field (identical on every rank after the ``allgather``).

    Raises:
        TypeError: If ``distributed_space.global_space`` is not a
            :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``kind`` is not recognized, or if ``func`` returns an output
            with an invalid shape (0-D, more than 2-D, or wrong leading dimension).

    Note:
        All internal computation uses ``float64``.  The global control points are cast
        to ``global_space.dtype`` before assembly, consistent with the serial
        :func:`~pantr.bspline.quasi_interpolate_bspline`.

    Example:
        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> from pantr.mpi import quasi_interpolate_bspline_distributed  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> ds = create_distributed_space(space, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> dfn = quasi_interpolate_bspline_distributed(  # doctest: +SKIP
        ...     lambda p: np.sin(p[:, 0]) * np.cos(p[:, 1]), ds
        ... )
        >>> local = dfn.local  # rank-local Bspline on the windowed space  # doctest: +SKIP
    """
    _ensure_default_thread_policy()

    global_space = distributed_space.global_space
    if not isinstance(global_space, BsplineSpace):
        raise TypeError(
            f"distributed_space.global_space must be a BsplineSpace; "
            f"got {type(global_space).__name__!r}."
        )
    if kind not in get_args(QIKind):
        valid = ", ".join(repr(v) for v in get_args(QIKind))
        raise ValueError(f"Unknown kind {kind!r}; expected one of {valid}.")

    comm = distributed_space.comm
    local = distributed_space.local

    if local is not None:
        # local.space is BsplineSpace because global_space is BsplineSpace (checked above).
        assert isinstance(local.space, BsplineSpace)
        # Run serial LLM QI on the windowed local space.
        local_bspline = quasi_interpolate_bspline(func, local.space, kind=kind)
        # Flatten control points to (n_local_dofs, rank_dim); shape is (*num_basis, rank_dim).
        cp = np.asarray(local_bspline.control_points, dtype=np.float64)
        cp_flat = cp.reshape(local.space.num_total_basis, -1)
        # Restrict to owned DOFs and record their global indices.
        owned_mask = local.owned_dof_mask
        owned_global_dofs: npt.NDArray[np.int64] = local.local_to_global_dof[owned_mask]
        owned_coeffs: npt.NDArray[np.float64] = cp_flat[owned_mask]
    else:
        owned_global_dofs = np.empty(0, dtype=np.int64)
        owned_coeffs = np.empty((0, 0), dtype=np.float64)

    # Single MPI collective: each rank contributes its owned-DOF (index, coefficient) pairs.
    gathered: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]] = list(
        comm.allgather((owned_global_dofs, owned_coeffs))
    )

    # Determine rank_dim from the first non-empty contribution.
    rank_dim = 1
    for _, coeffs in gathered:
        c = np.asarray(coeffs)
        if c.ndim == 2 and c.shape[0] > 0:  # noqa: PLR2004
            rank_dim = int(c.shape[1])
            break

    # Assemble global control points from per-rank contributions.
    n_global = global_space.num_total_basis
    global_cp_flat = np.empty((n_global, rank_dim), dtype=global_space.dtype)
    for gdofs, coeffs in gathered:
        gdofs_arr = np.asarray(gdofs, dtype=np.int64)
        if gdofs_arr.size == 0:
            continue
        coeffs_arr = np.asarray(coeffs, dtype=global_space.dtype)
        global_cp_flat[gdofs_arr] = coeffs_arr.reshape(gdofs_arr.size, rank_dim)

    # Reshape to Bspline convention: (*num_basis, rank_dim).
    num_basis = tuple(global_space.num_basis)
    global_cp = global_cp_flat.reshape(*num_basis, rank_dim)

    global_bspline = Bspline(global_space, global_cp)
    return DistributedFunction(global_bspline, distributed_space)


__all__ = ["quasi_interpolate_bspline_distributed"]
