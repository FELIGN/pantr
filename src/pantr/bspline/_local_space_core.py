"""Numba kernels for distributed B-spline ownership and halo computation.

Layer-3 counterparts of :func:`~pantr.bspline.dof_owner` and
:func:`~pantr.bspline.compute_halo`. Both replace a Python loop over every global
degree of freedom (or every owned cell) with a compiled loop.

The ownership kernel relies on one ordering fact: a flat C-order cell id is monotone
in the lexicographic order of its multi-index, so traversing a support box in C-order
visits cells in strictly increasing flat-id order. The first active cell met is
therefore the smallest-id active cell, which is exactly the lex-first-active-cell
rule, and no minimum has to be accumulated.

Both kernels are deliberately **serial**. Measured on a 1e6-control-point 3D space at
degree 3, the worst case -- every cell inactive, so each dof walks its full 64-cell
support box -- takes 81 ms single-threaded against a 3 s budget, so threading buys
nothing that matters here. It would cost something: ``parallel=True`` kernels go
through Numba's workqueue threading layer, which aborts the process when entered
concurrently from two Python threads, and pantr runs an async JIT warmup thread at
import time. ``pantr/__init__.py`` documents that hazard as the reason parallel kernels
are kept out of the warmup list; keeping these serial removes the exposure from a
public API path instead of managing it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(nopython=True, cache=True)
def _dof_owner_core(  # noqa: PLR0913
    fc_flat: npt.NDArray[np.int64],
    lc_flat: npt.NDArray[np.int64],
    axis_starts: npt.NDArray[np.int64],
    num_intervals: npt.NDArray[np.int64],
    num_basis: npt.NDArray[np.int64],
    cell_owner: npt.NDArray[np.int32],
    dofs: npt.NDArray[np.int64],
    out: npt.NDArray[np.int32],
) -> None:
    """Resolve the owner rank of each requested degree of freedom.

    For each dof, walks its support box in C-order and takes the owner of the first
    active cell (``cell_owner >= 0``). A dof whose support holds no active cell gets
    ``-1``.

    Args:
        fc_flat (npt.NDArray[np.int64]): Per-axis first support cell of each basis
            function, axes concatenated.
        lc_flat (npt.NDArray[np.int64]): Per-axis last support cell of each basis
            function, axes concatenated.
        axis_starts (npt.NDArray[np.int64]): ``(dim + 1,)`` offsets of each axis into
            ``fc_flat`` and ``lc_flat``.
        num_intervals (npt.NDArray[np.int64]): ``(dim,)`` cells per direction.
        num_basis (npt.NDArray[np.int64]): ``(dim,)`` basis functions per direction.
        cell_owner (npt.NDArray[np.int32]): Owner rank of every cell, ``-1`` if
            inactive.
        dofs (npt.NDArray[np.int64]): Flat C-order dof ids to resolve.
        out (npt.NDArray[np.int32]): ``(dofs.size,)`` destination, written in full.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`~pantr.bspline.dof_owner_windowed` instead.
    """
    dim = num_basis.shape[0]

    # Row-major strides of the cell grid, so a box walk can accumulate flat ids.
    cell_strides = np.empty(dim, dtype=np.int64)
    stride = 1
    for axis in range(dim - 1, -1, -1):
        cell_strides[axis] = stride
        stride *= num_intervals[axis]

    for i in range(dofs.shape[0]):
        dof = dofs[i]
        owner = np.int32(-1)

        lo = np.empty(dim, dtype=np.int64)
        span = np.empty(dim, dtype=np.int64)
        counter = np.zeros(dim, dtype=np.int64)

        # Decode the dof's multi-index (C-order) and read its per-axis support range.
        remainder = dof
        for axis in range(dim - 1, -1, -1):
            index = remainder % num_basis[axis]
            remainder //= num_basis[axis]
            start = axis_starts[axis]
            lo[axis] = fc_flat[start + index]
            span[axis] = lc_flat[start + index] - fc_flat[start + index] + 1

        total = 1
        for axis in range(dim):
            total *= span[axis]

        base = 0
        for axis in range(dim):
            base += lo[axis] * cell_strides[axis]

        flat = base
        for _ in range(total):
            if cell_owner[flat] >= 0:
                owner = cell_owner[flat]
                break
            # Odometer over the box in C-order: advance the last axis fastest, so the
            # flat id increases monotonically and the first hit is the smallest id.
            for axis in range(dim - 1, -1, -1):
                counter[axis] += 1
                flat += cell_strides[axis]
                if counter[axis] < span[axis]:
                    break
                flat -= counter[axis] * cell_strides[axis]
                counter[axis] = 0

        out[i] = owner


@nb_jit(nopython=True, cache=True)
def _halo_mask_core(  # noqa: PLR0913
    lo_flat: npt.NDArray[np.int64],
    hi_flat: npt.NDArray[np.int64],
    axis_starts: npt.NDArray[np.int64],
    num_intervals: npt.NDArray[np.int64],
    owned_multi: npt.NDArray[np.int64],
    mask: npt.NDArray[np.bool_],
) -> None:
    """Mark every cell in the support closure of the owned cells.

    Args:
        lo_flat (npt.NDArray[np.int64]): Per-axis first cell of each interval's support
            closure, axes concatenated.
        hi_flat (npt.NDArray[np.int64]): Per-axis last cell of each interval's support
            closure, axes concatenated.
        axis_starts (npt.NDArray[np.int64]): ``(dim + 1,)`` offsets of each axis into
            ``lo_flat`` and ``hi_flat``.
        num_intervals (npt.NDArray[np.int64]): ``(dim,)`` cells per direction.
        owned_multi (npt.NDArray[np.int64]): ``(n_owned, dim)`` per-direction indices of
            the owned cells.
        mask (npt.NDArray[np.bool_]): ``(num_total_intervals,)`` flat mask, pre-zeroed;
            set to ``True`` on the closure.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`~pantr.bspline.compute_halo` instead.

        Writes are idempotent -- every write stores ``True`` -- so the closure does not
        depend on the order the owned cells are visited in.
    """
    dim = num_intervals.shape[0]

    cell_strides = np.empty(dim, dtype=np.int64)
    stride = 1
    for axis in range(dim - 1, -1, -1):
        cell_strides[axis] = stride
        stride *= num_intervals[axis]

    for i in range(owned_multi.shape[0]):
        lo = np.empty(dim, dtype=np.int64)
        span = np.empty(dim, dtype=np.int64)
        counter = np.zeros(dim, dtype=np.int64)

        for axis in range(dim):
            start = axis_starts[axis]
            index = owned_multi[i, axis]
            lo[axis] = lo_flat[start + index]
            span[axis] = hi_flat[start + index] - lo_flat[start + index] + 1

        total = 1
        for axis in range(dim):
            total *= span[axis]

        flat = 0
        for axis in range(dim):
            flat += lo[axis] * cell_strides[axis]

        for _ in range(total):
            mask[flat] = True
            for axis in range(dim - 1, -1, -1):
                counter[axis] += 1
                flat += cell_strides[axis]
                if counter[axis] < span[axis]:
                    break
                flat -= counter[axis] * cell_strides[axis]
                counter[axis] = 0
