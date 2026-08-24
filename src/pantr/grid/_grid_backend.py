"""Which implementation of a grid kernel runs: the Numba one or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for
:mod:`pantr.grid`, and it lives here rather than beside the policy for the reason
that module's own docstring gives: a catalogue imports the kernels it hands out, so
keeping each one beside its own kernels is what stops the policy module from
importing the library it has to stay independent of.

Eight accessors over nine kernels
---------------------------------

The ninth is :func:`~pantr.grid._hier_core._block_of_midx`, and it is absent
structurally rather than by oversight. On the Numba side it is called from inside
:func:`~pantr.grid._hier_core._hier_locate_points_core` at a ``nopython`` call
site, and no dispatch can be inserted between two Numba kernels, so the boundary is
forced up to the eight places Layer 2 reaches for a kernel. Its C++ counterpart is
reachable under another name: the oracle's :func:`_encode_midx_core` is a one-line
call to it and *is* a Layer 2 seam, so the two are one function in C++ and the
accessor carries the wrapper's name.

Eight bare callables and no record, which is the rule
``design/cross_backend_types.md`` states: a record when a consumer needs more than
one kernel at once, a bare callable when it does not. The BVH looks like the
exception and is not. :meth:`pantr.grid.BVH.query_aabb` uses count and then emit in
one call, but it needs the count's *answer* to size the emit's output, so the two
are sequential rather than simultaneous and nothing is served by handing them over
together.

Every accessor adapts, and that is the price of the keyword-only bindings
-------------------------------------------------------------------------

The C++ signatures are almost entirely keyword-only, deliberately: they are full of
adjacent, same-typed, mutually unordered ``int64`` descriptor arrays --
``block_lo``/``block_hi``, ``node_left``/``node_right``,
``knot_starts``/``cells_per_axis``/``strides`` -- and every one of them type-checks,
runs and returns a plausible wrong answer if transposed. ``cpp/bindings/grid.cpp``
says why at length, and ``bezier_root_finding.cpp`` records two such traps found by
an audit rather than by a test.

The Numba kernels take those arrays positionally. So each accessor below wraps a
thin positional-to-keyword adapter, which is what lets the Layer 2 call sites keep
one call shape for both backends. Eight three-line adapters is the cost; a
transposition that runs is what it buys.

The adapters do **not** wrap output arrays in :func:`numpy.ascontiguousarray`, and
that omission is the point. A non-contiguous ``out`` would be silently copied,
written and discarded, which is exactly the failure ``.noconvert()`` exists to
prevent on the C++ side. Leaving it unwrapped means the binding rejects it loudly
instead. Input arrays are wrapped, where a copy is harmless and Layer 2 has usually
made one unnecessary anyway.

What crosses the boundary
-------------------------

Arrays and scalars only. Every coordinate array is ``float64`` and every descriptor
and output array is ``int64``, in both backends: the oracle is ``float64``-only --
none of the three kernel modules mentions ``float32`` -- so unlike
:mod:`pantr.bezier` there is no dtype axis here and the C++ registers no ``float``
overload. ``design/backend_parity.md`` Rule 8 is the reason it should not: a parity
claim is only defined where the comparison can say something, and a ``float32``
surface would have no oracle behind it.

- :func:`locate_points_kernel`, :func:`bvh_build_kernel`,
  :func:`bvh_query_count_kernel`, :func:`bvh_query_emit_kernel`,
  :func:`encode_midx_kernel`, :func:`decode_flat_id_kernel`,
  :func:`hier_locate_points_kernel`, :func:`hier_collect_cell_bounds_kernel`:
  the accessors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ._bvh_core import _bvh_build_core, _bvh_query_count_core, _bvh_query_emit_core
from ._hier_core import (
    _decode_flat_id_core,
    _encode_midx_core,
    _hier_collect_cell_bounds_core,
    _hier_locate_points_core,
)
from ._locate_core import _locate_points_core

_K = TypeVar("_K")
"""One kernel's signature, so :func:`_select` returns the type it was given."""

_Coords = npt.NDArray[np.float64]
"""A coordinate array. Always float64: the grid kernels have no dtype axis."""

_Index = npt.NDArray[np.int64]
"""A descriptor or output array of cell indices."""

_LocateFunc = Callable[[_Coords, _Coords, _Index, _Index, _Index, _Index], None]
"""``(points, knots_flat, knot_starts, cells_per_axis, strides, out) -> None``."""

_BvhBuildFunc = Callable[[_Coords, _Coords, _Coords, _Coords, _Index, _Index, _Index], None]
"""``(cell_lo, cell_hi, node_lo, node_hi, node_left, node_right, node_cell) -> None``."""

_BvhCountFunc = Callable[[_Coords, _Coords, _Coords, _Coords, _Index, _Index, _Index], int]
"""``(qlo, qhi, node_lo, node_hi, node_left, node_right, node_cell) -> count``."""

_BvhEmitFunc = Callable[[_Coords, _Coords, _Coords, _Coords, _Index, _Index, _Index, _Index], int]
"""As :data:`_BvhCountFunc`, plus a trailing ``out`` array; returns the count."""

_EncodeFunc = Callable[[int, _Index, _Index, _Index, _Index, _Index], int]
"""``(level, midx, block_lo, block_hi, block_base, level_block_start) -> cid``."""

_DecodeFunc = Callable[[int, _Index, _Index, _Index, _Index, _Index], int]
"""``(cid, block_lo, block_hi, block_base, level_block_start, out_midx) -> level``."""

_HierLocateFunc = Callable[
    [_Coords, _Coords, _Index, _Index, _Index, _Index, _Index, _Index, _Index, _Index], None
]
"""The hierarchical locate: points, the root descriptor, the block descriptor, ``out``."""

_HierBoundsFunc = Callable[
    [_Coords, _Index, _Index, _Index, _Index, _Index, _Index, _Coords, _Coords], None
]
"""The bounds collector: the root and block descriptors, then ``out_lo`` and ``out_hi``."""


def _select(backend: Backend | None, python_kernel: _K, cpp_kernel: _K) -> _K:
    """Pick one kernel's implementation for the requested backend.

    The one place the never-fall-back rule is applied for this package, so the
    accessors below cannot drift from each other on it.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
        python_kernel (_K): The Numba implementation, always available.
        cpp_kernel (_K): The C++ implementation, available only when the extension
            was built.

    Returns:
        _K: The kernel of the chosen backend.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        return python_kernel

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return cpp_kernel


def _cpp_locate_points(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    points: _Coords,
    knots_flat: _Coords,
    knot_starts: _Index,
    cells_per_axis: _Index,
    strides: _Index,
    out: _Index,
) -> None:
    """Locate a batch of points through the C++ binding.

    Args:
        points (_Coords): Query points, shape ``(npts, ndim)``.
        knots_flat (_Coords): Concatenated per-axis knot vectors.
        knot_starts (_Index): Per-axis offset into ``knots_flat``.
        cells_per_axis (_Index): Per-axis cell counts.
        strides (_Index): Per-axis C-order flat strides.
        out (_Index): Output flat cell ids, shape ``(npts,)``.

    Note:
        No input validation is performed here. Shape and dtype were established by
        Layer 2; the binding re-checks dtype, rank, contiguity and the descriptors'
        mutual consistency.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    _pantr_cpp.locate_points(
        np.ascontiguousarray(points),
        knots_flat=np.ascontiguousarray(knots_flat),
        knot_starts=knot_starts,
        cells_per_axis=cells_per_axis,
        strides=strides,
        out=out,
    )


def _cpp_bvh_build(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    cell_lo: _Coords,
    cell_hi: _Coords,
    node_lo: _Coords,
    node_hi: _Coords,
    node_left: _Index,
    node_right: _Index,
    node_cell: _Index,
) -> None:
    """Build the BVH node arrays through the C++ binding.

    Args:
        cell_lo (_Coords): Per-cell lo corners, shape ``(n_cells, ndim)``.
        cell_hi (_Coords): Per-cell hi corners, same shape.
        node_lo (_Coords): Output per-node lo corners.
        node_hi (_Coords): Output per-node hi corners.
        node_left (_Index): Output left-child indices.
        node_right (_Index): Output right-child indices.
        node_cell (_Index): Output per-leaf cell ids.

    Note:
        No input validation is performed here. The binding checks the shapes, that
        every cell has ``hi >= lo`` and finite corners, and that the cell count
        cannot produce a tree deeper than the traversal stack allows.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    _pantr_cpp.bvh_build(
        cell_lo=np.ascontiguousarray(cell_lo),
        cell_hi=np.ascontiguousarray(cell_hi),
        node_lo=node_lo,
        node_hi=node_hi,
        node_left=node_left,
        node_right=node_right,
        node_cell=node_cell,
    )


def _cpp_bvh_query_count(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    qlo: _Coords,
    qhi: _Coords,
    node_lo: _Coords,
    node_hi: _Coords,
    node_left: _Index,
    node_right: _Index,
    node_cell: _Index,
) -> int:
    """Count the overlapping leaves through the C++ binding.

    Args:
        qlo (_Coords): Query box lo corner, shape ``(ndim,)``.
        qhi (_Coords): Query box hi corner, same shape.
        node_lo (_Coords): Per-node lo corners.
        node_hi (_Coords): Per-node hi corners.
        node_left (_Index): Left-child indices.
        node_right (_Index): Right-child indices.
        node_cell (_Index): Per-leaf cell ids.

    Returns:
        int: The number of overlapping leaves.

    Note:
        No input validation is performed here, and in particular the tree's depth is
        not revalidated: :class:`pantr.grid.BVH` establishes it once at
        construction, which is also where the oracle establishes it.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.bvh_query_count(
        qlo=np.ascontiguousarray(qlo),
        qhi=np.ascontiguousarray(qhi),
        node_lo=node_lo,
        node_hi=node_hi,
        node_left=node_left,
        node_right=node_right,
        node_cell=node_cell,
    )


def _cpp_bvh_query_emit(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    qlo: _Coords,
    qhi: _Coords,
    node_lo: _Coords,
    node_hi: _Coords,
    node_left: _Index,
    node_right: _Index,
    node_cell: _Index,
    out: _Index,
) -> int:
    """Write the overlapping cell ids through the C++ binding.

    Args:
        qlo (_Coords): Query box lo corner, shape ``(ndim,)``.
        qhi (_Coords): Query box hi corner, same shape.
        node_lo (_Coords): Per-node lo corners.
        node_hi (_Coords): Per-node hi corners.
        node_left (_Index): Left-child indices.
        node_right (_Index): Right-child indices.
        node_cell (_Index): Per-leaf cell ids.
        out (_Index): Output cell ids, sized from a prior count.

    Returns:
        int: The number of overlapping leaves.

    Note:
        No input validation is performed here. Unlike the oracle, the binding raises
        rather than corrupting memory when ``out`` cannot hold the matches.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.bvh_query_emit(
        qlo=np.ascontiguousarray(qlo),
        qhi=np.ascontiguousarray(qhi),
        node_lo=node_lo,
        node_hi=node_hi,
        node_left=node_left,
        node_right=node_right,
        node_cell=node_cell,
        out=out,
    )


def _cpp_encode_midx(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    level: int,
    midx: _Index,
    block_lo: _Index,
    block_hi: _Index,
    block_base: _Index,
    level_block_start: _Index,
) -> int:
    """Encode a ``(level, midx)`` position through the C++ binding.

    Args:
        level (int): Hierarchy level of the queried position.
        midx (_Index): Per-axis index in level coordinates.
        block_lo (_Index): Packed block lower bounds.
        block_hi (_Index): Packed block upper bounds.
        block_base (_Index): Flat-id base per block.
        level_block_start (_Index): Block index range per level.

    Returns:
        int: The flat cell id, or ``-1`` when the position is not an active leaf.

    Note:
        No input validation is performed here. The binding checks the block
        descriptor's self-consistency and that ``level`` is in range.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.encode_midx(
        level,
        midx=np.ascontiguousarray(midx),
        block_lo=block_lo,
        block_hi=block_hi,
        block_base=block_base,
        level_block_start=level_block_start,
    )


def _cpp_decode_flat_id(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    cid: int,
    block_lo: _Index,
    block_hi: _Index,
    block_base: _Index,
    level_block_start: _Index,
    out_midx: _Index,
) -> int:
    """Decode a flat cell id through the C++ binding.

    Args:
        cid (int): Flat cell id.
        block_lo (_Index): Packed block lower bounds.
        block_hi (_Index): Packed block upper bounds.
        block_base (_Index): Flat-id base per block.
        level_block_start (_Index): Block index range per level.
        out_midx (_Index): Output per-axis index, shape ``(ndim,)``.

    Returns:
        int: The cell's level.

    Note:
        No input validation is performed here. The binding checks the block
        descriptor and refuses a ``cid`` below the first block's base, which the
        oracle's upper-bound search would index as ``-1``.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.decode_flat_id(
        cid,
        block_lo=block_lo,
        block_hi=block_hi,
        block_base=block_base,
        level_block_start=level_block_start,
        out_midx=out_midx,
    )


def _cpp_hier_locate_points(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    points: _Coords,
    knots_flat: _Coords,
    knot_starts: _Index,
    root_cells_per_axis: _Index,
    factor: _Index,
    block_lo: _Index,
    block_hi: _Index,
    block_base: _Index,
    level_block_start: _Index,
    out: _Index,
) -> None:
    """Locate a batch of points on a hierarchical grid through the C++ binding.

    Args:
        points (_Coords): Query points, shape ``(npts, ndim)``.
        knots_flat (_Coords): Root per-axis breakpoints, concatenated.
        knot_starts (_Index): Per-axis offset into ``knots_flat``.
        root_cells_per_axis (_Index): Per-axis root cell counts.
        factor (_Index): Per-axis subdivision factor.
        block_lo (_Index): Packed block lower bounds.
        block_hi (_Index): Packed block upper bounds.
        block_base (_Index): Flat-id base per block.
        level_block_start (_Index): Block index range per level.
        out (_Index): Output flat cell ids, shape ``(npts,)``.

    Note:
        No input validation is performed here. The binding checks the descriptors'
        mutual consistency, the knot ranges and that every subdivision factor is at
        least two.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    _pantr_cpp.hier_locate_points(
        np.ascontiguousarray(points),
        knots_flat=np.ascontiguousarray(knots_flat),
        knot_starts=knot_starts,
        root_cells_per_axis=root_cells_per_axis,
        factor=factor,
        block_lo=block_lo,
        block_hi=block_hi,
        block_base=block_base,
        level_block_start=level_block_start,
        out=out,
    )


def _cpp_hier_collect_cell_bounds(  # noqa: PLR0913, PLR0917 -- mirrors the kernel's flat argument list
    knots_flat: _Coords,
    knot_starts: _Index,
    factor: _Index,
    block_lo: _Index,
    block_hi: _Index,
    block_base: _Index,
    level_block_start: _Index,
    out_lo: _Coords,
    out_hi: _Coords,
) -> None:
    """Materialize per-cell bounds through the C++ binding.

    Args:
        knots_flat (_Coords): Root per-axis breakpoints, concatenated.
        knot_starts (_Index): Per-axis offset into ``knots_flat``.
        factor (_Index): Per-axis subdivision factor.
        block_lo (_Index): Packed block lower bounds.
        block_hi (_Index): Packed block upper bounds.
        block_base (_Index): Flat-id base per block.
        level_block_start (_Index): Block index range per level.
        out_lo (_Coords): Output lower corners, shape ``(num_cells, ndim)``.
        out_hi (_Coords): Output upper corners, same shape.

    Note:
        No input validation is performed here. The binding computes, per block, both
        the output rows and the largest root cell the block reaches, since either
        one being short is an out-of-bounds access rather than a wrong answer.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    _pantr_cpp.hier_collect_cell_bounds(
        knots_flat=np.ascontiguousarray(knots_flat),
        knot_starts=knot_starts,
        factor=factor,
        block_lo=block_lo,
        block_hi=block_hi,
        block_base=block_base,
        level_block_start=level_block_start,
        out_lo=out_lo,
        out_hi=out_hi,
    )


def locate_points_kernel(backend: Backend | None = None) -> _LocateFunc:
    """Get the batch point-location kernel for a tensor-product grid.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _LocateFunc: ``(points, knots_flat, knot_starts, cells_per_axis, strides,
            out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _locate_points_core, _cpp_locate_points)


def bvh_build_kernel(backend: Backend | None = None) -> _BvhBuildFunc:
    """Get the BVH construction kernel.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _BvhBuildFunc: ``(cell_lo, cell_hi, node_lo, node_hi, node_left,
            node_right, node_cell) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _bvh_build_core, _cpp_bvh_build)


def bvh_query_count_kernel(backend: Backend | None = None) -> _BvhCountFunc:
    """Get the BVH overlap-counting kernel.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _BvhCountFunc: ``(qlo, qhi, node_lo, node_hi, node_left, node_right,
            node_cell) -> count``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _bvh_query_count_core, _cpp_bvh_query_count)


def bvh_query_emit_kernel(backend: Backend | None = None) -> _BvhEmitFunc:
    """Get the BVH overlap-emitting kernel.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _BvhEmitFunc: as :func:`bvh_query_count_kernel`'s, plus a trailing ``out``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _bvh_query_emit_core, _cpp_bvh_query_emit)


def encode_midx_kernel(backend: Backend | None = None) -> _EncodeFunc:
    """Get the hierarchical position-to-flat-id kernel.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _EncodeFunc: ``(level, midx, block_lo, block_hi, block_base,
            level_block_start) -> cid``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _encode_midx_core, _cpp_encode_midx)


def decode_flat_id_kernel(backend: Backend | None = None) -> _DecodeFunc:
    """Get the hierarchical flat-id-to-position kernel.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _DecodeFunc: ``(cid, block_lo, block_hi, block_base, level_block_start,
            out_midx) -> level``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _decode_flat_id_core, _cpp_decode_flat_id)


def hier_locate_points_kernel(backend: Backend | None = None) -> _HierLocateFunc:
    """Get the batch point-location kernel for a hierarchical grid.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _HierLocateFunc: the hierarchical locate kernel.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _hier_locate_points_core, _cpp_hier_locate_points)


def hier_collect_cell_bounds_kernel(backend: Backend | None = None) -> _HierBoundsFunc:
    """Get the hierarchical per-cell bounds collector.

    Args:
        backend (Backend | None): Backend to use, or ``None`` for the active one.

    Returns:
        _HierBoundsFunc: the bounds-collecting kernel.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _hier_collect_cell_bounds_core, _cpp_hier_collect_cell_bounds)


__all__ = [
    "bvh_build_kernel",
    "bvh_query_count_kernel",
    "bvh_query_emit_kernel",
    "decode_flat_id_kernel",
    "encode_midx_kernel",
    "hier_collect_cell_bounds_kernel",
    "hier_locate_points_kernel",
    "locate_points_kernel",
]
