"""Type stub for the `pantr.bspline` types bound in `cpp/bindings/bspline_types.cpp`.

Two registrations rather than one generic class, because the storage format is part
of the value: `pantr.bspline.BsplineSpace1D` stores whatever float dtype it is handed
and its `dtype` property is public, so the class of the handle is the only thing left
to carry it.

## Tensor-product extraction kernels


Bound by ``cpp/bindings/bspline_extraction.cpp``, over the four dimension-generic
functions in ``cpp/include/pantr/bspline/extraction_kernels.hpp``.

Twenty-four entry points rather than four, because the names, arities and
argument order mirror ``pantr.bspline._extraction_kernels`` exactly: the catalogue
in :mod:`pantr.bspline._extraction_backend` then selects between two functions
with one signature rather than between two conventions.

Every one of these is the **C++ half of Layer 2** in the layering of CLAUDE.md,
not Layer 3. The kernels behind them validate nothing, so each entry point checks
what a typed signature cannot express -- that the operand and output lengths are
the products of the operators' extents, that ``scratch`` is at least what the
stage sequence addresses, that ``out`` shares no memory with the operand, and that
a direction flagged identity is square. An unvalidated call would reach undefined
behaviour rather than an exception.

Two conventions differ from the sibling stubs and both are deliberate. There is no
``*`` before ``out``: the Layer 2 dispatcher calls these positionally, as it calls
the Numba kernels, so the bindings carry no ``nb::kw_only()``. And the identity
flags are plain ``bool`` while the batch forms take them as arrays, which is the
oracle's own split between a resolved cell and a batch of them.

See ``__init__.pyi`` for what this package promises and who has to keep it.
"""

import numpy as np
from numpy import typing as npt

class BsplineSpace1D32:
    """A ``float32`` 1D B-spline space owned by the C++ core.

    Wrapped by :class:`pantr.bspline.BsplineSpace1D`, which is the class a caller
    holds; this one is reached only through it. The constructor refuses a knot
    vector of any other dtype rather than casting it, because widening a
    ``float32`` vector into this class would change the space's tolerance by four
    orders and narrowing one into the ``float64`` class would move its knots.

    The knots are **copied** at construction and handed back, like every array
    below, as a **read-only view** of storage the space owns.

    Attributes:
        knots (npt.NDArray[np.float32]): The knot vector, read-only.
        degree (int): Polynomial degree, non-negative.
        periodic (bool): Whether the space is periodic.
        tolerance (float): Absolute parametric tolerance, a ``float`` at both
            storage widths.
        num_basis (int): Number of basis functions.
        num_intervals (int): Number of in-domain intervals, at least 1.
        domain (tuple[float, float]): The domain ends, as Python floats; the
            wrapper is what presents them as numpy scalars.
    """

    def __init__(
        self,
        knots: npt.NDArray[np.float32],
        degree: int,
        periodic: bool = False,
        snap_knots: bool = True,
    ) -> None: ...
    @property
    def knots(self) -> npt.NDArray[np.float32]: ...
    @property
    def degree(self) -> int: ...
    @property
    def periodic(self) -> bool: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def num_basis(self) -> int: ...
    @property
    def num_intervals(self) -> int: ...
    @property
    def domain(self) -> tuple[float, float]: ...
    def get_unique_knots_and_multiplicity(
        self, in_domain: bool = False
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    def first_basis_per_interval(self) -> npt.NDArray[np.int64]: ...
    def has_left_end_open(self) -> bool: ...
    def has_right_end_open(self) -> bool: ...
    def has_open_knots(self) -> bool: ...
    def has_Bezier_like_knots(self) -> bool: ...

class BsplineSpace1D64:
    """The ``float64`` twin of :class:`BsplineSpace1D32`; see it for what the two share.

    Attributes:
        knots (npt.NDArray[np.float64]): The knot vector, read-only.
        degree (int): Polynomial degree, non-negative.
        periodic (bool): Whether the space is periodic.
        tolerance (float): Absolute parametric tolerance.
        num_basis (int): Number of basis functions.
        num_intervals (int): Number of in-domain intervals, at least 1.
        domain (tuple[float, float]): The domain ends, as Python floats.
    """

    def __init__(
        self,
        knots: npt.NDArray[np.float64],
        degree: int,
        periodic: bool = False,
        snap_knots: bool = True,
    ) -> None: ...
    @property
    def knots(self) -> npt.NDArray[np.float64]: ...
    @property
    def degree(self) -> int: ...
    @property
    def periodic(self) -> bool: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def num_basis(self) -> int: ...
    @property
    def num_intervals(self) -> int: ...
    @property
    def domain(self) -> tuple[float, float]: ...
    def get_unique_knots_and_multiplicity(
        self, in_domain: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...
    def first_basis_per_interval(self) -> npt.NDArray[np.int64]: ...
    def has_left_end_open(self) -> bool: ...
    def has_right_end_open(self) -> bool: ...
    def has_open_knots(self) -> bool: ...
    def has_Bezier_like_knots(self) -> bool: ...

_Array = npt.NDArray[np.float32 | np.float64]
"""A float32 or float64 array; the two dtypes these kernels are built for."""

_Index = npt.NDArray[np.intp]
"""A compact index map or a block of per-direction cell indices."""

_Mask = npt.NDArray[np.bool_]
"""A per-direction identity mask."""

def apply_kron_1d(
    M_0: _Array,
    is_id_0: bool,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M @ v`` for one cell, ``M = kron(M_0, …, M_0)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        v (_Array): Input vector.
        out (_Array): Output vector. Must not share memory with ``v``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_2d(
    M_0: _Array,
    M_1: _Array,
    is_id_0: bool,
    is_id_1: bool,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M @ v`` for one cell, ``M = kron(M_0, …, M_1)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        v (_Array): Input vector.
        out (_Array): Output vector. Must not share memory with ``v``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_3d(
    M_0: _Array,
    M_1: _Array,
    M_2: _Array,
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M @ v`` for one cell, ``M = kron(M_0, …, M_2)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        M_2 (_Array): Direction 2's operator, shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_2 (bool): Whether direction 2 is the identity. Its operator's
            values are then unread and it must be square.
        v (_Array): Input vector.
        out (_Array): Output vector. Must not share memory with ``v``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_T_1d(
    M_0: _Array,
    is_id_0: bool,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M^T @ v`` for one cell, ``M = kron(M_0, …, M_0)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        v (_Array): Input vector.
        out (_Array): Output vector. Must not share memory with ``v``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_T_2d(
    M_0: _Array,
    M_1: _Array,
    is_id_0: bool,
    is_id_1: bool,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M^T @ v`` for one cell, ``M = kron(M_0, …, M_1)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        v (_Array): Input vector.
        out (_Array): Output vector. Must not share memory with ``v``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_T_3d(
    M_0: _Array,
    M_1: _Array,
    M_2: _Array,
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M^T @ v`` for one cell, ``M = kron(M_0, …, M_2)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        M_2 (_Array): Direction 2's operator, shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_2 (bool): Whether direction 2 is the identity. Its operator's
            values are then unread and it must be square.
        v (_Array): Input vector.
        out (_Array): Output vector. Must not share memory with ``v``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_MT_K_M_1d(
    M_0: _Array,
    is_id_0: bool,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M^T @ K @ M`` for one cell, ``M = kron(M_0, …, M_0)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        K (_Array): Input matrix, square.
        out (_Array): Output matrix, square. Must not share memory with ``K``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_MT_K_M_2d(
    M_0: _Array,
    M_1: _Array,
    is_id_0: bool,
    is_id_1: bool,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M^T @ K @ M`` for one cell, ``M = kron(M_0, …, M_1)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        K (_Array): Input matrix, square.
        out (_Array): Output matrix, square. Must not share memory with ``K``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_MT_K_M_3d(
    M_0: _Array,
    M_1: _Array,
    M_2: _Array,
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M^T @ K @ M`` for one cell, ``M = kron(M_0, …, M_2)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        M_2 (_Array): Direction 2's operator, shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_2 (bool): Whether direction 2 is the identity. Its operator's
            values are then unread and it must be square.
        K (_Array): Input matrix, square.
        out (_Array): Output matrix, square. Must not share memory with ``K``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_M_K_MT_1d(
    M_0: _Array,
    is_id_0: bool,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M @ K @ M^T`` for one cell, ``M = kron(M_0, …, M_0)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        K (_Array): Input matrix, square.
        out (_Array): Output matrix, square. Must not share memory with ``K``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_M_K_MT_2d(
    M_0: _Array,
    M_1: _Array,
    is_id_0: bool,
    is_id_1: bool,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M @ K @ M^T`` for one cell, ``M = kron(M_0, …, M_1)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        K (_Array): Input matrix, square.
        out (_Array): Output matrix, square. Must not share memory with ``K``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_M_K_MT_3d(
    M_0: _Array,
    M_1: _Array,
    M_2: _Array,
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out = M @ K @ M^T`` for one cell, ``M = kron(M_0, …, M_2)``.

    Args:
        M_0 (_Array): Direction 0's operator, shape ``(n_out_0, n_in_0)``.
        M_1 (_Array): Direction 1's operator, shape ``(n_out_1, n_in_1)``.
        M_2 (_Array): Direction 2's operator, shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Whether direction 0 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_1 (bool): Whether direction 1 is the identity. Its operator's
            values are then unread and it must be square.
        is_id_2 (bool): Whether direction 2 is the identity. Its operator's
            values are then unread and it must be square.
        K (_Array): Input matrix, square.
        out (_Array): Output matrix, square. Must not share memory with ``K``.
        scratch (_Array): Work buffer, at least the size
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            returns.

    Raises:
        ValueError: If an extent, a length or the scratch size is inconsistent,
            if ``out`` overlaps the operand, or if a direction flagged identity
            is not square.
    """

def apply_kron_apply_many_1d(
    ops_0: _Array,
    idx_map_0: _Index,
    is_id_0: _Mask,
    cell_indices: _Index,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c @ v[c]`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 1)``.
        v (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``v``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_apply_many_2d(
    ops_0: _Array,
    ops_1: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    cell_indices: _Index,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c @ v[c]`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 2)``.
        v (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``v``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_apply_many_3d(
    ops_0: _Array,
    ops_1: _Array,
    ops_2: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    idx_map_2: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    is_id_2: _Mask,
    cell_indices: _Index,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c @ v[c]`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        ops_2 (_Array): Direction 2's compact operator stack, shape
            ``(n_compact_2, n_out_2, n_in_2)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        idx_map_2 (_Index): Direction 2's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        is_id_2 (_Mask): Direction 2's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 3)``.
        v (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``v``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_apply_T_many_1d(
    ops_0: _Array,
    idx_map_0: _Index,
    is_id_0: _Mask,
    cell_indices: _Index,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c^T @ v[c]`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 1)``.
        v (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``v``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_apply_T_many_2d(
    ops_0: _Array,
    ops_1: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    cell_indices: _Index,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c^T @ v[c]`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 2)``.
        v (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``v``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_apply_T_many_3d(
    ops_0: _Array,
    ops_1: _Array,
    ops_2: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    idx_map_2: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    is_id_2: _Mask,
    cell_indices: _Index,
    v: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c^T @ v[c]`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        ops_2 (_Array): Direction 2's compact operator stack, shape
            ``(n_compact_2, n_out_2, n_in_2)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        idx_map_2 (_Index): Direction 2's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        is_id_2 (_Mask): Direction 2's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 3)``.
        v (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``v``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_MT_K_M_many_1d(
    ops_0: _Array,
    idx_map_0: _Index,
    is_id_0: _Mask,
    cell_indices: _Index,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c^T @ K[c] @ M_c`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 1)``.
        K (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``K``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_MT_K_M_many_2d(
    ops_0: _Array,
    ops_1: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    cell_indices: _Index,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c^T @ K[c] @ M_c`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 2)``.
        K (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``K``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_MT_K_M_many_3d(
    ops_0: _Array,
    ops_1: _Array,
    ops_2: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    idx_map_2: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    is_id_2: _Mask,
    cell_indices: _Index,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c^T @ K[c] @ M_c`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        ops_2 (_Array): Direction 2's compact operator stack, shape
            ``(n_compact_2, n_out_2, n_in_2)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        idx_map_2 (_Index): Direction 2's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        is_id_2 (_Mask): Direction 2's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 3)``.
        K (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``K``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_M_K_MT_many_1d(
    ops_0: _Array,
    idx_map_0: _Index,
    is_id_0: _Mask,
    cell_indices: _Index,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c @ K[c] @ M_c^T`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 1)``.
        K (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``K``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_M_K_MT_many_2d(
    ops_0: _Array,
    ops_1: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    cell_indices: _Index,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c @ K[c] @ M_c^T`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 2)``.
        K (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``K``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """

def apply_kron_M_K_MT_many_3d(
    ops_0: _Array,
    ops_1: _Array,
    ops_2: _Array,
    idx_map_0: _Index,
    idx_map_1: _Index,
    idx_map_2: _Index,
    is_id_0: _Mask,
    is_id_1: _Mask,
    is_id_2: _Mask,
    cell_indices: _Index,
    K: _Array,
    out: _Array,
    scratch: _Array,
) -> None:
    """Compute ``out[c] = M_c @ K[c] @ M_c^T`` for every cell in the batch.

    Serial, where the oracle's counterpart is ``parallel=True`` over cells. Each
    cell writes only its own rows and there is no reduction, so the answer cannot
    move with the thread count either way; only the speed does.

    Args:
        ops_0 (_Array): Direction 0's compact operator stack, shape
            ``(n_compact_0, n_out_0, n_in_0)``.
        ops_1 (_Array): Direction 1's compact operator stack, shape
            ``(n_compact_1, n_out_1, n_in_1)``.
        ops_2 (_Array): Direction 2's compact operator stack, shape
            ``(n_compact_2, n_out_2, n_in_2)``.
        idx_map_0 (_Index): Direction 0's compact index map.
        idx_map_1 (_Index): Direction 1's compact index map.
        idx_map_2 (_Index): Direction 2's compact index map.
        is_id_0 (_Mask): Direction 0's identity mask.
        is_id_1 (_Mask): Direction 1's identity mask.
        is_id_2 (_Mask): Direction 2's identity mask.
        cell_indices (_Index): Per-direction indices, shape ``(n_cells, 3)``.
        K (_Array): Operands, one per cell.
        out (_Array): Results, one per cell. Must not share memory with ``K``.
        scratch (_Array): Work buffers, one row per cell.

    Raises:
        ValueError: If a shape, a row count, a cell index or the scratch size is
            inconsistent, or if ``out`` overlaps the operand.
    """
