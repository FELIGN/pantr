"""Tensor-product change-of-basis extraction across B-spline elements.

This module exposes :class:`SpanwiseElementExtraction`, a lazy tensor-product
change-of-basis object. It eagerly caches the per-direction 1D extraction
operators once at construction time and, on demand, dispatches to the Layer-3
Kronecker kernels in ``pantr.bspline._extraction_kernels`` to apply the
d-dimensional operator for a single element.

Three targets are supported (the source basis is always the B-spline basis):

- ``"bezier"``:   Bernstein (Bézier) basis on each element.
- ``"lagrange"``: Lagrange basis on each element, at the chosen point
  distribution (see :class:`pantr.basis.LagrangeVariant`).
- ``"cardinal"``: cardinal B-spline basis on each element.

Identity short-circuit is used wherever possible: for ``"cardinal"`` the
structural mask from :meth:`BsplineSpace1D.get_cardinal_intervals` labels each
interval whose knot spans are uniform (a *cardinal* interval); on such an
interval the cardinal extraction operator is exactly the identity matrix. For
``"bezier"`` and ``"lagrange"`` a per-element numerical test
``|C - I|_max < tol`` marks trivial elements (for instance, a C⁰ Bézier space
has identity Bézier extraction everywhere).
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import numpy.typing as npt

from ..basis import LagrangeVariant
from ..basis._basis_utils import _allocate_or_validate_out
from ._extraction_helpers import (
    OpKind,
    _operation_shapes,
    _prepare_apply_call,
    _prepare_apply_many_call,
)

if TYPE_CHECKING:
    from ._bspline_space_nd import BsplineSpace


Target = Literal["bezier", "lagrange", "cardinal"]
"""Supported target bases for spanwise element extraction."""


CellIndex = int | tuple[int, ...] | list[int] | npt.NDArray[np.int_]
"""Accepted cell-index forms: flat ``int``, or a per-direction integer sequence."""

CellIndicesBatch = npt.NDArray[np.int_] | list[int] | list[tuple[int, ...]] | list[list[int]]
"""Accepted batch cell-index forms.

May be:

- 1-D integer array or list of ``n_cells`` flat indices (row-major over
  :attr:`~SpanwiseElementExtraction.num_intervals`).
- 2-D integer array of shape ``(n_cells, d)`` with per-direction indices.
- List of per-direction integer tuples or lists of length ``d``.
"""


class SpanwiseElementExtraction:
    """Tensor-product change-of-basis operator across B-spline elements.

    For a :class:`BsplineSpace` of dimension ``d`` and a chosen ``target``
    basis, this class eagerly builds the per-direction 1D extraction operator
    arrays ``ops_1d[k]`` of shape ``(n_elements_k, n_out_k, n_in_k)``. Per-
    element d-dimensional operators are never materialized unless explicitly
    requested via :meth:`operator` or :meth:`tabulate`: instead the apply-style
    methods dispatch to the matrix-free Kronecker kernels in
    ``pantr.bspline._extraction_kernels``.

    With the current 1D builders all per-direction operators are square of
    size ``(degree_k + 1, degree_k + 1)``. The class also supports non-square
    per-direction operators, so new 1D builders can plug in without changes.

    The per-direction data is also exposed as tuples of NumPy arrays
    (:attr:`ops_1d`, :attr:`is_identity_mask_1d`) so that downstream Numba
    code can consume the raw arrays and call the Layer-3 kernels directly.

    Attributes:
        _space (BsplineSpace): Underlying multi-dimensional B-spline space.
        _target (Target): Target basis tag.
        _lagrange_variant (LagrangeVariant): Point distribution used when
            ``target == "lagrange"``; ignored otherwise.
        _identity_tol (float): Numerical tolerance used for identity
            detection on ``"bezier"`` and ``"lagrange"`` targets.
        _ops_1d (tuple[npt.NDArray[np.float32 | np.float64], ...]):
            Per-direction 3D operator arrays of shape
            ``(n_elements_k, n_out_k, n_in_k)``.
        _is_identity_mask_1d (tuple[npt.NDArray[bool], ...]): Per-direction
            identity masks of shape ``(n_elements_k,)``.
    """

    _space: BsplineSpace
    _target: Target
    _lagrange_variant: LagrangeVariant
    _identity_tol: float
    _ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...]
    _is_identity_mask_1d: tuple[npt.NDArray[np.bool_], ...]

    def __init__(
        self,
        space: BsplineSpace,
        target: Target,
        *,
        lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
        identity_tol: float | None = None,
    ) -> None:
        """Build the per-direction operators and identity masks.

        Args:
            space (BsplineSpace): Multi-dimensional B-spline space.
            target (Target): One of ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
            lagrange_variant (LagrangeVariant): Point distribution used when
                ``target == "lagrange"``. Defaults to
                :attr:`pantr.basis.LagrangeVariant.EQUISPACES`.
            identity_tol (float | None): Absolute tolerance for numerical
                identity detection on ``"bezier"`` and ``"lagrange"`` targets.
                If ``None``, defaults to ``space.tolerance``.

        Raises:
            ValueError: If ``target`` is not a recognized tag.
            ValueError: If ``identity_tol`` is negative or NaN.
            NotImplementedError: If any direction of ``space`` is periodic;
                periodic support is deferred to a later version.
        """
        if target not in get_args(Target):
            valid = ", ".join(repr(v) for v in get_args(Target))
            raise ValueError(f"Unknown target {target!r}; expected one of {valid}")

        if any(s.periodic for s in space.spaces):
            raise NotImplementedError(
                "SpanwiseElementExtraction does not yet support periodic directions. "
                "Convert the B-spline to open form first (see Bspline.to_open_bspline)."
            )

        self._space = space
        self._target = target
        self._lagrange_variant = lagrange_variant
        tol = float(space.tolerance) if identity_tol is None else float(identity_tol)
        if not (tol >= 0.0):
            raise ValueError(f"identity_tol must be non-negative; got {tol!r}")
        self._identity_tol = tol

        ops_1d: list[npt.NDArray[np.float32 | np.float64]] = []
        masks_1d: list[npt.NDArray[np.bool_]] = []
        for space_1d in space.spaces:
            if target == "bezier":
                ops = space_1d.tabulate_Bezier_extraction_operators()
                mask = _numerical_identity_mask(ops, self._identity_tol)
            elif target == "lagrange":
                ops = space_1d.tabulate_Lagrange_extraction_operators(
                    lagrange_variant=lagrange_variant
                )
                mask = _numerical_identity_mask(ops, self._identity_tol)
            else:  # target == "cardinal"
                ops = space_1d.tabulate_cardinal_extraction_operators()
                mask = space_1d.get_cardinal_intervals()
            ops.flags.writeable = False
            mask.flags.writeable = False
            ops_1d.append(ops)
            masks_1d.append(mask)

        self._ops_1d = tuple(ops_1d)
        self._is_identity_mask_1d = tuple(masks_1d)

        if len(self._ops_1d) > 1:
            dtype_0 = self._ops_1d[0].dtype
            for k, _ops in enumerate(self._ops_1d[1:], start=1):
                if _ops.dtype != dtype_0:
                    raise ValueError(
                        f"Per-direction operators have inconsistent dtypes: "
                        f"ops_1d[0].dtype={dtype_0}, ops_1d[{k}].dtype={_ops.dtype}"
                    )

    # ---------------------------------------------------------------- properties

    @property
    def space(self) -> BsplineSpace:
        """Get the underlying B-spline space.

        Returns:
            BsplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def target(self) -> Target:
        """Get the target basis tag.

        Returns:
            Target: One of ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
        """
        return self._target

    @property
    def lagrange_variant(self) -> LagrangeVariant:
        """Get the Lagrange point distribution used for ``"lagrange"`` target.

        Returns:
            LagrangeVariant: The point distribution. Meaningless for other targets.
        """
        return self._lagrange_variant

    @property
    def identity_tol(self) -> float:
        """Get the tolerance used for numerical identity detection.

        Returns:
            float: The absolute tolerance.
        """
        return self._identity_tol

    @property
    def dim(self) -> int:
        """Get the number of tensor-product directions.

        Returns:
            int: The dimension ``d`` of the space.
        """
        return self._space.dim

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype shared by all operators.

        Returns:
            npt.DTypeLike: The dtype inherited from the space (``float32`` or ``float64``).
        """
        return self._space.dtype

    @property
    def num_intervals(self) -> tuple[int, ...]:
        """Get the per-direction number of elements (intervals).

        Returns:
            tuple[int, ...]: Length-``d`` tuple ``(n_elements_0, …, n_elements_{d-1})``.
        """
        return self._space.num_intervals

    @property
    def num_total_intervals(self) -> int:
        """Get the total number of elements across the tensor-product grid.

        Returns:
            int: ``prod(num_intervals)``.
        """
        return self._space.num_total_intervals

    @property
    def ops_1d(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the per-direction 1D operator arrays.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: Length-``d`` tuple
            of read-only 3D arrays; ``ops_1d[k]`` has shape
            ``(n_elements_k, n_out_k, n_in_k)``. Intended for consumption by
            downstream ``@njit`` code.
        """
        return self._ops_1d

    @property
    def is_identity_mask_1d(self) -> tuple[npt.NDArray[np.bool_], ...]:
        """Get the per-direction identity masks.

        For the ``"cardinal"`` target each entry reflects whether the interval
        has a uniform (cardinal) knot span, which is exactly when the cardinal
        extraction operator is the identity matrix. For ``"bezier"`` and
        ``"lagrange"`` targets the mask is computed by a numerical
        ``|C - I|_max < identity_tol`` test per element.

        Returns:
            tuple[npt.NDArray[bool], ...]: Length-``d`` tuple of read-only
            1D boolean arrays; ``is_identity_mask_1d[k][i]`` is ``True`` iff
            the ``i``-th element in direction ``k`` has an identity operator.
        """
        return self._is_identity_mask_1d

    @property
    def input_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction input sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_in_0, …, n_in_{d-1})``.
        """
        return tuple(int(ops.shape[2]) for ops in self._ops_1d)

    @property
    def output_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction output sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_out_0, …, n_out_{d-1})``.
        """
        return tuple(int(ops.shape[1]) for ops in self._ops_1d)

    # ---------------------------------------------------------------- identity queries

    def is_identity_at(self, cell_idx: CellIndex) -> bool:
        """Check whether the per-element operator is identity along every direction.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            bool: ``True`` iff the ``d``-dimensional operator at ``cell_idx``
            is the identity (all per-direction operators are identity).
        """
        multi = self._normalize_cell_idx(cell_idx)
        return all(bool(mask[i]) for mask, i in zip(self._is_identity_mask_1d, multi, strict=True))

    @functools.cached_property
    def num_identity_elements(self) -> int:
        """Count elements whose per-direction operators are all identity.

        Returns:
            int: The number of fully-identity elements on the tensor-product grid.
        """
        count = 1
        for mask in self._is_identity_mask_1d:
            count *= int(np.count_nonzero(mask))
        return count

    @property
    def is_identity(self) -> bool:
        """Check whether every element on the grid has an identity operator.

        Returns:
            bool: ``True`` iff all per-direction identity masks are all-``True``,
            meaning every element's operator is the identity.
        """
        return all(bool(mask.all()) for mask in self._is_identity_mask_1d)

    def per_direction_identity_flags(self, cell_idx: CellIndex) -> tuple[bool, ...]:
        """Return the per-direction identity flags for a single element.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            tuple[bool, ...]: Length-``d`` tuple of identity flags for the element.
        """
        multi = self._normalize_cell_idx(cell_idx)
        return tuple(
            bool(mask[i]) for mask, i in zip(self._is_identity_mask_1d, multi, strict=True)
        )

    # ---------------------------------------------------------------- per-cell applies

    def apply(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M @ v`` for the element at ``cell_idx``.

        Here ``M = kron(M_0, …, M_{d-1})`` with ``M_k`` the 1D operator at
        ``cell_idx`` in direction ``k``; identity directions short-circuit.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Input vector of shape
                ``(prod(input_shape_per_dir),)``.
            cell_idx (CellIndex): Element index (flat or per-direction).
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(prod(output_shape_per_dir),)``. Allocated
                if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result array (the same
            array as ``out`` when ``out`` was provided).

        Raises:
            NotImplementedError: If the space has more than 3 directions;
                specialized kernels only exist for ``d in {1, 2, 3}``.
        """
        return self._apply(v, cell_idx, "apply", out, scratch)

    def apply_transpose(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M^T @ v`` for the element at ``cell_idx``.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Input vector of shape
                ``(prod(output_shape_per_dir),)``.
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(prod(input_shape_per_dir),)``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result array.

        Raises:
            NotImplementedError: If the space has more than 3 directions;
                specialized kernels only exist for ``d in {1, 2, 3}``.
        """
        return self._apply(v, cell_idx, "apply_T", out, scratch)

    def apply_MT_K_M(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M^T @ K @ M`` for the element at ``cell_idx``.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Input matrix of shape
                ``(N_out, N_out)`` with ``N_out = prod(output_shape_per_dir)``.
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                matrix of shape ``(N_in, N_in)`` with
                ``N_in = prod(input_shape_per_dir)``. Must not alias ``K``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result matrix.

        Raises:
            NotImplementedError: If the space has more than 3 directions;
                specialized kernels only exist for ``d in {1, 2, 3}``.
        """
        return self._apply(K, cell_idx, "MT_K_M", out, scratch)

    def apply_M_K_MT(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M @ K @ M^T`` for the element at ``cell_idx``.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Input matrix of shape
                ``(N_in, N_in)`` with ``N_in = prod(input_shape_per_dir)``.
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                matrix of shape ``(N_out, N_out)`` with
                ``N_out = prod(output_shape_per_dir)``. Must not alias ``K``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result matrix.

        Raises:
            NotImplementedError: If the space has more than 3 directions;
                specialized kernels only exist for ``d in {1, 2, 3}``.
        """
        return self._apply(K, cell_idx, "M_K_MT", out, scratch)

    def operator(
        self,
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Materialize the full ``(N_out, N_in)`` operator for one element.

        Assembles the full Kronecker product in memory using :func:`numpy.kron`.
        Prefer the matrix-free apply methods in production code when the full
        matrix is not needed explicitly.

        Args:
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                matrix of shape ``(N_out, N_in)``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The full Kronecker operator.
        """
        ops = self._ops_for_cell(cell_idx)
        n_out = int(np.prod(self.output_shape_per_dir))
        n_in = int(np.prod(self.input_shape_per_dir))
        out = _allocate_or_validate_out(out, (n_out, n_in), self.dtype)
        result = ops[0]
        for M in ops[1:]:
            result = np.kron(result, M)
        out[...] = result
        return out

    def tabulate(
        self,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Materialize per-element operators for every element on the grid.

        Args:
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(num_total_intervals, N_out, N_in)``. Cells
                are ordered row-major over :attr:`num_intervals` (so flat
                index ``f`` corresponds to multi-index
                ``np.unravel_index(f, num_intervals)``).

        Returns:
            npt.NDArray[np.float32 | np.float64]: Stacked per-element operators.
        """
        n_out = int(np.prod(self.output_shape_per_dir))
        n_in = int(np.prod(self.input_shape_per_dir))
        expected = (self.num_total_intervals, n_out, n_in)
        out = _allocate_or_validate_out(out, expected, self.dtype)
        for flat in range(self.num_total_intervals):
            self.operator(flat, out=out[flat])
        return out

    # ---------------------------------------------------------------- indexing / iteration

    def __len__(self) -> int:
        """Return the total number of elements on the tensor-product grid.

        Returns:
            int: Equal to :attr:`num_total_intervals`.
        """
        return self.num_total_intervals

    def __getitem__(
        self, cell_idx: CellIndex
    ) -> tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]:
        """Return the per-direction operators and identity flags for one element.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]:
            ``(ops_for_cell, identity_flags)`` where ``ops_for_cell[k]`` is a
            view into :attr:`ops_1d` ``[k]`` and ``identity_flags`` is the
            per-direction identity mask at this element.
        """
        multi = self._normalize_cell_idx(cell_idx)
        ops = tuple(ops_dir[i] for ops_dir, i in zip(self._ops_1d, multi, strict=True))
        flags = tuple(
            bool(mask[i]) for mask, i in zip(self._is_identity_mask_1d, multi, strict=True)
        )
        return ops, flags

    def __iter__(
        self,
    ) -> Iterator[tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]]:
        """Iterate over all elements in row-major order over :attr:`num_intervals`.

        Yields:
            tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]:
            Same shape as :meth:`__getitem__`'s return value.
        """
        for flat in range(self.num_total_intervals):
            yield self[flat]

    # ---------------------------------------------------------------- internals

    def _normalize_cell_idx(self, cell_idx: CellIndex) -> tuple[int, ...]:
        """Convert a flat or per-direction index into a validated per-direction tuple.

        Args:
            cell_idx (CellIndex): Flat ``int`` or per-direction sequence.

        Returns:
            tuple[int, ...]: Length-``d`` tuple of non-negative element indices.

        Raises:
            IndexError: If a flat index is out of range (negative indices are
                not supported and are also rejected), or a per-direction entry
                is out of range for its direction.
            ValueError: If a per-direction index has the wrong length.
            TypeError: If ``cell_idx`` is not an ``int`` or sequence of ``int``.
        """
        num_intervals = self.num_intervals
        d = len(num_intervals)
        if isinstance(cell_idx, int | np.integer):
            flat = int(cell_idx)
            total = self.num_total_intervals
            if flat < 0 or flat >= total:
                raise IndexError(f"Flat cell index {flat} out of range for {total} elements")
            multi = np.unravel_index(flat, num_intervals)
            return tuple(int(i) for i in multi)
        if isinstance(cell_idx, tuple | list | np.ndarray):
            seq = tuple(int(x) for x in cell_idx)
            if len(seq) != d:
                raise ValueError(f"Per-direction cell index has length {len(seq)}, expected {d}")
            for k, (i, n) in enumerate(zip(seq, num_intervals, strict=True)):
                if i < 0 or i >= n:
                    raise IndexError(
                        f"Cell index {i} out of range for direction {k} with {n} elements"
                    )
            return seq
        raise TypeError(f"cell_idx must be int or sequence of int; got {type(cell_idx).__name__}")

    def _ops_for_cell(
        self, cell_idx: CellIndex
    ) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Return the per-direction operators at one element.

        Args:
            cell_idx (CellIndex): Element index.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: Per-direction
            2D operators, each of shape ``(n_out_k, n_in_k)``.
        """
        multi = self._normalize_cell_idx(cell_idx)
        return tuple(ops_dir[i] for ops_dir, i in zip(self._ops_1d, multi, strict=True))

    def _apply(
        self,
        operand: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        op_kind: OpKind,
        out: npt.NDArray[np.float32 | np.float64] | None,
        scratch: npt.NDArray[np.float32 | np.float64] | None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Dispatch a single-element apply variant through the Layer-2 helper.

        Args:
            operand (npt.NDArray[np.float32 | np.float64]): Input vector or
                matrix (shape depends on ``op_kind``).
            cell_idx (CellIndex): Element index.
            op_kind (OpKind): Which apply variant to dispatch.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional scratch.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result array.
        """
        multi = self._normalize_cell_idx(cell_idx)
        ops = tuple(ops_dir[i] for ops_dir, i in zip(self._ops_1d, multi, strict=True))
        flags = tuple(
            bool(mask[i]) for mask, i in zip(self._is_identity_mask_1d, multi, strict=True)
        )
        kernel, args, result = _prepare_apply_call(ops, flags, operand, out, scratch, op_kind)
        kernel(*args)
        return result

    def _apply_many(
        self,
        operand: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        op_kind: OpKind,
        out: npt.NDArray[np.float32 | np.float64] | None,
        scratch: npt.NDArray[np.float32 | np.float64] | None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Dispatch a batch apply variant through the Layer-2 helper.

        Args:
            operand (npt.NDArray[np.float32 | np.float64]): Batch input array.
            cell_indices (CellIndicesBatch): Flat or per-direction cell indices.
            op_kind (OpKind): Which apply variant to dispatch.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional scratch.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result array.
        """
        idx2d = normalize_cell_indices(cell_indices, self.num_intervals)
        kernel, args, result = _prepare_apply_many_call(
            self._ops_1d,
            self._is_identity_mask_1d,
            idx2d,
            operand,
            out,
            scratch,
            op_kind,
        )
        kernel(*args)
        return result

    # ---------------------------------------------------------------- batch applies

    def apply_many(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c @ v[c]`` for all cells in the batch.

        ``M_c = kron(M_0[c_0], …, M_{d-1}[c_{d-1}])`` with ``M_k[c_k]`` the
        1D operator at element ``c_k`` in direction ``k``; identity directions
        short-circuit per cell.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Batch input vectors,
                shape ``(n_cells, N_in)`` with
                ``N_in = prod(input_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat 1-D array of
                shape ``(n_cells,)`` or per-direction 2-D array of shape
                ``(n_cells, d)``.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_out)``. Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array of shape ``(n_cells, s)`` with
                ``s >= scratch_size_per_cell``. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_out)``.

        Raises:
            NotImplementedError: If the space has more than 3 directions.
        """
        return self._apply_many(v, cell_indices, "apply", out, scratch)

    def apply_transpose_many(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c^T @ v[c]`` for all cells in the batch.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Batch input vectors,
                shape ``(n_cells, N_out)`` with
                ``N_out = prod(output_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat or per-direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_in)``. Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_in)``.

        Raises:
            NotImplementedError: If the space has more than 3 directions.
        """
        return self._apply_many(v, cell_indices, "apply_T", out, scratch)

    def apply_MT_K_M_many(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c^T @ K[c] @ M_c`` for all cells in the batch.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Batch input matrices,
                shape ``(n_cells, N_out, N_out)`` with
                ``N_out = prod(output_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat or per-direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_in, N_in)``. Must not alias ``K``.
                Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_in, N_in)``.

        Raises:
            NotImplementedError: If the space has more than 3 directions.
        """
        return self._apply_many(K, cell_indices, "MT_K_M", out, scratch)

    def apply_M_K_MT_many(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c @ K[c] @ M_c^T`` for all cells in the batch.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Batch input matrices,
                shape ``(n_cells, N_in, N_in)`` with
                ``N_in = prod(input_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat or per-direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_out, N_out)``. Must not alias ``K``.
                Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_out, N_out)``.

        Raises:
            NotImplementedError: If the space has more than 3 directions.
        """
        return self._apply_many(K, cell_indices, "M_K_MT", out, scratch)


def normalize_cell_indices(
    cell_indices: CellIndicesBatch,
    num_intervals: tuple[int, ...],
) -> npt.NDArray[np.intp]:
    """Convert batch cell indices to a validated ``(n_cells, d)`` integer array.

    Accepts flat indices (row-major over ``num_intervals``) or per-direction
    indices, in array or list form. Validates that all values are within their
    respective per-direction bounds and that the shape is consistent.

    Args:
        cell_indices (CellIndicesBatch): Flat 1-D array or list of ``n_cells``
            flat indices, or 2-D array/list of shape ``(n_cells, d)`` with
            per-direction indices.
        num_intervals (tuple[int, ...]): Per-direction element counts
            ``(n_el_0, …, n_el_{d-1})``.

    Returns:
        npt.NDArray[np.intp]: 2-D integer array of shape ``(n_cells, d)``.

    Raises:
        ValueError: If ``cell_indices`` has wrong shape or out-of-range values.
        TypeError: If ``cell_indices`` is not an array-like of integers.
    """
    d = len(num_intervals)
    arr = np.asarray(cell_indices, dtype=np.intp)
    if arr.ndim == 1:
        n_cells = arr.shape[0]
        total = 1
        for n in num_intervals:
            total *= n
        if n_cells > 0 and (int(arr.min()) < 0 or int(arr.max()) >= total):
            raise ValueError(
                f"Flat cell indices must be in [0, {total}); "
                f"got range [{int(arr.min())}, {int(arr.max())}]"
            )
        rows = np.unravel_index(arr, num_intervals)
        return np.stack(rows, axis=1)
    if arr.ndim == 2:  # noqa: PLR2004
        if arr.shape[1] != d:
            raise ValueError(
                f"Per-direction cell_indices must have shape (n_cells, {d}); got shape {arr.shape}"
            )
        n_cells = arr.shape[0]
        if n_cells > 0:
            for k, n_el in enumerate(num_intervals):
                col = arr[:, k]
                if int(col.min()) < 0 or int(col.max()) >= n_el:
                    raise ValueError(
                        f"cell_indices[:, {k}] must be in [0, {n_el}); "
                        f"got range [{int(col.min())}, {int(col.max())}]"
                    )
        return arr
    raise ValueError(f"cell_indices must be 1-D (flat) or 2-D (per-direction); got ndim={arr.ndim}")


def _numerical_identity_mask(
    ops: npt.NDArray[np.float32 | np.float64], tol: float
) -> npt.NDArray[np.bool_]:
    """Compute a per-element identity mask by comparing each matrix against ``I``.

    Non-square matrices are always reported as non-identity.

    Args:
        ops (npt.NDArray[np.float32 | np.float64]): Stack of 2D operators,
            shape ``(n_elements, n_out, n_in)``.
        tol (float): Absolute tolerance used element-wise.

    Returns:
        npt.NDArray[bool]: Boolean array of shape ``(n_elements,)``.
    """
    n_elements, n_out, n_in = ops.shape
    if n_out != n_in:
        return np.zeros(n_elements, dtype=np.bool_)
    eye = np.eye(n_out, dtype=ops.dtype)
    result: npt.NDArray[np.bool_] = np.max(np.abs(ops - eye[np.newaxis]), axis=(1, 2)) <= tol
    return result


# The op_kind shape helper is kept import-local so downstream callers can build
# operands of the right shape without reaching into Layer 2 directly.
def operand_shape(
    extraction: SpanwiseElementExtraction, op_kind: OpKind
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the expected ``(input_shape, output_shape)`` for an apply variant.

    Args:
        extraction (SpanwiseElementExtraction): Extraction object supplying
            per-direction shapes.
        op_kind (OpKind): One of the :data:`OpKind` literals: ``"apply"``,
            ``"apply_T"``, ``"MT_K_M"``, ``"M_K_MT"``.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: ``(input_shape, output_shape)``.
    """
    return _operation_shapes(
        extraction.input_shape_per_dir, extraction.output_shape_per_dir, op_kind
    )


__all__ = [
    "CellIndex",
    "CellIndicesBatch",
    "SpanwiseElementExtraction",
    "Target",
    "normalize_cell_indices",
    "operand_shape",
]
