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

Identity short-circuit is used wherever possible. All three targets use
structural (multiplicity-based) identity predicates:

- ``"bezier"``: element ``e`` is identity iff both its boundary unique knots
  have multiplicity ``>= degree + 1``, i.e. the element is already a Bézier
  patch. Knot multiplicities are computed using ``space.tolerance``.
- ``"lagrange"``: for ``degree == 0`` every element is trivially identity.
  For ``degree > 0`` an element is identity iff its Bézier extraction is
  identity and the Lagrange-to-Bernstein matrix equals ``I`` (which holds when
  the Lagrange nodes coincide with the Bernstein abscissae, e.g. ``degree == 1``
  with equispaced, GLL, or Chebyshev-2nd nodes).
- ``"cardinal"``: structural mask from
  :meth:`BsplineSpace1D.get_cardinal_intervals` labels uniform-span intervals,
  on which the cardinal extraction operator is exactly the identity.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import numpy.typing as npt

from ..basis import LagrangeVariant
from ..basis._basis_utils import _allocate_or_validate_out
from ..change_basis import _cached_lagrange_to_bernstein_matrix
from ._bspline_extraction import _bezier_structural_identity_mask_core
from ._extraction_helpers import (
    OpKind,
    _operation_shapes,
    _prepare_apply_call,
    _prepare_apply_many_call,
)

if TYPE_CHECKING:
    from ._bspline_space_1d import BsplineSpace1D
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
    basis, this class eagerly builds per-direction compact operator storage:
    only the non-identity rows of each direction's extraction operator array
    are retained, reducing memory for identity-heavy spaces (e.g. cardinal
    spaces on uniform meshes). Per-element d-dimensional operators are never
    materialized unless explicitly requested via :meth:`operator` or
    :meth:`tabulate`: instead the apply-style methods dispatch to the
    matrix-free Kronecker kernels in ``pantr.bspline._extraction_kernels``.

    With the current 1D builders all per-direction operators are square of
    size ``(degree_k + 1, degree_k + 1)``. The class also supports non-square
    per-direction operators, so new 1D builders can plug in without changes.

    The per-direction data is exposed as two complementary representations:

    - *Compact* (:attr:`compact_ops_1d`, :attr:`idx_maps_1d`,
      :attr:`is_identity_mask_1d`): primary storage, suitable for downstream
      ``@njit`` code that calls the Layer-3 batch kernels directly.
    - *Dense* (:attr:`ops_1d`): the full ``(n_elements_k, n_out_k, n_in_k)``
      layout, reconstructed lazily from compact storage on first access.

    Attributes:
        _space (BsplineSpace): Underlying multi-dimensional B-spline space.
        _target (Target): Target basis tag.
        _lagrange_variant (LagrangeVariant): Point distribution used when
            ``target == "lagrange"``; ignored otherwise.
        _compact_ops_1d (tuple[npt.NDArray[np.float32 | np.float64], ...]):
            Per-direction compact 3D operator arrays of shape
            ``(n_compact_k, n_out_k, n_in_k)``; only non-identity rows are
            stored. Always has at least one row to ensure safe Numba indexing.
        _idx_maps_1d (tuple[npt.NDArray[np.intp], ...]): Per-direction compact
            index maps of shape ``(n_elements_k,)``; ``_idx_maps_1d[k][e]`` is
            the row index into ``_compact_ops_1d[k]`` for element ``e``
            (undefined for identity elements, stored as 0).
        _is_identity_mask_1d (tuple[npt.NDArray[bool], ...]): Per-direction
            identity masks of shape ``(n_elements_k,)``.
    """

    _space: BsplineSpace
    _target: Target
    _lagrange_variant: LagrangeVariant
    _compact_ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...]
    _idx_maps_1d: tuple[npt.NDArray[np.intp], ...]
    _is_identity_mask_1d: tuple[npt.NDArray[np.bool_], ...]

    def __init__(
        self,
        space: BsplineSpace,
        target: Target,
        *,
        lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    ) -> None:
        """Build the per-direction operators and identity masks.

        Args:
            space (BsplineSpace): Multi-dimensional B-spline space.
            target (Target): One of ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
            lagrange_variant (LagrangeVariant): Point distribution used when
                ``target == "lagrange"``. Defaults to
                :attr:`pantr.basis.LagrangeVariant.EQUISPACES`.

        Raises:
            ValueError: If ``target`` is not a recognized tag.
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

        compact_ops_1d: list[npt.NDArray[np.float32 | np.float64]] = []
        idx_maps_1d: list[npt.NDArray[np.intp]] = []
        masks_1d: list[npt.NDArray[np.bool_]] = []
        for space_1d in space.spaces:
            if target == "bezier":
                ops = space_1d.tabulate_Bezier_extraction_operators()
                mask = _bezier_structural_identity_mask(space_1d)
            elif target == "lagrange":
                ops = space_1d.tabulate_Lagrange_extraction_operators(
                    lagrange_variant=lagrange_variant
                )
                mask = _lagrange_structural_identity_mask(space_1d, lagrange_variant)
            else:  # target == "cardinal"
                ops = space_1d.tabulate_cardinal_extraction_operators()
                mask = space_1d.get_cardinal_intervals()
            non_id_idx = np.where(~mask)[0]
            n_non_id = int(non_id_idx.shape[0])
            n_out, n_in = int(ops.shape[1]), int(ops.shape[2])
            if n_non_id > 0:
                compact_ops = ops[non_id_idx].copy()
            else:
                compact_ops = np.zeros((1, n_out, n_in), dtype=ops.dtype)
            idx_map = np.zeros(int(mask.shape[0]), dtype=np.intp)
            idx_map[non_id_idx] = np.arange(n_non_id, dtype=np.intp)
            compact_ops.flags.writeable = False
            idx_map.flags.writeable = False
            mask.flags.writeable = False
            compact_ops_1d.append(compact_ops)
            idx_maps_1d.append(idx_map)
            masks_1d.append(mask)

        self._compact_ops_1d = tuple(compact_ops_1d)
        self._idx_maps_1d = tuple(idx_maps_1d)
        self._is_identity_mask_1d = tuple(masks_1d)

        if len(self._compact_ops_1d) > 1:
            dtype_0 = self._compact_ops_1d[0].dtype
            for k, _ops in enumerate(self._compact_ops_1d[1:], start=1):
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

    @functools.cached_property
    def ops_1d(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the per-direction 1D operator arrays (dense, reconstructed lazily).

        Reconstructs the full ``(n_elements_k, n_out_k, n_in_k)`` array from
        compact storage on first access and caches the result. Identity elements
        are filled with the identity matrix; non-identity elements are read from
        :attr:`compact_ops_1d`.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: Length-``d`` tuple
            of read-only 3D arrays; ``ops_1d[k]`` has shape
            ``(n_elements_k, n_out_k, n_in_k)``. Intended for consumption by
            downstream ``@njit`` code when the full dense layout is required.
            For compact-aware downstream code, prefer :attr:`compact_ops_1d` and
            :attr:`idx_maps_1d`.
        """
        dense: list[npt.NDArray[np.float32 | np.float64]] = []
        for compact_ops, idx_map, mask in zip(
            self._compact_ops_1d, self._idx_maps_1d, self._is_identity_mask_1d, strict=True
        ):
            n_el = int(mask.shape[0])
            n_out, n_in = int(compact_ops.shape[1]), int(compact_ops.shape[2])
            full: npt.NDArray[np.float32 | np.float64] = np.empty(
                (n_el, n_out, n_in), dtype=compact_ops.dtype
            )
            eye = np.eye(n_out, n_in, dtype=compact_ops.dtype)
            for e in range(n_el):
                if mask[e]:
                    full[e] = eye
                else:
                    full[e] = compact_ops[idx_map[e]]
            full.flags.writeable = False
            dense.append(full)
        return tuple(dense)

    @property
    def compact_ops_1d(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the per-direction compact operator arrays (non-identity rows only).

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: Length-``d`` tuple
            of read-only 3D arrays; ``compact_ops_1d[k]`` has shape
            ``(n_compact_k, n_out_k, n_in_k)`` where ``n_compact_k`` is the
            number of non-identity elements in direction ``k`` (at least 1 to
            ensure safe Numba indexing). Intended for downstream ``@njit`` code
            alongside :attr:`idx_maps_1d` and :attr:`is_identity_mask_1d`.
        """
        return self._compact_ops_1d

    @property
    def idx_maps_1d(self) -> tuple[npt.NDArray[np.intp], ...]:
        """Get the per-direction compact index maps.

        Returns:
            tuple[npt.NDArray[np.intp], ...]: Length-``d`` tuple of read-only
            1D integer arrays; ``idx_maps_1d[k]`` has shape ``(n_elements_k,)``
            and ``idx_maps_1d[k][e]`` is the row index into
            :attr:`compact_ops_1d` ``[k]`` for element ``e``. For identity
            elements the stored value is 0 (unused; the kernel short-circuits on
            :attr:`is_identity_mask_1d`). Intended for downstream ``@njit`` code.
        """
        return self._idx_maps_1d

    @property
    def is_identity_mask_1d(self) -> tuple[npt.NDArray[np.bool_], ...]:
        """Get the per-direction identity masks.

        All three targets use structural (multiplicity-based) identity predicates.
        For ``"bezier"``, an element is identity iff both its boundary unique knots
        have multiplicity ``>= degree + 1``; multiplicities are computed using
        ``space.tolerance``. For ``"lagrange"``, the mask delegates to the Bézier
        mask when the Lagrange-to-Bernstein matrix equals ``I`` (e.g. ``degree == 1``
        with equispaced or GLL nodes), returns all-``True`` for ``degree == 0``, and
        all-``False`` otherwise. For ``"cardinal"``, the mask is the structural output
        of :meth:`BsplineSpace1D.get_cardinal_intervals`.

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
        return tuple(int(ops.shape[2]) for ops in self._compact_ops_1d)

    @property
    def output_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction output sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_out_0, …, n_out_{d-1})``.
        """
        return tuple(int(ops.shape[1]) for ops in self._compact_ops_1d)

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
        ops = tuple(ops_dir[i] for ops_dir, i in zip(self.ops_1d, multi, strict=True))
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
        return tuple(ops_dir[i] for ops_dir, i in zip(self.ops_1d, multi, strict=True))

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
        ops = tuple(ops_dir[i] for ops_dir, i in zip(self.ops_1d, multi, strict=True))
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
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_out)`` for vector kinds or ``(n_cells, N_out, N_out)`` /
            ``(n_cells, N_in, N_in)`` for bilateral kinds.

        Raises:
            IndexError: If any cell index is out of range.
            ValueError: If operand shape/dtype or ``out``/``scratch`` are invalid.
            TypeError: If ``cell_indices`` contains non-integer values.
            NotImplementedError: If the space has more than 3 directions.
        """
        idx2d = normalize_cell_indices(cell_indices, self.num_intervals)
        kernel, args, result = _prepare_apply_many_call(
            self._compact_ops_1d,
            self._idx_maps_1d,
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
        IndexError: If any value is out of range for its per-direction bound.
        ValueError: If ``cell_indices`` has the wrong shape or ndim.
        TypeError: If ``cell_indices`` contains non-integer values.
    """
    d = len(num_intervals)
    arr_raw = np.asarray(cell_indices)
    if arr_raw.size > 0 and not np.issubdtype(arr_raw.dtype, np.integer):
        raise TypeError(f"cell_indices must contain integers; got dtype {arr_raw.dtype}")
    arr = arr_raw.astype(np.intp)
    if arr.ndim == 1:
        n_cells = arr.shape[0]
        total = 1
        for n in num_intervals:
            total *= n
        if n_cells > 0 and (int(arr.min()) < 0 or int(arr.max()) >= total):
            raise IndexError(
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
                    raise IndexError(
                        f"cell_indices[:, {k}] must be in [0, {n_el}); "
                        f"got range [{int(col.min())}, {int(col.max())}]"
                    )
        return arr
    raise ValueError(f"cell_indices must be 1-D (flat) or 2-D (per-direction); got ndim={arr.ndim}")


def _bezier_structural_identity_mask(
    space_1d: BsplineSpace1D,
) -> npt.NDArray[np.bool_]:
    """Compute the per-element Bézier identity mask from knot multiplicities.

    Element ``e`` is identity iff both its boundary unique knots (in-domain)
    have multiplicity ``>= degree + 1``, meaning the element is already a
    Bézier patch with no continuity coupling to its neighbours.

    Knot multiplicities are computed via the space's own tolerance
    (``space_1d.tolerance``), which groups coincident knots before counting.

    Args:
        space_1d (BsplineSpace1D): A 1D B-spline space.

    Returns:
        npt.NDArray[np.bool_]: Boolean array of shape ``(n_elements,)``.
    """
    _, mults = space_1d.get_unique_knots_and_multiplicity(in_domain=True)
    n_elements = len(mults) - 1
    out = np.empty(n_elements, dtype=np.bool_)
    _bezier_structural_identity_mask_core(mults, space_1d.degree, out)
    return out


def _lagrange_structural_identity_mask(
    space_1d: BsplineSpace1D,
    lagrange_variant: LagrangeVariant,
) -> npt.NDArray[np.bool_]:
    """Compute the per-element Lagrange identity mask.

    For ``degree == 0`` every element is trivially identity (the 1x1
    extraction matrix is ``[[1.0]]``). For ``degree > 0`` the Lagrange
    extraction operator at element ``e`` equals ``bezier_op[e] @ lagr_to_bzr``.
    This is the identity iff ``bezier_op[e] == I`` and ``lagr_to_bzr == I``.
    ``lagr_to_bzr`` equals ``I`` when the Lagrange nodes coincide with the
    Bernstein abscissae ``i / degree`` — e.g. for ``degree == 1`` with
    equispaced, GLL, or Chebyshev-2nd nodes.  For all other cases no element
    can have an identity Lagrange extraction operator.

    Args:
        space_1d (BsplineSpace1D): A 1D B-spline space.
        lagrange_variant (LagrangeVariant): Lagrange node distribution.

    Returns:
        npt.NDArray[np.bool_]: Boolean array of shape ``(n_elements,)``.
    """
    n_elements = space_1d.num_intervals
    if space_1d.degree == 0:
        return np.ones(n_elements, dtype=np.bool_)
    dtype = space_1d.knots.dtype
    lagr_to_bzr = _cached_lagrange_to_bernstein_matrix(space_1d.degree, lagrange_variant, dtype)
    if np.array_equal(lagr_to_bzr, np.eye(space_1d.degree + 1, dtype=dtype)):
        return _bezier_structural_identity_mask(space_1d)
    return np.zeros(n_elements, dtype=np.bool_)


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
