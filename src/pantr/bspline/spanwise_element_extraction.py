"""Tensor-product change-of-basis extraction across B-spline elements.

This module exposes :class:`SpanwiseElementExtraction`, a lazy tensor-product
change-of-basis object. It eagerly caches the per-direction 1D extraction
operators once at construction time and, on demand, dispatches to the Layer-3
Kronecker kernels in ``pantr.bspline._extraction_kernels`` to apply the
d-dimensional operator for a single element.

Three targets are supported, named by :class:`ExtractionTarget` (the source basis
is always the B-spline basis):

- ``BEZIER``:   Bernstein (Bézier) basis on each element.
- ``LAGRANGE``: Lagrange basis on each element, at the chosen point
  distribution (see :class:`pantr.basis.LagrangeVariant`).
- ``CARDINAL``: cardinal B-spline basis on each element.

Identity short-circuit is used wherever possible. All three targets use
structural (multiplicity-based) identity predicates:

- ``BEZIER``: element ``e`` is identity iff both its boundary unique knots
  have multiplicity ``>= degree + 1``, i.e. the element is already a Bézier
  patch. Knot multiplicities are computed using ``space.tolerance``.
- ``LAGRANGE``: for ``degree == 0`` every element is trivially identity.
  For ``degree > 0`` an element is identity iff its Bézier extraction is
  identity and the Lagrange-to-Bernstein matrix equals ``I`` (which holds when
  the Lagrange nodes coincide with the Bernstein abscissae, e.g. ``degree == 1``
  with equispaced, GLL, or Chebyshev-2nd nodes).
- ``CARDINAL``: structural mask from
  :meth:`BsplineSpace1D.get_cardinal_intervals` labels uniform-span intervals,
  on which the cardinal extraction operator is exactly the identity.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, NoReturn, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ..basis import LagrangeVariant
from ..basis._basis_utils import _allocate_or_validate_out
from ..change_basis import _cached_lagrange_to_bernstein_matrix
from ._bspline_space_nd import _impl_class as _space_impl_class
from ._extraction_backend import bezier_identity_mask_kernel, lagrange_identity_mask_kernel
from ._extraction_helpers import (
    OpKind,
    _operation_shapes,
    _prepare_apply_call,
    _prepare_apply_many_call,
)

if TYPE_CHECKING:
    from .._pantr_cpp import SpanwiseElementExtraction32 as _CppExtraction32
    from .._pantr_cpp import SpanwiseElementExtraction64 as _CppExtraction64
    from ._bspline_space_1d import BsplineSpace1D
    from ._bspline_space_nd import BsplineSpace

    _Impl: TypeAlias = "_SpanwiseElementExtractionPython | _CppExtraction32 | _CppExtraction64"
    """The implementation a :class:`SpanwiseElementExtraction` holds.

    The same alias shape :mod:`pantr.bspline._bspline_space_nd` declares: three
    unrelated nominal types that happen to offer one surface, which is the port's
    whole claim.
    """

    _SpaceImpl: TypeAlias = "Any"
    """The space's implementation, as either backend holds it.

    Deliberately opaque, for :mod:`pantr.bspline._bspline_space_nd`'s reason: that
    module owns the union of the concrete types and restating it here would be a
    second place to keep in step.
    """

_Ops1D: TypeAlias = "tuple[npt.NDArray[np.float32 | np.float64], ...]"
"""One per-direction 3D operator array per direction, in axis order."""


class ExtractionTarget(IntEnum):
    """The element-local basis a spanwise extraction maps the B-spline basis onto.

    An :class:`~enum.IntEnum` rather than a string, per the project's convention
    that a closed set of choices is never stringly typed (``pantr._backend``'s
    ``Backend`` states the same rule). Two properties
    follow from the choice and neither is available to a string: an integer member
    is what crosses the backend seam (``design/cross_backend_types.md``), and it is
    what a ``nopython`` Numba kernel can hold.

    Attributes:
        BEZIER: Bernstein (Bézier) basis on each element.
        LAGRANGE: Lagrange basis on each element, at the point distribution named
            by :class:`pantr.basis.LagrangeVariant`.
        CARDINAL: Cardinal B-spline basis on each element.
    """

    BEZIER = 0
    LAGRANGE = 1
    CARDINAL = 2


Target = Literal["bezier", "lagrange", "cardinal"]
"""Legacy string spelling of :class:`ExtractionTarget`, still accepted on input.

Kept so that ``SpanwiseElementExtraction(space, "bezier")`` keeps working. Pass an
:class:`ExtractionTarget` in new code: the string form cannot cross the backend
seam and is not what the ``target`` property hands back.
"""


TargetLike = ExtractionTarget | Target
"""What the public constructors accept for a target: the enum or its legacy string."""


_TARGET_BY_NAME: Final[dict[str, ExtractionTarget]] = {
    "bezier": ExtractionTarget.BEZIER,
    "lagrange": ExtractionTarget.LAGRANGE,
    "cardinal": ExtractionTarget.CARDINAL,
}
"""The legacy string spellings, mapped onto the enum they name."""


def _coerce_target(target: TargetLike) -> ExtractionTarget:
    """Resolve a target argument to an :class:`ExtractionTarget`.

    The string branch is the compatibility boundary and the only place the legacy
    spelling is understood; everything downstream of it sees the enum.

    Args:
        target (TargetLike): An :class:`ExtractionTarget`, or one of the legacy
            strings ``"bezier"``, ``"lagrange"``, ``"cardinal"``.

    Returns:
        ExtractionTarget: The resolved target.

    Raises:
        ValueError: If ``target`` is neither an :class:`ExtractionTarget` nor a
            recognized string spelling.
    """
    if isinstance(target, ExtractionTarget):
        return target
    resolved = _TARGET_BY_NAME.get(target) if isinstance(target, str) else None
    if resolved is None:
        valid = ", ".join(repr(name) for name in _TARGET_BY_NAME)
        raise ValueError(
            f"Unknown target {target!r}; expected an ExtractionTarget or one of {valid}"
        )
    return resolved


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


def _build_direction_operators(
    space: BsplineSpace,
    target: ExtractionTarget,
    lagrange_variant: LagrangeVariant,
) -> tuple[list[npt.NDArray[np.float32 | np.float64]], list[npt.NDArray[np.bool_]]]:
    """Build each direction's dense extraction operators and identity mask.

    Common mode between the two backends, and deliberately so: the 1D builders and
    the two mask predicates are already dispatched on their own by
    :mod:`pantr.bspline._extraction_backend`, so building them here keeps one
    implementation of each and keeps the cardinal target -- which has no C++ builder,
    because it needs the cardinal-interval scan -- working under both backends.

    Args:
        space (BsplineSpace): The space to extract from.
        target (ExtractionTarget): The element-local basis, already resolved.
        lagrange_variant (LagrangeVariant): Point distribution, used only for
            :attr:`ExtractionTarget.LAGRANGE`.

    Returns:
        tuple[list, list]: ``(operators, masks)``, one entry each per direction in
        axis order. ``operators[k]`` has shape ``(n_elements_k, n_out_k, n_in_k)``
        and ``masks[k]`` has shape ``(n_elements_k,)``.
    """
    operators: list[npt.NDArray[np.float32 | np.float64]] = []
    masks: list[npt.NDArray[np.bool_]] = []
    for space_1d in space.spaces:
        if target is ExtractionTarget.BEZIER:
            ops = space_1d.tabulate_Bezier_extraction_operators()
            mask = _bezier_structural_identity_mask(space_1d)
        elif target is ExtractionTarget.LAGRANGE:
            ops = space_1d.tabulate_Lagrange_extraction_operators(lagrange_variant=lagrange_variant)
            mask = _lagrange_structural_identity_mask(space_1d, lagrange_variant)
        else:  # target is ExtractionTarget.CARDINAL
            ops = space_1d.tabulate_cardinal_extraction_operators()
            mask = space_1d.get_cardinal_intervals()
        operators.append(ops)
        masks.append(mask)
    return operators, masks


class _SpanwiseElementExtractionPython:
    """The pure-Python spanwise element extraction: the port's parity oracle.

    Holds the space *implementation* rather than its wrapper, which is what makes it
    the exact counterpart of ``pantr::bspline::SpanwiseElementExtraction<T>``: that
    type holds a ``BsplineSpace<T>`` handle, and this one holds whatever
    :class:`~pantr.bspline.BsplineSpace` selected for the same backend. The two
    constructors therefore take the same five arguments in the same order, which is
    what lets :class:`SpanwiseElementExtraction` build either without a branch.

    The per-direction operators arrive built. That is the type's seam rather than a
    gap, and ``cpp/include/pantr/bspline/spanwise_extraction.hpp`` carries the
    argument; the short version is that the builders are their own port and the
    cardinal target has none in C++.

    What this class does own is the *compaction* -- only the non-identity rows are
    kept -- plus the two derived quantities ``design/bspline_derived_caches.md``
    assigns here: :attr:`ops_1d`, the dense block, memoised; and
    :attr:`num_identity_elements`, eager.

    No ``__slots__``, matching :class:`pantr.bspline._bspline_space_1d`'s oracle:
    :func:`functools.cached_property` needs an instance dictionary, and the immutability
    that matters to a caller is enforced on the wrapper, which is the object a caller
    ever holds.

    Attributes:
        _space (Any): The space implementation, as either backend holds it.
        _target (int): The :class:`ExtractionTarget` member's integer value, carried
            as the integer because that is what crosses the binding.
        _lagrange_variant (str): The :class:`~pantr.basis.LagrangeVariant` member's
            own string, carried and handed back unread.
        _compact_ops_1d (tuple[npt.NDArray[np.float32 | np.float64], ...]):
            Per-direction compact operators, shape ``(n_compact_k, n_out_k, n_in_k)``.
            At least one row, of zeros where every element is the identity, so that an
            index a Numba kernel computed before checking the mask is in range.
        _idx_maps_1d (tuple[npt.NDArray[np.intp], ...]): Per-direction row indices
            into :attr:`compact_ops_1d`, shape ``(n_elements_k,)``.
        _is_identity_mask_1d (tuple[npt.NDArray[np.bool_], ...]): Per-direction
            identity masks, shape ``(n_elements_k,)``.
        _num_identity_elements (int): Fully-identity elements on the grid.
    """

    def __init__(
        self,
        space: _SpaceImpl,
        target: int,
        lagrange_variant: str,
        operators: Sequence[npt.NDArray[np.float32 | np.float64]],
        masks: Sequence[npt.NDArray[np.bool_]],
    ) -> None:
        """Compact the given per-direction operators and hold them.

        Args:
            space (Any): The space implementation, already validated by
                :class:`SpanwiseElementExtraction`.
            target (int): The :class:`ExtractionTarget` member's integer value.
            lagrange_variant (str): The :class:`~pantr.basis.LagrangeVariant` member's
                own string.
            operators (Sequence[npt.NDArray[np.float32 | np.float64]]): One dense
                ``(n_elements_k, n_out_k, n_in_k)`` block per direction.
            masks (Sequence[npt.NDArray[np.bool_]]): One ``(n_elements_k,)`` identity
                mask per direction.
        """
        self._space = space
        self._target = target
        self._lagrange_variant = lagrange_variant

        compact_ops_1d: list[npt.NDArray[np.float32 | np.float64]] = []
        idx_maps_1d: list[npt.NDArray[np.intp]] = []
        masks_1d: list[npt.NDArray[np.bool_]] = []
        num_identity = 1
        for ops, mask_in in zip(operators, masks, strict=True):
            mask = np.array(mask_in, dtype=np.bool_)
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
            num_identity *= int(np.count_nonzero(mask))

        self._compact_ops_1d = tuple(compact_ops_1d)
        self._idx_maps_1d = tuple(idx_maps_1d)
        self._is_identity_mask_1d = tuple(masks_1d)
        self._num_identity_elements = num_identity

    @property
    def space(self) -> _SpaceImpl:
        """Get the space implementation this extraction was built over.

        Returns:
            Any: The implementation, shared rather than copied, exactly as the C++
            counterpart shares its handle.
        """
        return self._space

    @property
    def target(self) -> int:
        """Get the target basis, as the enum's integer value.

        Returns:
            int: The :class:`ExtractionTarget` member's value.
        """
        return self._target

    @property
    def lagrange_variant(self) -> str:
        """Get the Lagrange point distribution's own name.

        Returns:
            str: The :class:`~pantr.basis.LagrangeVariant` member's value.
        """
        return self._lagrange_variant

    @property
    def dim(self) -> int:
        """Get the number of tensor-product directions.

        Returns:
            int: The dimension of the space.
        """
        return int(self._space.dim)

    @property
    def num_intervals(self) -> tuple[int, ...]:
        """Get the per-direction number of elements.

        Returns:
            tuple[int, ...]: One count per direction, in axis order.
        """
        return tuple(int(n) for n in self._space.num_intervals)

    @property
    def num_total_intervals(self) -> int:
        """Get the total number of elements on the tensor-product grid.

        Returns:
            int: The product of :attr:`num_intervals`.
        """
        return int(self._space.num_total_intervals)

    @property
    def compact_ops_1d(self) -> _Ops1D:
        """Get the per-direction compact operator arrays.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: One read-only
            ``(n_compact_k, n_out_k, n_in_k)`` array per direction.
        """
        return self._compact_ops_1d

    @property
    def idx_maps_1d(self) -> tuple[npt.NDArray[np.intp], ...]:
        """Get the per-direction compact index maps.

        Returns:
            tuple[npt.NDArray[np.intp], ...]: One read-only ``(n_elements_k,)`` array
            per direction.
        """
        return self._idx_maps_1d

    @property
    def is_identity_mask_1d(self) -> tuple[npt.NDArray[np.bool_], ...]:
        """Get the per-direction identity masks.

        Returns:
            tuple[npt.NDArray[np.bool_], ...]: One read-only ``(n_elements_k,)`` array
            per direction.
        """
        return self._is_identity_mask_1d

    @functools.cached_property
    def ops_1d(self) -> _Ops1D:
        """Get the per-direction dense operator arrays, decompressed on first access.

        Identity elements read as ``numpy.eye(n_out, n_in)``; everything else is read
        from :attr:`compact_ops_1d`.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: One read-only
            ``(n_elements_k, n_out_k, n_in_k)`` array per direction.
        """
        dense: list[npt.NDArray[np.float32 | np.float64]] = []
        for compact_ops, idx_map, mask in zip(
            self._compact_ops_1d, self._idx_maps_1d, self._is_identity_mask_1d, strict=True
        ):
            n_el = int(mask.shape[0])
            # shape[1] and shape[2] are direction-wide constants (same for compact and full)
            n_out, n_in = int(compact_ops.shape[1]), int(compact_ops.shape[2])
            full: npt.NDArray[np.float32 | np.float64] = np.empty(
                (n_el, n_out, n_in), dtype=compact_ops.dtype
            )
            eye = np.eye(n_out, n_in, dtype=compact_ops.dtype)
            full[mask] = eye
            full[~mask] = compact_ops[idx_map[~mask]]
            full.flags.writeable = False
            dense.append(full)
        return tuple(dense)

    @property
    def input_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction input sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_in_0, …, n_in_{d-1})``.
        """
        # shape[2] is the per-direction input size, identical between compact and full layouts
        return tuple(int(ops.shape[2]) for ops in self._compact_ops_1d)

    @property
    def output_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction output sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_out_0, …, n_out_{d-1})``.
        """
        # shape[1] is the per-direction output size, identical between compact and full layouts
        return tuple(int(ops.shape[1]) for ops in self._compact_ops_1d)

    @property
    def num_identity_elements(self) -> int:
        """Count elements whose per-direction operators are all identity.

        Returns:
            int: The number of fully-identity elements on the grid.
        """
        return self._num_identity_elements

    @property
    def is_identity(self) -> bool:
        """Check whether every element on the grid has an identity operator.

        Returns:
            bool: ``True`` iff every per-direction mask is all-``True``.
        """
        return all(bool(mask.all()) for mask in self._is_identity_mask_1d)


def _impl_class(dtype: np.dtype[Any]) -> type[_SpanwiseElementExtractionPython] | type[Any]:
    """Get the implementation class the active backend and the dtype select.

    The backend is per process rather than per instance, for the reason
    :func:`pantr.bspline._bspline_space_nd._impl_class` gives.

    Args:
        dtype (np.dtype[Any]): The storage format the operators share.

    Returns:
        type: The oracle under the Python backend, and the C++ class for that storage
        format otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if active_backend() is Backend.PYTHON:
        return _SpanwiseElementExtractionPython
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    if dtype == np.float32:
        return _pantr_cpp.SpanwiseElementExtraction32
    return _pantr_cpp.SpanwiseElementExtraction64


def _new_impl(
    space: BsplineSpace,
    target: ExtractionTarget,
    lagrange_variant: LagrangeVariant,
) -> _Impl:
    """Build an extraction in whichever implementation the backend selects.

    Args:
        space (BsplineSpace): The space to extract from, already checked for periodic
            directions.
        target (ExtractionTarget): The element-local basis, already resolved.
        lagrange_variant (LagrangeVariant): The point distribution.

    Returns:
        _Impl: The implementation object; an oracle instance or a C++ handle.

    Raises:
        ValueError: If the per-direction operators do not share one dtype, or if
            ``space`` was built under a different backend.
        RuntimeError: If the C++ backend is requested and is not available.
    """
    operators, masks = _build_direction_operators(space, target, lagrange_variant)

    # A type-kind check, so it stays here: `SpanwiseElementExtraction<T>` can hold only
    # one width and a mixed collection is not representable, exactly as
    # `BsplineSpace<T>` cannot hold directions of two dtypes. The message is the
    # oracle's, character for character.
    if len(operators) > 1:
        dtype_0 = operators[0].dtype
        for k, ops in enumerate(operators[1:], start=1):
            if ops.dtype != dtype_0:
                raise ValueError(
                    f"Per-direction operators have inconsistent dtypes: "
                    f"ops_1d[0].dtype={dtype_0}, ops_1d[{k}].dtype={ops.dtype}"
                )

    dtype = np.dtype(space.dtype)
    cls = _impl_class(dtype)
    if cls is _SpanwiseElementExtractionPython:
        return _SpanwiseElementExtractionPython(
            space._impl, int(target), str(lagrange_variant), operators, masks
        )
    # `cls` is one of the C++ classes here. Handing it a space built under the Python
    # backend raises a nanobind `TypeError` naming C++ types, which is loud but
    # unreadable, so the same refusal `_bspline_space_nd._new_impl` writes is written
    # here, in the oracle's own vocabulary.
    if not isinstance(space._impl, _space_impl_class(dtype)):
        raise ValueError(
            "The B-spline space must come from the active backend; it was built under "
            "a different one."
        )
    cpp_cls: Any = cls
    return cast(
        "_Impl",
        cpp_cls(space._impl, int(target), str(lagrange_variant), operators, masks),
    )


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

    For one element at a time, :meth:`factors` returns the per-direction
    ``(is_identity, operator)`` pairs without allocating anything, which is what
    a consumer serializing the 1D factors should use; ``__getitem__`` is the
    convenience form that materializes an explicit identity matrix instead.

    **This class is a wrapper.** The value -- the space, the target, the point
    distribution and the three per-direction array bundles -- is owned by an
    implementation chosen by ``_impl_class``, which is the C++ type
    (``cpp/include/pantr/bspline/spanwise_extraction.hpp``) or the oracle
    ``_SpanwiseElementExtractionPython``. The *operations* below -- the apply
    family, :meth:`operator`, :meth:`tabulate`, :meth:`factors`, the indexing and
    the cell-index normalisation -- are computations *over* an extraction rather
    than properties *of* one, so they are unchanged, still run on the Layer-3
    kernels, and live on the wrapper. Only the state moved.

    Instances are immutable, and that is enforced rather than documented:
    ``__slots__`` means there is no ``__dict__`` to attach anything to, and
    ``__setattr__`` refuses even a rebinding of the slots. The wrapper fills them
    through ``object.__setattr__``, which is the pattern
    ``design/bspline_ownership_lifetime.md`` asks for and
    :class:`~pantr.bspline.BsplineSpace` already ships.

    Attributes:
        _impl: The implementation this wrapper holds; see ``_impl_class``. Its type
            is the private ``_Impl`` alias, a union of three unrelated nominal types
            with no documented form to name here.
        _space (BsplineSpace): The space wrapper this extraction was built from, so
            that ``extraction.space is space`` holds -- ``design/bspline_ownership_
            lifetime.md`` F6's identity contract. A *presentation* memo and never a
            second truth: every count comes from ``_impl`` on every access.
        _compact_ops_1d (tuple[npt.NDArray[np.float32 | np.float64], ...]): The
            implementation's compact 3D operator arrays of shape
            ``(n_compact_k, n_out_k, n_in_k)``, read once at construction; only
            non-identity rows are stored, and there is always at least one row to
            ensure safe Numba indexing. Held rather than re-read because under the
            C++ backend each property access builds a fresh tuple of fresh views over
            the same storage, which is an allocation per element in the per-cell
            paths and loses the object identity the Python backend has always had.
        _idx_maps_1d (tuple[npt.NDArray[np.intp], ...]): Likewise for the
            per-direction compact index maps of shape ``(n_elements_k,)``;
            ``_idx_maps_1d[k][e]`` is the row index into ``_compact_ops_1d[k]`` for
            element ``e`` (undefined for identity elements, stored as 0).
        _is_identity_mask_1d (tuple[npt.NDArray[np.bool_], ...]): Likewise for the
            per-direction identity masks of shape ``(n_elements_k,)``.
        _dense_ops_1d (tuple[npt.NDArray[np.float32 | np.float64], ...] | None): The
            same memo for :attr:`ops_1d`, filled on first read rather than at
            construction, because the implementation's own block is lazy and
            decompressing it eagerly would undo that.
    """

    __slots__ = (
        "_compact_ops_1d",
        "_dense_ops_1d",
        "_idx_maps_1d",
        "_impl",
        "_is_identity_mask_1d",
        "_space",
    )

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

    _space: BsplineSpace
    """The space wrapper this extraction was built from; see the class docstring."""

    _compact_ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...]
    """The implementation's compact operators; see the class docstring."""

    _idx_maps_1d: tuple[npt.NDArray[np.intp], ...]
    """The implementation's index maps; see the class docstring."""

    _is_identity_mask_1d: tuple[npt.NDArray[np.bool_], ...]
    """The implementation's identity masks; see the class docstring."""

    _dense_ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...] | None
    """The implementation's dense operators once read; see the class docstring."""

    def __init__(
        self,
        space: BsplineSpace,
        target: TargetLike,
        *,
        lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    ) -> None:
        """Build the per-direction operators and identity masks.

        Args:
            space (BsplineSpace): Multi-dimensional B-spline space.
            target (TargetLike): An :class:`ExtractionTarget`, or its legacy string
                spelling ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
            lagrange_variant (LagrangeVariant): Point distribution used when
                ``target`` is :attr:`ExtractionTarget.LAGRANGE`. Defaults to
                :attr:`pantr.basis.LagrangeVariant.EQUISPACES`.

        Raises:
            ValueError: If ``target`` is not a recognized tag, if the per-direction
                operators do not share one dtype, or if ``space`` was built under a
                different backend.
            NotImplementedError: If any direction of ``space`` is periodic;
                periodic support is deferred to a later version.
            RuntimeError: If the C++ backend is requested and is not available.
        """
        resolved_target = _coerce_target(target)

        if any(s.periodic for s in space.spaces):
            raise NotImplementedError(
                "SpanwiseElementExtraction does not yet support periodic directions. "
                "Convert the B-spline to open form first (see Bspline.to_open_bspline)."
            )

        impl = _new_impl(space, resolved_target, lagrange_variant)
        object.__setattr__(self, "_impl", impl)
        object.__setattr__(self, "_space", space)
        object.__setattr__(self, "_compact_ops_1d", tuple(impl.compact_ops_1d))
        object.__setattr__(self, "_idx_maps_1d", tuple(impl.idx_maps_1d))
        object.__setattr__(self, "_is_identity_mask_1d", tuple(impl.is_identity_mask_1d))
        object.__setattr__(self, "_dense_ops_1d", None)

    def __setattr__(self, name: str, value: object) -> NoReturn:
        """Refuse to set an attribute, because an extraction is immutable.

        Args:
            name (str): The attribute a caller tried to set.
            value (object): The value it tried to set.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        """Refuse to delete an attribute, because an extraction is immutable.

        Args:
            name (str): The attribute a caller tried to delete.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable; cannot delete {name!r}")

    def __reduce__(
        self,
    ) -> tuple[object, tuple[BsplineSpace, ExtractionTarget, LagrangeVariant]]:
        """Pickle by the constructor's arguments rather than by implementation.

        The C++ handle is not picklable and must not become part of the wire format:
        a pickle written under the C++ backend has to load under the Python one and
        the other way round, or the backend switch would silently become a
        data-format switch. ``design/bspline_pickle_tolerance.md`` fixes the rule.

        The space goes out as its *wrapper*, so its own ``__reduce__`` runs and the
        knot vectors survive rather than a handle being smuggled through -- and with
        them the tolerance-drift bound that note derives for a univariate space. It
        is also what makes sharing survive a single pickle for free, since ``pickle``
        memoises: dumping ``(ext, ext.space)`` restores a pair that still satisfies
        ``ext.space is space``. Sharing does not survive two independent ``dumps``
        calls, which is true of every other type in this front.

        The operators are **not** in the payload. They are a function of the space,
        the target and the point distribution, and rebuilding them is what keeps a
        pickle valid across a change of backend: shipping the arrays instead would
        pin a reader to the writer's own builders.

        A module-level rebuilder rather than ``type(self)`` directly, because
        ``lagrange_variant`` is keyword-only and ``__reduce__``'s argument tuple is
        positional.

        Returns:
            tuple: The rebuilder and the three arguments to rebuild from.
        """
        return (
            _rebuild_spanwise_element_extraction,
            (self.space, self.target, self.lagrange_variant),
        )

    # ---------------------------------------------------------------- properties

    @property
    def space(self) -> BsplineSpace:
        """Get the underlying B-spline space.

        The object a caller passed to the constructor, not a re-wrapping of what the
        implementation holds, so ``extraction.space is space`` holds under both
        backends. ``design/bspline_ownership_lifetime.md`` F6 records why: no C++
        object can supply the constructor argument's own Python object, and only the
        wrapper can, by keeping what it was built from.

        Returns:
            BsplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def target(self) -> ExtractionTarget:
        """Get the target basis.

        Returns:
            ExtractionTarget: The element-local basis this extraction maps onto.
                A string passed at construction is resolved to its enum member,
                so this never returns a string.
        """
        return ExtractionTarget(self._impl.target)

    @property
    def lagrange_variant(self) -> LagrangeVariant:
        """Get the Lagrange point distribution used for the Lagrange target.

        Returns:
            LagrangeVariant: The point distribution. Meaningless for other targets.
        """
        return LagrangeVariant(self._impl.lagrange_variant)

    @property
    def dim(self) -> int:
        """Get the number of tensor-product directions.

        Returns:
            int: The dimension ``d`` of the space.
        """
        return int(self._impl.dim)

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
        return tuple(int(n) for n in self._impl.num_intervals)

    @property
    def num_total_intervals(self) -> int:
        """Get the total number of elements across the tensor-product grid.

        Returns:
            int: ``prod(num_intervals)``.
        """
        return int(self._impl.num_total_intervals)

    @property
    def ops_1d(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the per-direction 1D operator arrays (dense, reconstructed lazily).

        The implementation reconstructs the full ``(n_elements_k, n_out_k, n_in_k)``
        array from compact storage on first access and memoises it; this hands back a
        read-only view *of that memo* and never a copy. Identity elements are filled
        with ``numpy.eye(n_out, n_in)``; non-identity elements are read from
        :attr:`compact_ops_1d`. Every extraction target builds square per-direction
        operators, and :meth:`apply` and :meth:`apply_many` refuse an
        identity-flagged operator that is not square.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: Length-``d`` tuple
            of read-only 3D arrays; ``ops_1d[k]`` has shape
            ``(n_elements_k, n_out_k, n_in_k)``. Intended for consumption by
            downstream ``@njit`` code when the full dense layout is required.
            For compact-aware downstream code, prefer :attr:`compact_ops_1d` and
            :attr:`idx_maps_1d`.
        """
        dense = self._dense_ops_1d
        if dense is None:
            dense = tuple(self._impl.ops_1d)
            object.__setattr__(self, "_dense_ops_1d", dense)
        return dense

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
        return tuple(int(n) for n in self._impl.input_shape_per_dir)

    @property
    def output_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction output sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_out_0, …, n_out_{d-1})``.
        """
        return tuple(int(n) for n in self._impl.output_shape_per_dir)

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

    @property
    def num_identity_elements(self) -> int:
        """Count elements whose per-direction operators are all identity.

        An eager field of the implementation rather than a memo here, which is what
        ``design/bspline_derived_caches.md`` assigns it: the count is one pass over
        the masks at construction, and the oracle memoised it only because a Python
        attribute read is dearer than the count.

        Returns:
            int: The number of fully-identity elements on the tensor-product grid.
        """
        return int(self._impl.num_identity_elements)

    @property
    def is_identity(self) -> bool:
        """Check whether every element on the grid has an identity operator.

        Returns:
            bool: ``True`` iff all per-direction identity masks are all-``True``,
            meaning every element's operator is the identity.
        """
        return bool(self._impl.is_identity)

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

    def factors(
        self, cell_idx: CellIndex
    ) -> tuple[tuple[bool, npt.NDArray[np.float32 | np.float64] | None], ...]:
        """Return the per-direction 1D factors of one element's operator.

        The element operator is ``kron(F_0, …, F_{d-1})`` where ``F_k`` is the
        returned direction-``k`` operator, or the identity where the flag is set.
        This is the same factorization :meth:`operator` and ``__getitem__``
        use, exposed without materializing anything: identity directions yield
        ``None`` rather than an identity matrix, and non-identity directions yield
        a read-only view into :attr:`compact_ops_1d` rather than a copy. Elements
        sharing a compact row therefore share its memory -- the view object itself
        is new on each call, as numpy basic indexing always builds one, but no
        operator data is copied.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            tuple[tuple[bool, npt.NDArray[np.float32 | np.float64] | None], ...]:
            Length-``d`` tuple whose entry ``k`` is ``(is_identity_k, op_k)``.
            ``op_k`` is ``None`` exactly when ``is_identity_k`` is ``True``;
            otherwise it is a read-only ``(n_out_k, n_in_k)`` view.

        Raises:
            IndexError: If a flat index is out of range, or a per-direction entry
                is out of range for its direction.
            ValueError: If a per-direction index has the wrong length.
            TypeError: If ``cell_idx`` is not an ``int`` or sequence of ``int``.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
            >>> space = BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 4, 4], 2)])
            >>> ext = SpanwiseElementExtraction(space, "cardinal")
            >>> ext.factors(1)
            ((True, None),)
            >>> flag, op = ext.factors(0)[0]
            >>> flag, op.shape, op.flags.writeable
            (False, (3, 3), False)
        """
        multi = self._normalize_cell_idx(cell_idx)
        factors: list[tuple[bool, npt.NDArray[np.float32 | np.float64] | None]] = []
        for compact, idx_map, mask, i in zip(
            self._compact_ops_1d, self._idx_maps_1d, self._is_identity_mask_1d, multi, strict=True
        ):
            if bool(mask[i]):
                factors.append((True, None))
            else:
                factors.append((False, compact[int(idx_map[i])]))
        return tuple(factors)

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
                array of shape ``(prod(output_shape_per_dir),)``. Must not
                alias ``v``. Allocated
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
                array of shape ``(prod(input_shape_per_dir),)``. Must not
                alias ``w``.
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
            2D array of shape ``(n_out_k, n_in_k)`` (identity elements return a
            fresh ``numpy.eye``; non-identity elements return a row from
            :attr:`compact_ops_1d`) and ``identity_flags`` is the per-direction
            identity mask at this element.
        """
        ops, flags = self._ops_and_flags_for_cell(self._normalize_cell_idx(cell_idx))
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

    def _ops_and_flags_for_cell(
        self, multi: tuple[int, ...]
    ) -> tuple[
        tuple[npt.NDArray[np.float32 | np.float64], ...],
        tuple[bool, ...],
    ]:
        """Extract per-direction operators and identity flags from compact storage.

        Args:
            multi (tuple[int, ...]): Already-validated per-direction element indices.

        Returns:
            tuple[tuple[npt.NDArray, ...], tuple[bool, ...]]: ``(ops, flags)``
            where each ``ops[k]`` is a 2D ``(n_out_k, n_in_k)`` array from
            compact storage (or a fresh ``numpy.eye`` for identity elements).
        """
        ops: list[npt.NDArray[np.float32 | np.float64]] = []
        flags: list[bool] = []
        for compact, idx_map, mask, i in zip(
            self._compact_ops_1d, self._idx_maps_1d, self._is_identity_mask_1d, multi, strict=True
        ):
            n_out, n_in = int(compact.shape[1]), int(compact.shape[2])
            is_id = bool(mask[i])
            flags.append(is_id)
            if is_id:
                ops.append(np.eye(n_out, n_in, dtype=compact.dtype))
            else:
                ops.append(compact[int(idx_map[i])])
        return tuple(ops), tuple(flags)

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
        ops, _ = self._ops_and_flags_for_cell(self._normalize_cell_idx(cell_idx))
        return ops

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
        ops, flags = self._ops_and_flags_for_cell(self._normalize_cell_idx(cell_idx))
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
            npt.NDArray[np.float32 | np.float64]: Result array; shape is
            ``(n_cells, N_out)`` for ``"apply"``, ``(n_cells, N_in)`` for
            ``"apply_T"``, ``(n_cells, N_in, N_in)`` for ``"MT_K_M"``, or
            ``(n_cells, N_out, N_out)`` for ``"M_K_MT"``.

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
                array of shape ``(n_cells, N_out)``. Must not alias ``v``.
                Allocated if ``None``.
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
                array of shape ``(n_cells, N_in)``. Must not alias ``w``.
                Allocated if ``None``.
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


def _rebuild_spanwise_element_extraction(
    space: BsplineSpace,
    target: ExtractionTarget,
    lagrange_variant: LagrangeVariant,
) -> SpanwiseElementExtraction:
    """Rebuild an extraction from its constructor's arguments, for :mod:`pickle`.

    Module-level rather than :class:`SpanwiseElementExtraction` itself, because
    ``lagrange_variant`` is keyword-only and ``__reduce__``'s argument tuple is
    positional. It re-runs the constructor in full, so a pickle written under one
    backend rebuilds under whichever is active at load time.

    Args:
        space (BsplineSpace): The space to extract from.
        target (ExtractionTarget): The element-local basis.
        lagrange_variant (LagrangeVariant): The point distribution.

    Returns:
        SpanwiseElementExtraction: The rebuilt extraction.
    """
    return SpanwiseElementExtraction(space, target, lagrange_variant=lagrange_variant)


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
    bezier_identity_mask_kernel()(mults, space_1d.degree, out)
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
    can have an identity Lagrange extraction operator; the argument for that
    half, which is a claim rather than a definition, is written out beside
    ``lagrange_structural_identity_mask`` in
    ``cpp/include/pantr/bspline/extraction.hpp``.

    The ``degree == 0`` branch stays here rather than moving into the dispatched
    kernel, and it is not a special case that could be folded away:
    :func:`pantr.change_basis.compute_lagrange_to_bernstein_1d` refuses a degree
    below 1, so there is no matrix to pass and no kernel call to make.

    Args:
        space_1d (BsplineSpace1D): A 1D B-spline space.
        lagrange_variant (LagrangeVariant): Lagrange node distribution.

    Returns:
        npt.NDArray[np.bool_]: Boolean array of shape ``(n_elements,)``.
    """
    if space_1d.degree == 0:
        return np.ones(space_1d.num_intervals, dtype=np.bool_)
    dtype = space_1d.knots.dtype
    lagr_to_bzr = _cached_lagrange_to_bernstein_matrix(space_1d.degree, lagrange_variant, dtype)
    _, mults = space_1d.get_unique_knots_and_multiplicity(in_domain=True)
    out = np.empty(len(mults) - 1, dtype=np.bool_)
    lagrange_identity_mask_kernel()(mults, space_1d.degree, lagr_to_bzr, out)
    return out


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


class ExtractionStructView(NamedTuple):
    """Immutable struct view of a :class:`SpanwiseElementExtraction` for ``@njit`` callers.

    A :class:`typing.NamedTuple` bundling the compact per-direction operator
    storage, index maps, identity masks, and shape metadata into a single
    object that Numba can unbox. Each array field is homogeneous in dtype and
    array dimensionality across all directions (per-direction shapes may
    differ), so Numba represents those tuple fields as ``UniTuple`` inside an
    ``@njit`` function.
    This makes ``ExtractionStructView`` a drop-in replacement for the separate
    ``(ops_1d, idx_maps_1d, is_identity_mask_1d, …)`` bundle when calling the
    Layer-3 batch kernels in ``pantr.bspline._extraction_kernels`` from
    downstream Numba code.

    Construct via :func:`make_struct_view`. Field semantics mirror the
    same-named members of :class:`SpanwiseElementExtraction`:

    - ``compact_ops_1d`` — per-direction compact 3D operator arrays of shape
      ``(n_compact_k, n_out_k, n_in_k)``; only non-identity rows are stored.
      Always has at least one row (sentinel zeros) to ensure safe Numba
      indexing.
    - ``idx_maps_1d`` — per-direction compact index maps of shape
      ``(n_elements_k,)``.
    - ``is_identity_mask_1d`` — per-direction identity masks of shape
      ``(n_elements_k,)``.
    - ``num_intervals`` — per-direction number of elements.
    - ``input_shape_per_dir`` — per-direction input sizes
      ``(n_in_0, …, n_in_{d-1})``.
    - ``output_shape_per_dir`` — per-direction output sizes
      ``(n_out_0, …, n_out_{d-1})``.
    - ``dim`` — number of tensor-product directions ``d``.
    """

    compact_ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...]
    idx_maps_1d: tuple[npt.NDArray[np.intp], ...]
    is_identity_mask_1d: tuple[npt.NDArray[np.bool_], ...]
    num_intervals: tuple[int, ...]
    input_shape_per_dir: tuple[int, ...]
    output_shape_per_dir: tuple[int, ...]
    dim: int


def make_struct_view(extraction: SpanwiseElementExtraction) -> ExtractionStructView:
    """Bundle a :class:`SpanwiseElementExtraction` into a Numba-passable struct view.

    Shares the underlying per-direction arrays (no copies). The arrays are
    already marked read-only by :class:`SpanwiseElementExtraction`, so the
    returned view is safe to pass into ``@njit`` code without risk of
    accidental mutation.

    Args:
        extraction (SpanwiseElementExtraction): Source extraction object.

    Returns:
        ExtractionStructView: Named tuple wrapping the extraction's compact
        storage and shape metadata. Suitable for direct use as a single
        argument to ``@njit`` functions that call the Layer-3 batch kernels
        in ``pantr.bspline._extraction_kernels``.

    Example:
        >>> from pantr.bspline import BsplineSpace1D, BsplineSpace
        >>> from pantr.bspline import SpanwiseElementExtraction, make_struct_view
        >>> sp = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
        >>> space = BsplineSpace([sp, sp])
        >>> ext = SpanwiseElementExtraction(space, "bezier")
        >>> view = make_struct_view(ext)
        >>> view.dim
        2
    """
    return ExtractionStructView(
        compact_ops_1d=extraction.compact_ops_1d,
        idx_maps_1d=extraction.idx_maps_1d,
        is_identity_mask_1d=extraction.is_identity_mask_1d,
        num_intervals=tuple(int(n) for n in extraction.num_intervals),
        input_shape_per_dir=extraction.input_shape_per_dir,
        output_shape_per_dir=extraction.output_shape_per_dir,
        dim=int(extraction.dim),
    )


__all__ = [
    "CellIndex",
    "CellIndicesBatch",
    "ExtractionStructView",
    "ExtractionTarget",
    "SpanwiseElementExtraction",
    "Target",
    "TargetLike",
    "make_struct_view",
    "normalize_cell_indices",
    "operand_shape",
]
