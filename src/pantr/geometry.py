"""Axis-aligned bounding boxes and geometry primitives.

This module exposes :class:`AABB`, a lightweight, immutable value type for an
axis-aligned bounding box in any spatial dimension ``ndim >= 1``. It is the
shared box / domain primitive for PaNTr (for example, the parametric domain of
a spline space or the bounds of a grid cell) and for libraries built on PaNTr.

An :class:`AABB` stores two read-only ``float64`` corner arrays :attr:`lo` and
:attr:`hi` of shape ``(ndim,)``. Entries may be finite or ``+/- numpy.inf`` (for
unbounded axes); ``numpy.nan`` is rejected. A point ``x`` is inside the box when
``lo[i] <= x[i] <= hi[i]`` on every axis; a box with ``lo[i] > hi[i]`` on some
axis is *empty* and reported by :meth:`AABB.is_empty`.

Main exports:

- :class:`AABB` -- immutable axis-aligned bounding box.
"""

from __future__ import annotations

from typing import Final, Protocol

import numpy as np
from numpy import typing as npt

# Shape constants for :meth:`AABB.from_bounds` / :meth:`AABB.as_bounds`.
_BOUNDS_NDIM: Final[int] = 2  # bounds array must be exactly 2-D (axes x {lo,hi})
_BOUNDS_AXIS_LEN: Final[int] = 2  # last axis encodes [lo, hi]


class _AffineMap(Protocol):
    """Structural protocol for an affine map ``T(x) = matrix @ x + offset``.

    Any object exposing :attr:`dim`, :attr:`matrix`, and :attr:`offset` satisfies
    :meth:`AABB.transform`. This decouples the box primitive from a specific
    affine-transform class (for example :class:`pantr.transform.AffineTransform`
    or a downstream library's own affine type).
    """

    @property
    def dim(self) -> int:
        """Get the spatial dimension of the map.

        Returns:
            int: The dimension ``n``.
        """
        ...

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Get the linear part ``A`` of the map.

        Returns:
            npt.NDArray[np.float64]: The ``(n, n)`` matrix.
        """
        ...

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Get the translation part ``b`` of the map.

        Returns:
            npt.NDArray[np.float64]: The ``(n,)`` vector.
        """
        ...


def _as_float64(arr: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    """Coerce ``arr`` to a ``float64`` array, preserving rank.

    Integer and unsigned-integer inputs are cast to ``float64``; boolean and
    non-numeric inputs are rejected. Results of rank ``>= 1`` are made
    C-contiguous.

    Args:
        arr (npt.ArrayLike): Input array or array-like.
        name (str): Argument name, used in error messages.

    Returns:
        npt.NDArray[np.float64]: A ``float64`` view or copy of ``arr``.

    Raises:
        TypeError: If ``arr`` cannot be converted to an ndarray, or its dtype
            is neither integer nor floating-point (e.g. boolean, complex,
            object).
    """
    try:
        a = np.asarray(arr)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} could not be converted to an ndarray: {exc}") from exc
    if a.dtype.kind not in ("f", "i", "u"):
        raise TypeError(f"{name} must have a numeric (int or float) dtype; got {a.dtype!r}.")
    a = a.astype(np.float64, copy=False)
    if a.ndim >= 1 and not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return a


class AABB:
    """Axis-aligned bounding box in any spatial dimension ``ndim >= 1``.

    An :class:`AABB` stores two 1-D arrays :attr:`lo` and :attr:`hi` of equal
    shape ``(ndim,)``. Entries may be finite or ``+/- numpy.inf`` (for unbounded
    axes); ``numpy.nan`` is rejected. Instances are frozen: the stored arrays
    are C-contiguous, ``float64``, and flagged read-only, and ``__setattr__``
    rejects attempts to replace the bound attributes.

    The sign convention is the natural one: a point ``x`` is "inside" the box
    when ``lo[i] <= x[i] <= hi[i]`` on every axis. A box with ``lo[i] > hi[i]``
    on some axis is *empty* (no point satisfies the inequality);
    :meth:`is_empty` reports this.

    Attributes:
        lo (npt.NDArray[np.float64]): Lower corner, shape ``(ndim,)``, read-only.
        hi (npt.NDArray[np.float64]): Upper corner, shape ``(ndim,)``, read-only.
    """

    __slots__ = ("hi", "lo")

    lo: npt.NDArray[np.float64]
    hi: npt.NDArray[np.float64]

    def __init__(self, lo: npt.ArrayLike, hi: npt.ArrayLike) -> None:
        """Build and validate an AABB from array-like corners.

        The arguments are coerced to C-contiguous ``float64`` arrays and
        checked for NaN. The resulting buffers are cached as read-only
        attributes on ``self``.

        Args:
            lo (npt.ArrayLike): Lower corner; array-like of finite-or-infinite floats.
                Any rank is accepted and ravelled to 1-D of length ``ndim``.
            hi (npt.ArrayLike): Upper corner, same shape and semantics as ``lo``.

        Raises:
            ValueError: If the shapes mismatch, contain a NaN, or the implied
                ``ndim`` is less than 1.
            TypeError: If ``lo`` or ``hi`` has a non-numeric dtype.
        """
        lo_arr = _as_float64(lo, name="lo").ravel()
        hi_arr = _as_float64(hi, name="hi").ravel()
        if lo_arr.shape != hi_arr.shape:
            raise ValueError(
                f"AABB.lo and AABB.hi must share shape; got {lo_arr.shape} vs {hi_arr.shape}."
            )
        ndim = int(lo_arr.shape[0])
        if ndim < 1:
            raise ValueError(f"AABB ndim must be >= 1; got {ndim}.")
        if np.any(np.isnan(lo_arr)) or np.any(np.isnan(hi_arr)):
            raise ValueError(
                f"AABB bounds must not contain NaN; got lo={lo_arr.tolist()!r}, "
                f"hi={hi_arr.tolist()!r}."
            )
        frozen_lo = lo_arr.copy()
        frozen_hi = hi_arr.copy()
        frozen_lo.flags.writeable = False
        frozen_hi.flags.writeable = False
        object.__setattr__(self, "lo", frozen_lo)
        object.__setattr__(self, "hi", frozen_hi)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject post-construction attribute writes.

        Raises:
            AttributeError: Always -- :class:`AABB` is immutable.
        """
        raise AttributeError(f"AABB is immutable; cannot set attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion.

        Raises:
            AttributeError: Always -- :class:`AABB` is immutable.
        """
        raise AttributeError(f"AABB is immutable; cannot delete attribute {name!r}.")

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Returns:
            str: ``"AABB(lo=..., hi=...)"`` with corner values as Python lists.
        """
        return f"AABB(lo={self.lo.tolist()!r}, hi={self.hi.tolist()!r})"

    def __eq__(self, other: object) -> bool:
        """Compare value-based equality: two AABBs are equal iff their bounds match.

        This is *value* equality, not geometric equality: two empty AABBs whose
        corner arrays differ (e.g. ``AABB.empty(2)`` vs a hand-constructed empty)
        compare unequal even though they contain the same set of points.

        Args:
            other (object): Expected to be an :class:`AABB`.

        Returns:
            bool: ``True`` when both corner arrays are element-wise equal
            (``+inf == +inf`` and ``-inf == -inf`` as usual).

        Note:
            Returns :data:`NotImplemented` for non-:class:`AABB` ``other`` so
            that Python's reflected equality protocol works correctly.
        """
        if not isinstance(other, AABB):
            return NotImplemented
        return bool(np.array_equal(self.lo, other.lo) and np.array_equal(self.hi, other.hi))

    def __hash__(self) -> int:
        """Hash based on the immutable corner bytes.

        Returns:
            int: Hash compatible with :meth:`__eq__`; equal AABBs hash equal.
        """
        return hash((self.lo.tobytes(), self.hi.tobytes()))

    @property
    def ndim(self) -> int:
        """Get the spatial dimensionality of the box.

        Returns:
            int: The number of axes (``>= 1``).
        """
        return int(self.lo.shape[0])

    def is_empty(self) -> bool:
        """Check whether the box contains no points.

        A box is empty iff ``lo[i] > hi[i]`` on at least one axis. Useful when
        constructing boxes manually, accepting boxes from external sources, or
        after :meth:`pad` with a negative radius.

        Returns:
            bool: ``True`` when the box is empty.
        """
        return bool(np.any(self.lo > self.hi))

    def contains_point(self, x: npt.ArrayLike) -> bool:
        """Check whether point ``x`` lies inside or on the boundary of the box.

        A point is inside iff ``lo[i] <= x[i] <= hi[i]`` on every axis.
        An empty box contains no points.

        Args:
            x (npt.ArrayLike): Point to test; array-like of floats, ravelled to
                1-D of length ``ndim``.

        Returns:
            bool: ``True`` when ``x`` is inside or on the boundary.

        Raises:
            ValueError: If the ravelled ``x`` does not have length ``ndim``,
                or if ``x`` contains NaN.
            TypeError: If ``x`` has a non-numeric dtype.
        """
        x_arr = _as_float64(x, name="x").ravel()
        if x_arr.shape != (self.ndim,):
            raise ValueError(f"contains_point: x must have length {self.ndim}; got {x_arr.shape}.")
        if np.any(np.isnan(x_arr)):
            raise ValueError("contains_point: x must not contain NaN.")
        return bool(np.all((x_arr >= self.lo) & (x_arr <= self.hi)))

    def overlaps(self, other: AABB) -> bool:
        """Check whether ``self`` and ``other`` share at least one point.

        Empty boxes (on either side) overlap nothing.

        Args:
            other (AABB): The box to test against; must share :attr:`ndim`.

        Returns:
            bool: ``True`` when the two boxes intersect, ``False`` otherwise.

        Raises:
            ValueError: If ``other.ndim != self.ndim``.
        """
        _require_same_ndim(self, other, op="overlaps")
        if self.is_empty() or other.is_empty():
            return False
        lo = np.maximum(self.lo, other.lo)
        hi = np.minimum(self.hi, other.hi)
        return bool(np.all(lo <= hi))

    def union(self, other: AABB) -> AABB:
        """Return the smallest axis-aligned box that contains ``self`` and ``other``.

        Empty boxes act as neutral elements: ``union(empty, x) == x``.

        Args:
            other (AABB): Box to union with; must share :attr:`ndim`.

        Returns:
            AABB: The union bounding box.

        Raises:
            ValueError: If ``other.ndim != self.ndim``.
        """
        _require_same_ndim(self, other, op="union")
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        return AABB(np.minimum(self.lo, other.lo), np.maximum(self.hi, other.hi))

    def intersect(self, other: AABB) -> AABB | None:
        """Return the axis-aligned intersection, or ``None`` if disjoint.

        Args:
            other (AABB): Box to intersect with; must share :attr:`ndim`.

        Returns:
            AABB | None: The intersection, or ``None`` when the two boxes do not
            overlap (including the case where either operand is empty).

        Raises:
            ValueError: If ``other.ndim != self.ndim``.
        """
        _require_same_ndim(self, other, op="intersect")
        if self.is_empty() or other.is_empty():
            return None
        lo = np.maximum(self.lo, other.lo)
        hi = np.minimum(self.hi, other.hi)
        if np.any(lo > hi):
            return None
        return AABB(lo, hi)

    def pad(self, r: float | npt.ArrayLike) -> AABB:
        """Inflate the box by ``r`` on every axis (symmetric on both sides).

        A scalar ``r`` expands every axis by the same amount; a length-``ndim``
        array expands each axis by its own entry. Padding an unbounded axis
        leaves it unbounded. Padding by a negative ``r`` shrinks the box and can
        produce an empty AABB, which is allowed and reported by
        :meth:`is_empty`.

        Args:
            r (float | npt.ArrayLike): Per-side padding amount. Scalar or
                length-``ndim`` array-like of finite reals.

        Returns:
            AABB: The padded box.

        Raises:
            ValueError: If ``r`` has the wrong shape or non-finite entries.
        """
        arr = _as_float64(r, name="r")
        if arr.ndim == 0:
            pad_vec = np.full(self.ndim, float(arr), dtype=np.float64)
        else:
            pad_vec = arr.ravel()
            if pad_vec.shape != (self.ndim,):
                raise ValueError(
                    f"pad(r) requires r scalar or shape ({self.ndim},); got {pad_vec.shape}."
                )
        if not np.all(np.isfinite(pad_vec)):
            raise ValueError(f"pad(r) entries must be finite; got {pad_vec.tolist()!r}.")
        # Inf bounds are preserved by numpy: (+inf) + finite == +inf; (-inf) - finite == -inf.
        return AABB(self.lo - pad_vec, self.hi + pad_vec)

    def transform(self, affine: _AffineMap) -> AABB:
        """Return the axis-aligned wrap of the image of ``self`` under ``affine``.

        For an affine map ``T(x) = A x + b``, the tight axis-aligned box
        containing ``T(self)`` is computed from the per-entry sign of ``A``:
        each output axis ``i`` receives contributions ``A[i, j] * lo[j]`` or
        ``A[i, j] * hi[j]`` -- whichever gives the minimum / maximum -- summed
        over ``j``. Zero entries of ``A`` contribute nothing even when
        ``lo[j]`` / ``hi[j]`` is infinite; this preserves the correct wrap for
        unbounded axes that the transform projects out.

        Args:
            affine (_AffineMap): The affine map to apply -- any object exposing
                ``dim``, ``matrix`` (``A``), and ``offset`` (``b``). Its
                dimension must match :attr:`ndim`.

        Returns:
            AABB: The axis-aligned wrap of the transformed box.

        Raises:
            ValueError: If ``affine.dim != self.ndim``, if ``affine.matrix`` is
                not square ``(ndim, ndim)``, or if the wrap produces NaN (e.g.
                a matrix containing NaN, or ``inf - inf`` arithmetic from
                opposing infinite bounds in the same row).
        """
        if self.is_empty():
            return AABB.empty(self.ndim)
        if affine.dim != self.ndim:
            raise ValueError(
                f"transform(): affine dim ({affine.dim}) must match AABB ndim ({self.ndim})."
            )
        a = affine.matrix
        if a.shape != (self.ndim, self.ndim):
            raise ValueError(
                f"transform(): affine.matrix must be ({self.ndim}, {self.ndim}); got {a.shape}."
            )
        b = affine.offset
        # Per output axis i and input axis j the contribution is the pair
        #   (A[i,j] * lo[j], A[i,j] * hi[j]);
        # min / max of this pair go into new_lo[i] and new_hi[i]. Zeros in ``A``
        # must contribute nothing, even when ``lo[j]`` / ``hi[j]`` is infinite --
        # otherwise ``0 * inf == NaN`` would poison the output. ``np.errstate``
        # suppresses the harmless warning; ``np.where`` masks those NaN slots.
        lo = self.lo
        hi = self.hi
        mask = a != 0.0
        # Suppress invalid-value warnings: 0*inf=NaN is masked away by np.where, and
        # inf+(-inf)=NaN in the row sums is caught explicitly by the NaN check below.
        with np.errstate(invalid="ignore"):
            term_lo = a * lo[np.newaxis, :]
            term_hi = a * hi[np.newaxis, :]
            term_lo = np.where(mask, term_lo, 0.0)
            term_hi = np.where(mask, term_hi, 0.0)
            contrib_min = np.minimum(term_lo, term_hi)
            contrib_max = np.maximum(term_lo, term_hi)
            new_lo = np.sum(contrib_min, axis=1) + b
            new_hi = np.sum(contrib_max, axis=1) + b
        if np.any(np.isnan(new_lo)) or np.any(np.isnan(new_hi)):
            raise ValueError(
                "AABB.transform produced NaN bounds; the transform is incompatible with "
                "this AABB (for example, a singular matrix combined with infinite bounds)."
            )
        return AABB(new_lo, new_hi)

    @staticmethod
    def unbounded(ndim: int) -> AABB:
        """Build the everywhere-true AABB ``(-inf, +inf)^ndim``.

        Args:
            ndim (int): Spatial dimension (``>= 1``).

        Returns:
            AABB: The unbounded AABB.

        Raises:
            ValueError: If ``ndim < 1``.
        """
        if ndim < 1:
            raise ValueError(f"AABB.unbounded: ndim must be >= 1; got {ndim}.")
        lo = np.full(ndim, -np.inf, dtype=np.float64)
        hi = np.full(ndim, np.inf, dtype=np.float64)
        return AABB(lo, hi)

    @staticmethod
    def empty(ndim: int) -> AABB:
        """Build an empty (zero-volume) AABB with ``lo > hi``.

        An empty AABB contains no points (detected by :meth:`is_empty`) and acts
        as the neutral element of :meth:`union` (``union(empty, x) == x``).
        Symmetric counterpart to :meth:`unbounded`.

        Args:
            ndim (int): Spatial dimension (``>= 1``).

        Returns:
            AABB: An empty AABB (``lo = +inf``, ``hi = -inf``).

        Raises:
            ValueError: If ``ndim < 1``.
        """
        if ndim < 1:
            raise ValueError(f"AABB.empty: ndim must be >= 1; got {ndim}.")
        lo = np.full(ndim, np.inf, dtype=np.float64)
        hi = np.full(ndim, -np.inf, dtype=np.float64)
        return AABB(lo, hi)

    @staticmethod
    def from_bounds(bounds: npt.ArrayLike) -> AABB:
        """Build an AABB from a ``(ndim, 2)`` ``[[lo, hi], ...]`` array.

        The dual of :meth:`as_bounds`; useful when interoperating with
        ``numpy.histogramdd``-style bounds arguments.

        Args:
            bounds (npt.ArrayLike): ``(ndim, 2)`` array-like of ``[lo, hi]`` rows.

        Returns:
            AABB: The constructed AABB.

        Raises:
            ValueError: If ``bounds`` does not have shape ``(ndim, 2)`` with
                ``ndim >= 1``.
            TypeError: If ``bounds`` has a non-numeric dtype.
        """
        arr = _as_float64(bounds, name="bounds")
        if arr.ndim != _BOUNDS_NDIM or arr.shape[-1] != _BOUNDS_AXIS_LEN:
            raise ValueError(f"from_bounds: bounds must have shape (ndim, 2); got {arr.shape}.")
        return AABB(arr[:, 0], arr[:, 1])

    def as_bounds(self) -> npt.NDArray[np.float64]:
        """Return the AABB as a ``(ndim, 2)`` ``[[lo, hi], ...]`` array.

        Returns:
            npt.NDArray[np.float64]: A freshly-allocated, writeable, C-contiguous
            ``(ndim, 2)`` array.
        """
        out = np.empty((self.ndim, _BOUNDS_AXIS_LEN), dtype=np.float64)
        out[:, 0] = self.lo
        out[:, 1] = self.hi
        return out


def _require_same_ndim(a: AABB, b: AABB, *, op: str) -> None:
    """Raise ``ValueError`` if two AABBs have mismatched dimensions.

    Args:
        a (AABB): First operand.
        b (AABB): Second operand.
        op (str): Operation name for the error message.

    Raises:
        ValueError: If ``a.ndim != b.ndim``.
    """
    if a.ndim != b.ndim:
        raise ValueError(f"AABB.{op}: dimension mismatch (a.ndim={a.ndim} vs b.ndim={b.ndim}).")


__all__ = ["AABB"]
