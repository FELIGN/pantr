"""Affine transformations for geometric objects.

Provides the :class:`AffineTransform` class, which represents an affine map
``T(x) = A @ x + b`` in *n*-dimensional space, together with factory methods
for common transformations (translation, rotation, scaling, mirroring, shear)
and composition operators.

Main exports:

- :class:`AffineTransform` — logically immutable affine-transformation object.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from pantr._backend import Backend, active_backend, available_backends

if TYPE_CHECKING:
    from typing import TypeAlias

    from pantr._pantr_cpp import AffineTransform as _CppAffine

    _AffineImpl: TypeAlias = "_AffineTransformPython | _CppAffine"
    """The implementation an :class:`AffineTransform` holds: the oracle, or the C++ map."""

import numpy as np
from numpy import typing as npt

__all__ = ["AffineTransform"]


class _AffineTransformPython:
    """The pure-Python affine map, kept as the port's oracle.

    This was the public :class:`AffineTransform` until 2026-08-28, when the map
    followed :class:`pantr.geometry.AABB` into the C++ core under the 2026-08-27
    amendment to ``design/cross_backend_types.md``. It survives as what the parity
    suite compares against and as what runs under ``PANTR_BACKEND=python``; both
    are temporary and go when the C++ core stops being optional.

    Not a second implementation of the public type: :class:`AffineTransform` is
    the only class a caller holds, and this one is reachable only through it.

    An affine transformation T(x) = A x + b in n-dimensional space.

    The transformation is defined by a square matrix ``A`` (the linear part)
    and a translation vector ``b``.  Instances are treated as immutable: every
    factory method and operator returns a new :class:`_AffineTransformPython`; no
    method mutates an existing instance.

    Attributes:
        _matrix (npt.NDArray[np.float64]): The ``(n, n)`` linear part.
        _translation (npt.NDArray[np.float64]): The ``(n,)`` translation.
    """

    _matrix: npt.NDArray[np.float64]
    _translation: npt.NDArray[np.float64]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        matrix: npt.ArrayLike,
        translation: npt.ArrayLike | None = None,
    ) -> None:
        """Create an affine transformation from a matrix and translation.

        Args:
            matrix (npt.ArrayLike): The ``(n, n)`` linear part of the
                transformation.  Must be a square 2-D array.
            translation (npt.ArrayLike | None): The ``(n,)`` translation
                vector.  If ``None``, defaults to the zero vector.

        Raises:
            ValueError: If *matrix* is not 2-D or not square.
            ValueError: If *translation* length does not match the matrix
                dimension.

        Note:
            Both *matrix* and *translation* are stored as C-contiguous,
            read-only ``float64`` arrays.
        """
        mat = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:  # noqa: PLR2004
            raise ValueError(f"matrix must be a square 2-D array, got shape {mat.shape}.")

        n = mat.shape[0]

        tvec: npt.NDArray[np.float64]
        if translation is None:
            tvec = np.zeros(n, dtype=np.float64)
        else:
            tvec = np.ascontiguousarray(np.asarray(translation, dtype=np.float64))
            if tvec.shape != (n,):
                raise ValueError(f"translation must have shape ({n},), got {tvec.shape}.")

        mat.flags.writeable = False
        tvec.flags.writeable = False
        self._matrix = mat
        self._translation = tvec

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Get the spatial dimension of the transformation.

        Returns:
            int: Dimension *n* of the transformation.
        """
        return int(self._matrix.shape[0])

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Get the linear part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n, n)`` matrix.
        """
        return self._matrix

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Get the translation (offset) part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n,)`` vector.
        """
        return self._translation

    @functools.cached_property
    def inverse(self) -> _AffineTransformPython:
        """Get the inverse transformation.

        Computed once and cached; subsequent accesses are free. Safe because
        neither the matrix nor the translation is ever modified after
        construction.

        Returns:
            _AffineTransformPython: The inverse such that ``T @ T.inverse`` is the
            identity.

        Raises:
            ValueError: If the matrix is singular.
        """
        try:
            inv_mat = np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Cannot invert a singular affine transformation.") from exc
        inv_trans = -inv_mat @ self._translation
        return _AffineTransformPython(inv_mat, inv_trans)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def identity(n: int) -> _AffineTransformPython:
        """Create the identity transformation in *n* dimensions.

        Args:
            n (int): Spatial dimension.

        Returns:
            _AffineTransformPython: The identity map.
        """
        return _AffineTransformPython(np.eye(n))

    @staticmethod
    def translation(offset: npt.ArrayLike) -> _AffineTransformPython:
        """Create a pure translation.

        Args:
            offset (npt.ArrayLike): Translation vector of length *n*.

        Returns:
            _AffineTransformPython: A translation by *offset*.
        """
        b = np.asarray(offset, dtype=np.float64).ravel()
        return _AffineTransformPython(np.eye(len(b)), b)

    @staticmethod
    def scaling(
        factors: float | npt.ArrayLike,
        *,
        center: npt.ArrayLike | None = None,
    ) -> _AffineTransformPython:
        """Create a scaling transformation.

        Args:
            factors (float | npt.ArrayLike): If a scalar, isotropic scaling
                is applied and *center* (or a separate call) determines the
                dimension.  If an array, anisotropic scaling along each axis.
            center (npt.ArrayLike | None): Optional center point. If given,
                the scaling is performed about this point rather than the
                origin.

        Returns:
            _AffineTransformPython: The scaling transformation.

        Raises:
            ValueError: If *factors* is a scalar and *center* is ``None``
                (dimension cannot be inferred).
            ValueError: If *factors* is a scalar and *center* is not a 1-D
                array-like.
            ValueError: If any factor is non-finite or zero (singular
                transform).
            ValueError: If *factors* is an array and *center* has the wrong
                shape.
        """
        f = np.asarray(factors, dtype=np.float64)
        if f.ndim == 0:
            # Scalar — need center to know dimension.
            if center is None:
                raise ValueError(
                    "An isotropic scaling factor requires a center or an "
                    "array of per-axis factors so the dimension can be "
                    "inferred."
                )
            fval = float(f)
            if not np.isfinite(fval):
                raise ValueError(f"scaling factors must be finite, got {fval!r}.")
            if fval == 0.0:
                raise ValueError(
                    f"scaling factors must be non-zero (singular transform), got {fval!r}."
                )
            c = np.asarray(center, dtype=np.float64)
            if c.ndim != 1:
                raise ValueError(f"center must be a 1-D array, got shape {c.shape}.")
            f = np.full(len(c), fval)
        else:
            f = f.ravel()
            if not np.all(np.isfinite(f)):
                raise ValueError(f"scaling factors must be finite, got {f!r}.")
            if np.any(f == 0.0):
                raise ValueError(
                    f"scaling factors must be non-zero (singular transform), got {f!r}."
                )

        mat = np.diag(f)
        return _with_optional_center(mat, center)

    @staticmethod
    def rotation_2d(
        angle: float,
        *,
        center: npt.ArrayLike | None = None,
    ) -> _AffineTransformPython:
        """Create a 2-D counter-clockwise rotation.

        Args:
            angle (float): Rotation angle in radians.
            center (npt.ArrayLike | None): Optional center of rotation.

        Returns:
            _AffineTransformPython: The 2-D rotation.

        Raises:
            ValueError: If *angle* is non-finite.
        """
        angle_f = float(angle)
        if not np.isfinite(angle_f):
            raise ValueError(f"angle must be finite, got {angle_f!r}.")
        c, s = np.cos(angle_f), np.sin(angle_f)
        mat = np.array([[c, -s], [s, c]], dtype=np.float64)
        return _with_optional_center(mat, center)

    @staticmethod
    def rotation_3d(
        angle: float,
        axis: int | npt.ArrayLike = 2,
        *,
        center: npt.ArrayLike | None = None,
    ) -> _AffineTransformPython:
        """Create a 3-D rotation via the Rodrigues formula.

        Args:
            angle (float): Rotation angle in radians.
            axis (int | npt.ArrayLike): Rotation axis. An ``int`` in
                ``{0, 1, 2}`` selects the corresponding coordinate axis
                (x, y, z).  An array-like of length 3 specifies an arbitrary
                axis (will be normalised internally).
            center (npt.ArrayLike | None): Optional center of rotation.

        Returns:
            _AffineTransformPython: The 3-D rotation.

        Raises:
            ValueError: If *angle* is non-finite.
            ValueError: If an integer axis is not in ``{0, 1, 2}``.
            ValueError: If a vector axis does not have shape ``(3,)``.
            ValueError: If a vector axis is zero or non-finite.
        """
        if isinstance(axis, int | np.integer):
            axis_int = int(axis)
            if axis_int not in (0, 1, 2):
                raise ValueError(f"Integer axis must be 0, 1, or 2, got {axis_int}.")
            u = np.zeros(3, dtype=np.float64)
            u[axis_int] = 1.0
        else:
            u = np.asarray(axis, dtype=np.float64).ravel()
            if u.shape != (3,):
                raise ValueError(f"Rotation axis must have shape (3,), got {u.shape}.")
            norm = np.linalg.norm(u)
            if norm == 0.0 or not np.isfinite(norm):
                raise ValueError(f"Rotation axis must be a finite non-zero vector, got {u!r}.")
            u = u / norm

        angle_f = float(angle)
        if not np.isfinite(angle_f):
            raise ValueError(f"angle must be finite, got {angle_f!r}.")
        # Rodrigues rotation matrix: R = I cos(t) + (1-cos(t)) u u^T + sin(t) [u]x
        c, s = np.cos(angle_f), np.sin(angle_f)
        ux, uy, uz = u
        K = np.array(
            [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]],
            dtype=np.float64,
        )
        mat = c * np.eye(3) + (1.0 - c) * np.outer(u, u) + s * K

        return _with_optional_center(mat, center)

    @staticmethod
    def mirror(
        normal: npt.ArrayLike,
        *,
        center: npt.ArrayLike | None = None,
    ) -> _AffineTransformPython:
        """Create a reflection (mirror) across a hyperplane.

        The hyperplane passes through the origin (or *center*) and has the
        given *normal* vector.  The Householder formula is used:
        ``A = I - 2 n nᵀ``.

        Args:
            normal (npt.ArrayLike): Normal vector of the mirror plane.  Will
                be normalised internally.
            center (npt.ArrayLike | None): Optional point on the mirror
                plane.

        Returns:
            _AffineTransformPython: The reflection.

        Raises:
            ValueError: If *normal* is zero or non-finite.
        """
        n = np.asarray(normal, dtype=np.float64).ravel()
        norm = np.linalg.norm(n)
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError(f"Mirror normal must be a finite non-zero vector, got {n!r}.")
        n = n / norm
        mat = np.eye(len(n)) - 2.0 * np.outer(n, n)
        return _with_optional_center(mat, center)

    @staticmethod
    def shear(
        dim: int,
        component: int,
        direction: int,
        factor: float,
    ) -> _AffineTransformPython:
        """Create a shear transformation.

        The resulting map adds ``factor * x[direction]`` to
        ``x[component]``, leaving all other components unchanged.

        Args:
            dim (int): Spatial dimension.
            component (int): The axis that is modified.
            direction (int): The axis whose value drives the shear.
            factor (float): Shear magnitude.

        Returns:
            _AffineTransformPython: The shear transformation.

        Raises:
            ValueError: If *component* equals *direction*.
            ValueError: If *component* or *direction* is out of range.
            ValueError: If *factor* is non-finite.
        """
        if component == direction:
            raise ValueError("component and direction must differ.")
        if not (0 <= component < dim):
            raise ValueError(f"component must be in [0, {dim}), got {component}.")
        if not (0 <= direction < dim):
            raise ValueError(f"direction must be in [0, {dim}), got {direction}.")
        factor_f = float(factor)
        if not np.isfinite(factor_f):
            raise ValueError(f"factor must be finite, got {factor_f!r}.")
        mat = np.eye(dim, dtype=np.float64)
        mat[component, direction] = factor_f
        return _AffineTransformPython(mat)

    # ------------------------------------------------------------------
    # Composition and application
    # ------------------------------------------------------------------

    def compose(self, other: _AffineTransformPython) -> _AffineTransformPython:
        """Compose this transformation with *other*.

        Returns the transformation ``self(other(x))``.

        Args:
            other (_AffineTransformPython): The inner transformation.

        Returns:
            _AffineTransformPython: The composed transformation.

        Raises:
            ValueError: If the dimensions do not match.
        """
        if self.dim != other.dim:
            raise ValueError(
                f"Cannot compose transforms of different dimensions ({self.dim} and {other.dim})."
            )
        new_mat = self._matrix @ other._matrix
        new_trans = self._matrix @ other._translation + self._translation
        return _AffineTransformPython(new_mat, new_trans)

    def __matmul__(self, other: object) -> _AffineTransformPython:
        """Compose via the ``@`` operator.

        Args:
            other (object): Must be an :class:`_AffineTransformPython`.

        Returns:
            _AffineTransformPython: The composed transformation (``self`` after
            ``other``).
        """
        if not isinstance(other, _AffineTransformPython):
            return NotImplemented
        return self.compose(other)

    def __call__(
        self,
        points: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Apply the transformation to a set of points.

        Args:
            points (npt.ArrayLike): Points with shape ``(..., n)``.

        Returns:
            npt.NDArray[np.float64]: Transformed points with the same shape.

        Raises:
            ValueError: If the last dimension of *points* does not match
                ``self.dim``.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != self.dim:
            raise ValueError(
                f"Points last dimension ({pts.shape[-1]}) must match "
                f"transform dimension ({self.dim})."
            )
        return np.asarray(pts @ self._matrix.T + self._translation, dtype=np.float64)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: Representation showing dimension and matrix/translation.
        """
        return (
            f"AffineTransform(dim={self.dim}, "
            f"matrix={self._matrix.tolist()}, "
            f"translation={self._translation.tolist()})"
        )


# ------------------------------------------------------------------
# Module-private helpers
# ------------------------------------------------------------------


def _apply_center(
    transform: _AffineTransformPython,
    center: npt.ArrayLike,
) -> _AffineTransformPython:
    """Conjugate *transform* by a translation to/from *center*.

    Computes ``translate(center) @ transform @ translate(-center)`` so that
    the linear part of *transform* is applied about *center* rather than the
    origin.

    Args:
        transform (_AffineTransformPython): A linear (or affine) transformation.
        center (npt.ArrayLike): The center point.

    Returns:
        _AffineTransformPython: The re-centred transformation.

    Raises:
        ValueError: If *center* does not have shape ``(transform.dim,)``.
    """
    c = np.asarray(center, dtype=np.float64).ravel()
    if c.shape != (transform.dim,):
        raise ValueError(
            f"center must have shape ({transform.dim},), got {np.asarray(center).shape}."
        )
    t_neg = _AffineTransformPython.translation(-c)
    t_pos = _AffineTransformPython.translation(c)
    return t_pos @ transform @ t_neg


def _with_optional_center(
    mat: npt.NDArray[np.float64],
    center: npt.ArrayLike | None,
) -> _AffineTransformPython:
    """Build an :class:`_AffineTransformPython` from ``mat``, re-centred if requested.

    Args:
        mat (npt.NDArray[np.float64]): The linear part of the transform.
        center (npt.ArrayLike | None): Optional center point; when given, the
            transform is conjugated about it via :func:`_apply_center`.

    Returns:
        _AffineTransformPython: The transform, about ``center`` when provided.
    """
    t = _AffineTransformPython(mat)
    if center is not None:
        t = _apply_center(t, center)
    return t


def _use_python() -> bool:
    """Whether the active backend selects the pure-Python oracle.

    The choice is per process rather than per instance, for the reason
    :class:`pantr.geometry.AABB` records: two maps built under different backends
    could otherwise meet in :meth:`AffineTransform.compose`, and reconciling them
    would mean converting one implementation into the other, which
    ``design/cross_backend_types.md`` forbids.

    Returns:
        bool: ``True`` under the Python backend.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if active_backend() is Backend.PYTHON:
        return True
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    return False


def _cpp_class() -> type[_CppAffine]:
    """The bound C++ map class.

    Split from :func:`_use_python` so that a caller past the branch has a single
    concrete type rather than a union, which is what lets the checker verify the
    factories instead of taking them on trust.

    Returns:
        type[_CppAffine]: The class exposed by the extension.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.AffineTransform


def _f64(value: npt.ArrayLike, *, ravel: bool) -> npt.NDArray[np.float64]:
    """Normalize an argument to the contiguous float64 array the binding needs.

    The oracle accepts anything array-like; the binding refuses a non-contiguous
    or wrongly-typed array outright. Normalizing here keeps ``PANTR_BACKEND`` from
    changing what the library accepts.

    ``ravel`` is explicit and per call site rather than inferred from a rank,
    because the oracle is not uniform: :meth:`AffineTransform.translation` rejects
    a ``(n, 1)`` vector where :meth:`mirror`, :meth:`scaling`, ``rotation_3d``'s
    axis and every ``center`` flatten it first. A helper that guessed reshaped a
    rejected argument before its shape reached the error message, so
    ``AffineTransform(np.zeros((2, 3, 4)))`` reported ``got shape (24,)``.

    Args:
        value (npt.ArrayLike): The caller's argument.
        ravel (bool): Whether the oracle flattens this argument before checking it.

    Returns:
        npt.NDArray[np.float64]: A contiguous ``float64`` array.
    """
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return np.ascontiguousarray(arr.ravel()) if ravel else arr


class AffineTransform:
    """An affine transformation ``T(x) = A x + b`` in ``n``-dimensional space.

    **This class is a wrapper.** Since the 2026-08-27 amendment to
    ``design/cross_backend_types.md`` the map itself is owned by the C++ core
    (``cpp/include/pantr/transform/affine.hpp``) and this class holds one.
    Ownership moved rather than being duplicated: there is one implementation of
    an affine map and one Python class in front of it.

    Under ``PANTR_BACKEND=python`` the thing held is
    :class:`_AffineTransformPython`, the port's oracle, which is temporary.

    Instances are immutable: every factory and operator returns a new map.

    Attributes:
        dim (int): The spatial dimension ``n``.
        matrix (npt.NDArray[np.float64]): Read-only ``(n, n)`` linear part.
        offset (npt.NDArray[np.float64]): Read-only ``(n,)`` translation.
    """

    __slots__ = ("__dict__", "_impl")

    _impl: _AffineImpl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

    def __init__(
        self,
        matrix: npt.ArrayLike,
        translation: npt.ArrayLike | None = None,
    ) -> None:
        """Create an affine transformation from a matrix and translation.

        Args:
            matrix (npt.ArrayLike): The ``(n, n)`` linear part. Must be square.
            translation (npt.ArrayLike | None): The ``(n,)`` translation. If
                ``None``, the zero vector.

        Raises:
            ValueError: If *matrix* is not 2-D or not square, or if *translation*
                does not match the matrix dimension.
        """
        if _use_python():
            object.__setattr__(self, "_impl", _AffineTransformPython(matrix, translation))
            return
        cls = _cpp_class()
        mat = _f64(matrix, ravel=False)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:  # noqa: PLR2004 -- a matrix is 2-D
            raise ValueError(f"matrix must be a square 2-D array, got shape {mat.shape}.")
        off = np.zeros(mat.shape[0]) if translation is None else _f64(translation, ravel=False)
        object.__setattr__(self, "_impl", cls(mat, off))

    @classmethod
    def _wrap(cls, impl: _AffineImpl) -> AffineTransform:
        """Wrap an implementation object that is already valid.

        Args:
            impl (_AffineImpl): The implementation to adopt.

        Returns:
            AffineTransform: A wrapper around it, with no re-validation.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "_impl", impl)
        return self

    def _peer(self, other: AffineTransform) -> Any:  # noqa: ANN401 -- see the Returns section
        """The other map's implementation, once it is known to match this one's.

        Args:
            other (AffineTransform): The right-hand map.

        Returns:
            Any: ``other``'s implementation. Untyped because the two are
            unrelated nominal types and this method's job is the check that makes
            the call safe.

        Raises:
            TypeError: If the two maps hold different implementations.
        """
        mine, theirs = type(self._impl), type(other._impl)
        if mine is not theirs:
            raise TypeError(
                f"AffineTransform: cannot combine maps from different backends "
                f"({mine.__name__} and {theirs.__name__}); the backend is chosen "
                f"per process, so this means one was built under a different one."
            )
        return other._impl

    @property
    def dim(self) -> int:
        """Get the spatial dimension of the transformation.

        Returns:
            int: Dimension ``n``.
        """
        return int(self._impl.dim)

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Get the linear part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n, n)`` matrix.
        """
        return self._impl.matrix

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Get the translation part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n,)`` vector.
        """
        return self._impl.offset

    @functools.cached_property
    def inverse(self) -> AffineTransform:
        """Get the inverse transformation.

        Cached here rather than in the implementation. The C++ type holds no
        mutable member on purpose, so there is nothing to reason about across
        threads on that side; the cache lives where it already did, and where it
        is unobservable.

        Returns:
            AffineTransform: The inverse.

        Raises:
            ValueError: If the matrix is singular.
        """
        impl = self._impl
        got = impl.inverse if isinstance(impl, _AffineTransformPython) else impl.inverse()
        return AffineTransform._wrap(got)

    def compose(self, other: AffineTransform) -> AffineTransform:
        """Compose this transformation with *other*, giving ``self(other(x))``.

        Args:
            other (AffineTransform): The inner transformation.

        Returns:
            AffineTransform: The composed transformation.

        Raises:
            ValueError: If the dimensions do not match.
        """
        return AffineTransform._wrap(self._impl.compose(self._peer(other)))

    def __matmul__(self, other: object) -> AffineTransform:
        """Compose via the ``@`` operator.

        Args:
            other (object): Must be an :class:`AffineTransform`.

        Returns:
            AffineTransform: The composed transformation.
        """
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return self.compose(other)

    def __call__(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the transformation to a set of points.

        Args:
            points (npt.ArrayLike): Points with shape ``(..., n)``.

        Returns:
            npt.NDArray[np.float64]: Transformed points, same shape.

        Raises:
            ValueError: If the last dimension does not match :attr:`dim`.
        """
        impl = self._impl
        if isinstance(impl, _AffineTransformPython):
            return impl(points)
        # The rank check runs on `asarray`, NOT on `ascontiguousarray`, and the
        # difference is the whole point: `ascontiguousarray` promotes a 0-d input
        # to shape (1,), so a guard written after it can never see rank 0. It was
        # written after it, and the consequence was that `identity(1)(5.0)`
        # silently returned `array([5.])` under the C++ backend while the oracle
        # raised. Indexing `shape[-1]` on a 0-d array raises `IndexError`, which
        # is exactly what the oracle does -- reproducing that is what a port owes
        # its oracle, even where the behaviour looks accidental. Whether the
        # oracle SHOULD raise `IndexError` there is a separate question about the
        # oracle.
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != self.dim:
            raise ValueError(
                f"Points last dimension ({pts.shape[-1]}) must match "
                f"transform dimension ({self.dim})."
            )
        flat = np.ascontiguousarray(pts).reshape(-1, self.dim)
        out = np.empty_like(flat)
        impl.apply(flat, out)
        return out.reshape(pts.shape)

    def __reduce__(self) -> tuple[type[AffineTransform], tuple[npt.NDArray[np.float64], ...]]:
        """Pickle by matrix and offset rather than by implementation.

        The C++ handle is not picklable and must not become part of the wire
        format: a pickle written with the C++ backend has to load under the
        Python one and the other way round, or the backend switch would silently
        become a data-format switch. This is also the route ``copy.deepcopy``
        takes, which is how a grid holding a reference map gets copied.

        The cached ``inverse`` is deliberately not carried: it is a memo over
        the two arrays below, so the reconstructed map recomputes it on demand.

        Returns:
            tuple: The class and the ``(matrix, offset)`` pair to rebuild it from.
        """
        return (type(self), (self.matrix, self.offset))

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Formatted here rather than by the implementation, so the two backends
        print identically.

        Returns:
            str: Representation showing dimension, matrix and translation.
        """
        return (
            f"AffineTransform(dim={self.dim}, "
            f"matrix={self.matrix.tolist()}, "
            f"translation={self.offset.tolist()})"
        )

    @staticmethod
    def identity(n: int) -> AffineTransform:
        """Create the identity transformation in ``n`` dimensions.

        Args:
            n (int): Spatial dimension.

        Returns:
            AffineTransform: The identity.
        """
        if _use_python():
            return AffineTransform._wrap(_AffineTransformPython.identity(n))
        return AffineTransform._wrap(_cpp_class().identity(n))

    @staticmethod
    def translation(offset: npt.ArrayLike) -> AffineTransform:
        """Create a pure translation.

        Args:
            offset (npt.ArrayLike): The ``(n,)`` translation vector.

        Returns:
            AffineTransform: The translation.
        """
        if _use_python():
            return AffineTransform._wrap(_AffineTransformPython.translation(offset))
        cls = _cpp_class()
        return AffineTransform._wrap(cls.translation(_f64(offset, ravel=False)))

    @staticmethod
    def scaling(
        factors: float | npt.ArrayLike,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a scaling transformation.

        Args:
            factors (float | npt.ArrayLike): A scalar (isotropic, requiring
                *center* to fix the dimension) or one factor per axis.
            center (npt.ArrayLike | None): Optional centre of scaling.

        Returns:
            AffineTransform: The scaling.

        Raises:
            ValueError: If a scalar factor is given without a centre, or if a
                factor is zero or non-finite.
        """
        if _use_python():
            return AffineTransform._wrap(_AffineTransformPython.scaling(factors, center=center))
        cls = _cpp_class()
        f = np.asarray(factors, dtype=np.float64)
        if f.ndim == 0:
            if center is None:
                raise ValueError(
                    "An isotropic scaling factor requires a center or an "
                    "array of per-axis factors so the dimension can be "
                    "inferred."
                )
            # The factor is checked BEFORE the centre, because that is the
            # oracle's order. Deferring the factor to C++ and checking the centre
            # here inverted it, so two simultaneously bad arguments produced
            # different messages on the two backends.
            fval = float(f)
            if not np.isfinite(fval):
                raise ValueError(f"scaling factors must be finite, got {fval!r}.")
            if fval == 0.0:
                raise ValueError(
                    f"scaling factors must be non-zero (singular transform), got {fval!r}."
                )
            c = np.asarray(center, dtype=np.float64)
            if c.ndim != 1:
                raise ValueError(f"center must be a 1-D array, got shape {c.shape}.")
            f = np.full(len(c), fval)
        return AffineTransform._centred(cls.scaling(_f64(f, ravel=True)), center)

    @staticmethod
    def rotation_2d(angle: float, *, center: npt.ArrayLike | None = None) -> AffineTransform:
        """Create a rotation of the plane.

        Args:
            angle (float): Rotation angle in radians.
            center (npt.ArrayLike | None): Optional centre of rotation.

        Returns:
            AffineTransform: The rotation.

        Raises:
            ValueError: If *angle* is non-finite.
        """
        if _use_python():
            return AffineTransform._wrap(_AffineTransformPython.rotation_2d(angle, center=center))
        cls = _cpp_class()
        return AffineTransform._centred(cls.rotation_2d(float(angle)), center)

    @staticmethod
    def rotation_3d(
        angle: float,
        axis: int | npt.ArrayLike = 2,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a 3-D rotation via the Rodrigues formula.

        Args:
            angle (float): Rotation angle in radians.
            axis (int | npt.ArrayLike): An ``int`` in ``{0, 1, 2}`` selecting a
                coordinate axis, or a length-3 vector (normalised internally).
            center (npt.ArrayLike | None): Optional centre of rotation.

        Returns:
            AffineTransform: The rotation.

        Raises:
            ValueError: If *angle* is non-finite, an integer axis is out of
                range, or a vector axis is the wrong shape, zero or non-finite.
        """
        if _use_python():
            return AffineTransform._wrap(
                _AffineTransformPython.rotation_3d(angle, axis, center=center)
            )
        cls = _cpp_class()
        # The integer-axis spelling is a Python convenience, not part of the
        # kernel seam: it is resolved to a vector here, exactly as the oracle
        # resolves it before touching any arithmetic.
        if isinstance(axis, int | np.integer):
            axis_int = int(axis)
            if axis_int not in (0, 1, 2):
                raise ValueError(f"Integer axis must be 0, 1, or 2, got {axis_int}.")
            vec = np.zeros(3)
            vec[axis_int] = 1.0
        else:
            vec = _f64(axis, ravel=True)
            # The axis is validated BEFORE the angle, matching the oracle. The C++
            # side checks the angle first, so leaving both to it inverted the order
            # whenever a caller got both wrong at once.
            if vec.shape != (3,):
                raise ValueError(f"Rotation axis must have shape (3,), got {vec.shape}.")
            norm = float(np.linalg.norm(vec))
            if norm == 0.0 or not np.isfinite(norm):
                raise ValueError(f"Rotation axis must be a finite non-zero vector, got {vec!r}.")
        return AffineTransform._centred(cls.rotation_3d(float(angle), vec), center)

    @staticmethod
    def mirror(normal: npt.ArrayLike, *, center: npt.ArrayLike | None = None) -> AffineTransform:
        """Create a reflection across a hyperplane.

        Args:
            normal (npt.ArrayLike): Normal of the mirror plane, normalised
                internally.
            center (npt.ArrayLike | None): Optional point on the plane.

        Returns:
            AffineTransform: The reflection.

        Raises:
            ValueError: If *normal* is zero or non-finite.
        """
        if _use_python():
            return AffineTransform._wrap(_AffineTransformPython.mirror(normal, center=center))
        cls = _cpp_class()
        return AffineTransform._centred(cls.mirror(_f64(normal, ravel=True)), center)

    @staticmethod
    def shear(dim: int, component: int, direction: int, factor: float) -> AffineTransform:
        """Create a shear that adds ``factor * x[direction]`` to ``x[component]``.

        Args:
            dim (int): Spatial dimension.
            component (int): The axis that is modified.
            direction (int): The axis whose value drives the shear.
            factor (float): Shear magnitude.

        Returns:
            AffineTransform: The shear.

        Raises:
            ValueError: If the two axes coincide, either is out of range, or
                *factor* is non-finite.
        """
        if _use_python():
            return AffineTransform._wrap(
                _AffineTransformPython.shear(dim, component, direction, factor)
            )
        cls = _cpp_class()
        if component == direction:
            raise ValueError("component and direction must differ.")
        for name, value in (("component", component), ("direction", direction)):
            if not 0 <= value < dim:
                raise ValueError(f"{name} must be in [0, {dim}), got {value}.")
        return AffineTransform._wrap(cls.shear(dim, component, direction, float(factor)))

    @staticmethod
    def _centred(impl: _CppAffine, center: npt.ArrayLike | None) -> AffineTransform:
        """Conjugate an implementation about ``center``, when one is given.

        Args:
            impl (_CppAffine): The C++ map to re-centre.
            center (npt.ArrayLike | None): The centre, or ``None``.

        Returns:
            AffineTransform: The map, about ``center`` when provided.

        Raises:
            ValueError: If ``center`` has the wrong length.
        """
        if center is None:
            return AffineTransform._wrap(impl)
        c = _f64(center, ravel=True)
        if c.shape != (impl.dim,):
            # The oracle ravels for the CHECK and reports the ORIGINAL shape, so
            # a (3, 1) centre is named as (3, 1) and not as (3,). Reporting the
            # ravelled shape is a smaller mistake than it looks: it tells the
            # caller their argument had a shape it never had.
            raise ValueError(
                f"center must have shape ({impl.dim},), got {np.asarray(center).shape}."
            )
        return AffineTransform._wrap(impl.about_center(c))
