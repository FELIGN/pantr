"""Bezier geometric object: the Bezier class.

Since the 2026-08-27 amendment to ``design/cross_backend_types.md`` the value
itself is owned by C++ (``cpp/include/pantr/bezier/bezier.hpp``) and
:class:`Bezier` is a wrapper holding one implementation of it. Ownership moves;
it is never duplicated. There are two implementations and they are not two
Béziers: :class:`_BezierPython` is the oracle the port is checked against, and
the C++ handle is the thing being checked. :func:`_impl_class` picks between
them, per process and per dtype.

The operations are unchanged and still live in the sibling ``_bezier_*`` modules
as free functions over a :class:`Bezier`. Only the *state* moved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, cast, overload

import numpy as np
from numpy import typing as npt

from .._backend import Backend, active_backend, available_backends
from .._transform_control_points import _apply_affine_to_control_points
from ._bezier_collapse import _collapse_along_axis
from ._bezier_compose import _compose_bezier
from ._bezier_degree import (
    _degree_elevate_bezier,
    _degree_reduce_bezier,
    _degree_reduction_l2_error,
    _minimize_degree_bezier,
)
from ._bezier_derivative import _derivative_bezier
from ._bezier_eval import _evaluate_bezier, _evaluate_bezier_deriv
from ._bezier_product import _multiply_bezier
from ._bezier_restrict import _restrict_bezier
from ._bezier_slice import _slice_bezier
from ._bezier_split import _split_bezier

if TYPE_CHECKING:
    from .._pantr_cpp import Bezier32 as _CppBezier32
    from .._pantr_cpp import Bezier64 as _CppBezier64
    from ..bspline import Bspline
    from ..quad import PointsLattice
    from ..transform import AffineTransform

    _Impl: TypeAlias = "_BezierPython | _CppBezier32 | _CppBezier64"
    """The implementation a :class:`Bezier` holds: the oracle, or a C++ handle.

    Type-checking only. The three are unrelated nominal types that happen to
    offer the same surface, which is the port's whole claim; naming the union
    here is what lets the checker verify it instead of taking it on trust.
    """

_ControlPoints: TypeAlias = "npt.NDArray[np.float32 | np.float64]"
"""A control-point array in one of the two storage formats a Bézier may use."""

_SUPPORTED_DTYPES: Final = (np.dtype(np.float32), np.dtype(np.float64))
"""The two storage formats a Bézier may hold, after integer input has been cast.

Narrower than what the oracle's constructor used to accept, which was anything
:func:`numpy.asarray` produced -- ``float16`` and ``bool`` included, both of which
build a Bézier no kernel below it can evaluate. The type annotations, the kernel
catalogue in :mod:`pantr.bezier._bezier_backend` and now the C++ registration all
say two formats, so the constructor says two as well. Rejecting is the wrapper's
job rather than the C++ type's: a dtype is a type-kind check, and nanobind has no
path that produces a :class:`TypeError`.
"""


class _BezierPython:
    """The Bézier value as the Python backend stores it: the port's oracle.

    Holds the control points and the rationality flag, and answers the four
    questions they determine. Everything the class used to do beyond that is on
    :class:`Bezier`, which is the only class a caller ever holds; this one is
    reachable only through it. When the C++ core stops being optional this class
    goes and :class:`Bezier` collapses onto the handle.

    **It aliases the caller's array, and that is a defect it keeps on purpose.**
    ``__init__`` stores what :func:`numpy.asarray` returns and
    :attr:`control_points` hands the same object back, so a caller can mutate a
    constructed Bézier through either end. That is the shape of FELIGN/pantr#338,
    the C++ implementation copies instead, and a port never edits its own oracle
    -- so the two backends differ here, deliberately, until the ticket that fixes
    this side lands.

    Attributes:
        _control_points (npt.NDArray[np.float32 | np.float64]): Control point
            array with shape ``(*degrees_plus_1, rank)``, where the last axis
            is the output rank (including the homogeneous weight for rational).
        _is_rational (bool): Whether the Bézier is rational (last control
            point coordinate is a homogeneous weight).
    """

    __slots__ = ("_control_points", "_is_rational")

    def __init__(
        self,
        control_points: npt.ArrayLike,
        is_rational: bool = False,
    ) -> None:
        """Initialize a Bézier from control points.

        Args:
            control_points (npt.ArrayLike): Control points. Shape must be at
                least 2D: ``(*degrees_plus_1, rank)``. A 1D input of shape
                ``(n,)`` is reshaped to ``(n, 1)`` (scalar field). Integer
                arrays are cast to ``float64``.
            is_rational (bool): Whether the Bézier is rational. Defaults to
                False.

        Raises:
            ValueError: If the control points have fewer than 1 entry in any
                parametric direction.
            ValueError: If the Bézier has rank smaller than 1.
        """
        cp = np.asarray(control_points)
        if np.issubdtype(cp.dtype, np.integer):
            cp = cp.astype(np.float64)
        if cp.ndim < 1:
            raise ValueError("Control points must be at least 1D.")
        if cp.ndim == 1:
            cp = cp[:, np.newaxis]

        if cp.ndim < 2:  # pragma: no cover - guarded by the reshape above  # noqa: PLR2004
            raise ValueError("Control points must be at least 2D after reshape.")

        for d in range(cp.ndim - 1):
            if cp.shape[d] < 1:
                raise ValueError(
                    f"Control points must have at least 1 entry in parametric "
                    f"direction {d}, got {cp.shape[d]}."
                )

        self._control_points: _ControlPoints = cp
        self._is_rational = is_rational

        if self.rank <= 0:
            raise ValueError(f"The Bézier must have at least rank one. Got rank {self.rank}.")

    def _replace_control_points(self, control_points: _ControlPoints) -> None:
        """Adopt an already-validated control-point array, in place.

        The one mutating operation on this class, and it exists only because
        ``reverse``, ``permute_directions`` and ``transform`` take an
        ``in_place`` flag that :class:`Bezier` ports faithfully. It re-validates
        nothing: every caller derives the array from this Bézier's own, so the
        shape either did not change or changed by a permutation.

        Args:
            control_points (npt.NDArray[np.float32 | np.float64]): The array to
                store.
        """
        self._control_points = control_points

    @property
    def dim(self) -> int:
        """Get the parametric dimension of the Bézier.

        Returns:
            int: Number of parametric dimensions.
        """
        return self._control_points.ndim - 1

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the polynomial degrees per parametric direction.

        Returns:
            tuple[int, ...]: Polynomial degree for each parametric dimension,
            computed as ``shape[d] - 1``.
        """
        return tuple(s - 1 for s in self._control_points.shape[:-1])

    @property
    def control_points(self) -> _ControlPoints:
        """Get the control points of the Bézier.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Control point array with
            shape ``(*degrees_plus_1, rank)``. The stored array itself, not a
            copy; see the class docstring.
        """
        return self._control_points

    @property
    def is_rational(self) -> bool:
        """Check whether the Bézier is rational.

        Returns:
            bool: True if the Bézier is rational (i.e., the last control point
            coordinate is a homogeneous weight), False otherwise.
        """
        return self._is_rational

    @property
    def rank(self) -> int:
        """Get the output rank of the Bézier.

        The rank is the number of value dimensions produced by the mapping.
        For a scalar field it is 1; for a 3D curve it is 3. For rational
        Béziers the weight coordinate is excluded.

        Returns:
            int: Output rank of the Bézier.
        """
        rk = int(self._control_points.shape[-1])
        return rk - 1 if self.is_rational else rk


def _stored_dtype(control_points: npt.ArrayLike) -> tuple[_ControlPoints, np.dtype[Any]]:
    """Normalize an array-like to the array a Bézier would store, and its dtype.

    The two steps are the oracle's own first two lines, lifted out because the
    wrapper has to know the dtype *before* it can choose an implementation: the
    C++ side registers one class per storage format, so the format is what picks
    the class.

    Integer input is cast to ``float64``, which the constructor documents.
    Everything else keeps its dtype, which is how ``float32`` survives.

    Args:
        control_points (npt.ArrayLike): The caller's argument.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], np.dtype[Any]]: The array
        and its dtype. The dtype is not yet known to be one a Bézier supports;
        :func:`_new_impl` is what rejects the rest.

    Raises:
        TypeError: If the argument cannot be read as an array at all.
    """
    cp = np.asarray(control_points)
    if np.issubdtype(cp.dtype, np.integer):
        cp = cp.astype(np.float64)
    return cp, cp.dtype


def _impl_class(dtype: np.dtype[Any]) -> type[_BezierPython] | type[_CppBezier32 | _CppBezier64]:
    """The implementation class the active backend and the dtype select.

    The backend is per process rather than per instance, deliberately, and for
    the reason ``pantr.geometry`` states: two Béziers built under different
    backends could otherwise meet in a binary operation, and reconciling them
    would mean converting one implementation into the other, which is the shape
    ``design/cross_backend_types.md`` forbids.

    The dtype, by contrast, *is* per instance. It has to be: ``float32`` is a
    supported storage format for a Bézier, unlike for :class:`pantr.geometry.AABB`,
    and the C++ side carries the format in the class name because the class of
    the handle is the only thing left to carry it.

    Args:
        dtype (np.dtype[Any]): The dtype the control points will be stored in.

    Returns:
        type[_BezierPython] | type[_CppBezier32 | _CppBezier64]: The oracle under
        the Python backend, and the C++ class for that storage format otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if active_backend() is Backend.PYTHON:
        return _BezierPython
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.Bezier32 if dtype == np.float32 else _pantr_cpp.Bezier64


def _new_impl(control_points: npt.ArrayLike, is_rational: bool) -> _Impl:
    """Build a Bézier value in whichever implementation the active backend selects.

    Args:
        control_points (npt.ArrayLike): Control points, in any of the forms
            :meth:`Bezier.__init__` accepts.
        is_rational (bool): Whether the last coordinate is a homogeneous weight.

    Returns:
        _Impl: The implementation object; a :class:`_BezierPython` or a C++ handle.

    Raises:
        TypeError: If the control points have a dtype no Bézier can store.
        ValueError: If the shape does not describe a Bézier.
        RuntimeError: If the C++ backend is requested and is not available.
    """
    cp, dtype = _stored_dtype(control_points)
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"Bezier control points must be float32, float64 or an integer type; got dtype {dtype}."
        )

    cls = _impl_class(dtype)
    if cls is _BezierPython:
        return _BezierPython(cp, is_rational)

    # `np.ascontiguousarray` promotes a 0-d array to shape `(1,)`, which would turn
    # the "must be at least 1D" rejection into a silently accepted degree-0 curve.
    # The same trap is recorded at `pantr.transform.AffineTransform.apply`. A 0-d
    # array is contiguous anyway, so skipping it costs nothing.
    contiguous = cp if cp.ndim == 0 else np.ascontiguousarray(cp)
    # Each C++ class accepts only its own storage format and refuses rather than
    # casts, so the array's dtype and the class have to agree. They do: the class
    # was chosen from that dtype one line above. That is a correlation between a
    # value and a type, which the checker cannot state, and the cast is what
    # stands in for it.
    return cls(cast("Any", contiguous), bool(is_rational))


class Bezier:
    """A parametric Bézier curve, surface, or volume defined by control points.

    Stores only control points and an ``is_rational`` flag. The polynomial
    degree in each parametric direction is inferred from the control point
    array shape: ``degree[d] = control_points.shape[d] - 1``.

    **This class is a wrapper.** Since the 2026-08-27 amendment to
    ``design/cross_backend_types.md`` the value is owned by C++
    (``cpp/include/pantr/bezier/bezier.hpp``) and this class holds one
    implementation of it, chosen by :func:`_impl_class`. The operations are still
    Python and still live in the sibling ``_bezier_*`` modules; only the state
    moved.

    One behaviour depends on which implementation is in hand, and it is the point
    of the port rather than an oversight: the C++ value **copies** the control
    points at construction and hands out a read-only view, so neither end aliases
    the caller's array. The Python oracle aliases at both ends, which is the
    defect FELIGN/pantr#338 names, and it is not fixed here because a port does
    not edit its own oracle.

    Attributes:
        _impl (_Impl): The implementation this wrapper holds; see
            :func:`_impl_class`.
    """

    __slots__ = ("_impl",)

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

    def __init__(
        self,
        control_points: npt.ArrayLike,
        is_rational: bool = False,
    ) -> None:
        """Initialize a Bézier from control points.

        Args:
            control_points (npt.ArrayLike): Control points. Shape must be at
                least 2D: ``(*degrees_plus_1, rank)``. A 1D input of shape
                ``(n,)`` is reshaped to ``(n, 1)`` (scalar field). Integer
                arrays are cast to ``float64``; ``float32`` and ``float64``
                are stored as they are.
            is_rational (bool): Whether the Bézier is rational. Defaults to
                False.

        Raises:
            TypeError: If the control points have a dtype no Bézier can store.
            ValueError: If the control points have fewer than 1 entry in any
                parametric direction.
            ValueError: If the Bézier has rank smaller than 1.
        """
        self._impl = _new_impl(control_points, is_rational)

    @classmethod
    def _wrap(cls, impl: _Impl) -> Bezier:
        """Wrap an implementation object that is already valid.

        Args:
            impl (_Impl): The implementation object to adopt.

        Returns:
            Bezier: A wrapper around ``impl``, with no re-validation.
        """
        self = object.__new__(cls)
        self._impl = impl
        return self

    def _mutate_control_points(self, rebuild: Callable[[_ControlPoints], _ControlPoints]) -> None:
        """Replace this Bézier's control points with ``rebuild``'s result, in place.

        The single place the two implementations differ in *kind* rather than in
        provenance, so the branch lives here once instead of in each of the three
        ``in_place=True`` methods.

        Under the Python backend ``rebuild`` is handed the stored array itself, so
        a helper writing into it mutates the Bézier exactly as it always has --
        ``id(bezier.control_points)`` included, which ``tests/test_transform.py``
        pins. Under the C++ backend the storage belongs to the C++ object and is
        read-only, so ``rebuild`` gets a writable copy and the *implementation* is
        replaced. Both leave this wrapper carrying the new value; only the array's
        identity differs, and only where the oracle's aliasing is what defined it.

        Rebuilding the implementation reads the *active* backend, so mutating a
        Bézier inside a :func:`~pantr._backend.use_backend` block that selects a
        different one would silently hand the caller back an object of the other
        implementation -- and, going from C++ to Python, silently drop the
        read-only guarantee on an array they still hold. Reconciling two
        implementations of one type by converting between them is the shape
        ``design/cross_backend_types.md`` forbids, so this refuses rather than
        converts, exactly as :meth:`pantr.geometry.AABB._peer` does for a binary
        operation. Immutable ported types cannot reach this; ``Bezier`` is the
        first with observable mutation.

        Args:
            rebuild (Callable[[NDArray], NDArray]): Given a writable control-point
                array, returns the new one. It may write into the array it is
                given and return it.

        Raises:
            TypeError: If the active backend no longer selects this Bézier's own
                implementation.
        """
        impl = self._impl
        mine = type(impl)
        theirs = _impl_class(impl.control_points.dtype)
        if mine is not theirs:
            raise TypeError(
                f"Bezier: cannot mutate a Bezier built under a different backend "
                f"({mine.__name__} against the active {theirs.__name__}); the "
                f"backend is chosen per process, so this means the active one "
                f"changed after this Bezier was built."
            )
        if isinstance(impl, _BezierPython):
            impl._replace_control_points(rebuild(impl.control_points))
        else:
            self._impl = _new_impl(rebuild(np.array(impl.control_points)), impl.is_rational)

    def __reduce__(self) -> tuple[type[Bezier], tuple[_ControlPoints, bool]]:
        """Pickle by the constructor's arguments rather than by implementation.

        The C++ handle is not picklable and must not become part of the wire
        format: a pickle written under the C++ backend has to load under the
        Python one and the other way round, or the backend switch would silently
        become a data-format switch.

        Returns:
            tuple: The class, and the control points and rationality flag to
            rebuild it from.
        """
        control_points = self.control_points
        if not control_points.flags.writeable:
            # The C++ backend hands out a read-only view and pickle preserves that
            # flag (measured). Reconstructing under the Python backend would then
            # store a read-only array, where `reverse(in_place=True)` raises. The
            # copy is what keeps one backend's storage decision out of the wire
            # format.
            control_points = np.array(control_points)
        return (type(self), (control_points, self.is_rational))

    @property
    def dim(self) -> int:
        """Get the parametric dimension of the Bézier.

        Returns:
            int: Number of parametric dimensions.
        """
        return int(self._impl.dim)

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the polynomial degrees per parametric direction.

        Returns:
            tuple[int, ...]: Polynomial degree for each parametric dimension,
            computed as ``shape[d] - 1``.
        """
        return self._impl.degree

    @property
    def control_points(self) -> _ControlPoints:
        """Get the control points of the Bézier.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Control point array with
            shape ``(*degrees_plus_1, rank)``. Under the C++ backend this is a
            **read-only view** of the Bézier's own storage: writing through it
            raises, and it stays valid after the Bézier is dropped. Under the
            Python backend it is the stored array itself, writable, which is the
            aliasing defect the class docstring names.
        """
        return self._impl.control_points

    @property
    def is_rational(self) -> bool:
        """Check whether the Bézier is rational.

        Returns:
            bool: True if the Bézier is rational (i.e., the last control point
            coordinate is a homogeneous weight), False otherwise.
        """
        return bool(self._impl.is_rational)

    @property
    def rank(self) -> int:
        """Get the output rank of the Bézier.

        The rank is the number of value dimensions produced by the mapping.
        For a scalar field it is 1; for a 3D curve it is 3. For rational
        Béziers the weight coordinate is excluded.

        Returns:
            int: Output rank of the Bézier.
        """
        return int(self._impl.rank)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the Bézier.

        Read off the control points rather than delegated: the C++ handle carries
        its storage format in its class name, not as a numpy dtype it could hand
        back.

        Returns:
            npt.DTypeLike: The numpy dtype (float32 or float64) of the control
            point array.
        """
        return self.control_points.dtype

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate the Bézier at the given parametric points.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate. For 1D Bézier, must be
                a 1D array of shape ``(n_pts,)``. For multi-dimensional Bézier,
                must be a 2D array of shape ``(n_pts, dim)`` or a
                :class:`~pantr.quad.PointsLattice`.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array. Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Bézier values at the given
            points.

        Raises:
            ValueError: If the points dtype does not match the Bézier dtype,
                or if ``out`` has incorrect shape or dtype.
        """
        return _evaluate_bezier(self, pts, out)

    def evaluate_derivatives(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate a specific partial derivative of the Bézier.

        Computes the single partial derivative specified by ``orders``,
        where ``orders[d]`` is the derivative order in parametric direction
        ``d``. For rational Bézier the generalised quotient rule is applied.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate.
            orders (int | Sequence[int]): Derivative order(s). A single
                ``int`` is broadcast to all ``self.dim`` directions. A sequence
                must contain one non-negative integer per parametric direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional
                pre-allocated output array. Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Mixed partial derivative
            values.

        Raises:
            ValueError: If ``len(orders) != self.dim``, if any order is
                negative, or if the points dtype does not match the Bézier dtype.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [2.0]]))  # f(t) = 2t
            >>> np.allclose(f.evaluate_derivatives(np.array([0.3, 0.7]), 1), [2.0, 2.0])
            True
            >>> cp = np.zeros((2, 2, 1))
            >>> cp[1, 1, 0] = 1.0  # f(u, v) = u * v
            >>> surf = Bezier(cp)
            >>> np.allclose(surf.evaluate_derivatives(np.array([[0.3, 0.4]]), [1, 1]), [1.0])
            True
        """
        orders_seq: Sequence[int] = [orders] * self.dim if isinstance(orders, int) else orders
        return _evaluate_bezier_deriv(self, pts, orders_seq, out)

    # ------------------------------------------------------------------
    # Derivative (returns new Bezier)
    # ------------------------------------------------------------------

    def derivative(self, direction: int = 0, *, keep_degree: bool = False) -> Bezier:
        """Return a Bézier representing the first derivative in the given direction.

        Computes the hodograph: a new Bézier whose value at every parametric
        point equals the partial derivative of this Bézier with respect to
        parametric direction ``direction``.

        For non-rational Bézier of degree ``p`` in direction ``d``, the
        result has degree ``p - 1`` (or ``p`` when ``keep_degree=True``).

        For rational Bézier, the quotient rule is applied, producing a
        rational Bézier of degree ``2p`` in direction ``d`` (or the original
        degree when ``keep_degree=True``).

        Args:
            direction (int): Parametric direction for differentiation.
                Must be in ``[0, dim)``. Defaults to 0.
            keep_degree (bool): If ``True``, the result preserves the same
                degree as the original Bézier by fusing derivative and degree
                elevation. This is useful, for instance, when computing
                derivatives of rational polynomials (in the numerator).
                Defaults to ``False``.

        Returns:
            Bezier: A new Bézier representing the derivative.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If the degree in the given direction is 0.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [0.0], [1.0]]))  # Bernstein form of t**2
            >>> fp = f.derivative()
            >>> fp.degree
            (1,)
            >>> np.allclose(fp.evaluate(np.array([0.3])), [0.6])
            True
            >>> f.derivative(keep_degree=True).degree
            (2,)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        if self.degree[direction] < 1:
            raise ValueError("Derivative of a degree-0 Bézier is not defined.")
        return _derivative_bezier(self, direction, keep_degree=keep_degree)

    # ------------------------------------------------------------------
    # Degree elevation
    # ------------------------------------------------------------------

    def elevate_degree(self, degree_increments: int | Sequence[int]) -> Bezier:
        """Elevate the polynomial degree of the Bézier.

        Creates a new Bézier that represents the same mapping as the original
        but with higher polynomial degree.

        Args:
            degree_increments (int | Sequence[int]): Number of degrees to
                increase. If an integer, the same increment is applied to all
                parametric directions. If a sequence, must have length equal
                to ``self.dim``.

        Returns:
            Bezier: A new Bézier with elevated degrees.

        Raises:
            ValueError: If any degree increment is negative.
            ValueError: If all degree increments are zero.
            ValueError: If the number of increments does not match the dimension.
        """
        if isinstance(degree_increments, int):
            increments = (degree_increments,) * self.dim
        else:
            increments = tuple(degree_increments)

        if len(increments) != self.dim:
            raise ValueError(
                f"Number of degree increments ({len(increments)}) "
                f"must match dimension ({self.dim})."
            )

        if any(inc < 0 for inc in increments):
            raise ValueError("Degree increments must be non-negative.")

        if all(inc == 0 for inc in increments):
            raise ValueError("At least one degree increment must be positive.")

        return _degree_elevate_bezier(self, increments)

    # ------------------------------------------------------------------
    # Degree reduction
    # ------------------------------------------------------------------

    def reduce_degree(self, degree_decrements: int | Sequence[int]) -> Bezier:
        r"""Reduce the polynomial degree of the Bézier, interpolating the endpoints.

        Creates a new Bézier whose degree is lower by the requested amount in
        each parametric direction.  Among the polynomials of the lower degree
        that reproduce this one at the ends of the parametric domain, the result
        is the one closest in :math:`L^2([0, 1]^{\dim})`; the endpoint values are
        reproduced exactly, bit for bit.

        Unlike :meth:`elevate_degree`, this operation is **not exact** in
        general: the result is an approximation of the original mapping.  Use
        :meth:`degree_reduction_error` to find out how close it is before
        committing to it.

        The endpoint conditions cost accuracy in the interior — dropping them
        would lower the :math:`L^2` error by a factor between 1.1 (degree 16) and
        4.5 (reduction to a straight line) — and buy an approximation that joins
        its neighbours exactly, which is what makes the B-spline case work.  A
        reduction to degree 0 cannot honour two conditions with one coefficient
        and returns the plain :math:`L^2` projection, the mean of the control
        points.

        Args:
            degree_decrements (int | Sequence[int]): Number of degrees to
                reduce. If an integer, the same decrement is applied to all
                parametric directions. If a sequence, must have length equal
                to ``self.dim``.

        Returns:
            Bezier: A new Bézier with reduced degrees.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.
        """
        return _degree_reduce_bezier(self, self._checked_decrements(degree_decrements))

    def degree_reduction_error(self, degree_decrements: int | Sequence[int]) -> float:
        r"""Compute the :math:`L^2` error that :meth:`reduce_degree` would introduce.

        The value is exact rather than an estimate: it is
        :math:`\lVert f - g \rVert_{L^2}` for the reduction *g* this Bézier would
        produce, obtained from the Bernstein mass matrix rather than by sampling.
        Rank components are combined in the Euclidean sense, and for a rational
        Bézier the norm is taken over the homogeneous coefficients, not over the
        projected mapping.

        The convex-hull bound on the coefficient residual is deliberately not
        offered instead: measured against the true supremum of the error it runs
        from 1.3 to 1600 times too large, the ratio growing with degree, so it is
        of little use as a stopping criterion.

        Args:
            degree_decrements (int | Sequence[int]): Number of degrees to
                reduce, as in :meth:`reduce_degree`.

        Returns:
            float: The :math:`L^2` norm of the error, in the units of the
            control points.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.

        Example:
            >>> import numpy as np
            >>> from pantr.bezier import Bezier
            >>> bezier = Bezier(np.array([[0.0], [1.0], [0.0], [1.0]]))
            >>> round(bezier.degree_reduction_error(1), 6)
            0.138013
        """
        decrements = self._checked_decrements(degree_decrements)
        return _degree_reduction_l2_error(self, decrements)

    def _checked_decrements(self, degree_decrements: int | Sequence[int]) -> tuple[int, ...]:
        """Normalise and validate a degree-decrement argument.

        Args:
            degree_decrements (int | Sequence[int]): Decrement for every
                direction, or one per direction.

        Returns:
            tuple[int, ...]: One decrement per parametric direction.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.
        """
        if isinstance(degree_decrements, int):
            decrements = (degree_decrements,) * self.dim
        else:
            decrements = tuple(degree_decrements)

        if len(decrements) != self.dim:
            raise ValueError(
                f"Number of degree decrements ({len(decrements)}) "
                f"must match dimension ({self.dim})."
            )

        if any(dec < 0 for dec in decrements):
            raise ValueError("Degree decrements must be non-negative.")

        if all(dec == 0 for dec in decrements):
            raise ValueError("At least one degree decrement must be positive.")

        for d, dec in enumerate(decrements):
            if dec > self.degree[d]:
                raise ValueError(
                    f"Degree decrement ({dec}) in direction {d} exceeds "
                    f"current degree ({self.degree[d]})."
                )

        return decrements

    def minimize_degree(self, tol: float | None = None) -> Bezier:
        """Find the lowest degree that preserves accuracy within tolerance.

        Iterates over each parametric direction and repeatedly tries to
        reduce the degree by 1.  A reduction is accepted when the
        round-trip (reduce then elevate) relative L2 error stays below
        ``tol``.  For vector-valued Bézier, all rank components are
        checked simultaneously.

        For a **rational** Bézier that error is measured on the projected
        mapping, not on the homogeneous control coefficients, so ``tol`` is a
        budget on the geometry the caller sees at any coordinate scale.  Two
        consequences: it is then a quadrature estimate rather than an exact
        value, accurate to better than 3% of the true relative deviation over
        degrees 3 to 20 and weight ratios up to 100; and a reduction is refused
        outright when a weight is not strictly positive on the domain, since the
        mapping then has a pole and no projected deviation is defined.

        Args:
            tol (float | None): Relative tolerance for accepting a degree
                reduction.  If *None*, uses a default based on machine
                epsilon (``1e3 * eps``).

        Returns:
            Bezier: A new Bézier with the lowest degree that preserves
            accuracy.  If no reduction is possible, returns a copy.

        Example:
            >>> import numpy as np
            >>> b = Bezier(np.array([3.0, 3.0, 3.0, 1.0]).reshape(4, 1))
            >>> b.degree
            (3,)
            >>> b_min = b.minimize_degree()
        """
        return _minimize_degree_bezier(self, tol)

    # ------------------------------------------------------------------
    # Multiply
    # ------------------------------------------------------------------

    def multiply(self, other: Bezier) -> Bezier:
        """Return the pointwise product of this Bézier and another.

        Given Bézier ``self`` and ``other`` over the same parametric domain
        ``[0, 1]^dim``, returns a new Bézier ``h`` such that
        ``h(t) = self(t) * other(t)``. The result has degree ``p_d + q_d``
        per direction.

        That degree represents the product **exactly**: no approximation is made
        in choosing it. The control points are not exact, and this docstring used
        to say they were. Each is a binomial-weighted sum evaluated in floating
        point, so each carries that sum's roundings; the size of them is measured
        in ``tests/parity/test_bezier_product.py``.

        Args:
            other (Bezier): The second Bézier operand. Must have the same
                dimension, dtype, and rank as ``self``.

        Returns:
            Bezier: A new Bézier representing ``self * other``.

        Raises:
            ValueError: If the operands have different dimensions, dtypes,
                or ranks.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [1.0]]))  # f(t) = t
            >>> g = Bezier(np.array([[0.0], [1.0]]))  # g(t) = t
            >>> h = f.multiply(g)
            >>> h.degree
            (2,)
            >>> np.allclose(h.evaluate(np.array([0.5])), [0.25])
            True
            >>> np.allclose((f * g).evaluate(np.array([0.5])), [0.25])  # same via __mul__
            True
        """
        return _multiply_bezier(self, other)

    __mul__ = multiply

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self, inner: Bezier) -> Bezier:
        """Compose this Bézier with another: ``result(t) = self(inner(t))``.

        Composes two non-rational Bézier objects. The result is a new Bézier with
        parametric dimension equal to ``inner.dim``, rank equal to ``self.rank``,
        and degree ``sum(self.degree) * inner.degree[s]`` in each direction ``s``.

        That degree represents the composition **exactly**; the control points do
        not, and this docstring used to call the composition itself exact. They
        are built from repeated Bernstein products in floating point -- one per
        power of the inner map and one per tensor term -- and carry every one of
        those roundings.

        Args:
            inner (Bezier): The inner Bézier (reparametrization). Must be
                non-rational and satisfy ``inner.rank == self.dim``.

        Returns:
            Bezier: A new Bézier representing ``self(inner(t))``.

        Raises:
            TypeError: If either Bézier is rational.
            ValueError: If ``inner.rank != self.dim``.
            ValueError: If the operands have different dtypes.

        Example:
            >>> f = Bezier(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))
            >>> g = Bezier(np.array([[0.2], [0.8]]))
            >>> h = f.compose(g)
        """
        return _compose_bezier(self, inner)

    # ------------------------------------------------------------------
    # Reverse and permute
    # ------------------------------------------------------------------

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[False] = ...) -> Bezier: ...

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[True]) -> None: ...

    def reverse(self, direction: int = 0, *, in_place: bool = False) -> Bezier | None:
        """Reverse the orientation of one parametric direction.

        Flips the control points along the given parametric axis so that the
        mapping is reparametrised in the opposite sense along that direction.

        Args:
            direction (int): Parametric direction to reverse. Must be in
                ``[0, dim)``. Defaults to 0.
            in_place (bool): If ``True``, modify this Bézier in place and
                return ``None``. If ``False`` (default), return a new Bézier.

        Returns:
            Bezier | None: The reversed Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [1.0]]))  # f(t) = t
            >>> rev = f.reverse()
            >>> np.allclose(rev.evaluate(np.array([0.3])), [0.7])  # rev(t) = 1 - t
            True
            >>> f.reverse(in_place=True)  # mutates in place, returns None
            >>> np.allclose(f.evaluate(np.array([0.3])), [0.7])
            True
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")

        from .._control_points_utils import _reverse_control_points  # noqa: PLC0415

        if in_place:
            self._mutate_control_points(
                lambda cp: _reverse_control_points(cp, direction, in_place=True)
            )
            return None
        new_cp = _reverse_control_points(self.control_points, direction, in_place=False)
        return Bezier(new_cp, is_rational=self.is_rational)

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[False] = ...
    ) -> Bezier: ...

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[True]
    ) -> None: ...

    def permute_directions(
        self, permutation: Sequence[int], *, in_place: bool = False
    ) -> Bezier | None:
        """Reorder the parametric directions according to a permutation.

        Given a permutation ``[i_0, i_1, …]``, the new direction ``k`` is
        the old direction ``permutation[k]``. For example, ``[1, 2, 0]`` on
        a 3D volume maps old direction 1 → new 0, old 2 → new 1, old 0 → new 2.

        Args:
            permutation (Sequence[int]): A permutation of ``range(dim)``.
            in_place (bool): If ``True``, modify this Bézier in place and
                return ``None``. If ``False`` (default), return a new Bézier.

        Returns:
            Bezier | None: The permuted Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``permutation`` is not a valid permutation of
                ``range(dim)``.

        Example:
            >>> import numpy as np
            >>> surf = Bezier(np.zeros((2, 3, 1)))
            >>> surf.degree
            (1, 2)
            >>> surf.permute_directions([1, 0]).degree  # swap u <-> v
            (2, 1)
        """
        perm = list(permutation)
        if sorted(perm) != list(range(self.dim)):
            raise ValueError(f"permutation must be a permutation of range({self.dim}), got {perm}.")

        from .._control_points_utils import _permute_control_points  # noqa: PLC0415

        if in_place:
            self._mutate_control_points(lambda cp: _permute_control_points(cp, perm, self.dim))
            return None
        new_cp = _permute_control_points(self.control_points, perm, self.dim)
        return Bezier(new_cp, is_rational=self.is_rational)

    # ------------------------------------------------------------------
    # Affine transformation
    # ------------------------------------------------------------------

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[False] = ...) -> Bezier: ...

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[True]) -> None: ...

    def transform(
        self,
        affine: AffineTransform,
        *,
        in_place: bool = False,
    ) -> Bezier | None:
        """Apply an affine transformation to the control points.

        For non-rational Bézier, every control point ``P`` is mapped to
        ``A @ P + b``.  For rational Bézier the weighted homogeneous
        coordinates are updated so that the projected geometry undergoes the
        same affine map while the weights are preserved.

        Args:
            affine (~pantr.transform.AffineTransform): The affine
                transformation to apply.
            in_place (bool): If ``True``, the control points are modified in
                place and ``None`` is returned.  If ``False`` (default), a
                new :class:`Bezier` is returned.

        Returns:
            Bezier | None: The transformed Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If the transform dimension does not match the
                geometric rank of the Bézier.

        Example:
            >>> import numpy as np
            >>> from pantr.transform import AffineTransform
            >>> curve = Bezier(np.array([[0.0, 0.0], [1.0, 1.0]]))  # rank 2
            >>> T = AffineTransform.translation([1.0, 2.0])
            >>> shifted = curve.transform(T)
            >>> np.allclose(shifted.control_points, [[1.0, 2.0], [2.0, 3.0]])
            True
        """
        if in_place:
            self._mutate_control_points(
                lambda cp: _apply_affine_to_control_points(
                    cp, self.is_rational, affine.matrix, affine.offset, in_place=True
                )
            )
            return None
        new_cp = _apply_affine_to_control_points(
            self.control_points,
            self.is_rational,
            affine.matrix,
            affine.offset,
            in_place=False,
        )
        return Bezier(new_cp, is_rational=self.is_rational)

    # ------------------------------------------------------------------
    # Restrict
    # ------------------------------------------------------------------

    def restrict(
        self,
        bounds: tuple[float, float] | Sequence[tuple[float, float] | None],
    ) -> Bezier:
        """Return a Bézier restricted to a sub-region of ``[0, 1]^dim``.

        Extracts the portion of the Bézier defined on the given sub-interval
        and reparametrizes the result back to ``[0, 1]^dim``.  The returned
        Bézier has the same degree but different control points that encode
        the restricted mapping.

        Uses two de Casteljau passes per direction for direct Bernstein
        coefficient computation without B-spline conversion. The order
        of the passes is chosen for numerical stability.

        Args:
            bounds (tuple[float, float] | Sequence[tuple[float, float] | None]):
                For a 1D Bézier, a ``(lower, upper)`` tuple within ``[0, 1]``.
                For multi-dimensional Bézier, a sequence of length ``dim``
                where each element is a ``(lower, upper)`` tuple for that
                direction, or ``None`` to keep the full ``[0, 1]`` range.
                At least one direction must have non-``None`` bounds that
                restrict the domain.

        Returns:
            Bezier: New Bézier on ``[0, 1]^dim`` representing the restriction.

        Raises:
            ValueError: If the sequence length does not match ``dim``.
            ValueError: If all directions are ``None`` or match the full domain.
            ValueError: If any bound lies outside ``[0, 1]``.
            ValueError: If ``lower >= upper`` in any direction.
        """
        if self.dim == 1:
            bounds_per_dim: list[tuple[float, float] | None] = [
                bounds  # type: ignore[list-item]
            ]
        else:
            seq = list(bounds)  # type: ignore[arg-type,unused-ignore]
            if len(seq) != self.dim:
                raise ValueError(
                    f"bounds sequence length ({len(seq)}) must match dim ({self.dim})."
                )
            bounds_per_dim = seq  # type: ignore[assignment]

        # Validate bounds.
        for i, b in enumerate(bounds_per_dim):
            if b is None:
                continue
            lower, upper = b
            if lower >= upper:
                raise ValueError(
                    f"Lower bound ({lower}) must be strictly less than upper bound ({upper}) "
                    f"in direction {i}."
                )
            if lower < 0.0 or upper > 1.0:
                raise ValueError(
                    f"Bounds ({lower}, {upper}) must lie within [0, 1] in direction {i}."
                )

        return _restrict_bezier(self, bounds_per_dim)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(self, direction: int, value: float) -> tuple[Bezier, Bezier]:
        """Split the Bézier into two at a parameter value in one direction.

        Uses the de Casteljau algorithm to subdivide the Bézier into a
        left half (representing the original on ``[0, value]``) and a right
        half (representing the original on ``[value, 1]``), both
        reparametrized to ``[0, 1]``.

        Args:
            direction (int): Parametric direction along which to split.
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to split.  Must lie
                strictly inside ``(0, 1)``.

        Returns:
            tuple[Bezier, Bezier]: A pair ``(left, right)`` of Béziers on
            ``[0, 1]^dim``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is not strictly inside ``(0, 1)``.

        Example:
            >>> import numpy as np
            >>> curve = Bezier(np.array([[0.0], [1.0]]))  # f(t) = t
            >>> left, right = curve.split(0, 0.5)
            >>> np.allclose(left.evaluate(np.array([1.0])), [0.5])
            True
            >>> np.allclose(right.evaluate(np.array([0.0])), [0.5])
            True
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"value must be strictly inside (0, 1), got {value}.")

        return _split_bezier(self, direction, value)

    # ------------------------------------------------------------------
    # Slice and boundary
    # ------------------------------------------------------------------

    def slice(self, axis: int, value: float) -> Bezier | npt.NDArray[np.float32 | np.float64]:
        """Slice the Bézier by fixing one parametric direction at a given value.

        Reduces the parametric dimension by one using the de Casteljau
        algorithm on the control points.  A surface becomes a curve, a curve
        becomes a point (returned as a NumPy array).

        At the boundary values ``0`` and ``1`` the result is obtained in
        O(1) by direct control point lookup.

        Args:
            axis (int): Parametric direction to fix (0-indexed).
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to slice.  Must lie
                within ``[0, 1]``.

        Returns:
            Bezier | npt.NDArray[np.float32 | np.float64]:
            A Bézier with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.
            Rational Béziers preserve the rational structure when ``dim >= 2``;
            for ``dim == 1`` the result is projected to physical coordinates.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is outside ``[0, 1]``.

        Example:
            >>> import numpy as np
            >>> cp = np.zeros((2, 2, 1))
            >>> cp[1, 1, 0] = 1.0  # f(u, v) = u * v
            >>> surf = Bezier(cp)
            >>> curve = surf.slice(1, 0.5)  # fix v=0.5: curve(u) = 0.5u
            >>> curve.dim
            1
            >>> pt = surf.slice(1, 0.5).slice(0, 0.2)  # surface -> curve -> point
            >>> np.allclose(pt, [0.1])
            True
        """
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"value must be in [0, 1], got {value}.")

        return _slice_bezier(self, axis, value)

    def boundary(self, axis: int, side: int) -> Bezier | npt.NDArray[np.float32 | np.float64]:
        """Extract the boundary of the Bézier along one parametric direction.

        Returns the restriction of the Bézier to one end of the ``[0, 1]``
        domain in the given direction.

        Args:
            axis (int): Parametric direction (0-indexed).
                Must be in ``[0, dim)``.
            side (int): Which end of the domain: ``0`` for the start,
                ``1`` for the end.

        Returns:
            Bezier | npt.NDArray[np.float32 | np.float64]:
            A Bézier with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``side`` is not 0 or 1.

        Example:
            >>> import numpy as np
            >>> cp = np.zeros((2, 2, 1))
            >>> cp[1, 1, 0] = 1.0  # f(u, v) = u * v
            >>> surf = Bezier(cp)
            >>> right_edge = surf.boundary(0, 1)  # u = 1: f(1, v) = v
            >>> np.allclose(right_edge.evaluate(np.array([0.7])), [0.7])
            True
        """
        if side not in (0, 1):
            raise ValueError(f"side must be 0 or 1, got {side}.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        value = 0.0 if side == 0 else 1.0
        return self.slice(axis, value)

    # ------------------------------------------------------------------
    # Collapse along axis
    # ------------------------------------------------------------------

    def collapse_along_axis(
        self,
        axis: int,
        values: npt.ArrayLike,
    ) -> Bezier:
        """Collapse to a univariate Bézier along one parametric direction.

        Fixes all parametric directions except ``axis`` at the given parameter
        values, producing a 1D Bézier whose control points are the Bernstein
        coefficients along ``axis``.  This is a tensor contraction: for each
        collapsed direction, the Bernstein basis is evaluated at the given
        value and contracted with the control point array.

        Args:
            axis (int): Parametric direction to keep (0-indexed).
                Must be in ``[0, dim)``.
            values (npt.ArrayLike): Parameter values for all directions
                except ``axis``.  Must have length ``dim - 1`` with all
                values in ``[0, 1]``.  ``values[i]`` corresponds to
                direction ``i`` for ``i < axis``, and direction ``i + 1``
                for ``i >= axis``.

        Returns:
            Bezier: A 1D Bézier with degree ``self.degree[axis]`` and
            the same rank and rationality as the input.

        Raises:
            ValueError: If ``dim < 2`` (nothing to collapse).
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``values`` does not have length ``dim - 1``.
            ValueError: If any value is outside ``[0, 1]``.

        Example:
            >>> import numpy as np
            >>> cp = np.zeros((2, 2, 2, 1))
            >>> cp[1, 1, 1, 0] = 1.0  # f(u, v, w) = u * v * w
            >>> vol = Bezier(cp)
            >>> curve = vol.collapse_along_axis(1, [0.3, 0.7])  # fix u=0.3, w=0.7
            >>> curve.degree
            (1,)
            >>> np.allclose(curve.evaluate(np.array([0.5])), [0.3 * 0.5 * 0.7])
            True
        """
        if self.dim < 2:  # noqa: PLR2004
            raise ValueError("collapse_along_axis requires dim >= 2.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        return _collapse_along_axis(self, axis, values)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_bspline(self, *, copy: bool = True) -> Bspline:
        """Convert to an equivalent B-spline with Bézier knot vectors.

        Creates a :class:`~pantr.bspline.Bspline` with open knot vectors
        ``[0]*(p+1) + [1]*(p+1)`` in each parametric direction.

        Args:
            copy (bool): If ``True`` (default), the control points are
                deep-copied into the new B-spline. If ``False``, the
                B-spline shares the same underlying control point array --
                which, under the C++ backend, is the Bézier's own storage and
                is read-only, so the B-spline is then read-only too.

        Returns:
            ~pantr.bspline.Bspline: Equivalent B-spline representation.
        """
        from ..bspline import Bspline as BsplineCls  # noqa: PLC0415
        from ..bspline import BsplineSpace, BsplineSpace1D  # noqa: PLC0415

        dtype = self.dtype
        spaces: list[BsplineSpace1D] = []
        for p in self.degree:
            knots = np.zeros(2 * (p + 1), dtype=dtype)
            knots[p + 1 :] = 1.0
            spaces.append(BsplineSpace1D(knots, p))

        cp = self.control_points.copy() if copy else self.control_points
        return BsplineCls(BsplineSpace(spaces), cp, self.is_rational)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        color: str | None = None,
        show_control_polygon: bool = False,
        **plotter_kwargs: Any,  # noqa: ANN401
    ) -> object:
        """Quick interactive visualization of this Bézier (requires pyvista).

        For finer control, use ``pantr.viz.Scene`` directly.

        Args:
            color: Surface color.
            show_control_polygon: Render control polygon (points and wireframe).
            **plotter_kwargs: Additional keyword arguments for ``pv.Plotter()``.

        Returns:
            object: The pyvista ``Plotter`` after showing.

        Raises:
            ImportError: If pyvista is not installed.
        """
        from ..viz import plot as _plot  # noqa: PLC0415

        return _plot(
            self,
            color=color,
            show_control_polygon=show_control_polygon,
            **plotter_kwargs,
        )


def create_from_bspline(bspline: Bspline, *, copy: bool = True) -> Bezier:
    """Create a Bézier from a B-spline with Bézier-like knot vectors.

    Validates that the B-spline has Bézier-like knots (open knots with
    ``num_basis == degree + 1`` in each direction) and extracts the
    control points.

    Args:
        bspline (~pantr.bspline.Bspline): A B-spline with Bézier-like
            knot structure.
        copy (bool): If ``True`` (default), the control points are
            deep-copied into the new Bézier. If ``False``, the Bézier
            shares the same underlying control point array -- **under the
            Python backend only**. The C++ value owns its storage and copies
            at construction, so there ``copy=False`` saves nothing and shares
            nothing.

    Returns:
        Bezier: The equivalent Bézier.

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
    """
    if not bspline.space.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots. Cannot convert to Bézier.")
    cp = bspline.control_points.copy() if copy else bspline.control_points
    return Bezier(cp, is_rational=bspline.is_rational)
