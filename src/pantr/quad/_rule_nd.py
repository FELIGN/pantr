"""Multi-dimensional quadrature rules on the unit cube (:class:`QuadratureRule`).

Since the 2026-08-27 amendment to ``design/cross_backend_types.md`` the rule
itself is owned by the C++ core (``cpp/include/pantr/quad/rule.hpp``) and
:class:`QuadratureRule` here is a wrapper holding one. Ownership moved rather
than being duplicated: there is one implementation of a rule and one Python class
in front of it. Under ``PANTR_BACKEND=python`` the thing held is
:class:`_QuadratureRulePython`, the port's oracle, which is temporary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

from pantr._backend import Backend, active_backend, available_backends

from ._rules import get_gauss_legendre_1d

if TYPE_CHECKING:
    from pantr._pantr_cpp import QuadratureRule as _CppQuadratureRule

    _Impl: TypeAlias = "_QuadratureRulePython | _CppQuadratureRule"
    """The implementation a :class:`QuadratureRule` holds; see :func:`_use_python`.

    Type-checking only. The two are unrelated nominal types that happen to offer
    the same surface, which is the port's whole claim; naming the union here is
    what lets the checker verify it instead of taking it on trust.
    """

_POINTS_NDIM = 2
"""Rank of the ``points`` argument: ``(num_points, ndim)``."""


def _validate_ranks(pts: npt.NDArray[np.float64], wts: npt.NDArray[np.float64]) -> None:
    """Reject a points or weights array of the wrong rank.

    Shared by the oracle and the wrapper's C++ branch rather than left to the C++
    type, and that is forced rather than chosen: nanobind's typed signature
    refuses a wrong-rank array with a ``TypeError``, and
    ``cpp/include/pantr/core/error.hpp`` records that no C++ exception reaches
    Python as one. The oracle raises ``ValueError`` here, so the check has to
    happen before the call.

    Args:
        pts (npt.NDArray[np.float64]): The points array.
        wts (npt.NDArray[np.float64]): The weights array.

    Raises:
        ValueError: If ``pts`` is not 2D or ``wts`` is not 1D.
    """
    if pts.ndim != _POINTS_NDIM:
        raise ValueError(f"points must be 2D (num_points, ndim); got shape {pts.shape}.")
    if wts.ndim != 1:
        raise ValueError(f"weights must be 1D (num_points,); got shape {wts.shape}.")


class _QuadratureRulePython:
    """The pure-Python quadrature rule, kept as the port's oracle.

    This was the public :class:`QuadratureRule` until the amendment named in the
    module docstring. It survives for two reasons and both are temporary: it is
    what the parity suite compares the C++ rule against, and it is what runs
    under ``PANTR_BACKEND=python``, which is how the package still imports in a
    tree with no compiled extension.

    **It is not a second implementation of the public type.**
    :class:`QuadratureRule` is the only class a caller ever holds; this one is
    reachable only through it. When the C++ core stops being optional, this class
    goes and :class:`QuadratureRule` collapses into a plain wrapper.

    Immutable quadrature rule on the unit cube ``[0, 1]^ndim``.

    Bundles quadrature points and weights as the *reference* rule that
    :func:`pantr.grid.cell_quadrature` affinely maps onto each cell of a grid.
    Points lie in the closed unit cube; the factory-built rules
    (:func:`tensor_product_quadrature`, :func:`gauss_legendre_quadrature`) have
    weights summing to ``1``, the measure of the unit cube, so the rule
    integrates the constant ``1`` to within the rounding of that sum and not
    exactly: a ``ndim``-fold product of rounded 1D weights, summed over
    ``num_points`` terms, cannot land on ``1.0`` bitwise. Measured over Gauss-
    Legendre rules of 1 to 40 points per direction, the largest ``|sum - 1|`` is
    2 ulp in 1D, 3 ulp in 2D and 4 ulp in 3D. The stored arrays are read-only.
    """

    __slots__ = ("_ndim", "_num_points", "_points", "_weights")

    def __init__(self, points: npt.ArrayLike, weights: npt.ArrayLike) -> None:
        """Build and validate a quadrature rule on the unit cube.

        Args:
            points (npt.ArrayLike): ``(num_points, ndim)`` array-like; every
                coordinate must lie in ``[0, 1]``.
            weights (npt.ArrayLike): ``(num_points,)`` array-like of weights.

        Raises:
            ValueError: If ``points`` is not 2D, ``weights`` is not 1D, their
                lengths disagree, either is empty, any value is non-finite, or any
                point lies outside ``[0, 1]``.
        """
        pts = np.array(points, dtype=np.float64)
        wts = np.array(weights, dtype=np.float64)
        _validate_ranks(pts, wts)
        if pts.shape[0] == 0 or pts.shape[1] == 0:
            raise ValueError(f"points must be non-empty; got shape {pts.shape}.")
        if wts.shape[0] != pts.shape[0]:
            raise ValueError(
                f"weights length {wts.shape[0]} must match the number of points {pts.shape[0]}."
            )
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must contain only finite values.")
        if not np.all(np.isfinite(wts)):
            raise ValueError("weights must contain only finite values.")
        if np.any(pts < 0.0) or np.any(pts > 1.0):
            raise ValueError("points must lie in the unit cube [0, 1]^ndim.")
        pts = np.ascontiguousarray(pts)
        wts = np.ascontiguousarray(wts)
        pts.flags.writeable = False
        wts.flags.writeable = False
        self._points = pts
        self._weights = wts
        self._ndim = int(pts.shape[1])
        self._num_points = int(pts.shape[0])

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the rule.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def num_points(self) -> int:
        """Get the number of quadrature points.

        Returns:
            int: Point count (``>= 1``).
        """
        return self._num_points

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Get the quadrature points on the unit cube.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points, ndim)`` array in
            ``[0, 1]^ndim``.
        """
        return self._points

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Get the quadrature weights.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points,)`` array.
        """
        return self._weights

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Returns:
            str: ``"QuadratureRule(ndim=..., num_points=...)"``.
        """
        return f"QuadratureRule(ndim={self._ndim}, num_points={self._num_points})"


def _use_python() -> bool:
    """Report whether the active backend selects the pure-Python oracle.

    The choice is per process rather than per instance, deliberately: two rules
    built under different backends could otherwise meet, and reconciling them
    would mean converting one implementation into the other -- the shape
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


def _cpp_class() -> type[_CppQuadratureRule]:
    """The bound C++ rule class.

    Split from :func:`_use_python` so that a caller past the branch has a single
    concrete type rather than a union, which is what lets the checker verify the
    factories instead of taking them on trust.

    Returns:
        type[_CppQuadratureRule]: The class exposed by the extension.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.QuadratureRule


def _f64(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Normalize an argument to the contiguous float64 array the binding needs.

    The oracle accepts anything array-like; the binding refuses a non-contiguous
    or wrongly-typed array outright. Normalizing here keeps ``PANTR_BACKEND`` from
    changing what the library accepts.

    Args:
        value (npt.ArrayLike): The caller's argument.

    Returns:
        npt.NDArray[np.float64]: A contiguous ``float64`` array.
    """
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


class QuadratureRule:
    """Immutable quadrature rule on the unit cube ``[0, 1]^ndim``.

    Bundles quadrature points and weights as the *reference* rule that
    :func:`pantr.grid.cell_quadrature` affinely maps onto each cell of a grid.
    Points lie in the closed unit cube; the factory-built rules
    (:func:`tensor_product_quadrature`, :func:`gauss_legendre_quadrature`) have
    weights summing to ``1``, the measure of the unit cube, so the rule
    integrates the constant ``1`` to within the rounding of that sum and not
    exactly: a ``ndim``-fold product of rounded 1D weights, summed over
    ``num_points`` terms, cannot land on ``1.0`` bitwise. Measured over Gauss-
    Legendre rules of 1 to 40 points per direction, the largest ``|sum - 1|`` is
    2 ulp in 1D, 3 ulp in 2D and 4 ulp in 3D. The stored arrays are read-only.

    **This class is a wrapper.** The rule itself is owned by the C++ core
    (``cpp/include/pantr/quad/rule.hpp``) and this class holds one; see the module
    docstring. Under ``PANTR_BACKEND=python`` it holds
    :class:`_QuadratureRulePython` instead.

    Attributes:
        ndim (int): Number of axes (``>= 1``).
        num_points (int): Number of quadrature points (``>= 1``).
        points (npt.NDArray[np.float64]): Read-only ``(num_points, ndim)`` array.
        weights (npt.NDArray[np.float64]): Read-only ``(num_points,)`` array.
    """

    __slots__ = ("_impl", "_points", "_weights")

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_use_python`."""

    _points: npt.NDArray[np.float64] | None
    """The points array once read, or ``None``; see :attr:`points`."""

    _weights: npt.NDArray[np.float64] | None
    """The weights array once read, or ``None``; see :attr:`points`."""

    def __init__(self, points: npt.ArrayLike, weights: npt.ArrayLike) -> None:
        """Build and validate a quadrature rule on the unit cube.

        Args:
            points (npt.ArrayLike): ``(num_points, ndim)`` array-like; every
                coordinate must lie in ``[0, 1]``.
            weights (npt.ArrayLike): ``(num_points,)`` array-like of weights.

        Raises:
            ValueError: If ``points`` is not 2D, ``weights`` is not 1D, their
                lengths disagree, either is empty, any value is non-finite, or any
                point lies outside ``[0, 1]``.
        """
        if _use_python():
            self._adopt(_QuadratureRulePython(points, weights))
            return
        pts = _f64(points)
        wts = _f64(weights)
        _validate_ranks(pts, wts)
        self._adopt(_cpp_class()(pts, wts))

    def _adopt(self, impl: _Impl) -> None:
        """Hold an implementation object, with an empty array cache.

        Args:
            impl (_Impl): The implementation to hold.
        """
        self._impl = impl
        self._points = None
        self._weights = None

    @classmethod
    def _wrap(cls, impl: _Impl) -> QuadratureRule:
        """Wrap an implementation object that is already valid.

        Args:
            impl (_Impl): The implementation object to adopt.

        Returns:
            QuadratureRule: A wrapper around ``impl``, with no re-validation.
        """
        self = object.__new__(cls)
        self._adopt(impl)
        return self

    def __reduce__(self) -> tuple[type[QuadratureRule], tuple[npt.NDArray[np.float64], ...]]:
        """Pickle by points and weights rather than by implementation.

        The C++ handle is not picklable and must not become part of the wire
        format: a pickle written with the C++ backend has to load under the Python
        one and the other way round, or the backend switch would silently become a
        data-format switch. ``pantr.quad.__init__`` rebinds ``__module__`` so the
        pickle names the public path, and ``pantr.mpi`` sends these across
        collective calls, so this is load-bearing rather than a convenience.

        Returns:
            tuple: The class and the ``(points, weights)`` pair to rebuild it from.
        """
        return (type(self), (self.points, self.weights))

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the rule.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return int(self._impl.ndim)

    @property
    def num_points(self) -> int:
        """Get the number of quadrature points.

        Returns:
            int: Point count (``>= 1``).
        """
        return int(self._impl.num_points)

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Get the quadrature points on the unit cube.

        Cached on first read. The C++ binding hands out a fresh copy on every
        access -- a view could outlive the rule, which is immutable precisely so
        that nobody has to reason about when its storage changes -- and a rule can
        carry tens of thousands of points, so an uncached property would copy the
        whole table on every ``rule.points[i]``. Caching also restores the
        oracle's own semantics, where the array is the same object every time.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points, ndim)`` array in
            ``[0, 1]^ndim``.
        """
        if self._points is None:
            self._points = self._impl.points
        return self._points

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Get the quadrature weights.

        Cached on first read; see :attr:`points`.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points,)`` array.
        """
        if self._weights is None:
            self._weights = self._impl.weights
        return self._weights

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Formatted here rather than by the implementation, so that the two backends
        print identically.

        Returns:
            str: ``"QuadratureRule(ndim=..., num_points=...)"``.
        """
        return f"QuadratureRule(ndim={self.ndim}, num_points={self.num_points})"


def _coerce_axis_rules(
    rules: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Coerce the per-axis rules and reject the ranks C++ cannot reject.

    Unpacking the pair is a Python-shaped operation and the rank check cannot
    produce a ``ValueError`` from C++, so both stay here; see
    :func:`_validate_ranks` and ``cpp/bindings/quad_types.cpp``.

    Args:
        rules (Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]): One
            ``(nodes, weights)`` pair per axis.

    Returns:
        list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]: The same
        pairs as contiguous ``float64`` arrays.

    Raises:
        ValueError: If any axis's nodes or weights are not 1D.
    """
    coerced: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for d, pair in enumerate(rules):
        nodes_d = _f64(pair[0])
        weights_d = _f64(pair[1])
        if nodes_d.ndim != 1 or weights_d.ndim != 1:
            raise ValueError(
                f"tensor_product_quadrature: axis {d} nodes and weights must be 1D; "
                f"got shapes {nodes_d.shape} and {weights_d.shape}."
            )
        coerced.append((nodes_d, weights_d))
    return coerced


def _tensor_product_quadrature_python(
    rules: Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
) -> _QuadratureRulePython:
    """Build the tensor product of per-axis 1D rules, in pure Python.

    The port's oracle for :func:`tensor_product_quadrature`; see
    :class:`_QuadratureRulePython`. Ranks are already checked by
    :func:`_coerce_axis_rules`.

    Args:
        rules (Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]):
            One ``(nodes, weights)`` pair per axis, each 1D and ``float64``.

    Returns:
        _QuadratureRulePython: The tensor-product rule on ``[0, 1]^ndim``.

    Raises:
        ValueError: If ``rules`` is empty, any axis pair is empty or of mismatched
            length, or (via :class:`_QuadratureRulePython`) any node lies outside
            ``[0, 1]``.
    """
    if len(rules) == 0:
        raise ValueError("tensor_product_quadrature: rules must have at least one axis.")
    nodes_per_axis: list[npt.NDArray[np.float64]] = []
    weights_per_axis: list[npt.NDArray[np.float64]] = []
    for d, (nodes_d, weights_d) in enumerate(rules):
        if nodes_d.shape[0] == 0 or nodes_d.shape != weights_d.shape:
            raise ValueError(
                f"tensor_product_quadrature: axis {d} needs matching non-empty "
                f"(nodes, weights); got shapes {nodes_d.shape} and {weights_d.shape}."
            )
        nodes_per_axis.append(nodes_d)
        weights_per_axis.append(weights_d)
    node_mesh = np.meshgrid(*nodes_per_axis, indexing="ij")
    weight_mesh = np.meshgrid(*weights_per_axis, indexing="ij")
    points = np.stack([m.ravel() for m in node_mesh], axis=1)
    weights = np.prod(np.stack([w.ravel() for w in weight_mesh], axis=0), axis=0)
    return _QuadratureRulePython(points, weights)


def tensor_product_quadrature(
    rules: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
) -> QuadratureRule:
    """Build a tensor-product :class:`QuadratureRule` from per-axis 1D rules.

    Each axis contributes a 1D rule ``(nodes, weights)`` on ``[0, 1]``; the
    d-dimensional rule is their tensor product. Points are enumerated in
    row-major (C) order -- the last axis varies fastest, matching
    :class:`pantr.grid.TensorProductGrid` cell ids -- and each weight is the
    product of the corresponding per-axis weights.

    Args:
        rules (Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]): One
            ``(nodes, weights)`` pair per axis. Within a pair, ``nodes`` and
            ``weights`` must be 1D of equal, non-zero length, with nodes in
            ``[0, 1]``.

    Returns:
        QuadratureRule: The tensor-product rule on ``[0, 1]^ndim`` with
        ``ndim == len(rules)`` and ``num_points`` the product of the per-axis
        point counts.

    Raises:
        ValueError: If ``rules`` is empty, or any axis pair is not a matching
            pair of non-empty 1D arrays, or (via :class:`QuadratureRule`) any
            node lies outside ``[0, 1]``.
    """
    coerced = _coerce_axis_rules(rules)
    if _use_python():
        return QuadratureRule._wrap(_tensor_product_quadrature_python(coerced))
    if len(coerced) == 0:
        # Refused here as well as in C++, because nanobind's `std::vector` caster
        # accepts an empty list happily and would call the factory with a span
        # this side had already decided was illegal. The C++ factory rejects it
        # too, with the same message, for a caller that has no Python.
        raise ValueError("tensor_product_quadrature: rules must have at least one axis.")
    return QuadratureRule._wrap(_cpp_class().tensor_product(coerced))


def _normalize_npts(ndim: int, npts: int | Sequence[int]) -> tuple[int, ...]:
    """Broadcast the point count over the axes and check the Python-only contract.

    Both checks here belong to the ``(ndim, npts)`` calling convention, which
    exists in Python and not in C++: ``QuadratureRule::gauss_legendre`` takes one
    span and reads the dimension off it, so there is no second argument for the
    two to disagree about, and a negative ``ndim`` cannot be recovered from the
    broadcast tuple in order to be reported. ``cpp/include/pantr/core/error.hpp``'s
    rule still holds for everything else: the per-entry range check lives in C++.

    Args:
        ndim (int): Number of axes (``>= 1``).
        npts (int | Sequence[int]): Points per axis, scalar or per-axis.

    Returns:
        tuple[int, ...]: The per-axis counts, of length ``ndim``.

    Raises:
        ValueError: If ``ndim < 1`` or ``npts`` is a sequence of the wrong length.
    """
    if ndim < 1:
        raise ValueError(f"gauss_legendre_quadrature: ndim must be >= 1; got {ndim}.")
    npts_tuple = (int(npts),) * ndim if isinstance(npts, int) else tuple(int(n) for n in npts)
    if len(npts_tuple) != ndim:
        raise ValueError(
            f"gauss_legendre_quadrature: npts must be a scalar or a length-{ndim} sequence; "
            f"got length {len(npts_tuple)}."
        )
    return npts_tuple


def _gauss_legendre_quadrature_python(npts: tuple[int, ...]) -> _QuadratureRulePython:
    """Build the tensor-product Gauss-Legendre rule, in pure Python.

    The port's oracle for :func:`gauss_legendre_quadrature`. The 1D rules come
    from :func:`pantr.quad.get_gauss_legendre_1d`, which maps them onto ``[0, 1]``
    on this side; ``design/backend_parity.md`` Rule 1 is why that map matters.

    Args:
        npts (tuple[int, ...]): Points per axis, already broadcast to ``ndim``.

    Returns:
        _QuadratureRulePython: The tensor-product Gauss-Legendre rule.

    Raises:
        ValueError: If any count is ``< 1``.
    """
    if any(n < 1 for n in npts):
        raise ValueError(f"gauss_legendre_quadrature: every npts entry must be >= 1; got {npts!r}.")
    rules = [get_gauss_legendre_1d(n, dtype=np.float64) for n in npts]
    return _tensor_product_quadrature_python(_coerce_axis_rules(rules))


def gauss_legendre_quadrature(ndim: int, npts: int | Sequence[int]) -> QuadratureRule:
    """Build a tensor-product Gauss-Legendre :class:`QuadratureRule` on the unit cube.

    Args:
        ndim (int): Number of axes (``>= 1``).
        npts (int | Sequence[int]): Points per axis. A scalar is broadcast to
            every axis; a length-``ndim`` sequence gives per-axis counts. Each
            count must be ``>= 1``.

    Returns:
        QuadratureRule: The tensor product of per-axis ``npts``-point
        Gauss-Legendre rules. Exact for tensor-product polynomials of per-axis
        degree ``2 * npts - 1``; weights sum to ``1``.

    Raises:
        ValueError: If ``ndim < 1``, ``npts`` is a sequence of the wrong length,
            or any count is ``< 1``.

    References:
        Nodes and weights follow the classical Gauss-Legendre construction
        :cite:p:`golub1969gauss`.
    """
    npts_tuple = _normalize_npts(ndim, npts)
    if _use_python():
        return QuadratureRule._wrap(_gauss_legendre_quadrature_python(npts_tuple))
    return QuadratureRule._wrap(_cpp_class().gauss_legendre(list(npts_tuple)))
