"""Quadrature kernels and the quadrature rule type of the compiled extension.

The kernels are bound by ``cpp/bindings/quad.cpp`` and :class:`QuadratureRule`
by ``cpp/bindings/quad_types.cpp``; both land in the one extension module, so
one stub module covers them. See ``__init__.pyi`` for what this package promises
and who has to keep it.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

def gauss_legendre_symmetric(
    n: int,
    *,
    out_nodes: npt.NDArray[np.float64],
    out_weights: npt.NDArray[np.float64],
) -> None:
    """Compute the ``n``-point Gauss-Legendre rule on ``[-1, 1]`` in float64.

    The **C++ half of Layer 2**, same division of labour as
    :func:`tabulate_cardinal_bspline_1d`. Unlike that function this one has a
    single overload rather than two: the kernel behind it is ``double``-only
    by design (instantiating Newton's method at ``float`` was measured to give
    1.46e-3 relative weight error at ``n = 200``), and the caller narrows
    afterwards.

    What is checked, and where:

    * dtype (float64 only), rank, C-contiguity, device and writability of both
      arrays, by nanobind's typed signature, before the body runs;
    * that ``n`` is a non-negative integer, by that same signature -- the C++
      parameter is ``unsigned``;
    * that ``n >= 1``, that ``n`` fits a C ``int``, and that both arrays have
      length exactly ``n``, in the function body;
    * that ``out_nodes`` and ``out_weights`` are passed by keyword, by that same
      signature. Both outputs share a type, rank, dtype and contiguity, so
      nothing about a positional call would catch them transposed -- keyword-only
      makes that a ``TypeError`` instead of a silently wrong answer.

    Call :func:`pantr.quad.get_gauss_legendre_1d` for the ordinary path, which
    maps the result onto ``[0, 1]``, narrows to the requested dtype, and
    allocates the arrays for you.

    Args:
        n (int): Number of points. Must be at least 1 and must fit a C ``int``.
        out_nodes (npt.NDArray[np.float64]): Output array of length ``n``, 1D,
            C-contiguous and writable. Written in full, ascending in
            ``(-1, 1)``.
        out_weights (npt.NDArray[np.float64]): Output array of length ``n``,
            matching ``out_nodes`` in shape. Written in full; the weights sum
            to 2 up to the rounding of that sum.

    Raises:
        TypeError: If either array has a dtype other than float64, the wrong
            rank, or is not C-contiguous. A non-contiguous array is **refused
            rather than converted**, for the reason given on
            :func:`tabulate_cardinal_bspline_1d`. Also raised if ``out_nodes``
            or ``out_weights`` is passed positionally.
        ValueError: If ``n`` is less than 1, if ``n`` is too large to fit a C
            ``int``, or if either array's length is not exactly ``n``.
    """

def lambert_w_principal(x: float) -> float:
    """Solve ``w e^w = x`` on the principal branch, by Halley's method.

    This is **the C++ half of Layer 2**, and here the two backends
    deliberately diverge. ``pantr.quad._rules._lambert_w_principal`` is a
    Layer 3 kernel: it documents the precondition on ``x`` and never checks
    it, because its only caller, ``_generate_tanh_sinh``, never passes an
    argument below about 1.885 (the value at ``n = 2``). This binding has no
    such guarantee on its caller -- it is a public attribute of a public
    module -- so it validates. That is why the two sides differ only here and
    nowhere else in this stub.

    What is checked, and where:

    * that ``x`` is convertible to a C ``double``, by nanobind's typed
      signature, before the body runs;
    * that ``x`` is finite and at least about 1.61, in the function body.
      Below that the branch-free asymptotic starting guess lands on the wrong
      branch and no number of Halley steps recovers: measured on the Python
      original, an argument corresponding to a decay factor of 0.50 leaves
      2.8e4 units of roundoff, 0.40 leaves 2.4e16, and below about 0.31 the
      inner logarithm is not real.

    Call :func:`pantr.quad._rules._lambert_w_principal` for the Python
    original; there is no public path onto this function, since its sole
    caller in both backends is the tanh-sinh rule generator.

    Args:
        x (float): The argument. Must be finite and at least about 1.61.

    Returns:
        float: ``W(x)``, within about one unit of roundoff of ``W``.

    Raises:
        ValueError: If ``x`` is not finite or is below about 1.61.
    """

def generate_tanh_sinh(
    n: int,
    min_gap: float,
    *,
    out_nodes: npt.NDArray[np.float64],
    out_weights: npt.NDArray[np.float64],
) -> int:
    """Generate the tanh-sinh (double-exponential) rule on ``[-1, 1]``.

    The **C++ half of Layer 2**. This is the one seam in ``pantr.quad`` whose
    signature is not shaped like the others: **the return value is the
    effective node count and may be below ``n``**. Generation stops at the
    last node whose distance to the endpoint given by ``min_gap`` is still
    representable, so ``out_nodes`` and ``out_weights`` are sized for the
    worst case and only the first ``m`` entries -- ``m`` being the return
    value -- are written.

    What is checked, and where:

    * dtype (float64 only), rank, C-contiguity, device and writability of both
      arrays, by nanobind's typed signature, before the body runs;
    * that ``n`` is a non-negative integer, by that same signature;
    * that ``n >= 1``, that ``n`` fits a C ``int``, that ``min_gap`` is finite
      and strictly positive (a non-positive value never terminates the
      generation loop by its own test), and that both arrays have length at
      least ``n`` -- not exactly ``n``, since the caller sizes for the worst
      case -- in the function body;
    * that ``out_nodes`` and ``out_weights`` are passed by keyword, by that same
      signature, for the reason given on :func:`gauss_legendre_symmetric`.

    Call :func:`pantr.quad.get_tanh_sinh_1d` for the ordinary path, which
    derives ``min_gap`` from the requested dtype, maps the result onto
    ``[0, 1]``, narrows, and allocates and trims the arrays for you.

    Args:
        n (int): Requested number of points. Must be at least 1 and must fit
            a C ``int``.
        min_gap (float): Smallest endpoint distance ``1 - |x|`` a node may
            carry. Must be finite and strictly positive.
        out_nodes (npt.NDArray[np.float64]): Output array of length at least
            ``n``, 1D, C-contiguous and writable. Only the first ``m`` entries
            are written, where ``m`` is the return value.
        out_weights (npt.NDArray[np.float64]): Output array of length at
            least ``n``, written the same way.

    Returns:
        int: The effective node count ``m``, which may be less than ``n``.

    Raises:
        TypeError: If either array has a dtype other than float64, the wrong
            rank, or is not C-contiguous. A non-contiguous array is **refused
            rather than converted**, for the reason given on
            :func:`tabulate_cardinal_bspline_1d`. Also raised if ``out_nodes``
            or ``out_weights`` is passed positionally.
        ValueError: If ``n`` is less than 1, if ``n`` is too large to fit a C
            ``int``, if ``min_gap`` is not finite or not strictly positive, or
            if either array's length is less than ``n``.
    """

def trapezoidal(
    n: int,
    *,
    out_nodes: npt.NDArray[np.float64],
    out_weights: npt.NDArray[np.float64],
) -> None:
    """Tabulate the ``n``-point trapezoidal rule on ``[0, 1]``.

    The **C++ half of Layer 2**. Unlike the other rules bound here the result
    is already on ``[0, 1]``, so the ordinary path only narrows it -- there is
    no map onto the unit interval to apply. A single overload, as for
    :func:`gauss_legendre_symmetric`: the kernel emits ``double`` and the
    caller narrows, which was measured to reproduce
    ``np.linspace(..., dtype=float32)`` bit for bit.

    What is checked, and where:

    * dtype (float64 only), rank, C-contiguity, device and writability of both
      arrays, by nanobind's typed signature, before the body runs;
    * that ``n`` is a non-negative integer, by that same signature;
    * that ``n >= 1``, that ``n`` fits a C ``int``, and that both arrays have
      length exactly ``n``, in the function body;
    * that ``out_nodes`` and ``out_weights`` are passed by keyword, by that same
      signature, for the reason given on :func:`gauss_legendre_symmetric`.

    Call :func:`pantr.quad.get_trapezoidal_1d` for the ordinary path, which
    narrows to the requested dtype and allocates the arrays for you.

    Args:
        n (int): Number of points. Must be at least 1 and must fit a C
            ``int``.
        out_nodes (npt.NDArray[np.float64]): Output array of length ``n``,
            1D, C-contiguous and writable. Written in full, ascending from 0
            to 1.
        out_weights (npt.NDArray[np.float64]): Output array of length ``n``,
            matching ``out_nodes`` in shape. Written in full.

    Raises:
        TypeError: If either array has a dtype other than float64, the wrong
            rank, or is not C-contiguous. A non-contiguous array is **refused
            rather than converted**, for the reason given on
            :func:`tabulate_cardinal_bspline_1d`. Also raised if ``out_nodes``
            or ``out_weights`` is passed positionally.
        ValueError: If ``n`` is less than 1, if ``n`` is too large to fit a C
            ``int``, or if either array's length is not exactly ``n``.
    """

def modified_chebyshev_nodes(
    n: int,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Tabulate the ``n`` modified Chebyshev interpolation nodes on ``[0, 1]``.

    The **C++ half of Layer 2**. The only kernel bound here that is templated
    on the storage type rather than fixed at ``double``: the Python computes
    in the storage format too, and a double-then-narrow port was measured to
    disagree with a float32 ``cos`` on 17% of arguments, so the template
    parameter carries the parity claim rather than being cosmetic.

    What is checked, and where:

    * dtype, rank, C-contiguity, device and writability of ``out``, by
      nanobind's typed signature, before the body runs -- ``out``'s dtype
      picks which of the two bound overloads (``float32`` or ``float64``)
      runs;
    * that ``n`` is a non-negative integer, by that same signature;
    * that ``n >= 2`` (the kernel divides by ``n - 1``), that ``n`` fits a C
      ``int``, and that ``out`` has length exactly ``n``, in the function
      body;
    * that ``out`` is passed by keyword, by that same signature -- there is
      only one output here to transpose against nothing, but the convention is
      shared uniformly across :mod:`pantr._pantr_cpp`.

    Call :func:`pantr.quad.get_modified_chebyshev_nodes_1d` for the ordinary
    path, which allocates ``out`` for you.

    Args:
        n (int): Number of nodes. Must be at least 2 and must fit a C
            ``int``.
        out (npt.NDArray[np.float32 | np.float64]): Output array of length
            ``n``, 1D, C-contiguous and writable. Written in full, from 0 to
            1.

    Raises:
        TypeError: If ``out`` has the wrong dtype or rank, or is not
            C-contiguous. A non-contiguous array is **refused rather than
            converted**, for the reason given on
            :func:`tabulate_cardinal_bspline_1d`. Also raised if ``out`` is
            passed positionally.
        ValueError: If ``n`` is less than 2, if ``n`` is too large to fit a C
            ``int``, or if ``out``'s length is not exactly ``n``.
    """

class QuadratureRule:
    """Reference quadrature rule on the unit cube, owned by the C++ core.

    The third type this extension exposes rather than a free function, under the
    2026-08-27 amendment to ``design/cross_backend_types.md``. Wrapped by
    :class:`pantr.quad.QuadratureRule`, which is the class a caller holds; this
    one is reached only through it.

    Every method here is ``double``-only and there is no second overload, unlike
    :func:`modified_chebyshev_nodes` above. The reason is not symmetry: the
    oracle casts points and weights to ``float64`` unconditionally, so a
    ``float32`` surface would have no oracle behind it
    (``design/backend_parity.md`` Rule 8), and ``scripts/ci_local.sh`` asserts
    that no template appears in ``cpp/include/pantr/quad/rule.hpp`` for the
    measured reason recorded there.

    What is checked, and where:

    * dtype (float64 only), rank, C-contiguity and device of ``points`` and
      ``weights``, by nanobind's typed signature, before the body runs -- so a
      wrong-rank array reaches Python as ``TypeError`` where the oracle raises
      ``ValueError``, and :mod:`pantr.quad._rule_nd` checks rank before calling;
    * emptiness, the length agreement between points and weights, finiteness,
      and membership of the closed unit cube, in the C++ constructor;
    * for :meth:`tensor_product`, that there is at least one axis and that each
      axis carries a matching non-empty ``(nodes, weights)`` pair, in C++;
    * for :meth:`gauss_legendre`, that ``npts`` is non-empty and every entry is
      at least 1, in C++. The Python-only ``(ndim, npts)`` convention -- a
      negative ``ndim``, or a sequence of the wrong length -- has no counterpart
      here and is checked by :mod:`pantr.quad._rule_nd`.

    Call :class:`pantr.quad.QuadratureRule` and its two factories for the
    ordinary path.

    Attributes:
        ndim (int): Spatial dimension, ``>= 1``.
        num_points (int): Number of quadrature points, ``>= 1``.
        points (npt.NDArray[np.float64]): ``(num_points, ndim)`` array in
            ``[0, 1]^ndim``, freshly allocated and read-only on every access.
        weights (npt.NDArray[np.float64]): ``(num_points,)`` array, likewise.
    """

    def __init__(
        self, points: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
    ) -> None: ...
    @property
    def ndim(self) -> int: ...
    @property
    def num_points(self) -> int: ...
    @property
    def points(self) -> npt.NDArray[np.float64]: ...
    @property
    def weights(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def tensor_product(
        rules: Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    ) -> QuadratureRule: ...
    @staticmethod
    def gauss_legendre(npts: Sequence[int]) -> QuadratureRule: ...
