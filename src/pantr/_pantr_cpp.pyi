"""Type stub for the compiled C++ extension.

The extension is a nanobind module, so mypy cannot see into it: without this
file every use of :mod:`pantr._pantr_cpp` is an ``attr-defined`` error under the
project's strict configuration, and the kernel adapters under ``pantr.basis``
and ``pantr.quad`` are what use it. The stub is written by hand rather than
generated, because it is short enough to read in one sitting and a generated
one would need regenerating on every signature change anyway.

**It is a promise that has to be kept by hand.** Nothing checks this file
against ``cpp/bindings/basis.cpp`` and ``cpp/bindings/quad.cpp``; if a
binding's signature changes and this does not, mypy will happily typecheck a
call that fails at run time. The parity tests exercise the real calls, which
is what actually catches it.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

__compiler__: Final[str]
"""Compiler and version that built the extension, e.g. ``"gcc 14.4.0"``."""

__has_std_mdspan__: Final[bool]
"""Whether the build used ``std::mdspan`` rather than the Kokkos fallback."""

__fp_contract__: Final[str]
"""Whether the target ISA can fuse a multiply-add.

``"available"`` or ``"unavailable-on-target-isa"``. Read by
``tests/parity/test_basis_cardinal_bspline.py`` to choose between asserting bit-exact parity with
the numba oracle and asserting the derived FMA bound: with no fused instruction
on the target there is no rounding difference for a tolerance to absorb.
"""

def tabulate_cardinal_bspline_1d(
    degree: int,
    points: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Tabulate the cardinal B-spline basis of ``degree`` at ``points``.

    The **C++ half of Layer 2** in the layering of CLAUDE.md, not Layer 3: it
    validates every precondition the kernel behind it assumes and never checks.
    That is not optional here -- the kernel indexes ``out`` using ``degree`` and
    ``points.size`` with no bounds check of its own, so an unvalidated call
    reaches undefined behaviour rather than an exception.

    What is checked, and where:

    * dtype, rank, C-contiguity, device and writability, by nanobind's typed
      signature, before the body runs;
    * that ``degree`` is a non-negative integer, by that same signature -- the
      C++ parameter is ``unsigned``, so a negative value is rejected by the
      caster and never reaches pantr's code;
    * that ``out.shape == (points.size, degree + 1)``, in the function body;
    * that ``out`` is passed by keyword. It is the only output parameter here,
      so there is nothing within this call to transpose, but the convention is
      shared with :func:`gauss_legendre_symmetric` and the rest of
      :mod:`pantr._pantr_cpp`, where two same-typed outputs make a positional
      call transposable.

    Call :func:`pantr.basis.tabulate_cardinal_bspline_1d` for the ordinary path,
    which additionally takes points of any shape and allocates ``out`` for you.

    Args:
        degree (int): Degree of the basis. Must be non-negative and must fit a
            C ``int``.
        points (npt.NDArray[np.float32 | np.float64]): Evaluation points, 1D and
            C-contiguous.
        out (npt.NDArray[np.float32 | np.float64]): Output array of shape
            ``(points.size, degree + 1)``, matching ``points`` in dtype,
            C-contiguous and writable. Written in full.

    Raises:
        TypeError: If ``degree`` is negative or is not an integer, if either
            array has the wrong dtype or rank, or if either is not C-contiguous.
            A non-contiguous array is **refused rather than converted**:
            converting an ``out`` would fill a temporary and leave the caller's
            array untouched, which is why ``.noconvert()`` is on it. Also raised
            if ``out`` is passed positionally.
        ValueError: If ``out.shape`` is not ``(points.size, degree + 1)``, or if
            ``degree`` is too large to express as a C ``int``.
    """

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

def tabulate_bernstein_1d(
    degree: int,
    points: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Tabulate the Bernstein basis of ``degree`` at ``points``.

    The **C++ half of Layer 2**, with the same contract, the same checks and the
    same reasons as :func:`tabulate_cardinal_bspline_1d`; only the basis differs.

    Args:
        degree (int): Degree of the basis. Must be non-negative and must fit a
            C ``int``.
        points (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous
            evaluation points.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(points.size, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If ``points`` or ``out`` has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, or if ``out``
            does not have shape ``(points.size, degree + 1)``.
    """

def tabulate_legendre_1d(
    degree: int,
    points: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Tabulate the orthonormal shifted Legendre basis of ``degree`` at ``points``.

    The **C++ half of Layer 2**, with the same contract, the same checks and the
    same reasons as :func:`tabulate_cardinal_bspline_1d`; only the basis differs.

    Args:
        degree (int): Degree of the basis. Must be non-negative and must fit a
            C ``int``.
        points (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous
            evaluation points.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(points.size, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If ``points`` or ``out`` has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, or if ``out``
            does not have shape ``(points.size, degree + 1)``.
    """

def lagrange_to_bernstein_1d(
    degree: int,
    nodes: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Build the Lagrange-to-Bernstein matrix at the given Lagrange nodes.

    The **C++ half of Layer 2** for the change-of-basis builders. The nodes are an
    argument rather than a computation: two of the five node families
    :class:`pantr.basis.LagrangeVariant` offers are ones
    :mod:`pantr._backend` records as deliberately never dispatched, so resolving
    a variant to nodes stays on the Python side.

    What is checked, and where: dtype, rank, C-contiguity, device and writability
    by nanobind's typed signature; a non-negative ``degree`` by that signature's
    ``unsigned`` parameter; and ``nodes.size == degree + 1`` together with
    ``out.shape == (degree + 1, degree + 1)`` in the function body.

    Call :func:`pantr.change_basis.compute_lagrange_to_bernstein_1d` for the
    ordinary path, which resolves the variant and allocates ``out`` for you.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        nodes (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous array of
            exactly ``degree + 1`` Lagrange nodes.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If an array has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, if ``nodes``
            does not have length ``degree + 1``, or if ``out`` is not the square
            matrix that degree calls for.
    """

def bernstein_to_lagrange_1d(
    degree: int,
    nodes: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Build the Bernstein-to-Lagrange matrix at the given Lagrange nodes.

    The inverse of :func:`lagrange_to_bernstein_1d`, by one LU solve, with the
    same contract and the same checks.

    **The degree domain is not checked here.** Whether the solve is still defined
    at a given ``(degree, dtype)`` is Layer 2's Python half to enforce, and
    :func:`pantr.change_basis.compute_bernstein_to_lagrange_1d` does. Calling this
    outside that domain returns whatever the LU produces.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        nodes (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous array of
            exactly ``degree + 1`` Lagrange nodes.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If an array has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, if ``nodes``
            does not have length ``degree + 1``, or if ``out`` is not the square
            matrix that degree calls for.
    """

def bernstein_to_cardinal_1d(
    degree: int,
    nodes: npt.NDArray[np.float32 | np.float64],
    weights: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Project the cardinal B-spline basis onto the Bernstein basis.

    The **C++ half of Layer 2**. ``nodes`` and ``weights`` are the Gauss-Legendre
    rule on ``[0, 1]`` with ``degree + 1`` points, passed in rather than computed,
    and they stay two arrays because that is the shape
    :func:`pantr.quad.get_gauss_legendre_1d` returns.

    Both are keyword-checkable only by position here, so ``out`` is keyword-only
    for a sharper reason than elsewhere in this module: two same-dtype,
    same-length arrays in a row make a transposed positional call type-check, run,
    and return a plausible matrix.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        nodes (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous quadrature
            nodes, exactly ``degree + 1`` of them.
        weights (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous weights
            of the same length.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If an array has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, if ``nodes`` or
            ``weights`` does not have length ``degree + 1``, or if ``out`` is not
            the square matrix that degree calls for.
    """

def cardinal_to_bernstein_1d(
    degree: int,
    nodes: npt.NDArray[np.float32 | np.float64],
    weights: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Build the cardinal-to-Bernstein matrix, the inverse of the projection.

    Same contract and checks as :func:`bernstein_to_cardinal_1d`. **The degree
    domain is not checked here**; see :func:`bernstein_to_lagrange_1d` for why.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        nodes (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous quadrature
            nodes, exactly ``degree + 1`` of them.
        weights (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous weights
            of the same length.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If an array has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, if ``nodes`` or
            ``weights`` does not have length ``degree + 1``, or if ``out`` is not
            the square matrix that degree calls for.
    """

def legendre_to_cardinal_1d(
    degree: int,
    nodes: npt.NDArray[np.float32 | np.float64],
    weights: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Project the cardinal B-spline basis onto the orthonormal Legendre basis.

    Same contract and checks as :func:`bernstein_to_cardinal_1d`. This one's Gram
    matrix is the identity up to round-off, since the new basis is orthonormal and
    the quadrature integrates every product formed exactly.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        nodes (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous quadrature
            nodes, exactly ``degree + 1`` of them.
        weights (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous weights
            of the same length.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If an array has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, if ``nodes`` or
            ``weights`` does not have length ``degree + 1``, or if ``out`` is not
            the square matrix that degree calls for.
    """

def cardinal_to_legendre_1d(
    degree: int,
    nodes: npt.NDArray[np.float32 | np.float64],
    weights: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Build the cardinal-to-Legendre matrix, the inverse of the projection.

    Same contract and checks as :func:`bernstein_to_cardinal_1d`. **The degree
    domain is not checked here**; see :func:`bernstein_to_lagrange_1d` for why.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        nodes (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous quadrature
            nodes, exactly ``degree + 1`` of them.
        weights (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous weights
            of the same length.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If an array has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, if ``nodes`` or
            ``weights`` does not have length ``degree + 1``, or if ``out`` is not
            the square matrix that degree calls for.
    """

def cardinal_dual_legendre_coeffs_1d(
    degree: int,
    nodes: npt.NDArray[np.float32 | np.float64],
    weights: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Build the cardinal-dual L2 functionals expressed in the Legendre basis.

    The transpose of :func:`cardinal_to_legendre_1d`, with the same contract and
    checks. **The degree domain is not checked here**; see
    :func:`bernstein_to_lagrange_1d` for why.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        nodes (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous quadrature
            nodes, exactly ``degree + 1`` of them.
        weights (npt.NDArray[np.float32 | np.float64]): 1D, C-contiguous weights
            of the same length.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If an array has the wrong dtype or rank, or is not
            C-contiguous, or if ``degree`` is negative, or if ``out`` is passed
            positionally.
        ValueError: If ``degree`` is too large to fit a C ``int``, if ``nodes`` or
            ``weights`` does not have length ``degree + 1``, or if ``out`` is not
            the square matrix that degree calls for.
    """

def monomial_to_bernstein_1d(
    degree: int,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Build the monomial-to-Bernstein matrix on ``[0, 1]``.

    ``M[i, j] = C(i, j) / C(degree, j)`` for ``j <= i`` and zero above the
    diagonal. The only builder that needs neither nodes nor a quadrature rule, and
    the only one that carries no degree domain at all: it runs no solve.

    The binomials are exact integers while they stay at or below ``2**53``, i.e.
    through degree 56, so the entries below that are the correctly rounded values
    of exact rationals. Past it both backends round and may round differently.

    Args:
        degree (int): Polynomial degree. Must be non-negative and fit a C ``int``.
        out (npt.NDArray[np.float32 | np.float64]): 2D, C-contiguous, writable
            output of shape ``(degree + 1, degree + 1)`` and matching dtype.
            Keyword-only.

    Raises:
        TypeError: If ``out`` has the wrong dtype or rank, is not C-contiguous, or
            is passed positionally, or if ``degree`` is negative.
        ValueError: If ``degree`` is too large to fit a C ``int``, or if ``out`` is
            not the square matrix that degree calls for.
    """

def evaluate_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    points: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier at ``points``, fusing basis and contraction.

    Runs the Bernstein ratio recurrence and contracts each term with the control
    points in one pass, mirroring about ``u = 1/2`` so the seed cannot underflow
    at high degree.

    Call :meth:`pantr.bezier.Bezier.evaluate` for the ordinary path, which takes
    points of any shape and allocates ``out``.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, rank)``, 2D and C-contiguous.
        points (npt.NDArray[np.float32 | np.float64]): Evaluation points, 1D and
            C-contiguous.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(points.size, rank)``, matching dtype, C-contiguous and writable.
            Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``out`` is not the shape the other two arguments call for.
    """

def evaluate_bezier_deriv_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    points: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier and its derivatives up to order ``n_deriv``.

    Algorithm A2.3 of Piegl & Tiller specialised to Bernstein polynomials.

    Call :meth:`pantr.bezier.Bezier.evaluate_derivatives` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, rank)``, 2D and C-contiguous.
        points (npt.NDArray[np.float32 | np.float64]): Evaluation points, 1D and
            C-contiguous.
        n_deriv (int): Highest derivative order. Must be non-negative and fit a C
            ``int``.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(points.size, n_deriv + 1, rank)``, matching dtype, 3D,
            C-contiguous and writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            if ``out`` is passed positionally, or if ``n_deriv`` is negative.
        ValueError: If ``out`` is not the shape the other arguments call for.
    """

def degree_elevate_bezier_1d(
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
    degree_increment: int,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Degree-elevate a single Bézier segment by ``degree_increment``.

    Call :meth:`pantr.bezier.Bezier.elevate_degree` for the ordinary path.

    Args:
        degree (int): Original degree. Non-negative.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, rank)``, 2D and C-contiguous.
        degree_increment (int): Degrees to add. Non-negative.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(degree + degree_increment + 1, rank)``, matching dtype,
            C-contiguous and writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            if ``out`` is passed positionally, or if either degree is negative.
        ValueError: If ``ctrl`` does not have ``degree + 1`` rows, if ``out`` is
            the wrong shape, or if ``degree + degree_increment`` exceeds the
            exact-integer binomial envelope of 61.
    """

def slice_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    value: float,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier at a single parameter, per column, by de Casteljau.

    Call :meth:`pantr.bezier.Bezier.slice` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, n_cols)``, 2D and C-contiguous.
        value (float): Parameter in ``[0, 1]``.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(n_cols,)``, matching dtype, 1D, C-contiguous and writable.
            Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``out`` does not have ``ctrl.shape[1]`` entries.
    """

def split_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    value: float,
    *,
    out_left: npt.NDArray[np.float32 | np.float64],
    out_right: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Split a 1D Bézier at ``value`` into its two halves.

    The two outputs share a dtype and a shape, so nothing in the type system
    separates them and a positional call could exchange the halves silently.
    Both are keyword-only for that reason.

    Call :meth:`pantr.bezier.Bezier.split` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, n_cols)``, 2D and C-contiguous.
        value (float): Parameter in ``[0, 1]``.
        out_left (npt.NDArray[np.float32 | np.float64]): Left half, shape
            ``(degree + 1, n_cols)``, matching dtype, C-contiguous and writable.
            Written in full. Keyword-only.
        out_right (npt.NDArray[np.float32 | np.float64]): Right half, same
            requirements. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if either output is passed positionally.
        ValueError: If either output is not the shape ``ctrl`` calls for.
    """

def restrict_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    lower: float,
    upper: float,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Restrict a 1D Bézier to ``[lower, upper]``, reparametrized to ``[0, 1]``.

    Two de Casteljau passes, ordered so that neither divides by a small number.

    Call :meth:`pantr.bezier.Bezier.restrict` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, n_cols)``, 2D and C-contiguous.
        lower (float): Left bound in ``[0, 1)``.
        upper (float): Right bound in ``(0, 1]``.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(degree + 1, n_cols)``, matching dtype, C-contiguous and writable.
            Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``out`` is not the shape ``ctrl`` calls for.
    """

def scalar_bernstein_product_1d(
    a: npt.NDArray[np.float32 | np.float64],
    b: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Multiply two scalar 1D Béziers in the Bernstein basis.

    ``c_k = (1 / C(p+q, k)) * sum_i C(p, i) C(q, k-i) a_i b_{k-i}``.

    Args:
        a (npt.NDArray[np.float32 | np.float64]): Control points of the first
            curve, 1D and C-contiguous, at least one entry.
        b (npt.NDArray[np.float32 | np.float64]): Control points of the second
            curve, same requirements and dtype.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(a.size + b.size - 1,)``, matching dtype, C-contiguous and
            writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If either input is empty, if ``out`` is the wrong length, or
            if the summed degree exceeds the exact-integer binomial envelope
            of 61.
    """

def apply_reduction_operator(
    operator: npt.NDArray[np.float64],
    ctrl: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Apply a dense degree-reduction operator: ``out = operator @ ctrl``.

    Accumulates in ``float64`` regardless of the control points' dtype and rounds
    once on the write, which is the contract the numba original states and this
    one keeps. Rows of the operator that pin an endpoint are unit vectors, so
    those outputs reproduce their inputs bit for bit.

    The operator itself is assembled in exact rational arithmetic on the Python
    side and converted to ``float64`` before it reaches here.

    Args:
        operator (npt.NDArray[np.float64]): Reduction operator of shape
            ``(q + 1, p + 1)``. Always ``float64``, 2D and C-contiguous.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p + 1, rank)``, 2D and C-contiguous.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(q + 1, rank)``, matching ``ctrl``'s dtype, C-contiguous and
            writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``ctrl`` does not have as many rows as the operator has
            columns, or if ``out`` is the wrong shape.
    """

def yuksel_roots(
    coeff: npt.NDArray[np.float32 | np.float64],
    tol: float,
    *,
    out: npt.NDArray[np.float64],
) -> int:
    """Find every root on [0, 1] by Yuksel's monotone decomposition.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients,
            C-contiguous and non-empty.
        tol (float): Bracket-width tolerance. Finite and strictly positive.
        out (npt.NDArray[np.float64]): Receives the roots, unsorted. Always
            ``float64`` whatever ``coeff`` is, C-contiguous, writable, and at least
            ``max(degree, 1)`` long. Only the returned count of entries is written.
            Keyword-only.

    Returns:
        int: How many entries of ``out`` are valid.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``coeff`` is empty, if ``tol`` is not finite and positive,
            or if ``out`` is too short.
    """

def clip_roots(
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
    *,
    out: npt.NDArray[np.float64],
) -> int:
    """Find every root on [0, 1] by Bézier clipping.

    The candidates are unsorted and may repeat: the same root reaches the output
    from several converging intervals, and merging them is :func:`dedup_roots`.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients,
            C-contiguous and non-empty.
        param_tol (float): Bracket-width termination tolerance. Finite, positive.
        geom_tol (float): Geometric tolerance for near-zero detection. Finite,
            positive.
        out (npt.NDArray[np.float64]): Receives the candidates. Always ``float64``,
            C-contiguous, writable, and at least ``3 * degree + 4`` long, which is
            the kernel's own worst case before the merge. Keyword-only.

    Returns:
        int: How many entries of ``out`` are valid.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``coeff`` is empty, if either tolerance is not finite and
            positive, or if ``out`` is too short.
    """

def dedup_roots(
    raw_roots: npt.NDArray[np.float64],
    n_roots: int,
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
    *,
    out: npt.NDArray[np.float64],
) -> int:
    """Sort root candidates and merge the duplicates, with a derivative-aware radius.

    Args:
        raw_roots (npt.NDArray[np.float64]): Candidates, of which the first
            ``n_roots`` are valid. C-contiguous.
        n_roots (int): Number of valid candidates, in ``[0, len(raw_roots)]``.
        coeff (npt.NDArray[np.float32 | np.float64]): Original Bernstein
            coefficients, used for the derivative. C-contiguous and non-empty.
        param_tol (float): Parametric tolerance. Finite and positive.
        geom_tol (float): Geometric tolerance. Finite and positive.
        out (npt.NDArray[np.float64]): Receives the merged roots, sorted ascending.
            C-contiguous, writable, at least ``n_roots`` long. Keyword-only.

    Returns:
        int: How many entries of ``out`` are valid.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``coeff`` is empty, if either tolerance is not finite and
            positive, if ``n_roots`` is not a valid count, or if ``out`` is too
            short.
    """

def solve_monotone_root(
    coeff: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> float:
    """Find the unique root of a monotone scalar Bernstein polynomial on [0, 1].

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients of
            a monotone polynomial. C-contiguous and non-empty.
        tol (float): Bracket-width termination tolerance. Finite and positive.

    Returns:
        float: The root parameter, or NaN when no sign change is detected across
            [0, 1].

    Raises:
        TypeError: If ``coeff`` has the wrong dtype or rank, or is not C-contiguous.
        ValueError: If ``coeff`` is empty or ``tol`` is not finite and positive.
    """

def find_roots_batch(
    coeffs: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
    *,
    out_roots: npt.NDArray[np.float64],
    out_counts: npt.NDArray[np.int64],
) -> None:
    """Find the roots of many same-degree scalar Bernstein polynomials.

    Each polynomial is dispatched between Yuksel and clipping on its own degree and
    dynamic range, then deduplicated. Rows are independent and each writes only its
    own, so no reduction crosses them.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of shape
            ``(n_polys, degree + 1)``, C-contiguous with at least one column.
        param_tol (float): Parametric tolerance. Finite and positive.
        geom_tol (float): Geometric tolerance. Finite and positive.
        out_roots (npt.NDArray[np.float64]): Shape ``(n_polys, max(degree, 1))``,
            C-contiguous and writable. Entries past each row's count are left as the
            caller set them, which is NaN. Keyword-only.
        out_counts (npt.NDArray[np.int64]): Shape ``(n_polys,)``, C-contiguous and
            writable, receiving the per-row root count. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if an output is passed positionally.
        ValueError: If ``coeffs`` has no columns, if either tolerance is not finite
            and positive, or if the outputs do not have ``n_polys`` rows.
    """

def solve_monotone_root_batch(
    coeffs: npt.NDArray[np.float32 | np.float64],
    tol: float,
    *,
    out_roots: npt.NDArray[np.float64],
) -> None:
    """Solve for the monotone root of many same-degree Bernstein polynomials.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of shape
            ``(n_polys, degree + 1)``, C-contiguous with at least one column.
        tol (float): Bracket-width termination tolerance. Finite and positive.
        out_roots (npt.NDArray[np.float64]): Shape ``(n_polys,)``, C-contiguous and
            writable, **pre-filled with NaN by the caller**: a row whose polynomial
            has no root is left untouched rather than written. Keyword-only.

    Raises:
        TypeError: If ``coeffs`` or ``out_roots`` has the wrong dtype or rank, is
            not C-contiguous, or if ``out_roots`` is passed positionally.
        ValueError: If ``coeffs`` has no columns, if ``tol`` is not finite and
            positive, or if ``out_roots`` does not have ``n_polys`` entries.
    """
