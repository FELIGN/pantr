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
    * that ``out.shape == (points.size, degree + 1)``, in the function body.

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
            array untouched, which is why ``.noconvert()`` is on it.
        ValueError: If ``out.shape`` is not ``(points.size, degree + 1)``, or if
            ``degree`` is too large to express as a C ``int``.
    """

def gauss_legendre_symmetric(
    n: int,
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
      length exactly ``n``, in the function body.

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
            :func:`tabulate_cardinal_bspline_1d`.
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
      case -- in the function body.

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
            :func:`tabulate_cardinal_bspline_1d`.
        ValueError: If ``n`` is less than 1, if ``n`` is too large to fit a C
            ``int``, if ``min_gap`` is not finite or not strictly positive, or
            if either array's length is less than ``n``.
    """

def trapezoidal(
    n: int,
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
      length exactly ``n``, in the function body.

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
            :func:`tabulate_cardinal_bspline_1d`.
        ValueError: If ``n`` is less than 1, if ``n`` is too large to fit a C
            ``int``, or if either array's length is not exactly ``n``.
    """

def modified_chebyshev_nodes(
    n: int,
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
      body.

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
            :func:`tabulate_cardinal_bspline_1d`.
        ValueError: If ``n`` is less than 2, if ``n`` is too large to fit a C
            ``int``, or if ``out``'s length is not exactly ``n``.
    """
