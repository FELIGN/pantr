"""Type stub for the compiled C++ extension.

The extension is a nanobind module, so mypy cannot see into it: without this
file every use of :mod:`pantr._pantr_cpp` is an ``attr-defined`` error under the
project's strict configuration, and the kernel adapters under ``pantr.basis``
are what use it. The stub is written by hand rather than generated, because it is fifteen
lines and a generated one would need regenerating on every signature change
anyway.

**It is a promise that has to be kept by hand.** Nothing checks this file
against ``cpp/bindings/pantr_cpp.cpp``; if the binding's signature changes and
this does not, mypy will happily typecheck a call that fails at run time. The
parity test exercises the real call, which is what actually catches it.
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
``tests/test_cpp_parity.py`` to choose between asserting bit-exact parity with
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
