"""Basis tabulation and change-of-basis kernels of the compiled extension.

Bound by ``cpp/bindings/basis.cpp`` and ``cpp/bindings/change_basis.cpp``.
Change of basis has no file of its own: the split gives one file to each area
the port is still moving, and those kernels are done.

See ``__init__.pyi`` for what this package promises and who has to keep it.
"""

import numpy as np
import numpy.typing as npt

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
