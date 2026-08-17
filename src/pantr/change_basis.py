r"""Change of basis operators for various polynomial bases in 1D.

This module provides functions to create transformation matrices between different
polynomial bases including Lagrange, Bernstein, cardinal B-spline, and monomial
bases.

Architecturally, this module serves as the **bridge between different basis types**,
providing pure mathematical functions to compute the exact $(degree+1, degree+1)$
transformation matrices without tying the dense numerical quadrature logic directly
into the core Spline space objects.

Every public builder is named ``compute_A_to_B_1d`` and returns the matrix $M$ with
$M \\, [A\\ values](x) = [B\\ values](x)$.
"""

import functools
from collections.abc import Callable
from math import comb

import numpy as np
import numpy.typing as npt

from .basis import (
    LagrangeVariant,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline_1d,
    tabulate_legendre_1d,
)
from .basis._basis_lagrange import _get_lagrange_points
from .basis._basis_utils import (
    _allocate_or_validate_out,
    _validate_float_dtype,
)
from .quad import get_gauss_legendre_1d


def _prepare_square_out(
    degree: int,
    dtype: npt.DTypeLike,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate ``dtype`` and allocate/validate a ``(degree+1, degree+1)`` matrix.

    Shared prologue for the ``compute_*_1d`` change-of-basis builders, which all
    return a square ``(degree+1, degree+1)`` matrix in a validated float dtype.

    Args:
        degree (int): Polynomial degree; the matrix is ``(degree+1, degree+1)``.
        dtype (npt.DTypeLike): Output dtype; must be ``float32`` or ``float64``.
        out (npt.NDArray[np.float32 | np.float64] | None): Caller-provided output
            array to validate, or ``None`` to allocate a fresh one.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The validated or freshly allocated
        ``(degree+1, degree+1)`` matrix.

    Raises:
        ValueError: If ``dtype`` is not float32/float64, or ``out`` has the wrong
            shape, dtype, or is not writeable.
    """
    _validate_float_dtype(dtype)
    return _allocate_or_validate_out(out, (degree + 1, degree + 1), dtype)


def compute_lagrange_to_bernstein_1d(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Construct the matrix mapping Lagrange basis evaluations to Bernstein basis evaluations.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering (see https://en.wikipedia.org/wiki/Bernstein_polynomial).

    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Lagrange values] = [Bernstein values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is lower than 1, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")
    out = _prepare_square_out(degree, dtype, out)

    points = _get_lagrange_points(lagrange_variant, degree + 1, dtype)
    tabulate_bernstein_1d(degree, points, out=out.T)
    return out


def compute_bernstein_to_lagrange_1d(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Construct the matrix mapping Bernstein basis evaluations to Lagrange basis evaluations.

    The inverse direction of :func:`compute_lagrange_to_bernstein_1d`: with ``C`` from
    that function, this returns the matrix solving ``C @ result = I``, computed by one
    LU solve against the identity (:func:`numpy.linalg.solve`). No explicit inverse is
    formed.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering
        (see https://en.wikipedia.org/wiki/Bernstein_polynomial).

        Only this direction involves a solve. The forward direction needs no
        projection at all: because the Lagrange basis is cardinal at its own nodes,
        ``C[j, k]`` is just $B_j(x_k)$, so ``C`` costs one basis tabulation and carries
        no solve error. That is why this pair is much better conditioned than the
        Bernstein/cardinal and Legendre/cardinal ones.

    Warning:
        $\kappa(C)$ grows with degree and with the node family: with the default
        equispaced nodes it is ``1.2e1`` at degree 4, ``4.6e2`` at degree 8 and
        ``3.0e3`` at degree 10, so the attainable accuracy of the result, by any
        algorithm, is bounded by roughly $\kappa(C)\varepsilon$. In float64 the round
        trip $\lVert C B - I \rVert_2$ holds to ``3e-16`` at degree 4, ``1e-14`` at
        degree 8 and ``1e-13`` at degree 10. In float32 the same bound stays below 0.1
        through degree 11 for equispaced nodes and through degree 14 for the Gauss and
        Chebyshev variants, so unlike the other two pairs this one remains usable in
        single precision over the whole range of practical degrees.

    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Bernstein values] = [Lagrange values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is lower than 1, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")
    out = _prepare_square_out(degree, dtype, out)

    forward = compute_lagrange_to_bernstein_1d(degree, lagrange_variant, dtype)
    out[:] = np.linalg.solve(forward, np.eye(degree + 1, dtype=out.dtype))
    return out


def _compute_change_basis_1D(
    new_basis_eval: Callable[
        [npt.NDArray[np.float32 | np.float64]], npt.NDArray[np.float32 | np.float64]
    ],
    old_basis_eval: Callable[
        [npt.NDArray[np.float32 | np.float64]], npt.NDArray[np.float32 | np.float64]
    ],
    n_quad_pts: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a change of basis operator using numerical quadrature.

    This function computes the transformation matrix M that satisfies:
        old_basis(x) = M @ new_basis(x)

    That is, row ``i`` of ``M`` holds the coefficients of the ``i``-th *old* basis
    function expanded in the *new* basis. Note the direction: the public
    ``compute_A_to_B_1d`` wrappers document ``M @ [A values] = [B values]`` and
    therefore pass ``new_basis_eval=A``, ``old_basis_eval=B``.

    The matrix is computed by solving the system C = G M^T where:
    - G is the Gram matrix of the new basis
    - C is the mixed inner product matrix between new and old bases

    Args:
        new_basis_eval (callable): Function that evaluates the new basis at points.
        old_basis_eval (callable): Function that evaluates the old basis at points.
        n_quad_pts (int): Number of quadrature points for numerical integration.
            Must be positive.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype matching the `dtype` parameter
            if provided. The shape is determined by the number of basis functions
            in the old and new bases. This follows NumPy's style for output arrays.
            Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Change of basis transformation matrix.
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If number of quadrature points is not positive, dtype is not float32 or
            float64, or if `out` is provided and has incorrect shape or dtype.
    """
    if n_quad_pts < 1:
        raise ValueError("Number of quadrature points must be positive.")
    _validate_float_dtype(dtype)

    # 1. Get Gauss-Legendre quadrature points and weights for the inner product on [0, 1]
    points, weights = get_gauss_legendre_1d(n_quad_pts, dtype)

    # 2. Pre-evaluate all basis functions at all quadrature points for efficiency
    new_basis = new_basis_eval(points)
    old_basis = old_basis_eval(points)

    out = _allocate_or_validate_out(out, (old_basis.shape[1], new_basis.shape[1]), dtype)

    # 3. Compute the Gram matrix G for the new basis B: G_kj = <b_k, b_j>
    # The inner product <f, g> is approximated by sum(w_m * f(x_m) * g(x_m))
    weights_diag = np.diag(weights)
    G = new_basis.T @ weights_diag @ new_basis

    # 4. Compute the mixed inner product matrix C: C_ki = <b_k, a_i>
    C = new_basis.T @ weights_diag @ old_basis

    # 5. Solve the system C = G M^T for M^T, which means M = (G^-1 C)^T
    out[:] = np.linalg.solve(G, C).T
    return out


def compute_bernstein_to_cardinal_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from Bernstein to cardinal B-spline basis.

    The projected direction of the Bernstein/cardinal pair, obtained by one Gram
    projection; :func:`compute_cardinal_to_bernstein_1d` inverts the result rather
    than projecting again. This is the direction worth projecting because *its* Gram
    matrix is the Bernstein one, which is far better conditioned than the cardinal
    one: since ``cardinal = A @ bernstein``, the two are related by
    $G_{\mathrm{card}} = A \, G_{\mathrm{bern}} A^\mathsf{T}$ and therefore
    $\kappa(G_{\mathrm{card}}) \le \kappa(A)^2 \kappa(G_{\mathrm{bern}})$. Measured at
    degree 8: ``2.4e4`` against ``8.4e16``, the latter already saturated against
    $1/\varepsilon$.

    The quadrature is exact: ``degree + 1`` Gauss-Legendre points integrate
    polynomials up to degree ``2 * degree + 1`` exactly, while every inner product
    formed here is between two polynomials of degree ``degree``.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Bernstein values] = [cardinal values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)

    return _compute_change_basis_1D(
        new_basis_eval=functools.partial(tabulate_bernstein_1d, degree),
        old_basis_eval=functools.partial(tabulate_cardinal_bspline_1d, degree),
        n_quad_pts=degree + 1,
        dtype=dtype,
        out=out,
    )


def compute_cardinal_to_bernstein_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from cardinal B-spline to Bernstein basis.

    The inverse direction of :func:`compute_bernstein_to_cardinal_1d`: with ``A`` from
    that function, this returns ``B`` solving ``A @ B = I``, computed by one LU solve
    against the identity (:func:`numpy.linalg.solve`). No explicit inverse is formed.

    Note:
        A second Gram projection -- swapping the two evaluators and solving with the
        *cardinal* Gram matrix -- is the obvious alternative and is the wrong one. It
        is equivalent in exact arithmetic, but that Gram matrix satisfies
        $\kappa(G_{\mathrm{card}}) \le \kappa(A)^2 \kappa(G_{\mathrm{bern}})$ (see
        :func:`compute_bernstein_to_cardinal_1d`), so it loses twice as many digits.
        Measured round-trip error $\lVert A B - I \rVert_2$ in float64: ``2.9e-2`` at
        degree 8 and ``23.9`` at degree 9 via that projection, against ``1.3e-10`` and
        ``3.8e-10`` here.

    Warning:
        The cardinal-to-Bernstein map is intrinsically ill-conditioned at high degree
        -- $\kappa(A)$ is ``1.0e2`` at degree 4, ``9.9e3`` at degree 6, ``1.9e6`` at
        degree 8 and ``6.3e8`` at degree 10 -- so the attainable accuracy of ``B``, by
        any algorithm, is bounded by roughly $\kappa(A)\varepsilon$. In float64 the
        returned matrix reproduces the Bernstein basis to about ``4e-15`` through
        degree 4, ``3e-13`` at degree 6, ``3e-11`` at degree 8 and ``2e-8`` at degree
        10, and the round trip holds to the same order; in float32 the bound exceeds
        0.1 from degree 7 on, so the result is meaningless beyond degree 6. This is a
        property of the two bases, not of the implementation.

        Far past that point the entries themselves stop being representable: in
        float32 the exact inverse exceeds the format's range from degree 34 (``6.8e40``
        at degree 36 against a maximum of ``3.4e38``), so the result contains
        infinities and NumPy reports an overflow.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [cardinal values] = [Bernstein values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.

    Example:
        >>> import numpy as np
        >>> A = compute_bernstein_to_cardinal_1d(3)
        >>> B = compute_cardinal_to_bernstein_1d(3)
        >>> np.allclose(A @ B, np.eye(4), atol=1e-13)
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)

    forward = compute_bernstein_to_cardinal_1d(degree, dtype)
    out[:] = np.linalg.solve(forward, np.eye(degree + 1, dtype=out.dtype))
    return out


def compute_legendre_to_cardinal_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from orthonormal shifted Legendre to cardinal B-spline basis.

    The Legendre basis is the orthonormal shifted Legendre basis on ``[0, 1]``
    (see :func:`~pantr.basis.tabulate_legendre_1d`), so row ``j`` of the result
    holds the coefficients of the ``j``-th cardinal B-spline in that basis.

    The quadrature is exact: ``degree + 1`` Gauss-Legendre points integrate
    polynomials up to degree ``2 * degree + 1`` exactly, while every inner
    product formed here is between two polynomials of degree ``degree`` and so
    has degree ``2 * degree``. Consequently the Legendre Gram matrix is the
    identity up to round-off, and the returned matrix carries no quadrature
    error beyond floating-point round-off.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix A such
            that ``A @ [Legendre values] = [cardinal values]``. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.

    Example:
        >>> import numpy as np
        >>> A = compute_legendre_to_cardinal_1d(1)
        >>> np.allclose(A, [[0.5, -0.28867513], [0.5, 0.28867513]])
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)

    return _compute_change_basis_1D(
        new_basis_eval=functools.partial(tabulate_legendre_1d, degree),
        old_basis_eval=functools.partial(tabulate_cardinal_bspline_1d, degree),
        n_quad_pts=degree + 1,
        dtype=dtype,
        out=out,
    )


def compute_cardinal_to_legendre_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from cardinal B-spline to orthonormal shifted Legendre basis.

    The inverse direction of :func:`compute_legendre_to_cardinal_1d`: with ``A``
    from that function, this returns ``W`` solving ``A @ W = I``, computed by one
    LU solve against the identity (:func:`numpy.linalg.solve`). No explicit
    inverse is formed.

    Note:
        A second Gram projection -- swapping the two evaluators and solving with
        the *cardinal* Gram matrix -- is the obvious alternative and is the wrong
        one. It is equivalent in exact arithmetic, but the cardinal Gram matrix
        satisfies $\kappa(G) = \kappa(A)^2$ (verified numerically to three digits
        for degrees 0-7; at degree 8 it saturates against $1/\varepsilon$), so it
        loses twice as many digits. Measured round-trip error at degree 8 in
        float64: ``5e-6`` via the Gram projection against ``5e-10`` here. The
        forward direction does not face this because *its* Gram matrix is the
        Legendre one, which is the identity.

    Warning:
        The cardinal-to-Legendre map is intrinsically ill-conditioned at high
        degree -- $\kappa(A)$ is ``1.1e3`` at degree 4 and ``3.0e8`` at degree 8
        -- so the attainable accuracy of ``W``, by any algorithm, is bounded by
        roughly $\kappa(A)\varepsilon$. In float64 the round trip holds to about
        ``1e-13`` through degree 6 and degrades to ``5e-10`` at degree 8; in
        float32 it is meaningless beyond degree 4. This is a property of the
        bases, not of the implementation.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix W such
            that ``W @ [cardinal values] = [Legendre values]``. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.

    Example:
        >>> import numpy as np
        >>> A = compute_legendre_to_cardinal_1d(3)
        >>> W = compute_cardinal_to_legendre_1d(3)
        >>> np.allclose(W @ A, np.eye(4), atol=1e-13)
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)

    forward = compute_legendre_to_cardinal_1d(degree, dtype)
    out[:] = np.linalg.solve(forward, np.eye(degree + 1, dtype=out.dtype))
    return out


def compute_cardinal_dual_legendre_coeffs_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create the cardinal-dual $L^2$ functionals, expressed in the Legendre basis.

    Row ``i`` holds the coefficients, in the orthonormal shifted Legendre basis,
    of the dual function $D_i$ biorthogonal to the cardinal B-spline basis:

    \[
    \int_0^1 D_i(x) B_{p,j}(x) \, dx = \delta_{ij}
    \]

    Derivation: write the cardinal basis in the Legendre basis as
    ``cardinal = A @ legendre`` with ``A`` from
    :func:`compute_legendre_to_cardinal_1d`, and let $D_i = \sum_k T_{ik}
    \tilde{p}_k$. Because the Legendre basis is orthonormal,
    $\int D_i B_j = \sum_k T_{ik} A_{jk} = (A T^\mathsf{T})_{ji}$, so
    biorthogonality holds exactly when $T^\mathsf{T} = A^{-1}$, i.e.
    $T = A^{-\mathsf{T}}$. No inverse is formed: since
    :func:`compute_cardinal_to_legendre_1d` returns ``W`` with ``W @ A = I`` from
    an LU solve, the result is simply ``W.T``.

    Warning:
        Biorthogonality is limited by the same conditioning that bounds ``W`` --
        see the warning on :func:`compute_cardinal_to_legendre_1d`. In float64
        $\int D_i B_j = \delta_{ij}$ holds to about ``1e-13`` through degree 6 and
        to ``5e-10`` at degree 8.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) matrix whose row ``i``
            holds the Legendre coefficients of the dual function $D_i$. If `out` was
            provided, returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.

    Example:
        >>> import numpy as np
        >>> D = compute_cardinal_dual_legendre_coeffs_1d(2)
        >>> W = compute_cardinal_to_legendre_1d(2)
        >>> np.allclose(D, W.T)
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)

    out[:] = compute_cardinal_to_legendre_1d(degree, dtype).T
    return out


def compute_monomial_to_bernstein_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from monomial to Bernstein basis on [0, 1].

    Given a polynomial of degree ``degree`` written in the monomial basis on
    ``[0, 1]``, the returned matrix ``M`` converts its coefficient vector to
    the Bernstein basis: ``bern_coeffs = M @ mono_coeffs``. Equivalently, on
    basis evaluations, ``monomial(x) = M.T @ bernstein(x)``.

    The entries are ``M[i, j] = C(i, j) / C(degree, j)`` for ``j <= i``, else
    ``0``, where ``C(n, k)`` is the binomial coefficient.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) lower-triangular
            transformation matrix ``M`` such that ``M @ [monomial coefficients] =
            [Bernstein coefficients]``. If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out`
            is provided and has incorrect shape or dtype.
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)
    out[:] = 0

    for i in range(degree + 1):
        for j in range(i + 1):
            out[i, j] = comb(i, j) / comb(degree, j)

    return out


@functools.lru_cache(maxsize=64)
def _cached_lagrange_to_bernstein_matrix(
    degree: int,
    lagrange_variant: LagrangeVariant,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only Lagrange-to-Bernstein change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_lagrange_to_bernstein_1d` used internally by
    the extraction-operator routines.  Because the matrix depends only on
    ``(degree, lagrange_variant, dtype)`` — never on the knot vector — it is
    safe to share a single immutable copy across all calls with matching
    arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        lagrange_variant (LagrangeVariant): Lagrange node distribution.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``C @ lagrange_values = bernstein_values``.
    """
    mat = compute_lagrange_to_bernstein_1d(degree, lagrange_variant, dtype)
    mat.flags.writeable = False
    return mat


@functools.lru_cache(maxsize=64)
def _cached_cardinal_to_bernstein_matrix(
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only cardinal-B-spline-to-Bernstein change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_cardinal_to_bernstein_1d` used internally by
    the extraction-operator routines.  Because the matrix depends only on
    ``(degree, dtype)`` — never on the knot vector — it is safe to share a
    single immutable copy across all calls with matching arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``C @ cardinal_values = bernstein_values``.
    """
    mat = compute_cardinal_to_bernstein_1d(degree, dtype)
    mat.flags.writeable = False
    return mat


@functools.lru_cache(maxsize=64)
def _cached_legendre_to_cardinal_matrix(
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only Legendre-to-cardinal-B-spline change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_legendre_to_cardinal_1d`.  Because the matrix depends only on
    ``(degree, dtype)`` — never on the knot vector — it is safe to share a
    single immutable copy across all calls with matching arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``A @ legendre_values = cardinal_values``.
    """
    mat = compute_legendre_to_cardinal_1d(degree, dtype)
    mat.flags.writeable = False
    return mat


@functools.lru_cache(maxsize=64)
def _cached_cardinal_to_legendre_matrix(
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only cardinal-B-spline-to-Legendre change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_cardinal_to_legendre_1d`.  Because the matrix depends only on
    ``(degree, dtype)`` — never on the knot vector — it is safe to share a
    single immutable copy across all calls with matching arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``W @ cardinal_values = legendre_values``.
    """
    mat = compute_cardinal_to_legendre_1d(degree, dtype)
    mat.flags.writeable = False
    return mat


__all__ = [
    "compute_bernstein_to_cardinal_1d",
    "compute_bernstein_to_lagrange_1d",
    "compute_cardinal_dual_legendre_coeffs_1d",
    "compute_cardinal_to_bernstein_1d",
    "compute_cardinal_to_legendre_1d",
    "compute_lagrange_to_bernstein_1d",
    "compute_legendre_to_cardinal_1d",
    "compute_monomial_to_bernstein_1d",
]
