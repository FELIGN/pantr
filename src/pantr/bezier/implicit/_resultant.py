"""Resultant and discriminant computation for multivariate Bernstein polynomials.

Computes resultants and pseudo-discriminants by evaluating determinants of
Sylvester/Bezout matrices at Chebyshev interpolation nodes, then recovering
the resulting Bernstein polynomial via SVD-based interpolation.

Main exports:

- :func:`resultant_2d` -- resultant of two 2D TP Bernstein polynomials along axis k.
- :func:`resultant_3d` -- resultant of two 3D TP Bernstein polynomials along axis k.
- :func:`discriminant_2d` -- pseudo-discriminant of a 2D polynomial along axis k.
- :func:`discriminant_3d` -- pseudo-discriminant of a 3D polynomial along axis k.

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit quadrature module.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.bezier.implicit._bernstein import (
    _collapse_2d,
    _collapse_3d,
    _degree_elevate_1d,
    _elevated_derivative_along_axis_2d,
    _elevated_derivative_along_axis_3d,
    _eval_bernstein_basis_1d,
)

_SVD_TOL_FACTOR: float = 100.0
"""Multiplier on machine epsilon for SVD truncation."""


# ---------------------------------------------------------------------------
# Section A: Chebyshev interpolation nodes
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _chebyshev_nodes(n: int) -> npt.NDArray[np.float64]:
    """Compute modified Chebyshev nodes on [0, 1], inclusive of endpoints.

    Uses the formula ``x_i = 0.5 + 0.5 * cos(i/(n-1) * pi)`` for
    ``i = 0, ..., n-1``, reversed to ascending order.

    Args:
        n (int): Number of nodes (>= 1).

    Returns:
        npt.NDArray[np.float64]: Nodes of shape ``(n,)`` in ascending order.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if n == 1:
        return np.array([0.5], dtype=np.float64)
    nodes = np.empty(n, dtype=np.float64)
    for i in range(n):
        nodes[n - 1 - i] = 0.5 + 0.5 * np.cos(float(i) / float(n - 1) * np.pi)
    return nodes


# ---------------------------------------------------------------------------
# Section B: Sylvester and Bezout matrices
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _bincoeff(n: int, k: int) -> float:
    """Compute binomial coefficient C(n, k) as a float.

    Args:
        n (int): Top argument.
        k (int): Bottom argument.

    Returns:
        float: C(n, k).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:  # noqa: PLR1714
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result = result * float(n - i) / float(i + 1)
    return result


@nb_jit(nopython=True, cache=True)
def _sylvester_matrix(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build the scaled Sylvester resultant matrix for two 1D Bernstein polynomials.

    For f of degree n and g of degree m, the matrix is (n+m) x (n+m).
    Row layout: first m rows from f, next n rows from g. Entries are
    scaled by binomial coefficients per the Bernstein formulation in
    Winkler (2000), with diagonal scaling D^{-1}.

    Args:
        f (npt.NDArray[np.float64]): Coefficients of f, length n+1.
        g (npt.NDArray[np.float64]): Coefficients of g, length m+1.

    Returns:
        npt.NDArray[np.float64]: Sylvester matrix of shape ``(n+m, n+m)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(f) - 1
    m = len(g) - 1
    size = n + m
    S = np.zeros((size, size), dtype=np.float64)

    # First m rows from f.
    for row in range(m):
        for j in range(n + 1):
            col = row + j
            S[row, col] = f[j] * _bincoeff(n, j)
        # Scale row by 1/C(n+m-1, row) ... actually, apply D^{-1} scaling.

    # Next n rows from g.
    for row in range(n):
        for j in range(m + 1):
            col = row + j
            S[m + row, col] = g[j] * _bincoeff(m, j)

    # Apply diagonal scaling D^{-1}: D_i = C(n+m-1, i).
    for col in range(size):
        d_inv = 1.0 / _bincoeff(size - 1, col)
        for row in range(size):
            S[row, col] *= d_inv

    return S


@nb_jit(nopython=True, cache=True)
def _bezout_matrix(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build the Bezout resultant matrix for two 1D Bernstein polynomials of equal degree.

    The matrix B = (b_{i,j}) of size n x n is defined by the recurrence:
    - b_{i,1} = (n/i)(f_i*g_0 - f_0*g_i) for 1 <= i <= n
    - b_{n,j+1} = (n/(n-j))(f_n*g_j - f_j*g_n) for 1 <= j <= n-1
    - b_{i,j+1} = (n^2/(i*(n-j)))(f_i*g_j - f_j*g_i)
                  + (i*(n-i))/(i*(n-j)) * b_{i+1,j} for 1<=i, j<=n-1

    The matrix is symmetric: b_{i,j} = b_{j,i}.

    Args:
        f (npt.NDArray[np.float64]): Coefficients of f, length n+1.
        g (npt.NDArray[np.float64]): Coefficients of g, length n+1.

    Returns:
        npt.NDArray[np.float64]: Bezout matrix of shape ``(n, n)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(f) - 1
    if n == 0:
        return np.zeros((1, 1), dtype=np.float64)

    B = np.zeros((n, n), dtype=np.float64)
    fn = float(n)

    # Build lower-left triangular portion, then copy to upper-right by symmetry.
    # Using 1-based indexing internally, converting to 0-based for array access.

    # First column (j=1 -> j_idx=0): b_{i,1} = (n/i)(f_i*g_0 - f_0*g_i)
    for i in range(1, n + 1):
        B[i - 1, 0] = (fn / float(i)) * (f[i] * g[0] - f[0] * g[i])

    # Last row (i=n -> i_idx=n-1): b_{n,j+1} = (n/(n-j))(f_n*g_j - f_j*g_n)
    for j in range(1, n):
        B[n - 1, j] = (fn / float(n - j)) * (f[n] * g[j] - f[j] * g[n])

    # Interior: traverse from bottom-right to top-left.
    # b_{i,j+1} = n^2/(i*(n-j)) * (f_i*g_j - f_j*g_i) + (i*(n-i))/(i*(n-j)) * b_{i+1,j}
    # Note: need b_{i+1,j} which is already computed (we go bottom-up).
    for j in range(1, n - 1):
        for i in range(n - 1, 0, -1):
            coeff1 = fn * fn / (float(i) * float(n - j))
            coeff2 = float(n - i) / float(n - j)
            B[i - 1, j] = coeff1 * (f[i] * g[j] - f[j] * g[i]) + coeff2 * B[i, j - 1]

    # Copy lower triangular to upper (symmetric).
    for i in range(n):
        for j in range(i + 1, n):
            B[i, j] = B[j, i]

    return B


# ---------------------------------------------------------------------------
# Section C: SVD-based Bernstein interpolation
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _bernstein_interpolate_1d(
    values: npt.NDArray[np.float64],
    nodes: npt.NDArray[np.float64],
    degree: int,
) -> npt.NDArray[np.float64]:
    """Recover Bernstein coefficients from nodal values via SVD.

    Builds a Vandermonde-like matrix of Bernstein basis evaluations at the
    given nodes, then solves the interpolation system using truncated SVD.

    Args:
        values (npt.NDArray[np.float64]): Function values at nodes, shape ``(n,)``.
        nodes (npt.NDArray[np.float64]): Interpolation nodes in [0, 1], shape ``(n,)``.
        degree (int): Target Bernstein polynomial degree.

    Returns:
        npt.NDArray[np.float64]: Bernstein coefficients of shape ``(degree + 1,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n_pts = len(nodes)
    n_coeffs = degree + 1

    # Build Vandermonde matrix: V[i, j] = B^degree_j(nodes[i]).
    V = np.empty((n_pts, n_coeffs), dtype=np.float64)
    for i in range(n_pts):
        basis = _eval_bernstein_basis_1d(degree, nodes[i])
        for j in range(n_coeffs):
            V[i, j] = basis[j]

    # SVD solve with truncation.
    U, s, Vt = np.linalg.svd(V, False)

    # Determine truncation threshold.
    tol = s[0] * max(n_pts, n_coeffs) * 2.2204460492503131e-16 * _SVD_TOL_FACTOR

    # Compute coefficients: c = V^+ @ values.
    coeffs = np.zeros(n_coeffs, dtype=np.float64)
    for k in range(len(s)):
        if s[k] > tol:
            proj = 0.0
            for i in range(n_pts):
                proj += U[i, k] * values[i]
            for j in range(n_coeffs):
                coeffs[j] += (proj / s[k]) * Vt[k, j]

    return coeffs


# ---------------------------------------------------------------------------
# Section D: 1D resultant (determinant of Sylvester/Bezout matrix)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _resultant_1d(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
) -> float:
    """Compute the resultant of two 1D Bernstein polynomials.

    Uses the Bezout matrix if degrees are equal, Sylvester otherwise.

    Args:
        f (npt.NDArray[np.float64]): Coefficients of f.
        g (npt.NDArray[np.float64]): Coefficients of g.

    Returns:
        float: Resultant value (det of the matrix).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(f) - 1
    m = len(g) - 1

    if n == m:
        B = _bezout_matrix(f, g)
        return float(np.linalg.det(B))
    else:
        S = _sylvester_matrix(f, g)
        return float(np.linalg.det(S))


# ---------------------------------------------------------------------------
# Section E: Multivariate resultant via Chebyshev interpolation
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def resultant_2d(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute the resultant of two 2D TP Bernstein polynomials along axis *k*.

    Evaluates the 1D resultant at Chebyshev nodes in the tangential direction,
    then interpolates back to Bernstein form.

    The output degree in the tangential direction is
    ``n_k * m_tang + m_k * n_tang`` (equation 6 from paper).

    Args:
        f (npt.NDArray[np.float64]): 2D coefficient array of f.
        g (npt.NDArray[np.float64]): 2D coefficient array of g.
        k (int): Elimination axis (0 or 1).

    Returns:
        npt.NDArray[np.float64]: 1D Bernstein coefficients of the resultant.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0_f, n1_f = f.shape[0] - 1, f.shape[1] - 1
    n0_g, n1_g = g.shape[0] - 1, g.shape[1] - 1

    if k == 0:
        nk_f, nt_f = n0_f, n1_f
        nk_g, nt_g = n0_g, n1_g
    else:
        nk_f, nt_f = n1_f, n0_f
        nk_g, nt_g = n1_g, n0_g

    # Output degree in tangential direction.
    out_deg = nk_f * nt_g + nk_g * nt_f
    n_nodes = out_deg + 1
    nodes = _chebyshev_nodes(n_nodes)

    # Evaluate resultant at each node.
    values = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        f_1d = _collapse_2d(f, k, nodes[i])
        g_1d = _collapse_2d(g, k, nodes[i])
        # Degree-elevate to common degree if needed.
        nf = len(f_1d) - 1
        ng = len(g_1d) - 1
        if nf < nk_f:
            f_1d = _degree_elevate_1d(f_1d, nk_f - nf)
        if ng < nk_g:
            g_1d = _degree_elevate_1d(g_1d, nk_g - ng)
        values[i] = _resultant_1d(f_1d, g_1d)

    # Interpolate to Bernstein form.
    return _bernstein_interpolate_1d(values, nodes, out_deg)


@nb_jit(nopython=True, cache=True)
def resultant_3d(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute the resultant of two 3D TP Bernstein polynomials along axis *k*.

    Evaluates the 1D resultant on a tensor-product Chebyshev grid in the two
    tangential directions, then interpolates to a 2D Bernstein polynomial
    dimension by dimension.

    Args:
        f (npt.NDArray[np.float64]): 3D coefficient array of f.
        g (npt.NDArray[np.float64]): 3D coefficient array of g.
        k (int): Elimination axis (0, 1, or 2).

    Returns:
        npt.NDArray[np.float64]: 2D Bernstein coefficients of the resultant.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    shape_f = (f.shape[0] - 1, f.shape[1] - 1, f.shape[2] - 1)
    shape_g = (g.shape[0] - 1, g.shape[1] - 1, g.shape[2] - 1)

    nk_f = shape_f[k]
    nk_g = shape_g[k]

    # Tangential directions (in order, skipping k).
    tang_dirs = np.empty(2, dtype=np.int64)
    idx = 0
    for d in range(3):
        if d != k:
            tang_dirs[idx] = d
            idx += 1

    t0, t1 = tang_dirs[0], tang_dirs[1]

    # Output degrees in tangential directions.
    out_deg0 = nk_f * shape_g[t0] + nk_g * shape_f[t0]
    out_deg1 = nk_f * shape_g[t1] + nk_g * shape_f[t1]

    n0 = out_deg0 + 1
    n1 = out_deg1 + 1
    nodes0 = _chebyshev_nodes(n0)
    nodes1 = _chebyshev_nodes(n1)

    # Evaluate resultant on tensor-product grid.
    values = np.empty((n0, n1), dtype=np.float64)
    x_tang = np.empty(2, dtype=np.float64)
    for i0 in range(n0):
        for i1 in range(n1):
            x_tang[0] = nodes0[i0]
            x_tang[1] = nodes1[i1]
            f_1d = _collapse_3d(f, k, x_tang)
            g_1d = _collapse_3d(g, k, x_tang)
            values[i0, i1] = _resultant_1d(f_1d, g_1d)

    # Interpolate dimension by dimension: first along dir 0, then dir 1.
    # Step 1: For each column i1, interpolate the n0 values to Bernstein in dir 0.
    interp_step1 = np.empty((out_deg0 + 1, n1), dtype=np.float64)
    for i1 in range(n1):
        col_vals = np.empty(n0, dtype=np.float64)
        for i0 in range(n0):
            col_vals[i0] = values[i0, i1]
        coeffs_0 = _bernstein_interpolate_1d(col_vals, nodes0, out_deg0)
        for i0 in range(out_deg0 + 1):
            interp_step1[i0, i1] = coeffs_0[i0]

    # Step 2: For each row i0, interpolate the n1 values to Bernstein in dir 1.
    result = np.empty((out_deg0 + 1, out_deg1 + 1), dtype=np.float64)
    for i0 in range(out_deg0 + 1):
        row_vals = np.empty(n1, dtype=np.float64)
        for i1 in range(n1):
            row_vals[i1] = interp_step1[i0, i1]
        coeffs_1 = _bernstein_interpolate_1d(row_vals, nodes1, out_deg1)
        for i1 in range(out_deg1 + 1):
            result[i0, i1] = coeffs_1[i1]

    return result


# ---------------------------------------------------------------------------
# Section F: Discriminant (pseudo-discriminant)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def discriminant_2d(
    f: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute the pseudo-discriminant of a 2D TP Bernstein polynomial along axis *k*.

    The pseudo-discriminant is defined as ``Delta(f; k) = R(f, d_k f; k)``
    where ``d_k f`` is the elevated derivative of *f* in direction *k*.

    Args:
        f (npt.NDArray[np.float64]): 2D coefficient array.
        k (int): Direction for discriminant computation.

    Returns:
        npt.NDArray[np.float64]: 1D Bernstein coefficients of the discriminant.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    deriv_f = _elevated_derivative_along_axis_2d(f, k)
    return resultant_2d(f, deriv_f, k)


@nb_jit(nopython=True, cache=True)
def discriminant_3d(
    f: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute the pseudo-discriminant of a 3D TP Bernstein polynomial along axis *k*.

    Args:
        f (npt.NDArray[np.float64]): 3D coefficient array.
        k (int): Direction for discriminant computation.

    Returns:
        npt.NDArray[np.float64]: 2D Bernstein coefficients of the discriminant.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    deriv_f = _elevated_derivative_along_axis_3d(f, k)
    return resultant_3d(f, deriv_f, k)
