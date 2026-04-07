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
    _auto_reduce_1d,
    _auto_reduce_2d,
    _bincoeff,
    _collapse_2d,
    _collapse_3d,
    _degree_elevate_1d,
    _elevated_derivative_along_axis_2d,
    _elevated_derivative_along_axis_3d,
    _eval_bernstein_basis_1d,
)

_SVD_TOL_FACTOR: float = 100.0
"""Multiplier on machine epsilon for SVD truncation."""

_AUTO_REDUCE_TOL: float = 1e4 * 2.2204460492503131e-16
"""Tolerance for auto-reduction of resultant polynomial degree.

Matches algoim's ``1e4 * eps`` used in ``resultant_core``.
"""


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

    # Next n rows from g.
    for row in range(n):
        for j in range(m + 1):
            col = row + j
            S[m + row, col] = g[j] * _bincoeff(m, j)

    # Apply diagonal scaling D^{-1}: D_i = C(n+m-1, i).
    # Guard against overflow for high-degree inputs where bincoeff → inf.
    _BINCOEFF_OVERFLOW: float = 1e300
    for col in range(size):
        bc = _bincoeff(size - 1, col)
        d_inv = 0.0 if bc == 0.0 or bc > _BINCOEFF_OVERFLOW else 1.0 / bc
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
                  + (j*(n-i))/(i*(n-j)) * b_{i+1,j} for 1<=i, j<=n-1

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

    # Interior: traverse rows bottom-up, columns left-to-right.
    # b_{i,j+1} = n^2/(i*(n-j)) * (f_i*g_j - f_j*g_i) + j*(n-i)/(i*(n-j)) * b_{i+1,j}
    # Note: need b_{i+1,j} which is already computed (we go bottom-up).
    # Loop order matches algoim: outer i from n-1 down to 1, inner j from 1 to i-1.
    for i in range(n - 1, 0, -1):
        for j in range(1, i):
            coeff1 = fn * fn / (float(i) * float(n - j))
            coeff2 = float(j) * float(n - i) / (float(i) * float(n - j))
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
# Section D: QR-based determinant (Givens rotations with column pivoting)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _det_qr(A: npt.NDArray[np.float64]) -> float:  # noqa: PLR0912
    """Compute the determinant of a square matrix via Givens QR with column pivoting.

    Implements the same algorithm as algoim's ``det_qr`` (Saye, JCP 2022,
    supplementary ``quadrature_multipoly.hpp``). At each step, the column
    with largest Euclidean norm is pivoted into position, then Givens
    rotations eliminate the sub-diagonal entries. The determinant is the
    product of the R diagonal, with sign flips for each column swap.

    This is more numerically stable than LU-based ``np.linalg.det`` for
    the Sylvester/Bezout matrices arising in resultant computations.

    Args:
        A (npt.NDArray[np.float64]): Square matrix of shape ``(n, n)``.
            **Overwritten** during computation.

    Returns:
        float: Determinant of the input matrix.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = A.shape[0]

    # Fast paths for small matrices (common for degree-1 and degree-2 polynomials).
    if n == 1:
        return float(A[0, 0])
    if n == 2:  # noqa: PLR2004
        return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if n == 3:  # noqa: PLR2004
        return float(
            A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
        )

    det = 1.0

    # Pre-compute column norms squared (updated incrementally).
    col_norms_sq = np.empty(n, dtype=np.float64)
    for col in range(n):
        s = 0.0
        for row in range(n):
            s += A[row, col] * A[row, col]
        col_norms_sq[col] = s

    for j in range(n):
        # Column pivoting: find column k >= j with largest norm (using cached norms).
        best_norm = col_norms_sq[j]
        best_k = j
        for col in range(j + 1, n):
            if col_norms_sq[col] > best_norm:
                best_norm = col_norms_sq[col]
                best_k = col

        # Swap columns j and best_k.
        if best_k != j:
            for row in range(n):
                A[row, j], A[row, best_k] = A[row, best_k], A[row, j]
            col_norms_sq[j], col_norms_sq[best_k] = col_norms_sq[best_k], col_norms_sq[j]
            det = -det

        # Givens rotations to zero out sub-diagonal in column j.
        for i in range(n - 1, j, -1):
            a_val = A[i - 1, j]
            b_val = A[i, j]
            # Compute Givens rotation coefficients.
            if b_val == 0.0:
                c = 1.0
                s_val = 0.0
            elif abs(b_val) > abs(a_val):
                tmp = a_val / b_val
                s_val = 1.0 / np.sqrt(1.0 + tmp * tmp)
                c = tmp * s_val
            else:
                tmp = b_val / a_val
                c = 1.0 / np.sqrt(1.0 + tmp * tmp)
                s_val = tmp * c
            # Apply rotation to rows i-1 and i, columns j..n-1.
            for col in range(j, n):
                x = A[i - 1, col]
                y = A[i, col]
                A[i - 1, col] = c * x + s_val * y
                A[i, col] = -s_val * x + c * y

        det *= A[j, j]

        # Update column norms: subtract the eliminated row's contribution.
        for col in range(j + 1, n):
            col_norms_sq[col] -= A[j, col] * A[j, col]
            col_norms_sq[col] = max(col_norms_sq[col], 0.0)

    return det


# ---------------------------------------------------------------------------
# Section D2: 1D resultant (determinant of Sylvester/Bezout matrix)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _resultant_1d(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
) -> float:
    """Compute the resultant of two 1D Bernstein polynomials.

    Uses the Bezout matrix if degrees are equal, Sylvester otherwise.
    The determinant is computed via QR with Givens rotations and column
    pivoting for numerical stability.

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
        return _det_qr(B)
    else:
        S = _sylvester_matrix(f, g)
        return _det_qr(S)


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
    result = _resultant_2d_at_degree(f, g, k, nk_f, nk_g, out_deg)

    # Try auto-reduction: if the resultant is effectively lower degree,
    # recompute at the reduced degree for better conditioning.
    reduced = _auto_reduce_1d(result, _AUTO_REDUCE_TOL)
    if len(reduced) < len(result):
        new_deg = len(reduced) - 1
        result = _resultant_2d_at_degree(f, g, k, nk_f, nk_g, new_deg)

    return result


@nb_jit(nopython=True, cache=True)
def _resultant_2d_at_degree(  # noqa: PLR0913
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    k: int,
    nk_f: int,
    nk_g: int,
    out_deg: int,
) -> npt.NDArray[np.float64]:
    """Compute resultant interpolated at a given output degree.

    Args:
        f (npt.NDArray[np.float64]): 2D coefficient array of f.
        g (npt.NDArray[np.float64]): 2D coefficient array of g.
        k (int): Elimination axis.
        nk_f (int): Degree of f in direction k.
        nk_g (int): Degree of g in direction k.
        out_deg (int): Output Bernstein degree.

    Returns:
        npt.NDArray[np.float64]: 1D Bernstein coefficients.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n_nodes = out_deg + 1
    nodes = _chebyshev_nodes(n_nodes)

    values = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        f_1d = _collapse_2d(f, k, nodes[i])
        g_1d = _collapse_2d(g, k, nodes[i])
        nf = len(f_1d) - 1
        ng = len(g_1d) - 1
        if nf < nk_f:
            f_1d = _degree_elevate_1d(f_1d, nk_f - nf)
        if ng < nk_g:
            g_1d = _degree_elevate_1d(g_1d, nk_g - ng)
        values[i] = _resultant_1d(f_1d, g_1d)

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

    result = _resultant_3d_at_degrees(f, g, k, nk_f, nk_g, out_deg0, out_deg1)

    # Try auto-reduction.
    reduced = _auto_reduce_2d(result, _AUTO_REDUCE_TOL)
    if reduced.shape[0] < result.shape[0] or reduced.shape[1] < result.shape[1]:
        new_deg0 = reduced.shape[0] - 1
        new_deg1 = reduced.shape[1] - 1
        result = _resultant_3d_at_degrees(f, g, k, nk_f, nk_g, new_deg0, new_deg1)

    return result


@nb_jit(nopython=True, cache=True)
def _resultant_3d_at_degrees(  # noqa: PLR0913
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    k: int,
    nk_f: int,
    nk_g: int,
    out_deg0: int,
    out_deg1: int,
) -> npt.NDArray[np.float64]:
    """Compute 3D resultant interpolated at given output degrees.

    Args:
        f (npt.NDArray[np.float64]): 3D coefficient array of f.
        g (npt.NDArray[np.float64]): 3D coefficient array of g.
        k (int): Elimination axis.
        nk_f (int): Degree of f in direction k.
        nk_g (int): Degree of g in direction k.
        out_deg0 (int): Output degree in first tangential direction.
        out_deg1 (int): Output degree in second tangential direction.

    Returns:
        npt.NDArray[np.float64]: 2D Bernstein coefficients.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0 = out_deg0 + 1
    n1 = out_deg1 + 1
    nodes0 = _chebyshev_nodes(n0)
    nodes1 = _chebyshev_nodes(n1)

    values = np.empty((n0, n1), dtype=np.float64)
    x_tang = np.empty(2, dtype=np.float64)
    for i0 in range(n0):
        for i1 in range(n1):
            x_tang[0] = nodes0[i0]
            x_tang[1] = nodes1[i1]
            f_1d = _collapse_3d(f, k, x_tang)
            g_1d = _collapse_3d(g, k, x_tang)
            values[i0, i1] = _resultant_1d(f_1d, g_1d)

    # Interpolate dimension by dimension.
    interp_step1 = np.empty((out_deg0 + 1, n1), dtype=np.float64)
    for i1 in range(n1):
        col_vals = np.empty(n0, dtype=np.float64)
        for i0 in range(n0):
            col_vals[i0] = values[i0, i1]
        coeffs_0 = _bernstein_interpolate_1d(col_vals, nodes0, out_deg0)
        for i0 in range(out_deg0 + 1):
            interp_step1[i0, i1] = coeffs_0[i0]

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
