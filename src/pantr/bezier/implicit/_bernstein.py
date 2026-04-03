"""Tensor-product Bernstein polynomial operations for implicit quadrature.

All functions are Numba nopython-compiled and operate on raw coefficient arrays.
Polynomials are stored as dense numpy arrays of shape ``(n0+1, n1+1, ...)`` where
``ni`` is the degree in direction ``i``.

Main exports:

- :func:`_eval_bernstein_basis_1d` -- evaluate all basis functions at a single point.
- :func:`_collapse_2d` / :func:`_collapse_3d` -- collapse ND polynomial to 1D.
- :func:`_face_restrict_2d` / :func:`_face_restrict_3d` -- extract boundary face.
- :func:`_derivative_along_axis_2d` / :func:`_derivative_along_axis_3d` -- partial derivative.
- :func:`_elevated_derivative_along_axis_2d` / ``_3d`` -- derivative in same-degree basis.
- :func:`_eval_gradient_2d` / :func:`_eval_gradient_3d` -- evaluate gradient vector.
- :func:`_eval_bernstein_2d` / :func:`_eval_bernstein_3d` -- evaluate polynomial at a point.
- :func:`_normalize` -- scale coefficients to unit max-norm.
- :func:`_degree_elevate_axis` -- degree elevation along one axis.

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit quadrature module.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit

# ---------------------------------------------------------------------------
# Section A: 1D Bernstein basis evaluation
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _eval_bernstein_basis_1d(
    degree: int,
    x: float,
) -> npt.NDArray[np.float64]:
    """Evaluate all 1D Bernstein basis functions of given degree at *x*.

    Uses the numerically stable recurrence relation:
    ``B[0] = (1-x)^n``, ``B[i] = B[i-1] * (n-i+1)/i * x/(1-x)``.

    Args:
        degree (int): Polynomial degree (>= 0).
        x (float): Parameter value in [0, 1].

    Returns:
        npt.NDArray[np.float64]: Basis values of shape ``(degree + 1,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = degree
    basis = np.empty(n + 1, dtype=np.float64)

    if n == 0:
        basis[0] = 1.0
        return basis

    # Handle boundary cases exactly.
    if x == 0.0:
        basis[:] = 0.0
        basis[0] = 1.0
        return basis
    if x == 1.0:
        basis[:] = 0.0
        basis[n] = 1.0
        return basis

    s = 1.0 - x
    basis[0] = s**n
    ratio = x / s
    for i in range(1, n + 1):
        basis[i] = basis[i - 1] * (float(n - i + 1) / float(i)) * ratio

    return basis


# ---------------------------------------------------------------------------
# Section B: Collapse (evaluate tangential directions, keep height direction)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _collapse_2d(
    coeffs: npt.NDArray[np.float64],
    k: int,
    x_tang: float,
) -> npt.NDArray[np.float64]:
    """Collapse a 2D TP Bernstein polynomial to 1D along axis *k*.

    Evaluates the Bernstein basis in the tangential direction (the one that
    is NOT *k*) at *x_tang*, and contracts the coefficient array along that
    direction, leaving a 1D polynomial in direction *k*.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array of shape
            ``(n0+1, n1+1)``.
        k (int): Height direction (0 or 1) to keep.
        x_tang (float): Parameter value for the tangential direction.

    Returns:
        npt.NDArray[np.float64]: 1D coefficients of shape ``(nk+1,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0, n1 = coeffs.shape[0] - 1, coeffs.shape[1] - 1

    if k == 0:
        # Keep axis 0, contract axis 1 at x_tang.
        basis = _eval_bernstein_basis_1d(n1, x_tang)
        out = np.zeros(n0 + 1, dtype=np.float64)
        for i0 in range(n0 + 1):
            s = 0.0
            for i1 in range(n1 + 1):
                s += coeffs[i0, i1] * basis[i1]
            out[i0] = s
        return out
    else:
        # Keep axis 1, contract axis 0 at x_tang.
        basis = _eval_bernstein_basis_1d(n0, x_tang)
        out = np.zeros(n1 + 1, dtype=np.float64)
        for i1 in range(n1 + 1):
            s = 0.0
            for i0 in range(n0 + 1):
                s += coeffs[i0, i1] * basis[i0]
            out[i1] = s
        return out


@nb_jit(nopython=True, cache=True)
def _collapse_3d(
    coeffs: npt.NDArray[np.float64],
    k: int,
    x_tang: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Collapse a 3D TP Bernstein polynomial to 1D along axis *k*.

    Evaluates the Bernstein bases in the two tangential directions at the
    corresponding components of *x_tang*, and contracts the coefficient array,
    leaving a 1D polynomial in direction *k*.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array of shape
            ``(n0+1, n1+1, n2+1)``.
        k (int): Height direction (0, 1, or 2) to keep.
        x_tang (npt.NDArray[np.float64]): Parameter values of shape ``(2,)``
            for the two tangential directions, ordered by increasing axis
            index (skipping *k*).

    Returns:
        npt.NDArray[np.float64]: 1D coefficients of shape ``(nk+1,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0 = coeffs.shape[0] - 1
    n1 = coeffs.shape[1] - 1
    n2 = coeffs.shape[2] - 1

    if k == 0:
        # Keep axis 0, contract axes 1 and 2.
        basis1 = _eval_bernstein_basis_1d(n1, x_tang[0])
        basis2 = _eval_bernstein_basis_1d(n2, x_tang[1])
        out = np.zeros(n0 + 1, dtype=np.float64)
        for i0 in range(n0 + 1):
            s = 0.0
            for i1 in range(n1 + 1):
                b1 = basis1[i1]
                for i2 in range(n2 + 1):
                    s += coeffs[i0, i1, i2] * b1 * basis2[i2]
            out[i0] = s
        return out
    elif k == 1:
        # Keep axis 1, contract axes 0 and 2.
        basis0 = _eval_bernstein_basis_1d(n0, x_tang[0])
        basis2 = _eval_bernstein_basis_1d(n2, x_tang[1])
        out = np.zeros(n1 + 1, dtype=np.float64)
        for i1 in range(n1 + 1):
            s = 0.0
            for i0 in range(n0 + 1):
                b0 = basis0[i0]
                for i2 in range(n2 + 1):
                    s += coeffs[i0, i1, i2] * b0 * basis2[i2]
            out[i1] = s
        return out
    else:
        # Keep axis 2, contract axes 0 and 1.
        basis0 = _eval_bernstein_basis_1d(n0, x_tang[0])
        basis1 = _eval_bernstein_basis_1d(n1, x_tang[1])
        out = np.zeros(n2 + 1, dtype=np.float64)
        for i2 in range(n2 + 1):
            s = 0.0
            for i0 in range(n0 + 1):
                b0 = basis0[i0]
                for i1 in range(n1 + 1):
                    s += coeffs[i0, i1, i2] * b0 * basis1[i1]
            out[i2] = s
        return out


# ---------------------------------------------------------------------------
# Section C: Face restriction (extract x_k=0 or x_k=1 boundary)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _face_restrict_2d(
    coeffs: npt.NDArray[np.float64],
    k: int,
    side: int,
) -> npt.NDArray[np.float64]:
    """Extract a face from a 2D TP Bernstein polynomial.

    For Bernstein basis, ``B^n_0(0) = 1`` and all others are 0, so the face
    at ``x_k = 0`` is the first slice along axis *k*. Similarly, the face at
    ``x_k = 1`` is the last slice.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array of shape
            ``(n0+1, n1+1)``.
        k (int): Axis to restrict (0 or 1).
        side (int): 0 for lower face (x_k=0), 1 for upper face (x_k=1).

    Returns:
        npt.NDArray[np.float64]: 1D coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if k == 0:
        idx = 0 if side == 0 else coeffs.shape[0] - 1
        return coeffs[idx, :].copy()
    else:
        idx = 0 if side == 0 else coeffs.shape[1] - 1
        return coeffs[:, idx].copy()


@nb_jit(nopython=True, cache=True)
def _face_restrict_3d(
    coeffs: npt.NDArray[np.float64],
    k: int,
    side: int,
) -> npt.NDArray[np.float64]:
    """Extract a face from a 3D TP Bernstein polynomial.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array of shape
            ``(n0+1, n1+1, n2+1)``.
        k (int): Axis to restrict (0, 1, or 2).
        side (int): 0 for lower face (x_k=0), 1 for upper face (x_k=1).

    Returns:
        npt.NDArray[np.float64]: 2D coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if k == 0:
        idx = 0 if side == 0 else coeffs.shape[0] - 1
        return coeffs[idx, :, :].copy()
    elif k == 1:
        idx = 0 if side == 0 else coeffs.shape[1] - 1
        return coeffs[:, idx, :].copy()
    else:
        idx = 0 if side == 0 else coeffs.shape[2] - 1
        return coeffs[:, :, idx].copy()


# ---------------------------------------------------------------------------
# Section D: Derivatives along an axis
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _derivative_along_axis_1d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute derivative of a 1D Bernstein polynomial.

    For degree *n*, the derivative has degree *n-1* with coefficients
    ``d[i] = n * (c[i+1] - c[i])`` for ``i = 0, ..., n-1``.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D coefficient array of shape ``(n+1,)``.

    Returns:
        npt.NDArray[np.float64]: Derivative coefficients of shape ``(n,)``.
            Returns empty array if degree is 0.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    deriv = np.empty(n, dtype=np.float64)
    fn = float(n)
    for i in range(n):
        deriv[i] = fn * (coeffs[i + 1] - coeffs[i])
    return deriv


@nb_jit(nopython=True, cache=True)
def _derivative_along_axis_2d(
    coeffs: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute partial derivative of a 2D TP Bernstein polynomial along axis *k*.

    The result has degree reduced by 1 in direction *k*.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array of shape
            ``(n0+1, n1+1)``.
        k (int): Differentiation direction (0 or 1).

    Returns:
        npt.NDArray[np.float64]: Derivative coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0, n1 = coeffs.shape[0] - 1, coeffs.shape[1] - 1

    if k == 0:
        if n0 == 0:
            return np.zeros((1, n1 + 1), dtype=np.float64)
        deriv = np.empty((n0, n1 + 1), dtype=np.float64)
        fn = float(n0)
        for i0 in range(n0):
            for i1 in range(n1 + 1):
                deriv[i0, i1] = fn * (coeffs[i0 + 1, i1] - coeffs[i0, i1])
        return deriv
    else:
        if n1 == 0:
            return np.zeros((n0 + 1, 1), dtype=np.float64)
        deriv = np.empty((n0 + 1, n1), dtype=np.float64)
        fn = float(n1)
        for i0 in range(n0 + 1):
            for i1 in range(n1):
                deriv[i0, i1] = fn * (coeffs[i0, i1 + 1] - coeffs[i0, i1])
        return deriv


@nb_jit(nopython=True, cache=True)
def _derivative_along_axis_3d(  # noqa: PLR0912
    coeffs: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute partial derivative of a 3D TP Bernstein polynomial along axis *k*.

    The result has degree reduced by 1 in direction *k*.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array of shape
            ``(n0+1, n1+1, n2+1)``.
        k (int): Differentiation direction (0, 1, or 2).

    Returns:
        npt.NDArray[np.float64]: Derivative coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0 = coeffs.shape[0] - 1
    n1 = coeffs.shape[1] - 1
    n2 = coeffs.shape[2] - 1

    if k == 0:
        if n0 == 0:
            return np.zeros((1, n1 + 1, n2 + 1), dtype=np.float64)
        deriv = np.empty((n0, n1 + 1, n2 + 1), dtype=np.float64)
        fn = float(n0)
        for i0 in range(n0):
            for i1 in range(n1 + 1):
                for i2 in range(n2 + 1):
                    deriv[i0, i1, i2] = fn * (coeffs[i0 + 1, i1, i2] - coeffs[i0, i1, i2])
        return deriv
    elif k == 1:
        if n1 == 0:
            return np.zeros((n0 + 1, 1, n2 + 1), dtype=np.float64)
        deriv = np.empty((n0 + 1, n1, n2 + 1), dtype=np.float64)
        fn = float(n1)
        for i0 in range(n0 + 1):
            for i1 in range(n1):
                for i2 in range(n2 + 1):
                    deriv[i0, i1, i2] = fn * (coeffs[i0, i1 + 1, i2] - coeffs[i0, i1, i2])
        return deriv
    else:
        if n2 == 0:
            return np.zeros((n0 + 1, n1 + 1, 1), dtype=np.float64)
        deriv = np.empty((n0 + 1, n1 + 1, n2), dtype=np.float64)
        fn = float(n2)
        for i0 in range(n0 + 1):
            for i1 in range(n1 + 1):
                for i2 in range(n2):
                    deriv[i0, i1, i2] = fn * (coeffs[i0, i1, i2 + 1] - coeffs[i0, i1, i2])
        return deriv


# ---------------------------------------------------------------------------
# Section E: Elevated derivative (derivative in same-degree basis)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _degree_elevate_1d(
    coeffs: npt.NDArray[np.float64],
    increment: int,
) -> npt.NDArray[np.float64]:
    """Degree-elevate a 1D Bernstein polynomial by *increment* degrees.

    Uses the formula: ``Q[i] = sum_{j} C(p,j)*C(t,i-j)/C(p+t,i) * P[j]``
    applied iteratively one degree at a time for simplicity.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D coefficient array of shape ``(p+1,)``.
        increment (int): Number of degrees to add (>= 0).

    Returns:
        npt.NDArray[np.float64]: Elevated coefficients of shape ``(p+increment+1,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if increment <= 0:
        return coeffs.copy()

    cur = coeffs.copy()
    for _step in range(increment):
        p = len(cur) - 1
        new_p = p + 1
        elev = np.empty(new_p + 1, dtype=np.float64)
        elev[0] = cur[0]
        elev[new_p] = cur[p]
        for i in range(1, new_p):
            alpha = float(i) / float(new_p)
            elev[i] = alpha * cur[i - 1] + (1.0 - alpha) * cur[i]
        cur = elev
    return cur


@nb_jit(nopython=True, cache=True)
def _elevated_derivative_along_axis_2d(
    coeffs: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute the elevated derivative of a 2D TP Bernstein polynomial along axis *k*.

    This is the derivative followed by degree elevation back to the original
    degree in direction *k*. Equivalent to ``n * (c[...,i+1,...] - c[...,i,...])``
    with subsequent elevation, resulting in a polynomial of the same degree as
    the input.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array of shape
            ``(n0+1, n1+1)``.
        k (int): Differentiation direction (0 or 1).

    Returns:
        npt.NDArray[np.float64]: Elevated derivative coefficient array
            (same shape as input).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # First compute the derivative (reduces degree by 1 in direction k).
    deriv = _derivative_along_axis_2d(coeffs, k)

    # Then degree-elevate back by 1 in direction k.
    if k == 0:
        n0_d = deriv.shape[0] - 1
        n1 = deriv.shape[1] - 1
        new_n0 = n0_d + 1
        out = np.empty((new_n0 + 1, n1 + 1), dtype=np.float64)
        for i1 in range(n1 + 1):
            col = np.empty(n0_d + 1, dtype=np.float64)
            for i0 in range(n0_d + 1):
                col[i0] = deriv[i0, i1]
            elev = _degree_elevate_1d(col, 1)
            for i0 in range(new_n0 + 1):
                out[i0, i1] = elev[i0]
        return out
    else:
        n0 = deriv.shape[0] - 1
        n1_d = deriv.shape[1] - 1
        new_n1 = n1_d + 1
        out = np.empty((n0 + 1, new_n1 + 1), dtype=np.float64)
        for i0 in range(n0 + 1):
            row = np.empty(n1_d + 1, dtype=np.float64)
            for i1 in range(n1_d + 1):
                row[i1] = deriv[i0, i1]
            elev = _degree_elevate_1d(row, 1)
            for i1 in range(new_n1 + 1):
                out[i0, i1] = elev[i1]
        return out


@nb_jit(nopython=True, cache=True)
def _elevated_derivative_along_axis_3d(  # noqa: PLR0912
    coeffs: npt.NDArray[np.float64],
    k: int,
) -> npt.NDArray[np.float64]:
    """Compute the elevated derivative of a 3D TP Bernstein polynomial along axis *k*.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array of shape
            ``(n0+1, n1+1, n2+1)``.
        k (int): Differentiation direction (0, 1, or 2).

    Returns:
        npt.NDArray[np.float64]: Elevated derivative coefficient array
            (same shape as input).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    deriv = _derivative_along_axis_3d(coeffs, k)
    s0 = deriv.shape[0]
    s1 = deriv.shape[1]
    s2 = deriv.shape[2]

    if k == 0:
        out = np.empty((s0 + 1, s1, s2), dtype=np.float64)
        for i1 in range(s1):
            for i2 in range(s2):
                col = np.empty(s0, dtype=np.float64)
                for i0 in range(s0):
                    col[i0] = deriv[i0, i1, i2]
                elev = _degree_elevate_1d(col, 1)
                for i0 in range(s0 + 1):
                    out[i0, i1, i2] = elev[i0]
        return out
    elif k == 1:
        out = np.empty((s0, s1 + 1, s2), dtype=np.float64)
        for i0 in range(s0):
            for i2 in range(s2):
                col = np.empty(s1, dtype=np.float64)
                for i1 in range(s1):
                    col[i1] = deriv[i0, i1, i2]
                elev = _degree_elevate_1d(col, 1)
                for i1 in range(s1 + 1):
                    out[i0, i1, i2] = elev[i1]
        return out
    else:
        out = np.empty((s0, s1, s2 + 1), dtype=np.float64)
        for i0 in range(s0):
            for i1 in range(s1):
                col = np.empty(s2, dtype=np.float64)
                for i2 in range(s2):
                    col[i2] = deriv[i0, i1, i2]
                elev = _degree_elevate_1d(col, 1)
                for i2 in range(s2 + 1):
                    out[i0, i1, i2] = elev[i2]
        return out


# ---------------------------------------------------------------------------
# Section F: Polynomial evaluation and gradient
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _eval_bernstein_2d(
    coeffs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> float:
    """Evaluate a 2D TP Bernstein polynomial at point *x*.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array of shape
            ``(n0+1, n1+1)``.
        x (npt.NDArray[np.float64]): Point of shape ``(2,)``.

    Returns:
        float: Polynomial value at *x*.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0, n1 = coeffs.shape[0] - 1, coeffs.shape[1] - 1
    b0 = _eval_bernstein_basis_1d(n0, x[0])
    b1 = _eval_bernstein_basis_1d(n1, x[1])
    result = 0.0
    for i0 in range(n0 + 1):
        v0 = b0[i0]
        for i1 in range(n1 + 1):
            result += coeffs[i0, i1] * v0 * b1[i1]
    return result


@nb_jit(nopython=True, cache=True)
def _eval_bernstein_3d(
    coeffs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> float:
    """Evaluate a 3D TP Bernstein polynomial at point *x*.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array of shape
            ``(n0+1, n1+1, n2+1)``.
        x (npt.NDArray[np.float64]): Point of shape ``(3,)``.

    Returns:
        float: Polynomial value at *x*.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0 = coeffs.shape[0] - 1
    n1 = coeffs.shape[1] - 1
    n2 = coeffs.shape[2] - 1
    b0 = _eval_bernstein_basis_1d(n0, x[0])
    b1 = _eval_bernstein_basis_1d(n1, x[1])
    b2 = _eval_bernstein_basis_1d(n2, x[2])
    result = 0.0
    for i0 in range(n0 + 1):
        v0 = b0[i0]
        for i1 in range(n1 + 1):
            v01 = v0 * b1[i1]
            for i2 in range(n2 + 1):
                result += coeffs[i0, i1, i2] * v01 * b2[i2]
    return result


@nb_jit(nopython=True, cache=True)
def _eval_gradient_2d(
    coeffs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate the gradient of a 2D TP Bernstein polynomial at point *x*.

    Computes partial derivatives by first taking the Bernstein derivative
    coefficients along each axis, then evaluating the resulting polynomial.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array of shape
            ``(n0+1, n1+1)``.
        x (npt.NDArray[np.float64]): Point of shape ``(2,)``.

    Returns:
        npt.NDArray[np.float64]: Gradient vector of shape ``(2,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    grad = np.empty(2, dtype=np.float64)

    # d/dx0: derivative along axis 0, then evaluate.
    d0 = _derivative_along_axis_2d(coeffs, 0)
    grad[0] = _eval_bernstein_2d(d0, x)

    # d/dx1: derivative along axis 1, then evaluate.
    d1 = _derivative_along_axis_2d(coeffs, 1)
    grad[1] = _eval_bernstein_2d(d1, x)

    return grad


@nb_jit(nopython=True, cache=True)
def _eval_gradient_3d(
    coeffs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate the gradient of a 3D TP Bernstein polynomial at point *x*.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array of shape
            ``(n0+1, n1+1, n2+1)``.
        x (npt.NDArray[np.float64]): Point of shape ``(3,)``.

    Returns:
        npt.NDArray[np.float64]: Gradient vector of shape ``(3,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    grad = np.empty(3, dtype=np.float64)

    d0 = _derivative_along_axis_3d(coeffs, 0)
    grad[0] = _eval_bernstein_3d(d0, x)

    d1 = _derivative_along_axis_3d(coeffs, 1)
    grad[1] = _eval_bernstein_3d(d1, x)

    d2 = _derivative_along_axis_3d(coeffs, 2)
    grad[2] = _eval_bernstein_3d(d2, x)

    return grad


# ---------------------------------------------------------------------------
# Section G: Normalization
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _normalize_1d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Normalize a 1D Bernstein polynomial by its maximum absolute coefficient.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D coefficient array.

    Returns:
        npt.NDArray[np.float64]: Normalized coefficients. Returns a copy of
            the input if all coefficients are zero.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    max_val = 0.0
    for i in range(len(coeffs)):
        v = abs(coeffs[i])
        max_val = max(max_val, v)
    if max_val == 0.0:
        return coeffs.copy()
    out = np.empty(len(coeffs), dtype=np.float64)
    inv = 1.0 / max_val
    for i in range(len(coeffs)):
        out[i] = coeffs[i] * inv
    return out


@nb_jit(nopython=True, cache=True)
def _normalize_2d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Normalize a 2D Bernstein polynomial by its maximum absolute coefficient.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array.

    Returns:
        npt.NDArray[np.float64]: Normalized coefficients.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    max_val = 0.0
    for i0 in range(coeffs.shape[0]):
        for i1 in range(coeffs.shape[1]):
            v = abs(coeffs[i0, i1])
            max_val = max(max_val, v)
    if max_val == 0.0:
        return coeffs.copy()
    out = np.empty_like(coeffs)
    inv = 1.0 / max_val
    for i0 in range(coeffs.shape[0]):
        for i1 in range(coeffs.shape[1]):
            out[i0, i1] = coeffs[i0, i1] * inv
    return out


@nb_jit(nopython=True, cache=True)
def _normalize_3d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Normalize a 3D Bernstein polynomial by its maximum absolute coefficient.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array.

    Returns:
        npt.NDArray[np.float64]: Normalized coefficients.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    max_val = 0.0
    for i0 in range(coeffs.shape[0]):
        for i1 in range(coeffs.shape[1]):
            for i2 in range(coeffs.shape[2]):
                v = abs(coeffs[i0, i1, i2])
                max_val = max(max_val, v)
    if max_val == 0.0:
        return coeffs.copy()
    out = np.empty_like(coeffs)
    inv = 1.0 / max_val
    for i0 in range(coeffs.shape[0]):
        for i1 in range(coeffs.shape[1]):
            for i2 in range(coeffs.shape[2]):
                out[i0, i1, i2] = coeffs[i0, i1, i2] * inv
    return out


# ---------------------------------------------------------------------------
# Section H: Degree elevation along one axis (for resultant computation)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _degree_elevate_axis_2d(
    coeffs: npt.NDArray[np.float64],
    k: int,
    increment: int,
) -> npt.NDArray[np.float64]:
    """Degree-elevate a 2D TP Bernstein polynomial along axis *k*.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array.
        k (int): Axis to elevate (0 or 1).
        increment (int): Number of degrees to add.

    Returns:
        npt.NDArray[np.float64]: Elevated coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if increment <= 0:
        return coeffs.copy()

    s0, s1 = coeffs.shape

    if k == 0:
        # Elevate each "column" (fixed i1) along axis 0.
        new_s0 = s0 + increment
        out = np.empty((new_s0, s1), dtype=np.float64)
        for i1 in range(s1):
            col = np.empty(s0, dtype=np.float64)
            for i0 in range(s0):
                col[i0] = coeffs[i0, i1]
            elev = _degree_elevate_1d(col, increment)
            for i0 in range(new_s0):
                out[i0, i1] = elev[i0]
        return out
    else:
        # Elevate each "row" (fixed i0) along axis 1.
        new_s1 = s1 + increment
        out = np.empty((s0, new_s1), dtype=np.float64)
        for i0 in range(s0):
            row = np.empty(s1, dtype=np.float64)
            for i1 in range(s1):
                row[i1] = coeffs[i0, i1]
            elev = _degree_elevate_1d(row, increment)
            for i1 in range(new_s1):
                out[i0, i1] = elev[i1]
        return out


@nb_jit(nopython=True, cache=True)
def _degree_elevate_axis_3d(  # noqa: PLR0912
    coeffs: npt.NDArray[np.float64],
    k: int,
    increment: int,
) -> npt.NDArray[np.float64]:
    """Degree-elevate a 3D TP Bernstein polynomial along axis *k*.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array.
        k (int): Axis to elevate (0, 1, or 2).
        increment (int): Number of degrees to add.

    Returns:
        npt.NDArray[np.float64]: Elevated coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if increment <= 0:
        return coeffs.copy()

    s0, s1, s2 = coeffs.shape

    if k == 0:
        new_s0 = s0 + increment
        out = np.empty((new_s0, s1, s2), dtype=np.float64)
        for i1 in range(s1):
            for i2 in range(s2):
                col = np.empty(s0, dtype=np.float64)
                for i0 in range(s0):
                    col[i0] = coeffs[i0, i1, i2]
                elev = _degree_elevate_1d(col, increment)
                for i0 in range(new_s0):
                    out[i0, i1, i2] = elev[i0]
        return out
    elif k == 1:
        new_s1 = s1 + increment
        out = np.empty((s0, new_s1, s2), dtype=np.float64)
        for i0 in range(s0):
            for i2 in range(s2):
                col = np.empty(s1, dtype=np.float64)
                for i1 in range(s1):
                    col[i1] = coeffs[i0, i1, i2]
                elev = _degree_elevate_1d(col, increment)
                for i1 in range(new_s1):
                    out[i0, i1, i2] = elev[i1]
        return out
    else:
        new_s2 = s2 + increment
        out = np.empty((s0, s1, new_s2), dtype=np.float64)
        for i0 in range(s0):
            for i1 in range(s1):
                col = np.empty(s2, dtype=np.float64)
                for i2 in range(s2):
                    col[i2] = coeffs[i0, i1, i2]
                elev = _degree_elevate_1d(col, increment)
                for i2 in range(new_s2):
                    out[i0, i1, i2] = elev[i2]
        return out


# ---------------------------------------------------------------------------
# Section I: Squared L2 norm and degree auto-reduction
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
def _squared_l2_norm_1d(coeffs: npt.NDArray[np.float64]) -> float:
    """Compute the squared L2 norm of a 1D Bernstein polynomial on [0, 1].

    Uses the Bernstein basis Gram matrix: ``<B^n_i, B^n_j> = C(n,i)*C(n,j) / ((2n+1)*C(2n,i+j))``.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients of length ``n+1``.

    Returns:
        float: Squared L2 norm ``integral_0^1 p(x)^2 dx``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    result = 0.0
    inv_2n1 = 1.0 / float(2 * n + 1)
    for i in range(n + 1):
        ci = coeffs[i]
        bi = _bincoeff(n, i)
        for j in range(n + 1):
            cj = coeffs[j]
            bj = _bincoeff(n, j)
            bij = _bincoeff(2 * n, i + j)
            if bij > 0.0:
                result += ci * cj * bi * bj * inv_2n1 / bij
    return result


@nb_jit(nopython=True, cache=True)
def _degree_reduce_1d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reduce a 1D Bernstein polynomial by one degree via least-squares.

    Uses the simple averaging formula: for a degree-n polynomial reduced to
    degree n-1, the coefficients are approximately:
    ``q[i] = ((n-i)*c[i] + (i+1)*c[i+1]) / (n)`` ... but a cleaner approach
    is to use the pseudo-inverse of the degree elevation matrix.

    For simplicity, uses the formula from Eck (1993): the best L2 approximation
    of degree n by degree n-1 in Bernstein form.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients of length ``n+1``.

    Returns:
        npt.NDArray[np.float64]: Reduced coefficients of length ``n``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    if n <= 0:
        return coeffs.copy()

    # The degree elevation matrix E maps degree-(n-1) to degree-n:
    # E[i, j] = alpha if j == i, (1-alpha) if j == i-1, where alpha = i/n.
    # The least-squares reduction is E^+ * coeffs.
    # For the bidiagonal E, this can be solved efficiently.
    # Forward sweep: q[0] = c[0], q[i] = (c[i] - (i/n)*q[i-1]) / (1 - i/n).
    # Backward sweep: r[n-1] = c[n], r[i] = (c[i+1] - (1-(i+1)/n)*r[i+1]) / ((i+1)/n).
    # Average: reduced[i] = (q[i] + r[i]) / 2.

    fn = float(n)

    # Forward sweep.
    q = np.empty(n, dtype=np.float64)
    q[0] = coeffs[0]
    for i in range(1, n):
        alpha = float(i) / fn
        q[i] = (coeffs[i] - alpha * q[i - 1]) / (1.0 - alpha)

    # Backward sweep.
    r = np.empty(n, dtype=np.float64)
    r[n - 1] = coeffs[n]
    for i in range(n - 2, -1, -1):
        alpha = float(i + 1) / fn
        r[i] = (coeffs[i + 1] - (1.0 - alpha) * r[i + 1]) / alpha

    # Average.
    reduced = np.empty(n, dtype=np.float64)
    for i in range(n):
        reduced[i] = 0.5 * (q[i] + r[i])

    return reduced


@nb_jit(nopython=True, cache=True)
def _auto_reduce_1d(
    coeffs: npt.NDArray[np.float64],
    tol: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Attempt to reduce the degree of a 1D Bernstein polynomial.

    Iteratively reduces degree by 1 as long as the L2 residual (reduce then
    re-elevate minus original) is below *tol* times the original L2 norm.
    Matches algoim's ``autoReduction`` strategy.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients.
        tol (float): Relative L2 tolerance for accepting reduction.

    Returns:
        npt.NDArray[np.float64]: Possibly degree-reduced coefficients.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    current = coeffs.copy()
    orig_norm_sq = _squared_l2_norm_1d(current)
    if orig_norm_sq <= 0.0:
        return current

    orig_norm = np.sqrt(abs(orig_norm_sq))

    while len(current) > 2:  # noqa: PLR2004
        n = len(current) - 1
        # Reduce by 1.
        reduced = _degree_reduce_1d(current)
        # Re-elevate to original degree.
        re_elevated = _degree_elevate_1d(reduced, 1)
        # Compute residual.
        residual = np.empty(n + 1, dtype=np.float64)
        for i in range(n + 1):
            residual[i] = re_elevated[i] - current[i]
        res_norm_sq = _squared_l2_norm_1d(residual)
        res_norm = np.sqrt(abs(res_norm_sq))

        if res_norm < tol * orig_norm:
            current = reduced
        else:
            break

    return current


@nb_jit(nopython=True, cache=True)
def _auto_reduce_2d(  # noqa: PLR0912
    coeffs: npt.NDArray[np.float64],
    tol: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Attempt to reduce the degree of a 2D TP Bernstein polynomial.

    Tries reducing each axis independently, dimension by dimension.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D Bernstein coefficient array.
        tol (float): Relative L2 tolerance for accepting reduction.

    Returns:
        npt.NDArray[np.float64]: Possibly degree-reduced coefficients.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    current = coeffs.copy()
    changed = True
    while changed:
        changed = False
        s0, s1 = current.shape

        # Try reducing axis 0.
        if s0 > 2:  # noqa: PLR2004
            # Reduce each column along axis 0, then check residual.
            reduced_0 = np.empty((s0 - 1, s1), dtype=np.float64)
            for j in range(s1):
                col = np.empty(s0, dtype=np.float64)
                for i in range(s0):
                    col[i] = current[i, j]
                red = _degree_reduce_1d(col)
                for i in range(s0 - 1):
                    reduced_0[i, j] = red[i]
            # Re-elevate.
            re_elev_0 = _degree_elevate_axis_2d(reduced_0, 0, 1)
            # Residual norm.
            res_sq = 0.0
            orig_sq = 0.0
            for i in range(s0):
                for j in range(s1):
                    res_sq += (re_elev_0[i, j] - current[i, j]) ** 2
                    orig_sq += current[i, j] ** 2
            if orig_sq > 0.0 and np.sqrt(res_sq) < tol * np.sqrt(orig_sq):
                current = reduced_0
                changed = True
                continue

        s0, s1 = current.shape
        # Try reducing axis 1.
        if s1 > 2:  # noqa: PLR2004
            reduced_1 = np.empty((s0, s1 - 1), dtype=np.float64)
            for i in range(s0):
                row = np.empty(s1, dtype=np.float64)
                for j in range(s1):
                    row[j] = current[i, j]
                red = _degree_reduce_1d(row)
                for j in range(s1 - 1):
                    reduced_1[i, j] = red[j]
            re_elev_1 = _degree_elevate_axis_2d(reduced_1, 1, 1)
            res_sq = 0.0
            orig_sq = 0.0
            for i in range(s0):
                for j in range(s1):
                    res_sq += (re_elev_1[i, j] - current[i, j]) ** 2
                    orig_sq += current[i, j] ** 2
            if orig_sq > 0.0 and np.sqrt(res_sq) < tol * np.sqrt(orig_sq):
                current = reduced_1
                changed = True
                continue

        break

    return current


# ---------------------------------------------------------------------------
# Section J: Square-free factoring via monomial GCD
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _bernstein_to_monomial_1d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert 1D Bernstein coefficients to monomial (power) basis.

    For degree n: ``p(x) = sum_i c_i B^n_i(x) = sum_j a_j x^j``.

    Args:
        coeffs (npt.NDArray[np.float64]): Bernstein coefficients of length ``n+1``.

    Returns:
        npt.NDArray[np.float64]: Monomial coefficients (ascending power) of length ``n+1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    # Build Bernstein-to-monomial matrix T.
    # T[j, i] = C(n, i) * C(n-i, j-i) * (-1)^{j-i} for j >= i, else 0.
    mono = np.zeros(n + 1, dtype=np.float64)
    for j in range(n + 1):
        for i in range(j + 1):
            sign = 1.0 if (j - i) % 2 == 0 else -1.0
            t_ji = _bincoeff(n, i) * _bincoeff(n - i, j - i) * sign
            mono[j] += t_ji * coeffs[i]
    return mono


@nb_jit(nopython=True, cache=True)
def _monomial_to_bernstein_1d(
    mono: npt.NDArray[np.float64],
    degree: int,
) -> npt.NDArray[np.float64]:
    """Convert monomial (power) coefficients to Bernstein of given degree.

    Args:
        mono (npt.NDArray[np.float64]): Monomial coefficients (ascending power).
        degree (int): Target Bernstein degree (>= len(mono) - 1).

    Returns:
        npt.NDArray[np.float64]: Bernstein coefficients of length ``degree+1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = degree
    # M[i, j] = C(i, j) / C(n, j) for j <= i, else 0.
    bern = np.zeros(n + 1, dtype=np.float64)
    m = min(len(mono), n + 1)
    for i in range(n + 1):
        for j in range(min(i + 1, m)):
            cn = _bincoeff(n, j)
            if cn > 0.0:
                bern[i] += (_bincoeff(i, j) / cn) * mono[j]
    return bern


@nb_jit(nopython=True, cache=True)
def _monomial_gcd(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    tol: float,
) -> npt.NDArray[np.float64]:
    """Compute approximate GCD of two monomial polynomials via Euclidean algorithm.

    Uses a tolerance-based stopping criterion: stops when the remainder has
    all coefficients below ``tol`` times the leading coefficient of the input.

    Args:
        p (npt.NDArray[np.float64]): Monomial coefficients of p (ascending power).
        q (npt.NDArray[np.float64]): Monomial coefficients of q (ascending power).
        tol (float): Relative tolerance for stopping.

    Returns:
        npt.NDArray[np.float64]: Monomial coefficients of GCD (ascending power),
            normalized to monic.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """

    # Trim trailing zeros (find actual degree).
    def _trim(poly: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = len(poly)
        while n > 1 and abs(poly[n - 1]) < tol:
            n -= 1
        return poly[:n].copy()

    a = _trim(p)
    b = _trim(q)

    # Ensure deg(a) >= deg(b).
    if len(b) > len(a):
        a, b = b, a

    scale = 0.0
    for i in range(len(a)):
        scale = max(scale, abs(a[i]))
    if scale < 1e-300:  # noqa: PLR2004
        return np.ones(1, dtype=np.float64)

    abs_tol = tol * scale

    # Euclidean algorithm.
    for _iter in range(len(a)):
        b = _trim(b)

        # Check if b is effectively zero (remainder vanished → a is the GCD).
        b_max = 0.0
        for i in range(len(b)):
            b_max = max(b_max, abs(b[i]))
        if b_max < abs_tol:
            break

        if len(b) <= 1:
            # GCD is constant (degree 0) — polynomials are coprime.
            return np.ones(1, dtype=np.float64)

        # Polynomial long division remainder: a mod b.
        rem = a.copy()
        deg_a = len(rem) - 1
        deg_b = len(b) - 1
        for i in range(deg_a - deg_b, -1, -1):
            if abs(b[deg_b]) < 1e-300:  # noqa: PLR2004
                break
            coeff = rem[i + deg_b] / b[deg_b]
            for j in range(deg_b + 1):
                rem[i + j] -= coeff * b[j]
        a = b
        b = _trim(rem)

    # Normalize to monic.
    result = a.copy()
    lc = result[len(result) - 1]
    if abs(lc) > 1e-300:  # noqa: PLR2004
        for i in range(len(result)):
            result[i] /= lc

    return result


@nb_jit(nopython=True, cache=True)
def _monomial_div(
    p: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Divide monomial polynomial p by d (exact or approximate division).

    Args:
        p (npt.NDArray[np.float64]): Dividend (ascending power).
        d (npt.NDArray[np.float64]): Divisor (ascending power).

    Returns:
        npt.NDArray[np.float64]: Quotient (ascending power).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    deg_p = len(p) - 1
    deg_d = len(d) - 1
    if deg_d > deg_p:
        return p.copy()
    deg_q = deg_p - deg_d
    q = np.zeros(deg_q + 1, dtype=np.float64)
    rem = p.copy()
    for i in range(deg_q, -1, -1):
        if abs(d[deg_d]) < 1e-300:  # noqa: PLR2004
            break
        q[i] = rem[i + deg_d] / d[deg_d]
        for j in range(deg_d + 1):
            rem[i + j] -= q[i] * d[j]
    return q


@nb_jit(nopython=True, cache=True)
def _make_square_free_1d(
    coeffs: npt.NDArray[np.float64],
    tol: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """Remove repeated root factors from a 1D Bernstein polynomial.

    Computes ``p / gcd(p, p')`` in monomial form, then converts back to
    Bernstein. If the polynomial has no repeated roots, returns the original.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients.
        tol (float): Tolerance for GCD computation.

    Returns:
        npt.NDArray[np.float64]: Square-free Bernstein coefficients (possibly
            lower degree).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    if n <= 1:
        return coeffs.copy()

    # Convert to monomial form.
    p_mono = _bernstein_to_monomial_1d(coeffs)

    # Compute derivative in monomial form: a'[j] = (j+1) * a[j+1].
    dp_mono = np.empty(n, dtype=np.float64)
    for j in range(n):
        dp_mono[j] = float(j + 1) * p_mono[j + 1]

    # Compute GCD(p, p').
    gcd = _monomial_gcd(p_mono, dp_mono, tol)

    gcd_deg = len(gcd) - 1
    if gcd_deg == 0:
        # No repeated roots — return original.
        return coeffs.copy()

    # Divide p by GCD to get square-free part.
    sf_mono = _monomial_div(p_mono, gcd)

    # Convert back to Bernstein.
    sf_deg = len(sf_mono) - 1
    return _monomial_to_bernstein_1d(sf_mono, sf_deg)
