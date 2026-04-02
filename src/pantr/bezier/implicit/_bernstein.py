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
