"""Monomial-to-Bernstein conversion utilities for implicit quadrature.

Converts polynomials from monomial (power) form on arbitrary domains to
tensor-product Bernstein form on [0,1]^d. This is the standard way to
prepare input for :class:`ImplicitQuadrature`.

Main exports:

- :func:`monomial_to_bernstein_2d` -- convert 2D monomial polynomial.
- :func:`monomial_to_bernstein_3d` -- convert 3D monomial polynomial.
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy import typing as npt


def _validate_degrees(
    mono_shape: tuple[int, ...],
    degrees: tuple[int, ...],
) -> None:
    """Validate that target degrees are >= monomial degrees.

    Args:
        mono_shape (tuple[int, ...]): Shape of the monomial coefficient array.
        degrees (tuple[int, ...]): Target Bernstein degrees.

    Raises:
        ValueError: If any target degree is less than the corresponding
            monomial degree.
    """
    for axis, (size, deg) in enumerate(zip(mono_shape, degrees, strict=True)):
        mono_deg = size - 1
        if deg < mono_deg:
            msg = f"Target degree {deg} in axis {axis} is less than monomial degree {mono_deg}."
            raise ValueError(msg)


def monomial_to_bernstein_2d(
    mono: npt.NDArray[np.float64],
    degrees: tuple[int, int],
    domain_lo: npt.NDArray[np.float64],
    domain_hi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert a 2D monomial polynomial to Bernstein form on a given domain.

    The monomial polynomial is ``phi(x, y) = sum_{i,j} mono[i, j] * x^i * y^j``
    defined on a rectangular domain ``[domain_lo, domain_hi]``. The result is
    a tensor-product Bernstein coefficient array on ``[0, 1]^2``.

    Args:
        mono (npt.NDArray[np.float64]): Monomial coefficient array where
            ``mono[i, j]`` is the coefficient of ``x^i * y^j``.
        degrees (tuple[int, int]): Target Bernstein degrees ``(deg_x, deg_y)``.
            Must be >= the monomial degree in each direction.
        domain_lo (npt.NDArray[np.float64]): Lower corner of the domain, shape ``(2,)``.
        domain_hi (npt.NDArray[np.float64]): Upper corner of the domain, shape ``(2,)``.

    Returns:
        npt.NDArray[np.float64]: Bernstein coefficient array of shape
            ``(deg_x + 1, deg_y + 1)``.

    Raises:
        ValueError: If any target degree is less than the corresponding
            monomial degree.

    Example:
        >>> import numpy as np
        >>> from pantr.bezier.implicit._convert_core import monomial_to_bernstein_2d
        >>> # phi(x, y) = x^2 + 4y^2 - 1 on [-1, 1]^2
        >>> mono = np.zeros((3, 3))
        >>> mono[2, 0] = 1.0   # x^2
        >>> mono[0, 2] = 4.0   # 4y^2
        >>> mono[0, 0] = -1.0  # -1
        >>> bern = monomial_to_bernstein_2d(
        ...     mono, (2, 2), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        ... )
    """
    dx, dy = degrees
    _validate_degrees(mono.shape, degrees)
    if domain_hi[0] <= domain_lo[0] or domain_hi[1] <= domain_lo[1]:
        msg = "domain_hi must be strictly greater than domain_lo in every dimension."
        raise ValueError(msg)
    lo_x, lo_y = float(domain_lo[0]), float(domain_lo[1])
    hx = float(domain_hi[0]) - lo_x
    hy = float(domain_hi[1]) - lo_y

    # Substitute x = lo_x + hx*t, y = lo_y + hy*s into monomial form.
    mapped = np.zeros((dx + 1, dy + 1))
    for ix in range(mono.shape[0]):
        for iy in range(mono.shape[1]):
            c = mono[ix, iy]
            if c == 0.0:
                continue
            for p in range(min(ix, dx) + 1):
                cx = comb(ix, p) * lo_x ** (ix - p) * hx**p
                for q in range(min(iy, dy) + 1):
                    cy = comb(iy, q) * lo_y ** (iy - q) * hy**q
                    mapped[p, q] += c * cx * cy

    return _m2b_mat(dx) @ mapped @ _m2b_mat(dy).T


def monomial_to_bernstein_3d(
    mono: npt.NDArray[np.float64],
    degrees: tuple[int, int, int],
    domain_lo: npt.NDArray[np.float64],
    domain_hi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert a 3D monomial polynomial to Bernstein form on a given domain.

    The monomial polynomial is ``phi(x, y, z) = sum_{i,j,k} mono[i, j, k] * x^i * y^j * z^k``
    defined on a rectangular domain ``[domain_lo, domain_hi]``.

    Args:
        mono (npt.NDArray[np.float64]): Monomial coefficient array where
            ``mono[i, j, k]`` is the coefficient of ``x^i * y^j * z^k``.
        degrees (tuple[int, int, int]): Target Bernstein degrees.
        domain_lo (npt.NDArray[np.float64]): Lower corner, shape ``(3,)``.
        domain_hi (npt.NDArray[np.float64]): Upper corner, shape ``(3,)``.

    Returns:
        npt.NDArray[np.float64]: Bernstein coefficient array of shape
            ``(deg_x + 1, deg_y + 1, deg_z + 1)``.

    Raises:
        ValueError: If any target degree is less than the corresponding
            monomial degree.
    """
    dx, dy, dz = degrees
    _validate_degrees(mono.shape, degrees)
    if domain_hi[0] <= domain_lo[0] or domain_hi[1] <= domain_lo[1] or domain_hi[2] <= domain_lo[2]:
        msg = "domain_hi must be strictly greater than domain_lo in every dimension."
        raise ValueError(msg)
    lo_x, lo_y, lo_z = float(domain_lo[0]), float(domain_lo[1]), float(domain_lo[2])
    hx = float(domain_hi[0]) - lo_x
    hy = float(domain_hi[1]) - lo_y
    hz = float(domain_hi[2]) - lo_z

    mapped = np.zeros((dx + 1, dy + 1, dz + 1))
    for ix in range(mono.shape[0]):
        for iy in range(mono.shape[1]):
            for iz in range(mono.shape[2]):
                c = mono[ix, iy, iz]
                if c == 0.0:
                    continue
                for p in range(min(ix, dx) + 1):
                    cx = comb(ix, p) * lo_x ** (ix - p) * hx**p
                    for q in range(min(iy, dy) + 1):
                        cy = comb(iy, q) * lo_y ** (iy - q) * hy**q
                        for r in range(min(iz, dz) + 1):
                            cz = comb(iz, r) * lo_z ** (iz - r) * hz**r
                            mapped[p, q, r] += c * cx * cy * cz

    mx, my, mz = _m2b_mat(dx), _m2b_mat(dy), _m2b_mat(dz)
    tmp1 = np.einsum("ip,pqr->iqr", mx, mapped)
    tmp2 = np.einsum("jq,iqr->ijr", my, tmp1)
    return np.asarray(np.einsum("kr,ijr->ijk", mz, tmp2), dtype=np.float64)


def _m2b_mat(n: int) -> npt.NDArray[np.float64]:
    """Monomial-to-Bernstein conversion matrix for degree *n*.

    ``M[i, j] = C(i, j) / C(n, j)`` for ``j <= i``, else 0.

    Args:
        n (int): Polynomial degree.

    Returns:
        npt.NDArray[np.float64]: Lower-triangular matrix of shape ``(n+1, n+1)``.
    """
    mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            mat[i, j] = comb(i, j) / comb(n, j)
    return mat
