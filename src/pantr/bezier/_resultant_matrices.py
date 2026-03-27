"""Sylvester and Bezout matrices for Bernstein polynomials.

Provides :func:`_sylvester_matrix` and :func:`_bezout_matrix`, which construct
matrices whose determinants give the resultant of two univariate Bernstein
polynomials.  These are core building blocks for the dimension-elimination step
in the algoim-style implicit quadrature pipeline (R. I. Saye, *J. Comput. Phys.*
448, 110720, 2022).

- The **Sylvester matrix** applies to polynomials of *arbitrary* degrees and has
  size ``(p + q) x (p + q)``, where ``p`` and ``q`` are the polynomial degrees.
- The **Bezout matrix** applies to polynomials of *equal* degree ``n`` and has
  size ``n x n``.  It is symmetric and generally better conditioned than the
  Sylvester matrix for same-degree pairs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt


def _sylvester_matrix(
    a: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    out: npt.NDArray[np.floating[Any]] | None = None,
) -> npt.NDArray[np.floating[Any]]:
    r"""Build the Sylvester matrix for two Bernstein polynomials.

    Given Bernstein coefficient vectors ``a`` (degree *p*) and ``b``
    (degree *q*), the Sylvester matrix is the ``(p + q) x (p + q)`` matrix
    whose determinant equals the resultant of the two polynomials (up to a
    known constant involving binomial coefficients).

    The matrix entries are

    .. math::

        S_{i,\,j+i}   = a_j \frac{\binom{p}{j}}{\binom{p+q-1}{j+i}},
        \quad i = 0,\dots,q-1,\; j = 0,\dots,p

    .. math::

        S_{i+q,\,j+i} = b_j \frac{\binom{q}{j}}{\binom{p+q-1}{j+i}},
        \quad i = 0,\dots,p-1,\; j = 0,\dots,q

    with all other entries zero.

    Args:
        a (npt.NDArray[np.floating[Any]]): Bernstein coefficients of the first
            polynomial, shape ``(p + 1,)``.  Must have ``p >= 1``.
        b (npt.NDArray[np.floating[Any]]): Bernstein coefficients of the second
            polynomial, shape ``(q + 1,)``.  Must have ``q >= 1``.
        out (npt.NDArray[np.floating[Any]] | None): Optional pre-allocated output
            array of shape ``(p + q, p + q)`` and matching dtype.  If *None*,
            a new array is allocated.

    Returns:
        npt.NDArray[np.floating[Any]]: The Sylvester matrix, shape ``(p + q, p + q)``.

    Raises:
        ValueError: If input arrays are not 1-D, have non-floating dtype,
            have degree < 1, or if the ``out`` array has wrong shape / dtype /
            writability.
    """
    _validate_coeff_array(a, "a", min_len=2)
    _validate_coeff_array(b, "b", min_len=2)

    p = len(a) - 1  # degree of a
    q = len(b) - 1  # degree of b
    m = p + q  # matrix size
    dtype = np.result_type(a.dtype, b.dtype)

    out = _prepare_out(out, (m, m), dtype)
    out[:] = 0.0

    # Precompute binomial rows.
    bp = np.array([math.comb(p, j) for j in range(p + 1)], dtype=dtype)
    bq = np.array([math.comb(q, j) for j in range(q + 1)], dtype=dtype)
    inv_bpq = np.array([1.0 / math.comb(p + q - 1, k) for k in range(p + q)], dtype=dtype)

    # Upper block: out[i, j+i] = a[j]*binom(p,j) / binom(p+q-1, j+i)
    # for i in 0..q-1, j in 0..p.
    i_a = np.arange(q)[:, np.newaxis]  # (q, 1)
    j_a = np.arange(p + 1)[np.newaxis, :]  # (1, p+1)
    out[i_a, j_a + i_a] = (a * bp)[np.newaxis, :] * inv_bpq[j_a + i_a]

    # Lower block: out[i+q, j+i] = b[j]*binom(q,j) / binom(p+q-1, j+i)
    # for i in 0..p-1, j in 0..q.
    i_b = np.arange(p)[:, np.newaxis]  # (p, 1)
    j_b = np.arange(q + 1)[np.newaxis, :]  # (1, q+1)
    out[i_b + q, j_b + i_b] = (b * bq)[np.newaxis, :] * inv_bpq[j_b + i_b]

    return out


def _bezout_matrix(
    a: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    out: npt.NDArray[np.floating[Any]] | None = None,
) -> npt.NDArray[np.floating[Any]]:
    r"""Build the Bezout matrix for two Bernstein polynomials of equal degree.

    Given Bernstein coefficient vectors ``a`` and ``b`` of equal degree *n*,
    the Bezout matrix is the ``n x n`` symmetric matrix whose determinant
    equals the resultant of the two polynomials (up to a known constant).

    The matrix is built via the recurrence (Bini & Gemignani, 2004):

    .. math::

        B_{i-1,\,0}  = (a_i b_0 - a_0 b_i) \frac{n}{i},
        \quad i = 1,\dots,n

    .. math::

        B_{n-1,\,j}  = (a_n b_j - a_j b_n) \frac{n}{n - j},
        \quad j = 1,\dots,n-1

    .. math::

        B_{i-1,\,j}  = (a_i b_j - a_j b_i) \frac{n^2}{i\,(n - j)}
                       + B_{i,\,j-1} \frac{j\,(n - i)}{i\,(n - j)},
        \quad i = n-1,\dots,1,\; j = 1,\dots,i-1

    The result is then symmetrized: :math:`B_{i,j} = B_{j,i}` for ``j > i``.

    Args:
        a (npt.NDArray[np.floating[Any]]): Bernstein coefficients of the first
            polynomial, shape ``(n + 1,)``.  Must have ``n >= 1``.
        b (npt.NDArray[np.floating[Any]]): Bernstein coefficients of the second
            polynomial, shape ``(n + 1,)``.  Must have ``n >= 1``.
        out (npt.NDArray[np.floating[Any]] | None): Optional pre-allocated output
            array of shape ``(n, n)`` and matching dtype.  If *None*, a new
            array is allocated.

    Returns:
        npt.NDArray[np.floating[Any]]: The symmetric Bezout matrix, shape ``(n, n)``.

    Raises:
        ValueError: If input arrays are not 1-D, have non-floating dtype,
            have degree < 1, have different lengths, or if the ``out`` array
            has wrong shape / dtype / writability.
    """
    _validate_coeff_array(a, "a", min_len=2)
    _validate_coeff_array(b, "b", min_len=2)

    if len(a) != len(b):
        raise ValueError(
            f"Coefficient arrays must have equal length for the Bezout matrix. "
            f"Got len(a)={len(a)} and len(b)={len(b)}."
        )

    n = len(a) - 1  # polynomial degree
    dtype = np.result_type(a.dtype, b.dtype)

    out = _prepare_out(out, (n, n), dtype)
    out[:] = 0.0

    fn = dtype.type(n)

    # Precompute the antisymmetric product: D[i,j] = a[i]*b[j] - a[j]*b[i].
    ab = np.outer(a, b)
    D = ab - ab.T  # shape (n+1, n+1)

    # First column: out[i-1, 0] = D[i, 0] * n / i  for i = 1..n.
    idx = np.arange(1, n + 1, dtype=dtype)
    out[:, 0] = D[1 : n + 1, 0] * fn / idx

    # Last row: out[n-1, j] = D[n, j] * n / (n - j)  for j = 1..n-1.
    jdx = np.arange(1, n, dtype=dtype)
    out[n - 1, 1:n] = D[n, 1:n] * fn / (fn - jdx)

    # Interior (backwards recurrence over rows, vectorised over columns).
    # out[i-1, j] = D[i,j] * n^2 / (i*(n-j)) + out[i, j-1] * j*(n-i) / (i*(n-j))
    # The dependency out[i,j-1] means we cannot vectorise across j directly,
    # but we can rewrite the row as a first-order linear scan.
    for i in range(n - 1, 1, -1):
        # j runs from 1 to i-1 (inclusive).
        js = np.arange(1, i, dtype=dtype)
        inv_denom = 1.0 / (dtype.type(i) * (fn - js))
        src = D[i, 1:i] * fn * fn * inv_denom
        mult = js * (fn - dtype.type(i)) * inv_denom
        # Scan: out[i-1, j] = src[j-1] + mult[j-1] * out[i-1, j-1]
        # (here j-1 because js starts at 1 but arrays are 0-indexed).
        # out[i, 0] is already set (first column), used as the seed via out[i, j-1].
        row = np.empty(i - 1, dtype=dtype)
        row[0] = src[0] + mult[0] * out[i, 0]
        for k in range(1, i - 1):
            row[k] = src[k] + mult[k] * row[k - 1]
        out[i - 1, 1:i] = row

    # Symmetrise: copy lower triangle to upper triangle.
    il = np.tril_indices(n, -1)
    out[il[1], il[0]] = out[il[0], il[1]]

    return out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_coeff_array(
    arr: npt.NDArray[np.floating[Any]],
    name: str,
    min_len: int,
) -> None:
    """Validate a 1-D Bernstein coefficient array.

    Args:
        arr (npt.NDArray[np.floating[Any]]): Array to validate.
        name (str): Parameter name for error messages.
        min_len (int): Minimum required length (degree + 1).

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"`{name}` must be a numpy array, got {type(arr).__name__}.")
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be 1-D, got ndim={arr.ndim}.")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"`{name}` must have floating dtype, got {arr.dtype}.")
    if len(arr) < min_len:
        raise ValueError(
            f"`{name}` must have at least {min_len} coefficients "
            f"(degree >= {min_len - 1}), got {len(arr)}."
        )


def _prepare_out(
    out: npt.NDArray[np.floating[Any]] | None,
    shape: tuple[int, int],
    dtype: np.dtype[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Allocate or validate the output array.

    Args:
        out (npt.NDArray[np.floating[Any]] | None): Caller-supplied array, or *None*.
        shape (tuple[int, int]): Expected shape.
        dtype (np.dtype[np.floating[Any]]): Expected dtype.

    Returns:
        npt.NDArray[np.floating[Any]]: Ready-to-write output array.

    Raises:
        ValueError: If ``out`` has wrong shape, dtype, or is not writeable.
    """
    if out is None:
        return np.empty(shape, dtype=dtype)

    if out.shape != shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected {shape}.")
    if out.dtype != dtype:
        raise ValueError(f"Output array has dtype {out.dtype}, but expected {dtype}.")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable.")
    return out
