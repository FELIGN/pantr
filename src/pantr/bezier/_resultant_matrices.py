"""Sylvester and Bezout matrices for Bernstein polynomials.

Provides :func:`_sylvester_matrix` and :func:`_bezout_matrix`, which construct
matrices whose determinants give the resultant of two univariate Bernstein
polynomials.  These are core building blocks for the dimension-elimination step
in the algoim-style implicit quadrature pipeline (R. I. Saye, *J. Comput. Phys.*
448, 110720, 2022).

- The **Sylvester matrix** applies to polynomials of *arbitrary* degrees and has
  size ``(p + q) × (p + q)``, where ``p`` and ``q`` are the polynomial degrees.
- The **Bezout matrix** applies to polynomials of *equal* degree ``n`` and has
  size ``n × n``.  It is symmetric and generally better conditioned than the
  Sylvester matrix for same-degree pairs.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def _sylvester_matrix(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    out: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    r"""Build the Sylvester matrix for two Bernstein polynomials.

    Given Bernstein coefficient vectors ``a`` (degree *p*) and ``b``
    (degree *q*), the Sylvester matrix is the ``(p + q) × (p + q)`` matrix
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
        a (npt.NDArray[np.floating]): Bernstein coefficients of the first
            polynomial, shape ``(p + 1,)``.  Must have ``p >= 1``.
        b (npt.NDArray[np.floating]): Bernstein coefficients of the second
            polynomial, shape ``(q + 1,)``.  Must have ``q >= 1``.
        out (npt.NDArray[np.floating] | None): Optional pre-allocated output
            array of shape ``(p + q, p + q)`` and matching dtype.  If *None*,
            a new array is allocated.

    Returns:
        npt.NDArray[np.floating]: The Sylvester matrix, shape ``(p + q, p + q)``.

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

    # Upper block: Q-1 rows from polynomial a.
    for i in range(q):
        for j in range(p + 1):
            out[i, j + i] = a[j] * bp[j] * inv_bpq[j + i]

    # Lower block: P-1 rows from polynomial b.
    for i in range(p):
        for j in range(q + 1):
            out[i + q, j + i] = b[j] * bq[j] * inv_bpq[j + i]

    return out


def _bezout_matrix(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    out: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    r"""Build the Bezout matrix for two Bernstein polynomials of equal degree.

    Given Bernstein coefficient vectors ``a`` and ``b`` of equal degree *n*,
    the Bezout matrix is the ``n × n`` symmetric matrix whose determinant
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
        a (npt.NDArray[np.floating]): Bernstein coefficients of the first
            polynomial, shape ``(n + 1,)``.  Must have ``n >= 1``.
        b (npt.NDArray[np.floating]): Bernstein coefficients of the second
            polynomial, shape ``(n + 1,)``.  Must have ``n >= 1``.
        out (npt.NDArray[np.floating] | None): Optional pre-allocated output
            array of shape ``(n, n)`` and matching dtype.  If *None*, a new
            array is allocated.

    Returns:
        npt.NDArray[np.floating]: The symmetric Bezout matrix, shape ``(n, n)``.

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

    fn = float(n)

    # First column: i = 1..n → row i-1.
    for i in range(1, n + 1):
        out[i - 1, 0] = (a[i] * b[0] - a[0] * b[i]) * fn / i

    # Last row: j = 1..n-1.
    for j in range(1, n):
        out[n - 1, j] = (a[n] * b[j] - a[j] * b[n]) * fn / (n - j)

    # Interior (backwards recurrence).
    for i in range(n - 1, 0, -1):
        for j in range(1, i):
            out[i - 1, j] = (a[i] * b[j] - a[j] * b[i]) * fn * fn / (i * (n - j)) + out[
                i, j - 1
            ] * j * (n - i) / (i * (n - j))

    # Symmetrize upper triangle.
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i]

    return out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_coeff_array(
    arr: npt.NDArray[np.floating],
    name: str,
    min_len: int,
) -> None:
    """Validate a 1-D Bernstein coefficient array.

    Args:
        arr (npt.NDArray[np.floating]): Array to validate.
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
    out: npt.NDArray[np.floating] | None,
    shape: tuple[int, int],
    dtype: np.dtype[np.floating],
) -> npt.NDArray[np.floating]:
    """Allocate or validate the output array.

    Args:
        out (npt.NDArray[np.floating] | None): Caller-supplied array, or *None*.
        shape (tuple[int, int]): Expected shape.
        dtype (np.dtype[np.floating]): Expected dtype.

    Returns:
        npt.NDArray[np.floating]: Ready-to-write output array.

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
