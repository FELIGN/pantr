"""Numba kernels for the B-spline extraction operators.

Layer 3: the Bézier operator builder and the structural identity mask, with no
validation and no allocation the caller cannot predict.

Split out of :mod:`pantr.bspline._bspline_extraction`, which is Layer 2 and now
holds no Numba, so that :mod:`pantr.bspline._extraction_backend` can import these
kernels while Layer 2 imports the catalogue. That is the same shape
:mod:`pantr.basis._basis_core` and :mod:`pantr.change_basis._change_basis_core`
have, and it is what keeps the catalogue rule of
``design/cross_backend_types.md`` from needing a lazy import to break a cycle:
a catalogue imports the kernels it hands out, so the kernels cannot live in the
module that imports the catalogue.

The **Bézier** and **Lagrange** targets have kernels here; the cardinal one does
not, because it additionally needs the cardinal-interval scan, which is not ported.

Two of the four are Numba and two are not, and the split is deliberate. The Lagrange
pair is the Bézier operator post-multiplied by a change-of-basis matrix, and that
product is :func:`numpy.matmul` -- a BLAS ``gemm``, already compiled, and the thing
the C++ twin has to agree with. Rewriting it as a ``nopython`` loop would not speed
it up and would change the oracle's summation order, which is the one property
``design/backend_parity.md`` Rule 9 says a port must reproduce rather than improve.
``numpy.matmul`` is in any case unsupported for a rank-3 operand inside ``nopython``,
so the loop would have to be written out by hand.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit
from ._bspline_knots import (
    _get_multiplicity_of_first_knot_in_domain_impl,
    _get_unique_knots_and_multiplicity_impl,
)


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _tabulate_Bspline_Bezier_1D_extraction_core(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    r"""Core implementation to compute Bézier extraction operators, writing to output array.

    This function computes the extraction operators that transform Bernstein
    into B-spline basis functions for each interval.
    For each interval \( i \), the Bézier extraction operator \( C_i \) satisfies:

        \[
        N_i(x) = C_i @ B(ξ)
        \]

    where:
      - N_i(x) is the vector of B-spline basis functions nonzero on the interval \( i \),
        evaluated at \( x \),
      - \( B(ξ) \) is the vector of Bernstein basis functions on the reference interval \([0, 1]\),
        evaluated at \( ξ \),
      - \( C_i \) is the extraction matrix for interval \( i \),
      - \( x \) is the physical coordinate, \( ξ \) is the local (reference) referred to \([0, 1]\).

    Every arithmetic step runs in ``knots.dtype``, which is what ``one`` below is
    spelled ``dtype.type(1.0)`` for: a bare ``1.0`` would promote the whole
    combination to ``float64`` at ``float32`` storage. ``design/backend_parity.md``
    Rule 9 is the rule, and the C++ twin in
    ``cpp/include/pantr/bspline/extraction.hpp`` reproduces the same width.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.float32 | np.float64]): Output array where results will be written.
            Must have the correct shape (n_elems, degree+1, degree+1) and dtype matching knots
            (no validation performed inside this numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-validated inputs and assumes the output array has the
        correct shape and dtype. Inputs are assumed to be correct (no validation performed).
        For general use, call _tabulate_Bspline_Bezier_1D_extraction_impl instead.
    """
    unique_knots, mults = _get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=True
    )

    n_elems = len(unique_knots) - 1

    dtype = knots.dtype
    one = dtype.type(1.0)

    # Initialize identity matrix for every element.
    out.fill(0.0)
    out[:, : degree + 1, : degree + 1] = np.eye(degree + 1, dtype=dtype)

    mult = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)

    # If not open first knot, additional knot insertion is needed.
    if mult < (degree + 1):
        C = out[0]
        reg = degree - mult

        t = knots[degree]
        for r in range(reg):
            lcl_knots = knots[r:]
            for k in range(1, degree - r):
                alpha = (t - lcl_knots[k]) / (lcl_knots[k + degree - r] - lcl_knots[k])
                C[:, k - 1] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

    alphas = np.zeros(max(degree - 1, 0), dtype=dtype)  # degree 0: no insertion coefficients

    knt_id = degree
    mult = 0

    for elem_id in range(n_elems):
        knt_id += mult
        mult = mults[elem_id + 1]

        if mult >= degree:
            continue

        lcl_knots = knots[knt_id : knt_id + degree + 1]
        alphas[: degree - mult] = (lcl_knots[1] - lcl_knots[0]) / (
            lcl_knots[mult + 1 :] - lcl_knots[0]
        )

        C = out[elem_id]

        reg = degree - mult
        for r in range(1, reg + 1):
            s = mult + r
            for k in range(degree, s - 1, -1):
                alpha = alphas[k - s]
                C[:, k] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

            if elem_id < (n_elems - 1):
                out[elem_id + 1, reg - r : reg + 1, reg - r] = C[degree - r : degree + 1, degree]


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _bezier_structural_identity_mask_core(
    multiplicities: npt.NDArray[np.intp],
    degree: int,
    out: npt.NDArray[np.bool_],
) -> None:
    """Compute a per-element Bézier identity mask from knot multiplicities.

    Element ``e`` spanning ``[unique_knots[e], unique_knots[e+1]]`` has an
    identity Bézier extraction operator if and only if both boundary unique
    knots have multiplicity ``>= degree + 1``, meaning the element is already
    a Bézier patch (fully isolated from its neighbours).

    Args:
        multiplicities (npt.NDArray[np.intp]): Per-unique-knot multiplicity
            array of length ``n_elements + 1`` (in-domain unique knots
            including both endpoints).
        degree (int): B-spline degree.
        out (npt.NDArray[np.bool_]): Output boolean array of length
            ``n_elements = len(multiplicities) - 1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call
        ``spanwise_element_extraction._bezier_structural_identity_mask`` instead.
    """
    threshold = degree + 1
    n_elements = len(multiplicities) - 1
    for e in range(n_elements):
        out[e] = multiplicities[e] >= threshold and multiplicities[e + 1] >= threshold


def _tabulate_Bspline_Lagrange_1D_extraction_core(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    lagrange_to_bernstein: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    r"""Compute the Lagrange extraction operators, writing to the output array.

    ``A_e = C_e @ L``, with ``C_e`` the Bézier operator of interval ``e`` and ``L``
    the Lagrange-to-Bernstein matrix ``L[j, k] = B_j(x_k)`` of the same degree, so
    that \( N_e(x) = A_e @ Lag(ξ) \) for the Lagrange basis on the reference
    interval. :func:`numpy.matmul` broadcasts the ``(degree+1, degree+1)`` matrix
    over the leading interval axis, so the whole stack is one call.

    The matrix is an argument rather than something this builds: it depends only on
    ``(degree, lagrange_variant, dtype)``, :mod:`pantr.change_basis` caches it on
    exactly that key, and the variant is a :class:`~enum.StrEnum` that must not
    reach a kernel.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        lagrange_to_bernstein (npt.NDArray[np.float32 | np.float64]): The
            ``(degree+1, degree+1)`` change-of-basis matrix, in ``knots``' dtype.
        out (npt.NDArray[np.float32 | np.float64]): Output array of shape
            ``(n_elems, degree+1, degree+1)``, overwritten in full.

    Note:
        Not a Numba kernel, and the module docstring says why. Inputs are assumed to
        be correct (no validation performed). For general use, call
        ``_tabulate_Bspline_Lagrange_1D_extraction_impl`` instead.
    """
    _tabulate_Bspline_Bezier_1D_extraction_core(knots, degree, tol, out)
    # In place, and legally so: `numpy.matmul` detects the overlap between `out` and
    # its own first operand and buffers, which is what the shipped Layer 2 has always
    # relied on. The C++ twin does the same product over a private copy of the row it
    # is overwriting.
    np.matmul(out, lagrange_to_bernstein, out=out)


def _lagrange_structural_identity_mask_core(
    multiplicities: npt.NDArray[np.intp],
    degree: int,
    lagrange_to_bernstein: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.bool_],
) -> None:
    """Compute a per-element Lagrange identity mask.

    ``A_e = C_e @ L`` is the identity exactly when ``L`` is and ``C_e`` is, so the
    mask is the Bézier one when ``L`` is the identity and all-false otherwise. The
    comparison against the identity is exact, because the question is a verdict
    rather than a value: a matrix that misses the identity by an ulp produces an
    operator that is not the identity.

    ``L`` is the identity when every Lagrange node coincides with the Bernstein
    abscissa of the same index, which happens at degree 1 for the equispaced,
    Gauss-Lobatto-Legendre and second-kind Chebyshev families.

    Args:
        multiplicities (npt.NDArray[np.intp]): Per-unique-knot multiplicity array of
            length ``n_elements + 1``.
        degree (int): B-spline degree, at least 1.
        lagrange_to_bernstein (npt.NDArray[np.float32 | np.float64]): The
            ``(degree+1, degree+1)`` change-of-basis matrix.
        out (npt.NDArray[np.bool_]): Output boolean array of length
            ``n_elements = len(multiplicities) - 1``.

    Note:
        Not a Numba kernel, for the reason the module docstring gives of its sibling:
        it is the Python half of a pair whose C++ twin it must agree with, and the
        work is a whole-array comparison rather than a loop. Inputs are assumed to be
        correct (no validation performed). Degree 0 does not reach here, since
        :func:`pantr.change_basis.compute_lagrange_to_bernstein_1d` refuses it and
        there is no matrix to pass. For general use, call
        ``spanwise_element_extraction._lagrange_structural_identity_mask`` instead.
    """
    identity = np.eye(degree + 1, dtype=lagrange_to_bernstein.dtype)
    if np.array_equal(lagrange_to_bernstein, identity):
        _bezier_structural_identity_mask_core(multiplicities, degree, out)
        return
    out.fill(False)


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2
    n_elems = 1
    out_dummy = np.empty((n_elems, degree_dummy + 1, degree_dummy + 1), dtype=np.float64)

    # Warmup Bezier extraction core with float64
    _tabulate_Bspline_Bezier_1D_extraction_core(knots_dummy, degree_dummy, tol_dummy, out_dummy)

    # Warmup structural identity mask kernel
    mults_dummy = np.array([degree_dummy + 1, degree_dummy + 1], dtype=np.intp)
    mask_dummy = np.empty(1, dtype=np.bool_)
    _bezier_structural_identity_mask_core(mults_dummy, degree_dummy, mask_dummy)


__all__ = [
    "_bezier_structural_identity_mask_core",
    "_lagrange_structural_identity_mask_core",
    "_tabulate_Bspline_Bezier_1D_extraction_core",
    "_tabulate_Bspline_Lagrange_1D_extraction_core",
]
