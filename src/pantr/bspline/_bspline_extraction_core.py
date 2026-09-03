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

Only the **Bézier** target has a kernel here. Lagrange and cardinal are this
operator post-multiplied by a change-of-basis matrix, which Layer 2 does with
:func:`numpy.matmul`, and the cardinal target additionally needs the
cardinal-interval scan.
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
    "_tabulate_Bspline_Bezier_1D_extraction_core",
]
