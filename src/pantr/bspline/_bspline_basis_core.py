"""Layer 2 of the general-knot B-spline basis tabulation.

Validates what the kernels assume and never check -- shapes, dtypes, domain membership,
the width a mixed call computes at -- then asks
:mod:`pantr.bspline._basis_backend` which backend's kernel to run and hands it a
pre-validated buffer. The Bézier-like fast path is chosen here too, so both backends
take it on the same inputs.

The kernels themselves are in :mod:`pantr.bspline._bspline_basis_kernels`, which this
module no longer holds: see that module's docstring for why the split exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..basis._basis_1D import _tabulate_Bernstein_basis_1D_impl
from ..basis._basis_backend import bernstein_deriv_core
from ..basis._basis_core import _PARALLEL_MIN_NUM_PTS
from ..basis._basis_utils import (
    _compute_final_output_shape_1D,
    _compute_final_output_shape_1D_deriv,
    _normalize_points_1D,
    _reshaped_out,
    _validate_out_array,
)
from ._basis_backend import bspline_basis_core, bspline_basis_deriv_core
from ._bspline_knots import _is_in_domain_impl

if TYPE_CHECKING:
    from ._bspline_space_1d import BsplineSpace1D


def _tabulate_Bspline_basis_Bernstein_like_1D(
    spline: BsplineSpace1D,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions when they reduce to Bernstein polynomials.

    This function is used when the B-spline has Bézier-like knots, allowing
    direct evaluation using Bernstein basis functions.

    Args:
        spline (BsplineSpace1D): B-spline object with Bézier-like knots.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points (already normalized to 1D).
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape (num_pts, degree+1) and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape (num_pts,) and dtype np.int_ if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (basis_values, first_basis_indices) where basis_values is an array of shape
            (number pts, degree+1) that contains the Bernstein basis function values and
            first_basis_indices contains the indices of the first non-zero basis function
            for each point. If `out_basis` or `out_first_basis` was provided,
            returns the same array(s).

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
        ValueError: If `out_basis` or `out_first_basis` is provided and has incorrect shape
            or dtype.
    """
    if not spline.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots.")

    # map the points to the reference interval [0, 1].
    #
    # The two bounds are read at `pts`'s width, which Layer 2 has already promoted to
    # the call's. `pts - k0` would widen on its own, but `k1 - k0` is scalar against
    # scalar and would otherwise be computed at the space's storage width, so a
    # mixed-width call would carry the narrow span's rounding into every point. On a
    # same-width call the cast is the identity.
    k0, k1 = (pts.dtype.type(bound) for bound in spline.domain)
    pts_normalized = (pts - k0) / (k1 - k0)

    num_pts = pts.size
    expected_first_basis_shape = (num_pts,)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    else:
        _validate_out_array(out_first_basis, expected_first_basis_shape, np.int_)

    # the first basis function is always the 0
    out_first_basis.fill(0)

    # Compute Bernstein basis - pass out_basis directly since pts_normalized is already 1D
    # and _tabulate_Bernstein_basis_1D_impl will handle shape validation
    B = _tabulate_Bernstein_basis_1D_impl(spline.degree, pts_normalized, out=out_basis)

    return B, out_first_basis


def _tabulate_Bspline_basis_Bernstein_like_deriv_1D(
    spline: BsplineSpace1D,
    pts: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    out_deriv: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis derivatives for Bézier-like knots via Bernstein polynomials.

    Maps the evaluation points to the reference interval [0, 1], delegates to the
    parallel Bernstein derivative kernel, then applies the chain-rule correction
    ``(1/(b-a))^k`` to each k-th derivative slice.

    Args:
        spline (BsplineSpace1D): B-spline with Bézier-like knots.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points (1D, already
            normalized by :func:`_normalize_points_1D`).
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out_deriv (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array
            of shape ``(n_pts, n_deriv+1, degree+1)`` and dtype matching ``pts``.
        out_first_basis (npt.NDArray[np.int_]): Pre-allocated output array of shape
            ``(n_pts,)`` and dtype int.

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
    """
    if not spline.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots.")

    # At `pts`'s width, which Layer 2 has already promoted: see the note in
    # `_tabulate_Bspline_basis_Bernstein_like_1D`. `k1 - k0` is scalar against scalar
    # and would otherwise carry the space's storage width into the span and, through
    # `inv_span` below, into every chain-rule factor.
    k0, k1 = (pts.dtype.type(bound) for bound in spline.domain)
    pts_normalized = (pts - k0) / (k1 - k0)  # map to [0, 1]

    kernels = bernstein_deriv_core()
    if kernels.serial is not None and pts_normalized.shape[0] < _PARALLEL_MIN_NUM_PTS:
        kernels.serial(np.int32(spline.degree), pts_normalized, n_deriv, out_deriv)
    else:
        kernels.parallel(np.int32(spline.degree), pts_normalized, n_deriv, out_deriv)

    # Chain-rule: d^k/dx^k f(x) = d^k/ds^k f(s) * (ds/dx)^k = d^k/ds^k f(s) * (1/(k1-k0))^k
    inv_span: float = 1.0 / float(k1 - k0)
    scale: float = inv_span
    for k in range(1, n_deriv + 1):
        out_deriv[:, k, :] = out_deriv[:, k, :] * scale
        scale *= inv_span

    out_first_basis.fill(0)


def _promote_for_mixed_width(
    knots: npt.NDArray[np.float32 | np.float64],
    pts: npt.NDArray[np.float32 | np.float64],
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Bring a mixed-width call up to the wider of its two dtypes.

    A call whose knots and points have different widths -- a ``float32`` space
    evaluated at ``float64`` points, or the reverse -- computes at the **wider** of the
    two, and returns it. That is the library's contract rather than an implementation
    detail, so it is applied here, above the backend seam, where both backends see the
    same arrays: ``design/backend_parity.md`` Rule 1, a shared map on the common side
    cancels exactly. Applying it in one backend only would leave the two computing
    different values for the same input, which is what the parity suite exists to
    prevent.

    It replaces an older answer, under which the oracle opened its kernels with
    ``dtype = knots.dtype`` for their scratch while reading points at the points' own
    width, so a mixed call truncated every intermediate to the *narrower* of the two.
    The C++ kernels are templated on one scalar type and cannot express that at all,
    and the seam fell back to Numba for exactly this call shape.

    A same-width call is untouched, and neither array is copied.

    **The space's tolerance is deliberately not promoted with the knots**, and that has
    a consequence worth stating rather than discovering. It is an absolute parametric
    tolerance fixed when the space was built, derived from that knot vector's extent
    *and its storage format*: a ``float32`` vector over ``[0, 3]`` carries a tolerance
    of 2.9e-6 against a ``float64`` one's 5.3e-15. Widening the array moves no knot, and
    adds no resolution to knots that were already rounded, so the tolerance stays the
    narrow one and the mesh keeps saying what it can actually distinguish.

    So a mixed call is **not** equivalent to a space built at the wide width in every
    respect: the domain-membership gate that ``validate=True`` applies is the narrow
    space's, and is correspondingly looser. Measured: a point 1.4e-6 past the right
    endpoint is accepted by the ``float32`` space and refused by its widened twin. What
    the promotion makes equal is the **arithmetic** on the points that are accepted.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The space's knot vector.
        pts (npt.NDArray[np.float32 | np.float64]): The evaluation points, normalized.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 |
        np.float64]]: The knots and the points, both at the promoted dtype.
    """
    if knots.dtype == pts.dtype:
        return knots, pts
    work_dtype = np.promote_types(knots.dtype, pts.dtype)
    return knots.astype(work_dtype, copy=False), pts.astype(work_dtype, copy=False)


def _tabulate_Bspline_basis_1D_impl(
    spline: BsplineSpace1D,
    pts: npt.ArrayLike,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
    *,
    validate: bool = True,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at given points.

    This function automatically selects the most efficient evaluation method:
    - For Bézier-like knots: direct Bernstein evaluation
    - For general knots: BasisFuncs (Piegl & Tiller A2.2).  Batches smaller
      than ``_PARALLEL_MIN_NUM_PTS`` use the serial (non-parallel) kernel to
      avoid fork/join overhead; larger batches use the ``parallel=True`` kernel.

    In both cases it calls vectorized or numba implementations.

    Args:
        spline (BsplineSpace1D): B-spline object defining the basis.
        pts (npt.ArrayLike): Evaluation points.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape and dtype np.int_ if provided. This follows NumPy's style for
            output arrays. Defaults to None.
        validate (bool): If True (default), check that every point lies inside the
            spline domain. Pass False only when the caller guarantees in-domain
            points (e.g. points generated inside a knot span); out-of-domain
            points are then undefined behavior. Defaults to True.

    Returns:
        tuple[
            npt.NDArray[np.float32] | npt.NDArray[np.float64],
            npt.NDArray[np.int_]
        ]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32] | npt.NDArray[np.float64])
              Array of shape matching `pts` with the last dimension length (degree+1),
              containing the basis function values evaluated at each point.
              If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              1D integer array indicating the index of the first nonzero basis function
              for each evaluation point. The length is the same as the number of evaluation points.
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If ``validate`` is True and any evaluation point is outside the
            B-spline domain, or if `out_basis` or `out_first_basis` is provided and
            has incorrect shape or dtype.

    Example:
        >>> from pantr.bspline import BsplineSpace1D
        >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
        >>> values, first = _tabulate_Bspline_basis_1D_impl(bspline, [0.0, 0.5, 0.75, 1.0])
        >>> np.allclose(
        ...     values,
        ...     [
        ...         [1.0, 0.0, 0.0],
        ...         [0.12698413, 0.5643739, 0.30864198],
        ...         [0.69444444, 0.27777778, 0.02777778],
        ...         [0.0, 0.0, 1.0],
        ...     ],
        ... )
        True
        >>> first.tolist()
        [0, 1, 3, 3]
    """
    input_shape = np.shape(pts)
    pts = _normalize_points_1D(pts)

    if validate and not np.all(
        _is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)
    ):
        raise ValueError(
            f"One or more values in pts are outside the knot vector domain {spline.domain}"
        )

    knots, pts = _promote_for_mixed_width(spline.knots, pts)

    num_pts = pts.shape[0]
    n_basis = spline.degree + 1
    expected_final_shape = _compute_final_output_shape_1D(input_shape, n_basis)
    expected_dtype = pts.dtype
    expected_first_basis_shape = input_shape

    if out_basis is None:
        out_basis = np.empty(expected_final_shape, dtype=expected_dtype)
    _validate_out_array(out_basis, expected_final_shape, expected_dtype)
    basis_normalized, copy_back_basis = _reshaped_out(out_basis, (num_pts, n_basis))

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    _validate_out_array(out_first_basis, expected_first_basis_shape, np.int_)
    first_indices_normalized, copy_back_first = _reshaped_out(out_first_basis, (num_pts,))

    if spline.has_Bezier_like_knots():
        _tabulate_Bspline_basis_Bernstein_like_1D(
            spline, pts, basis_normalized, first_indices_normalized
        )
    else:
        kernels = bspline_basis_core()
        kernel = (
            kernels.serial
            if kernels.serial is not None and num_pts < _PARALLEL_MIN_NUM_PTS
            else kernels.parallel
        )
        kernel(
            knots,
            spline.degree,
            spline.periodic,
            spline.tolerance,
            pts,
            basis_normalized,
            first_indices_normalized,
        )

    if copy_back_basis:
        out_basis[...] = basis_normalized.reshape(out_basis.shape)
    if copy_back_first:
        out_first_basis[...] = first_indices_normalized.reshape(out_first_basis.shape)

    return out_basis, out_first_basis


def _tabulate_Bspline_basis_deriv_1D_impl(  # noqa: PLR0913
    spline: BsplineSpace1D,
    pts: npt.ArrayLike,
    n_deriv: int,
    out_deriv: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
    *,
    validate: bool = True,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis function derivatives at given points.

    Implements Algorithm A2.3 (DerBasisFuncs) from Piegl & Tiller.  Uses a
    fast Bernstein path for Bézier-like knots (parallel kernel + chain-rule
    scaling) and falls back to the general DerBasisFuncs kernel otherwise.
    For general knots, batches smaller than ``_PARALLEL_MIN_NUM_PTS`` use the
    serial twin to avoid fork/join overhead.
    The 0th slice of the result is identical to the output of
    :func:`_tabulate_Bspline_basis_1D_impl`.  For ``n_deriv > degree`` all
    rows beyond ``degree`` are identically zero.

    Args:
        spline (BsplineSpace1D): B-spline object defining the basis.
        pts (npt.ArrayLike): Evaluation points.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out_deriv (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            for derivative values. If None, a new array is allocated. Must have shape
            ``(*pts_shape, n_deriv+1, degree+1)`` and dtype matching ``pts`` if provided.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array for first
            basis indices. If None, a new array is allocated. Must have shape ``pts_shape``
            and dtype ``np.int_`` if provided. Defaults to None.
        validate (bool): If True (default), check that every point lies inside the
            spline domain. Pass False only when the caller guarantees in-domain
            points (e.g. points generated inside a knot span); out-of-domain
            points are then undefined behavior. Defaults to True.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            ``(deriv_values, first_basis_indices)``.
            ``deriv_values[..., k, i]`` is the k-th derivative of the i-th local basis
            function at each point.

    Raises:
        ValueError: If ``n_deriv < 0``, if ``validate`` is True and any evaluation
            point is outside the domain, or ``out_deriv`` / ``out_first_basis`` has
            incorrect shape or dtype.

    Example:
        >>> from pantr.bspline import BsplineSpace1D
        >>> bspline = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        >>> d, first = _tabulate_Bspline_basis_deriv_1D_impl(bspline, [0.5], n_deriv=1)
        >>> d.shape
        (1, 2, 3)
    """
    if n_deriv < 0:
        raise ValueError(f"n_deriv must be non-negative, got {n_deriv}")

    input_shape = np.shape(pts)
    pts = _normalize_points_1D(pts)

    if validate and not np.all(
        _is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)
    ):
        raise ValueError(
            f"One or more values in pts are outside the knot vector domain {spline.domain}"
        )

    knots, pts = _promote_for_mixed_width(spline.knots, pts)

    num_pts = pts.shape[0]
    order = spline.degree + 1
    expected_dtype = pts.dtype
    expected_deriv_shape = _compute_final_output_shape_1D_deriv(input_shape, n_deriv, order)
    expected_first_basis_shape = input_shape

    if out_deriv is None:
        out_deriv = np.empty(expected_deriv_shape, dtype=expected_dtype)
    _validate_out_array(out_deriv, expected_deriv_shape, expected_dtype)
    deriv_normalized, copy_back_deriv = _reshaped_out(out_deriv, (num_pts, n_deriv + 1, order))

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    _validate_out_array(out_first_basis, expected_first_basis_shape, np.int_)
    first_indices_normalized, copy_back_first = _reshaped_out(out_first_basis, (num_pts,))

    if spline.has_Bezier_like_knots():
        _tabulate_Bspline_basis_Bernstein_like_deriv_1D(
            spline, pts, n_deriv, deriv_normalized, first_indices_normalized
        )
    else:
        deriv_kernels = bspline_basis_deriv_core()
        deriv_kernel = (
            deriv_kernels.serial
            if deriv_kernels.serial is not None and num_pts < _PARALLEL_MIN_NUM_PTS
            else deriv_kernels.parallel
        )
        deriv_kernel(
            knots,
            spline.degree,
            spline.periodic,
            spline.tolerance,
            n_deriv,
            pts,
            deriv_normalized,
            first_indices_normalized,
        )

    if copy_back_deriv:
        out_deriv[...] = deriv_normalized.reshape(out_deriv.shape)
    if copy_back_first:
        out_first_basis[...] = first_indices_normalized.reshape(out_first_basis.shape)

    return out_deriv, out_first_basis


__all__ = [
    "_tabulate_Bspline_basis_1D_impl",
    "_tabulate_Bspline_basis_Bernstein_like_1D",
    "_tabulate_Bspline_basis_Bernstein_like_deriv_1D",
    "_tabulate_Bspline_basis_deriv_1D_impl",
]
