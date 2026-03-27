"""Mask operations for Bernstein polynomial subcell grids.

Provides functions to create, query, and manipulate boolean masks over a
regular ``M x M x ... x M`` subdivision of the reference cube ``[0,1]^N``.
A True entry indicates the associated polynomial may have zeros in that
subcell; False guarantees the polynomial is nonzero there.

This module forms **Layer 2** of the mask implementation: it validates inputs,
allocates output arrays, and dispatches to the Numba kernels in
``_mask_core``.

This is a translation of the masking logic from the algoim library
(R. I. Saye, *J. Comput. Phys.* 448, 110720, 2022).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import wait_for_jit_warmup
from ._mask_core import (
    _intersection_mask_2d_core,
    _intersection_mask_3d_core,
    _line_intersects_mask_core,
    _nonzero_mask_1d_core,
    _nonzero_mask_2d_core,
    _nonzero_mask_3d_core,
    _point_within_mask_core,
)

if TYPE_CHECKING:
    from . import Bezier


# ---------------------------------------------------------------------------
# Pure numpy operations
# ---------------------------------------------------------------------------


def _mask_empty(mask: npt.NDArray[np.bool_]) -> bool:
    """Test if a mask is entirely False (no active subcells).

    Args:
        mask (npt.NDArray[np.bool_]): Boolean mask of shape ``(M,)*N``.

    Returns:
        bool: True if every entry is False.
    """
    return not np.any(mask)


def _collapse_mask(
    mask: npt.NDArray[np.bool_],
    axis: int,
) -> npt.NDArray[np.bool_]:
    """OR-reduce a mask along one axis, producing an (N-1)-D mask.

    For each (N-1)-D cell, the result is True if *any* cell along the
    collapsed axis is True.

    Args:
        mask (npt.NDArray[np.bool_]): Boolean mask of shape ``(M,)*N``
            where ``N >= 2``.
        axis (int): Axis to collapse, in ``[0, N)``.

    Returns:
        npt.NDArray[np.bool_]: Collapsed mask of shape ``(M,)*(N-1)``.

    Raises:
        ValueError: If ``mask.ndim < 2`` or ``axis`` is out of range.
    """
    if mask.ndim < 2:  # noqa: PLR2004
        raise ValueError(f"Cannot collapse a {mask.ndim}-D mask; need at least 2 dimensions.")
    if not 0 <= axis < mask.ndim:
        raise ValueError(f"axis {axis} is out of range for a {mask.ndim}-D mask.")
    result: npt.NDArray[np.bool_] = np.any(mask, axis=axis)
    return result


def _restrict_to_face(
    mask: npt.NDArray[np.bool_],
    axis: int,
    side: int,
) -> npt.NDArray[np.bool_]:
    """Extract an (N-1)-D face mask from an N-D mask.

    Args:
        mask (npt.NDArray[np.bool_]): Boolean mask of shape ``(M,)*N``
            where ``N >= 2``.
        axis (int): Axis perpendicular to the face, in ``[0, N)``.
        side (int): ``0`` for the lower face, ``1`` for the upper face.

    Returns:
        npt.NDArray[np.bool_]: Face mask of shape ``(M,)*(N-1)``.

    Raises:
        ValueError: If ``mask.ndim < 2``, ``axis`` is out of range, or
            ``side`` is not 0 or 1.
    """
    if mask.ndim < 2:  # noqa: PLR2004
        raise ValueError(
            f"Cannot restrict a {mask.ndim}-D mask to a face; need at least 2 dimensions."
        )
    if not 0 <= axis < mask.ndim:
        raise ValueError(f"axis {axis} is out of range for a {mask.ndim}-D mask.")
    if side not in (0, 1):
        raise ValueError(f"side must be 0 or 1, got {side}.")

    M = mask.shape[0]
    idx = 0 if side == 0 else M - 1

    # Build a tuple of slices to extract the face.
    slices: list[int | slice] = [slice(None)] * mask.ndim
    slices[axis] = idx
    result: npt.NDArray[np.bool_] = mask[tuple(slices)]
    return np.ascontiguousarray(result)


# ---------------------------------------------------------------------------
# Numba-backed operations
# ---------------------------------------------------------------------------


def _point_within_mask(
    mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.floating[Any]],
    M: int = 8,
) -> bool:
    """Test if point ``x`` falls in a True subcell of a mask.

    Discretizes ``x`` to grid coordinates and checks the mask.

    Args:
        mask (npt.NDArray[np.bool_]): Boolean mask of shape ``(M,)*N``.
        x (npt.NDArray[np.floating]): Point in ``[0,1]^N``, shape ``(N,)``.
        M (int): Grid resolution per axis. Defaults to 8.

    Returns:
        bool: True if the subcell containing ``x`` is marked True.

    Raises:
        ValueError: If ``x`` has wrong length for the mask dimensions.
    """
    N = mask.ndim
    if x.shape[0] != N:
        raise ValueError(f"Point has {x.shape[0]} components but mask has {N} dimensions.")
    wait_for_jit_warmup()
    x_f64 = np.asarray(x, dtype=np.float64).ravel()
    return bool(_point_within_mask_core(mask.ravel(), x_f64, M, N))


def _line_intersects_mask(
    mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.floating[Any]],
    axis: int,
    M: int = 8,
) -> bool:
    """Test if line ``{x + alpha * e_axis}`` hits a True subcell for some alpha in [0,1].

    The point ``x`` has ``N-1`` components (axis excluded).

    Args:
        mask (npt.NDArray[np.bool_]): Boolean mask of shape ``(M,)*N``.
        x (npt.NDArray[np.floating]): Base point, shape ``(N-1,)``.
        axis (int): Axis along which to scan.
        M (int): Grid resolution per axis. Defaults to 8.

    Returns:
        bool: True if any subcell along the line is marked True.

    Raises:
        ValueError: If ``axis`` is out of range or ``x`` has wrong length.
    """
    N = mask.ndim
    if not 0 <= axis < N:
        raise ValueError(f"axis {axis} is out of range for a {N}-D mask.")
    expected_len = N - 1
    if x.shape[0] != expected_len:
        raise ValueError(
            f"Point has {x.shape[0]} components but expected {expected_len} (N-1 where N={N})."
        )
    wait_for_jit_warmup()
    x_f64 = np.asarray(x, dtype=np.float64).ravel()
    return bool(_line_intersects_mask_core(mask.ravel(), x_f64, axis, M, N))


def _extract_scalar_coeffs(
    bezier: Bezier,
) -> npt.NDArray[np.float64]:
    """Extract scalar Bernstein coefficients from a Bezier as a contiguous N-D float64 array.

    Args:
        bezier (~pantr.bezier.Bezier): A scalar (rank == 1), non-rational Bezier.

    Returns:
        npt.NDArray[np.float64]: Contiguous coefficient array of shape
        ``(p0+1, p1+1, ...)``.

    Raises:
        TypeError: If ``bezier`` is rational.
        ValueError: If ``bezier`` is not scalar (rank != 1).
    """
    if bezier.is_rational:
        raise TypeError("Mask operations require non-rational Bézier polynomials.")
    if bezier.rank != 1:
        raise ValueError(
            f"Mask operations require a scalar Bézier (rank == 1), got rank {bezier.rank}."
        )
    # Remove the trailing rank-1 axis.
    coeffs: npt.NDArray[np.float64] = np.ascontiguousarray(
        bezier.control_points[..., 0], dtype=np.float64
    )
    return coeffs


def _nonzero_mask(
    bezier: Bezier,
    mask: npt.NDArray[np.bool_] | None = None,
    M: int = 8,
) -> npt.NDArray[np.bool_]:
    """Compute conservative nonzero mask of subcells where polynomial may have zeros.

    For each of the ``M^N`` subcells, determines whether the polynomial
    could have a zero crossing there.  Uses recursive subdivision with
    de Casteljau restriction and uniform sign detection.

    Args:
        bezier (~pantr.bezier.Bezier): A scalar (rank == 1), non-rational
            Bézier of parametric dimension 1, 2, or 3.
        mask (npt.NDArray[np.bool_] | None): Input mask to restrict the
            search.  If None, an all-True mask is used. Defaults to None.
        M (int): Grid resolution per axis. Defaults to 8.

    Returns:
        npt.NDArray[np.bool_]: Boolean mask of shape ``(M,)*dim``.

    Raises:
        TypeError: If ``bezier`` is rational.
        ValueError: If ``bezier`` is not scalar, or ``dim > 3``.
    """
    coeffs = _extract_scalar_coeffs(bezier)
    dim = bezier.dim

    if dim > 3:  # noqa: PLR2004
        raise ValueError(f"nonzero_mask supports dim <= 3, got {dim}.")

    shape = tuple(M for _ in range(dim))
    if mask is None:
        fmask = np.ones(shape, dtype=np.bool_)
    else:
        fmask = np.ascontiguousarray(mask, dtype=np.bool_)

    out = np.zeros(shape, dtype=np.bool_)

    wait_for_jit_warmup()
    if dim == 1:
        _nonzero_mask_1d_core(coeffs, fmask, out, M)
    elif dim == 2:  # noqa: PLR2004
        _nonzero_mask_2d_core(coeffs, fmask, out, M)
    else:
        _nonzero_mask_3d_core(coeffs, fmask, out, M)

    return out


def _intersection_mask(
    bezier_f: Bezier,
    mask_f: npt.NDArray[np.bool_],
    bezier_g: Bezier,
    mask_g: npt.NDArray[np.bool_],
    M: int = 8,
) -> npt.NDArray[np.bool_]:
    """Compute intersection mask where two polynomials may share a common zero.

    Uses recursive subdivision with the orthant test to determine subcells
    where both polynomials could simultaneously vanish.

    Args:
        bezier_f (~pantr.bezier.Bezier): First scalar non-rational Bézier.
        mask_f (npt.NDArray[np.bool_]): Nonzero mask of ``bezier_f``,
            shape ``(M,)*dim``.
        bezier_g (~pantr.bezier.Bezier): Second scalar non-rational Bézier.
        mask_g (npt.NDArray[np.bool_]): Nonzero mask of ``bezier_g``,
            shape ``(M,)*dim``.
        M (int): Grid resolution per axis. Defaults to 8.

    Returns:
        npt.NDArray[np.bool_]: Intersection mask of shape ``(M,)*dim``.

    Raises:
        TypeError: If either Bézier is rational.
        ValueError: If either is not scalar, dimensions don't match, or dim > 3.
    """
    coeffs_f = _extract_scalar_coeffs(bezier_f)
    coeffs_g = _extract_scalar_coeffs(bezier_g)

    if bezier_f.dim != bezier_g.dim:
        raise ValueError(
            f"Dimension mismatch: bezier_f has dim {bezier_f.dim}, bezier_g has dim {bezier_g.dim}."
        )
    dim = bezier_f.dim
    if dim > 3:  # noqa: PLR2004
        raise ValueError(f"intersection_mask supports dim <= 3, got {dim}.")
    if dim < 2:  # noqa: PLR2004
        raise ValueError(f"intersection_mask requires dim >= 2, got {dim}.")

    shape = tuple(M for _ in range(dim))
    fmask = np.ascontiguousarray(mask_f, dtype=np.bool_)
    gmask = np.ascontiguousarray(mask_g, dtype=np.bool_)
    out = np.zeros(shape, dtype=np.bool_)

    wait_for_jit_warmup()
    if dim == 2:  # noqa: PLR2004
        _intersection_mask_2d_core(coeffs_f, fmask, coeffs_g, gmask, out, M)
    else:
        _intersection_mask_3d_core(coeffs_f, fmask, coeffs_g, gmask, out, M)

    return out
