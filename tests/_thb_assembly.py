"""Test-only THB assembly helpers for the paper-reproduction validation suite.

This module is **not** part of the public ``pantr`` API and is intentionally
underscore-prefixed so pytest does not collect it as a test module.  It provides the
global cell-by-cell mass-matrix assembly and L2 projection that the THB validation
suite needs.  The hierarchical basis is not tensor-product, so (unlike
:func:`pantr.bspline.l2_project_bspline`, which is a pure Kronecker solve) the mass
matrix has no separable structure and must be assembled by looping over the active
leaf cells — the FEM-style assembly deliberately kept out of the library (#152 Q0).

Functions:

- :func:`gram_matrix`: the global mass/Gram matrix ``M[i,j] = ∫ φ_i φ_j``.
- :func:`l2_project_thb`: the L2 best approximation of a callable as a
  :class:`~pantr.bspline.THBSpline`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pantr.bspline import THBSpline, THBSplineSpace
from pantr.grid import cell_quadrature
from pantr.quad import gauss_legendre_quadrature

if TYPE_CHECKING:
    from collections.abc import Callable


def _assemble(
    thb: THBSplineSpace,
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike] | None,
    n_quad: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Assemble the global mass matrix and (optionally) the load vector.

    Loops over the active leaf cells, mapping a Gauss-Legendre rule onto each via
    :func:`pantr.grid.cell_quadrature` (volume-scaled weights) and scattering the
    local ``(K, K)`` block ``φᵀ W φ`` into the global matrix using the cell's
    :meth:`~pantr.bspline.THBSplineSpace.active_basis` dofs.

    Args:
        thb (THBSplineSpace): The hierarchical space.
        func (Callable | None): Scalar integrand for the load vector, called on an
            ``(n_pts, dim)`` point array and returning ``(n_pts,)``.  If ``None``,
            the returned load vector is zero.
        n_quad (int | None): Gauss points per direction.  ``None`` uses
            ``max(degrees) + 1``, exact for the per-cell degree-``2p`` mass products
            when all degrees are equal (each active function is degree ``p`` on a leaf
            cell, so a ``p + 1``-point rule is exact to degree ``2p + 1``).

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ``(M, b)`` of shapes
        ``(num_active, num_active)`` and ``(num_active,)``.
    """
    dim = thb.dim
    n = thb.num_active_functions
    nq = max(thb.degrees) + 1 if n_quad is None else n_quad
    rule = gauss_legendre_quadrature(dim, nq)
    pts_all, wts_all = cell_quadrature(thb.grid, rule)

    mass = np.zeros((n, n), dtype=np.float64)
    load = np.zeros(n, dtype=np.float64)
    for cid in range(thb.grid.num_cells):
        pts = np.ascontiguousarray(pts_all[cid], dtype=np.float64)
        wts = np.asarray(wts_all[cid], dtype=np.float64)
        vals = np.asarray(thb.tabulate_basis(cid, pts), dtype=np.float64)
        dofs = thb.active_basis(cid)
        mass[np.ix_(dofs, dofs)] += vals.T @ (vals * wts[:, None])
        if func is not None:
            fvals = np.asarray(func(pts), dtype=np.float64).reshape(pts.shape[0])
            load[dofs] += vals.T @ (wts * fvals)
    return mass, load


def gram_matrix(thb: THBSplineSpace, *, n_quad: int | None = None) -> npt.NDArray[np.float64]:
    """Return the global Gram (mass) matrix ``M[i, j] = ∫ φ_i φ_j``.

    Args:
        thb (THBSplineSpace): The hierarchical space.
        n_quad (int | None): Gauss points per direction.  Defaults to
            ``max(degrees) + 1``.

    Returns:
        npt.NDArray[np.float64]: Symmetric matrix of shape ``(num_active, num_active)``;
        positive definite for the linearly independent THB basis, only positive
        semidefinite for a non-truncated HB basis.
    """
    return _assemble(thb, None, n_quad)[0]


def l2_project_thb(
    thb: THBSplineSpace,
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    *,
    n_quad: int | None = None,
) -> THBSpline:
    """Return the L2 best approximation of ``func`` in the THB space.

    Assembles the global mass matrix ``M`` and load ``b = ∫ f φ_i`` and solves
    ``M c = b`` (numpy direct solve; ``M`` is SPD for the linearly independent THB
    basis).

    Args:
        thb (THBSplineSpace): The hierarchical space.
        func (Callable): Scalar function, called on an ``(n_pts, dim)`` point array
            and returning ``(n_pts,)``.
        n_quad (int | None): Gauss points per direction.  Defaults to
            ``max(degrees) + 1``.

    Returns:
        THBSpline: The L2 projection of ``func``.
    """
    mass, load = _assemble(thb, func, n_quad)
    coeffs = np.asarray(np.linalg.solve(mass, load), dtype=np.float64)
    return THBSpline(thb, coeffs)


def l2_error(
    spline: THBSpline,
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    *,
    n_quad: int | None = None,
) -> float:
    """Return the L2 error ``sqrt(∫ (func - spline)^2)`` over the hierarchical domain.

    Args:
        spline (THBSpline): The (scalar) approximation.
        func (Callable): The exact function, called on an ``(n_pts, dim)`` point array
            and returning ``(n_pts,)``.
        n_quad (int | None): Gauss points per direction.  Defaults to
            ``max(degrees) + 3`` (a few extra to resolve a smooth ``func``).

    Returns:
        float: The L2 error.
    """
    thb = spline.space
    nq = max(thb.degrees) + 3 if n_quad is None else n_quad
    rule = gauss_legendre_quadrature(thb.dim, nq)
    pts_all, wts_all = cell_quadrature(thb.grid, rule)
    err_sq = 0.0
    for cid in range(thb.grid.num_cells):
        pts = np.ascontiguousarray(pts_all[cid], dtype=np.float64)
        wts = np.asarray(wts_all[cid], dtype=np.float64)
        diff = np.asarray(func(pts), dtype=np.float64).reshape(pts.shape[0]) - np.asarray(
            spline.evaluate(pts), dtype=np.float64
        ).reshape(pts.shape[0])
        err_sq += float(np.sum(wts * diff**2))
    return float(np.sqrt(err_sq))
