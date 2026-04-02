"""High-order quadrature on domains implicitly defined by multivariate polynomials.

Implements the dimension-reduction algorithm of Saye (JCP 2022) entirely in
Numba nopython mode for near-C++ performance. The algorithm recasts implicitly
defined geometry as the graph of a multi-valued height function and applies
recursive dimension reduction down to one-dimensional quadrature.

The algorithm has two phases:

1. **Build phase**: Given tensor-product Bernstein polynomials defining the
   implicit geometry, construct a dimension-reduction hierarchy. This is done
   once per set of polynomials and can be reused for different quadrature orders.

2. **Construction phase**: Given the hierarchy and a quadrature order *q*,
   generate quadrature points and weights. Supports volume integrals
   (over {phi < 0}) and surface integrals (over {phi = 0}).

Main exports:

- :class:`ImplicitPolyQuadrature` -- build + query interface.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr.bezier.implicit._bernstein import _eval_bernstein_2d, _eval_bernstein_3d
from pantr.bezier.implicit._build import build_2d, build_3d
from pantr.bezier.implicit._construct import (
    surface_quad_2d,
    volume_quad_2d,
    volume_quad_3d,
)
from pantr.bezier.implicit._mask import (
    compute_nonzero_mask_2d,
    compute_nonzero_mask_3d,
)

if TYPE_CHECKING:
    from pantr.bezier._bezier import Bezier


class QuadStrategy(IntEnum):
    """Strategy for choosing 1D quadrature method at each level."""

    GL_ONLY = 0
    """Gauss-Legendre on all intervals."""
    TS_ONLY = 1
    """Tanh-sinh on all intervals."""
    AUTO_MIXED = 2
    """Tanh-sinh on outer integrals with branching points, GL on inner."""


class ImplicitPolyQuadrature:
    """High-order quadrature on domains implicitly defined by Bernstein polynomials.

    Builds a dimension-reduction hierarchy from one or more tensor-product
    Bernstein polynomials and generates quadrature points/weights for volume
    and surface integrals.

    Attributes:
        dim (int): Parametric dimension (2 or 3).
        n_polys (int): Number of input polynomials.
    """

    def __init__(self, *polynomials: Bezier | npt.NDArray[np.float64]) -> None:
        """Build the quadrature hierarchy.

        Args:
            *polynomials: One or more tensor-product Bernstein polynomials,
                given either as :class:`~pantr.bezier.Bezier` objects or as
                raw coefficient arrays of shape ``(n0+1, n1+1, ...)`` where
                each ``ni`` is the degree in direction *i*.

        Raises:
            ValueError: If no polynomials are given, dimensions are
                inconsistent, or dimension is not 2 or 3.
        """
        if len(polynomials) == 0:
            msg = "At least one polynomial is required."
            raise ValueError(msg)

        # Extract coefficient arrays.
        coeffs_arrays: list[npt.NDArray[np.float64]] = []
        for p in polynomials:
            if hasattr(p, "control_points"):
                # Bezier object: extract scalar control points.
                cp = np.asarray(p.control_points, dtype=np.float64)
                if cp.ndim > p.dim:  # type: ignore[union-attr]
                    # Scalar Bezier: last axis is rank=1, squeeze it.
                    cp = cp[..., 0]
                coeffs_arrays.append(cp)
            else:
                coeffs_arrays.append(np.asarray(p, dtype=np.float64))

        # Validate dimensions.
        self.dim = coeffs_arrays[0].ndim
        for ca in coeffs_arrays:
            if ca.ndim != self.dim:
                msg = f"All polynomials must have the same dimension, got {ca.ndim} and {self.dim}."
                raise ValueError(msg)

        if self.dim not in (2, 3):
            msg = f"Only 2D and 3D are supported, got {self.dim}D."
            raise ValueError(msg)

        self.n_polys = len(coeffs_arrays)
        self._coeffs = coeffs_arrays

        # Compute masks and build hierarchy.
        if self.dim == 2:
            coeffs_list = NumbaList()
            masks_list = NumbaList()
            for ca in coeffs_arrays:
                coeffs_list.append(ca)
                masks_list.append(compute_nonzero_mask_2d(ca))
            self._build_result: tuple[Any, ...] = build_2d(coeffs_list, masks_list)
        else:
            coeffs_list_3d = NumbaList()
            masks_list_3d = NumbaList()
            for ca in coeffs_arrays:
                coeffs_list_3d.append(ca)
                masks_list_3d.append(compute_nonzero_mask_3d(ca))
            self._build_result = build_3d(coeffs_list_3d, masks_list_3d)

    def volume_quad(
        self,
        q: int,
        strategy: QuadStrategy = QuadStrategy.AUTO_MIXED,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate volume quadrature points and weights.

        The quadrature integrates over the full domain [0,1]^d. To compute
        an integral over {phi < 0}, filter the points by the sign of phi.

        Args:
            q (int): Number of 1D quadrature points per sub-interval.
            strategy (QuadStrategy): Quadrature method selection.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                ``(points, weights)`` with shapes ``(n_pts, dim)`` and
                ``(n_pts,)``. Weights sum to 1 (the volume of [0,1]^d).
        """
        gl_nodes, gl_weights = _gauss_legendre_01(q)
        ts_nodes, ts_weights = _tanh_sinh_01(q)
        strat = int(strategy)

        r = self._build_result
        if self.dim == 2:
            return volume_quad_2d(
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                r[7],
                r[8],
                r[9],
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )
        else:
            return volume_quad_3d(
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                r[7],
                r[8],
                r[9],
                r[10],
                r[11],
                r[12],
                r[13],
                r[14],
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )

    def surface_quad(
        self,
        q: int,
        strategy: QuadStrategy = QuadStrategy.AUTO_MIXED,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Generate surface quadrature points and weights.

        Computes quadrature over the zero level set {phi = 0} in flux form.
        The scalar weights integrate |f| and the normal weights integrate
        f * n where n is the outward normal.

        Args:
            q (int): Number of 1D quadrature points per sub-interval.
            strategy (QuadStrategy): Quadrature method selection.

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
                ``(points, scalar_weights, normal_weights)`` with shapes
                ``(n_pts, dim)``, ``(n_pts,)``, ``(n_pts, dim)``.
        """
        gl_nodes, gl_weights = _gauss_legendre_01(q)
        ts_nodes, ts_weights = _tanh_sinh_01(q)
        strat = int(strategy)

        r = self._build_result
        if self.dim == 2:
            input_c = NumbaList()
            for ca in self._coeffs:
                input_c.append(ca)
            return surface_quad_2d(
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                r[7],
                r[8],
                r[9],
                self.n_polys,
                input_c,
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )
        else:
            msg = "3D surface quadrature not yet implemented."
            raise NotImplementedError(msg)

    def eval_poly(self, poly_idx: int, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate a polynomial at the given points.

        Args:
            poly_idx (int): Index of the polynomial (0-based).
            points (npt.NDArray[np.float64]): Points of shape ``(n_pts, dim)``.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(n_pts,)``.
        """
        coeffs = self._coeffs[poly_idx]
        n = points.shape[0]
        values = np.empty(n, dtype=np.float64)
        if self.dim == 2:
            for i in range(n):
                values[i] = _eval_bernstein_2d(coeffs, points[i])
        else:
            for i in range(n):
                values[i] = _eval_bernstein_3d(coeffs, points[i])
        return values


# ---------------------------------------------------------------------------
# Quadrature node generation (Python-level, pre-computed once per q)
# ---------------------------------------------------------------------------


def _gauss_legendre_01(q: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Gauss-Legendre nodes and weights on [0, 1].

    Args:
        q (int): Number of quadrature points.

    Returns:
        tuple: (nodes, weights) both of shape ``(q,)``.
    """
    from numpy.polynomial.legendre import leggauss  # noqa: PLC0415

    pts, wts = leggauss(q)  # type: ignore[no-untyped-call]
    return np.ascontiguousarray(0.5 * (pts + 1.0)), np.ascontiguousarray(0.5 * wts)


def _tanh_sinh_01(q: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute tanh-sinh nodes and weights on [0, 1].

    Args:
        q (int): Number of quadrature points.

    Returns:
        tuple: (nodes, weights) both of shape ``(q,)``.
    """
    from pantr.quad import get_tanh_sinh_1d  # noqa: PLC0415

    nodes, weights = get_tanh_sinh_1d(q)
    return np.ascontiguousarray(nodes), np.ascontiguousarray(weights)
