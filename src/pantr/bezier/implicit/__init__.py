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
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr.bezier.implicit._bernstein import _eval_bernstein_2d, _eval_bernstein_3d
from pantr.bezier.implicit._build import (
    build_2d,
    build_2d_forced_k,
    build_3d,
    build_3d_forced_k,
)
from pantr.bezier.implicit._construct import (
    surface_quad_2d,
    surface_quad_2d_aggregate,
    surface_quad_3d,
    surface_quad_3d_aggregate,
    volume_quad_2d,
    volume_quad_3d,
)
from pantr.bezier.implicit._convert import (
    monomial_to_bernstein_2d,
    monomial_to_bernstein_3d,
)
from pantr.bezier.implicit._mask import (
    compute_nonzero_mask_2d,
    compute_nonzero_mask_3d,
)

if TYPE_CHECKING:
    from pantr.bezier._bezier import Bezier

VolQuadResult: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""Volume quadrature result: ``(points, weights)``."""

SurfQuadResult: TypeAlias = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]
"""Surface quadrature result: ``(points, scalar_weights, normal_weights)``."""

__all__ = [
    "ImplicitPolyQuadrature",
    "QuadStrategy",
    "SurfQuadResult",
    "VolQuadResult",
    "monomial_to_bernstein_2d",
    "monomial_to_bernstein_3d",
]


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
        dim (int): Parametric dimension (2 or 3). Read-only.
        n_polys (int): Number of input polynomials. Read-only.
    """

    def __init__(self, *polynomials: Bezier | npt.NDArray[np.float64]) -> None:  # noqa: PLR0912
        """Build the quadrature hierarchy.

        Args:
            *polynomials: One or more tensor-product Bernstein polynomials,
                given either as :class:`~pantr.bezier.Bezier` objects or as
                raw coefficient arrays of shape ``(n0+1, n1+1, ...)`` where
                each ``ni`` is the degree in direction *i*.

        Raises:
            ValueError: If no polynomials are given, dimensions are
                inconsistent, dimension is not 2 or 3, or a polynomial
                is identically zero.
        """
        if len(polynomials) == 0:
            msg = "At least one polynomial is required."
            raise ValueError(msg)

        # Extract coefficient arrays.
        coeffs_arrays: list[npt.NDArray[np.float64]] = []
        for p in polynomials:
            if isinstance(p, np.ndarray):
                coeffs_arrays.append(np.asarray(p, dtype=np.float64))
            elif hasattr(p, "control_points"):
                # Bezier object: extract scalar control points.
                cp = np.asarray(p.control_points, dtype=np.float64)
                if cp.ndim > p.dim:
                    if cp.shape[-1] != 1:
                        msg = (
                            f"Only scalar Bezier polynomials are supported, "
                            f"got rank {cp.shape[-1]}."
                        )
                        raise ValueError(msg)
                    cp = cp[..., 0]
                coeffs_arrays.append(cp)
            else:
                coeffs_arrays.append(np.asarray(p, dtype=np.float64))

        # Validate dimensions.
        dim = coeffs_arrays[0].ndim
        for ca in coeffs_arrays:
            if ca.ndim != dim:
                msg = f"All polynomials must have the same dimension, got {ca.ndim} and {dim}."
                raise ValueError(msg)

        if dim not in (2, 3):
            msg = f"Only 2D and 3D are supported, got {dim}D."
            raise ValueError(msg)

        # Reject identically-zero polynomials (undefined implicit domain).
        for i, ca in enumerate(coeffs_arrays):
            if np.all(ca == 0.0):
                msg = f"Polynomial {i} is identically zero; the implicit domain is undefined."
                raise ValueError(msg)

        self._dim = dim
        self._n_polys = len(coeffs_arrays)
        self._coeffs = coeffs_arrays

        # Build typed lists (cached for reuse across quad calls).
        self._coeffs_list = NumbaList()
        for ca in coeffs_arrays:
            self._coeffs_list.append(ca)

        # Compute masks and build hierarchy.
        if self._dim == 2:  # noqa: PLR2004
            self._masks_list = NumbaList()
            for ca in coeffs_arrays:
                self._masks_list.append(compute_nonzero_mask_2d(ca))
            self._build_result: tuple[Any, ...] = build_2d(self._coeffs_list, self._masks_list)
        else:
            self._masks_list = NumbaList()
            for ca in coeffs_arrays:
                self._masks_list.append(compute_nonzero_mask_3d(ca))
            self._build_result = build_3d(self._coeffs_list, self._masks_list)

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: 2 or 3.
        """
        return self._dim

    @property
    def n_polys(self) -> int:
        """Get the number of input polynomials.

        Returns:
            int: Number of polynomials.
        """
        return self._n_polys

    def volume_quad(
        self,
        q: int,
        strategy: QuadStrategy = QuadStrategy.AUTO_MIXED,
    ) -> VolQuadResult:
        """Generate volume quadrature points and weights.

        The quadrature integrates over the full domain [0,1]^d. To compute
        an integral over {phi < 0}, filter the points by the sign of phi.

        Args:
            q (int): Number of 1D quadrature points per sub-interval.
                Must be >= 1.
            strategy (QuadStrategy): Quadrature method selection.

        Returns:
            VolQuadResult: ``(points, weights)`` with shapes
                ``(n_pts, dim)`` and ``(n_pts,)``. When integrated against
                the constant function 1, the weights recover the volume
                of ``[0, 1]^d``.

        Raises:
            ValueError: If ``q < 1``.
        """
        _validate_q(q)
        gl_nodes, gl_weights = _gauss_legendre_01(q)
        ts_nodes, ts_weights = _tanh_sinh_01(q)
        strat = int(strategy)

        if self._dim == 2:  # noqa: PLR2004
            return volume_quad_2d(  # type: ignore[call-arg]
                *self._build_result, gl_nodes, gl_weights, ts_nodes, ts_weights, strat
            )
        else:
            return volume_quad_3d(  # type: ignore[call-arg]
                *self._build_result, gl_nodes, gl_weights, ts_nodes, ts_weights, strat
            )

    def surface_quad(
        self,
        q: int,
        strategy: QuadStrategy = QuadStrategy.AUTO_MIXED,
        aggregate: bool = False,
    ) -> SurfQuadResult:
        """Generate surface quadrature points and weights.

        Computes quadrature over the zero level set {phi = 0} in flux form.
        The scalar weights integrate |f| and the normal weights integrate
        f * n where n is the outward normal.

        When *aggregate* is True, the algorithm runs for each possible height
        direction and sums the flux-form contributions. This is more robust
        when vertical tangents exist in every coordinate direction (paper §3.7).

        .. note::

           In non-aggregate mode, quadrature points at vertical tangents
           (where the gradient component along the height direction is zero)
           are silently skipped. Use ``aggregate=True`` for geometries with
           vertical tangents in every coordinate direction.

        Args:
            q (int): Number of 1D quadrature points per sub-interval.
                Must be >= 1.
            strategy (QuadStrategy): Quadrature method selection.
            aggregate (bool): Use aggregate mode (run all d directions).

        Returns:
            SurfQuadResult: ``(points, scalar_weights, normal_weights)``
                with shapes ``(n_pts, dim)``, ``(n_pts,)``, ``(n_pts, dim)``.

        Raises:
            ValueError: If ``q < 1``.
        """
        _validate_q(q)
        gl_nodes, gl_weights = _gauss_legendre_01(q)
        ts_nodes, ts_weights = _tanh_sinh_01(q)
        strat = int(strategy)

        if aggregate:
            return self._surface_quad_aggregate(
                q, gl_nodes, gl_weights, ts_nodes, ts_weights, strat
            )

        if self._dim == 2:  # noqa: PLR2004
            return surface_quad_2d(  # type: ignore[call-arg]
                *self._build_result,
                self._n_polys,
                self._coeffs_list,
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )
        else:
            return surface_quad_3d(  # type: ignore[call-arg]
                *self._build_result,
                self._n_polys,
                self._coeffs_list,
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )

    def _surface_quad_aggregate(  # noqa: PLR0913
        self,
        q: int,
        gl_nodes: npt.NDArray[np.float64],
        gl_weights: npt.NDArray[np.float64],
        ts_nodes: npt.NDArray[np.float64],
        ts_weights: npt.NDArray[np.float64],
        strat: int,
    ) -> SurfQuadResult:
        """Aggregate surface quadrature: run for each height direction, combine.

        For each direction k, builds a hierarchy with forced k, computes the
        flux-form surface integral, and combines the results. The k-th
        component of the normal weight comes from the k-th hierarchy.

        Args:
            q (int): Quadrature order.
            gl_nodes (npt.NDArray[np.float64]): GL nodes on [0, 1].
            gl_weights (npt.NDArray[np.float64]): GL weights on [0, 1].
            ts_nodes (npt.NDArray[np.float64]): Tanh-sinh nodes on [0, 1].
            ts_weights (npt.NDArray[np.float64]): Tanh-sinh weights on [0, 1].
            strat (int): Strategy code (see :class:`QuadStrategy`).

        Returns:
            SurfQuadResult: ``(points, scalar_weights, normal_weights)``.
        """
        all_pts_list: list[npt.NDArray[np.float64]] = []
        all_sw_list: list[npt.NDArray[np.float64]] = []
        all_nw_list: list[npt.NDArray[np.float64]] = []

        if self._dim == 2:  # noqa: PLR2004
            for k in range(2):
                r_raw = build_2d_forced_k(self._coeffs_list, self._masks_list, k)
                # Aggregate mode always uses TS for robustness with vertical tangents.
                r = _force_ts_flags(r_raw, 2)
                pts, sw, nw = surface_quad_2d_aggregate(  # type: ignore[call-arg]
                    *r,
                    self._n_polys,
                    self._coeffs_list,
                    gl_nodes,
                    gl_weights,
                    ts_nodes,
                    ts_weights,
                    strat,
                )
                if len(sw) > 0:
                    all_pts_list.append(pts)
                    all_sw_list.append(sw)
                    all_nw_list.append(nw)
        else:
            for k in range(3):
                r_raw_3d = build_3d_forced_k(self._coeffs_list, self._masks_list, k)
                # Aggregate mode always uses TS for robustness with vertical tangents.
                r_3d = _force_ts_flags(r_raw_3d, 3)
                pts, sw, nw = surface_quad_3d_aggregate(  # type: ignore[call-arg]
                    *r_3d,
                    self._n_polys,
                    self._coeffs_list,
                    gl_nodes,
                    gl_weights,
                    ts_nodes,
                    ts_weights,
                    strat,
                )
                if len(sw) > 0:
                    all_pts_list.append(pts)
                    all_sw_list.append(sw)
                    all_nw_list.append(nw)

        if not all_pts_list:
            return (
                np.empty((0, self._dim), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty((0, self._dim), dtype=np.float64),
            )

        return (
            np.concatenate(all_pts_list, axis=0),
            np.concatenate(all_sw_list, axis=0),
            np.concatenate(all_nw_list, axis=0),
        )

    def eval_poly(self, poly_idx: int, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate a polynomial at the given points.

        Args:
            poly_idx (int): Index of the polynomial (0-based).
            points (npt.NDArray[np.float64]): Points of shape ``(n_pts, dim)``.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(n_pts,)``.

        Raises:
            IndexError: If ``poly_idx`` is out of range ``[0, n_polys)``.
            ValueError: If ``points`` does not have shape ``(n, dim)``.
        """
        if poly_idx < 0 or poly_idx >= self._n_polys:
            msg = f"poly_idx {poly_idx} out of range [0, {self._n_polys})."
            raise IndexError(msg)
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self._dim:  # noqa: PLR2004
            msg = f"points must have shape (n, {self._dim}), got {points.shape}."
            raise ValueError(msg)
        coeffs = self._coeffs[poly_idx]
        n = points.shape[0]
        values = np.empty(n, dtype=np.float64)
        if self._dim == 2:  # noqa: PLR2004
            for i in range(n):
                values[i] = _eval_bernstein_2d(coeffs, points[i])
        else:
            for i in range(n):
                values[i] = _eval_bernstein_3d(coeffs, points[i])
        return values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_q(q: int) -> None:
    """Validate that ``q >= 1``.

    Args:
        q (int): Number of quadrature points.

    Raises:
        ValueError: If ``q < 1``.
    """
    if q < 1:
        msg = f"q must be >= 1, got {q}."
        raise ValueError(msg)


def _force_ts_flags(build_result: tuple[Any, ...], dim: int) -> tuple[Any, ...]:
    """Replace ``use_ts`` flags with ``True`` at each level of a build result.

    The build result is a flat tuple with 5 fields per level
    (coeffs, masks, k, use_ts, type). This function sets the ``use_ts``
    field (index 3 within each level) to ``True``.

    Args:
        build_result (tuple[Any, ...]): Raw build result tuple.
        dim (int): Parametric dimension (2 or 3). Determines number of levels.

    Returns:
        tuple[Any, ...]: Modified build result with all ``use_ts`` flags set to ``True``.
    """
    result = list(build_result)
    for level in range(dim):
        result[3 + level * 5] = True
    return tuple(result)


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
