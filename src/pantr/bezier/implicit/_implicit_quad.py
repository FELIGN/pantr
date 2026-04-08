"""Implementation of :class:`ImplicitQuadrature` and supporting helpers.

This module contains the main quadrature class, the :class:`ReparamResult`
dataclass, the :class:`QuadStrategy` enum, type aliases, and all Python-level
helpers (root finding via eigenvalues, base-level partition precomputation,
quadrature node generation, Lagrange-cell reparameterization, etc.).

The Numba kernel files (``_bernstein_core``, ``_build_core``, ``_construct_core``,
``_reparameterize_core``, etc.) handle the heavy computation; this module
orchestrates them.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr.bezier.implicit._bernstein_core import _eval_bernstein_2d, _eval_bernstein_3d
from pantr.bezier.implicit._build_core import (
    build_2d,
    build_2d_forced_k,
    build_3d,
    build_3d_forced_k,
)
from pantr.bezier.implicit._construct_core import (
    _MERGE_TOL,
    _collect_and_partition_1d,
    surface_quad_2d,
    surface_quad_2d_aggregate,
    surface_quad_3d,
    surface_quad_3d_aggregate,
    volume_quad_2d,
    volume_quad_3d,
)
from pantr.bezier.implicit._mask_core import (
    _point_within_1d,
    compute_nonzero_mask_2d,
    compute_nonzero_mask_3d,
)
from pantr.bezier.implicit._reparameterize_core import (
    surface_reparam_2d,
    surface_reparam_3d,
    volume_reparam_2d,
    volume_reparam_3d,
)
from pantr.bezier.implicit._roots_core import find_roots

# Numba-jitted helpers to compute masks in a single Numba call, avoiding
# per-polynomial Python->Numba transitions.
if not TYPE_CHECKING:
    from pantr._numba_compat import nb_jit as _nb_jit

    @_nb_jit(nopython=True, cache=True)
    def _compute_masks_2d(coeffs_list: NumbaList) -> NumbaList:
        """Compute nonzero masks for all 2D polynomials in a single Numba call.

        Args:
            coeffs_list: Typed list of 2D coefficient arrays.

        Returns:
            Typed list of 2D boolean mask arrays.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        masks = NumbaList()
        # Force-type the list for the empty case.
        _dm = np.empty((1, 1), dtype=np.bool_)
        masks.append(_dm)
        masks.pop()
        for i in range(len(coeffs_list)):
            masks.append(compute_nonzero_mask_2d(coeffs_list[i]))
        return masks

    @_nb_jit(nopython=True, cache=True)
    def _compute_masks_3d(coeffs_list: NumbaList) -> NumbaList:
        """Compute nonzero masks for all 3D polynomials in a single Numba call.

        Args:
            coeffs_list: Typed list of 3D coefficient arrays.

        Returns:
            Typed list of 3D boolean mask arrays.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        masks = NumbaList()
        _dm = np.empty((1, 1, 1), dtype=np.bool_)
        masks.append(_dm)
        masks.pop()
        for i in range(len(coeffs_list)):
            masks.append(compute_nonzero_mask_3d(coeffs_list[i]))
        return masks


if TYPE_CHECKING:
    from pantr.bezier._bezier import Bezier

    def _compute_masks_2d(coeffs_list: NumbaList) -> NumbaList: ...

    def _compute_masks_3d(coeffs_list: NumbaList) -> NumbaList: ...


def _is_bezier(obj: object) -> bool:
    """Check if *obj* is a :class:`~pantr.bezier.Bezier` instance (lazy import)."""
    from pantr.bezier._bezier import Bezier as _Bezier  # noqa: PLC0415

    return isinstance(obj, _Bezier)


VolQuadResult: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""Volume quadrature result: ``(points, weights)``."""

SurfQuadResult: TypeAlias = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]
"""Surface quadrature result: ``(points, scalar_weights, normal_weights)``."""


@dataclass(frozen=True)
class ReparamResult:
    """Result of implicit domain reparameterization.

    Cell *i* occupies rows ``[i * pts_per_cell : (i + 1) * pts_per_cell]``
    of :attr:`points`, stored in natural tensor-product (row-major) order.

    Attributes:
        points (npt.NDArray[np.float64]): All cell nodes, shape
            ``(n_cells * pts_per_cell, dim)``.  Read-only.
        n_cells (int): Number of Lagrange cells (>= 0).
        q (int): Lagrange order per direction (>= 2).
        dim (int): Parametric dimension (2 or 3).
        cell_dim (int): Topological dimension of each cell
            (``dim`` for volume, ``dim - 1`` for surface).
    """

    points: npt.NDArray[np.float64]
    """All cell nodes, shape ``(n_cells * pts_per_cell, dim)``.  Read-only."""

    n_cells: int
    """Number of Lagrange cells."""

    q: int
    """Lagrange order per direction."""

    dim: int
    """Parametric dimension (2 or 3)."""

    cell_dim: int
    """Topological dimension of each cell."""

    @property
    def pts_per_cell(self) -> int:
        """Get the number of nodes per cell.

        Returns:
            int: ``q ** cell_dim``.
        """
        return int(self.q**self.cell_dim)

    def __post_init__(self) -> None:
        """Validate invariants between fields.

        Raises:
            ValueError: If ``q < 2``, ``n_cells < 0``, ``dim`` not in
                ``{2, 3}``, ``cell_dim`` invalid, or ``points`` has wrong
                shape or dtype.
        """
        if self.q < 2:  # noqa: PLR2004
            msg = f"q must be >= 2, got {self.q}"
            raise ValueError(msg)
        if self.n_cells < 0:
            msg = f"n_cells must be >= 0, got {self.n_cells}"
            raise ValueError(msg)
        if self.dim not in (2, 3):
            msg = f"dim must be 2 or 3, got {self.dim}"
            raise ValueError(msg)
        if self.cell_dim not in (self.dim - 1, self.dim):
            msg = f"cell_dim must be {self.dim - 1} or {self.dim}, got {self.cell_dim}"
            raise ValueError(msg)
        # Coerce dtype and freeze the array.
        pts = np.asarray(self.points, dtype=np.float64)
        pts.flags.writeable = False
        object.__setattr__(self, "points", pts)
        expected_rows = self.n_cells * self.pts_per_cell
        if self.points.shape != (expected_rows, self.dim):
            msg = f"points.shape must be ({expected_rows}, {self.dim}), got {self.points.shape}"
            raise ValueError(msg)


class QuadStrategy(IntEnum):
    """Strategy for choosing 1D quadrature method at each level."""

    GL_ONLY = 0
    """Gauss-Legendre on all intervals."""
    TS_ONLY = 1
    """Tanh-sinh on all intervals."""
    AUTO_MIXED = 2
    """Tanh-sinh on outer integrals with branching points, GL on inner."""


_OVERFLOW_WARNING: str = (
    "Bezier clipping stack overflow detected during root finding. "
    "Some roots may have been missed, leading to inaccurate results."
)
"""Shared warning message for overflow detection in quadrature methods."""

_REPARAM_OVERFLOW_WARNING: str = (
    "Numerical issues detected during reparameterization (root-finding "
    "overflow or internal buffer capacity exceeded). Some cells may be "
    "inaccurate or missing."
)
"""Warning message for overflow detection in reparameterization methods."""

_EIGVALS_DEGREE_THRESHOLD: int = 20
"""Use companion-matrix eigenvalues for root finding above this degree.

Resultant and discriminant computations produce high-degree 1D polynomials
at the base level (degree 32-128+ for 3D inputs).  These polynomials are
often ill-conditioned, with coefficients spanning many orders of magnitude.
The companion-matrix eigenvalue method (via ``np.linalg.eigvals``) is both
faster and more numerically robust than Bezier clipping for these cases.

The threshold must stay above the highest degree produced by the
discriminant of a single polynomial (~10-16 for degree-4 input, as
opposed to nested resultants of multiple polynomials), since the
Bernstein-to-monomial conversion used by eigvals loses accuracy at
moderate degree.  Setting it to 20 ensures only the truly high-degree
nested-resultant polynomials (degree 32-128+) use this path.
"""

_NEAR_ZERO: float = 1e-300
"""Guard against division by zero in monomial leading coefficient."""


@functools.lru_cache(maxsize=32)
def _bernstein_to_monomial_matrix(n: int) -> npt.NDArray[np.float64]:
    """Build the (n+1)x(n+1) Bernstein-to-monomial conversion matrix.

    Cached by degree to avoid redundant O(n^2) construction when multiple
    polynomials share the same degree.

    Args:
        n: Polynomial degree.

    Returns:
        Conversion matrix of shape ``(n+1, n+1)``.
    """
    from math import comb  # noqa: PLC0415

    mat = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        for i in range(j, n + 1):
            mat[j, i] = (-1) ** (i - j) * comb(n, i) * comb(i, j)
    mat.flags.writeable = False
    return mat


def _bernstein_to_monomial(bern: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Bernstein coefficients to monomial (power) basis.

    Args:
        bern: Bernstein coefficient array of shape ``(n+1,)``.

    Returns:
        Monomial coefficient array of shape ``(n+1,)``.
    """
    return _bernstein_to_monomial_matrix(len(bern) - 1) @ bern


def _find_roots_eigvals(
    bern_coeffs: npt.NDArray[np.float64],
    tol: float = 1e-10,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Find real roots in [0,1] via companion matrix eigenvalues.

    Falls back to :func:`find_roots` (Bezier clipping / Yuksel) if the
    Bernstein-to-monomial conversion overflows or produces non-finite
    coefficients, the leading monomial coefficient is near-zero, or the
    eigenvalue solver fails.

    Args:
        bern_coeffs: Bernstein coefficients.
        tol: Tolerance for imaginary-part filtering.

    Returns:
        tuple[npt.NDArray[np.float64], bool]: ``(roots, overflowed)`` --
            sorted array of real roots in [0, 1] and whether the Bezier
            clipping stack overflowed in any fallback call.
    """
    deg = len(bern_coeffs) - 1
    if deg <= 0:
        return np.empty(0, dtype=np.float64), False

    try:
        mono = _bernstein_to_monomial(bern_coeffs)
    except (OverflowError, FloatingPointError):
        roots, n_roots, overflow = find_roots(bern_coeffs)
        return np.array(roots[:n_roots], dtype=np.float64), overflow

    if not np.all(np.isfinite(mono)) or abs(mono[-1]) < _NEAR_ZERO:
        roots, n_roots, overflow = find_roots(bern_coeffs)
        return np.array(roots[:n_roots], dtype=np.float64), overflow

    mono_monic = mono / mono[-1]

    n = deg
    companion = np.zeros((n, n))
    if n > 1:
        companion[1:, :-1] = np.eye(n - 1)
    companion[:, -1] = -mono_monic[:-1]

    try:
        ev = np.linalg.eigvals(companion)
    except np.linalg.LinAlgError:
        roots, n_roots, overflow = find_roots(bern_coeffs)
        return np.array(roots[:n_roots], dtype=np.float64), overflow

    real_mask = np.abs(ev.imag) < tol
    real_roots = ev[real_mask].real
    in_unit = real_roots[(real_roots >= -tol) & (real_roots <= 1 + tol)]
    candidates = np.sort(np.clip(in_unit, 0.0, 1.0))

    # Residual check: reject roots where the polynomial value is too large
    # relative to the coefficient scale, guarding against garbage eigvals
    # from ill-conditioned companion matrices.
    coeff_scale = np.max(np.abs(bern_coeffs))
    residual_tol = 1e-6 * coeff_scale
    verified: list[float] = []
    for r in candidates:
        # De Casteljau evaluation at r.
        tmp = bern_coeffs.copy()
        for k in range(deg):
            for j in range(deg - k):
                tmp[j] = tmp[j] * (1.0 - r) + tmp[j + 1] * r
        if abs(tmp[0]) <= residual_tol:
            verified.append(float(r))

    return np.array(verified, dtype=np.float64), False


def _precompute_base_partition(  # noqa: PLR0912
    coeffs_1d: NumbaList,
    masks_1d: NumbaList,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Pre-compute the 1D base-level partition using eigvals for high-degree polys.

    For polynomials with degree > ``_EIGVALS_DEGREE_THRESHOLD``, uses the
    companion-matrix eigenvalue method (via ``np.linalg.eigvals``) which is
    much faster than Bezier clipping for high-degree polynomials with many
    roots or ill-conditioned coefficients.

    For low-degree polynomials, delegates to the Numba-jitted
    ``_collect_and_partition_1d`` which uses ``find_roots`` (Yuksel for
    degree < 6, Bezier clipping otherwise).

    Args:
        coeffs_1d: Typed list of 1D Bernstein coefficient arrays.
        masks_1d: Typed list of 1D boolean mask arrays.

    Returns:
        tuple: ``(bounds, n_bounds, any_overflow)`` -- sorted partition
        boundaries including 0 and 1, the count, and a boolean indicating
        whether the Bezier clipping stack overflowed during any root-finding
        call.
    """
    if len(coeffs_1d) == 0:
        bounds = np.array([0.0, 1.0], dtype=np.float64)
        return bounds, 2, False

    max_degree = max(len(coeffs_1d[i]) - 1 for i in range(len(coeffs_1d)))

    if max_degree <= _EIGVALS_DEGREE_THRESHOLD:
        return _collect_and_partition_1d(coeffs_1d, masks_1d)

    # Mixed path: use eigvals for high-degree polynomials, Numba for the rest.
    # First, collect eigval roots for high-degree polynomials (Python).
    eigval_roots: list[float] = []
    low_degree_indices: list[int] = []
    any_overflow = False

    for i in range(len(coeffs_1d)):
        deg = len(coeffs_1d[i]) - 1
        if deg > _EIGVALS_DEGREE_THRESHOLD:
            c = np.asarray(coeffs_1d[i])
            roots, ovf = _find_roots_eigvals(c)
            any_overflow |= ovf
            m = masks_1d[i]
            for root in roots:
                if _point_within_1d(m, float(root)):
                    eigval_roots.append(float(root))
        else:
            low_degree_indices.append(i)

    # Get the base partition from low-degree polynomials via Numba.
    if low_degree_indices:
        low_coeffs = NumbaList()
        low_masks = NumbaList()
        for idx in low_degree_indices:
            low_coeffs.append(coeffs_1d[idx])
            low_masks.append(masks_1d[idx])
        base_bounds, base_nb, overflow = _collect_and_partition_1d(low_coeffs, low_masks)
        any_overflow |= overflow
    else:
        base_bounds = np.array([0.0, 1.0], dtype=np.float64)
        base_nb = 2

    # Merge eigval roots into the base partition.
    if not eigval_roots:
        return base_bounds, base_nb, any_overflow

    all_nodes: list[float] = list(base_bounds[:base_nb])
    all_nodes.extend(eigval_roots)
    all_sorted = np.sort(np.array(all_nodes, dtype=np.float64))

    # Single-pass tolerance merge (avoids set() on floats).
    merged = [all_sorted[0]]
    for v in all_sorted[1:]:
        if v - merged[-1] > _MERGE_TOL:
            merged.append(v)
    result = np.array(merged, dtype=np.float64)
    return result, len(result), any_overflow


class ImplicitQuadrature:
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
            elif _is_bezier(p):
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
        # Use batched Numba helpers to avoid per-polynomial Python->Numba overhead.
        if self._dim == 2:  # noqa: PLR2004
            self._masks_list = _compute_masks_2d(self._coeffs_list)
            self._build_result: tuple[Any, ...] = build_2d(self._coeffs_list, self._masks_list)
        else:
            self._masks_list = _compute_masks_3d(self._coeffs_list)
            self._build_result = build_3d(self._coeffs_list, self._masks_list)

        # Pre-compute base-level partition (1D roots) using eigvals for
        # high-degree polynomials produced by the resultant chain.
        coeffs_1d = self._build_result[0]
        masks_1d = self._build_result[1]
        self._base_bounds, self._base_nb, self._base_overflow = _precompute_base_partition(
            coeffs_1d, masks_1d
        )

        # Pre-unpack build result args (avoids tuple unpacking on every quad call).
        self._build_args = self._build_result[2:]

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

        The quadrature is adapted to the implicit geometry: the interval
        partition ensures each sub-interval is smooth with respect to the
        defining polynomials.  Points span the full domain ``[0,1]^d``, so
        to compute an integral restricted to ``{phi < 0}`` the caller
        should evaluate an appropriate indicator or weight function at the
        returned points.

        Args:
            q (int): Number of 1D quadrature points per sub-interval.
                Must be >= 1.
            strategy (QuadStrategy): Quadrature method selection.

        Returns:
            VolQuadResult: ``(points, weights)`` with shapes
                ``(n_pts, dim)`` and ``(n_pts,)``.  When integrated against
                the constant function 1 the weights sum to the volume of
                ``[0, 1]^d``.

        Raises:
            ValueError: If ``q < 1``.
        """
        _validate_q(q)
        gl_nodes, gl_weights = _gauss_legendre_01(q)
        ts_nodes, ts_weights = _tanh_sinh_01(q)
        strat = int(strategy)

        if self._dim == 2:  # noqa: PLR2004
            pts, wts, overflow = volume_quad_2d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )
        else:
            pts, wts, overflow = volume_quad_3d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )

        if overflow or self._base_overflow:
            warnings.warn(_OVERFLOW_WARNING, RuntimeWarning, stacklevel=2)
        return pts, wts

    def surface_quad(
        self,
        q: int,
        strategy: QuadStrategy = QuadStrategy.AUTO_MIXED,
        aggregate: bool = False,
    ) -> SurfQuadResult:
        """Generate surface quadrature points and weights.

        Computes quadrature over the zero level set ``{phi = 0}`` in flux
        form.  The scalar weights, when summed, approximate the surface
        area.  The normal weights encode the flux form: summing them
        approximates ``integral_S n dS`` where *n* is the outward unit
        normal scaled by the surface measure.

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
            pts, sw, nw, overflow = surface_quad_2d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                self._n_polys,
                self._coeffs_list,
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )
        else:
            pts, sw, nw, overflow = surface_quad_3d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                self._n_polys,
                self._coeffs_list,
                gl_nodes,
                gl_weights,
                ts_nodes,
                ts_weights,
                strat,
            )

        if overflow or self._base_overflow:
            warnings.warn(_OVERFLOW_WARNING, RuntimeWarning, stacklevel=2)
        return pts, sw, nw

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
        any_overflow = False

        if self._dim == 2:  # noqa: PLR2004
            for k in range(2):
                r_raw = build_2d_forced_k(self._coeffs_list, self._masks_list, k)
                # Aggregate mode always uses TS for robustness with vertical tangents.
                r = _force_ts_flags(r_raw, 2)
                bb, bn, base_ovf = _precompute_base_partition(r[0], r[1])
                any_overflow |= base_ovf
                pts, sw, nw, ovf = surface_quad_2d_aggregate(  # type: ignore[call-arg]
                    bb,
                    bn,
                    *r[2:],
                    self._n_polys,
                    self._coeffs_list,
                    gl_nodes,
                    gl_weights,
                    ts_nodes,
                    ts_weights,
                    strat,
                )
                any_overflow |= ovf
                if len(sw) > 0:
                    all_pts_list.append(pts)
                    all_sw_list.append(sw)
                    all_nw_list.append(nw)
        else:
            for k in range(3):
                r_raw_3d = build_3d_forced_k(self._coeffs_list, self._masks_list, k)
                # Aggregate mode always uses TS for robustness with vertical tangents.
                r_3d = _force_ts_flags(r_raw_3d, 3)
                bb, bn, base_ovf = _precompute_base_partition(r_3d[0], r_3d[1])
                any_overflow |= base_ovf
                pts, sw, nw, ovf = surface_quad_3d_aggregate(  # type: ignore[call-arg]
                    bb,
                    bn,
                    *r_3d[2:],
                    self._n_polys,
                    self._coeffs_list,
                    gl_nodes,
                    gl_weights,
                    ts_nodes,
                    ts_weights,
                    strat,
                )
                any_overflow |= ovf
                if len(sw) > 0:
                    all_pts_list.append(pts)
                    all_sw_list.append(sw)
                    all_nw_list.append(nw)

        if any_overflow or self._base_overflow:
            warnings.warn(_OVERFLOW_WARNING, RuntimeWarning, stacklevel=3)

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

    # ------------------------------------------------------------------
    # Reparameterization
    # ------------------------------------------------------------------

    def volume_reparam(
        self,
        q: int,
        signs: Sequence[int],
        *,
        node_type: Literal["equispaced", "chebyshev", "gll"] = "chebyshev",
    ) -> ReparamResult:
        """Reparameterize the implicit volume domain with Lagrange cells.

        Generates Lagrange cells (quads in 2D, hexes in 3D) that tile the
        region of ``[0,1]^d`` satisfying the sign condition for each
        polynomial.

        Args:
            q (int): Number of Lagrange nodes per direction per cell.
                Must be >= 2.
            signs (Sequence[int]): Sign condition per polynomial.
                ``+1`` selects ``{phi > 0}``, ``-1`` selects ``{phi < 0}``,
                ``0`` ignores that polynomial.
            node_type: Node distribution.  ``"chebyshev"`` (default) uses
                modified Chebyshev (Chebyshev-Lobatto) nodes, optimal for
                polynomial interpolation.  ``"equispaced"`` and ``"gll"``
                are also supported.

        Returns:
            ReparamResult: Lagrange cell data.

        Raises:
            ValueError: If ``q < 2`` or ``len(signs) != n_polys``.
        """
        _validate_q_reparam(q)
        signs_arr = _validate_signs(signs, self._n_polys)
        nodes = _lagrange_nodes(q, node_type)

        if self._dim == 2:  # noqa: PLR2004
            pts, n_cells, overflow = volume_reparam_2d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                nodes,
                signs_arr,
                self._n_polys,
            )
        else:
            pts, n_cells, overflow = volume_reparam_3d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                nodes,
                signs_arr,
                self._n_polys,
            )

        if overflow or self._base_overflow:
            warnings.warn(_REPARAM_OVERFLOW_WARNING, RuntimeWarning, stacklevel=2)

        return ReparamResult(
            points=pts,
            n_cells=n_cells,
            q=q,
            dim=self._dim,
            cell_dim=self._dim,
        )

    def surface_reparam(
        self,
        q: int,
        poly_idx: int,
        signs: Sequence[int] | None = None,
        *,
        node_type: Literal["equispaced", "chebyshev", "gll"] = "chebyshev",
    ) -> ReparamResult:
        """Reparameterize a levelset surface with Lagrange cells.

        Generates Lagrange cells (curves in 2D, quads in 3D) tracing the
        zero set ``{phi_{poly_idx} = 0}``, optionally restricted by sign
        conditions on the other polynomials.  The surface is identified
        via sign transitions between adjacent intervals (inside/outside).

        Args:
            q (int): Number of Lagrange nodes per direction per cell.
                Must be >= 2.
            poly_idx (int): Index of the polynomial whose zero set to trace.
                When ``signs`` is ``None``, this determines which polynomial
                defines the "inside" region (``signs[poly_idx] = -1``, all
                others ``0``).  When ``signs`` is provided explicitly,
                ``signs[poly_idx]`` must be non-zero.
            signs (Sequence[int] | None): Sign condition per polynomial.
                Defines the "inside" region; the surface is its boundary
                (where the sign condition transitions).  If ``None``,
                defaults to ``signs[poly_idx] = -1`` with all others ``0``
                (traces only the zero set of the selected polynomial).
            node_type: Node distribution (``"chebyshev"``, ``"equispaced"``,
                or ``"gll"``).

        Returns:
            ReparamResult: Lagrange cell data.

        Raises:
            ValueError: If ``q < 2``, ``len(signs) != n_polys``, or
                ``signs[poly_idx] == 0``.
            IndexError: If ``poly_idx`` is out of range.
        """
        _validate_q_reparam(q)
        if poly_idx < 0 or poly_idx >= self._n_polys:
            msg = f"poly_idx {poly_idx} out of range [0, {self._n_polys})."
            raise IndexError(msg)

        if signs is None:
            # Default: trace only the zero set of poly_idx.
            signs_arr = np.zeros(self._n_polys, dtype=np.int64)
            signs_arr[poly_idx] = -1
        else:
            signs_arr = _validate_signs(signs, self._n_polys)
            if signs_arr[poly_idx] == 0:
                msg = f"signs[poly_idx={poly_idx}] must be non-zero to trace a surface, got 0."
                raise ValueError(msg)

        nodes = _lagrange_nodes(q, node_type)

        if self._dim == 2:  # noqa: PLR2004
            pts, n_cells, overflow = surface_reparam_2d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                nodes,
                signs_arr,
                self._n_polys,
            )
        else:
            pts, n_cells, overflow = surface_reparam_3d(  # type: ignore[call-arg]
                self._base_bounds,
                self._base_nb,
                *self._build_args,
                nodes,
                signs_arr,
                self._n_polys,
            )

        if overflow or self._base_overflow:
            warnings.warn(_REPARAM_OVERFLOW_WARNING, RuntimeWarning, stacklevel=2)

        return ReparamResult(
            points=pts,
            n_cells=n_cells,
            q=q,
            dim=self._dim,
            cell_dim=self._dim - 1,
        )


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


def _validate_q_reparam(q: int) -> None:
    """Validate that ``q >= 2`` for reparameterization.

    Args:
        q (int): Number of Lagrange nodes per direction.

    Raises:
        ValueError: If ``q < 2``.
    """
    if q < 2:  # noqa: PLR2004
        msg = f"q must be >= 2 for reparameterization, got {q}."
        raise ValueError(msg)


def _validate_signs(signs: Sequence[int], n_polys: int) -> npt.NDArray[np.int64]:
    """Validate and convert signs to an int64 array.

    Args:
        signs: Sequence of sign conditions (+1, -1, or 0).
        n_polys: Expected number of polynomials.

    Returns:
        npt.NDArray[np.int64]: Validated sign array.

    Raises:
        ValueError: If length mismatch or invalid sign values.
    """
    arr = np.asarray(signs, dtype=np.int64)
    if arr.shape != (n_polys,):
        msg = f"signs must have length {n_polys}, got {arr.shape}."
        raise ValueError(msg)
    if not np.all((arr == -1) | (arr == 0) | (arr == 1)):
        bad = arr[~((arr == -1) | (arr == 0) | (arr == 1))][0]
        msg = f"signs entries must be -1, 0, or +1, got {bad}."
        raise ValueError(msg)
    return arr


@functools.lru_cache(maxsize=32)
def _lagrange_nodes(q: int, node_type: str) -> npt.NDArray[np.float64]:
    """Get 1D Lagrange reference nodes on [0, 1].

    Args:
        q: Number of nodes.
        node_type: ``"chebyshev"``, ``"equispaced"``, or ``"gll"``.

    Returns:
        npt.NDArray[np.float64]: Nodes of shape ``(q,)``, read-only.

    Raises:
        ValueError: If *node_type* is not recognized.
    """
    if node_type == "chebyshev":
        from pantr.quad import get_modified_chebyshev_nodes_1d  # noqa: PLC0415

        nodes = np.ascontiguousarray(
            get_modified_chebyshev_nodes_1d(q),
            dtype=np.float64,
        )
    elif node_type == "equispaced":
        nodes = np.linspace(0.0, 1.0, q)
    elif node_type == "gll":
        from pantr.quad import get_gauss_lobatto_legendre_1d  # noqa: PLC0415

        gll_nodes, _wts = get_gauss_lobatto_legendre_1d(q)
        nodes = np.ascontiguousarray(gll_nodes, dtype=np.float64)
    else:
        msg = f"Unknown node_type {node_type!r}, expected 'chebyshev', 'equispaced', or 'gll'."
        raise ValueError(msg)
    nodes.flags.writeable = False
    return nodes


# ---------------------------------------------------------------------------
# Quadrature node generation (Python-level, pre-computed once per q)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _gauss_legendre_01(q: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Gauss-Legendre nodes and weights on [0, 1].

    Results are cached across calls for the same *q*.  The returned arrays
    are read-only to prevent accidental mutation of the cached data.

    Args:
        q (int): Number of quadrature points. Must be >= 1.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            (nodes, weights) both of shape ``(q,)``.

    Raises:
        ValueError: If ``q < 1``.
    """
    from pantr.quad import get_gauss_legendre_1d  # noqa: PLC0415

    nodes, weights = get_gauss_legendre_1d(q)
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return cast(npt.NDArray[np.float64], nodes), cast(npt.NDArray[np.float64], weights)


@functools.lru_cache(maxsize=32)
def _tanh_sinh_01(q: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute tanh-sinh nodes and weights on [0, 1].

    Results are cached across calls for the same *q*.  The returned arrays
    are read-only to prevent accidental mutation of the cached data.

    Args:
        q (int): Number of quadrature points. Must be >= 1.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            (nodes, weights) both of shape ``(q,)``.

    Raises:
        ValueError: If ``q < 1``.
    """
    from pantr.quad import get_tanh_sinh_1d  # noqa: PLC0415

    nodes, weights = get_tanh_sinh_1d(q)
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return cast(npt.NDArray[np.float64], nodes), cast(npt.NDArray[np.float64], weights)
