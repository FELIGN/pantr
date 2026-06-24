"""Quadrature rules and evaluation grid helpers for 1D integration.

This module provides:

- 1D quadrature rules on ``[0, 1]``: trapezoidal (equispaced), Gauss-Legendre,
  Gauss-Lobatto-Legendre, Chebyshev-Gauss (1st and 2nd kind).
- :class:`PointsLattice`: a multi-dimensional tensor-product evaluation grid.
- :func:`create_lagrange_points_lattice`: factory for Lagrange-node lattices.
- :class:`QuadratureRule`: a d-dimensional quadrature rule on the unit cube,
  with :func:`tensor_product_quadrature` and :func:`gauss_legendre_quadrature`
  factories; the reference rule consumed by :func:`pantr.grid.cell_quadrature`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from numpy.polynomial import chebyshev, legendre

from ._array_utils import _validate_float_dtype

if TYPE_CHECKING:
    from .basis import LagrangeVariant


def _scale_and_cast_nodes_and_weights(
    nodes: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], dtype: npt.DTypeLike
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Scale and cast nodes and weights to the given dtype.

    The nodes and weights are scaled from the interval [-1, 1] to the interval [0, 1]
    and then cast to the given dtype.

    Args:
        nodes (npt.NDArray[np.float64]): The nodes.
        weights (npt.NDArray[np.float64]): The weights.
        dtype (npt.DTypeLike): The dtype of the nodes and weights.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The scaled and cast nodes and weights.
    """
    nodes = ((nodes + 1.0) * 0.5).astype(dtype)
    weights = (weights * 0.5).astype(dtype)
    return nodes, weights


def _validate_n_pts_and_dtype(n_pts: int, dtype: npt.DTypeLike, min_pts: int = 1) -> None:
    """Validate the number of points and dtype.

    Args:
        n_pts (int): The number of points. Must be at least ``min_pts``.
        dtype (npt.DTypeLike): The dtype of the nodes. Must be float32 or float64.
        min_pts (int): Minimum required number of points. Defaults to 1.

    Raises:
        ValueError: If ``n_pts`` is less than ``min_pts`` or dtype is not
            float32 or float64.
    """
    if n_pts < min_pts:
        raise ValueError(f"n_pts must be at least {min_pts}")

    _validate_float_dtype(dtype)


def get_trapezoidal_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get trapezoidal quadrature nodes on [0, 1] for the given number of points.

    If n_pts == 1, the nodes are [0.5] and the weights are [1.0].

    Args:
        n_pts (int): The number of points. Must be at least 1.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 1 or dtype is not float32 or float64.
    """
    _validate_n_pts_and_dtype(n_pts, dtype)

    if n_pts == 1:
        return np.array([0.5], dtype=dtype), np.array([1.0], dtype=dtype)

    nodes = np.linspace(0, 1, n_pts, dtype=dtype)

    h = 1.0 / float(n_pts - 1)
    weights = np.full(n_pts, h, dtype=dtype)
    weights[0] = weights[-1] = 0.5 * h

    return nodes, weights


def get_gauss_legendre_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get Gauss-Legendre quadrature nodes on [0, 1] for the given number of points.

    Args:
        n_pts (int): The number of points. Must be at least 1.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 1 or dtype is not float32 or float64.
    """
    _validate_n_pts_and_dtype(n_pts, dtype)

    leggauss_t = cast(
        Callable[[int], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
        legendre.leggauss,
    )
    nodes, weights = leggauss_t(n_pts)

    return _scale_and_cast_nodes_and_weights(nodes, weights, dtype)


def get_gauss_lobatto_legendre_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get Gauss-Lobatto-Legendre quadrature nodes on [0, 1] for the given number of points.

    Args:
        n_pts (int): The number of points. Must be at least 2.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 2 or dtype is not float32 or float64.
    """
    _validate_n_pts_and_dtype(n_pts, dtype, min_pts=2)

    # Degree N = n_pts - 1 Legendre polynomial P_N
    # GLL nodes are [-1, roots of P_N'(x), 1] on [-1, 1]
    N = n_pts - 1
    basis_t = cast(Callable[[int], Any], legendre.Legendre.basis)
    P_N = basis_t(N)
    P_prime = P_N.deriv()
    interior_nodes = cast(npt.NDArray[np.float64], P_prime.roots())
    nodes = np.concatenate((np.array([-1.0]), interior_nodes, np.array([1.0])))

    # Weights on [-1, 1]: w_i = 2 / (N (N+1) [P_N(x_i)]^2)
    P_vals = cast(npt.NDArray[np.float64], P_N(nodes))
    weights = 2.0 / (float(N) * float(N + 1)) / (P_vals * P_vals)

    return _scale_and_cast_nodes_and_weights(nodes, weights, dtype)


def get_chebyshev_gauss_1st_kind_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get Chebyshev-Gauss quadrature of the first kind on [0, 1] for the given number of points.

    If n_pts == 1, the nodes are [0.5] and the weights are [1.0].

    Args:
        n_pts (int): Number of quadrature points. Must be at least 1.
        dtype (npt.DTypeLike): Floating dtype for nodes/weights; float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 1 or dtype is not float32 or float64.
    """
    _validate_n_pts_and_dtype(n_pts, dtype)

    cheb1_t = cast(Callable[[int], npt.NDArray[np.float64]], chebyshev.chebpts1)
    nodes = cheb1_t(n_pts)
    weights = np.full(n_pts, np.pi / float(n_pts))

    return _scale_and_cast_nodes_and_weights(nodes, weights, dtype)


def get_modified_chebyshev_nodes_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[np.float32 | np.float64]:
    """Get modified Chebyshev nodes on [0, 1] for the given number of points.

    Returns Chebyshev nodes of the second kind (Chebyshev-Lobatto points)
    mapped to [0, 1].  These include both endpoints and are suitable for
    polynomial interpolation into the Bernstein basis.

    Unlike the quadrature functions in this module, this returns only nodes
    (no weights) since it is intended for interpolation, not integration.

    Args:
        n_pts (int): Number of nodes.  Must be at least 2.
        dtype (npt.DTypeLike): Floating dtype; float32 or float64.
            Defaults to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape ``(n_pts,)`` with
        nodes in [0, 1], starting at 0 and ending at 1.

    Raises:
        ValueError: If *n_pts* < 2 or *dtype* is not float32 or float64.
    """
    _validate_n_pts_and_dtype(n_pts, dtype, min_pts=2)

    dtype_obj = np.dtype(dtype)
    i = np.arange(n_pts, dtype=dtype_obj)
    nodes: npt.NDArray[np.float32 | np.float64] = 0.5 - 0.5 * np.cos(np.pi * i / (n_pts - 1))
    return nodes


def get_chebyshev_gauss_2nd_kind_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    r"""Get Chebyshev-Gauss quadrature of the second kind on [0, 1] for the given number of points.

    The rule integrates against the Chebyshev second-kind weight function
    mapped to [0, 1]: :math:`\int_0^1 f(x) \sqrt{1 - (2x - 1)^2}\, dx \approx
    \sum_k w_k f(x_k)`, exactly for polynomials of degree up to
    ``2 * n_pts - 1``.  The nodes are the mapped roots of the Chebyshev
    polynomial of the second kind :math:`U_{n_pts}` -- all interior (no
    endpoints).  For endpoint-including Chebyshev-Lobatto *interpolation*
    nodes, use :func:`get_modified_chebyshev_nodes_1d` instead.

    Args:
        n_pts (int): Number of quadrature points. Must be at least 2.
        dtype (npt.DTypeLike): Floating dtype for nodes/weights; float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes (ascending, interior) and weights.

    Raises:
        ValueError: If n_pts is less than 2 or dtype is not float32 or float64.
    """
    _validate_n_pts_and_dtype(n_pts, dtype, min_pts=2)

    angles = np.arange(1, n_pts + 1, dtype=np.float64) * np.pi / float(n_pts + 1)
    # Ascending Gauss-Chebyshev-U nodes on [-1, 1]; the weight formula is
    # symmetric under k -> n_pts + 1 - k, so the pairing stays exact.
    nodes = -np.cos(angles)
    weights = np.pi / float(n_pts + 1) * np.sin(angles) ** 2

    return _scale_and_cast_nodes_and_weights(nodes, weights, dtype)


# Step-size tuning constant for the double-exponential rule.  Fixing the
# truncation point so the smallest retained node sits a constant number of decay
# e-folds from the endpoint turns the equation for the step ``h`` into a
# ``u * exp(u)`` form whose root is the Lambert W function (see
# `_generate_tanh_sinh`).  The value resolves the endpoint cluster within the
# float64 range and keeps the rule numerically interchangeable across releases.
_TANH_SINH_DECAY_FACTOR: float = 0.6
"""Decay-rate factor selecting the uniform step ``h`` in transform space."""


def _generate_tanh_sinh(n: int) -> tuple[npt.NDArray[np.float64], int]:
    r"""Generate tanh-sinh quadrature nodes and weights on [-1, 1].

    Builds an *n*-point double-exponential (tanh-sinh) scheme.  Under the
    change of variables :math:`x(t) = \tanh\!\big(\tfrac{\pi}{2}\sinh t\big)`
    the integral over ``[-1, 1]`` becomes an integral over ``t in R`` of an
    integrand that decays double-exponentially, so the trapezoidal rule with
    uniform step *h* converges rapidly.  The Jacobian gives the weight
    :math:`w(t) = \tfrac{\pi}{2}\,\cosh t \,/\, \cosh^2\!\big(\tfrac{\pi}{2}
    \sinh t\big)`.  Nodes are generated symmetrically about ``t = 0`` (with a
    central node at the origin for odd *n*); the step *h* is chosen from the
    truncation balance solved via the Lambert W function (see
    :data:`_TANH_SINH_DECAY_FACTOR`).

    Nodes so close to :math:`\pm 1` that ``1 - |x|`` underflows to ``0`` in
    float64 are snapped to the boundary and their weights accumulated onto the
    shared endpoint pair, so the effective number of nodes *m* may be less than
    *n*.  The weights are finally rescaled to sum to ``2`` (the measure of
    ``[-1, 1]``).

    Args:
        n (int): Requested number of quadrature points (must be >= 1).

    Returns:
        tuple[npt.NDArray[np.float64], int]: A pair ``(data, m)`` where
        *data* has shape ``(m, 2)`` with columns ``[node, weight]`` on
        ``[-1, 1]``, and *m* is the effective node count.

    Note:
        Nodes and weights follow the double-exponential formulas of Takahasi &
        Mori (1974), *Publ. RIMS, Kyoto Univ.* 9(3), 721-741; the step-size root
        is evaluated with :func:`scipy.special.lambertw`.
    """
    if n == 1:
        return np.array([[0.0, 2.0]]), 1

    from scipy.special import lambertw  # noqa: PLC0415

    half_pi = 0.5 * np.pi
    # Uniform step in transform space; the argument of W follows from the
    # large-argument truncation balance described in _TANH_SINH_DECAY_FACTOR.
    decay_arg = 2.0 * _TANH_SINH_DECAY_FACTOR * half_pi * (n - 1)
    h = 2.0 * float(lambertw(decay_arg).real) / n

    buf = np.empty((n, 2), dtype=np.float64)  # worst case: n nodes
    count = 0

    odd = bool(n % 2)
    if odd:
        # Central node at t = 0: x = 0, w = (pi / 2) * cosh(0) / cosh(0)^2.
        buf[count] = [0.0, half_pi]
        count += 1

    endpoint_snapped = False

    for i in range(n // 2):
        # Odd n samples t = h, 2h, ...; even n offsets by half a step.
        t = (i + 1) * h if odd else (i + 0.5) * h
        omega = half_pi * np.sinh(t)
        w = half_pi * np.cosh(t) / np.cosh(omega) ** 2
        # gap = 1 - tanh(omega), the node's distance from the +1 endpoint,
        # via the algebraically equal form 2 / (1 + e^{2 omega}).  This keeps
        # gap a small but nonzero float right up to the underflow boundary,
        # whereas 1 - np.tanh(omega) saturates to 0 a step too early and would
        # snap nodes prematurely.
        gap = 2.0 / (1.0 + np.exp(2.0 * omega))

        if (np.float64(1.0) - np.float64(gap)) == np.float64(1.0):
            # gap underflowed: the node is numerically at the boundary.
            if endpoint_snapped:
                buf[count - 2, 1] += w
                buf[count - 1, 1] += w
            else:
                buf[count] = [-1.0, w]
                count += 1
                buf[count] = [1.0, w]
                count += 1
                endpoint_snapped = True
        else:
            # Symmetric pair at -(1 - gap) and +(1 - gap); writing both from
            # gap keeps the coordinates exact negatives of each other.
            buf[count] = [-(1.0 - gap), w]
            count += 1
            buf[count] = [1.0 - gap, w]
            count += 1

    data = buf[:count].copy()

    # Rescale weights to integrate the constant 1 exactly over [-1, 1].
    data[:, 1] *= 2.0 / np.sum(data[:, 1])

    return data, count


def get_tanh_sinh_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get tanh-sinh quadrature nodes on [0, 1] for the given number of points.

    Tanh-sinh (double-exponential) quadrature clusters nodes near the
    endpoints of the interval, making it well suited for integrands with
    endpoint singularities or steep boundary layers.  The scheme is
    symmetric and nodes near the endpoints that are indistinguishable from
    0 or 1 in floating-point arithmetic are snapped to the boundary, so
    the effective number of returned nodes may be less than *n_pts*.

    Args:
        n_pts (int): Requested number of quadrature points.  Must be at
            least 1.
        dtype (npt.DTypeLike): Floating-point dtype for the output arrays.
            Must be ``float32`` or ``float64``.  Defaults to ``float64``.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            ``(nodes, weights)`` on ``[0, 1]``.  Both arrays have the same
            length, which may be less than *n_pts* due to endpoint
            snapping.  Weights sum to 1.

    Raises:
        ValueError: If *n_pts* < 1 or *dtype* is not ``float32``/``float64``.

    Example:
        >>> nodes, weights = get_tanh_sinh_1d(5)
        >>> nodes.shape[0] <= 5
        True
        >>> abs(weights.sum() - 1.0) < 1e-14
        True
    """
    _validate_n_pts_and_dtype(n_pts, dtype)

    data, _ = _generate_tanh_sinh(n_pts)
    return _scale_and_cast_nodes_and_weights(data[:, 0], data[:, 1], dtype)


class PointsLattice:
    """A tensor-product grid of evaluation points in multiple dimensions.

    Stores one 1D array of coordinates per spatial direction and provides
    helpers for constructing the full set of grid points or querying grid
    metadata.

    Attributes:
        _pts_per_dir (tuple[npt.NDArray[np.float32 | np.float64], ...]): One
            1D coordinate array per spatial dimension. All arrays share the same
            dtype.
    """

    def __init__(self, pts_per_dir: Iterable[npt.NDArray[np.float32 | np.float64]]) -> None:
        """Initialize the points lattice.

        Args:
            pts_per_dir (Iterable[npt.NDArray[np.float32 | np.float64]]): The points per dimension.
                All points must have the same dtype.

        Raises:
            ValueError: If the dimension is less than 1 or the points have different dtypes.
        """
        self._pts_per_dir: tuple[npt.NDArray[np.float32 | np.float64], ...] = tuple(pts_per_dir)
        self._validate_pts_per_dir()

    def _validate_pts_per_dir(self) -> None:
        """Validate the per-direction coordinate arrays.

        Raises:
            ValueError: If the number of dimensions is less than 1, if arrays
                have differing dtypes, if any array is not 1D, or if any array
                is empty.
        """
        if self.dim < 1:
            raise ValueError("Points lattice must have at least 1 dimension")
        if not all(pts.dtype == self.dtype for pts in self._pts_per_dir):
            raise ValueError("All points must have the same dtype")
        for pts in self._pts_per_dir:
            if pts.ndim != 1:
                raise ValueError("All points must be 1D")
            if pts.shape[0] == 0:
                raise ValueError("All points must have at least 1 point")

    @property
    def dim(self) -> int:
        """Get the dimension of the points lattice.

        Returns:
            int: Number of spatial dimensions.
        """
        return len(self._pts_per_dir)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the dtype of the points lattice.

        Returns:
            npt.DTypeLike: The numpy floating-point dtype shared by all
            coordinate arrays.
        """
        return self._pts_per_dir[0].dtype

    @property
    def pts_per_dir(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the points per dimension.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: One 1D coordinate
            array for each spatial dimension.
        """
        return self._pts_per_dir

    def get_all_points(
        self, order: Literal["C", "F"] = "C"
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Get all points in the points lattice.

        Args:
            order (Literal["C", "F"]): The order of the points. Defaults to "C".
                "C" means the last index varies fastest, "F" means the first index varies fastest.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The dim-dimensional points
                in the lattice. It has shape: (n_pts, dim).
        """
        tp_coords = np.meshgrid(*self._pts_per_dir, indexing="ij")
        if order == "C":  # Last index varies fastest
            return cast(
                npt.NDArray[np.float32 | np.float64],
                np.array(tp_coords).reshape(self.dim, -1).T,  # (n_pts, dim)
            )
        # order == "F": first index varies fastest.  meshgrid(indexing="xy") only
        # swaps the first two axes, so a Fortran-order ravel of each "ij"
        # coordinate grid is required for dim >= 3.
        return cast(
            npt.NDArray[np.float32 | np.float64],
            np.stack([c.ravel(order="F") for c in tp_coords], axis=-1),  # (n_pts, dim)
        )


def create_lagrange_points_lattice(
    lagrange_variant: LagrangeVariant,
    n_pts_per_dir: Iterable[int],
    dtype: npt.DTypeLike = np.float64,
) -> PointsLattice:
    """Create a Lagrange points lattice for tensor-product evaluation.

    Builds a :class:`PointsLattice` whose per-direction coordinate arrays are
    the Lagrange nodes of the specified variant on ``[0, 1]``.

    Args:
        lagrange_variant (LagrangeVariant): The variant of the Lagrange basis
            (e.g., equispaced, Gauss-Legendre, Gauss-Lobatto-Legendre, etc.).
        n_pts_per_dir (Iterable[int]): Number of points per spatial dimension.
            Each value must be at least 1.
        dtype (npt.DTypeLike): Floating-point dtype for the coordinates.
            Must be float32 or float64. Defaults to np.float64.

    Returns:
        PointsLattice: A lattice whose per-direction coordinate arrays are the
        Lagrange nodes for the given variant and point counts.

    Raises:
        ValueError: If any value in ``n_pts_per_dir`` is less than 1.
    """
    # Lazy import to avoid circular dependency
    from .basis._basis_lagrange import _get_lagrange_points  # noqa: PLC0415

    if any(n_pts < 1 for n_pts in n_pts_per_dir):
        raise ValueError("All number of points must be at least 1")

    pts_per_dir = tuple(
        _get_lagrange_points(lagrange_variant, n_pts, dtype) for n_pts in n_pts_per_dir
    )
    return PointsLattice(pts_per_dir)


class QuadratureRule:
    """Immutable quadrature rule on the unit cube ``[0, 1]^ndim``.

    Bundles quadrature points and weights as the *reference* rule that
    :func:`pantr.grid.cell_quadrature` affinely maps onto each cell of a grid.
    Points lie in the closed unit cube; the factory-built rules
    (:func:`tensor_product_quadrature`, :func:`gauss_legendre_quadrature`) have
    weights summing to ``1`` (the measure of the unit cube), so the rule
    integrates the constant ``1`` exactly. The stored arrays are read-only.
    """

    __slots__ = ("_ndim", "_num_points", "_points", "_weights")

    def __init__(self, points: npt.ArrayLike, weights: npt.ArrayLike) -> None:
        """Build and validate a quadrature rule on the unit cube.

        Args:
            points (npt.ArrayLike): ``(num_points, ndim)`` array-like; every
                coordinate must lie in ``[0, 1]``.
            weights (npt.ArrayLike): ``(num_points,)`` array-like of weights.

        Raises:
            ValueError: If ``points`` is not 2D, ``weights`` is not 1D, their
                lengths disagree, either is empty, any value is non-finite, or any
                point lies outside ``[0, 1]``.
        """
        pts = np.array(points, dtype=np.float64)
        wts = np.array(weights, dtype=np.float64)
        if pts.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"points must be 2D (num_points, ndim); got shape {pts.shape}.")
        if wts.ndim != 1:
            raise ValueError(f"weights must be 1D (num_points,); got shape {wts.shape}.")
        if pts.shape[0] == 0 or pts.shape[1] == 0:
            raise ValueError(f"points must be non-empty; got shape {pts.shape}.")
        if wts.shape[0] != pts.shape[0]:
            raise ValueError(
                f"weights length {wts.shape[0]} must match the number of points {pts.shape[0]}."
            )
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must contain only finite values.")
        if not np.all(np.isfinite(wts)):
            raise ValueError("weights must contain only finite values.")
        if np.any(pts < 0.0) or np.any(pts > 1.0):
            raise ValueError("points must lie in the unit cube [0, 1]^ndim.")
        pts = np.ascontiguousarray(pts)
        wts = np.ascontiguousarray(wts)
        pts.flags.writeable = False
        wts.flags.writeable = False
        self._points = pts
        self._weights = wts
        self._ndim = int(pts.shape[1])
        self._num_points = int(pts.shape[0])

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the rule.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def num_points(self) -> int:
        """Get the number of quadrature points.

        Returns:
            int: Point count (``>= 1``).
        """
        return self._num_points

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Get the quadrature points on the unit cube.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points, ndim)`` array in
            ``[0, 1]^ndim``.
        """
        return self._points

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Get the quadrature weights.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points,)`` array.
        """
        return self._weights

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Returns:
            str: ``"QuadratureRule(ndim=..., num_points=...)"``.
        """
        return f"QuadratureRule(ndim={self._ndim}, num_points={self._num_points})"


def tensor_product_quadrature(
    rules: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
) -> QuadratureRule:
    """Build a tensor-product :class:`QuadratureRule` from per-axis 1D rules.

    Each axis contributes a 1D rule ``(nodes, weights)`` on ``[0, 1]``; the
    d-dimensional rule is their tensor product. Points are enumerated in
    row-major (C) order -- the last axis varies fastest, matching
    :class:`pantr.grid.TensorProductGrid` cell ids -- and each weight is the
    product of the corresponding per-axis weights.

    Args:
        rules (Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]): One
            ``(nodes, weights)`` pair per axis. Within a pair, ``nodes`` and
            ``weights`` must be 1D of equal, non-zero length, with nodes in
            ``[0, 1]``.

    Returns:
        QuadratureRule: The tensor-product rule on ``[0, 1]^ndim`` with
        ``ndim == len(rules)`` and ``num_points`` the product of the per-axis
        point counts.

    Raises:
        ValueError: If ``rules`` is empty, or any axis pair is not a matching
            pair of non-empty 1D arrays, or (via :class:`QuadratureRule`) any
            node lies outside ``[0, 1]``.
    """
    if len(rules) == 0:
        raise ValueError("tensor_product_quadrature: rules must have at least one axis.")
    nodes_per_axis: list[npt.NDArray[np.float64]] = []
    weights_per_axis: list[npt.NDArray[np.float64]] = []
    for d, pair in enumerate(rules):
        nodes_d = np.asarray(pair[0], dtype=np.float64)
        weights_d = np.asarray(pair[1], dtype=np.float64)
        if nodes_d.ndim != 1 or weights_d.ndim != 1:
            raise ValueError(
                f"tensor_product_quadrature: axis {d} nodes and weights must be 1D; "
                f"got shapes {nodes_d.shape} and {weights_d.shape}."
            )
        if nodes_d.shape[0] == 0 or nodes_d.shape != weights_d.shape:
            raise ValueError(
                f"tensor_product_quadrature: axis {d} needs matching non-empty "
                f"(nodes, weights); got shapes {nodes_d.shape} and {weights_d.shape}."
            )
        nodes_per_axis.append(nodes_d)
        weights_per_axis.append(weights_d)
    node_mesh = np.meshgrid(*nodes_per_axis, indexing="ij")
    weight_mesh = np.meshgrid(*weights_per_axis, indexing="ij")
    points = np.stack([m.ravel() for m in node_mesh], axis=1)
    weights = np.prod(np.stack([w.ravel() for w in weight_mesh], axis=0), axis=0)
    return QuadratureRule(points, weights)


def gauss_legendre_quadrature(ndim: int, npts: int | Sequence[int]) -> QuadratureRule:
    """Build a tensor-product Gauss-Legendre :class:`QuadratureRule` on the unit cube.

    Args:
        ndim (int): Number of axes (``>= 1``).
        npts (int | Sequence[int]): Points per axis. A scalar is broadcast to
            every axis; a length-``ndim`` sequence gives per-axis counts. Each
            count must be ``>= 1``.

    Returns:
        QuadratureRule: The tensor product of per-axis ``npts``-point
        Gauss-Legendre rules. Exact for tensor-product polynomials of per-axis
        degree ``2 * npts - 1``; weights sum to ``1``.

    Raises:
        ValueError: If ``ndim < 1``, ``npts`` is a sequence of the wrong length,
            or any count is ``< 1``.

    References:
        Nodes and weights follow the classical Gauss-Legendre construction
        :cite:p:`golub1969gauss`.
    """
    if ndim < 1:
        raise ValueError(f"gauss_legendre_quadrature: ndim must be >= 1; got {ndim}.")
    npts_tuple = (int(npts),) * ndim if isinstance(npts, int) else tuple(int(n) for n in npts)
    if len(npts_tuple) != ndim:
        raise ValueError(
            f"gauss_legendre_quadrature: npts must be a scalar or a length-{ndim} sequence; "
            f"got length {len(npts_tuple)}."
        )
    if any(n < 1 for n in npts_tuple):
        raise ValueError(
            f"gauss_legendre_quadrature: every npts entry must be >= 1; got {npts_tuple!r}."
        )
    rules = [get_gauss_legendre_1d(n, dtype=np.float64) for n in npts_tuple]
    return tensor_product_quadrature(rules)


__all__ = [
    "PointsLattice",
    "QuadratureRule",
    "create_lagrange_points_lattice",
    "gauss_legendre_quadrature",
    "get_chebyshev_gauss_1st_kind_1d",
    "get_chebyshev_gauss_2nd_kind_1d",
    "get_gauss_legendre_1d",
    "get_gauss_lobatto_legendre_1d",
    "get_modified_chebyshev_nodes_1d",
    "get_tanh_sinh_1d",
    "get_trapezoidal_1d",
    "tensor_product_quadrature",
]
