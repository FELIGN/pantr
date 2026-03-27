"""Quadrature rules and evaluation grid helpers for 1D integration.

This module provides:

- 1D quadrature rules on ``[0, 1]``: trapezoidal (equispaced), Gauss-Legendre,
  Gauss-Lobatto-Legendre, Chebyshev-Gauss (1st and 2nd kind).
- :class:`PointsLattice`: a multi-dimensional tensor-product evaluation grid.
- :func:`create_lagrange_points_lattice`: factory for Lagrange-node lattices.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from numpy.polynomial import chebyshev, legendre

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


def _validate_n_pts_and_dtype(n_pts: int, dtype: npt.DTypeLike) -> None:
    """Validate the number of points and dtype.

    Args:
        n_pts (int): The number of points. Must be at least 1.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Raises:
        ValueError: If n_pts is less than 1 or dtype is not float32 or float64.
    """
    if n_pts < 1:
        raise ValueError("n_pts must be at least 1")

    dtype_obj = np.dtype(dtype)
    if dtype_obj.type not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")


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
    _validate_n_pts_and_dtype(n_pts, dtype)

    if n_pts < 2:  # noqa: PLR2004
        raise ValueError("n_pts must be at least 2")

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
    _validate_n_pts_and_dtype(n_pts, dtype)

    if n_pts < 2:  # noqa: PLR2004
        raise ValueError("n_pts must be at least 2")

    dtype_obj = np.dtype(dtype)
    i = np.arange(n_pts, dtype=dtype_obj)
    nodes: npt.NDArray[np.float32 | np.float64] = 0.5 - 0.5 * np.cos(np.pi * i / (n_pts - 1))
    return nodes


def get_chebyshev_gauss_2nd_kind_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get Chebyshev-Gauss quadrature of the second kind on [0, 1] for the given number of points.

    Args:
        n_pts (int): Number of quadrature points. Must be at least 2.
        dtype (npt.DTypeLike): Floating dtype for nodes/weights; float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 2 or dtype is not float32 or float64.
    """
    _validate_n_pts_and_dtype(n_pts, dtype)

    if n_pts < 2:  # noqa: PLR2004
        raise ValueError("n_pts must be at least 2")

    cheb2_t = cast(Callable[[int], npt.NDArray[np.float64]], chebyshev.chebpts2)
    nodes = cheb2_t(n_pts)
    n_pts_plus_1 = float(n_pts + 1)
    weights = np.pi / n_pts_plus_1 * (np.sin(np.arange(1, n_pts + 1) * np.pi / n_pts_plus_1) ** 2)

    return _scale_and_cast_nodes_and_weights(nodes, weights, dtype)


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
        if order == "C":  # Last index varies fastest
            tp_coords = np.meshgrid(*self._pts_per_dir, indexing="ij")
        else:  # if order == "F": # First index varies fastest
            tp_coords = np.meshgrid(*self._pts_per_dir, indexing="xy")

        return cast(
            npt.NDArray[np.float32 | np.float64],
            np.array(tp_coords).reshape(self.dim, -1).T,  # (n_pts, dim)
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


__all__ = [
    "PointsLattice",
    "create_lagrange_points_lattice",
    "get_chebyshev_gauss_1st_kind_1d",
    "get_chebyshev_gauss_2nd_kind_1d",
    "get_gauss_legendre_1d",
    "get_gauss_lobatto_legendre_1d",
    "get_modified_chebyshev_nodes_1d",
    "get_trapezoidal_1d",
]
