"""Multi-dimensional quadrature rules on the unit cube (:class:`QuadratureRule`)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ._rules import get_gauss_legendre_1d


class QuadratureRule:
    """Immutable quadrature rule on the unit cube ``[0, 1]^ndim``.

    Bundles quadrature points and weights as the *reference* rule that
    :func:`pantr.grid.cell_quadrature` affinely maps onto each cell of a grid.
    Points lie in the closed unit cube; the factory-built rules
    (:func:`tensor_product_quadrature`, :func:`gauss_legendre_quadrature`) have
    weights summing to ``1``, the measure of the unit cube, so the rule
    integrates the constant ``1`` to within the rounding of that sum and not
    exactly: a ``ndim``-fold product of rounded 1D weights, summed over
    ``num_points`` terms, cannot land on ``1.0`` bitwise. Measured over Gauss-
    Legendre rules of 1 to 40 points per direction, the largest ``|sum - 1|`` is
    2 ulp in 1D, 3 ulp in 2D and 4 ulp in 3D. The stored arrays are read-only.
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
