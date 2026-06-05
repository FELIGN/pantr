"""Quasi-interpolation onto tensor-product B-spline spaces (Lee-Lyche-Mørken).

This module provides :func:`quasi_interpolate_bspline`, a local quasi-interpolant
that maps a callable ``f`` to a :class:`~pantr.bspline.Bspline`.  It uses the
Lee-Lyche-Mørken local-spline-interpolation construction: for each basis function
``B_β`` the coefficient is a local, point-value functional
``λ_β(f) = Σ_k a_{β,k} f(x_{β,k})`` obtained by inverting a small collocation
matrix on one knot-interval cell inside ``supp(B_β)``.  On a single cell the
``p+1`` (per direction) nonzero B-splines span the degree-``p`` polynomials, so the
local interpolation recovers their exact coefficients and the operator is a
projector onto the spline space (``Q B_β = B_β``).

The same low-level helpers (:func:`_interval_interior_points`,
:func:`_local_weight_row`, :func:`_tensor_point_grid`, :func:`_contract_weights`)
back the hierarchical quasi-interpolant in ``_thb_quasi_interpolation``.

Main exports:

- :func:`quasi_interpolate_bspline`: quasi-interpolate a callable onto a
  tensor-product :class:`~pantr.bspline.BsplineSpace`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from numpy import typing as npt

from ._bspline import Bspline
from ._bspline_space_factory import get_greville_abscissae
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._bspline_space_1d import BsplineSpace1D

QIKind = Literal["llm"]
"""Supported quasi-interpolation kinds (Lee-Lyche-Mørken local projector)."""


# ---------------------------------------------------------------------------
# Local Lee-Lyche-Mørken building blocks (shared with the hierarchical QI)
# ---------------------------------------------------------------------------


def _interval_interior_points(lo: float, hi: float, order: int) -> npt.NDArray[np.float64]:
    """Return ``order`` equispaced interior points of ``[lo, hi]``.

    Interior placement keeps the points off the knots, where the local collocation
    block can become singular.

    Args:
        lo (float): Interval lower bound.
        hi (float): Interval upper bound.
        order (int): Number of points (``degree + 1``).

    Returns:
        npt.NDArray[np.float64]: Shape ``(order,)`` strictly inside ``(lo, hi)``.
    """
    frac = np.arange(1, order + 1, dtype=np.float64) / (order + 1)
    return np.asarray(lo + (hi - lo) * frac, dtype=np.float64)


def _local_weight_row(
    space1d: BsplineSpace1D,
    points: npt.NDArray[np.float64],
    target_index: int,
) -> npt.NDArray[np.float64]:
    """Return the local functional weights for one basis function on one interval.

    ``points`` must all lie in a single knot interval where ``target_index`` is one
    of the ``order`` nonzero B-splines.  The local collocation block is inverted so
    that ``λ(f) = weights · f(points)`` recovers the coefficient of
    ``B_{target_index}`` of the local degree-``p`` interpolant.

    Args:
        space1d (BsplineSpace1D): The 1D B-spline space.
        points (npt.NDArray[np.float64]): ``order`` distinct points in one interval.
        target_index (int): Global index of the target basis function.

    Returns:
        npt.NDArray[np.float64]: Weights of shape ``(order,)``.
    """
    basis, first_basis = space1d.tabulate_basis(points)
    block = np.asarray(basis, dtype=np.float64)
    inv = np.asarray(np.linalg.inv(block), dtype=np.float64)
    local = target_index - int(first_basis[0])
    return np.asarray(inv[local], dtype=np.float64)


def _tensor_point_grid(per_dir_points: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Build the tensor-product point grid from per-direction point arrays.

    Args:
        per_dir_points (list[npt.NDArray[np.float64]]): One ``(order_k,)`` array per
            direction.

    Returns:
        npt.NDArray[np.float64]: Points of shape ``(prod(order_k), dim)`` in C-order.
    """
    mesh = np.meshgrid(*per_dir_points, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float64, copy=False)


def _contract_weights(
    weights: list[npt.NDArray[np.float64]],
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Contract a local value tensor with per-direction weight vectors.

    Args:
        weights (list[npt.NDArray[np.float64]]): One ``(order_k,)`` weight vector per
            direction.
        values (npt.NDArray[np.float64]): Local values of shape
            ``(order_0, ..., order_{d-1}, rank)``.

    Returns:
        npt.NDArray[np.float64]: The coefficient of shape ``(rank,)``.
    """
    result = np.asarray(values, dtype=np.float64)
    for w in weights:
        result = np.asarray(np.tensordot(w, result, axes=([0], [0])), dtype=np.float64)
    return result


def _llm_tp_direction_data(
    space1d: BsplineSpace1D,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Precompute per-function local points and weights for a 1D space.

    For each basis function the local interval is the knot span containing its
    Greville abscissa (well inside the support, limiting extrapolation).

    Args:
        space1d (BsplineSpace1D): The 1D B-spline space.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ``(points, weights)``,
        each of shape ``(num_basis, degree + 1)``.
    """
    order = space1d.degree + 1
    n_basis = space1d.num_basis
    greville = np.asarray(get_greville_abscissae(space1d), dtype=np.float64)
    breakpoints, _ = space1d.get_unique_knots_and_multiplicity(in_domain=True)
    breaks = np.asarray(breakpoints, dtype=np.float64)
    n_intervals = breaks.shape[0] - 1

    points = np.empty((n_basis, order), dtype=np.float64)
    weights = np.empty((n_basis, order), dtype=np.float64)
    for j in range(n_basis):
        mu = int(np.searchsorted(breaks, greville[j], side="right")) - 1
        mu = min(max(mu, 0), n_intervals - 1)
        pts = _interval_interior_points(float(breaks[mu]), float(breaks[mu + 1]), order)
        points[j] = pts
        weights[j] = _local_weight_row(space1d, pts, j)
    return points, weights


# ---------------------------------------------------------------------------
# Batched evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_func(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Evaluate ``func`` on a point array and normalize the output.

    Args:
        func (Callable): Called on ``(M, dim)`` points; returns ``(M,)`` or
            ``(M, rank)``.
        points (npt.NDArray[np.float64]): Points of shape ``(M, dim)``.

    Returns:
        tuple[npt.NDArray[np.float64], int, bool]: ``(values, rank, scalar)`` with
        ``values`` of shape ``(M, rank)``.

    Raises:
        ValueError: If the output leading dimension does not match ``M``.
    """
    values = np.asarray(func(np.ascontiguousarray(points, dtype=np.float64)), dtype=np.float64)
    if values.shape[0] != points.shape[0]:
        raise ValueError(f"func returned {values.shape[0]} values for {points.shape[0]} points.")
    if values.ndim == 1:
        return values.reshape(-1, 1), 1, True
    return values, int(values.shape[1]), False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quasi_interpolate_bspline(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    space: BsplineSpace,
    *,
    kind: QIKind = "llm",
) -> Bspline:
    """Quasi-interpolate a callable onto a tensor-product B-spline space.

    Builds the Lee-Lyche-Mørken local projector: a local, point-value functional per
    basis function whose weights come from inverting a small collocation block.  The
    operator reproduces the spline space (``Q B_β = B_β``) and is local.

    Args:
        func (Callable): Function to quasi-interpolate.  Called once on an
            ``(M, dim)`` point array and must return ``(M,)`` (scalar) or
            ``(M, rank)`` (vector-valued).  Unlike the lattice-based
            :func:`interpolate_bspline` / :func:`l2_project_bspline`, ``func``
            receives a flat point array because the QI samples scattered local points.
        space (BsplineSpace): The target tensor-product B-spline space.
        kind (QIKind): Quasi-interpolant kind.  Only ``"llm"`` (Lee-Lyche-Mørken) is
            currently supported.  Defaults to ``"llm"``.

    Returns:
        Bspline: A non-rational B-spline whose evaluation quasi-interpolates ``func``.

    Raises:
        TypeError: If ``space`` is not a :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``kind`` is not recognized.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import create_uniform_space, quasi_interpolate_bspline
        >>> space = create_uniform_space([2], [4])
        >>> qi = quasi_interpolate_bspline(lambda p: p[:, 0] ** 2, space)
        >>> bool(np.isclose(qi.evaluate(np.array([[0.3]]))[0, 0], 0.09))
        True
    """
    if not isinstance(space, BsplineSpace):
        raise TypeError(f"space must be a BsplineSpace; got {type(space).__name__!r}.")
    if kind not in get_args(QIKind):
        valid = ", ".join(repr(v) for v in get_args(QIKind))
        raise ValueError(f"Unknown kind {kind!r}; expected one of {valid}.")

    dim = space.dim
    num_basis = tuple(space.num_basis)
    dir_data = [_llm_tp_direction_data(space.spaces[k]) for k in range(dim)]

    multi_indices = list(np.ndindex(*num_basis))
    orders = tuple(d + 1 for d in space.degrees)
    block = int(np.prod(orders))

    all_points = np.empty((len(multi_indices) * block, dim), dtype=np.float64)
    for f_idx, multi in enumerate(multi_indices):
        per_dir = [dir_data[k][0][multi[k]] for k in range(dim)]
        all_points[f_idx * block : (f_idx + 1) * block] = _tensor_point_grid(per_dir)

    values, rank, scalar = _evaluate_func(func, all_points)
    value_tensor = values.reshape(len(multi_indices), *orders, rank)

    coeffs = np.empty((len(multi_indices), rank), dtype=np.float64)
    for f_idx, multi in enumerate(multi_indices):
        weights = [dir_data[k][1][multi[k]] for k in range(dim)]
        coeffs[f_idx] = _contract_weights(weights, value_tensor[f_idx])

    control_points = coeffs.reshape(*num_basis, rank).astype(space.dtype, copy=False)
    return Bspline(space, control_points)
