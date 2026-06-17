"""Knot vector construction and space factory functions for B-splines.

Provides knot vector constructors (:func:`create_uniform_open_knots`,
:func:`create_uniform_periodic_knots`, :func:`create_cardinal_knots`), a convenience
space factory (:func:`create_uniform_space`), and Greville abscissa
utilities (:func:`get_greville_abscissae`, :func:`create_greville_lattice`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy import typing as npt

from ..quad import PointsLattice
from ._bspline_knots import (
    _get_knots_ends_and_dtype,
    _validate_knot_input,
)
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_nd import BsplineSpace


def create_uniform_open_knots(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform open knot vector.

    An open knot vector has the first and last knots repeated (degree+1) times,
    ensuring the B-spline interpolates the first and last control points.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Open knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_open_knots(2, 2, domain=(0.0, 1.0))
        array([0., 0., 0., 0.5, 1., 1., 1.])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_knots_ends_and_dtype(start_value, end_value, dtype)

    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)
    knots = np.array([start] * (degree + 1), dtype)

    interior_multiplicity = degree - continuity
    for knot in unique_knots[1:-1]:
        knots = np.append(knots, [knot] * interior_multiplicity)

    knots = np.append(knots, [end] * (degree + 1))

    return knots


def create_uniform_periodic_knots(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform periodic knot vector.

    A periodic knot vector extends beyond the domain boundaries to ensure
    periodicity of the B-spline basis functions.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If ``None``, inferred from the domain endpoints. Defaults to ``np.float64``.

    Returns:
        npt.NDArray[np.floating]: Periodic knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_periodic_knots(2, 2, domain=(0.0, 1.0))
        array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_knots_ends_and_dtype(start_value, end_value, dtype)
    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    # Create uniform spacing for unique interior knots
    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)

    # Build knot vector with repetitions
    knots = np.array([], dtype=dtype)

    multiplicity = degree - continuity

    # Starting periodic knots.
    length = (end - start) / num_intervals
    knots = np.linspace(
        start - length * (degree - multiplicity + 1),
        start,
        degree + 2 - multiplicity,
        dtype=dtype,
    )[:-1]

    # Interior knots with specified multiplicity
    for knot in unique_knots:
        knots = np.append(knots, [knot] * multiplicity)

    # End periodic knots.
    knots = np.append(
        knots,
        np.linspace(
            end,
            end + length * (degree - multiplicity + 1),
            degree + 2 - multiplicity,
            dtype=dtype,
        )[1:],
    )

    return knots


def create_cardinal_knots(
    num_intervals: int,
    degree: int,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a knot vector for cardinal B-spline basis functions.

    Cardinal B-splines are B-splines defined on uniform knot vectors with
    maximum continuity, where the basis functions in the central region
    have the same shape and are translated versions of each other.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be at least 1.
        degree (int): B-spline degree. Must be non-negative.
        dtype (npt.DTypeLike): Data type for the knot vector.
            It must be either float32 or float64. Defaults to np.float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Cardinal B-spline knot vector
            with uniform spacing.

    Raises:
        ValueError: If num_intervals < 1, degree < 0, or dtype is not float32/float64.

    Example:
        >>> create_cardinal_knots(2, 2)
        array([-2., -1.,  0.,  1.,  2.,  3., 4.])
    """
    if num_intervals < 1:
        raise ValueError("num_intervals must be at least 1")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    dtype_obj = np.dtype(dtype)
    if dtype_obj not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    start_value: np.float32 | np.float64
    end_value: np.float32 | np.float64
    if dtype_obj == np.dtype(np.float64):
        start_value = np.float64(0)
        end_value = np.float64(num_intervals)
    else:
        start_value = np.float32(0)
        end_value = np.float32(num_intervals)

    return create_uniform_periodic_knots(
        num_intervals,
        degree,
        continuity=degree - 1,
        domain=(start_value, end_value),
        dtype=dtype_obj,
    )


def get_greville_abscissae(
    space: BsplineSpace1D,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the Greville abscissae (knot averages) of a 1D B-spline space.

    Each Greville abscissa is the average of ``degree`` consecutive internal
    knots: ``g_i = (1/p) * sum(knots[i+1 : i+p+1])`` for ``i = 0, ..., n-1``,
    where ``n`` is the number of basis functions and ``p`` is the degree.

    For periodic spaces, the Greville points are computed from the full knot
    vector and then wrapped into the domain ``[a, b)``.

    Args:
        space (BsplineSpace1D): The 1D B-spline space.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape ``(num_basis,)``
            containing one Greville abscissa per basis function.

    Example:
        >>> from pantr.bspline import BsplineSpace1D, create_uniform_open_knots
        >>> knots = create_uniform_open_knots(4, 3)
        >>> space = BsplineSpace1D(knots, 3)
        >>> get_greville_abscissae(space)
        array([0.  , 0.08333333, 0.25, 0.5 , 0.75, 0.91666667, 1.  ])
    """
    if not isinstance(space, BsplineSpace1D):
        raise TypeError(f"Expected BsplineSpace1D, got {type(space).__name__}")

    knots = space.knots
    degree = space.degree
    n_basis = space.num_basis

    if degree == 0:
        # For degree 0, Greville points are midpoints of knot spans.
        greville = (knots[:n_basis] + knots[1 : n_basis + 1]) / 2
    else:
        greville = np.array(
            [np.mean(knots[i + 1 : i + degree + 1]) for i in range(n_basis)],
            dtype=knots.dtype,
        )

    if space.periodic:
        a, b = space.domain
        period = b - a
        greville = a + np.mod(greville - a, period)
        greville.sort()

    return greville


def create_greville_lattice(
    space: BsplineSpace,
) -> PointsLattice:
    """Compute the tensor-product Greville abscissae as a :class:`~pantr.quad.PointsLattice`.

    Returns a :class:`~pantr.quad.PointsLattice` whose per-direction arrays are
    the Greville abscissae of each 1D sub-space.

    Args:
        space (BsplineSpace): The multi-dimensional B-spline space.

    Returns:
        PointsLattice: Tensor-product grid of Greville abscissae.

    Example:
        >>> from pantr.bspline import BsplineSpace1D, BsplineSpace, create_uniform_open_knots
        >>> knots = create_uniform_open_knots(2, 2)
        >>> s1d = BsplineSpace1D(knots, 2)
        >>> space = BsplineSpace([s1d, s1d])
        >>> lattice = create_greville_lattice(space)
        >>> lattice.pts_per_dir[0]
        array([0. , 0.25, 0.75, 1.  ])
    """
    if not isinstance(space, BsplineSpace):
        raise TypeError(f"Expected BsplineSpace, got {type(space).__name__}")

    pts_per_dir = [get_greville_abscissae(s) for s in space.spaces]
    return PointsLattice(pts_per_dir)


def create_uniform_space(  # noqa: PLR0913
    degree: int | Sequence[int],
    num_intervals: int | Sequence[int],
    *,
    continuity: int | Sequence[int] | None = None,
    periodic: bool | Sequence[bool] = False,
    domain: (
        tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]
        | Sequence[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]
        | None
    ) = None,
    dtype: npt.DTypeLike = np.float64,
) -> BsplineSpace:
    """Create a tensor-product B-spline space with uniform knot vectors.

    Scalar arguments are broadcast to all parametric directions. The parametric
    dimension is inferred from whichever argument is given as a sequence (they
    must all agree in length when more than one is a sequence).

    Uses :func:`create_uniform_open_knots` for non-periodic directions and
    :func:`create_uniform_periodic_knots` for periodic ones.

    Args:
        degree (int | Sequence[int]): Polynomial degree per direction.
        num_intervals (int | Sequence[int]): Number of elements per direction.
        continuity (int | Sequence[int] | None): Interior knot continuity per
            direction. Defaults to ``degree - 1`` (maximum continuity).
        periodic (bool | Sequence[bool]): Whether each direction is periodic.
            Defaults to ``False``.
        domain: Domain boundaries per direction as ``(start, end)`` tuples.
            A single tuple is broadcast. Defaults to ``(0.0, 1.0)``.
        dtype (npt.DTypeLike): Data type for the knot vectors.
            Defaults to ``np.float64``.

    Returns:
        BsplineSpace: A tensor-product B-spline space.

    Raises:
        ValueError: If sequence lengths are inconsistent.

    Example:
        >>> space = create_uniform_space(3, 4, periodic=True, domain=(0.0, 2.0))
        >>> space.dim
        1
        >>> space.degrees
        (3,)
    """
    # Determine parametric dimension from sequence arguments.
    ndim = _infer_ndim(degree, num_intervals, continuity, periodic, domain)

    degrees = _broadcast_to_tuple(degree, ndim, "degree")
    n_intervals = _broadcast_to_tuple(num_intervals, ndim, "num_intervals")
    periodicities = _broadcast_bool_to_tuple(periodic, ndim, "periodic")

    if continuity is None:
        continuities: tuple[int | None, ...] = tuple(None for _ in range(ndim))
    elif isinstance(continuity, int):
        continuities = tuple(continuity for _ in range(ndim))
    else:
        cont_seq = tuple(continuity)
        if len(cont_seq) != ndim:
            raise ValueError(f"continuity has length {len(cont_seq)}, expected {ndim}")
        continuities = cont_seq

    if domain is None:
        domains: tuple[
            tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None, ...
        ] = tuple(None for _ in range(ndim))
    elif isinstance(domain, tuple) and len(domain) == 2 and not isinstance(domain[0], tuple):  # noqa: PLR2004
        # Single (start, end) pair — broadcast.
        domains = tuple(domain for _ in range(ndim))
    else:
        dom_seq = tuple(domain)
        if len(dom_seq) != ndim:
            raise ValueError(f"domain has length {len(dom_seq)}, expected {ndim}")
        domains = dom_seq

    spaces_1d: list[BsplineSpace1D] = []
    for d in range(ndim):
        if periodicities[d]:
            knots = create_uniform_periodic_knots(
                n_intervals[d],
                degrees[d],
                continuity=continuities[d],
                domain=domains[d],
                dtype=dtype,
            )
        else:
            knots = create_uniform_open_knots(
                n_intervals[d],
                degrees[d],
                continuity=continuities[d],
                domain=domains[d],
                dtype=dtype,
            )
        spaces_1d.append(BsplineSpace1D(knots, degrees[d], periodic=periodicities[d]))

    return BsplineSpace(spaces_1d)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _infer_ndim(
    *args: Any,  # noqa: ANN401
) -> int:
    """Infer parametric dimension from the first sequence-valued argument.

    Scalars and ``None`` are ignored. All sequences must have the same length.

    Args:
        *args: Arguments that may be scalars or sequences.

    Returns:
        int: The inferred parametric dimension (at least 1).

    Raises:
        ValueError: If sequences have inconsistent lengths.
    """
    ndim: int | None = None
    for arg in args:
        if arg is None or isinstance(arg, int | float | bool | np.integer | np.floating):
            continue
        if (
            isinstance(arg, tuple)
            and len(arg) == 2  # noqa: PLR2004
            and isinstance(arg[0], float | np.floating)
        ):
            continue  # single domain pair like (0.0, 1.0), treated as scalar
        if not isinstance(arg, Sequence):
            continue
        length = len(arg)
        if ndim is None:
            ndim = length
        elif length != ndim:
            raise ValueError(f"Inconsistent sequence lengths: got {length} and {ndim}")
    return ndim if ndim is not None else 1


def _broadcast_to_tuple(val: int | Sequence[int], ndim: int, name: str) -> tuple[int, ...]:
    """Broadcast a scalar int or sequence to a tuple of length *ndim*.

    Args:
        val (int | Sequence[int]): Value to broadcast.
        ndim (int): Target length.
        name (str): Parameter name for error messages.

    Returns:
        tuple[int, ...]: Tuple of length *ndim*.

    Raises:
        ValueError: If *val* is a sequence with wrong length.
    """
    if isinstance(val, int | np.integer):
        return tuple(int(val) for _ in range(ndim))
    seq = tuple(val)
    if len(seq) != ndim:
        raise ValueError(f"{name} has length {len(seq)}, expected {ndim}")
    return seq


def _broadcast_bool_to_tuple(val: bool | Sequence[bool], ndim: int, name: str) -> tuple[bool, ...]:
    """Broadcast a scalar bool or sequence to a tuple of length *ndim*.

    Args:
        val (bool | Sequence[bool]): Value to broadcast.
        ndim (int): Target length.
        name (str): Parameter name for error messages.

    Returns:
        tuple[bool, ...]: Tuple of length *ndim*.

    Raises:
        ValueError: If *val* is a sequence with wrong length.
    """
    if isinstance(val, bool | np.bool_):
        return tuple(bool(val) for _ in range(ndim))
    seq = tuple(val)
    if len(seq) != ndim:
        raise ValueError(f"{name} has length {len(seq)}, expected {ndim}")
    return seq
