"""Bézier degree elevation, reduction, and minimization.

This module provides :func:`_degree_elevate_bezier`, which raises the polynomial
degree of a Bézier in one or more parametric directions while preserving the
same geometric mapping, :func:`_degree_reduce_bezier`, which computes the
:math:`L^2`-optimal degree-reduced approximation that interpolates the segment
endpoints, and :func:`_minimize_degree_bezier`, which automatically finds the
lowest degree that preserves accuracy.

Reduction is driven by a *reduction operator* :math:`R`, a dense
``(q + 1) x (p + 1)`` matrix depending only on the degree pair, so that the
reduced control points are ``R @ ctrl``.  The operator is assembled in exact
rational arithmetic and rounded to ``float64`` once; see
:func:`_interpolating_reduction_operator` for why.

Errors are measured in the space the caller budgets in.  For a polynomial Bézier that
is the exact Bernstein-Gram :math:`L^2` norm of the coefficient difference
(:func:`_squared_l2_norm`); for a rational one it is the deviation of the *projected*
mapping, estimated on a Gauss-Legendre grid (:func:`_projected_relative_deviation`),
because a norm over homogeneous coefficients adds a weight to a weighted coordinate.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._array_utils import _flatten_along_axis, _unflatten_along_axis
from .._backend import backend_keyed_cache
from ..basis._basis_core import _tabulate_Bernstein_basis_1D_serial_core
from ..bspline._bspline_degree_core import _check_bincoeff_envelope
from ..quad import get_gauss_legendre_1d
from ._bezier_backend import _ReduceApplyFunc, degree_kernels

if TYPE_CHECKING:
    from . import Bezier

_ExactMatrix = list[list[Fraction]]
"""A matrix of exact rationals, used to assemble reduction operators."""

_AUTO_REDUCTION_TOL_FACTOR: float = 1.0e3
"""Default relative tolerance for automatic degree reduction, in units of eps.

When :func:`_minimize_degree_bezier` is called without an explicit ``tol``, a
degree-1 reduction is accepted whenever the round-trip (reduce then re-elevate)
relative :math:`L^2` error stays below ``_AUTO_REDUCTION_TOL_FACTOR * eps``,
where ``eps`` is the machine epsilon of the control-point dtype.

The factor of ``1e3`` gives roughly three decimal digits of headroom above the
unit-roundoff floor. The least-squares reduction and the subsequent
re-elevation each accumulate ``O(p)`` floating-point operations, so a curve that
is exactly reducible (for instance a degree-elevated lower-degree curve)
produces a round-trip error of a small multiple of ``eps`` rather than exactly
zero. A threshold at the bare ``eps`` level would spuriously reject such curves;
``1e3 * eps`` (``~2.2e-13`` for ``float64``) comfortably accepts genuine
reductions while still rejecting curves whose true degree cannot be lowered
without visible geometric error.

**For a rational Bézier the factor keeps this meaning verbatim**, because the error it
bounds is now the relative deviation of the *projected* mapping (see
:func:`_projected_relative_deviation`) rather than of the homogeneous coefficients.
Both are relative and both sit at the unit-roundoff floor for an exactly reducible
curve, so nothing about "three digits of headroom above round-off" changes -- only the
space the round-off is measured in, which is the space the caller cares about.

Measured, to check that the projected floor really is a round-off floor and not
something larger: over exactly-reducible rational nets (base degrees 1, 2, 5 elevated by
1, 3 and 8; weight ratios 1, 10 and 100; coordinate scales 1 and ``1e6``; 20 random nets
each), the projected round-trip relative deviation of the first trial never exceeded
``1.55 * eps``, and every net recovered its base degree at this default. That leaves
about ``650x`` of headroom, against the ``1e3`` the factor nominally provides -- the
difference being that the floor is a small multiple of ``eps`` rather than exactly one.
"""


def _degree_elevate_bezier(
    bezier: Bezier,
    increments: tuple[int, ...],
) -> Bezier:
    """Degree-elevate a Bézier in one or more parametric directions.

    For each direction with a positive increment, applies the Bézier degree
    elevation kernel via the shared flatten/unflatten helpers.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to elevate.
        increments (tuple[int, ...]): Degree increment per direction. All
            values must be non-negative; at least one must be positive.

    Returns:
        ~pantr.bezier.Bezier: New Bézier with elevated degrees and updated
        control points.

    Raises:
        ValueError: If an elevated degree would exceed the exactness envelope of the
            binomial-coefficient kernel (see
            :data:`~pantr.bspline._bspline_degree_core._BINCOEFF_MAX_N`).

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    degrees = bezier.degree
    elevate = degree_kernels().elevate

    # The elevation kernel builds its coefficient table from ``C(p + inc, i)``,
    # so the elevated degree is what has to stay in range.
    for d in range(bezier.dim):
        if increments[d] > 0:
            elevated = degrees[d] + increments[d]
            _check_bincoeff_envelope(
                elevated, f"Degree elevation to degree {elevated} in direction {d}"
            )

    for d in range(bezier.dim):
        inc = increments[d]
        if inc == 0:
            continue

        p = degrees[d]

        pts_2d, trailing_shape = _flatten_along_axis(ctrl, d)
        new_pts_2d = np.empty((p + inc + 1, pts_2d.shape[1]), dtype=pts_2d.dtype)
        elevate(p, pts_2d, inc, new_pts_2d)
        ctrl = _unflatten_along_axis(new_pts_2d, trailing_shape, d)

        # Update degrees for subsequent iterations.
        degrees = (*degrees[:d], p + inc, *degrees[d + 1 :])

    return BezierCls(ctrl, is_rational=bezier.is_rational)


def _bernstein_gram_exact(degree: int) -> _ExactMatrix:
    r"""Build the degree-``n`` Bernstein Gram matrix as exact rationals.

    Same closed form as :func:`_bernstein_gram_matrix_1d`, evaluated over
    :class:`~fractions.Fraction` so that the assembly of a reduction operator
    carries no rounding at all.

    Args:
        degree (int): Polynomial degree ``n`` (``>= 0``).

    Returns:
        _ExactMatrix: Symmetric ``(n + 1, n + 1)`` matrix of exact rationals.
    """
    n = degree
    return [
        [
            Fraction(
                math.comb(n, i) * math.comb(n, j),
                math.comb(2 * n, i + j) * (2 * n + 1),
            )
            for j in range(n + 1)
        ]
        for i in range(n + 1)
    ]


def _elevation_matrix_exact(degree: int, increment: int) -> _ExactMatrix:
    r"""Build the degree-elevation matrix as exact rationals.

    Elevation from degree :math:`q` to :math:`p = q + t` maps Bernstein
    coefficients by :math:`c = M \hat q` with

    .. math::

        M_{ij} = \frac{\binom{q}{j}\binom{t}{i-j}}{\binom{p}{i}},
        \qquad \max(0, i - t) \le j \le \min(q, i),

    and zero elsewhere.  This is the closed form of the ``bezalfs`` coefficients
    that :func:`~pantr.bezier._bezier_core._degree_elevate_bezier_1d_core`
    computes, assembled here as a matrix rather than applied.

    Args:
        degree (int): Starting degree ``q`` (``>= 0``).
        increment (int): Number of degrees to add, ``t >= 1``.

    Returns:
        _ExactMatrix: ``(q + t + 1, q + 1)`` matrix of exact rationals.
    """
    q = degree
    t = increment
    p = q + t
    return [
        [
            Fraction(math.comb(q, j) * math.comb(t, i - j), math.comb(p, i))
            if max(0, i - t) <= j <= min(q, i)
            else Fraction(0)
            for j in range(q + 1)
        ]
        for i in range(p + 1)
    ]


def _solve_exact(matrix: _ExactMatrix, rhs: _ExactMatrix) -> _ExactMatrix:
    """Solve ``matrix @ x = rhs`` in exact rational arithmetic.

    Gauss-Jordan elimination with partial pivoting.  Pivoting is not needed for
    stability here — the arithmetic is exact — only to step over a zero pivot.

    Args:
        matrix (_ExactMatrix): Square ``(n, n)`` non-singular matrix.
        rhs (_ExactMatrix): Right-hand sides, ``(n, k)``.

    Returns:
        _ExactMatrix: The solution ``(n, k)``.
    """
    n = len(matrix)
    aug = [list(matrix[i]) + list(rhs[i]) for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        inv_pivot = 1 / aug[col][col]
        aug[col] = [value * inv_pivot for value in aug[col]]
        for row in range(n):
            if row != col and aug[row][col]:
                factor = aug[row][col]
                aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col], strict=True)]

    return [row[n:] for row in aug]


@functools.lru_cache(maxsize=64)
def _l2_reduction_operator(degree: int, decrement: int) -> npt.NDArray[np.float64]:
    r"""Return the cached, read-only :math:`L^2` projection onto the lower degree.

    The reduced coefficients :math:`\hat q = R c` minimise the true
    :math:`L^2[0, 1]` distance :math:`\lVert \sum_j c_j B_{j,p} - \sum_i \hat q_i
    B_{i,q} \rVert` with no constraints, i.e. they solve the normal equations
    :math:`M^\top G_p M \hat q = M^\top G_p c`.  The system matrix is exactly the
    degree-:math:`q` Gram matrix, since :math:`M \hat q` and :math:`\hat q`
    describe the same polynomial: :math:`M^\top G_p M = G_q`.

    Note that this is *not* a new approximation: minimising the Euclidean norm of
    the Bernstein coefficient residual gives the same polynomial (Lutterkort,
    Peters & Reif, *Computer Aided Geometric Design* 16, 1999), which is what the
    bidiagonal least-squares route used before computed.

    Args:
        degree (int): Original degree ``p`` (``>= 1``).
        decrement (int): Degrees to remove, ``1 <= t <= p``.

    Returns:
        npt.NDArray[np.float64]: Read-only ``(p - t + 1, p + 1)`` operator.
    """
    p = degree
    q = p - decrement
    elevation = _elevation_matrix_exact(q, decrement)
    gram_p = _bernstein_gram_exact(p)

    projected = [
        [
            sum((elevation[k][i] * gram_p[k][j] for k in range(p + 1)), Fraction(0))
            for j in range(p + 1)
        ]
        for i in range(q + 1)
    ]
    rows = _solve_exact(_bernstein_gram_exact(q), projected)

    operator = np.array([[float(value) for value in row] for row in rows], dtype=np.float64)
    operator.flags.writeable = False
    return operator


@functools.lru_cache(maxsize=64)
def _interpolating_reduction_operator(degree: int, decrement: int) -> npt.NDArray[np.float64]:
    r"""Return the cached, read-only endpoint-interpolating reduction operator.

    Minimises the same :math:`L^2` distance as :func:`_l2_reduction_operator`
    subject to :math:`\hat q_0 = c_0` and :math:`\hat q_q = c_p`, so the reduced
    polynomial agrees with the original at both ends of the segment.  Splitting
    the columns of :math:`M` into the two constrained ones and the free block
    :math:`M_f` leaves the symmetric positive-definite system

    .. math::

        (M_f^\top G_p M_f)\, \hat q_f
        = \bigl(M^\top G_p c\bigr)_{1:q} - (G_q)_{1:q,0}\, c_0
          - (G_q)_{1:q,q}\, c_p ,

    whose matrix is the interior block of the degree-:math:`q` Gram matrix.  A
    degree-0 target cannot honour two interpolation conditions with a single
    coefficient, so ``decrement == degree`` falls back to
    :func:`_l2_reduction_operator`, whose degree-0 answer is the mean of ``c``.

    The assembly runs in exact rational arithmetic and rounds once, which is what
    makes the operator usable at high degree: that interior Gram block has
    condition number ``5.4e10`` at ``q = 19`` (it depends on ``q`` alone, not on
    ``p`` or ``t``), so a ``float64`` normal-equation solve loses eleven digits
    and a round trip through elevation then reduction comes back with an error of
    ``2.6e-11`` instead of ``4e-16``.  Solving exactly moves that cost to a cached
    one-off — 3 ms at degree 10, 32 ms at degree 21 — and the operator entries
    themselves are benign (``max |R| <= 2.5`` over that range), so applying it is
    accurate.  It also makes the operator bit-identical across platforms and BLAS
    versions.

    Args:
        degree (int): Original degree ``p`` (``>= 1``).
        decrement (int): Degrees to remove, ``1 <= t <= p``.

    Returns:
        npt.NDArray[np.float64]: Read-only ``(p - t + 1, p + 1)`` operator.
    """
    p = degree
    q = p - decrement
    if q == 0:
        return _l2_reduction_operator(degree, decrement)

    elevation = _elevation_matrix_exact(q, decrement)
    gram_p = _bernstein_gram_exact(p)
    gram_q = _bernstein_gram_exact(q)

    rows: _ExactMatrix = [[Fraction(0)] * (p + 1) for _ in range(q + 1)]
    rows[0][0] = Fraction(1)
    rows[q][p] = Fraction(1)

    if q >= 2:  # noqa: PLR2004
        interior = [row[1:q] for row in gram_q[1:q]]
        right = [
            [
                sum((elevation[k][i] * gram_p[k][j] for k in range(p + 1)), Fraction(0))
                for j in range(p + 1)
            ]
            for i in range(1, q)
        ]
        for i in range(q - 1):
            right[i][0] -= gram_q[1 + i][0]
            right[i][p] -= gram_q[1 + i][q]
        rows[1:q] = _solve_exact(interior, right)

    operator = np.array([[float(value) for value in row] for row in rows], dtype=np.float64)
    operator.flags.writeable = False
    return operator


def _reduce_along_axis(
    ctrl: npt.NDArray[np.float32 | np.float64],
    axis: int,
    operator: npt.NDArray[np.float64],
    apply_kernel: _ReduceApplyFunc,
) -> npt.NDArray[np.float32 | np.float64]:
    """Apply a reduction operator along one axis of a control-point array.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(*orders, rank)``.
        axis (int): Parametric direction to reduce.
        operator (npt.NDArray[np.float64]): Reduction operator of shape
            ``(new_order, ctrl.shape[axis])``.
        apply_kernel (_ReduceApplyFunc): The backend's reduction-operator apply.
            Passed in rather than resolved here so that a caller which also
            elevates gets both kernels from one backend selection.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Control points with ``axis``
        reduced, same dtype as *ctrl*.
    """
    pts_2d, trailing_shape = _flatten_along_axis(ctrl, axis)
    reduced = np.empty((operator.shape[0], pts_2d.shape[1]), dtype=pts_2d.dtype)
    apply_kernel(operator, np.ascontiguousarray(pts_2d), reduced)
    return _unflatten_along_axis(reduced, trailing_shape, axis)


def _degree_reduce_bezier(
    bezier: Bezier,
    decrements: tuple[int, ...],
) -> Bezier:
    """Degree-reduce a Bézier in one or more parametric directions.

    For each direction with a positive decrement, applies the
    endpoint-interpolating :math:`L^2`-optimal reduction operator.  The result is
    an approximation (not exact in general), but it reproduces the original
    exactly at the boundary of the parametric domain.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to reduce.
        decrements (tuple[int, ...]): Degree decrement per direction. All
            values must be non-negative; at least one must be positive.  No
            decrement may exceed the current degree in that direction.

    Returns:
        ~pantr.bezier.Bezier: New Bézier with reduced degrees and updated
        control points.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    reduce_apply = degree_kernels().reduce_apply

    for d in range(bezier.dim):
        dec = decrements[d]
        if dec == 0:
            continue

        operator = _interpolating_reduction_operator(bezier.degree[d], dec)
        ctrl = _reduce_along_axis(ctrl, d, operator, reduce_apply)

    return BezierCls(ctrl, is_rational=bezier.is_rational)


def _bernstein_gram_matrix_1d(degree: int) -> npt.NDArray[np.float64]:
    r"""Build the degree-``n`` univariate Bernstein mass (Gram) matrix on :math:`[0, 1]`.

    The entries are the exact inner products of the Bernstein basis functions,
    given by the closed form of Farouki & Rajan (*Computer Aided Geometric
    Design* 5, 1988):

    .. math::

        G_{ij} = \int_0^1 B_{i,n}(x) B_{j,n}(x)\,dx
        = \frac{1}{2n+1}\,
          \frac{\binom{n}{i}\binom{n}{j}}{\binom{2n}{i+j}},
        \qquad 0 \le i, j \le n .

    Args:
        degree (int): Polynomial degree ``n`` (``>= 0``).

    Returns:
        npt.NDArray[np.float64]: Symmetric ``(n + 1, n + 1)`` Gram matrix.
    """
    n = degree
    binom_n = np.array([math.comb(n, i) for i in range(n + 1)], dtype=np.float64)
    binom_2n = np.array([math.comb(2 * n, k) for k in range(2 * n + 1)], dtype=np.float64)

    idx = np.arange(n + 1)
    numerator = np.outer(binom_n, binom_n)
    denominator = binom_2n[idx[:, None] + idx[None, :]]
    gram: npt.NDArray[np.float64] = numerator / denominator / (2.0 * n + 1.0)
    return gram


def _squared_l2_norm(
    coeffs: npt.NDArray[np.floating[Any]],
) -> float:
    r"""Compute the squared :math:`L^2` norm of a Bernstein polynomial on :math:`[0, 1]^d`.

    A polynomial with Bernstein coefficients ``c`` has squared norm equal to the
    Bernstein-Gram quadratic form :math:`\lVert p\rVert_2^2 = c^\top G\, c`,
    where :math:`G` is the Bernstein mass matrix.  For a tensor-product basis the
    mass matrix factorises as a Kronecker product of the univariate Gram matrices
    :math:`G = G^{(0)} \otimes \cdots \otimes G^{(d-1)}`, so the quadratic form is
    evaluated by contracting each univariate :math:`G^{(k)}` against the
    coefficient tensor along its axis and taking the final inner product with the
    coefficients.

    The closed-form univariate Gram entries are due to Farouki & Rajan (*Computer
    Aided Geometric Design* 5, 1988); see :func:`_bernstein_gram_matrix_1d`.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients
            (any shape).

    Returns:
        float: The squared :math:`L^2` norm ``||p||_2^2``.
    """
    c = coeffs.astype(np.float64, copy=False)

    # Apply G = G^(0) (x) ... (x) G^(d-1) by contracting one axis at a time.
    g_c = c
    for axis in range(c.ndim):
        gram = _bernstein_gram_matrix_1d(c.shape[axis] - 1)
        g_c = np.moveaxis(np.tensordot(gram, g_c, axes=([1], [axis])), 0, axis)

    # Quadratic form c^T (G c); the analytic value is non-negative, so guard
    # against a small negative result from floating-point round-off.
    return abs(float(np.sum(c * g_c)))


def _projected_quadrature_size(degree: int) -> int:
    r"""Get the Gauss-Legendre node count that grades a rational reduction in one direction.

    ``2 * degree + 2``.

    **Derivation.** Writing the current homogeneous curve as :math:`(A, w)` and the
    round-tripped one as :math:`(\tilde A, \tilde w)`, the projected difference is

    .. math::

        \tilde C - C = \frac{\tilde A}{\tilde w} - \frac{A}{w}
                     = \frac{w \tilde A - \tilde w A}{w\, \tilde w} ,

    so the integrand of :math:`\lVert \tilde C - C \rVert_{L^2}^2` is a *rational*
    function whose numerator is a polynomial of degree :math:`4p` per direction and
    whose denominator :math:`(w \tilde w)^2` is a polynomial of the same degree,
    strictly positive on the closed domain whenever the control weights are (Bernstein
    positivity plus partition of unity). No Gauss rule integrates that exactly.

    An ``n``-node Gauss-Legendre rule is exact for polynomials of degree ``2n - 1``, so
    ``n = 2p + 1`` is the smallest rule that integrates the *numerator* exactly, i.e.
    the smallest rule that would be exact if the weights were constant -- which is
    precisely the non-rational case, where the measure reduces to the Bernstein-Gram
    value :func:`_squared_l2_norm` computes. This function adds one node of margin.

    What remains is the departure of :math:`1/(w \tilde w)^2` from a polynomial, which
    is *not* bounded here. Measured instead, against a 2e5-point reference over a
    battery of degrees ``p in {3, 6, 12, 20}``, weight ratios ``w_max / w_min`` up to
    ``1e2``, and coordinate offsets up to ``1e3``: the estimate agreed to five decimal
    digits at the median and to within 3% in the worst case (``p = 3``, weight ratio
    ``1e2``). The relative form helps -- the same denominators appear in the reference
    norm, so part of the quadrature error cancels.

    Args:
        degree (int): Polynomial degree ``p`` in the direction (``>= 0``).

    Returns:
        int: Number of Gauss-Legendre nodes to use in that direction.
    """
    return 2 * degree + 2


@backend_keyed_cache(maxsize=64)
def _bernstein_collocation_1d(degree: int, num_nodes: int) -> npt.NDArray[np.float64]:
    r"""Return the cached, read-only Bernstein values at the Gauss-Legendre nodes.

    Entry ``[k, i]`` is :math:`B_{i,\text{degree}}(t_k)` with ``t_k`` the ``num_nodes``
    Gauss-Legendre nodes on :math:`[0, 1]`. Sampling a control net is then one
    ``tensordot`` per parametric direction, the same contraction pattern
    :func:`_squared_l2_norm` uses with the Gram matrices.

    The values come from the serial Bernstein kernel rather than the parallel one: the
    batches here are a few dozen points, where the fork/join overhead dominates, and
    keeping :func:`_minimize_degree_bezier` clear of ``parallel=True`` kernels means it
    needs no JIT-warmup barrier.

    Args:
        degree (int): Polynomial degree of the basis (``>= 0``).
        num_nodes (int): Number of Gauss-Legendre nodes (``>= 1``).

    Returns:
        npt.NDArray[np.float64]: Read-only ``(num_nodes, degree + 1)`` array.
    """
    nodes = np.ascontiguousarray(get_gauss_legendre_1d(num_nodes, np.float64)[0], np.float64)
    basis = np.empty((num_nodes, degree + 1), dtype=np.float64)
    _tabulate_Bernstein_basis_1D_serial_core(np.int32(degree), nodes, basis)
    basis.flags.writeable = False
    return basis


@backend_keyed_cache(maxsize=64)
def _tensor_gauss_weights(num_nodes: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Return the cached, read-only tensor-product Gauss-Legendre weights on the unit cube.

    Args:
        num_nodes (tuple[int, ...]): Node count per parametric direction.

    Returns:
        npt.NDArray[np.float64]: Read-only array of shape ``num_nodes``.
    """
    weights = np.ones((), dtype=np.float64)
    for n in num_nodes:
        weights = np.multiply.outer(weights, get_gauss_legendre_1d(n, np.float64)[1])
    result = np.ascontiguousarray(weights, dtype=np.float64)
    result.flags.writeable = False
    return result


def _sample_projected(
    ctrl: npt.NDArray[np.floating[Any]], num_nodes: tuple[int, ...]
) -> npt.NDArray[np.float64] | None:
    """Evaluate a homogeneous control net on the Gauss grid and divide out the weights.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Homogeneous control points shaped
            ``(*orders, rank + 1)``, the last column the weights.
        num_nodes (tuple[int, ...]): Node count per parametric direction.

    Returns:
        npt.NDArray[np.float64] | None: Projected points of shape
        ``(*num_nodes, rank)``, or *None* if any sampled weight is not strictly
        positive.

    Note:
        A non-positive sample alongside a positive one means ``w`` changes sign, hence
        vanishes, hence the mapping has a pole on the domain. A uniformly non-positive
        weight field has no pole but is not a NURBS, and is refused on the strictly
        positive weight convention the rest of the library already requires (knot
        removal, ``locate`` and ``find_roots`` all state it). The converse does **not**
        hold: a sign change confined strictly between two nodes is not detected. Ruling
        that out would take the Bernstein convex-hull test on the control weights, which
        is deliberately not done here -- see :func:`_projected_relative_deviation`.
    """
    values = np.asarray(ctrl, dtype=np.float64)
    for axis in range(values.ndim - 1):
        basis = _bernstein_collocation_1d(ctrl.shape[axis] - 1, num_nodes[axis])
        values = np.asarray(
            np.moveaxis(np.tensordot(basis, values, axes=([1], [axis])), 0, axis), np.float64
        )

    weights = values[..., -1:]
    if not bool(np.all(weights > 0.0)):
        return None
    return np.asarray(values[..., :-1] / weights, dtype=np.float64)


def _projected_relative_deviation(
    values: npt.NDArray[np.float64] | None,
    trial_values: npt.NDArray[np.float64] | None,
    quad_weights: npt.NDArray[np.float64],
) -> float:
    r"""Grade a rational trial reduction by its deviation in *projected* space.

    Returns the quadrature estimate of :math:`\lVert \tilde C - C \rVert_{L^2} /
    \lVert C \rVert_{L^2}`, the relative deviation of the round-tripped mapping from
    the current one, both taken after dividing out the weights. This is the quantity a
    caller of :meth:`~pantr.bezier.Bezier.minimize_degree` budgets: grading the
    homogeneous coefficients instead compares a weight against a weighted coordinate,
    which are not the same quantity and, away from unit coordinate scale, not even the
    same order of magnitude.

    Hypotheses: the weights of both nets are strictly positive at the quadrature nodes
    (checked by :func:`_sample_projected`, which returns *None* otherwise), and the
    quadrature resolves the integrand -- see :func:`_projected_quadrature_size` for the
    node count and what it does and does not guarantee.

    The residual risk this leaves is a trial whose weight function dips through zero
    strictly *between* nodes: the deviation is then unbounded near the pole, and a rule
    that steps over it can under-report. Requiring the trial's *control* weights to be
    positive would exclude it outright, by Bernstein positivity, at the cost of also
    refusing trials whose weights are individually negative but whose weight function
    never vanishes. That guard is deliberately not taken: this measure is chosen for
    being tight, and the case needs a weight sign pattern that survives every node while
    failing in between.

    Args:
        values (npt.NDArray[np.float64] | None): Projected points of the current curve.
        trial_values (npt.NDArray[np.float64] | None): Projected points of the
            round-tripped trial curve.
        quad_weights (npt.NDArray[np.float64]): Tensor-product quadrature weights.

    Returns:
        float: The relative projected deviation, or :data:`math.inf` when it cannot be
        graded, so that the caller rejects the reduction.
    """
    if values is None or trial_values is None:
        return math.inf

    deviation2 = float(np.sum(quad_weights * np.sum((trial_values - values) ** 2, axis=-1)))
    reference2 = float(np.sum(quad_weights * np.sum(values**2, axis=-1)))

    deviation = math.sqrt(deviation2)
    if reference2 > 0.0:
        deviation /= math.sqrt(reference2)
    return deviation if math.isfinite(deviation) else math.inf


def _degree_reduction_l2_error(bezier: Bezier, decrements: tuple[int, ...]) -> float:
    r"""Compute the exact :math:`L^2` error of a degree reduction.

    Reduces, elevates the result back to the original degrees (an exact
    operation) and takes the :math:`L^2` norm of the coefficient difference
    through the Bernstein Gram matrix, so the value is the true
    :math:`\lVert f - g \rVert_{L^2([0,1]^d)}` rather than an estimate.  Rank
    components are combined in the Euclidean sense.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier that would be reduced.
        decrements (tuple[int, ...]): Degree decrement per direction.

    Returns:
        float: The :math:`L^2` norm of the error the reduction would introduce.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    reduced = _degree_reduce_bezier(bezier, decrements)
    restored = _degree_elevate_bezier(reduced, decrements)

    diff = restored.control_points - bezier.control_points
    rank = diff.shape[-1]
    return math.sqrt(sum(_squared_l2_norm(diff[..., r]) for r in range(rank)))


def _minimize_degree_bezier(
    bezier: Bezier,
    tol: float | None = None,
) -> Bezier:
    """Automatically reduce the degree of a Bézier while maintaining accuracy.

    Greedy, direction-by-direction degree reduction.  For each parametric
    direction the degree is lowered by one as long as the candidate reduction is
    accurate enough: the trial curve is degree-reduced (least squares) and then
    re-elevated back to the current degree, and the round-trip relative
    :math:`L^2` error is compared against ``tol``.  The first rejected trial in a
    direction stops further reduction in that direction (the error is
    monotonically non-decreasing as the degree drops, so there is nothing to gain
    by continuing).  For vector-valued Bézier all rank components are combined
    into a single error measure, so a reduction is accepted only when every
    component is preserved.

    **Rational input is graded in projected space.**  A rational control net holds
    homogeneous coordinates ``[w x, w y, ..., w]``, and an :math:`L^2` norm taken over
    all of its columns adds a weight to a weighted coordinate.  Those are different
    quantities, and at coordinate scale ``s`` they differ in magnitude by ``s``: a
    weight-carried round-trip deviation is measured as ``O(dw / s)`` relative while the
    projected deviation it actually causes is ``O(dw)`` relative, so the homogeneous
    measure under-reports by a factor of order ``s``.  Measured on the degree-6 rational
    curve of issue #297, one weight nudged by ``1e-6``, grading its first trial
    reduction:

    ===========  ====================  ====================  =============
    scale        homogeneous measure   projected deviation   under-report
    ===========  ====================  ====================  =============
    ``1``        ``6.01e-9``           ``7.12e-9``           ``1.2x``
    ``1e3``      ``1.10e-11``          ``7.12e-9``           ``646x``
    ``1e6``      ``1.10e-14``          ``7.12e-9``           ``6.5e5x``
    ===========  ====================  ====================  =============

    The projected column is constant, as it must be: scaling every coordinate is a
    similarity and cannot change a *relative* deviation.  The under-report grows by the
    full ``1e3`` per decade only once ``s`` dominates, since at ``s ~ 1`` the coordinate
    and weight columns are still comparable -- the same crossover
    :func:`~pantr.bspline._bspline_knot_removal._homogeneous_deviation_tolerance`
    describes for the max-norm case.

    So for a rational Bézier the trial is graded by
    :func:`_projected_relative_deviation`: both the current and the round-tripped nets
    are evaluated on a tensor Gauss-Legendre grid, the weights are divided out, and the
    relative :math:`L^2` deviation of the *mappings* is compared against ``tol``.  Under
    a uniform scaling both the deviation and the reference norm scale by the same
    factor, so the verdict is scale invariant -- which is exactly the property the
    homogeneous measure lacks.  (It is *not* translation invariant, because the
    reference norm is taken about the origin; neither is the non-rational branch, and
    that is what "relative" has always meant here.)  Two consequences to be aware of:

    * unlike the non-rational branch, the measure is a **quadrature estimate** rather
      than an exact value -- the integrand is rational, so no Gauss rule is exact on it.
      :func:`_projected_quadrature_size` gives the node count, the argument for it, and
      the measured agreement with a dense reference;
    * a reduction is **refused** whenever a weight is not strictly positive at a
      quadrature node, on either net, since the mapping then has a pole on the domain
      and no projected deviation is defined.  A net that violates this is returned
      unreduced rather than raising.

    The non-rational branch is untouched and still uses the exact Bernstein-Gram value.

    The alternative -- an :math:`L^2` analogue of Piegl & Tiller Eq. (5.30), bounding
    the projected deviation by the homogeneous one -- was prototyped and rejected.  The
    exact identity is ``dC = (dA - C dw) / (w + dw)``, whose numerator is invariant
    under translating the model; the triangle inequality ``|dA - C dw| <= |dA| +
    |C| |dw|`` is not, and it discards precisely the cancellation that keeps the true
    deviation small.  So on geometry that is not centred on the origin the bound is
    conservative by three to four orders of magnitude: measured median ``790x`` to
    ``1235x`` at a weight ratio of only 2 and a coordinate offset of ``1e3``.
    Recentring the bound at the control-net centroid restores that invariance but still
    leaves ``12x`` to ``85x`` at a weight ratio of ``1e2``, which would silently make
    ``minimize_degree`` refuse reductions the caller asked for.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to simplify.
        tol (float | None): Relative tolerance for accepting a degree
            reduction.  If *None*, uses
            :data:`_AUTO_REDUCTION_TOL_FACTOR` ``* eps``.

    Returns:
        ~pantr.bezier.Bezier: A new Bézier with the lowest degree that
        preserves accuracy within ``tol``.  If no reduction is possible,
        returns a copy of the input.

    Raises:
        ValueError: If a direction's degree exceeds the exactness envelope of the
            binomial-coefficient kernel (see
            :data:`~pantr.bspline._bspline_degree_core._BINCOEFF_MAX_N`).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    # Each trial re-elevates from ``degree - 1`` back to ``degree``, starting at the
    # direction's own degree, so that degree bounds the coefficients needed.  Corrupted
    # coefficients would not merely lose accuracy here: they feed the round-trip error
    # measure, and so the accept/reject verdict itself.
    for d, p in enumerate(bezier.degree):
        if p >= 1:
            _check_bincoeff_envelope(p, f"Degree minimization of a degree-{p} direction {d}")

    ctrl: npt.NDArray[np.floating[Any]] = bezier.control_points  # (*orders, rank)
    rank = ctrl.shape[-1]

    if tol is None:
        tol = _AUTO_REDUCTION_TOL_FACTOR * float(np.finfo(ctrl.dtype).eps)

    if tol <= 0.0:
        return BezierCls(ctrl.copy(), is_rational=bezier.is_rational)

    def total_squared_norm(arr: npt.NDArray[np.floating[Any]]) -> float:
        """Sum the squared L2 norms of every rank component of *arr*."""
        return float(sum(_squared_l2_norm(arr[..., r]) for r in range(rank)))

    result = ctrl
    changed = False

    # A rational net is graded in projected space, so the trials are sampled on one
    # tensor Gauss grid sized from the *input* degrees.  Degrees only ever fall, so a
    # grid fixed up front is never coarser than `_projected_quadrature_size` asks for at
    # the current degree, and it keeps the collocation cache warm across every trial.
    num_nodes = tuple(_projected_quadrature_size(p) for p in bezier.degree)
    quad_weights = _tensor_gauss_weights(num_nodes) if bezier.is_rational else None
    projected = _sample_projected(result, num_nodes) if bezier.is_rational else None

    # One backend selection for the whole search, and this is the call site that
    # makes the catalogue hand out a record rather than two callables: each trial
    # reduces and then re-elevates, and a round trip whose two halves came from
    # different implementations would not be a round trip.
    elevate, reduce_apply = degree_kernels()

    for dim in range(bezier.dim):
        # Each direction shrinks until a reduction is rejected.
        while result.shape[dim] >= 2:  # noqa: PLR2004
            degree = result.shape[dim] - 1

            # Reduce by one along `dim` (all rank components together) ...
            reduced = _reduce_along_axis(
                result, dim, _interpolating_reduction_operator(degree, 1), reduce_apply
            )

            # ... then re-elevate so the trial can be compared to `result`.
            flat_reduced, trailing_reduced = _flatten_along_axis(reduced, dim)
            flat_elevated = np.empty(
                (flat_reduced.shape[0] + 1, flat_reduced.shape[1]), dtype=flat_reduced.dtype
            )
            elevate(degree - 1, flat_reduced, 1, flat_elevated)
            elevated = _unflatten_along_axis(flat_elevated, trailing_reduced, dim)

            if quad_weights is not None:
                # Relative round-trip deviation of the projected mapping.
                rel_error = _projected_relative_deviation(
                    projected, _sample_projected(elevated, num_nodes), quad_weights
                )
            else:
                # Relative round-trip L2 error across all components.
                diff_norm2 = total_squared_norm(elevated - result)
                orig_norm2 = total_squared_norm(result)
                rel_error = math.sqrt(diff_norm2)
                if orig_norm2 > 0.0:
                    rel_error /= math.sqrt(orig_norm2)

            if rel_error >= tol:
                break
            result = reduced
            changed = True
            if quad_weights is not None:
                projected = _sample_projected(result, num_nodes)

    if not changed:
        return BezierCls(ctrl.copy(), is_rational=bezier.is_rational)
    return BezierCls(result, is_rational=bezier.is_rational)
