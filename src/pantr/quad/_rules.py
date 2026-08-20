"""1D quadrature rules and shared node/weight helpers.

Trapezoidal (equispaced), Gauss-Legendre, Gauss-Lobatto-Legendre,
Chebyshev-Gauss (1st and 2nd kind), modified Chebyshev interpolation nodes,
and tanh-sinh (double-exponential) quadrature, all on ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numpy.polynomial import chebyshev, legendre

from .._array_utils import _validate_float_dtype
from ..tolerance import get_machine_epsilon


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


_GAUSS_LEGENDRE_NEWTON_STEPS: int = 4
"""Newton steps used to place the Gauss-Legendre nodes.

Fixed rather than iterated to a residual test, for two reasons. A test on the
step is a comparison on the scalar, which the port's AD tiering keeps out of a
kernel, and it does not terminate reliably anyway: the step oscillates in the
last bit rather than reaching zero, so a loop waiting for zero runs to whatever
cap it was given.

Four rather than three, and the count is derived. At a root of ``P_n`` the
Legendre equation ``(1 - x^2) P'' - 2 x P' + n(n+1) P = 0`` gives
``P'' = 2 x P' / (1 - x^2)``, so Newton's error constant there is

    C = |P'' / (2 P')| = |x| / (1 - x^2)

which at the outermost node grows like ``n^2 / 5.78``. The asymptotic starting
guess below is accurate to ``O(1/n^2)`` at that same node, so the **product**
``C * e_0`` is independent of ``n``: measured 0.0182 at ``n = 6`` and 0.0201 at
``n = 1000``. Writing ``eps_k = C * e_k`` the Newton recurrence is
``eps_{k+1} = eps_k^2`` from ``eps_0 = 0.02``, giving 4e-4, 1.6e-7, 2.6e-14,
6.6e-28. Convergence needs ``eps_k < C * u``; at ``n = 6``, where ``C = 7.145``,
that threshold is 7.9e-16, which the third step misses and the fourth clears by
twelve orders.

The prediction is checkable and was checked: the chain puts the third step's
error at 1.68e-15, which is 7.6 units of roundoff, and three steps measure 7.5.
Four steps measure 0.5 units of roundoff against `numpy.polynomial.legendre
.leggauss` over every ``n`` from 1 to 1000 tried, and five and six measure the
same, so the fourth reaches the fixed point.
"""


def _legendre_and_derivative(
    n: int, x: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate ``P_n`` and ``P_n'`` by the three-term recurrence.

    The recurrence and the order its operations are written in are part of this
    function's contract, not an implementation detail: the C++ backend is an
    operation-for-operation transliteration of it, and the parity bound on the
    nodes is derived on the assumption that the two sides evaluate the same
    expression. Changing the association here changes the last bits of every
    node.

    The derivative uses ``P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1)``, which is
    singular at the endpoints and is never evaluated there: every Gauss node is
    interior, and the outermost sits about ``1 - c/n^2`` away.

    Args:
        n (int): Degree. Must be at least 1.
        x (npt.NDArray[np.float64]): Points, strictly inside ``(-1, 1)``.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ``P_n(x)`` and
            ``P_n'(x)``.
    """
    previous = np.ones_like(x)
    current = x.copy()
    for k in range(2, n + 1):
        previous, current = (
            current,
            ((2 * k - 1) * x * current - (k - 1) * previous) / k,
        )
    derivative = n * (x * current - previous) / (x * x - 1.0)
    return current, derivative


def _gauss_legendre_symmetric(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the ``n``-point Gauss-Legendre rule on ``[-1, 1]`` in float64.

    Newton on ``P_n`` rather than the eigenvalues of the Jacobi matrix. The two
    are not equally good here and the choice is recorded in
    ``design/quadrature_algorithms.md``: Newton converges to the root while an
    eigensolve inherits its sweep's accumulated backward error, so the node error
    is flat in ``n`` at well under one unit of roundoff instead of growing.

    Only the ``ceil(n/2)`` non-negative roots are computed; the rest follow from
    ``P_n(-x) = (-1)^n P_n(x)``, and writing each negative node as the exact
    negation of its partner keeps the rule symmetric to the last bit rather than
    to a tolerance.

    The result is on ``[-1, 1]`` and in float64 whatever the caller asked for.
    Mapping onto ``[0, 1]`` and narrowing are
    :func:`_scale_and_cast_nodes_and_weights`'s job, and keeping them there is
    deliberate: shared between the two backends, the map is common mode and
    cancels exactly in a parity comparison instead of contributing its own
    endpoint conditioning to the bound.

    Args:
        n (int): Number of points. Must be at least 1.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Nodes ascending
            in ``(-1, 1)`` and their weights, summing to 2 up to the rounding of
            that sum.
    """
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    half = (n + 1) // 2
    index = np.arange(1, half + 1, dtype=np.float64)
    # Tricomi's asymptotic approximation to the roots, counted inward from +1.
    # Its error is O(1/n^2) at the outermost root, which is what makes the step
    # count above independent of n.
    nodes = np.cos(np.pi * (index - 0.25) / (n + 0.5))

    for _ in range(_GAUSS_LEGENDRE_NEWTON_STEPS):
        value, derivative = _legendre_and_derivative(n, nodes)
        nodes = nodes - value / derivative

    _, derivative = _legendre_and_derivative(n, nodes)
    weights = 2.0 / ((1.0 - nodes * nodes) * derivative * derivative)

    # `nodes` runs from the largest root down to the smallest, so the ascending
    # rule is the reversed negatives, then the centre for odd n, then the tail.
    if n % 2:
        all_nodes = np.concatenate(
            [-nodes[: half - 1], nodes[half - 1 : half], nodes[half - 2 :: -1]]
        )
        all_weights = np.concatenate(
            [weights[: half - 1], weights[half - 1 : half], weights[half - 2 :: -1]]
        )
    else:
        all_nodes = np.concatenate([-nodes, nodes[::-1]])
        all_weights = np.concatenate([weights, weights[::-1]])
    return all_nodes, all_weights


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

    nodes, weights = _gauss_legendre_symmetric(n_pts)

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


def _tanh_sinh_min_gap(dtype: npt.DTypeLike) -> float:
    """Get the smallest endpoint gap ``1 - |x|`` a tanh-sinh node may carry on [-1, 1].

    A double-exponential rule is what it is because no node ever reaches an
    endpoint; that is what makes an endpoint singularity integrable, and it is
    the property :func:`get_tanh_sinh_1d` advertises.  The gap falls
    double-exponentially along the transform axis, so the rule has to stop
    somewhere, and the place to stop is where the gap stops being representable
    *in the frame the rule is returned in*.

    :func:`get_tanh_sinh_1d` maps ``[-1, 1]`` onto ``[0, 1]`` by ``(x + 1) / 2``
    and casts to *dtype*, so the node near the right end is reached as
    ``((1 - gap) + 1) / 2``, and it is that whole expression, not any one step of
    it, that has to land strictly below ``1``.  Two of the three steps round.
    Writing ``u = eps / 2`` for the spacing just below ``1``:

    * ``1 - gap`` rounds to ``1 - u`` while ``gap < 1.5 u``, and to ``1 - 2u``
      from there on;
    * ``+ 1`` maps ``1 - u`` to ``2 - u``, the midpoint of ``[2 - 2u, 2]``, which
      ties to even and so becomes ``2``; it maps ``1 - 2u`` to ``2 - 2u`` exactly;
    * the halving is exact.

    So the node collapses onto ``1`` exactly when ``gap < 1.5 u = 0.75 eps``, and
    returning ``eps`` clears that by one representable step of ``gap``, a factor
    ``4 / 3``.  Bisecting on ``gap`` puts the crossing at ``0.75 eps`` to fifty
    digits in ``float64`` and in ``float32`` alike, the argument being about the
    binade boundary at ``1`` and not about the width of the format.

    Note that reasoning about a single step gives the wrong answer here: ``2 -
    gap`` alone survives down to ``gap > eps / 2``, a factor 1.5 below the true
    threshold, because it ignores the rounding of ``1 - gap`` that precedes it.

    Truncating there costs nothing.  With ``gap = 1 - tanh(omega)`` one has
    ``cosh(omega)**-2 = gap * (2 - gap)``, so the discarded weight is
    ``w = (pi / 2) * cosh(t) * gap * (2 - gap) <= pi * cosh(t) * gap``.  In
    float64 the threshold puts ``omega`` at ``18.4`` and ``cosh(t)`` at ``11.8``,
    giving ``w <= 8.3e-15`` against a weight sum of ``2`` (largest measured over
    ``n`` from 2 to 400: ``8.19e-15``).  That is below the rounding of the sum
    the weight would have joined, and the rescaling in
    :func:`_generate_tanh_sinh` restores that sum.

    What this bound does *not* claim is that the returned nodes are all distinct.
    Two consecutive samples can both clear the threshold and still round onto one
    representable coordinate, first at ``n_pts = 544`` in ``float64`` and
    ``n_pts = 324`` in ``float32``, both at ``1 - eps``.  That is the format
    running out of coordinates rather than a node reaching the endpoint, and the
    two weights, ``2.9e-16`` together at ``n_pts = 544``, land where they belong.

    Args:
        dtype (npt.DTypeLike): Floating-point dtype the rule will be returned in.

    Returns:
        float: The smallest admissible gap, one machine epsilon of *dtype*.
    """
    return get_machine_epsilon(dtype)


def _generate_tanh_sinh(
    n: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float64], int]:
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

    Generation stops at the last node whose distance to the endpoint is still
    representable once the rule is mapped onto ``[0, 1]`` in *dtype* (see
    :func:`_tanh_sinh_min_gap`), so the effective number of nodes *m* may be less
    than *n* and no node ever sits on an endpoint.  The weights are finally
    rescaled onto the measure ``2`` of ``[-1, 1]``, up to the rounding of their
    own sum.

    Args:
        n (int): Requested number of quadrature points (must be >= 1).
        dtype (npt.DTypeLike): Floating-point dtype the rule will be returned in.
            It sets the truncation point and nothing else; the returned data is
            float64 either way.  Defaults to ``np.float64``.

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

    min_gap = _tanh_sinh_min_gap(dtype)
    buf = np.empty((n, 2), dtype=np.float64)  # worst case: n nodes
    count = 0

    odd = bool(n % 2)
    if odd:
        # Central node at t = 0: x = 0, w = (pi / 2) * cosh(0) / cosh(0)^2.
        buf[count] = [0.0, half_pi]
        count += 1

    for i in range(n // 2):
        # Odd n samples t = h, 2h, ...; even n offsets by half a step.
        t = (i + 1) * h if odd else (i + 0.5) * h
        omega = half_pi * np.sinh(t)
        w = half_pi * np.cosh(t) / np.cosh(omega) ** 2
        # gap = 1 - tanh(omega), the node's distance from the +1 endpoint,
        # via the algebraically equal form 2 / (1 + e^{2 omega}).  This keeps
        # gap a small but nonzero float right up to the resolution limit,
        # whereas 1 - np.tanh(omega) saturates to 0 a step too early and would
        # truncate the rule prematurely.
        gap = 2.0 / (1.0 + np.exp(2.0 * omega))

        # gap decreases monotonically in t, so the first node whose distance to
        # the endpoint no longer survives the mapping onto [0, 1] ends the rule.
        if gap < min_gap:
            break

        # Symmetric pair at -(1 - gap) and +(1 - gap); writing both from
        # gap keeps the coordinates exact negatives of each other.
        buf[count] = [-(1.0 - gap), w]
        count += 1
        buf[count] = [1.0 - gap, w]
        count += 1

    data = buf[:count].copy()

    # Rescale the weights onto the measure 2 of [-1, 1].  The sum of the rescaled
    # weights is 2 only up to the rounding of that sum: dividing by a computed sum
    # cannot make a floating-point sum exact.
    data[:, 1] *= 2.0 / np.sum(data[:, 1])

    return data, count


def get_tanh_sinh_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get tanh-sinh quadrature nodes on [0, 1] for the given number of points.

    Tanh-sinh (double-exponential) quadrature clusters nodes near the
    endpoints of the interval, making it well suited for integrands with
    endpoint singularities or steep boundary layers.  The scheme is
    symmetric, and every returned node lies strictly inside ``(0, 1)``: a node
    that would sit closer to an endpoint than *dtype* can resolve ends the rule
    instead of being moved onto the boundary, so the number of returned nodes
    may be less than *n_pts*, and fewer in ``float32`` than in ``float64``.
    Placing a node *on* the boundary would defeat the purpose of the rule, an
    endpoint singularity being exactly what it is meant to integrate.

    Args:
        n_pts (int): Requested number of quadrature points.  Must be at
            least 1.
        dtype (npt.DTypeLike): Floating-point dtype for the output arrays.
            Must be ``float32`` or ``float64``.  Defaults to ``float64``.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            ``(nodes, weights)`` on ``[0, 1]``.  Both arrays have the same
            length, which may be less than *n_pts* because the rule is
            truncated where the endpoint gap stops being resolvable.  Weights
            sum to 1 up to the rounding of that sum.

    Raises:
        ValueError: If *n_pts* < 1 or *dtype* is not ``float32``/``float64``.

    Note:
        Truncation puts a floor on what a singular integrand can reach, and
        raising *n_pts* past it buys nothing.  The rule covers
        ``[delta, 1 - delta]`` with ``delta`` of order ``eps / 2``, so the error
        is the integral over the two neglected tails.  For ``x**-0.5`` that is
        ``2 * sqrt(delta) = 2e-8`` (measured 2.0e-8 from *n_pts* 49 on) and for
        ``log(x)`` it is ``delta * (log(delta) - 1) = 4e-15`` (measured 3.9e-15).
        A smooth integrand is unaffected and converges to machine precision.

    Example:
        >>> nodes, weights = get_tanh_sinh_1d(5)
        >>> nodes.shape[0] <= 5
        True
        >>> bool(((nodes > 0.0) & (nodes < 1.0)).all())
        True
        >>> bool(abs(weights.sum() - 1.0) < 1e-14)
        True
    """
    _validate_n_pts_and_dtype(n_pts, dtype)

    data, _ = _generate_tanh_sinh(n_pts, dtype)
    return _scale_and_cast_nodes_and_weights(data[:, 0], data[:, 1], dtype)
