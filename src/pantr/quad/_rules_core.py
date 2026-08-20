"""Pure computation of the quadrature rules: no dispatch, no validation.

**Layer 3** in the layering of CLAUDE.md. Every function here assumes its inputs
are correct and checks nothing; the guarantees come from
:mod:`pantr.quad._rules`, which is Layer 2 and validates, and from the C++
bindings, which are Layer 2's other half.

The split from :mod:`pantr.quad._rules` exists to break an import cycle rather
than for tidiness. :mod:`pantr.quad._quad_backend` is the catalogue and has to
import the kernels it hands out, while the public entry points have to import the
catalogue to ask which kernel to call. Keeping the kernels in a third module that
imports neither is what makes the dependencies one-directional, and it is the
same shape :mod:`pantr.basis` already uses -- ``_basis_core`` under
``_basis_backend`` under ``_basis_1D``.

**These are the parity oracles.** Each has an operation-for-operation C++
transliteration under ``cpp/include/pantr/quad/``, and the association order of
the expressions below is part of the contract rather than an implementation
detail: re-associating one line changes the last bits of every node and turns a
bitwise parity claim into a fitted one. ``tests/parity/`` measures which of the
five agree bit for bit and which only within a bound.

- :func:`_legendre_and_derivative`, :func:`_gauss_legendre_symmetric_core`
- :func:`_lambert_w_principal_core`, :func:`_generate_tanh_sinh_core`
- :func:`_trapezoidal_core`, :func:`_modified_chebyshev_nodes_core`
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

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


def _gauss_legendre_symmetric_core(
    n: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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


_TANH_SINH_DECAY_FACTOR: float = 0.6
"""Decay-rate factor selecting the uniform step ``h`` in transform space."""


_LAMBERT_W_HALLEY_STEPS: int = 4
"""Halley steps used to solve ``w e^w = x`` for the tanh-sinh step size.

Fixed rather than iterated to a residual test, and four rather than three. Both
choices are derived, and the derivation is in
``design/quadrature_algorithms.md`` because it turns on a constant defined in
this module. The short form:

* A residual test is a comparison on the scalar, which the port's AD tiering
  keeps out of a kernel.
* The convergence chain in the **absolute** error, with a Lagrange constant
  ``M <= 0.6886`` over the argument range, reads 0.455, 6.5e-2, 1.1e-4, 4.9e-13,
  4.5e-38. Three steps miss the unit roundoff and four clear it.
* The argument that actually decides it is the validity threshold. Bisecting on
  :data:`_TANH_SINH_DECAY_FACTOR` at ``n = 2``, which is the binding case, three
  steps reach one unit of roundoff down to 0.5932 and four down to 0.5097. The
  shipped 0.6 therefore sits 1.1% away from failing with three steps and 18%
  away with four.

The extra step costs one exponential and one logarithm **per rule**, not per
node.
"""


def _lambert_w_principal_core(x: float) -> float:
    """Solve ``w e^w = x`` on the principal branch by Halley's method.

    Replaces ``scipy.special.lambertw``, which was this module's only use of
    scipy.

    **Precondition, and it is deliberately not checked here.** ``x`` must be
    large enough that the starting guess stays on the principal branch. The guess
    is the large-argument asymptotic form and is branch free, so for a small
    argument it lands negative and no number of Halley steps recovers: measured,
    an argument corresponding to a decay factor of 0.50 leaves 2.8e4 units of
    roundoff after four steps, 0.40 leaves 2.4e16, and below about 0.31 the inner
    logarithm is not real. The only caller is :func:`_generate_tanh_sinh_core`, whose
    argument is ``2 * _TANH_SINH_DECAY_FACTOR * (pi/2) * (n - 1)``, smallest at
    ``n = 2`` where it is 1.885. ``tests/test_quad_tanh_sinh.py`` pins that
    coupling, because neither constant records it.

    Args:
        x (float): The argument. Must be at least about 1.61.

    Returns:
        float: ``W(x)``, within about one unit of roundoff of ``W``.
    """
    log_x = np.log(x)
    log_log_x = np.log(log_x)
    w = log_x - log_log_x + log_log_x / log_x

    for _ in range(_LAMBERT_W_HALLEY_STEPS):
        exp_w = np.exp(w)
        residual = w * exp_w - x
        derivative = exp_w * (w + 1.0)
        w = w - residual / (derivative - (w + 2.0) * residual / (2.0 * w + 2.0))

    return float(w)


def _generate_tanh_sinh_core(n: int, min_gap: float) -> tuple[npt.NDArray[np.float64], int]:
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
    :func:`pantr.quad._rules._tanh_sinh_min_gap`), so the effective number of nodes *m* may be less
    than *n* and no node ever sits on an endpoint.  The weights are finally
    rescaled onto the measure ``2`` of ``[-1, 1]``, up to the rounding of their
    own sum.

    Args:
        n (int): Requested number of quadrature points (must be >= 1).
        min_gap (float): Smallest endpoint distance ``1 - |x|`` a node may
            carry, in the frame the rule will be returned in.  It sets the
            truncation point and nothing else; the returned data is float64
            whatever storage format it was derived from.  Layer 2 derives it,
            in :func:`pantr.quad._rules._tanh_sinh_min_gap`, and that helper is
            deliberately shared by both backends so the truncation decision is
            common mode.

    Returns:
        tuple[npt.NDArray[np.float64], int]: A pair ``(data, m)`` where
        *data* has shape ``(m, 2)`` with columns ``[node, weight]`` on
        ``[-1, 1]``, and *m* is the effective node count.

    Note:
        Nodes and weights follow the double-exponential formulas of Takahasi &
        Mori (1974), *Publ. RIMS, Kyoto Univ.* 9(3), 721-741; the step-size root
        is evaluated by :func:`_lambert_w_principal_core`.
    """
    if n == 1:
        return np.array([[0.0, 2.0]]), 1

    half_pi = 0.5 * np.pi
    # Uniform step in transform space; the argument of W follows from the
    # large-argument truncation balance described in _TANH_SINH_DECAY_FACTOR.
    decay_arg = 2.0 * _TANH_SINH_DECAY_FACTOR * half_pi * (n - 1)
    h = 2.0 * _lambert_w_principal_core(decay_arg) / n

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


def _trapezoidal_core(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the ``n``-point trapezoidal rule on ``[0, 1]`` in float64.

    Unlike the other rules here the output is already on ``[0, 1]``, so no
    mapping follows and Layer 2 only narrows.  Narrowing at the end rather than
    computing in the storage format is what :func:`numpy.linspace` itself does --
    it forms ``arange(n) * step + start`` in float64 and casts once -- so
    extracting it here changes nothing, which was checked: the float32 result is
    bit-identical either way.

    For ``n == 1`` the rule is the midpoint one, node ``0.5`` and weight ``1.0``.

    Args:
        n (int): Number of points. Must be at least 1.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Nodes ascending
            from 0 to 1, and their weights.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.quad.get_trapezoidal_1d` instead.
    """
    if n == 1:
        return np.array([0.5]), np.array([1.0])

    nodes = np.linspace(0.0, 1.0, n, dtype=np.float64)

    h = 1.0 / float(n - 1)
    weights = np.full(n, h, dtype=np.float64)
    weights[0] = weights[-1] = 0.5 * h

    return nodes, weights


def _modified_chebyshev_nodes_core(
    n: int, dtype: npt.DTypeLike
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the ``n`` modified Chebyshev interpolation nodes on ``[0, 1]``.

    Chebyshev nodes of the second kind, the Chebyshev-Lobatto points, mapped onto
    ``[0, 1]``.  Both endpoints are included.  These are interpolation nodes, so
    there are no weights.

    **The arithmetic happens in *dtype*, not in float64**, and that is
    load-bearing rather than incidental.  The index array is built in *dtype*, so
    under NEP 50 every operation after it -- the multiply by ``pi``, the divide,
    the cosine and both halvings -- stays in the storage format.  Computing in
    float64 and narrowing at the end is a different answer: the two were measured
    to differ on 2523 of 4096 float32 arguments, 62%.  (An earlier note quoted 712 of
    4096; that is a different measurement, the cosine alone handed an argument both
    sides already agree on.)  The C++ transliteration is
    templated for exactly this reason and is the only quad kernel that is.

    Args:
        n (int): Number of nodes. Must be at least 2.
        dtype (npt.DTypeLike): Floating dtype; float32 or float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape ``(n,)`` with nodes
            in ``[0, 1]``, starting at 0 and ending at 1.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.quad.get_modified_chebyshev_nodes_1d`
        instead.
    """
    dtype_obj = np.dtype(dtype)
    i = np.arange(n, dtype=dtype_obj)
    nodes: npt.NDArray[np.float32 | np.float64] = 0.5 - 0.5 * np.cos(np.pi * i / (n - 1))
    return nodes
