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
from ._quad_backend import quad_kernels

# Re-exported so the name keeps resolving where it always did. It predates the
# split of the kernels into ``_rules_core`` for the C++ port, and CLAUDE.md
# records that a downstream consumer imports pantr's private symbols, where this
# repository's CI cannot see the breakage. The three constants and helpers that
# were introduced together WITH the port are not re-exported: nothing outside
# this branch has ever been able to import them.
# The redundant alias is the PEP 484 explicit-re-export form, which mypy's
# no_implicit_reexport requires and ruff's PLC0414 objects to; ``__init__.py``
# is exempt from that rule by configuration and this module is not.
from ._rules_core import _TANH_SINH_DECAY_FACTOR as _TANH_SINH_DECAY_FACTOR  # noqa: PLC0414


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

    nodes, weights = quad_kernels().trapezoidal(n_pts)
    return nodes.astype(dtype), weights.astype(dtype)


def _gauss_legendre_symmetric(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the ``n``-point Gauss-Legendre rule on ``[-1, 1]`` in float64.

    Dispatches to the active backend.  The computation itself, and the reasoning
    behind the method, live in
    :func:`pantr.quad._rules_core._gauss_legendre_symmetric_core`; kept here under
    its original name because it is part of this module's private surface and
    CLAUDE.md records that a downstream consumer imports pantr's private symbols.

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
    return quad_kernels().gauss_legendre(n)


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

    return quad_kernels().chebyshev_nodes(n_pts, dtype)


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


def _lambert_w_principal(x: float) -> float:
    """Solve ``w e^w = x`` on the principal branch.

    Dispatches to the active backend; the derivation and the method are in
    :func:`pantr.quad._rules_core._lambert_w_principal_core`.  Replaces
    ``scipy.special.lambertw``, which was this module's only use of scipy.

    Args:
        x (float): The argument. Must be at least about 1.61; the precondition
            and what happens below it are stated on the core function.

    Returns:
        float: ``W(x)``, within about one unit of roundoff of ``W``.
    """
    return quad_kernels().lambert_w(x)


def _generate_tanh_sinh(
    n: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float64], int]:
    """Generate tanh-sinh quadrature nodes and weights on [-1, 1].

    Dispatches to the active backend.  The rule itself, and the reasoning behind
    the transform, the step size and the truncation, are in
    :func:`pantr.quad._rules_core._generate_tanh_sinh_core`.

    This function's signature and return shape are **frozen**: an external
    project consumes the rule and pins golden values off it, so the pair
    ``(data, m)`` with *data* of shape ``(m, 2)`` is a contract rather than a
    convenience.  ``tests/test_quad_tanh_sinh.py`` pins it.

    The truncation threshold is derived here rather than inside the kernel, by
    :func:`_tanh_sinh_min_gap`, and that placement is deliberate: shared by both
    backends, the threshold is common mode, so the discrete decision of how many
    nodes the rule has cannot differ between them for a reason that is not the
    arithmetic.

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
        Mori (1974), *Publ. RIMS, Kyoto Univ.* 9(3), 721-741.
    """
    return quad_kernels().tanh_sinh(n, _tanh_sinh_min_gap(dtype))


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
