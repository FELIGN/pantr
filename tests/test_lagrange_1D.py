"""Tests for Lagrange basis evaluations across all variants."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from numpy.polynomial import chebyshev, legendre

from pantr.basis import LagrangeVariant, tabulate_lagrange_1d
from pantr.basis._basis_lagrange import _get_lagrange_points
from pantr.tolerance import get_default


def _lagrange_nodes(
    variant: LagrangeVariant, n_pts: int, dtype: npt.DTypeLike
) -> npt.NDArray[np.floating[Any]]:
    """Recreate the interpolation nodes used by the implementation on [0, 1]."""
    dt = np.dtype(dtype)
    target_dtype = np.float32 if dt == np.dtype(np.float32) else np.float64

    if variant == LagrangeVariant.EQUISPACES:
        return np.linspace(0.0, 1.0, n_pts, dtype=target_dtype)

    if variant == LagrangeVariant.GAUSS_LEGENDRE:
        coefs64 = np.zeros(n_pts + 1, dtype=np.float64)
        coefs64[-1] = 1.0
        legroots_t = cast(
            "Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]",
            legendre.legroots,
        )
        nodes = legroots_t(coefs64)
    elif variant == LagrangeVariant.GAUSS_LOBATTO_LEGENDRE:
        if n_pts == 2:
            nodes = np.array([-1.0, 1.0], dtype=target_dtype)
        else:
            basis_t = cast("Callable[[int], Any]", legendre.Legendre.basis)
            P_basis = basis_t(n_pts - 1)
            P_prime = P_basis.deriv()
            interior = cast(npt.NDArray[np.float64], P_prime.roots())
            nodes = np.concatenate((np.array([-1.0]), interior, np.array([1.0]))).astype(
                target_dtype, copy=False
            )
    elif variant == LagrangeVariant.CHEBYSHEV_1ST:
        cheb1_t = cast("Callable[[int], npt.NDArray[np.float64]]", chebyshev.chebpts1)
        nodes = cheb1_t(n_pts)
        nodes = nodes.astype(target_dtype, copy=False)
    else:
        cheb2_t = cast("Callable[[int], npt.NDArray[np.float64]]", chebyshev.chebpts2)
        nodes = cheb2_t(n_pts)
        nodes = nodes.astype(target_dtype, copy=False)

    return ((nodes + 1.0) * 0.5).astype(target_dtype, copy=False)


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_degree_zero_all_variants(variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
    """Degree-0 Lagrange basis is constant 1 for all t in [0, 1]."""
    pts = np.linspace(0.0, 1.0, 5, dtype=dtype)
    res = tabulate_lagrange_1d(0, variant, pts)
    assert res.dtype == np.dtype(dtype)
    nptest.assert_allclose(res, np.ones((pts.shape[0], 1), dtype=dtype))


@pytest.mark.parametrize(
    ("variant", "degree"),
    [
        (LagrangeVariant.EQUISPACES, 1),
        (LagrangeVariant.GAUSS_LEGENDRE, 3),
        (LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, 1),  # hits special n_pts == 2 branch
        (LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, 4),
        (LagrangeVariant.CHEBYSHEV_1ST, 3),
        (LagrangeVariant.CHEBYSHEV_2ND, 5),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kronecker_delta_at_nodes(
    variant: LagrangeVariant, degree: int, dtype: npt.DTypeLike
) -> None:
    """Evaluate at interpolation nodes and verify identity matrix (delta property)."""
    n_pts = degree + 1
    nodes = _lagrange_nodes(variant, n_pts, dtype)
    res = tabulate_lagrange_1d(degree, variant, nodes)
    eye = np.eye(n_pts, dtype=dtype)
    rtol = get_default(dtype)
    nptest.assert_allclose(res, eye, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_partition_of_unity(variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
    """Sum over basis functions equals 1 for all t."""
    rng = np.random.default_rng(42)
    pts = rng.random(50).astype(dtype)
    res = tabulate_lagrange_1d(6, variant, pts)
    sums = np.sum(res, axis=-1)
    rtol = get_default(dtype)
    nptest.assert_allclose(sums, 1.0, rtol=rtol, atol=0.0)


_SNAP_CLEARANCE: float = 1e-5
"""Minimum distance from an evaluation point to the nearest node, for the test below.

``_tabulate_lagrange_basis_1D_core`` replaces any row within ``16 * eps`` of a node by a
hardcoded identity row, so a test meant to see the interpolator has to stay outside that
window. ``16 * eps`` is 3.6e-15; this leaves ten orders of magnitude. The tightest
approach the fixed points make is 1e-4, from ``0.5001`` against the node at ``0.5``
that every symmetric variant carries at even degree, so this is a guard against a later
edit moving a point onto a node, not a constraint the current data is near.
"""

_BARYCENTRIC_SAFETY: float = 4.0
"""Safety factor on the barycentric evaluation-error bound used below.

The bound itself is ``(degree + 1) * eps``, one relative rounding per term of the
barycentric quotient, applied to the largest cardinal value in the row so that it is a
bound on a value of that scale rather than on an absolute number. Measured against exact
rational arithmetic over the five variants at degrees 1 to 12, the worst relative error
is 1.55e-15 (equispaced, degree 12) against a bare bound of 2.89e-15 there, a margin of
only 1.9. Four takes that to 7.4 while still failing by twelve orders of magnitude on
anything that changes the polynomial rather than its rounding.
"""


def _exact_cardinal_row(nodes: npt.NDArray[np.floating[Any]], point: float) -> list[Fraction]:
    """Evaluate the cardinal basis of ``nodes`` at ``point`` in exact rational arithmetic.

    A float64 is itself a rational, so :class:`~fractions.Fraction` reproduces the
    interpolation nodes the implementation actually uses with no error at all, and the
    product form ``prod_{k != j} (t - x_k) / (x_j - x_k)`` is then evaluated exactly.
    This is an independent oracle rather than a mirror: it is the defining formula of the
    basis, not the barycentric quotient the implementation evaluates.

    Args:
        nodes (npt.NDArray[np.floating[Any]]): Interpolation nodes, in any order.
        point (float): Evaluation point.

    Returns:
        list[Fraction]: One exact cardinal value per node, in the order given.
    """
    xs = [Fraction(float(v)) for v in nodes]
    t = Fraction(float(point))
    row: list[Fraction] = []
    for j, xj in enumerate(xs):
        value = Fraction(1)
        for k, xk in enumerate(xs):
            if k != j:
                value *= (t - xk) / (xj - xk)
        row.append(value)
    return row


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
@pytest.mark.parametrize("degree", [1, 2, 3, 5, 8, 12])
def test_matches_exact_rational_arithmetic_away_from_the_nodes(
    variant: LagrangeVariant, degree: int
) -> None:
    """Off-node values match the defining product formula computed in exact arithmetic.

    The delta and partition-of-unity tests above cannot see the interpolator at all:
    ``_tabulate_lagrange_basis_1D_core`` overwrites any row within ``16 * eps`` of a node
    with a hardcoded identity row, so the first is satisfied by that snap, and the second
    is satisfied by any basis that sums to one, which a wrong-degree interpolant still
    does. A mutation that lowered the interpolator's degree left both of them green.
    Evaluating away from every node is what pins the polynomial itself.
    """
    points = np.array([0.13, 0.37, 0.5001, 0.86])
    # The implementation's own nodes, not the independent reconstruction
    # `_lagrange_nodes` builds: a cardinal basis is *defined* by its nodes, so an oracle
    # for the interpolator has to be given the same ones. The two differ by an ulp for
    # Gauss-Legendre (`leggauss` against `legroots`), and an ulp of node position moves
    # the exact cardinal value by more than the interpolator's own error budget.
    # Column j is the cardinal function of node j in this order, whatever that order is.
    nodes = _get_lagrange_points(variant, degree + 1, np.float64)
    clearance = float(np.min(np.abs(points[:, None] - nodes[None, :])))
    assert clearance > _SNAP_CLEARANCE, (
        f"the evaluation points must stay clear of the snap window around every node; "
        f"closest approach is {clearance:.3e}"
    )

    values = np.asarray(tabulate_lagrange_1d(degree, variant, points), dtype=np.float64)

    for i, point in enumerate(points):
        exact = _exact_cardinal_row(nodes, float(point))
        scale = max(abs(float(value)) for value in exact)
        bound = _BARYCENTRIC_SAFETY * (degree + 1) * float(np.finfo(np.float64).eps) * scale
        for j, value in enumerate(exact):
            assert abs(values[i, j] - float(value)) <= bound, (
                f"{variant.value} degree {degree}: cardinal {j} at {point} is "
                f"{values[i, j]!r}, exact {float(value)!r}, bound {bound:.3e}"
            )


def test_scalar_and_nd_shape_preservation() -> None:
    """Scalar, 2D, and 3D inputs preserve shape with trailing basis dimension."""
    # Scalar
    vec = tabulate_lagrange_1d(3, LagrangeVariant.EQUISPACES, 0.3)
    assert vec.shape == (4,)
    # 2D
    pts2 = np.array([[0.0, 0.25], [0.5, 0.75]], dtype=np.float64)
    res2 = tabulate_lagrange_1d(2, LagrangeVariant.GAUSS_LEGENDRE, pts2)
    assert res2.shape == (2, 2, 3)
    # 3D
    pts3 = np.array([[[0.0], [0.33]], [[0.66], [1.0]]], dtype=np.float32)
    res3 = tabulate_lagrange_1d(1, LagrangeVariant.CHEBYSHEV_2ND, pts3)
    assert res3.dtype == np.float32
    assert res3.shape == (2, 2, 1, 2)


def test_list_and_tuple_inputs_promote_and_preserve() -> None:
    """List/tuple inputs handled and output shape matches input container."""
    # List of lists → shape (2, 2, n+1)
    res_list = tabulate_lagrange_1d(3, LagrangeVariant.EQUISPACES, [[0.0, 0.25], [0.5, 0.75]])
    assert res_list.shape == (2, 2, 4)
    # Tuple → shape (3, n+1)
    res_tuple = tabulate_lagrange_1d(2, LagrangeVariant.CHEBYSHEV_1ST, (0.0, 0.5, 1.0))
    assert res_tuple.shape == (3, 3)


def test_dtype_preservation_float32_float64() -> None:
    """Input dtype float32/float64 preserved in outputs."""
    pts32 = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    pts64 = np.linspace(0.0, 1.0, 7, dtype=np.float64)
    out32 = tabulate_lagrange_1d(4, LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, pts32)
    out64 = tabulate_lagrange_1d(4, LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, pts64)
    assert out32.dtype == np.float32
    assert out64.dtype == np.float64


def test_negative_degree_raises_value_error() -> None:
    """Negative degree should raise ValueError."""
    with pytest.raises(ValueError, match="degree must be non-negative"):
        tabulate_lagrange_1d(-1, LagrangeVariant.EQUISPACES, [0.0, 0.5])
