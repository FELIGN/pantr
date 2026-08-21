r"""Change of basis operators for various polynomial bases in 1D.

This module provides functions to create transformation matrices between different
polynomial bases including Lagrange, Bernstein, cardinal B-spline, and monomial
bases.

Architecturally, this module serves as the **bridge between different basis types**,
providing pure mathematical functions to compute the exact ``(degree+1, degree+1)``
transformation matrices without tying the dense numerical quadrature logic directly
into the core Spline space objects.

Every public builder is named ``compute_A_to_B_1d`` and returns the matrix :math:`M` with
:math:`M \, [A\ \mathrm{values}](x) = [B\ \mathrm{values}](x)`.

Supported ``(degree, dtype)`` domain
------------------------------------

Five of the builders recover their result from a linear solve, and a solve is only
defined while the matrix it uses is not singular to working precision. Each of those five
declares the largest degree it supports per dtype and raises :class:`ValueError` past it,
naming the degree, the dtype and the supported range, rather than letting
:class:`numpy.linalg.LinAlgError` or an overflow to infinity escape. The two builders that
solve nothing carry no degree limit -- :func:`compute_lagrange_to_bernstein_1d`, which is
a tabulation, and :func:`compute_legendre_to_cardinal_1d`, whose Gram matrix is the
identity -- and :func:`compute_monomial_to_bernstein_1d` evaluates a closed form.

The boundary is the largest degree :math:`p` at which the matrix :math:`M_p` the builder
solves with satisfies

.. math::

    \kappa_\infty(M_p) \, \varepsilon < 1 ,

with :math:`\varepsilon` the dtype's machine epsilon. Nothing is tuned here. The standard
perturbation bound for a linear system puts the relative error of the computed solution at
order :math:`\kappa_\infty(M) \varepsilon` (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., SIAM 2002), so at :math:`\kappa_\infty \varepsilon = 1` that bound
reaches 100% and stops asserting anything at all, which is also where :math:`M` becomes
singular to working precision. It is the only threshold-free choice, and it is a
*necessary* condition rather than a promise of accuracy: each builder's ``Warning:``
section states the accuracy actually attained, and that degrades long before the boundary.

Two consequences of the inequality are worth stating, because they are the failures the
domain replaces. Inside the domain the inverse obeys
:math:`\lVert M^{-1} \rVert_\infty = \kappa_\infty(M) / \lVert M \rVert_\infty <
1 / (\varepsilon \lVert M \rVert_\infty)`, and :math:`\lVert M \rVert_\infty \ge 0.036`
over every supported range here, so no entry of an inverse can exceed
:math:`2.3 \times 10^{8}` in float32 -- thirty orders of magnitude below that format's
overflow threshold. The exactly zero pivot :class:`numpy.linalg.LinAlgError` reports is not
ruled out by that inequality -- the perturbation bound constrains the *exact* matrix, not the
one pantr forms -- but it is not observed anywhere near the boundary either. Measured, both
failures occur only in float32 and only in the cardinal-to-Bernstein and cardinal-to-Legendre
directions: the first infinity is 26 degrees past the boundary in both, and the first
``LinAlgError`` 29 and 32 degrees past.

The :math:`\kappa_\infty` values behind the tabulated limits come from matrices rebuilt in
exact rational (or 60-digit decimal) arithmetic from closed forms, rather than from the
matrices pantr computes. That distinction is load-bearing for the Bernstein/cardinal pair,
whose forward matrix is itself the output of a Gram solve and so is not exact: near the
boundary, ``numpy.linalg.cond(M, numpy.inf)`` applied to the matrix pantr forms reports 8.6
times too much at degree 13 and 4.9 times too little at degree 15, either of which moves the
crossing. (The norm has to be named: :func:`numpy.linalg.cond` defaults to the 2-norm, and
the criterion here is stated throughout in the infinity norm the perturbation bound uses.)
Applied to the *exact* matrix the same call tracks :math:`\kappa_\infty` to five digits at
:math:`1.1 \times 10^{16}` and still to four at :math:`3.7 \times 10^{17}`, so it is the
matrix and not the estimator that is at fault. Rebuilding from closed forms also keeps the
derivation independent of the code it grades. It runs as a test,
``tests/test_change_basis_domain.py``, so the tabulated degrees are checked rather than
asserted.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from .._backend import backend_keyed_cache
from ..basis import LagrangeVariant
from ..basis._basis_lagrange import _get_lagrange_points
from ..basis._basis_utils import (
    _allocate_or_validate_out,
    _validate_float_dtype,
)
from ..quad import get_gauss_legendre_1d
from ._change_basis_backend import (
    bernstein_to_cardinal_kernel,
    bernstein_to_lagrange_kernel,
    cardinal_dual_legendre_coeffs_kernel,
    cardinal_to_bernstein_kernel,
    cardinal_to_legendre_kernel,
    lagrange_to_bernstein_kernel,
    legendre_to_cardinal_kernel,
    monomial_to_bernstein_kernel,
)
from ._change_basis_core import _gram_projection


class _DegreeLimit(NamedTuple):
    """Largest degree a change-of-basis builder supports, per floating-point format.

    Attributes:
        float32 (int): Largest degree supported when the output dtype is ``float32``.
        float64 (int): Largest degree supported when the output dtype is ``float64``.
    """

    float32: int
    float64: int


_CARDINAL_TO_BERNSTEIN_MAX_DEGREE: Final = _DegreeLimit(float32=8, float64=14)
r"""Supported degrees of :func:`compute_cardinal_to_bernstein_1d`.

Governing matrix: ``A``, the Bernstein-to-cardinal matrix this builder inverts.
:math:`\kappa_\infty(A) \varepsilon` is ``0.46`` at degree 8 against ``6.7`` at degree 9 in
float32, and ``0.091`` at degree 14 against ``2.4`` at degree 15 in float64.

This builder is the one whose *accuracy* the solvability criterion does not bound, and the
distinction matters enough to state here rather than only in the ``Warning:``. ``A`` is not
exact: it comes out of :func:`compute_bernstein_to_cardinal_1d`'s own Gram solve, so it
carries a relative error of order :math:`\kappa_\infty(G_{\mathrm{bern}}) \varepsilon`, and
inverting a perturbed matrix multiplies that by :math:`\kappa_\infty(A)`. The bound on what
is returned is therefore the *product*
:math:`\kappa_\infty(A) \kappa_\infty(G_{\mathrm{bern}}) \varepsilon`, which reaches 100%
after degree 5 in float32 and after degree 10 in float64 -- well inside the domain. Those
two degrees are exact and platform-independent, and they are the honest accuracy statement;
they are deliberately *not* the domain, because the product bound is provably pessimistic
here (at degree 11 in float64 it stands at ``9.2`` while the returned matrix is accurate to
``3e-4``), so enforcing it would refuse results with four good digits.
"""

_BERNSTEIN_TO_CARDINAL_MAX_DEGREE: Final = _DegreeLimit(float32=12, float64=26)
r"""Supported degrees of :func:`compute_bernstein_to_cardinal_1d`.

Governing matrix: the Bernstein Gram matrix :math:`G_{\mathrm{bern}}` this builder's
projection solves with, whose condition number grows like :math:`4^p`.
:math:`\kappa_\infty(G_{\mathrm{bern}}) \varepsilon` is ``0.87`` at degree 12 against
``3.2`` at degree 13 in float32, and ``0.30`` at degree 26 against ``1.17`` at degree 27 in
float64. Measured against the exact rational answer, the returned matrix is still within
``5.3e-2`` relative at degree 12 in float32. That measurement only corroborates the
criterion here; it does not set the limit, and no limit in this module is set by a measured
quantity.
"""

_CARDINAL_TO_LEGENDRE_MAX_DEGREE: Final = _DegreeLimit(float32=6, float64=12)
r"""Supported degrees of the cardinal-to-Legendre direction.

Shared by :func:`compute_cardinal_to_legendre_1d` and
:func:`compute_cardinal_dual_legendre_coeffs_1d`, whose result is that matrix transposed.

Governing matrix: ``A``, the Legendre-to-cardinal matrix the first of the two inverts.
:math:`\kappa_\infty(A) \varepsilon` is ``0.068`` at degree 6 against ``1.6`` at degree 7
in float32, and ``0.17`` at degree 12 against ``8.2`` at degree 13 in float64. Unlike the
Bernstein pair, ``A`` here *is* accurate -- its Gram matrix is the identity -- so the
solvability criterion bounds the accuracy too, and no separate accuracy statement is needed.
"""

_BERNSTEIN_TO_LAGRANGE_MAX_DEGREE: Final[Mapping[LagrangeVariant, _DegreeLimit]] = MappingProxyType(
    {
        LagrangeVariant.EQUISPACES: _DegreeLimit(float32=17, float64=37),
        LagrangeVariant.GAUSS_LEGENDRE: _DegreeLimit(float32=22, float64=51),
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE: _DegreeLimit(float32=23, float64=52),
        LagrangeVariant.CHEBYSHEV_1ST: _DegreeLimit(float32=23, float64=52),
        LagrangeVariant.CHEBYSHEV_2ND: _DegreeLimit(float32=23, float64=52),
    }
)
r"""Supported degrees of :func:`compute_bernstein_to_lagrange_1d`, per node family.

Governing matrix: ``C``, the Lagrange-to-Bernstein matrix this builder inverts, whose
condition number depends on the nodes -- roughly :math:`2.7^p` for equispaced nodes against
:math:`2^p` for the other four families, which is why equispaced nodes lose five degrees in
float32 and fourteen in float64. :math:`\kappa_\infty(C) \varepsilon` at each entry's
degree and at the one past it: equispaced ``0.63``/``1.7`` (float32) and ``0.43``/``1.2``
(float64); Gauss-Legendre ``0.63``/``1.3`` and ``0.66``/``1.3``; Gauss-Lobatto-Legendre
``0.73``/``1.5`` and ``0.73``/``1.5``; Chebyshev 1st ``0.97``/``1.9`` and ``0.99``/``2.0``;
Chebyshev 2nd ``0.61``/``1.2`` and ``0.57``/``1.1``.

The nodes are the ones :func:`~pantr.basis._basis_lagrange._get_lagrange_points` returns,
carried into the exact computation as their float64 values, so these limits describe the
matrix pantr actually forms rather than an idealized one.
"""


def _validate_degree_in_domain(
    degree: int,
    dtype: npt.DTypeLike,
    max_degree: _DegreeLimit,
    builder: str,
) -> None:
    """Refuse a ``(degree, dtype)`` pair outside a builder's supported domain.

    See the module docstring for what sets the boundary. This is a pure precondition:
    inside the domain it does nothing at all, so it cannot perturb a returned matrix. It runs
    *before* the output array is allocated, so a degree far outside the domain is refused with
    this message rather than by whatever the allocation of a huge array would raise.

    Args:
        degree (int): Polynomial degree the caller asked for.
        dtype (npt.DTypeLike): Output dtype; validated here, since this runs before the
            output array exists.
        max_degree (_DegreeLimit): Largest supported degree per format.
        builder (str): Name of the calling builder, quoted in the message so the caller is
            told which function refused rather than only that some matrix was singular.

    Raises:
        ValueError: If ``dtype`` is not float32/float64, or ``degree`` exceeds what that
            dtype supports.
    """
    _validate_float_dtype(dtype)
    resolved = np.dtype(dtype)
    limit = max_degree.float32 if resolved == np.float32 else max_degree.float64
    if degree <= limit:
        return
    raise ValueError(
        f"{builder}: degree {degree} is outside the supported domain for {resolved.name} "
        f"(supported: degree <= {max_degree.float32} in float32, "
        f"degree <= {max_degree.float64} in float64). Past that degree the matrix this "
        f"builder solves with is singular to working precision, so the error bound on the "
        f"result reaches 100% and guarantees no correct digit; see the pantr.change_basis "
        f"module docstring for the derivation."
    )


def _prepare_square_out(
    degree: int,
    dtype: npt.DTypeLike,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate ``dtype`` and allocate/validate a ``(degree+1, degree+1)`` matrix.

    Shared prologue for the ``compute_*_1d`` change-of-basis builders, which all
    return a square ``(degree+1, degree+1)`` matrix in a validated float dtype.

    Args:
        degree (int): Polynomial degree; the matrix is ``(degree+1, degree+1)``.
        dtype (npt.DTypeLike): Output dtype; must be ``float32`` or ``float64``.
        out (npt.NDArray[np.float32 | np.float64] | None): Caller-provided output
            array to validate, or ``None`` to allocate a fresh one.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The validated or freshly allocated
        ``(degree+1, degree+1)`` matrix.

    Raises:
        ValueError: If ``dtype`` is not float32/float64, or ``out`` has the wrong
            shape, dtype, or is not writeable.
    """
    _validate_float_dtype(dtype)
    return _allocate_or_validate_out(out, (degree + 1, degree + 1), dtype)


def compute_lagrange_to_bernstein_1d(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Construct the matrix mapping Lagrange basis evaluations to Bernstein basis evaluations.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering (see https://en.wikipedia.org/wiki/Bernstein_polynomial).

        This builder carries no degree limit in either dtype, and so appears in no row of
        the module docstring's domain table: it runs no linear solve that could go
        singular, the Lagrange basis being cardinal at its own nodes, so every entry is a
        single Bernstein evaluation. The inverse direction,
        :func:`compute_bernstein_to_lagrange_1d`, is the one that carries a domain.

    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Lagrange values] = [Bernstein values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is lower than 1, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")
    out = _prepare_square_out(degree, dtype, out)

    points = _get_lagrange_points(lagrange_variant, degree + 1, dtype)
    lagrange_to_bernstein_kernel()(degree, points, out)
    return out


def compute_bernstein_to_lagrange_1d(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Construct the matrix mapping Bernstein basis evaluations to Lagrange basis evaluations.

    The inverse direction of :func:`compute_lagrange_to_bernstein_1d`: with ``C`` from
    that function, this returns ``L`` solving ``C @ L = I``, computed by one LU solve
    against the identity (:func:`numpy.linalg.solve`). No explicit inverse is formed.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering
        (see https://en.wikipedia.org/wiki/Bernstein_polynomial).

        Only this direction involves a solve. The forward direction needs no
        projection at all: because the Lagrange basis is cardinal at its own nodes,
        ``C[j, k]`` is just :math:`B_j(x_k)`, so ``C`` costs one basis tabulation and carries
        no solve error. That is why this pair is much better conditioned than the
        Bernstein/cardinal and Legendre/cardinal ones.

    Warning:
        :math:`\kappa(C)` grows with degree and with the node family: with the default
        equispaced nodes it is ``1.2e1`` at degree 4, ``4.6e2`` at degree 8 and
        ``3.0e3`` at degree 10, so the attainable accuracy of ``L``, by any algorithm,
        is bounded by roughly :math:`\kappa(C)\varepsilon`. In float64 the round trip
        :math:`\lVert C L - I \rVert_2` holds to ``3e-16`` at degree 4, ``1e-14`` at degree 8
        and ``1e-13`` at degree 10. The crossing degrees below are quoted for
        :math:`64\,\kappa(C)\varepsilon`, the bound with the safety factor the accuracy
        tests apply; the bare :math:`\kappa(C)\varepsilon` crosses several degrees later.
        In float32 that bound stays below 0.1 through
        degree 11 for equispaced nodes and through degree 14 for the Gauss and
        Chebyshev variants, and over those ranges the measured round trip never exceeds
        ``1.1e-4`` (worst case Gauss-Legendre at degree 14; below ``7.5e-5`` for every
        variant through degree 11), so unlike the other two pairs this one remains
        usable in single precision over the whole range of practical degrees.

    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix L such
            that ``L @ [Bernstein values] = [Lagrange values]``. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is lower than 1, dtype is not float32 or float64, if `out` is
            provided and has incorrect shape or dtype, or if the
            ``(degree, dtype)`` pair is outside the supported domain:
            in float32, degree at most 17 for equispaced nodes, 22 for
            Gauss-Legendre and 23 for the Gauss-Lobatto-Legendre and Chebyshev
            families; in float64, 37, 51 and 52 respectively.
            The domain is where the solve is still defined, not where the result is
            still accurate; see the module docstring for the derivation, and the
            ``Warning:`` above for the accuracy attained inside it.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")
    # `LagrangeVariant(...)` rather than a bare lookup: an unrecognized variant would
    # otherwise leave this function through `KeyError`, which is exactly the kind of
    # undocumented exception type the domain exists to stop escaping.
    variant = LagrangeVariant(lagrange_variant)
    _validate_degree_in_domain(
        degree,
        dtype,
        _BERNSTEIN_TO_LAGRANGE_MAX_DEGREE[variant],
        f"compute_bernstein_to_lagrange_1d with {variant.name} nodes",
    )
    out = _prepare_square_out(degree, dtype, out)

    points = _get_lagrange_points(variant, degree + 1, dtype)
    bernstein_to_lagrange_kernel()(degree, points, out)
    return out


def _compute_change_basis_1D(
    new_basis_eval: Callable[
        [npt.NDArray[np.float32 | np.float64]], npt.NDArray[np.float32 | np.float64]
    ],
    old_basis_eval: Callable[
        [npt.NDArray[np.float32 | np.float64]], npt.NDArray[np.float32 | np.float64]
    ],
    n_quad_pts: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a change of basis operator using numerical quadrature.

    Computes the matrix ``M`` with ``old_basis(x) = M @ new_basis(x)``, by solving
    ``C = G M^T`` with ``G`` the Gram matrix of the new basis and ``C`` the mixed
    inner-product matrix between the two. Row ``i`` of ``M`` holds the ``i``-th
    *old* basis function expanded in the *new* basis; the public
    ``compute_A_to_B_1d`` wrappers document ``M @ [A values] = [B values]`` and so
    pass ``new_basis_eval=A``, ``old_basis_eval=B``.

    **No public builder calls this any more.** Since the C++ port each one calls
    its own kernel through :mod:`pantr.change_basis._change_basis_backend`, so that
    an inverse builder inverts a matrix computed on the same side of the boundary
    rather than composing two backends. This remains because it is a module-level
    name that was importable from ``pantr.change_basis`` before the package split,
    and ``CLAUDE.md`` records that a downstream consumer this repository's CI
    cannot see imports pantr's private symbols. It delegates to the same
    :func:`~pantr.change_basis._change_basis_core._gram_projection` the Python
    kernels use, so there is one implementation of the projection rather than two.

    Args:
        new_basis_eval (Callable): Evaluates the new basis at an array of points.
        old_basis_eval (Callable): Evaluates the old basis at an array of points.
        n_quad_pts (int): Number of Gauss-Legendre points. Must be positive.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.
            Its shape is set by the two bases. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The change of basis matrix. If `out`
            was provided, returns the same array.

    Raises:
        ValueError: If ``n_quad_pts`` is not positive, ``dtype`` is not float32 or
            float64, or ``out`` has the wrong shape or dtype.
    """
    if n_quad_pts < 1:
        raise ValueError("Number of quadrature points must be positive.")
    _validate_float_dtype(dtype)

    points, weights = get_gauss_legendre_1d(n_quad_pts, dtype)
    new_basis = new_basis_eval(points)
    old_basis = old_basis_eval(points)

    out = _allocate_or_validate_out(out, (old_basis.shape[1], new_basis.shape[1]), dtype)
    _gram_projection(new_basis, old_basis, weights, out)
    return out


def compute_bernstein_to_cardinal_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from Bernstein to cardinal B-spline basis.

    The projected direction of the Bernstein/cardinal pair, obtained by one Gram
    projection; :func:`compute_cardinal_to_bernstein_1d` inverts the result rather
    than projecting again. This is the direction worth projecting because *its* Gram
    matrix is the Bernstein one, which is far better conditioned than the cardinal
    one: since ``cardinal = A @ bernstein``, the two are related by
    :math:`G_{\mathrm{card}} = A \, G_{\mathrm{bern}} A^\mathsf{T}` and therefore
    :math:`\kappa(G_{\mathrm{card}}) \le \kappa(A)^2 \kappa(G_{\mathrm{bern}})`. Measured at
    degree 8: ``2.4e4`` against ``8.4e16``, the latter already saturated against
    :math:`1/\varepsilon`.

    The quadrature is exact: ``degree + 1`` Gauss-Legendre points integrate
    polynomials up to degree ``2 * degree + 1`` exactly, while every inner product
    formed here is between two polynomials of degree ``degree``.

    Warning:
        Accuracy is set by the Bernstein Gram matrix this projection solves with, whose
        condition number grows like :math:`4^p`: :math:`\kappa_\infty(G_{\mathrm{bern}})` is
        ``1.7e2`` at degree 4, ``3.4e4`` at degree 8 and ``7.3e6`` at degree 12. Measured
        against the exact rational answer, the returned matrix is within ``2.4e-6`` relative
        at degree 4 and ``5.3e-2`` at degree 12 in float32, and within ``1.7e-15`` at degree
        4 in float64. This direction is much better behaved than its inverse -- see
        :func:`compute_cardinal_to_bernstein_1d` -- which is why it carries the larger
        domain of the pair.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix A such
            that ``A @ [Bernstein values] = [cardinal values]``. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, if `out` is
            provided and has incorrect shape or dtype, or if the
            ``(degree, dtype)`` pair is outside the supported domain:
            degree at most 12 in float32 and at most 26 in float64.
            The domain is where the solve is still defined, not where the result is
            still accurate; see the module docstring for the derivation, and the
            ``Warning:`` above for the accuracy attained inside it.
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    _validate_degree_in_domain(
        degree,
        dtype,
        _BERNSTEIN_TO_CARDINAL_MAX_DEGREE,
        "compute_bernstein_to_cardinal_1d",
    )
    out = _prepare_square_out(degree, dtype, out)

    points, weights = get_gauss_legendre_1d(degree + 1, dtype)
    bernstein_to_cardinal_kernel()(degree, points, weights, out)
    return out


def compute_cardinal_to_bernstein_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from cardinal B-spline to Bernstein basis.

    The inverse direction of :func:`compute_bernstein_to_cardinal_1d`: with ``A`` from
    that function, this returns ``B`` solving ``A @ B = I``, computed by one LU solve
    against the identity (:func:`numpy.linalg.solve`). No explicit inverse is formed.

    Note:
        A second Gram projection -- swapping the two evaluators and solving with the
        *cardinal* Gram matrix -- is the obvious alternative and is the wrong one. It
        is equivalent in exact arithmetic, but that Gram matrix satisfies
        :math:`\kappa(G_{\mathrm{card}}) \le \kappa(A)^2 \kappa(G_{\mathrm{bern}})` (see
        :func:`compute_bernstein_to_cardinal_1d`), so it loses twice as many digits.
        Measured round-trip error :math:`\lVert A B - I \rVert_2` in float64: ``2.9e-2`` at
        degree 8 and ``23.9`` at degree 9 via that projection, against ``1.3e-10`` and
        ``3.8e-10`` here.

    Warning:
        The cardinal-to-Bernstein map is intrinsically ill-conditioned at high degree
        -- :math:`\kappa(A)` is ``1.0e2`` at degree 4, ``9.9e3`` at degree 6, ``1.9e6`` at
        degree 8 and ``6.3e8`` at degree 10 -- so the attainable accuracy of ``B``, by
        any algorithm, is bounded by roughly :math:`\kappa(A)\varepsilon`. In float64 the
        returned matrix reproduces the Bernstein basis to about ``4e-15`` through
        degree 4, ``3e-13`` at degree 6, ``3e-11`` at degree 8 and ``2e-8`` at degree
        10, and the round trip holds to the same order. In float32
        :math:`64\,\kappa(A)\varepsilon` -- the bound with the safety factor the accuracy
        tests apply -- exceeds 0.1 from degree 7 on, so nothing is certified past degree
        6; the bare :math:`\kappa(A)\varepsilon` crosses one degree later. Measured,
        the round trip is still only ``7e-3`` at degree 8 but reaches ``1.3`` at degree
        9, so the result carries no information at all from degree 9. This is a
        property of the two bases, not of the implementation.

        Far past that point the entries themselves stop being representable: from
        degree 34 the float32 result contains infinities and NumPy reports an overflow,
        the true inverse having outgrown the format's range. How far it has outgrown it
        is not something float64 can be asked, since :math:`\kappa(A)` passes :math:`1/\varepsilon`
        for float64 by degree 16.

        The float64 domain runs to degree 14, past both of those: what the domain asserts is
        that the inversion is still *defined*, never that the result is accurate. The bound
        on the accuracy is the product
        :math:`\kappa_\infty(A) \kappa_\infty(G_{\mathrm{bern}}) \varepsilon`, because ``A``
        itself arrives from a Gram solve rather than exactly; it passes 100% after degree 5
        in float32 and after degree 10 in float64. Both figures come from exact arithmetic
        and hold on any platform.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix B such
            that ``B @ [cardinal values] = [Bernstein values]``. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, if `out` is
            provided and has incorrect shape or dtype, or if the
            ``(degree, dtype)`` pair is outside the supported domain:
            degree at most 8 in float32 and at most 14 in float64.
            The domain is where the solve is still defined, not where the result is
            still accurate; see the module docstring for the derivation, and the
            ``Warning:`` above for the accuracy attained inside it.

    Example:
        >>> import numpy as np
        >>> A = compute_bernstein_to_cardinal_1d(3)
        >>> B = compute_cardinal_to_bernstein_1d(3)
        >>> np.allclose(A @ B, np.eye(4), atol=1e-13)
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    _validate_degree_in_domain(
        degree,
        dtype,
        _CARDINAL_TO_BERNSTEIN_MAX_DEGREE,
        "compute_cardinal_to_bernstein_1d",
    )
    out = _prepare_square_out(degree, dtype, out)

    points, weights = get_gauss_legendre_1d(degree + 1, dtype)
    cardinal_to_bernstein_kernel()(degree, points, weights, out)
    return out


def compute_legendre_to_cardinal_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from orthonormal shifted Legendre to cardinal B-spline basis.

    The Legendre basis is the orthonormal shifted Legendre basis on ``[0, 1]``
    (see :func:`~pantr.basis.tabulate_legendre_1d`), so row ``j`` of the result
    holds the coefficients of the ``j``-th cardinal B-spline in that basis.

    The quadrature is exact: ``degree + 1`` Gauss-Legendre points integrate
    polynomials up to degree ``2 * degree + 1`` exactly, while every inner
    product formed here is between two polynomials of degree ``degree`` and so
    has degree ``2 * degree``. Consequently the Legendre Gram matrix is the
    identity up to round-off, and the returned matrix carries no quadrature
    error beyond floating-point round-off.

    Note:
        This builder carries no degree limit in either dtype, and so appears in no row of
        the module docstring's domain table. Its projection does solve, but with the
        Legendre Gram matrix, which is the identity up to round-off: measured
        :math:`\kappa_\infty` is ``1.00`` at every degree through 40. The inverse
        direction, :func:`compute_cardinal_to_legendre_1d`, is the one that carries a
        domain.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix A such
            that ``A @ [Legendre values] = [cardinal values]``. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.

    Example:
        >>> import numpy as np
        >>> A = compute_legendre_to_cardinal_1d(1)
        >>> np.allclose(A, [[0.5, -0.28867513], [0.5, 0.28867513]])
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)

    points, weights = get_gauss_legendre_1d(degree + 1, dtype)
    legendre_to_cardinal_kernel()(degree, points, weights, out)
    return out


def compute_cardinal_to_legendre_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from cardinal B-spline to orthonormal shifted Legendre basis.

    The inverse direction of :func:`compute_legendre_to_cardinal_1d`: with ``A``
    from that function, this returns ``W`` solving ``A @ W = I``, computed by one
    LU solve against the identity (:func:`numpy.linalg.solve`). No explicit
    inverse is formed.

    Note:
        A second Gram projection -- swapping the two evaluators and solving with
        the *cardinal* Gram matrix -- is the obvious alternative and is the wrong
        one. It is equivalent in exact arithmetic, but the cardinal Gram matrix
        satisfies :math:`\kappa(G) = \kappa(A)^2` (verified numerically to three digits
        for degrees 0-7; at degree 8 it saturates against :math:`1/\varepsilon`), so it
        loses twice as many digits. Measured round-trip error at degree 8 in
        float64: ``5e-6`` via the Gram projection against ``5e-10`` here. The
        forward direction does not face this because *its* Gram matrix is the
        Legendre one, which is the identity.

    Warning:
        The cardinal-to-Legendre map is intrinsically ill-conditioned at high
        degree -- :math:`\kappa(A)` is ``1.1e3`` at degree 4 and ``3.0e8`` at degree 8
        -- so the attainable accuracy of ``W``, by any algorithm, is bounded by
        roughly :math:`\kappa(A)\varepsilon`. In float64 the round trip holds to about
        ``1e-13`` through degree 6 and degrades to ``5e-10`` at degree 8; in
        float32 it is meaningless beyond degree 4. This is a property of the
        bases, not of the implementation.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix W such
            that ``W @ [cardinal values] = [Legendre values]``. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, if `out` is
            provided and has incorrect shape or dtype, or if the
            ``(degree, dtype)`` pair is outside the supported domain:
            degree at most 6 in float32 and at most 12 in float64.
            The domain is where the solve is still defined, not where the result is
            still accurate; see the module docstring for the derivation, and the
            ``Warning:`` above for the accuracy attained inside it.

    Example:
        Check ``A @ W``, not ``W @ A``. Solving ``A W = I`` bounds the *right*
        residual only; the left one carries no bound and is measurably worse --
        ``3.7e-15`` against ``5.1e-16`` already at degree 3, and ``8.4e-8``
        against ``2.7e-14`` by degree 10. See the ``Warning:`` above.

        >>> import numpy as np
        >>> A = compute_legendre_to_cardinal_1d(3)
        >>> W = compute_cardinal_to_legendre_1d(3)
        >>> np.allclose(A @ W, np.eye(4), atol=1e-13)
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    _validate_degree_in_domain(
        degree,
        dtype,
        _CARDINAL_TO_LEGENDRE_MAX_DEGREE,
        "compute_cardinal_to_legendre_1d",
    )
    out = _prepare_square_out(degree, dtype, out)

    points, weights = get_gauss_legendre_1d(degree + 1, dtype)
    cardinal_to_legendre_kernel()(degree, points, weights, out)
    return out


def compute_cardinal_dual_legendre_coeffs_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create the cardinal-dual :math:`L^2` functionals, expressed in the Legendre basis.

    Row ``i`` holds the coefficients, in the orthonormal shifted Legendre basis,
    of the dual function :math:`D_i` biorthogonal to the cardinal B-spline basis:

    .. math::

        \int_0^1 D_i(x) B_{p,j}(x) \, dx = \delta_{ij}

    Derivation: write the cardinal basis in the Legendre basis as
    ``cardinal = A @ legendre`` with ``A`` from
    :func:`compute_legendre_to_cardinal_1d`, and let
    :math:`D_i = \sum_k T_{ik} \tilde{p}_k`. Because the Legendre basis is orthonormal,
    :math:`\int D_i B_j = \sum_k T_{ik} A_{jk} = (A T^\mathsf{T})_{ji}`, so
    biorthogonality holds exactly when :math:`T^\mathsf{T} = A^{-1}`, i.e.
    :math:`T = A^{-\mathsf{T}}`. No inverse is formed: since
    :func:`compute_cardinal_to_legendre_1d` returns ``W`` with ``A @ W = I`` from
    an LU solve, the result is simply ``W.T``.

    The parity matters, and it is the one the derivation needs. Solving
    ``A W = I`` bounds :math:`\lVert A W - I\rVert` and *not*
    :math:`\lVert W A - I\rVert`; only one of the two residuals is guaranteed
    small, and which one follows from the side the system is solved on rather
    than from the matrices (Du Croz & Higham, *IMA J. Numer. Anal.* **12**
    (1992) 1-19, §4). The quantity above is :math:`(A T^\mathsf{T})_{ji}`, so it
    is the bounded one.

    Warning:
        Biorthogonality is limited by the same conditioning that bounds ``W`` --
        see the warning on :func:`compute_cardinal_to_legendre_1d`. In float64
        :math:`\int D_i B_j = \delta_{ij}` holds to about ``1e-13`` through degree 6 and
        to ``5e-10`` at degree 8.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) matrix whose row ``i``
            holds the Legendre coefficients of the dual function :math:`D_i`. If `out` was
            provided, returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, if `out` is
            provided and has incorrect shape or dtype, or if the
            ``(degree, dtype)`` pair is outside the supported domain:
            degree at most 6 in float32 and at most 12 in float64, inherited
            from :func:`compute_cardinal_to_legendre_1d`.
            The domain is where the solve is still defined, not where the result is
            still accurate; see the module docstring for the derivation, and the
            ``Warning:`` above for the accuracy attained inside it.

    Example:
        >>> import numpy as np
        >>> D = compute_cardinal_dual_legendre_coeffs_1d(2)
        >>> W = compute_cardinal_to_legendre_1d(2)
        >>> np.allclose(D, W.T)
        True
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    _validate_degree_in_domain(
        degree,
        dtype,
        _CARDINAL_TO_LEGENDRE_MAX_DEGREE,
        "compute_cardinal_dual_legendre_coeffs_1d",
    )
    out = _prepare_square_out(degree, dtype, out)

    points, weights = get_gauss_legendre_1d(degree + 1, dtype)
    cardinal_dual_legendre_coeffs_kernel()(degree, points, weights, out)
    return out


def compute_monomial_to_bernstein_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from monomial to Bernstein basis on [0, 1].

    Given a polynomial of degree ``degree`` written in the monomial basis on
    ``[0, 1]``, the returned matrix ``M`` converts its coefficient vector to
    the Bernstein basis: ``bern_coeffs = M @ mono_coeffs``. Equivalently, on
    basis evaluations, ``monomial(x) = M.T @ bernstein(x)``.

    The entries are ``M[i, j] = C(i, j) / C(degree, j)`` for ``j <= i``, else
    ``0``, where ``C(n, k)`` is the binomial coefficient.

    Note:
        This builder carries no degree limit in either dtype, and so appears in no row of
        the module docstring's domain table: the entries come straight from a closed form
        and no linear solve is involved at any degree.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) lower-triangular
            transformation matrix ``M`` such that ``M @ [monomial coefficients] =
            [Bernstein coefficients]``. If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out`
            is provided and has incorrect shape or dtype.
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    out = _prepare_square_out(degree, dtype, out)
    monomial_to_bernstein_kernel()(degree, out)
    return out


@backend_keyed_cache(maxsize=64)
def _cached_lagrange_to_bernstein_matrix(
    degree: int,
    lagrange_variant: LagrangeVariant,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only Lagrange-to-Bernstein change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_lagrange_to_bernstein_1d` used internally by
    the extraction-operator routines.  Because the matrix depends only on
    ``(degree, lagrange_variant, dtype)`` — never on the knot vector — it is
    safe to share a single immutable copy across all calls with matching
    arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        lagrange_variant (LagrangeVariant): Lagrange node distribution.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``C @ lagrange_values = bernstein_values``.
    """
    mat = compute_lagrange_to_bernstein_1d(degree, lagrange_variant, dtype)
    mat.flags.writeable = False
    return mat


@backend_keyed_cache(maxsize=64)
def _cached_cardinal_to_bernstein_matrix(
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only cardinal-B-spline-to-Bernstein change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_cardinal_to_bernstein_1d` used internally by
    the extraction-operator routines.  Because the matrix depends only on
    ``(degree, dtype)`` — never on the knot vector — it is safe to share a
    single immutable copy across all calls with matching arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``C @ cardinal_values = bernstein_values``.
    """
    mat = compute_cardinal_to_bernstein_1d(degree, dtype)
    mat.flags.writeable = False
    return mat


@backend_keyed_cache(maxsize=64)
def _cached_legendre_to_cardinal_matrix(
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only Legendre-to-cardinal-B-spline change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_legendre_to_cardinal_1d`.  Because the matrix depends only on
    ``(degree, dtype)`` — never on the knot vector — it is safe to share a
    single immutable copy across all calls with matching arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``A @ legendre_values = cardinal_values``.
    """
    mat = compute_legendre_to_cardinal_1d(degree, dtype)
    mat.flags.writeable = False
    return mat


@backend_keyed_cache(maxsize=64)
def _cached_cardinal_to_legendre_matrix(
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only cardinal-B-spline-to-Legendre change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_cardinal_to_legendre_1d`.  Because the matrix depends only on
    ``(degree, dtype)`` — never on the knot vector — it is safe to share a
    single immutable copy across all calls with matching arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``W @ cardinal_values = legendre_values``.
    """
    mat = compute_cardinal_to_legendre_1d(degree, dtype)
    mat.flags.writeable = False
    return mat


__all__ = [
    "compute_bernstein_to_cardinal_1d",
    "compute_bernstein_to_lagrange_1d",
    "compute_cardinal_dual_legendre_coeffs_1d",
    "compute_cardinal_to_bernstein_1d",
    "compute_cardinal_to_legendre_1d",
    "compute_lagrange_to_bernstein_1d",
    "compute_legendre_to_cardinal_1d",
    "compute_monomial_to_bernstein_1d",
]
