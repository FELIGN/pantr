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

Main exports
------------

- :func:`compute_lagrange_to_bernstein_1d` and
  :func:`compute_bernstein_to_lagrange_1d`
- :func:`compute_bernstein_to_cardinal_1d` and
  :func:`compute_cardinal_to_bernstein_1d`
- :func:`compute_legendre_to_cardinal_1d` and
  :func:`compute_cardinal_to_legendre_1d`
- :func:`compute_cardinal_dual_legendre_coeffs_1d`
- :func:`compute_monomial_to_bernstein_1d`

This was a single module until the C++ port; it is a package for the reason
:mod:`pantr.quad` is one, namely that a dispatched module needs somewhere to put
its catalogue and its Python kernels without either importing the other in a
circle.

**Every pantr name that was importable from ``pantr.change_basis`` before the
split still is**, public and private alike, and a test pins the list. What is no
longer reachable is the handful of stdlib and typing names the flat module
happened to bind -- ``np``, ``npt``, ``comb``, ``functools``, ``Callable``,
``Final``, ``Mapping``, ``MappingProxyType``, ``NamedTuple`` -- which nothing
could reasonably have imported from here.
"""

# Re-exported unchanged from what was a single module, because `CLAUDE.md`
# records that a downstream consumer this repository's CI cannot see imports
# pantr's private symbols. Splitting `change_basis.py` into a package moves where
# these are defined; it must not move where they are importable from.
#
# The second block is names the flat module bound only because it imported them
# for its own use. They were reachable as `pantr.change_basis.<name>` and are
# again. `LagrangeVariant` is the one that matters -- it is a public enum and the
# declared type of `lagrange_variant` in two of the eight functions below, so a
# caller had every reason to import it from here. The rest are restored because
# the cost is a line each and the alternative is guessing which of them the
# downstream consumer uses. `tests/test_change_basis_reexports.py` pins the set.
from .._backend import backend_keyed_cache as backend_keyed_cache
from ..basis import LagrangeVariant as LagrangeVariant
from ..basis import tabulate_bernstein_1d as tabulate_bernstein_1d
from ..basis import tabulate_cardinal_bspline_1d as tabulate_cardinal_bspline_1d
from ..basis import tabulate_legendre_1d as tabulate_legendre_1d
from ..basis._basis_lagrange import _get_lagrange_points as _get_lagrange_points
from ..basis._basis_utils import _allocate_or_validate_out as _allocate_or_validate_out
from ..basis._basis_utils import _validate_float_dtype as _validate_float_dtype
from ..quad import get_gauss_legendre_1d as get_gauss_legendre_1d
from ._builders import (
    _BERNSTEIN_TO_CARDINAL_MAX_DEGREE as _BERNSTEIN_TO_CARDINAL_MAX_DEGREE,
)
from ._builders import (
    _BERNSTEIN_TO_LAGRANGE_MAX_DEGREE as _BERNSTEIN_TO_LAGRANGE_MAX_DEGREE,
)
from ._builders import (
    _CARDINAL_TO_BERNSTEIN_MAX_DEGREE as _CARDINAL_TO_BERNSTEIN_MAX_DEGREE,
)
from ._builders import (
    _CARDINAL_TO_LEGENDRE_MAX_DEGREE as _CARDINAL_TO_LEGENDRE_MAX_DEGREE,
)
from ._builders import (
    _cached_cardinal_to_bernstein_matrix as _cached_cardinal_to_bernstein_matrix,
)
from ._builders import (
    _cached_cardinal_to_legendre_matrix as _cached_cardinal_to_legendre_matrix,
)
from ._builders import (
    _cached_lagrange_to_bernstein_matrix as _cached_lagrange_to_bernstein_matrix,
)
from ._builders import (
    _cached_legendre_to_cardinal_matrix as _cached_legendre_to_cardinal_matrix,
)
from ._builders import (
    _compute_change_basis_1D as _compute_change_basis_1D,
)
from ._builders import (
    _DegreeLimit as _DegreeLimit,
)
from ._builders import (
    _prepare_square_out as _prepare_square_out,
)
from ._builders import (
    _validate_degree_in_domain as _validate_degree_in_domain,
)
from ._builders import (
    compute_bernstein_to_cardinal_1d,
    compute_bernstein_to_lagrange_1d,
    compute_cardinal_dual_legendre_coeffs_1d,
    compute_cardinal_to_bernstein_1d,
    compute_cardinal_to_legendre_1d,
    compute_lagrange_to_bernstein_1d,
    compute_legendre_to_cardinal_1d,
    compute_monomial_to_bernstein_1d,
)

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
