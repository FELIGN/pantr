r"""Change of basis operators for various polynomial bases in 1D.

The eight ``compute_A_to_B_1d`` builders and the supported ``(degree, dtype)``
domain they are defined on. Every builder returns the matrix :math:`M` with
:math:`M \, [A\ \mathrm{values}](x) = [B\ \mathrm{values}](x)`; the full
account, including the derivation of the domain, is in
:mod:`pantr.change_basis._builders`.

Main exports:

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
circle. The public surface is unchanged: every name below was importable from
``pantr.change_basis`` before and still is.
"""

# Re-exported unchanged from what was a single module, because `CLAUDE.md`
# records that a downstream consumer this repository's CI cannot see imports
# pantr's private symbols. Splitting `change_basis.py` into a package moves where
# these are defined; it must not move where they are importable from.
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
