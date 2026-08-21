"""The names `pantr.change_basis` must keep exposing, and why a test holds them.

`change_basis` was a single module until the C++ port and is a package now.
`CLAUDE.md` records that a separate, not-yet-public downstream consumer imports
pantr's **private** symbols and that pantr's own CI cannot see breakage there, so
the split had to preserve every name the flat module made reachable.

**It did not, and nothing noticed.** Nine real pantr names stopped being
importable, among them :class:`~pantr.basis.LagrangeVariant`, which is a public
enum and the declared type of ``lagrange_variant`` in two of the eight public
builders. The package docstring asserted the opposite in prose, the changelog
repeated it, and no test checked either. This file is that check.

The list below is data, deliberately. It is the set of names bound at module scope
by ``src/pantr/change_basis.py`` at commit 1889076, minus the stdlib and typing
imports (``np``, ``npt``, ``comb``, ``functools``, ``Callable``, ``Final``,
``Mapping``, ``MappingProxyType``, ``NamedTuple``) that nothing could reasonably
have imported from a change-of-basis module.
"""

from __future__ import annotations

import importlib

import pytest

_PUBLIC = (
    "compute_bernstein_to_cardinal_1d",
    "compute_bernstein_to_lagrange_1d",
    "compute_cardinal_dual_legendre_coeffs_1d",
    "compute_cardinal_to_bernstein_1d",
    "compute_cardinal_to_legendre_1d",
    "compute_lagrange_to_bernstein_1d",
    "compute_legendre_to_cardinal_1d",
    "compute_monomial_to_bernstein_1d",
)
"""The eight builders, which are also ``__all__``."""

_MODULE_PRIVATE = (
    "_BERNSTEIN_TO_CARDINAL_MAX_DEGREE",
    "_BERNSTEIN_TO_LAGRANGE_MAX_DEGREE",
    "_CARDINAL_TO_BERNSTEIN_MAX_DEGREE",
    "_CARDINAL_TO_LEGENDRE_MAX_DEGREE",
    "_DegreeLimit",
    "_cached_cardinal_to_bernstein_matrix",
    "_cached_cardinal_to_legendre_matrix",
    "_cached_lagrange_to_bernstein_matrix",
    "_cached_legendre_to_cardinal_matrix",
    "_compute_change_basis_1D",
    "_prepare_square_out",
    "_validate_degree_in_domain",
)
"""Private names the flat module defined itself. `pantr.bspline` imports two."""

_BORROWED = (
    "LagrangeVariant",
    "_allocate_or_validate_out",
    "_get_lagrange_points",
    "_validate_float_dtype",
    "backend_keyed_cache",
    "get_gauss_legendre_1d",
    "tabulate_bernstein_1d",
    "tabulate_cardinal_bspline_1d",
    "tabulate_legendre_1d",
)
"""Names the flat module bound only because it imported them for its own use.

Reachable as ``pantr.change_basis.<name>`` before the split purely by accident of
being a single file, which does not make them any less reachable to a caller who
wrote the import. These are the nine that were lost.
"""


@pytest.mark.parametrize("name", _PUBLIC + _MODULE_PRIVATE + _BORROWED)
def test_the_name_is_still_importable_from_the_package_root(name: str) -> None:
    """Each name reachable before the package split is reachable after it.

    Args:
        name (str): The attribute to look for on :mod:`pantr.change_basis`.
    """
    module = importlib.import_module("pantr.change_basis")
    assert hasattr(module, name), (
        f"pantr.change_basis.{name} was importable before the package split and is not "
        f"now. CLAUDE.md records a downstream consumer that imports pantr's private "
        f"symbols and that this repository's CI cannot see it break, so a name is "
        f"removed deliberately or not at all"
    )


def test_only_the_eight_builders_are_public() -> None:
    """``__all__`` stays the eight builders, however many names are reachable.

    The re-exports above are a compatibility surface, not an enlargement of the
    public API, and this is what keeps the two from being confused.
    """
    module = importlib.import_module("pantr.change_basis")
    assert tuple(sorted(module.__all__)) == tuple(sorted(_PUBLIC))
