"""The names `pantr.bezier` must keep importable, and why a test holds them.

`CLAUDE.md` records that a separate, not-yet-public downstream consumer imports
pantr's **private** symbols, and that pantr's own CI cannot see breakage there. The
`change_basis` port learned that the hard way: nine names stopped being importable
when a flat module became a package, the docstring claimed otherwise, and no test
checked. `tests/test_change_basis_reexports.py` is that check for its module and
this is the one for this module.

**`bezier` is the first port that touches symbols the consumer actually uses.** A
grep over its checkout on 2026-08-21 found two:
``pantr.bezier._root_finding_core._de_casteljau_eval_scalar``, a `@nb_jit` kernel,
and ``pantr.bezier._bezier.Bezier``, which is public as ``pantr.bezier.Bezier``
with only the path private. Neither is in this port's scope, which is exactly why
they are pinned here: a later stage will reorganise `_root_finding_core`, and this
is the test that will notice.

The rest of the list is the surface this port did move. Routing Layer 2 through
`pantr.bezier._bezier_backend` removed the kernels from the namespaces of the six
modules that used to import them directly, so anything reachable through one of
those paths before is not reachable now. That is a deliberate change and the names
below are the ones that must survive it.
"""

from __future__ import annotations

import importlib

import pytest

_PUBLIC = (
    "Bezier",
    "create_from_bspline",
    "find_monotone_root",
    "find_roots",
    "fit_bezier",
    "interpolate_bezier",
)
"""The package's ``__all__``, unchanged by the port."""

_CONSUMER_PRIVATE = (
    ("pantr.bezier._root_finding_core", "_de_casteljau_eval_scalar"),
    ("pantr.bezier._bezier", "Bezier"),
)
"""Private paths the downstream consumer imports, measured 2026-08-21.

Pinned by full path rather than by name, because the path is the part that is
fragile: ``Bezier`` is public under another name and would survive a move of
``_bezier.py``, while the consumer's import would not.
"""

_KERNELS = (
    "_degree_elevate_bezier_1d_core",
    "_evaluate_bezier_1d_core",
    "_evaluate_bezier_deriv_1d_core",
    "_restrict_bezier_1d_core",
    "_scalar_bernstein_product_1d_core",
    "_slice_bezier_1d_core",
    "_split_bezier_1d_core",
)
"""The seven Layer 3 kernels, which stay importable from ``_bezier_core``.

They are the Numba half of the dual backend now rather than the only
implementation, so Layer 2 reaches them through the catalogue. The module that
defines them is still where they live, and a test that imports one directly, as
``tests/test_root_finding_stress.py`` does, must keep working.
"""

_CATALOGUE = (
    "DegreeKernels",
    "degree_kernels",
    "evaluate_kernel",
    "evaluate_deriv_kernel",
    "product_kernel",
    "restrict_kernel",
    "slice_kernel",
    "split_kernel",
)
"""The catalogue's accessors, added by this port."""


@pytest.mark.parametrize("name", _PUBLIC)
def test_the_public_surface_is_importable(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    module = importlib.import_module("pantr.bezier")
    assert hasattr(module, name), f"pantr.bezier lost {name}"


def test_all_matches_the_public_list() -> None:
    """``__all__`` is exactly the list above, so an addition has to be recorded here."""
    module = importlib.import_module("pantr.bezier")
    assert tuple(sorted(module.__all__)) == tuple(sorted(_PUBLIC))


@pytest.mark.parametrize(("path", "name"), _CONSUMER_PRIVATE)
def test_the_consumer_visible_private_paths_still_resolve(path: str, name: str) -> None:
    """The private paths a downstream consumer imports are still importable.

    Failing this does not mean the change is wrong. It means the consumer's
    checkout has to be updated in the same breath, which is the whole point of
    finding out here rather than there.
    """
    module = importlib.import_module(path)
    assert hasattr(module, name), f"{path} lost {name}, which a downstream consumer imports"


@pytest.mark.parametrize("name", _KERNELS)
def test_the_kernels_stay_where_they_were(name: str) -> None:
    """Every Layer 3 kernel is still reachable from ``pantr.bezier._bezier_core``."""
    module = importlib.import_module("pantr.bezier._bezier_core")
    assert hasattr(module, name), f"_bezier_core lost {name}"


@pytest.mark.parametrize("name", _CATALOGUE)
def test_the_catalogue_exposes_its_accessors(name: str) -> None:
    """Every accessor the port added is reachable on the catalogue module."""
    module = importlib.import_module("pantr.bezier._bezier_backend")
    assert hasattr(module, name), f"_bezier_backend lost {name}"
