"""The names `pantr.grid` must keep importable, and why a test holds them.

``CLAUDE.md`` records that a separate, not-yet-public downstream consumer imports
pantr's **private** symbols, and that pantr's own CI cannot see breakage there.
``tests/test_bezier_reexports.py`` is that check for its module and this is the one
for this one.

**One name here is imported by that consumer**, measured over its checkout on
2026-08-21: ``pantr.grid._bvh_core._BVH_STACK_DEPTH``. It is pinned by full path
rather than by value, because the path is the fragile part: this port added a C++
mirror of the constant in ``cpp/include/pantr/grid/bvh.hpp``, and the oracle's copy
stays the source of truth precisely so that the consumer's import keeps resolving.

**The BVH's five node arrays are the other consumer-visible surface, and they are
public API rather than an implementation detail.** ``design/bvh.md`` records why: the
consumer does not call :meth:`~pantr.grid.BVH.query_aabb` at all, it walks the tree
itself, probing at each internal node with its own predicate. So the properties are
pinned here too. A port that got the layout right for queries and wrong for
traversal would break that consumer without failing a query test.

The rest of the list is the surface this port moved. Routing Layer 2 through
``pantr.grid._grid_backend`` removed the kernels from the namespaces of the three
modules that used to import them directly, so anything reachable through one of
those paths before is not reachable now. That is deliberate, and the names below are
the ones that must survive it.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from pantr.grid import BVH

_PUBLIC = (
    "BVH",
    "CellTags",
    "FacetTags",
    "Grid",
    "GridRestriction",
    "HierarchicalGrid",
    "Partition",
    "TensorProductGrid",
    "cell_quadrature",
    "hierarchical_grid",
    "overlay",
    "partition_grid",
    "tensor_product_grid",
    "uniform_grid",
)
"""The package's ``__all__``, unchanged by the port."""

_CONSUMER_PRIVATE = (("pantr.grid._bvh_core", "_BVH_STACK_DEPTH"),)
"""Private paths the downstream consumer imports, measured 2026-08-21."""

_NODE_ARRAYS = ("node_lo", "node_hi", "node_left", "node_right", "node_cell")
"""The five node arrays a downstream consumer traverses itself. See `design/bvh.md`."""

_KERNELS = (
    ("pantr.grid._locate_core", "_locate_points_core"),
    ("pantr.grid._bvh_core", "_bvh_build_core"),
    ("pantr.grid._bvh_core", "_bvh_query_count_core"),
    ("pantr.grid._bvh_core", "_bvh_query_emit_core"),
    ("pantr.grid._hier_core", "_block_of_midx"),
    ("pantr.grid._hier_core", "_decode_flat_id_core"),
    ("pantr.grid._hier_core", "_encode_midx_core"),
    ("pantr.grid._hier_core", "_hier_collect_cell_bounds_core"),
    ("pantr.grid._hier_core", "_hier_locate_points_core"),
)
"""The nine Layer 3 kernels, which stay importable from the modules that define them.

They are the Numba half of the dual backend now rather than the only implementation,
so Layer 2 reaches them through the catalogue. The modules that define them are still
where they live, and a test that imports one directly must keep working.
"""

_CATALOGUE = (
    "bvh_build_kernel",
    "bvh_query_count_kernel",
    "bvh_query_emit_kernel",
    "decode_flat_id_kernel",
    "encode_midx_kernel",
    "hier_collect_cell_bounds_kernel",
    "hier_locate_points_kernel",
    "locate_points_kernel",
)
"""The catalogue's eight accessors, added by this port."""


@pytest.mark.parametrize("name", _PUBLIC)
def test_the_public_surface_is_importable(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    module = importlib.import_module("pantr.grid")
    assert hasattr(module, name), f"pantr.grid lost {name}"


def test_all_matches_the_public_list() -> None:
    """``__all__`` is exactly the list above, so an addition has to be recorded here."""
    module = importlib.import_module("pantr.grid")
    assert tuple(sorted(module.__all__)) == tuple(sorted(_PUBLIC))


@pytest.mark.parametrize(("path", "name"), _CONSUMER_PRIVATE)
def test_the_consumer_visible_private_paths_still_resolve(path: str, name: str) -> None:
    """The private paths a downstream consumer imports are still importable.

    Failing this does not mean the change is wrong. It means the consumer's checkout
    has to be updated in the same breath, which is the whole point of finding out
    here rather than there.
    """
    module = importlib.import_module(path)
    assert hasattr(module, name), f"{path} lost {name}, which a downstream consumer imports"


def test_the_stack_depth_constant_still_has_its_value() -> None:
    """``_BVH_STACK_DEPTH`` is 128 on both sides, and the oracle's copy is the source.

    Pinned as a value as well as a path because the port added a second copy in C++.
    Two spellings of one constant that can drift is exactly the failure this catches,
    and the consumer reads the Python one.
    """
    module = importlib.import_module("pantr.grid._bvh_core")
    assert module._BVH_STACK_DEPTH == 128


@pytest.mark.parametrize("name", _NODE_ARRAYS)
def test_the_bvh_node_arrays_are_still_properties(name: str) -> None:
    """Each of the five node arrays is still readable off a built BVH.

    The consumer traverses these directly, so they are contract rather than detail.
    """
    tree = BVH.from_cell_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    assert hasattr(tree, name), f"BVH lost the {name} property a consumer traverses"


@pytest.mark.parametrize(("path", "name"), _KERNELS)
def test_the_kernels_stay_where_they_were(path: str, name: str) -> None:
    """Every Layer 3 kernel is still reachable from the module that defines it."""
    module = importlib.import_module(path)
    assert hasattr(module, name), f"{path} lost {name}"


@pytest.mark.parametrize("name", _CATALOGUE)
def test_the_catalogue_accessors_resolve(name: str) -> None:
    """Every accessor this port added is importable from the catalogue."""
    module = importlib.import_module("pantr.grid._grid_backend")
    assert hasattr(module, name), f"pantr.grid._grid_backend lost {name}"


def test_the_catalogue_covers_every_dispatched_kernel() -> None:
    """``__all__`` is exactly the eight accessors, so a ninth has to be recorded here."""
    module = importlib.import_module("pantr.grid._grid_backend")
    assert tuple(sorted(module.__all__)) == tuple(sorted(_CATALOGUE))
