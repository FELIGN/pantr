"""Tests for the ``pantr.grid.Grid`` abstract base class and its defaults.

The generic, box-geometry defaults on :class:`Grid` are validated by wrapping a
:class:`TensorProductGrid` in a minimal subclass (:class:`_PlainGrid`) that
forwards only the five abstract methods and inherits every default. Comparing
the wrapper's default outputs against the tensor-product grid's specialized
overrides (notably ``locate_many`` and the lazy BVH built from
``_collect_cell_bounds``) checks the defaults for correctness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pantr.geometry import AABB
from pantr.grid import Grid, TensorProductGrid, uniform_grid

if TYPE_CHECKING:
    import numpy.typing as npt


class _PlainGrid(Grid):
    """Minimal Grid: forwards the abstract contract to a wrapped tensor grid.

    Every non-abstract method is left as the :class:`Grid` default, so this
    class exercises the base-class implementations rather than the
    tensor-product specializations.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: TensorProductGrid) -> None:
        super().__init__()
        self._inner = inner

    @property
    def ndim(self) -> int:
        return self._inner.ndim

    @property
    def num_cells(self) -> int:
        return self._inner.num_cells

    def cell_bounds(self, cid: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self._inner.cell_bounds(cid)

    def locate(self, pt: npt.ArrayLike) -> int | None:
        return self._inner.locate(pt)

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        return self._inner.neighbor_across_facet(cid, lfid)


def test_grid_is_abstract() -> None:
    """Grid cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        Grid()  # type: ignore[abstract]


def test_incomplete_subclass_is_abstract() -> None:
    """A subclass missing an abstract method cannot be instantiated."""

    class _Incomplete(Grid):
        @property
        def ndim(self) -> int:
            return 1

        @property
        def num_cells(self) -> int:
            return 0

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()  # type: ignore[abstract]


def test_default_locate_many_matches_kernel() -> None:
    """The default (looping) locate_many matches the tensor-grid kernel."""
    tpg = uniform_grid([[0.0, 4.0], [0.0, 3.0]], [4, 3])
    plain = _PlainGrid(tpg)
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1.0, 5.0, size=(40, 2))
    np.testing.assert_array_equal(plain.locate_many(pts), tpg.locate_many(pts))


def test_default_query_aabb_matches_specialized() -> None:
    """The default _collect_cell_bounds builds a BVH matching the tensor grid."""
    tpg = uniform_grid([[0.0, 5.0], [0.0, 5.0]], 5)
    plain = _PlainGrid(tpg)
    box = AABB([1.5, 1.5], [3.5, 3.5])
    assert sorted(plain.query_aabb(box).tolist()) == sorted(tpg.query_aabb(box).tolist())


def test_default_neighbors_and_boundary() -> None:
    """Default neighbours and boundary-facet detection match the tensor grid."""
    tpg = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    plain = _PlainGrid(tpg)
    for cid in range(tpg.num_cells):
        assert sorted(plain.neighbors(cid)) == sorted(tpg.neighbors(cid))
        for lfid in range(plain.num_local_facets(cid)):
            assert plain.is_mesh_boundary_facet(cid, lfid) == tpg.is_mesh_boundary_facet(cid, lfid)


def test_default_cell_aabb_and_reference_map() -> None:
    """Default cell_aabb and reference_map reproduce the cell geometry."""
    tpg = TensorProductGrid([[0.0, 2.0, 5.0], [0.0, 4.0]])
    plain = _PlainGrid(tpg)
    cid = tpg.flat_cell_index((1, 0))
    box = plain.cell_aabb(cid)
    assert isinstance(box, AABB)
    assert box.lo.tolist() == [2.0, 0.0]
    assert box.hi.tolist() == [5.0, 4.0]
    image = plain.reference_map(cid)(np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(image, [[5.0, 4.0]])


def test_default_facet_accessors() -> None:
    """Default facet count / axis-side / bounds follow the box convention."""
    tpg = uniform_grid([[0.0, 2.0], [0.0, 2.0]], 2)
    plain = _PlainGrid(tpg)
    assert plain.num_local_facets(0) == 4
    assert plain.local_facet_axis_side(0, 3) == (1, 1)
    lo, hi = plain.local_facet_bounds(0, 0)  # axis 0, low face
    assert lo[0] == hi[0] == 0.0


def test_default_level_children_and_iter() -> None:
    """Flat-grid defaults: level 0, no children, in-order iteration."""
    tpg = uniform_grid([[0.0, 3.0]], 3)
    plain = _PlainGrid(tpg)
    assert plain.cell_level(0) == 0
    assert plain.child_cells(0) == ()
    assert list(plain.iter_cells()) == [0, 1, 2]


def test_default_tags_available() -> None:
    """A bare subclass still gets working lazy tag registries."""
    plain = _PlainGrid(uniform_grid([[0.0, 4.0]], 4))
    plain.cell_tags.set("a", [0, 2], 1)
    assert plain.cell_tags.to_dense("a").tolist() == [1, 0, 1, 0]
    assert plain.facet_tags.facets_per_cell == 2 * plain.ndim


def test_check_cid_bounds() -> None:
    """Default accessors validate the cell id."""
    plain = _PlainGrid(uniform_grid([[0.0, 2.0]], 2))
    with pytest.raises(IndexError):
        plain.cell_level(5)
    with pytest.raises(IndexError):
        plain.num_local_facets(-1)


def test_locate_many_bad_shape_raises() -> None:
    """Default locate_many validates the trailing axis."""
    plain = _PlainGrid(uniform_grid([[0.0, 2.0], [0.0, 2.0]], 2))
    with pytest.raises(ValueError, match="shape"):
        plain.locate_many(np.zeros((4, 3)))
