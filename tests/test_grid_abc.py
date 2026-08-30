"""Tests for the ``pantr.grid.Grid`` contract and the defaults behind it.

The generic, box-geometry defaults are validated by wrapping a
:class:`TensorProductGrid` in a minimal subclass (:class:`_PlainGrid`) that
forwards only the five primitives and inherits every default. Comparing the
wrapper's default outputs against the tensor-product grid's specialized
overrides (notably ``locate_many`` and the lazy BVH built from
``collect_cell_bounds``) checks the defaults for correctness.

:class:`_PlainGrid` derives from ``_GridPython`` rather than from
:class:`Grid`, because that is where the primitives are enforced: ``Grid`` is a
:class:`typing.Protocol` listing every public member, so omitting a primitive
and omitting an inherited default fail it identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pantr.geometry import AABB
from pantr.grid import Grid, TensorProductGrid, uniform_grid
from pantr.grid._grid import _GridPython

if TYPE_CHECKING:
    import numpy.typing as npt


class _PlainGrid(_GridPython):
    """Minimal grid: forwards the five primitives to a wrapped tensor grid.

    Every other method is left as the inherited default, so this class
    exercises the generic implementations rather than the tensor-product
    specializations.
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
    """The Grid protocol cannot be instantiated directly.

    It survives the move to a Protocol only because the five primitives keep
    their ``@abc.abstractmethod`` decorators: ``object.__new__`` refuses an
    abstract class before ``Protocol.__init__`` gets to raise its own, less
    specific ``Protocols cannot be instantiated``. Drop those decorators and
    this is what tells you.
    """
    with pytest.raises(TypeError, match="abstract"):
        Grid()  # type: ignore[misc]


def test_incomplete_subclass_is_abstract() -> None:
    """A subclass missing a primitive cannot be instantiated.

    Pointed at ``_GridPython``, which is where the implementer's contract lives.
    ``Grid`` would answer here too -- ``typing._ProtocolMeta`` derives from
    ``abc.ABCMeta`` -- but it would answer the same way for a class omitting any
    of the nineteen defaults, which is not what this asserts.
    """

    class _Incomplete(_GridPython):
        @property
        def ndim(self) -> int:
            return 1

        @property
        def num_cells(self) -> int:
            return 0

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()  # type: ignore[abstract]


def test_grid_python_enforces_exactly_the_five_primitives() -> None:
    """The implementer's contract is five members, and it is enforced here.

    The ``Grid`` protocol lists all twenty-four public members, because
    structural typing has no inherited half; this is the only place the smaller
    claim is checked on the Python side. Its C++ counterpart is the ``GridLike``
    concept in ``cpp/include/pantr/grid/grid.hpp``.
    """
    assert set(_GridPython.__abstractmethods__) == {
        "ndim",
        "num_cells",
        "cell_bounds",
        "locate",
        "neighbor_across_facet",
    }


def test_grid_protocol_is_not_runtime_checkable() -> None:
    """An isinstance check against Grid raises rather than answering.

    Deliberate: a ``runtime_checkable`` protocol checks that the *names* exist
    and never the signatures, so it would answer ``True`` for an object that
    cannot serve as a grid. No site needs it; every grid ``isinstance`` in the
    tree tests a concrete type.
    """
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(uniform_grid([[0.0, 1.0]], 1), Grid)  # type: ignore[misc]


def test_concrete_grids_carry_no_instance_dict() -> None:
    """The protocol's ``__slots__ = ()`` keeps the slot discipline intact.

    ``Grid`` is now in every concrete grid's MRO, and ``Generic`` declares no
    ``__slots__``: omitting the one line on the protocol would give every grid a
    ``__dict__`` back, silently. Nothing else in the suite would notice.
    """
    for grid in (uniform_grid([[0.0, 2.0]], 2), _PlainGrid(uniform_grid([[0.0, 2.0]], 2))):
        assert not hasattr(grid, "__dict__")


def test_unbound_default_runs_against_a_specialized_grid() -> None:
    """``Grid.<name>(g)`` reaches the generic default, not the override.

    This is the differential oracle: it is how a specialization is compared
    against the default it replaces, and it is what would break if the protocol
    carried ``...`` bodies instead of real ones. ``TensorProductGrid``
    specializes ``locate_many``; the generic version loops over ``locate``.
    """
    tpg = uniform_grid([[0.0, 4.0], [0.0, 3.0]], [4, 3])
    rng = np.random.default_rng(11)
    pts = rng.uniform(-1.0, 5.0, size=(30, 2))
    np.testing.assert_array_equal(Grid.locate_many(tpg, pts), tpg.locate_many(pts))


def test_default_locate_many_matches_kernel() -> None:
    """The default (looping) locate_many matches the tensor-grid kernel."""
    tpg = uniform_grid([[0.0, 4.0], [0.0, 3.0]], [4, 3])
    plain = _PlainGrid(tpg)
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1.0, 5.0, size=(40, 2))
    np.testing.assert_array_equal(plain.locate_many(pts), tpg.locate_many(pts))


def test_default_query_aabb_matches_specialized() -> None:
    """The default collect_cell_bounds builds a BVH matching the tensor grid."""
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


def test_default_restrict_not_implemented() -> None:
    """The base Grid.restrict default raises NotImplementedError."""
    plain = _PlainGrid(uniform_grid([[0.0, 3.0]], 3))
    with pytest.raises(NotImplementedError, match="restrict"):
        plain.restrict([0, 1])
