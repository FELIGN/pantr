"""What the ``Grid`` protocol accepts and refuses, checked by mypy rather than by running.

Three claims, and the pair in the middle is the one that carries the argument.

``_AlmostGrid`` and ``_Duck`` differ by exactly one member -- ``hanging_neighbors`` --
and neither inherits anything from ``pantr.grid``. So the protocol is satisfiable
structurally, which is the whole premise of the port's seam, and a class that misses one
member is refused where a ``Grid`` is expected. Without the accepted half the refused
half would prove nothing: a harness that rejects everything rejects the bad case too.

The member they differ by is deliberately a **default**, not a primitive. Ticket #386's
``AC6`` asked for "a class that omits one of the six primitives", and that test cannot be
written: a protocol has no inherited half, so ``Grid`` must list every public member a
consumer may call, and omitting ``hanging_neighbors`` fails exactly as omitting
``cell_bounds`` would. The smaller claim -- that an *implementer* owes five members and
inherits the rest -- is enforced elsewhere, and the third case below is where: on
``_GridPython.__abstractmethods__``. Its C++ counterpart is the ``GridLike`` concept in
``cpp/include/pantr/grid/grid.hpp``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pantr.grid import Grid
from pantr.grid._grid import _GridPython

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from pantr.geometry import AABB
    from pantr.grid import BVH, CellTags, FacetTags, GridRestriction
    from pantr.transform import AffineTransform


def take_grid(grid: Grid) -> int:
    """Stand in for every ``grid: Grid`` annotation in the library.

    Args:
        grid (Grid): Any object satisfying the protocol.

    Returns:
        int: The grid's cell count.
    """
    return grid.num_cells


class _AlmostGrid:
    """Everything the protocol asks for except ``hanging_neighbors``.

    Inherits nothing from ``pantr.grid``: this is the shape a wrapper over a C++ handle
    will have, so the file also pins that such a wrapper can satisfy ``Grid`` at all.
    Every body raises; mypy checks signatures, and nothing here is ever run.
    """

    __slots__ = ("_bvh", "_cell_tags", "_facet_tags")

    _bvh: BVH | None
    _cell_tags: CellTags | None
    _facet_tags: FacetTags | None

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def num_cells(self) -> int:
        raise NotImplementedError

    def cell_bounds(self, cid: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError

    def locate(self, pt: npt.ArrayLike) -> int | None:
        raise NotImplementedError

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        raise NotImplementedError

    def iter_cells(self) -> Iterator[int]:
        raise NotImplementedError

    def cell_aabb(self, cid: int) -> AABB:
        raise NotImplementedError

    def cell_level(self, cid: int) -> int:
        raise NotImplementedError

    def child_cells(self, cid: int) -> tuple[int, ...]:
        raise NotImplementedError

    def reference_map(self, cid: int) -> AffineTransform:
        raise NotImplementedError

    def neighbors(self, cid: int) -> list[int]:
        raise NotImplementedError

    def restrict(self, cell_ids: npt.ArrayLike) -> GridRestriction:
        raise NotImplementedError

    def num_local_facets(self, cid: int) -> int:
        raise NotImplementedError

    def local_facet_axis_side(self, cid: int, lfid: int) -> tuple[int, int]:
        raise NotImplementedError

    def local_facet_bounds(
        self, cid: int, lfid: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError

    def is_mesh_boundary_facet(self, cid: int, lfid: int) -> bool:
        raise NotImplementedError

    def boundary_facets(self) -> npt.NDArray[np.int64]:
        raise NotImplementedError

    def locate_many(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        raise NotImplementedError

    def query_aabb(self, aabb: AABB) -> npt.NDArray[np.int64]:
        raise NotImplementedError

    def cell_bvh(self) -> BVH:
        raise NotImplementedError

    def collect_cell_bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError

    @property
    def cell_tags(self) -> CellTags:
        raise NotImplementedError

    @property
    def facet_tags(self) -> FacetTags:
        raise NotImplementedError

    def _check_cid(self, cid: int) -> None:
        raise NotImplementedError

    def _check_lfid(self, cid: int, lfid: int) -> None:
        raise NotImplementedError

    def _normalize_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        raise NotImplementedError


class _Duck(_AlmostGrid):
    """``_AlmostGrid`` plus the one member it lacks, and so a ``Grid``."""

    __slots__ = ()

    def hanging_neighbors(self, cid: int, lfid: int) -> tuple[int, ...]:
        raise NotImplementedError


class _IncompleteSubclass(_GridPython):
    """An implementer that supplies four of the five primitives."""

    __slots__ = ()

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def num_cells(self) -> int:
        raise NotImplementedError

    def cell_bounds(self, cid: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError

    def locate(self, pt: npt.ArrayLike) -> int | None:
        raise NotImplementedError


# Satisfying the protocol without inheriting from it: accepted.
take_grid(_Duck())

# One public member short: refused, and the member is a default rather than a primitive.
take_grid(_AlmostGrid())  # expect-error: arg-type

# One primitive short: refused at construction, which is the claim `AC6` wanted and the
# protocol cannot make.
_IncompleteSubclass()  # expect-error: abstract
