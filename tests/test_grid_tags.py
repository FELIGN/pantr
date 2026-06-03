"""Tests for the sparse cell/facet tag registries (``pantr.grid``)."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.grid import CellTags, FacetTags, uniform_grid


def test_cell_tags_set_get() -> None:
    """A cell tag stores sorted (ids, values) and reads them back."""
    tags = CellTags(num_cells=10)
    tags.set("location", [4, 1, 7], [2, 1, 2])
    ids, values = tags["location"]
    assert ids.tolist() == [1, 4, 7]  # sorted by id
    assert values.tolist() == [1, 2, 2]


def test_cell_tags_scalar_broadcast() -> None:
    """A scalar value broadcasts to every id."""
    tags = CellTags(num_cells=10)
    tags.set("inside", [2, 3, 5], 1)
    _, values = tags["inside"]
    assert values.tolist() == [1, 1, 1]


def test_cell_tags_to_dense() -> None:
    """to_dense scatters the sparse tag into a (num_cells,) array."""
    tags = CellTags(num_cells=6)
    tags.set("location", [0, 4], [1, 2])
    dense = tags.to_dense("location", fill=0)
    assert dense.tolist() == [1, 0, 0, 0, 2, 0]
    assert dense.shape == (6,)


def test_cell_tags_to_dense_custom_fill() -> None:
    """to_dense honours a custom fill for untagged cells."""
    tags = CellTags(num_cells=4)
    tags.set("m", [1], [9])
    assert tags.to_dense("m", fill=-1).tolist() == [-1, 9, -1, -1]


def test_cell_tags_membership_and_names() -> None:
    """The registry supports containment, names, len, and iteration."""
    tags = CellTags(num_cells=5)
    tags.set("a", [0], 1)
    tags.set("b", [1], 2)
    assert "a" in tags
    assert "c" not in tags
    assert set(tags.names) == {"a", "b"}
    assert len(tags) == 2  # noqa: PLR2004
    assert set(iter(tags)) == {"a", "b"}


def test_cell_tags_replace_and_remove() -> None:
    """Set replaces an existing tag; remove deletes it."""
    tags = CellTags(num_cells=5)
    tags.set("a", [0, 1], 1)
    tags.set("a", [2], 9)  # replace
    ids, values = tags["a"]
    assert ids.tolist() == [2]
    assert values.tolist() == [9]
    tags.remove("a")
    assert "a" not in tags
    with pytest.raises(KeyError):
        tags["a"]


def test_cell_tags_values_are_read_only() -> None:
    """Stored tag arrays are read-only."""
    tags = CellTags(num_cells=5)
    tags.set("a", [0, 1], [3, 4])
    ids, values = tags["a"]
    assert not ids.flags.writeable
    assert not values.flags.writeable


def test_cell_tags_validation() -> None:
    """Out-of-range ids, duplicates, length mismatch, and bad dtype are rejected."""
    tags = CellTags(num_cells=5)
    with pytest.raises(ValueError, match="in range|in \\[0"):
        tags.set("a", [5], 1)  # id == num_cells out of range
    with pytest.raises(ValueError, match="unique"):
        tags.set("a", [1, 1], 1)
    with pytest.raises(ValueError, match="length"):
        tags.set("a", [0, 1, 2], [1, 2])
    with pytest.raises(TypeError, match="integer"):
        tags.set("a", [0.5, 1.5], 1)


def test_cell_tags_negative_num_cells() -> None:
    """A negative num_cells is rejected."""
    with pytest.raises(ValueError, match=">= 0"):
        CellTags(num_cells=-1)


def test_facet_tags_set_get() -> None:
    """A facet tag stores sorted (keys, values) keyed by (cid, lfid)."""
    tags = FacetTags(num_cells=10, facets_per_cell=4)
    tags.set("bc", [[3, 1], [0, 0], [3, 0]], [7, 5, 6])
    keys, values = tags["bc"]
    # lexicographic sort by (cid, lfid)
    assert keys.tolist() == [[0, 0], [3, 0], [3, 1]]
    assert values.tolist() == [5, 6, 7]


def test_facet_tags_scalar_broadcast() -> None:
    """A scalar value broadcasts to every facet key."""
    tags = FacetTags(num_cells=4, facets_per_cell=4)
    tags.set("dirichlet", [[0, 0], [1, 2]], 1)
    _, values = tags["dirichlet"]
    assert values.tolist() == [1, 1]


def test_facet_tags_validation() -> None:
    """Bad key shape, out-of-range cid/lfid, duplicates are rejected."""
    tags = FacetTags(num_cells=4, facets_per_cell=4)
    with pytest.raises(ValueError, match="shape"):
        tags.set("a", [0, 1, 2], 1)  # not (M, 2)
    with pytest.raises(ValueError, match="cell ids"):
        tags.set("a", [[4, 0]], 1)  # cid out of range
    with pytest.raises(ValueError, match="facet ids"):
        tags.set("a", [[0, 4]], 1)  # lfid out of range
    with pytest.raises(ValueError, match="unique"):
        tags.set("a", [[0, 0], [0, 0]], 1)


def test_facet_tags_membership() -> None:
    """The facet registry supports containment, names, len, remove."""
    tags = FacetTags(num_cells=4, facets_per_cell=4)
    tags.set("a", [[0, 0]], 1)
    assert "a" in tags
    assert tags.names == ("a",)
    assert len(tags) == 1
    assert tags.facets_per_cell == 4  # noqa: PLR2004
    tags.remove("a")
    assert "a" not in tags


def test_grid_tags_are_lazy_and_cached() -> None:
    """A grid creates empty tag registries lazily and caches them."""
    g = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    assert g._cell_tags is None
    assert g._facet_tags is None
    ct = g.cell_tags
    assert ct is g.cell_tags  # cached
    assert len(ct) == 0
    assert g.facet_tags.facets_per_cell == 2 * g.ndim
    assert g.cell_tags.num_cells == g.num_cells


def test_grid_cell_tags_round_trip() -> None:
    """Tags set through a grid persist and scatter to a dense per-cell array."""
    g = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
    cut = [g.flat_cell_index((0, 0)), g.flat_cell_index((2, 1))]
    g.cell_tags.set("location", cut, 2)
    dense = g.cell_tags.to_dense("location", fill=0)
    assert dense.shape == (g.num_cells,)
    assert dense[cut[0]] == 2  # noqa: PLR2004
    assert dense[cut[1]] == 2  # noqa: PLR2004
    assert int(np.count_nonzero(dense)) == 2  # noqa: PLR2004


def test_cell_tags_to_dense_missing_key_raises() -> None:
    """to_dense raises KeyError for an unregistered tag name."""
    tags = CellTags(num_cells=4)
    with pytest.raises(KeyError):
        tags.to_dense("nonexistent")


def test_cell_tags_to_dense_custom_dtype() -> None:
    """to_dense respects a caller-supplied integer dtype."""
    tags = CellTags(num_cells=4)
    tags.set("m", [0, 2], [1, 2])
    dense = tags.to_dense("m", dtype=np.int32)
    assert dense.dtype == np.int32
    assert dense.tolist() == [1, 0, 2, 0]


def test_cell_tags_to_dense_float_dtype_raises() -> None:
    """to_dense rejects non-integer dtypes."""
    tags = CellTags(num_cells=4)
    tags.set("m", [0], [1])
    with pytest.raises(TypeError, match="integer"):
        tags.to_dense("m", dtype=np.float64)


def test_cell_tags_empty_set() -> None:
    """Setting an empty tag is valid; to_dense returns the fill value everywhere."""
    tags = CellTags(num_cells=4)
    tags.set("empty", np.array([], dtype=np.int64), 0)
    dense = tags.to_dense("empty", fill=7)
    assert dense.tolist() == [7, 7, 7, 7]


def test_facet_tags_empty_set() -> None:
    """Setting an empty facet tag is valid."""
    tags = FacetTags(num_cells=4, facets_per_cell=4)
    tags.set("none", np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64))
    keys, values = tags["none"]
    assert keys.shape == (0, 2)
    assert values.shape == (0,)


def test_facet_tags_to_dense() -> None:
    """to_dense scatters a facet tag into a (num_cells, facets_per_cell) array."""
    tags = FacetTags(num_cells=3, facets_per_cell=4)
    tags.set("bc", [[0, 0], [2, 3]], [1, 2])
    dense = tags.to_dense("bc", fill=0)
    assert dense.shape == (3, 4)
    assert dense[0, 0] == 1
    assert dense[2, 3] == 2  # noqa: PLR2004
    assert int(np.count_nonzero(dense)) == 2  # noqa: PLR2004


def test_facet_tags_to_dense_float_dtype_raises() -> None:
    """FacetTags.to_dense rejects non-integer dtypes."""
    tags = FacetTags(num_cells=2, facets_per_cell=4)
    tags.set("a", [[0, 0]], [1])
    with pytest.raises(TypeError, match="integer"):
        tags.to_dense("a", dtype=np.float32)


def test_cell_tags_to_dense_overflow_raises() -> None:
    """to_dense raises OverflowError when a narrow dtype cannot hold a stored value."""
    tags = CellTags(num_cells=4)
    tags.set("m", [0, 1], [200, 1])  # 200 does not fit in int8 (-128..127)
    with pytest.raises(OverflowError, match="truncation"):
        tags.to_dense("m", dtype=np.int8)
    # int16 range is -32768..32767; 200 fits fine.
    dense = tags.to_dense("m", dtype=np.int16)
    assert dense.tolist() == [200, 1, 0, 0]


def test_facet_tags_to_dense_overflow_raises() -> None:
    """FacetTags.to_dense raises OverflowError when a narrow dtype cannot hold a value."""
    tags = FacetTags(num_cells=2, facets_per_cell=4)
    tags.set("bc", [[0, 0]], [200])
    with pytest.raises(OverflowError, match="truncation"):
        tags.to_dense("bc", dtype=np.int8)
