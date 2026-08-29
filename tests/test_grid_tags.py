"""Tests for the sparse cell/facet tag registries (``pantr.grid``)."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.grid import CellTags, FacetTags, uniform_grid
from tests._parity_harness import demand_cpp_backend


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
    assert len(tags) == 2
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
    with pytest.raises(ValueError, match=r"in range|in \[0"):
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
    assert tags.facets_per_cell == 4
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
    assert dense[cut[0]] == 2
    assert dense[cut[1]] == 2
    assert int(np.count_nonzero(dense)) == 2


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
    assert dense[2, 3] == 2
    assert int(np.count_nonzero(dense)) == 2


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


# ---------------------------------------------------------------------------
# The port to C++ ownership (FELIGN/pantr#382)
# ---------------------------------------------------------------------------
#
# Everything above ran against the pure-Python registries and now runs against
# whichever implementation ``PANTR_BACKEND`` selects, unchanged. What follows is
# the part the port added, and each case is here because nothing above could see
# it: a view's lifetime, the insertion order across a replacement, and a pickle
# crossing between the two backends.


def _backend_pairs() -> list[tuple[Backend, Backend]]:
    """Every ordered pair of backends, for the serialization round trips.

    Returns:
        list[tuple[Backend, Backend]]: The four ``(writer, reader)`` pairs.
    """
    return [(writer, reader) for writer in Backend for reader in Backend]


@pytest.fixture
def both_backends() -> None:
    """Require the compiled extension for a test that uses both backends at once.

    Routed through the parity harness rather than a bare ``skipif`` for the reason
    that function's docstring gives: a bare skip is silent, and a suite that skips
    its way to green has let real failures through in this repository.
    """
    demand_cpp_backend()


def test_a_view_taken_before_a_replacement_still_holds_the_old_values() -> None:
    """``__getitem__``'s arrays survive a later ``set`` on the same name.

    The C++ registry hands back zero-copy views into its own storage, so a ``set``
    that replaced the buffer in place would free memory under a live numpy array --
    a use-after-free the pure-Python registry cannot have, because a replaced tuple
    stays alive as long as someone holds it. ``tags.hpp`` holds each tag behind a
    ``shared_ptr`` for exactly this, and nothing else in the suite reads a view
    across a replacement.
    """
    tags = CellTags(num_cells=10)
    tags.set("a", [1, 2], [10, 20])
    ids, values = tags["a"]

    tags.set("a", [5], [50])

    assert ids.tolist() == [1, 2]
    assert values.tolist() == [10, 20]
    assert tags["a"][0].tolist() == [5]

    tags.remove("a")
    assert ids.tolist() == [1, 2]
    assert values.tolist() == [10, 20]


def test_a_facet_view_taken_before_a_replacement_still_holds_the_old_values() -> None:
    """The facet registry's ``(keys, values)`` views have the same lifetime."""
    tags = FacetTags(num_cells=6, facets_per_cell=4)
    tags.set("bc", [[0, 1], [2, 3]], [7, 9])
    keys, values = tags["bc"]

    tags.set("bc", [[1, 0]], [4])

    assert keys.tolist() == [[0, 1], [2, 3]]
    assert values.tolist() == [7, 9]


def test_replacing_a_cell_tag_leaves_its_position_unchanged() -> None:
    """A replaced name keeps its index in ``names`` and in iteration.

    Asserted by position rather than by comparing the two backends: a registry that
    moved a replaced name to the end would be wrong in both backends at once, and a
    comparison between them would pass.
    """
    tags = CellTags(num_cells=10)
    for i, name in enumerate(("a", "b", "c")):
        tags.set(name, [i], 1)

    tags.set("b", [7], 99)

    assert list(tags.names) == ["a", "b", "c"]
    assert list(iter(tags)) == ["a", "b", "c"]
    assert tags["b"][0].tolist() == [7]

    tags.remove("a")
    assert list(tags.names) == ["b", "c"]


def test_replacing_a_facet_tag_leaves_its_position_unchanged() -> None:
    """The facet registry keeps insertion order across a replacement too."""
    tags = FacetTags(num_cells=6, facets_per_cell=4)
    for i, name in enumerate(("first", "second", "third")):
        tags.set(name, [[i, 0]], 1)

    tags.set("second", [[5, 3]], 42)

    assert list(tags.names) == ["first", "second", "third"]
    assert tags["second"][0].tolist() == [[5, 3]]


def test_remove_reports_an_unregistered_name_as_a_key_error() -> None:
    """``remove`` raises ``KeyError``, which nanobind cannot produce on its own.

    The registry's own C++ lookup throws ``std::out_of_range``, which nanobind maps
    to ``IndexError``; the wrapper checks membership first so that a caller catches
    what the docstring promises. ``__getitem__`` and ``to_dense`` are covered above.
    """
    tags = CellTags(num_cells=4)
    with pytest.raises(KeyError):
        tags.remove("nonexistent")

    facets = FacetTags(num_cells=4, facets_per_cell=4)
    with pytest.raises(KeyError):
        facets.remove("nonexistent")


def test_a_non_string_name_is_absent_rather_than_an_error() -> None:
    """``5 in tags`` is ``False`` and ``tags[5]`` is a ``KeyError``.

    ``set`` stores ``str(name)``, so no non-string key can exist. The C++
    registry's ``__contains__`` takes a ``str`` and would raise ``TypeError``, so
    the wrapper answers this itself.
    """
    tags = CellTags(num_cells=4)
    tags.set("5", [0], 1)
    assert 5 not in tags
    assert "5" in tags
    with pytest.raises(KeyError):
        tags[5]  # type: ignore[index]


def test_the_registries_print_their_counts_and_their_tags() -> None:
    """Both registries have a ``__repr__``, formatted by the wrapper.

    ``FacetTags`` had one before the port and ``CellTags`` did not; the addition is
    deliberate and is made on both sides at once, so the two backends and the two
    registries cannot drift apart. Computed by the wrapper, so the string does not
    depend on ``PANTR_BACKEND``.
    """
    cells = CellTags(num_cells=6)
    cells.set("inside", [0, 1], 1)
    assert repr(cells) == "CellTags(num_cells=6, tags=['inside'])"

    facets = FacetTags(num_cells=3, facets_per_cell=4)
    assert repr(facets) == "FacetTags(num_cells=3, facets_per_cell=4, tags=[])"
    facets.set("bc", [[0, 0]], 2)
    assert repr(facets) == "FacetTags(num_cells=3, facets_per_cell=4, tags=['bc'])"


def test_the_registries_refuse_attribute_writes() -> None:
    """A registry accumulates through ``set``; its own attributes are not settable."""
    tags = CellTags(num_cells=4)
    with pytest.raises(AttributeError):
        tags.num_cells = 9  # type: ignore[misc]

    facets = FacetTags(num_cells=4, facets_per_cell=4)
    with pytest.raises(AttributeError):
        facets.facets_per_cell = 9  # type: ignore[misc]


@pytest.mark.parametrize(("writer", "reader"), _backend_pairs())
def test_cell_tags_round_trip_across_every_backend_pair(
    both_backends: None, writer: Backend, reader: Backend
) -> None:
    """A pickled registry written under one backend loads under the other.

    Asserted on the reconstructed contents rather than on the absence of an
    exception: the failure this guards against is a wire format that quietly became
    backend-specific, and a handle that pickled to *something* would still fail
    that way.
    """
    del both_backends
    with use_backend(writer):
        tags = CellTags(num_cells=6)
        tags.set("labels", [3, 1], [30, 10])
        tags.set("inside", [0], 1)
        blob = pickle.dumps(tags)

    with use_backend(reader):
        loaded = pickle.loads(blob)
        assert loaded.num_cells == 6
        assert loaded.names == ("labels", "inside")
        np.testing.assert_array_equal(loaded["labels"][0], [1, 3])
        np.testing.assert_array_equal(loaded["labels"][1], [10, 30])
        np.testing.assert_array_equal(loaded["inside"][1], [1])
        assert loaded.to_dense("labels", fill=-1).tolist() == [-1, 10, -1, 30, -1, -1]


@pytest.mark.parametrize(("writer", "reader"), _backend_pairs())
def test_facet_tags_round_trip_across_every_backend_pair(
    both_backends: None, writer: Backend, reader: Backend
) -> None:
    """The facet registry pickles across the same four pairs."""
    del both_backends
    with use_backend(writer):
        tags = FacetTags(num_cells=3, facets_per_cell=4)
        tags.set("bc", [[2, 3], [0, 0]], [2, 1])
        blob = pickle.dumps(tags)

    with use_backend(reader):
        loaded = pickle.loads(blob)
        assert loaded.num_cells == 3
        assert loaded.facets_per_cell == 4
        np.testing.assert_array_equal(loaded["bc"][0], [[0, 0], [2, 3]])
        np.testing.assert_array_equal(loaded["bc"][1], [1, 2])


@pytest.mark.parametrize(("writer", "reader"), _backend_pairs())
def test_deepcopy_round_trips_across_every_backend_pair(
    both_backends: None, writer: Backend, reader: Backend
) -> None:
    """``deepcopy`` goes through the same reduction, so it crosses backends too.

    The copy is taken under ``reader`` while the original was built under
    ``writer``, which is the one arrangement that exercises the reduction rather
    than a same-backend shortcut.
    """
    del both_backends
    with use_backend(writer):
        tags = CellTags(num_cells=5)
        tags.set("m", [4, 2], [40, 20])

    with use_backend(reader):
        clone = copy.deepcopy(tags)
        assert clone.names == ("m",)
        np.testing.assert_array_equal(clone["m"][0], [2, 4])
        np.testing.assert_array_equal(clone["m"][1], [20, 40])
        clone.set("m", [0], 7)

    assert tags["m"][0].tolist() == [2, 4], "the copy must not share the original's storage"


def test_both_backends_agree_on_the_messages_a_caller_reads(both_backends: None) -> None:
    """Every rejection carries the same text under either backend.

    A port that changed what a caller reads or catches would pass every value
    assertion above. The pairs below are the ones the wrapper does *not* raise
    itself, so they are the ones that could have drifted.
    """
    del both_backends

    def messages() -> list[str]:
        tags = CellTags(num_cells=5)
        facets = FacetTags(num_cells=4, facets_per_cell=4)
        tags.set("m", [0, 1], [200, 1])
        facets.set("bc", [[0, 0]], [200])
        out: list[str] = []
        for call in (
            lambda: CellTags(num_cells=-1),
            lambda: tags.set("a", [5], 1),
            lambda: tags.set("a", [1, 1], 1),
            lambda: tags.set("a", [0, 1, 2], [1, 2]),
            lambda: tags.to_dense("m", dtype=np.int8),
            lambda: facets.set("a", [[4, 0]], 1),
            lambda: facets.set("a", [[0, 4]], 1),
            lambda: facets.set("a", [[0, 0], [0, 0]], 1),
            lambda: facets.to_dense("bc", dtype=np.int8),
        ):
            with pytest.raises((ValueError, TypeError, OverflowError)) as excinfo:
                call()
            out.append(f"{type(excinfo.value).__name__}: {excinfo.value}")
        return out

    with use_backend(Backend.PYTHON):
        from_python = messages()
    with use_backend(Backend.CPP):
        from_cpp = messages()

    assert from_cpp == from_python
