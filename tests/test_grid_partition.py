"""Tests for pantr.grid.Partition."""

from __future__ import annotations

import copy
import inspect
import pickle

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.grid import Partition
from tests._parity_harness import demand_cpp_backend


def test_construction_and_n_cells() -> None:
    p = Partition(np.array([0, 1, 0, -1, 1], dtype=np.int32), n_parts=2)
    assert p.n_cells == 5
    assert p.n_parts == 2


def test_cell_owner_is_readonly_int32() -> None:
    p = Partition([0, 1, -1, 1], n_parts=2)
    assert p.cell_owner.dtype == np.int32
    assert not p.cell_owner.flags.writeable


def test_active_mask() -> None:
    p = Partition([0, -1, 1, -1, 0], n_parts=2)
    np.testing.assert_array_equal(p.active_mask, [True, False, True, False, True])


def test_owned_cells() -> None:
    p = Partition([0, 1, 0, -1, 1, 0], n_parts=2)
    np.testing.assert_array_equal(p.owned_cells(0), [0, 2, 5])
    np.testing.assert_array_equal(p.owned_cells(1), [1, 4])


def test_owned_cells_empty_for_unused_rank() -> None:
    p = Partition([0, 0, -1], n_parts=3)
    assert p.owned_cells(2).size == 0


def test_owned_cells_bad_rank_raises() -> None:
    p = Partition([0, 1], n_parts=2)
    with pytest.raises(ValueError, match="rank"):
        p.owned_cells(2)
    with pytest.raises(ValueError, match="rank"):
        p.owned_cells(-1)


def test_accepts_list_input() -> None:
    p = Partition([0, 1, -1], n_parts=2)
    assert p.n_cells == 3
    assert p.cell_owner.dtype == np.int32


def test_invalid_n_parts_raises() -> None:
    with pytest.raises(ValueError, match="n_parts"):
        Partition([0, 0], n_parts=0)


def test_non_1d_raises() -> None:
    with pytest.raises(ValueError, match="1D integer"):
        Partition(np.zeros((2, 2), dtype=np.int32), n_parts=1)


def test_non_integer_raises() -> None:
    with pytest.raises(ValueError, match="1D integer"):
        Partition(np.array([0.0, 1.0]), n_parts=2)


def test_owner_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 2\)"):
        Partition([0, 2], n_parts=2)  # 2 == n_parts
    with pytest.raises(ValueError, match=r"\[-1, 2\)"):
        Partition([-2, 0], n_parts=2)  # -2 < -1


def test_frozen() -> None:
    p = Partition([0, 1], n_parts=2)
    with pytest.raises(AttributeError):
        p.n_parts = 3  # type: ignore[misc]


def test_empty_partition() -> None:
    p = Partition(np.array([], dtype=np.int32), n_parts=1)
    assert p.n_cells == 0
    assert p.owned_cells(0).size == 0


def test_owned_cells_dtype() -> None:
    p = Partition([0, 1, 0, 1], n_parts=2)
    assert p.owned_cells(0).dtype == np.int64


# ---------------------------------------------------------------------------
# The port to C++ ownership (FELIGN/pantr#381)
# ---------------------------------------------------------------------------
#
# Everything above ran against the pure-Python partition and now runs against
# whichever implementation ``PANTR_BACKEND`` selects, unchanged. What follows is
# what the port added: the pickle that has to cross the backends, the repr that has
# to be byte-identical across them, and the messages a caller catches.


def _backend_pairs() -> list[tuple[Backend, Backend]]:
    """Every ordered pair of backends, for the serialization round trips.

    Returns:
        list[tuple[Backend, Backend]]: The four ``(writer, reader)`` pairs.
    """
    return [(writer, reader) for writer in Backend for reader in Backend]


@pytest.fixture
def both_backends() -> None:
    """Require the compiled extension for a test that uses both backends at once.

    Routed through the parity harness rather than a bare ``skipif``: a bare skip is
    silent, and a suite that skips its way to green has let real failures through in
    this repository.
    """
    demand_cpp_backend()


def test_cell_owner_is_a_read_only_view_that_owns_its_storage(both_backends: None) -> None:
    """The owner array is read-only under both backends and aliases under C++.

    The second half is what catches the ownerless binding: a return with no owner
    silently *copies* and comes back writable, which would pass every value
    assertion and drop the read-only contract. ``base is not None`` is asserted only
    for the C++ backend, because the Python one hands back the array it stores and
    ``numpy.ascontiguousarray`` returns that array itself, base and all.
    """
    del both_backends
    with use_backend(Backend.PYTHON):
        py = Partition([0, 1, -1], 2)
    with use_backend(Backend.CPP):
        cpp = Partition([0, 1, -1], 2)

    for part in (py, cpp):
        assert not part.cell_owner.flags.writeable
        with pytest.raises(ValueError, match=r"read-only|assignment"):
            part.cell_owner[0] = 1

    assert cpp.cell_owner.base is not None, (
        "the C++ view must carry an owner: without one nanobind copies, and the copy "
        "comes back writable"
    )


@pytest.mark.parametrize(("writer", "reader"), _backend_pairs())
def test_pickle_round_trips_across_every_backend_pair(
    both_backends: None, writer: Backend, reader: Backend
) -> None:
    """A pickled partition written under one backend loads under the other.

    Asserted on the reconstructed field values rather than on the absence of an
    exception. This one is load-bearing beyond the backend switch: ``pantr.mpi``
    moves a partition between ranks through ``mpi4py``, which pickles.
    """
    del both_backends
    with use_backend(writer):
        original = Partition([0, 1, -1, 1, 0], 2)
        blob = pickle.dumps(original)

    with use_backend(reader):
        loaded = pickle.loads(blob)
        assert loaded.n_parts == 2
        assert loaded.n_cells == 5
        assert loaded.cell_owner.dtype == np.int32
        np.testing.assert_array_equal(loaded.cell_owner, [0, 1, -1, 1, 0])
        np.testing.assert_array_equal(loaded.owned_cells(0), [0, 4])
        np.testing.assert_array_equal(loaded.active_mask, [True, True, False, True, True])


@pytest.mark.parametrize(("writer", "reader"), _backend_pairs())
def test_deepcopy_round_trips_across_every_backend_pair(
    both_backends: None, writer: Backend, reader: Backend
) -> None:
    """``deepcopy`` goes through the same reduction, so it crosses backends too."""
    del both_backends
    with use_backend(writer):
        original = Partition([1, -1, 0], 2)

    with use_backend(reader):
        clone = copy.deepcopy(original)

    assert clone.n_parts == 2
    np.testing.assert_array_equal(clone.cell_owner, [1, -1, 0])
    np.testing.assert_array_equal(clone.owned_cells(1), [0])


def test_repr_is_byte_identical_under_both_backends(both_backends: None) -> None:
    """``repr`` is computed by the wrapper, so ``PANTR_BACKEND`` cannot move it."""
    del both_backends
    with use_backend(Backend.PYTHON):
        from_python = repr(Partition([0, 1, -1], 2))
    with use_backend(Backend.CPP):
        from_cpp = repr(Partition([0, 1, -1], 2))

    assert from_python == from_cpp
    assert from_python == "Partition(n_cells=3, n_parts=2)"


def test_both_backends_agree_on_the_messages_a_caller_reads(both_backends: None) -> None:
    """Every rejection carries the same class and the same text under either backend.

    A port that changed what a caller reads or catches would pass every value
    assertion above. Each case below violates exactly one contract, which is what
    makes the comparison a statement about the message rather than about the order
    two checks happen to run in.
    """
    del both_backends

    def messages() -> list[str]:
        part = Partition([0, 1], 2)
        out: list[str] = []
        for call in (
            lambda: Partition([0, 0], 0),
            lambda: Partition(np.zeros((2, 2), dtype=np.int32), 1),
            lambda: Partition(np.array([0.0, 1.0]), 2),
            lambda: Partition([0, 2], 2),
            lambda: Partition([-2, 0], 2),
            lambda: part.owned_cells(2),
            lambda: part.owned_cells(-1),
        ):
            with pytest.raises(ValueError) as excinfo:
                call()
            out.append(f"{type(excinfo.value).__name__}: {excinfo.value}")
        return out

    with use_backend(Backend.PYTHON):
        from_python = messages()
    with use_backend(Backend.CPP):
        from_cpp = messages()

    assert from_cpp == from_python


def test_two_backends_in_one_process_cannot_meet_in_an_operation(both_backends: None) -> None:
    """Two partitions built under different backends hold unrelated implementations.

    ``design/cross_backend_types.md`` forbids reconciling two implementations of one
    type by converting between them, and :class:`pantr.geometry.AABB` enforces that
    with a ``TypeError`` from ``_peer`` on every binary operation. **A partition has
    no binary operation**, so there is no site at which two of them could meet and
    nothing for such a guard to sit on -- and adding one would be dead code. This
    test pins both halves of that: the per-process selection really does produce two
    unrelated implementations in one process, and the public surface really does
    take no second partition.
    """
    del both_backends
    with use_backend(Backend.PYTHON):
        py = Partition([0, 1], 2)
    with use_backend(Backend.CPP):
        cpp = Partition([0, 1], 2)

    assert type(py) is type(cpp) is Partition
    assert type(py._impl) is not type(cpp._impl)

    takes_a_partition = [
        name
        for name, member in inspect.getmembers(Partition, callable)
        if not name.startswith("__")
        and any(
            parameter.annotation in (Partition, "Partition")
            for parameter in inspect.signature(member).parameters.values()
        )
    ]
    assert takes_a_partition == [], (
        f"{takes_a_partition} take a second Partition, so two backends CAN now meet in "
        f"one operation and each needs the refusal AABB._peer performs"
    )
