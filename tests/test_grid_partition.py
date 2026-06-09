"""Tests for pantr.grid.Partition."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.grid import Partition


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
