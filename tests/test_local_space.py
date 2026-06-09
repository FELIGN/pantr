"""Tests for pantr.bspline.compute_halo, dof_owner, build_local, and LocalSpace."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    LocalSpace,
    THBSplineSpace,
    build_local,
    compute_halo,
    dof_owner,
)
from pantr.bspline._local_space import _thb_dof_owner, _thb_halo
from pantr.grid import Partition, hierarchical_grid, uniform_grid


def _open_uniform_space(degrees: list[int], n_ints: list[int]) -> BsplineSpace:
    """Open-uniform tensor-product space: ``n_ints[d]`` unit intervals on each axis."""
    spaces: list[BsplineSpace1D] = []
    for p, n in zip(degrees, n_ints, strict=True):
        knots = [0.0] * (p + 1) + [float(i) for i in range(1, n)] + [float(n)] * (p + 1)
        spaces.append(BsplineSpace1D(knots, p))
    return BsplineSpace(spaces)


def _first_basis_per_axis(space: BsplineSpace) -> list[npt.NDArray[np.int64]]:
    """Per axis, the first non-zero basis index on each interval (via midpoints)."""
    fbs = []
    for sp in space.spaces:
        uk, _ = sp.get_unique_knots_and_multiplicity(in_domain=True)
        _, fb = sp.tabulate_basis(0.5 * (uk[:-1] + uk[1:]))
        fbs.append(np.asarray(fb, dtype=np.int64))
    return fbs


def _brute_closure(space: BsplineSpace, owned: list[int]) -> set[int]:
    """Cells sharing a B-spline function with any owned cell (function-index overlap).

    Independent cross-check of :func:`compute_halo`: cell ``c'`` is in the closure iff,
    for every axis, the function-index range ``[fb[c'], fb[c']+p]`` overlaps that of an
    owned cell.
    """
    fbs = _first_basis_per_axis(space)
    degs = space.degrees
    ni = space.num_intervals
    owned_multi = [tuple(int(i) for i in np.unravel_index(c, ni)) for c in owned]
    closure: set[int] = set()
    for cflat in range(space.num_total_intervals):
        cm = tuple(int(i) for i in np.unravel_index(cflat, ni))
        for om in owned_multi:
            if all(
                max(int(fbs[d][cm[d]]), int(fbs[d][om[d]]))
                <= min(int(fbs[d][cm[d]]), int(fbs[d][om[d]])) + degs[d]
                for d in range(space.dim)
            ):
                closure.add(cflat)
                break
    return closure


# ----------------------------------------------------------------- compute_halo


def test_halo_open_uniform_1d() -> None:
    space = _open_uniform_space([2], [7])
    # Degree-2 open-uniform: the support closure of cell 3 is cells [1, 5].
    np.testing.assert_array_equal(compute_halo(space, [3]), [1, 2, 4, 5])


def test_halo_matches_brute_force_1d() -> None:
    space = _open_uniform_space([3], [8])
    owned = [2, 3, 4]
    expected = sorted(_brute_closure(space, owned) - set(owned))
    np.testing.assert_array_equal(compute_halo(space, owned), expected)


def test_halo_matches_brute_force_2d_l_shape() -> None:
    space = _open_uniform_space([2, 2], [4, 5])
    owned = [1 * 5 + 1, 1 * 5 + 2, 2 * 5 + 1]  # non-convex L over num_intervals (4,5)
    expected = sorted(_brute_closure(space, owned) - set(owned))
    np.testing.assert_array_equal(compute_halo(space, owned), expected)


def test_halo_matches_brute_force_anisotropic() -> None:
    space = _open_uniform_space([1, 3], [5, 4])
    owned = [2 * 4 + 1, 2 * 4 + 2]
    expected = sorted(_brute_closure(space, owned) - set(owned))
    np.testing.assert_array_equal(compute_halo(space, owned), expected)


def test_halo_excludes_owned() -> None:
    space = _open_uniform_space([2, 2], [5, 5])
    owned = [12, 13]
    assert set(compute_halo(space, owned).tolist()).isdisjoint(owned)


def test_halo_is_readonly() -> None:
    halo = compute_halo(_open_uniform_space([2], [6]), [3])
    assert not halo.flags.writeable


def test_halo_full_owned_is_empty() -> None:
    space = _open_uniform_space([2], [5])
    assert compute_halo(space, list(range(space.num_total_intervals))).size == 0


def test_halo_periodic_rejected() -> None:
    from pantr.bspline import create_uniform_periodic_knots  # noqa: PLC0415

    per = BsplineSpace1D(create_uniform_periodic_knots(num_intervals=4, degree=2), 2, periodic=True)
    with pytest.raises(ValueError, match="periodic"):
        compute_halo(BsplineSpace([per]), [0])


def test_halo_out_of_range_raises() -> None:
    space = _open_uniform_space([2], [5])
    with pytest.raises(IndexError):
        compute_halo(space, [space.num_total_intervals])


def test_halo_out_of_range_negative() -> None:
    space = _open_uniform_space([2], [5])
    with pytest.raises(IndexError):
        compute_halo(space, [-1])


def test_halo_empty_owned() -> None:
    space = _open_uniform_space([1], [4])
    halo = compute_halo(space, [])
    assert halo.size == 0


def test_halo_non_integer_raises() -> None:
    space = _open_uniform_space([1], [4])
    with pytest.raises(TypeError, match="integer"):
        compute_halo(space, [0.0, 1.0])


# ------------------------------------------------------------------- dof_owner


def test_dof_owner_all_one_rank() -> None:
    space = _open_uniform_space([1], [4])  # num_basis 5
    part = Partition(np.zeros(space.num_total_intervals, dtype=np.int32), n_parts=1)
    owners = dof_owner(space, part)
    assert owners.shape == (space.num_total_basis,)
    np.testing.assert_array_equal(owners, np.zeros(5, dtype=np.int32))


def test_dof_owner_lex_first_active() -> None:
    space = _open_uniform_space([1], [4])  # degree 1: B_j supports cells [j-1, j]
    part = Partition(np.array([0, 0, 1, 1], dtype=np.int32), n_parts=2)
    # B_0{0}->0, B_1{0,1}->0, B_2{1,2}->cell1=0, B_3{2,3}->cell2=1, B_4{3}->1
    np.testing.assert_array_equal(dof_owner(space, part), [0, 0, 0, 1, 1])


def test_dof_owner_dead_dof() -> None:
    space = _open_uniform_space([1], [4])
    part = Partition(np.array([0, -1, -1, -1], dtype=np.int32), n_parts=1)
    # Only cell 0 active: DOFs whose support has no active cell are dead (-1).
    np.testing.assert_array_equal(dof_owner(space, part), [0, 0, -1, -1, -1])


def test_dof_owner_2d() -> None:
    space = _open_uniform_space([1, 1], [2, 2])  # num_basis (3,3) = 9, cells 4
    part = Partition(np.zeros(4, dtype=np.int32), n_parts=1)
    np.testing.assert_array_equal(dof_owner(space, part), np.zeros(9, dtype=np.int32))


def test_dof_owner_is_readonly() -> None:
    space = _open_uniform_space([1], [4])
    part = Partition(np.zeros(4, dtype=np.int32), n_parts=1)
    assert not dof_owner(space, part).flags.writeable


def test_dof_owner_cell_count_mismatch_raises() -> None:
    space = _open_uniform_space([2], [5])
    part = Partition(np.zeros(3, dtype=np.int32), n_parts=1)  # should be 5 cells
    with pytest.raises(ValueError, match="cells"):
        dof_owner(space, part)


def test_dof_owner_periodic_rejected() -> None:
    from pantr.bspline import create_uniform_periodic_knots  # noqa: PLC0415

    per = BsplineSpace1D(create_uniform_periodic_knots(num_intervals=4, degree=2), 2, periodic=True)
    part = Partition(np.zeros(4, dtype=np.int32), n_parts=1)
    with pytest.raises(ValueError, match="periodic"):
        dof_owner(BsplineSpace([per]), part)


def test_dof_owner_lex_first_active_degree2() -> None:
    # degree 2, 5 cells => 7 DOFs; partition splits at cell 2
    space = _open_uniform_space([2], [5])
    part = Partition(np.array([0, 0, 1, 1, 1], dtype=np.int32), n_parts=2)
    # B_0:[0]->0, B_1:[0,1]->0, B_2:[0,1,2]->0, B_3:[1,2,3]->cell1=rank0,
    # B_4:[2,3,4]->cell2=rank1, B_5:[3,4]->rank1, B_6:[4]->rank1
    np.testing.assert_array_equal(dof_owner(space, part), [0, 0, 0, 0, 1, 1, 1])


def test_dof_owner_2d_multirank() -> None:
    # degree (1,1), 2x2 cells => 3x3=9 DOFs; first two flat cells (row 0) -> rank 0, rest -> rank 1
    space = _open_uniform_space([1, 1], [2, 2])
    part = Partition(np.array([0, 0, 1, 1], dtype=np.int32), n_parts=2)
    # DOFs in rows 0-1 (i<2) have lex-first support cell in row 0 -> rank 0; row 2 -> rank 1
    np.testing.assert_array_equal(dof_owner(space, part), [0, 0, 0, 0, 0, 0, 1, 1, 1])


# ------------------------------------------------------------------ build_local


def _round_robin(space: BsplineSpace, n_parts: int) -> Partition:
    """Round-robin partition of all knot-span cells over ``n_parts`` ranks."""
    return Partition(
        np.arange(space.num_total_intervals, dtype=np.int32) % n_parts, n_parts=n_parts
    )


def _cell_midpoints(space: BsplineSpace, cells: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Midpoints of the given flat (C-order) knot-span cells of ``space``."""
    per_axis_uk = [sp.get_unique_knots_and_multiplicity(in_domain=True)[0] for sp in space.spaces]
    multi = np.unravel_index(cells, space.num_intervals)
    pts = np.empty((len(cells), space.dim), dtype=np.float64)
    for d in range(space.dim):
        uk = per_axis_uk[d]
        pts[:, d] = 0.5 * (uk[multi[d]] + uk[multi[d] + 1])
    return pts


def test_build_local_partitions_cells_and_dofs() -> None:
    # Over all ranks, build_local's owned cells/DOFs must exactly partition the active
    # cells / non-dead DOFs (complete and disjoint).
    space = _open_uniform_space([2, 2], [5, 4])
    part = _round_robin(space, 3)
    global_owner = dof_owner(space, part)

    owned_cells: list[int] = []
    owned_dofs: list[int] = []
    for rank in range(part.n_parts):
        loc = build_local(space, part, rank)
        assert isinstance(loc, LocalSpace)
        oc = loc.local_to_global_cell[loc.owned_cell_mask]
        od = loc.local_to_global_dof[loc.owned_dof_mask]
        np.testing.assert_array_equal(part.cell_owner[oc], rank)
        np.testing.assert_array_equal(global_owner[od], rank)
        owned_cells.extend(int(c) for c in oc)
        owned_dofs.extend(int(d) for d in od)

    assert sorted(owned_cells) == list(range(space.num_total_intervals))
    assert len(owned_cells) == len(set(owned_cells))
    assert sorted(owned_dofs) == sorted(int(d) for d in np.flatnonzero(global_owner >= 0))
    assert len(owned_dofs) == len(set(owned_dofs))


def test_build_local_reproduces_global_basis_on_owned_cells() -> None:
    space = _open_uniform_space([2, 2], [5, 5])
    part = _round_robin(space, 4)
    g_nb = space.num_basis
    degs = space.degrees
    for rank in range(part.n_parts):
        loc = build_local(space, part, rank)
        sub = loc.space
        assert isinstance(sub, BsplineSpace)
        owned_local = np.flatnonzero(loc.owned_cell_mask)
        pts = _cell_midpoints(sub, owned_local)
        gb, gfb = space.tabulate_basis(pts)
        wb, wfb = sub.tabulate_basis(pts)
        np.testing.assert_allclose(wb, gb, atol=1e-12)  # windowed basis == global on owned cells
        w_nb = sub.num_basis
        l2g = loc.local_to_global_dof
        for i in range(pts.shape[0]):
            w_axis = [wfb[i, d] + np.arange(degs[d] + 1) for d in range(space.dim)]
            g_axis = [gfb[i, d] + np.arange(degs[d] + 1) for d in range(space.dim)]
            w_flat = np.ravel_multi_index(
                [m.ravel() for m in np.meshgrid(*w_axis, indexing="ij")], w_nb
            )
            g_flat = np.ravel_multi_index(
                [m.ravel() for m in np.meshgrid(*g_axis, indexing="ij")], g_nb
            )
            np.testing.assert_array_equal(l2g[w_flat], g_flat)  # DOF map lines up


def test_build_local_shapes_and_readonly() -> None:
    space = _open_uniform_space([2, 2], [4, 4])
    loc = build_local(space, _round_robin(space, 2), 0)
    sub = loc.space
    assert isinstance(sub, BsplineSpace)
    n_cells = sub.num_total_intervals
    n_dofs = sub.num_total_basis
    assert loc.local_to_global_cell.shape == (n_cells,)
    assert loc.owned_cell_mask.shape == (n_cells,)
    assert loc.local_to_global_dof.shape == (n_dofs,)
    assert loc.owned_dof_mask.shape == (n_dofs,)
    assert loc.n_global_cells == space.num_total_intervals
    assert loc.n_global_dofs == space.num_total_basis
    for arr in (
        loc.local_to_global_cell,
        loc.local_to_global_dof,
        loc.owned_cell_mask,
        loc.owned_dof_mask,
    ):
        assert not arr.flags.writeable


def test_build_local_includes_halo_cells() -> None:
    space = _open_uniform_space([2], [8])
    owner = np.ones(8, dtype=np.int32)
    owner[[3, 4]] = 0  # rank 0 owns an interior sub-block
    loc = build_local(space, Partition(owner, n_parts=2), 0)
    assert isinstance(loc.space, BsplineSpace)
    assert int(loc.owned_cell_mask.sum()) == 2  # only cells 3, 4 owned
    assert loc.space.num_total_intervals > 2  # halo / bbox-fill cells present
    np.testing.assert_array_equal(
        sorted(loc.local_to_global_cell[loc.owned_cell_mask].tolist()), [3, 4]
    )


def test_build_local_periodic_rejected() -> None:
    from pantr.bspline import create_uniform_periodic_knots  # noqa: PLC0415

    per = BsplineSpace1D(create_uniform_periodic_knots(num_intervals=4, degree=2), 2, periodic=True)
    part = Partition(np.zeros(4, dtype=np.int32), n_parts=1)
    with pytest.raises(ValueError, match="periodic"):
        build_local(BsplineSpace([per]), part, 0)


def test_build_local_bad_rank_raises() -> None:
    space = _open_uniform_space([1], [4])
    part = Partition(np.zeros(4, dtype=np.int32), n_parts=2)
    with pytest.raises(ValueError, match="rank"):
        build_local(space, part, 2)
    with pytest.raises(ValueError, match="rank"):
        build_local(space, part, -1)


def test_build_local_cell_count_mismatch_raises() -> None:
    space = _open_uniform_space([2], [5])
    part = Partition(np.zeros(3, dtype=np.int32), n_parts=1)
    with pytest.raises(ValueError, match="cells"):
        build_local(space, part, 0)


def test_build_local_rank_owns_nothing_raises() -> None:
    space = _open_uniform_space([1], [4])
    part = Partition(np.zeros(4, dtype=np.int32), n_parts=2)  # rank 1 owns nothing
    with pytest.raises(ValueError, match="owns no cells"):
        build_local(space, part, 1)


def test_build_local_with_inactive_cells() -> None:
    # Cells 1 and 3 are inactive (-1); rank 0 owns cells 0 and 2.
    # Dead DOFs (no active cell in support) must appear with owned_dof_mask == False.
    space = _open_uniform_space([1], [4])  # degree 1, 4 cells, 5 DOFs
    owner = np.array([0, -1, 0, -1], dtype=np.int32)
    part = Partition(owner, n_parts=1)
    loc = build_local(space, part, 0)
    global_owners = dof_owner(space, part)
    expected_owned = global_owners[loc.local_to_global_dof] == 0
    np.testing.assert_array_equal(loc.owned_dof_mask, expected_owned)
    dead_local = np.flatnonzero(global_owners[loc.local_to_global_dof] == -1)
    assert not loc.owned_dof_mask[dead_local].any()


# -------------------------------------------------------- build_local (THB dispatch)


def _refined_thb(
    n: int = 6, lo: tuple[int, int] = (2, 2), hi: tuple[int, int] = (4, 4)
) -> THBSplineSpace:
    """Degree-2 ``n x n`` THB space on [0, n]^2, refining ``[lo, hi)`` to level 1."""
    knots = [0.0] * 3 + [float(i) for i in range(1, n)] + [float(n)] * 3
    sp = BsplineSpace1D(knots, 2)
    grid = hierarchical_grid(uniform_grid([[0.0, float(n)], [0.0, float(n)]], n), 2)
    grid.refine(0, list(lo), list(hi))
    return THBSplineSpace(BsplineSpace([sp, sp]), grid)


def test_build_local_thb_partitions_cells_and_dofs() -> None:
    g = _refined_thb()
    part = Partition(np.arange(g.grid.num_cells, dtype=np.int32) % 3, n_parts=3)
    global_owner = _thb_dof_owner(g, part)
    owned_cells: list[int] = []
    owned_dofs: list[int] = []
    for rank in range(part.n_parts):
        loc = build_local(g, part, rank)
        assert isinstance(loc, LocalSpace)
        assert isinstance(loc.space, THBSplineSpace)
        oc = loc.local_to_global_cell[loc.owned_cell_mask]
        od = loc.local_to_global_dof[loc.owned_dof_mask]
        np.testing.assert_array_equal(part.cell_owner[oc], rank)
        np.testing.assert_array_equal(global_owner[od], rank)
        owned_cells.extend(int(c) for c in oc)
        owned_dofs.extend(int(d) for d in od)
    assert sorted(owned_cells) == list(range(g.grid.num_cells))
    assert len(owned_cells) == len(set(owned_cells))
    assert sorted(owned_dofs) == sorted(int(d) for d in np.flatnonzero(global_owner >= 0))
    assert len(owned_dofs) == len(set(owned_dofs))


def test_build_local_thb_basis_matches_on_owned_cells() -> None:
    g = _refined_thb()
    part = Partition(np.arange(g.grid.num_cells, dtype=np.int32) % 4, n_parts=4)
    for rank in range(part.n_parts):
        loc = build_local(g, part, rank)
        sub = loc.space
        assert isinstance(sub, THBSplineSpace)
        for lcid in np.flatnonzero(loc.owned_cell_mask):
            gcid = int(loc.local_to_global_cell[lcid])
            lo, hi = sub.grid.cell_bounds(int(lcid))
            pt = (0.5 * (lo + hi)).reshape(1, -1)
            vs, ds = sub.tabulate_basis(int(lcid), pt)
            vg, dg = g.tabulate_basis(gcid, pt)
            mapped = loc.local_to_global_dof[ds]
            assert np.all(mapped >= 0)  # owned cells are interior
            gval = {int(d): float(v) for d, v in zip(dg, vg[0], strict=True)}
            for i, gd in enumerate(mapped):
                np.testing.assert_allclose(vs[0, i], gval[int(gd)], atol=1e-11)


def test_build_local_thb_includes_halo() -> None:
    g = _refined_thb()
    owner = np.ones(g.grid.num_cells, dtype=np.int32)
    central = [c for c in range(g.grid.num_cells) if g.grid.cell_level(c) == 1][:3]
    owner[central] = 0
    loc = build_local(g, Partition(owner, n_parts=2), 0)
    assert isinstance(loc.space, THBSplineSpace)
    assert int(loc.owned_cell_mask.sum()) == len(central)
    assert loc.space.grid.num_cells > len(central)  # halo / bbox-fill cells present
    # Owned global cell ids must exactly match what we assigned to rank 0
    np.testing.assert_array_equal(
        sorted(loc.local_to_global_cell[loc.owned_cell_mask].tolist()), sorted(central)
    )


def test_build_local_thb_returns_thb_localspace() -> None:
    g = _refined_thb()
    loc = build_local(g, Partition(np.zeros(g.grid.num_cells, dtype=np.int32), n_parts=1), 0)
    assert isinstance(loc, LocalSpace)
    assert isinstance(loc.space, THBSplineSpace)
    assert loc.n_global_cells == g.grid.num_cells
    assert loc.n_global_dofs == g.num_total_basis
    for arr in (
        loc.local_to_global_cell,
        loc.local_to_global_dof,
        loc.owned_cell_mask,
        loc.owned_dof_mask,
    ):
        assert not arr.flags.writeable


def test_build_local_thb_rank_owns_nothing_raises() -> None:
    g = _refined_thb()
    part = Partition(np.zeros(g.grid.num_cells, dtype=np.int32), n_parts=2)  # rank 1 owns nothing
    with pytest.raises(ValueError, match="owns no cells"):
        build_local(g, part, 1)


def test_build_local_thb_cell_count_mismatch_raises() -> None:
    g = _refined_thb()
    part = Partition(np.zeros(g.grid.num_cells - 1, dtype=np.int32), n_parts=1)
    with pytest.raises(ValueError, match="cells"):
        build_local(g, part, 0)


def test_build_local_thb_bad_rank_raises() -> None:
    g = _refined_thb()
    part = Partition(np.zeros(g.grid.num_cells, dtype=np.int32), n_parts=2)
    with pytest.raises(ValueError, match="rank"):
        build_local(g, part, 2)


def _corner_refined_thb() -> THBSplineSpace:
    """Degree-2 6x6 THB space with [0,3)x[0,3) refined to level 1.

    This produces globally-inactive level-0 functions (fully superseded by level-1
    functions), so a window covering only a few level-1 cells yields ``local_to_global_dof
    == -1`` entries — exercising the boundary-DOF path in ``_build_local_thb``.
    """
    knots = [0.0] * 3 + [float(i) for i in range(1, 6)] + [6.0] * 3
    sp = BsplineSpace1D(knots, 2)
    grid = hierarchical_grid(uniform_grid([[0.0, 6.0], [0.0, 6.0]], 6), 2)
    grid.refine(0, [0, 0], [3, 3])
    return THBSplineSpace(BsplineSpace([sp, sp]), grid)


def test_build_local_thb_boundary_dofs_not_owned() -> None:
    # Local DOFs mapping to -1 (window-boundary THB functions) must never be marked owned.
    g = _corner_refined_thb()
    l1 = [c for c in range(g.grid.num_cells) if g.grid.cell_level(c) == 1]
    owner = np.ones(g.grid.num_cells, dtype=np.int32)
    for c in l1[:4]:
        owner[c] = 0  # rank 0 owns just 4 level-1 cells — ensures boundary DOFs exist
    loc = build_local(g, Partition(owner, n_parts=2), 0)
    boundary = loc.local_to_global_dof < 0
    assert boundary.any(), "fixture produced no boundary DOFs; test is vacuous"
    assert not loc.owned_dof_mask[boundary].any()


def test_build_local_thb_full_owned_no_halo() -> None:
    # All cells owned by one rank: halo is empty, owned_cell_mask is all True.
    g = _refined_thb()
    loc = build_local(g, Partition(np.zeros(g.grid.num_cells, dtype=np.int32), n_parts=1), 0)
    assert isinstance(loc.space, THBSplineSpace)
    assert loc.owned_cell_mask.all()
    assert loc.space.grid.num_cells == g.grid.num_cells
    # _thb_halo with all cells owned should yield empty halo
    all_cells = np.arange(g.grid.num_cells, dtype=np.int64)
    assert _thb_halo(g, all_cells).size == 0


def test_thb_dof_owner_all_one_rank() -> None:
    # 1-rank partition: every non-dead DOF must be owned by rank 0.
    g = _refined_thb()
    part = Partition(np.zeros(g.grid.num_cells, dtype=np.int32), n_parts=1)
    owners = _thb_dof_owner(g, part)
    assert owners.shape == (g.num_total_basis,)
    assert not owners.flags.writeable
    # With all cells active and owned by rank 0, no DOF should be dead.
    np.testing.assert_array_equal(owners, np.zeros(g.num_total_basis, dtype=np.int32))


def test_thb_dof_owner_cell_count_mismatch_raises() -> None:
    g = _refined_thb()
    part = Partition(np.zeros(g.grid.num_cells - 1, dtype=np.int32), n_parts=1)
    with pytest.raises(ValueError, match="cells"):
        _thb_dof_owner(g, part)
