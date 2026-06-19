"""Evaluate a distributed THB-spline field at owned-cell midpoints.

This mirrors ``mpi_bspline_distribution.py`` step for step; the only differences
are that the global space is a :class:`~pantr.bspline.THBSplineSpace`, the field
is a :class:`~pantr.bspline.THBSpline`, and the partition is built on the
hierarchical grid (``thb.grid``). ``DistributedSpace`` consumes both space types
identically.

The space is built with the ergonomic :func:`~pantr.bspline.create_thb_space`
factory and refined with chained, immutable
:meth:`~pantr.bspline.THBSplineSpace.refine_region` calls -- the THB counterpart
of the tensor-product ``create_uniform_space`` flow.

Pipeline (every rank runs the same code, SPMD):

1. Build the *same* global THB space, control points and partition on every rank.
2. Wrap it in a :class:`~pantr.mpi.DistributedSpace`; each rank gets a local
   window over the cells it owns plus their support halo.
3. Slice the global control points to the local DOFs and build a *local* field.
4. Iterate over the owned cells and evaluate the field at each cell's midpoint.
5. Collectively check that every cell was evaluated once and that the
   distributed values reproduce the serial (global) field.

Run with::

    mpiexec -n 3 python examples/mpi_thb_distribution.py

Requires ``mpi4py`` and an MPI library (``pip install 'pantr[mpi]'``).
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from pantr.bspline import THBSpline, THBSplineSpace, create_thb_space, create_uniform_space
from pantr.grid import partition_grid
from pantr.mpi import DistributedSpace

VALUE_DIM = 2  # the field maps the 2-D parameter domain into R^2


def build_thb_space() -> THBSplineSpace:
    """Build a biquadratic THB space refined twice toward the lower-left corner."""
    root = create_uniform_space([2, 2], [8, 8])
    thb = create_thb_space(root, factor=2)  # trivial, single-level THB space
    thb = thb.refine_region(0, [0, 0], [4, 4])  # level 0 -> 1 on the lower-left quarter
    thb = thb.refine_region(1, [0, 0], [4, 4])  # level 1 -> 2 on the lower-left of that
    return thb


def main() -> None:
    """Build a distributed THB field and evaluate it at owned-cell midpoints."""
    comm = MPI.COMM_WORLD

    # 1. Global space + control points + partition: identical on every rank.
    thb = build_thb_space()
    rng = np.random.default_rng(42)  # same seed -> same field on every rank
    control_points = rng.standard_normal((thb.num_total_basis, VALUE_DIM))
    partition = partition_grid(thb.grid, comm.size)

    # 2. Per-rank distributed handle (no communication).
    ds = DistributedSpace(thb, partition, comm)

    # 3. Build this rank's local field from the local slice of the control points.
    global_cells = np.empty(0, dtype=np.int64)
    values = np.empty((0, VALUE_DIM), dtype=np.float64)
    if ds.owns_cells:
        local = ds.local
        assert local is not None
        local_field = THBSpline(local.space, control_points[local.local_to_global_dof])
        local_grid = local.space.grid

        # 4. Iterate over owned cells; evaluate the field at each cell's midpoint.
        owned_local_cells = np.flatnonzero(local.owned_cell_mask)
        global_cells = local.local_to_global_cell[owned_local_cells]
        values = np.empty((owned_local_cells.size, VALUE_DIM), dtype=np.float64)
        for i, local_cell in enumerate(owned_local_cells):
            lo, hi = local_grid.cell_bounds(int(local_cell))
            midpoint = 0.5 * (lo + hi)
            values[i] = local_field.evaluate(midpoint[None]).reshape(-1)

        print(
            f"[rank {ds.rank}/{ds.n_parts}] evaluated {global_cells.size} owned cells; "
            f"e.g. cell {int(global_cells[0])} midpoint -> {np.round(values[0], 4)}",
            flush=True,
        )
    else:
        print(f"[rank {ds.rank}/{ds.n_parts}] owns no cells", flush=True)

    # 5. Collective check: cells partition the globals and values match the serial field.
    all_cells = np.concatenate(comm.allgather(global_cells))
    all_values = np.concatenate(comm.allgather(values))
    if ds.rank == 0:
        order = np.argsort(all_cells)
        assert np.array_equal(all_cells[order], np.arange(thb.grid.num_cells))

        serial_field = THBSpline(thb, control_points)
        lo, hi = thb.grid.collect_cell_bounds()
        serial_values = serial_field.evaluate(0.5 * (lo + hi))
        assert np.allclose(all_values[order], serial_values)
        print(
            f"[rank 0] verified: {thb.grid.num_cells} cell midpoints evaluated "
            f"exactly once across {ds.n_parts} ranks, matching the serial field",
            flush=True,
        )


if __name__ == "__main__":
    main()
