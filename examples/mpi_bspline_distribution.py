"""Evaluate a distributed tensor-product B-spline field at owned-cell midpoints.

Pipeline (every rank runs the same code, SPMD):

1. Build the *same* global space, control points and partition on every rank
   (all deterministic, so no communication is needed for this).
2. Wrap it in a :class:`~pantr.mpi.DistributedSpace`; each rank gets a local,
   self-contained window over the cells it owns plus their support halo.
3. Slice the global control points down to the local DOFs and build a *local*
   :class:`~pantr.bspline.Bspline` on the windowed space.
4. Iterate over the cells this rank owns and evaluate the local field at each
   cell's parametric midpoint.
5. Collectively check that every cell was evaluated exactly once and that the
   distributed values reproduce the serial (global) field.

Run with::

    mpiexec -n 4 python examples/mpi_bspline_distribution.py

Requires ``mpi4py`` and an MPI library (``pip install 'pantr[mpi]'``).
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from pantr.bspline import Bspline, create_uniform_space
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import DistributedSpace

VALUE_DIM = 2  # the field maps the 2-D parameter domain into R^2


def main() -> None:
    """Build a distributed B-spline field and evaluate it at owned-cell midpoints."""
    comm = MPI.COMM_WORLD

    # 1. Global space + control points + partition: identical on every rank.
    space = create_uniform_space([2, 2], [8, 8])  # biquadratic, 8x8 elements
    rng = np.random.default_rng(42)  # same seed -> same field on every rank
    control_points = rng.standard_normal((space.num_total_basis, VALUE_DIM))
    partition = partition_grid(tensor_product_grid(space), comm.size)

    # 2. Per-rank distributed handle (no communication).
    ds = DistributedSpace(space, partition, comm)

    # 3. Build this rank's local field from the local slice of the control points.
    #    local.space's basis equals the global basis pointwise over owned cells,
    #    so the local field agrees with the global one there.
    global_cells = np.empty(0, dtype=np.int64)
    values = np.empty((0, VALUE_DIM), dtype=np.float64)
    if ds.owns_cells:
        local = ds.local
        assert local is not None
        local_field = Bspline(local.space, control_points[local.local_to_global_dof])
        local_grid = tensor_product_grid(local.space)

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
        assert np.array_equal(all_cells[order], np.arange(space.num_total_intervals))

        grid = tensor_product_grid(space)
        serial_field = Bspline(space, control_points)
        lo, hi = grid.collect_cell_bounds()
        serial_values = serial_field.evaluate(0.5 * (lo + hi))
        assert np.allclose(all_values[order], serial_values)
        print(
            f"[rank 0] verified: {space.num_total_intervals} cell midpoints evaluated "
            f"exactly once across {ds.n_parts} ranks, matching the serial field",
            flush=True,
        )


if __name__ == "__main__":
    main()
