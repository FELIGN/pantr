# Distributed spaces

PaNTr can distribute a tensor-product B-spline space (`BsplineSpace`) or a
hierarchical THB-spline space (`THBSplineSpace`) across MPI ranks for parallel
assembly. The design keeps a clear separation:

- a **serial windowing core** (in `pantr.grid` / `pantr.bspline`) that needs no MPI;
- an **optional MPI layer** (`pantr.mpi`) that wraps it per rank.

Every rank holds a **redundant, self-contained** local space covering the cells it
owns plus their support halo, so basis evaluation and element assembly are purely
local. Cross-rank coupling (e.g. a global linear solve) is the consumer's job
(typically a PETSc `PtAP`); PaNTr performs **no MPI DOF exchange**.

## The pipeline

Everything flows through one small descriptor, the `Partition` (which rank owns
each cell), regardless of where it came from:

```text
            partition_grid  ─┐
   coupling_graph + partition_graph  ─┤
            from_dolfinx     ─┘
                                  │
                                  ▼
                              Partition ──►  DistributedSpace(space, partition, comm)
                                                          │
                                                          ▼
                                                  LocalSpace  (this rank)
```

A `Partition` is just an integer owner per cell (`-1` for an inactive/excluded
cell). Any of the three *sources* below produces one; `DistributedSpace` consumes
it identically.

## Installation

```bash
pip install pantr                  # serial-only, no mpi4py dependency
pip install 'pantr[mpi]'           # + mpi4py, for distributed spaces (needs an MPI library)
pip install 'pantr[metis]'         # + pymetis, for partition_graph(backend="metis")
```

`import pantr.mpi` succeeds even without `mpi4py`; only code paths that genuinely
need MPI (e.g. `pantr.mpi.require_mpi()`) raise when it is absent. The
partition/windowing layer (`partition_grid`, `partition_graph`, `build_local`) is
pure serial and never imports MPI.

## Partitioning a grid

`pantr.grid.partition_grid(grid, n_parts, *, backend="auto", cell_weights=None,
cell_active=None)` splits a grid's cells into `n_parts` rank subdomains:

```python
from mpi4py import MPI
from pantr.bspline import create_uniform_space
from pantr.grid import partition_grid, tensor_product_grid

comm = MPI.COMM_WORLD
space = create_uniform_space([2, 2], [64, 64])      # identical on every rank
grid = tensor_product_grid(space)                   # the knot-span grid
partition = partition_grid(grid, comm.size)         # deterministic -> same on all ranks
```

| `backend` | dependency | balances | shape | good for |
|---|---|---|---|---|
| `"block"` | none | cell count | axis-aligned boxes | uniform tensor-product grids |
| `"rcb"` | none | total weight | boxes | THB / immersed; arbitrary / prime `n_parts` |
| `"auto"` (default) | none | — | — | `block` when tensor-product, unweighted, all-active and `n_parts` factors onto the axes; otherwise `rcb` |

The partitioner is **deterministic**, so every rank computes the *same* partition
with no communication.

## Coupling-graph partitioning

For irregular or hierarchical meshes, a geometric split can cut more shared
functions than necessary. Build the **cell-coupling graph** (cells that share a
basis function) and partition *that* to minimize cross-rank DOF coupling:

```python
from pantr.bspline import coupling_graph, partition_graph

graph = coupling_graph(space)                       # METIS/Scotch CSR; TP and THB
partition = partition_graph(graph, comm.size)       # backend="spectral" (default)
```

| `backend` | dependency | notes |
|---|---|---|
| `"spectral"` (default) | scipy only | recursive Fiedler bisection; never leaves a rank empty |
| `"metis"` | `pymetis` (`pip install 'pantr[metis]'`) | higher-quality k-way min-cut; raises a clear error if `pymetis` is absent |

## Consuming an external partition (dolfinx)

When a dolfinx-based consumer (e.g. qugar or tigarx) already partitioned a mesh,
ingest *its* cell ownership instead of re-partitioning:

```python
from pantr.mpi import from_dolfinx

n_cells = space.num_total_intervals     # TP;  THB: space.grid.num_cells
partition = from_dolfinx(mesh, n_cells)  # MPI-allgathers over mesh.comm
```

The correspondence between a dolfinx cell and a PaNTr cell is the dolfinx
*original* cell index; pass `dolfinx_to_pantr=<array>` if the mesh was not built in
PaNTr's C-order. Cells absent from the mesh (e.g. exterior cells an immersed code
trimmed away) get owner `-1`.

## Building the distributed space

`pantr.mpi.DistributedSpace(global_space, partition, comm)` is the per-rank,
SPMD handle. Every rank constructs its own from the *same* global space and
partition; construction performs **no communication** (it windows the global space
locally via `build_local`):

```python
from pantr.mpi import DistributedSpace

ds = DistributedSpace(space, partition, comm)

ds.rank          # this rank's id
ds.n_parts       # number of ranks (== comm.size == partition.n_parts)
ds.owned_cells   # global ids of the cells this rank owns
local = ds.local # this rank's LocalSpace, or None if it owns no cells
```

The `LocalSpace` (`local`) bundles a real, windowed `BsplineSpace` /
`THBSplineSpace` (`local.space`) with the maps relating it to the global space --
`local_to_global_cell`, `local_to_global_dof`, `owned_cell_mask`, `owned_dof_mask`.
Its basis equals the global basis pointwise over the rank's owned cells, so
per-element assembly on `local.space` is exact.

```{note}
A rank that owns no cells (an over-provisioned run, or a `from_dolfinx` partition
that leaves a rank empty) gets `ds.local is None` and `ds.owns_cells is False`
rather than failing -- guard with `if ds.owns_cells:` before assembling.
```

## Immersion hooks

PaNTr stores **no** geometric classification (interior / cut / exterior). An
immersed consumer expresses its classification through two transient hooks, and
PaNTr never interprets *why* a cell is weighted or inactive:

- `cell_weights` -- per-cell assembly cost; `rcb` and the graph backends balance
  total weight rather than cell count (cut cells cost more).
- `cell_active` -- a boolean mask; inactive cells get owner `-1` and drop out of
  the partition (and the coupling graph).

```python
partition = partition_grid(grid, comm.size, cell_weights=cost, cell_active=interior)
```

## Consumer patterns

**Native MPI (e.g. ocelat), no dolfinx.** Partition the grid (or coupling graph)
directly and build the distributed space:

```python
from mpi4py import MPI
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import DistributedSpace

comm = MPI.COMM_WORLD
partition = partition_grid(tensor_product_grid(space), comm.size)
ds = DistributedSpace(space, partition, comm)
if ds.owns_cells:
    assemble_local(ds.local)        # your element loop over ds.local.space
```

**dolfinx-driven (e.g. tigarx, qugar).** Let dolfinx own the partition and bridge
it in -- the immersed `cell_active` path falls out for free (trimmed cells become
owner `-1`):

```python
from pantr.mpi import from_dolfinx, DistributedSpace

n_cells = space.num_total_intervals      # THB: space.grid.num_cells
partition = from_dolfinx(mesh, n_cells)  # absent cells -> -1
ds = DistributedSpace(space, partition, mesh.comm)
```

## Testing distributed code

The distributed objects are duck-typed on the communicator (only `rank` / `size` /
`allgather` are read), so most behavior is unit-testable serially with a fake
comm. Genuine multi-rank tests live under `tests/mpi/`, are skipped unless
`PANTR_RUN_MPI` is set, and run under a launcher:

```bash
PANTR_RUN_MPI=1 mpiexec -n 3 python -m pytest tests/mpi/
```

See the [API reference](api/reference.md) for full signatures.
