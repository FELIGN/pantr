# THB-Splines

**Truncated hierarchical B-splines (THB-splines)** extend the tensor-product
B-spline framework with support for *local refinement*: individual elements can
be split without globally adding degrees of freedom.  The resulting basis retains
the partition-of-unity property, linear independence, and local support —
properties that hold for the non-truncated hierarchical (HB) basis but are more
favorable for THB due to truncation.

## Setup

A {class}`~pantr.bspline.THBSplineSpace` is built from a
**level-0 tensor-product space** and a **hierarchical grid** that drives
refinement.  The grid is mutable; the space is an immutable snapshot of it.

```python
import numpy as np
from pantr.bspline import BsplineSpace, create_uniform_space
from pantr.grid import TensorProductGrid, HierarchicalGrid, uniform_grid

# Level-0: 4×4 elements, degree 2 in each direction
root_space = create_uniform_space([2, 2], [4, 4])

# Matching grid: 4×4 cells on [0,1]²
root_grid = uniform_grid([[0, 1], [0, 1]], [4, 4])

# Hierarchical grid with subdivision factor 2 (each refined cell → 2² children)
grid = HierarchicalGrid(root_grid, factor=2)
```

`HierarchicalGrid` requires that `root_grid.cells_per_axis` matches
`root_space.num_intervals`.

## Creating a THBSplineSpace

```python
from pantr.bspline import THBSplineSpace

space = THBSplineSpace(root_space, grid)

space.num_levels       # 1 (only the root level at this point)
space.num_total_basis  # same as root_space.num_total_basis
space.degrees          # (2, 2)
space.domain           # matches root_space.domain
```

Pass `truncate=False` to build the non-truncated HB basis instead:

```python
hb_space = THBSplineSpace(root_space, grid, truncate=False)
```

## THBSpline geometry

A {class}`~pantr.bspline.THBSpline` pairs a THB space with a control-point array
of shape `(num_total_basis, rank)`:

```python
from pantr.bspline import THBSpline

cp = np.zeros((space.num_total_basis, 3))    # 3D surface
thb = THBSpline(space, cp)
```

Evaluation mirrors the `Bspline` interface:

```python
pts = np.random.rand(500, 2)
values = thb.evaluate(pts)                   # (500, 3)
derivs = thb.evaluate_derivatives(pts, orders=(1, 0))
```

## Local refinement

`refine` returns a **new** `THBSplineSpace` on a refined grid.
Neither `self` nor the original grid is mutated.

```python
# Mark cells to refine by their flat id (row-major, C order)
cells_to_refine = [5, 6, 9, 10]   # central 2×2 block in a 4×4 grid
fine_space = space.refine(cells_to_refine)

fine_space.num_levels       # 2
fine_space.num_total_basis  # larger — new fine-level functions activated
```

By default, `refine` enforces **admissibility class 2**: active cells span at
most two successive levels, preventing overly aggressive local refinement.  Pass
`admissible_class=None` to refine exactly the marked cells with no grading:

```python
fine_space_ungraded = space.refine(cells_to_refine, admissible_class=None)
```

## Querying the space

```python
# Active basis function indices at level l
idx_l0 = fine_space.active_function_indices(0)   # functions from level 0 still active
idx_l1 = fine_space.active_function_indices(1)   # new fine-level functions

# Active functions on a given cell
active = fine_space.active_basis(cell_id)         # 1D array of global dof indices

# Access the level-l tensor-product space
tp_space_l1 = fine_space.level_space(1)
```

## Updating the geometry after refinement

After refining the space you need to re-associate control points with the new
(larger) set of active basis functions.  The `restriction_to` method provides the
prolongation operator from a coarser space to a finer one:

```python
# Build the coarse-to-fine prolongation matrix
P = fine_space.restriction_to(space)    # shape (fine dofs, coarse dofs)

# Prolongate the control points
cp_fine = (P @ cp.reshape(space.num_total_basis, -1))  # (fine dofs, rank)
thb_fine = THBSpline(fine_space, cp_fine)
```

Alternatively, re-interpolate or quasi-interpolate a function on the fine space
directly (see [Approximation](approximation.md)).

## Regularity at refined levels

By default each refinement level uses maximum continuity knots
(`regularity = degree − 1`).  To lower the continuity at subdivision knots,
pass `regularity` when constructing the space:

```python
# C⁰ continuity at all inter-level subdivision knots
space_c0 = THBSplineSpace(root_space, grid, regularity=0)

# Per-direction: C¹ in u, C⁰ in v
space_mixed = THBSplineSpace(root_space, grid, regularity=[1, 0])
```
