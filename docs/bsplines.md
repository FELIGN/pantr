# B-Splines

The B-spline module provides two complementary objects: **spaces**
({class}`~pantr.bspline.BsplineSpace1D`, {class}`~pantr.bspline.BsplineSpace`)
that describe the parametric structure, and **geometries**
({class}`~pantr.bspline.Bspline`) that pair a space with a control-point array.

## Spaces

### BsplineSpace1D

A univariate B-spline space is defined by a knot vector and a degree:

```python
from pantr.bspline import BsplineSpace1D

# Degree-3 space: 4 knot spans on [0, 4], maximum C² continuity
knots = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
space_1d = BsplineSpace1D(knots, 3)

space_1d.degree        # 3
space_1d.num_basis     # 7
space_1d.num_intervals # 4
space_1d.domain        # (0.0, 4.0)
```

### Knot vector factories

For the most common cases the factory functions are more convenient than
spelling out knot vectors by hand:

```python
from pantr.bspline import (
    create_uniform_open_knots,
    create_uniform_periodic_knots,
    create_cardinal_knots,
)

# 4-element degree-3 open (clamped) knot vector on [0, 1]
knots_open = create_uniform_open_knots(4, 3)
# → [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]

# Same geometry but periodic (wraps around)
knots_per = create_uniform_periodic_knots(4, 3)

# Cardinal: 8 basis functions, degree 3, C² everywhere (maximum continuity)
knots_card = create_cardinal_knots(8, 3)

# Control continuity explicitly (C¹ interior knots)
knots_c1 = create_uniform_open_knots(4, 3, continuity=1)
```

### BsplineSpace (tensor-product)

`BsplineSpace` aggregates per-direction `BsplineSpace1D` objects:

```python
from pantr.bspline import BsplineSpace, BsplineSpace1D

s_u = BsplineSpace1D([0, 0, 0, 0, 1, 2, 2, 2, 2], 3)   # 2 elements, degree 3
s_v = BsplineSpace1D([0, 0, 1, 1], 1)                    # 1 element, degree 1
space = BsplineSpace([s_u, s_v])

space.dim            # 2
space.degrees        # (3, 1)
space.num_basis      # (5, 2)
space.num_intervals  # (2, 1)
```

`create_uniform_space` is the shorthand for uniform tensor-product spaces:

```python
from pantr.bspline import create_uniform_space

# Degree-3, 8 elements in each direction, 2D, default domain [0,1]²
space_2d = create_uniform_space([3, 3], [8, 8])

# Mixed degrees
space_mix = create_uniform_space([3, 2], [4, 6])

# Periodic in the u-direction, open in v
space_per = create_uniform_space([2, 2], [8, 8], periodic=[True, False])
```

### Greville abscissae

Greville abscissae are the canonical interpolation nodes — one per basis function
per direction.  For tensor-product spaces, `create_greville_lattice` builds a
{class}`~pantr.quad.PointsLattice` that can be passed directly to
{func}`~pantr.bspline.interpolate_bspline`:

```python
from pantr.bspline import get_greville_abscissae, create_greville_lattice

g = get_greville_abscissae(space_1d)       # 1D array, length num_basis
lattice = create_greville_lattice(space_2d) # PointsLattice for 2D space
```

## Bspline geometry

A `Bspline` pairs a space with an array of control points:

```python
import numpy as np
from pantr.bspline import Bspline, create_uniform_space

space = create_uniform_space(3, 4)            # degree-3, 4 elements on [0,1]
cp = np.zeros((space.num_total_basis, 2))     # 7 control points, 2D curve
cp[:, 0] = np.linspace(0, 1, 7)

curve = Bspline(space, cp)
curve.rank   # 2
```

The control-point array has shape `(*num_basis, rank)`.

### NURBS (rational B-splines)

For rational geometries, include the homogeneous weight as the last coordinate and
pass `is_rational=True`.  The `rank` property returns the *spatial* dimension (weight
excluded):

```python
from pantr.bspline import BsplineSpace, BsplineSpace1D

# Quarter-circle arc: rational degree-2 curve in 2D
w = 1.0 / np.sqrt(2)
cp_h = np.array([[1., 0., 1.], [1., 1., w], [0., 1., 1.]])  # (x*w, y*w, w)
arc = Bspline(
    BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)]),
    cp_h,
    is_rational=True,
)
arc.rank       # 2  (weight excluded)
arc.is_rational # True
```

## Evaluation

`evaluate(pts)` takes an `(n, dim)` array of parametric coordinates and returns an
`(n, rank)` array:

```python
u = np.linspace(0, 1, 200).reshape(-1, 1)
pts = curve.evaluate(u)               # (200, 2)
```

For surfaces, pass a 2-column array:

```python
import pantr.quad as pq

# 20×20 tensor-product grid
lat = pq.PointsLattice([np.linspace(0, 1, 20), np.linspace(0, 1, 20)])
u2d = lat.get_all_points()            # (400, 2)
pts_2d = surface.evaluate(u2d)        # (400, rank)
```

Mixed partial derivatives are available through `evaluate_derivatives`:

```python
# First derivative in u
du = curve.evaluate_derivatives(u, orders=(1,))   # (200, 2)

# Cross-derivative d²f/du dv for a surface
d_uv = surface.evaluate_derivatives(u2d, orders=(1, 1))
```

`derivative(direction)` returns the hodograph as a new `Bspline`:

```python
dcurve = curve.derivative(0)          # degree-2 curve
```

## Knot operations

All operations return new `Bspline` objects; `self` is never mutated.

### Knot insertion

```python
# Insert knots [0.25, 0.75] in the u-direction
refined = curve.insert_knots([0.25, 0.75])

# Multi-dimensional: insert in u only, skip v
surface_ref = surface.insert_knots([[0.25, 0.75], None])
```

### Knot removal

```python
# Remove knot 0.5 (as many times as geometry allows)
coarsened = curve.remove_knots([0.5])

# Set explicit tolerance and removal count
coarsened = curve.remove_knots([0.5], tol=1e-8, num=1)
```

### Subdivision

`subdivide` inserts uniform knots to split each element into `n` sub-elements.
The `regularity` argument controls the continuity at inserted knots
(default: `degree - 1`, maximum continuity):

```python
# Split each element into 4 equal sub-elements (maximum continuity)
fine = curve.subdivide([4])

# C⁰ continuity (breakpoints only)
fine_c0 = curve.subdivide([4], regularity=0)
```

### Knot topology conversion

```python
open_bsp  = periodic_curve.to_open_bspline()   # periodic → open (clamped)
per_bsp   = open_curve.to_periodic()            # open → periodic
```

## Degree operations

Degree elevation is exact; degree reduction is a least-squares approximation:

```python
elevated = curve.elevate_degree((1,))     # degree p → p+1, exact
reduced  = curve.reduce_degree((1,))      # degree p → p-1, approximate
```

## Domain operations

```python
left, right = curve.split(0, 0.5)          # split at u = 0.5
sub          = curve.restrict([(0.25, 0.75)])  # restrict to sub-interval
edge         = surface.boundary(0, 0)          # u_min boundary
section      = surface.slice(1, 0.5)           # fix v = 0.5 → curve
```

## Algebraic and geometric operations

```python
# Exact pointwise product
product = a.multiply(b)

# Reverse parametric direction
rev = curve.reverse(0)

# Swap u ↔ v for a surface
swapped = surface.permute_directions((1, 0))

# Affine transformation (translation, rotation, scale, …)
from pantr.cad import AffineTransform
moved = curve.transform(AffineTransform.translation([1.0, 0.0, 0.0]))
```

## Converting to Bézier

```python
# Single-element B-spline → Bézier
bezier = curve.to_bezier()

# Multi-element B-spline → array of per-element Bézier patches (cached)
beziers = curve.to_beziers()          # array of Bezier objects
for bez in beziers.ravel():
    print(bez.degree)
```
