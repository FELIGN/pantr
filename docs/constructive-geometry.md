# Constructive Geometry

The `pantr.cad` module provides CAD-style functions for building B-spline
curves, surfaces, and volumes from geometric descriptions.  All functions
return {class}`~pantr.bspline.Bspline` objects that integrate with the
rest of PaNTr (evaluation, derivatives, knot insertion, degree elevation, etc.).

## Primitives

Primitives create B-spline objects directly from geometric parameters.
All produce rank-3 (3D) output; lower-dimensional inputs are zero-padded.

### Lines and multilinear patches

```python
from pantr.cad import line, bilinear, trilinear

crv = line([0, 0, 0], [1, 0, 0])           # degree-1 curve
srf = bilinear()                            # unit square in XY
vol = trilinear()                           # unit cube
```

{func}`~pantr.cad.line` creates a degree-1 curve between two points.
{func}`~pantr.cad.bilinear` creates a degree-(1, 1) surface from four corners.
{func}`~pantr.cad.trilinear` creates a degree-(1, 1, 1) volume from eight corners.

### Circles and arcs

```python
from pantr.cad import circle
import numpy as np

full = circle(radius=2.0)                         # full circle, 4 spans
arc  = circle(angle=np.pi / 2)                    # quarter arc, 1 span
arc2 = circle(angle=(np.pi / 4, np.pi), radius=3) # arc from 45 to 180 deg
```

{func}`~pantr.cad.circle` builds an exact rational quadratic B-spline.
The number of spans depends on the sweep angle (one span per 90 degrees),
with C0 interior knots (multiplicity equal to degree).  This is the
standard conic representation.

### Derived shapes

```python
from pantr.cad import rectangle, disk, cylinder

rect = rectangle(corner=[0, 0, 0], width=2, height=3)  # closed curve
ann  = disk(radius_inner=0.5, radius_outer=1.0)         # annular sector
cyl  = cylinder(radius=1.0, height=5.0)                 # cylindrical surface
```

{func}`~pantr.cad.rectangle` returns a closed degree-1 curve visiting four corners.
{func}`~pantr.cad.disk` builds an annular sector (or full disk when `radius_inner=0`)
via {func}`~pantr.cad.ruled` between inner and outer circles.
{func}`~pantr.cad.cylinder` extrudes a circle along the z-axis.

## Operations

Operations create higher-dimensional objects from existing ones by adding
a new parametric direction.

### Extrude

```python
from pantr.cad import circle, extrude

pipe = extrude(circle(), [0, 0, 2])   # circle -> cylindrical surface
```

{func}`~pantr.cad.extrude` translates a curve or surface along a displacement
vector, appending a degree-1 parametric direction.

### Revolve

```python
from pantr.cad import line, revolve
import numpy as np

srf = revolve(line([1, 0, 0], [2, 0, 0]), point=0, axis=2)
quarter = revolve(line([1, 0, 0], [1, 0, 3]), point=0, axis=2,
                  angle=np.pi / 2)
```

{func}`~pantr.cad.revolve` rotates a curve or surface around an axis.
The angular direction inherits the same span structure as
{func}`~pantr.cad.circle` (one span per 90 degrees, C0 at arc junctions).
Supports coordinate axes (`axis=0, 1, 2`) and arbitrary axis vectors.

### Ruled

```python
from pantr.cad import circle, ruled

annulus = ruled(circle(radius=0.5), circle(radius=1.0))
```

{func}`~pantr.cad.ruled` linearly interpolates between two curves (or
surfaces) to produce a surface (or volume).  The inputs are automatically
made compatible via {func}`~pantr.cad.compat`.

### Sweep

```python
from pantr.cad import line, sweep

srf = sweep(line([0, 0, 0], [1, 0, 0]),    # section
            line([0, 0, 0], [0, 0, 3]))     # trajectory
```

{func}`~pantr.cad.sweep` creates a translational sweep:
$S(u, v) = \text{section}(u) + \text{trajectory}(v)$.

## Coons blending

### Surface

```python
from pantr.cad import coons_surface, line

c_u0 = line([0, 0, 0], [1, 0, 0])   # bottom
c_u1 = line([0, 1, 0], [1, 1, 0])   # top
c_v0 = line([0, 0, 0], [0, 1, 0])   # left
c_v1 = line([1, 0, 0], [1, 1, 0])   # right

srf = coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
```

{func}`~pantr.cad.coons_surface` builds a bilinearly blended surface from
four boundary curves using the formula $S = R_0 + R_1 - B$, where $R_0$
and $R_1$ are ruled surfaces and $B$ is the bilinear corner interpolant.

### Volume

```python
from pantr.cad import bilinear, coons_volume
import numpy as np

# Build 6 faces of a unit cube
face_u0 = bilinear(np.array([[[0,0,0],[0,0,1]],[[0,1,0],[0,1,1]]], dtype=float))
face_u1 = bilinear(np.array([[[1,0,0],[1,0,1]],[[1,1,0],[1,1,1]]], dtype=float))
face_v0 = bilinear(np.array([[[0,0,0],[0,0,1]],[[1,0,0],[1,0,1]]], dtype=float))
face_v1 = bilinear(np.array([[[0,1,0],[0,1,1]],[[1,1,0],[1,1,1]]], dtype=float))
face_w0 = bilinear(np.array([[[0,0,0],[0,1,0]],[[1,0,0],[1,1,0]]], dtype=float))
face_w1 = bilinear(np.array([[[0,0,1],[0,1,1]],[[1,0,1],[1,1,1]]], dtype=float))

vol = coons_volume(((face_u0, face_u1),
                    (face_v0, face_v1),
                    (face_w0, face_w1)))
```

{func}`~pantr.cad.coons_volume` builds a trilinearly blended volume from
six boundary faces using the inclusion-exclusion formula:

$$V = (R_u + R_v + R_w) - (B_{uv} + B_{uw} + B_{vw}) + T$$

where $R$ are ruled volumes from opposite face pairs, $B$ are bilinear
blend volumes from edge quadruples, and $T$ is the trilinear corner
interpolant.  Edges and corners are extracted automatically from face
boundaries.

## Compatibility and assembly

### compat

```python
from pantr.cad import compat, circle, line

c1 = line([0, 0, 0], [1, 0, 0])   # degree 1
c2 = circle(angle=np.pi / 2)       # degree 2, different knots
r1, r2 = compat(c1, c2)            # now same degree and knots
```

{func}`~pantr.cad.compat` makes N B-splines compatible along specified axes by:
1. Remapping domains to a common envelope.
2. Elevating degrees to the maximum.
3. Merging knot vectors (union of breakpoints with max multiplicities).

This is called internally by {func}`~pantr.cad.ruled`,
{func}`~pantr.cad.coons_surface`, {func}`~pantr.cad.coons_volume`, and
{func}`~pantr.cad.join`.

### join

```python
from pantr.cad import join, line

c1 = line([0, 0, 0], [1, 0, 0])
c2 = line([1, 0, 0], [2, 1, 0])
merged = join(c1, c2, axis=0)
```

{func}`~pantr.cad.join` concatenates two B-splines along a parametric
axis with C0 continuity.  Knots are automatically removed at the junction
when the geometry permits higher smoothness.
