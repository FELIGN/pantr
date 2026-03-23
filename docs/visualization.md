# Visualization

PaNTr includes an optional visualization module (`pantr.viz`) built on
[PyVista](https://docs.pyvista.org/) that renders B-spline and Bezier
geometries using **native VTK higher-order Bezier cell types**.  This means
exact polynomial geometry at any zoom level -- no tessellation to triangles.

## Installation

PyVista is not installed by default.  Add it with the `viz` extra:

```bash
pip install pantr[viz]
```

## Quick start

The simplest way to visualize a geometry is the `plot()` convenience method
available on both `Bspline` and `Bezier` objects:

```python
import numpy as np
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# Build a quadratic B-spline curve
space = BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)])
cp = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [1.0, 0.5, 0.5],
    [1.5, 1.0, 0.0],
    [2.0, 0.0, 0.0],
])
curve = Bspline(space, cp)

# Interactive visualization
curve.plot(show_control_polygon=True, show_knot_lines=True)
```

For multiple geometries or finer control, use `pantr.viz` directly:

```python
from pantr.viz import plot

plot(curve, surface, show_control_polygon=True)
```

## Scene composition

The `Scene` class lets you combine multiple geometries with individual
rendering options:

```python
from pantr.viz import Scene

scene = Scene()
scene.add(surface, color="lightblue", show_knot_lines=True)
scene.add(curve, color="red", show_control_polygon=True)
scene.add(boundary_curve, color="black")
scene.show()
```

Method chaining is supported:

```python
(
    Scene()
    .add(surface, color="lightblue", opacity=0.8)
    .add(curve, color="red")
    .show()
)
```

### Per-geometry options

| Option | Type | Default | Description |
|---|---|---|---|
| `color` | `str` or `None` | `None` | Surface color (uses colormap for scalar fields if `None`) |
| `opacity` | `float` | `1.0` | Surface opacity |
| `show_control_polygon` | `bool` | `False` | Render control polygon (points and wireframe) |
| `show_knot_lines` | `bool` | `False` | Render knot lines (B-splines only) |
| `control_point_color` | `str` | `"red"` | Color of control point spheres |
| `control_point_size` | `float` | `8.0` | Size of control point spheres |
| `control_polygon_color` | `str` | `"gray"` | Color of control polygon wireframe |
| `knot_line_color` | `str` | `"black"` | Color of knot lines |
| `knot_line_width` | `float` | `2.0` | Width of knot lines |
| `scalar_bar` | `bool` | `True` | Show color bar for scalar fields |
| `elevation` | `bool` | `False` | Use scalar value as z-coordinate (see below) |

## Scalar fields

When a B-spline or Bezier has `rank == 1` (a scalar field), the visualization
depends on the parametric dimension:

- **dim=1**: line plot -- the parametric coordinate is on the x-axis and the
  scalar value on the y-axis.
- **dim=2**: by default, a **color map** on the parametric domain `(u, v, 0)`.
  Set `elevation=True` to use the scalar as the z-coordinate: `(u, v, f(u,v))`.
- **dim=3**: color map on the parametric domain `(u, v, w)`.

```python
from pantr.bezier import Bezier

# A bilinear scalar field on a quad
cp = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
scalar_field = Bezier(cp)

# Flat color map (default)
scalar_field.plot()

# Elevation surface
scalar_field.plot(elevation=True)
```

## Control polygon

The control polygon shows both the control points (as small spheres) and the
wireframe connecting adjacent control points along each parametric direction:

- **Curves**: a single polyline through all control points.
- **Surfaces**: a grid of lines along both parametric directions.
- **Volumes**: edges of the 3D control point lattice.

For rational (NURBS) geometries, the **projected Euclidean coordinates** are
used (i.e. divided by the homogeneous weight).

```python
curve.plot(show_control_polygon=True)
```

## Knot lines

Knot lines are the images of iso-parametric lines at interior knot values.
They are only available for B-splines (not Bezier objects, which have a
single element).

- **Curves (dim=1)**: knot points rendered as dots on the curve.
- **Surfaces (dim=2)**: iso-parametric curves at each interior knot in each
  direction, rendered as wireframe lines.
- **Volumes (dim=3)**: iso-parametric surfaces at each interior knot.

```python
surface_bspline.plot(show_knot_lines=True)
```

## Working with pyvista directly

For advanced use cases, convert geometries to pyvista objects and use the
full pyvista API:

```python
from pantr.viz import to_pyvista, control_polygon_mesh

# Get an UnstructuredGrid with VTK Bezier cells
grid = to_pyvista(surface)

# Manipulate with pyvista
grid.plot(show_edges=True, cmap="viridis")

# Get the control polygon as pyvista PolyData (points + wireframe)
poly_mesh = control_polygon_mesh(surface)
```

Rational geometries include a `"RationalWeights"` point data array on the
grid. Scalar fields include the scalar values as point data (named `"scalar"`
by default, configurable via `scalar_name`).

## Exporting to VTK files

Export geometries to VTK files for visualization in
[ParaView](https://www.paraview.org/) (version 5.10+ supports Bezier cells
natively):

```python
from pantr.viz import save

# XML UnstructuredGrid format (recommended)
save(surface, "surface.vtu")

# Legacy VTK format
save(surface, "surface.vtk")
```

The file format is inferred from the extension.

## Supported geometries

| Parametric dim | Rank | VTK cell type | Description |
|---|---|---|---|
| 1 | 2--3 | Bezier curve | 2D/3D curve |
| 1 | 1 | Bezier curve | Scalar line plot |
| 2 | 2--3 | Bezier quadrilateral | 2D/3D surface |
| 2 | 1 | Bezier quadrilateral | Scalar color map / elevation |
| 3 | 3 | Bezier hexahedron | 3D volume |
| 3 | 1 | Bezier hexahedron | Scalar color map on volume |

Both non-rational and rational (NURBS) geometries are supported.  Periodic
and unclamped B-splines are automatically converted to open form before
visualization.
