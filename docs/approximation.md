# Approximation

PaNTr provides four strategies for constructing a spline that approximates a
given callable, ranging from exact interpolation to fast local projectors.
All functions are importable from `pantr.bspline`.

| Function | Method | Cost | Notes |
|---|---|---|---|
| `interpolate_bspline` | Interpolation (collocation) | Medium | Exact at nodes; Greville default |
| `fit_bspline` | Least-squares fit | Medium | Overdetermined system; scattered points ok |
| `l2_project_bspline` | L² projection | Higher | Global best approximation in L² |
| `quasi_interpolate_bspline` | Quasi-interpolation (LLM) | Low | Local, no linear system |
| `quasi_interpolate_thb_spline` | Hierarchical quasi-interpolation | Low | Hierarchical variant of LLM |

## Interpolation

`interpolate_bspline` evaluates the callable on a node lattice and recovers
B-spline coefficients by solving a collocation system.  By default it uses the
**Greville abscissae** of the space:

```python
import numpy as np
from pantr.bspline import create_uniform_space, interpolate_bspline

space = create_uniform_space(3, 8)    # degree-3, 8 elements on [0,1]

# Interpolate sin(2πu) — func receives a PointsLattice, must return (n,) or (n, rank)
result = interpolate_bspline(
    lambda lat: np.sin(2 * np.pi * lat.get_all_points()[:, 0]),
    space,
)
```

Custom interpolation nodes override the Greville default:

```python
# Gauss-Lobatto nodes (better conditioning for high degree)
from pantr.quad import get_gauss_lobatto_legendre_1d
nodes, _ = get_gauss_lobatto_legendre_1d(space.num_basis[0])  # returns (nodes, weights)
result = interpolate_bspline(func, space, nodes=nodes)
```

For multi-dimensional spaces, pass a list of per-direction node arrays, or a
{class}`~pantr.quad.PointsLattice`:

```python
space_2d = create_uniform_space([3, 3], [4, 4])
result_2d = interpolate_bspline(
    lambda lat: np.linalg.norm(lat.get_all_points(), axis=1),
    space_2d,
)
```

## Least-squares fit

`fit_bspline` takes *pre-evaluated* sample values and nodes, making it suitable
when the function has already been sampled or when using scattered data:

```python
from pantr.bspline import fit_bspline

# Tensor-product nodes (overdetermined: 100 points, 11 dofs)
nodes = np.linspace(0, 1, 100)
values = np.sin(2 * np.pi * nodes)
result = fit_bspline(values, [nodes], space)

# Scattered points in 2D
pts_scattered = np.random.rand(500, 2)
vals_scattered = np.sin(pts_scattered[:, 0]) * np.cos(pts_scattered[:, 1])
result_2d = fit_bspline(vals_scattered, pts_scattered, space_2d)
```

## L² projection

`l2_project_bspline` assembles per-element mass matrices and load vectors using
Gauss–Legendre quadrature and solves the resulting normal equations.  The result
minimises the L² error over the domain:

```python
from pantr.bspline import l2_project_bspline

result = l2_project_bspline(
    lambda lat: np.sin(2 * np.pi * lat.get_all_points()[:, 0]),
    space,
)

# More quadrature points per element (default is degree+1)
result_fine = l2_project_bspline(func, space, n_quad=6)

# Interpolate boundary values instead of projecting them
result_bc = l2_project_bspline(func, space, boundary_interpolation=True)
```

## Quasi-interpolation

`quasi_interpolate_bspline` implements the **Lee-Lyche-Mørken (LLM)** local
projector.  It samples the function at a small local set of points per basis
function and assembles coefficients without solving any global linear system.
This makes it the fastest option for smooth functions:

```python
from pantr.bspline import quasi_interpolate_bspline

# func receives a flat (M, dim) point array — not a PointsLattice
result = quasi_interpolate_bspline(
    lambda pts: np.sin(2 * np.pi * pts[:, 0]),
    space,
)
```

### THB quasi-interpolation

For hierarchical spaces, `quasi_interpolate_thb_spline` uses the
**Speleers-Manni hierarchical quasi-interpolant**: each active dof gets the LLM
coefficient evaluated at a leaf cell at the function's own level.

```python
from pantr.bspline import THBSplineSpace, quasi_interpolate_thb_spline
from pantr.grid import HierarchicalGrid, uniform_grid

root_space = create_uniform_space([2, 2], [4, 4])
root_grid  = uniform_grid([[0, 1], [0, 1]], [4, 4])
grid       = HierarchicalGrid(root_grid, factor=2)
space_thb  = THBSplineSpace(root_space, grid)

# Refine some cells
space_fine = space_thb.refine([5, 6, 9, 10])

result_thb = quasi_interpolate_thb_spline(
    lambda pts: np.sin(2 * np.pi * pts[:, 0]) * pts[:, 1],
    space_fine,
)
```

## Bézier approximation

Both interpolation and least-squares fitting are also available for
{class}`~pantr.bezier.Bezier` objects:

```python
from pantr.bezier import interpolate_bezier, fit_bezier

# Interpolate a callable: sampled at n_pts nodes, degree inferred as n_pts - 1.
# func is called as func(lattice) with a PointsLattice and returns (n_total, rank).
def f(lattice):
    u = lattice.get_all_points()[:, 0]
    return np.column_stack([u, np.sin(np.pi * u)])

bezier = interpolate_bezier(f, 5)        # exact, degree 4 at 5 Chebyshev nodes

# Least-squares fit from pre-evaluated samples (values at known nodes)
nodes  = np.linspace(0, 1, 5)
values = np.array([[0., 0.], [0.3, 0.9], [0.6, 0.7], [0.8, 0.3], [1., 0.]])
bezier_fit = fit_bezier(values, nodes, degree=3)   # degree 3 < 5 nodes → least squares
```
