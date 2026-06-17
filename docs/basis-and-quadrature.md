# Polynomial Basis and Quadrature

PaNTr provides tabulation routines for several polynomial bases, change-of-basis
matrices between them, and a suite of 1D quadrature rules with tensor-product
extensions.  These tools operate on fixed-degree single-element domains
$[0,1]^{\text{dim}}$ and are the building blocks used internally by the spline
machinery — but they are also useful directly, for example in finite-element
or spectral-element workflows.

## Polynomial bases (`pantr.basis`)

Four bases are supported.  All live on the reference interval $[0, 1]$:

| Basis | Function | Node distribution |
|---|---|---|
| Bernstein | `tabulate_bernstein_1d` | — |
| Cardinal B-spline | `tabulate_cardinal_bspline_1d` | uniform, max continuity |
| Lagrange | `tabulate_lagrange_1d` | configurable (see below) |
| Shifted Legendre | `tabulate_legendre_1d` | — (orthonormal on $[0,1]$) |

All 1D functions share the same calling convention:

```python
import numpy as np
from pantr.basis import tabulate_bernstein_1d

pts = np.linspace(0, 1, 50)
B = tabulate_bernstein_1d(3, pts)   # shape (50, 4): 4 degree-3 Bernstein polynomials
```

The return array has shape `(n_pts, degree+1)`.  An optional `out` argument accepts
a pre-allocated array (NumPy style):

```python
out = np.empty((50, 4))
tabulate_bernstein_1d(3, pts, out=out)
```

### Lagrange variants

The Lagrange basis depends on a node distribution selected through
{class}`~pantr.basis.LagrangeVariant`:

```python
from pantr.basis import tabulate_lagrange_1d, LagrangeVariant

# Gauss-Lobatto-Legendre nodes (recommended for high degree — low Lebesgue constant)
L = tabulate_lagrange_1d(4, LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, pts)
```

Available variants:

| Variant | Nodes |
|---|---|
| `EQUISPACES` | Equispaced on $[0,1]$ |
| `GAUSS_LEGENDRE` | Roots of the Legendre polynomial |
| `GAUSS_LOBATTO_LEGENDRE` | GLL nodes (includes endpoints) |
| `CHEBYSHEV_1ST` | Chebyshev points of the first kind |
| `CHEBYSHEV_2ND` | Chebyshev points of the second kind (Lobatto) |

### Multi-dimensional bases

Tensor-product generalizations of Bernstein, cardinal B-spline, and Lagrange bases
accept a sequence of per-direction degrees and either a flat `(n_pts, dim)` array
of scattered points or a {class}`~pantr.quad.PointsLattice`:

```python
from pantr.basis import tabulate_bernstein, tabulate_lagrange
from pantr.quad import PointsLattice

# Scattered points: 100 random points in [0,1]²
pts_2d = np.random.rand(100, 2)
B2 = tabulate_bernstein([2, 3], pts_2d)    # (100, 12): 3×4 tensor-product basis

# Tensor-product grid via PointsLattice (more efficient for structured grids)
lattice = PointsLattice([np.linspace(0, 1, 10), np.linspace(0, 1, 8)])
B_lat = tabulate_bernstein([2, 3], lattice)  # (80, 12)
```

The output is always `(n_pts, n_basis)` where `n_basis = prod(degree_i + 1)`.

The `funcs_order` argument (`"C"` by default, or `"F"`) controls the linearization
order of the multi-index over basis functions.

## Change of basis (`pantr.change_basis`)

All change-of-basis functions return a square `(degree+1, degree+1)` matrix $M$
such that

$$\text{new\_basis}(x) = M \cdot \text{old\_basis}(x)$$

on $[0,1]$. The Bernstein basis is the hub — all other bases connect through it:

```python
from pantr.change_basis import (
    compute_lagrange_to_bernstein_1d,
    compute_bernstein_to_lagrange_1d,
    compute_bernstein_to_cardinal_1d,
    compute_cardinal_to_bernstein_1d,
    compute_monomial_to_bernstein_1d,
)
from pantr.basis import LagrangeVariant

degree = 3

# Lagrange (GLL) ↔ Bernstein
L2B = compute_lagrange_to_bernstein_1d(degree, LagrangeVariant.GAUSS_LOBATTO_LEGENDRE)
B2L = compute_bernstein_to_lagrange_1d(degree, LagrangeVariant.GAUSS_LOBATTO_LEGENDRE)

# Cardinal B-spline ↔ Bernstein
C2B = compute_cardinal_to_bernstein_1d(degree)
B2C = compute_bernstein_to_cardinal_1d(degree)

# Monomial → Bernstein (lower-triangular; M[i,j] = C(i,j)/C(degree,j))
M2B = compute_monomial_to_bernstein_1d(degree)
```

These matrices are used internally by {class}`~pantr.bspline.SpanwiseElementExtraction`
for element-local basis conversion, and are LRU-cached for repeated calls with the
same arguments.

## Quadrature (`pantr.quad`)

### 1D rules

All 1D rules return a `(nodes, weights)` pair on $[0, 1]$:

```python
from pantr.quad import (
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_trapezoidal_1d,
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_tanh_sinh_1d,
)

nodes, weights = get_gauss_legendre_1d(5)         # 5-point GL rule on [0,1]
nodes, weights = get_gauss_lobatto_legendre_1d(5)  # includes endpoints
nodes, weights = get_tanh_sinh_1d(8)              # double-exponential, endpoint clustering
```

| Rule | Use case |
|---|---|
| `get_gauss_legendre_1d` | Default; exact for polynomials of degree $\le 2n-1$ |
| `get_gauss_lobatto_legendre_1d` | Matches GLL Lagrange nodes; includes endpoints |
| `get_trapezoidal_1d` | Periodic integrands; equispaced |
| `get_chebyshev_gauss_1st_kind_1d` | Integration against Chebyshev weight |
| `get_chebyshev_gauss_2nd_kind_1d` | Interior nodes only |
| `get_tanh_sinh_1d` | Endpoint singularities or steep boundary layers |

`get_modified_chebyshev_nodes_1d` returns Chebyshev-Lobatto **nodes only** (no
weights) — use it for polynomial interpolation into the Bernstein basis, not for
integration:

```python
from pantr.quad import get_modified_chebyshev_nodes_1d

cheb_nodes = get_modified_chebyshev_nodes_1d(6)   # (6,) array, includes 0 and 1
```

### PointsLattice

{class}`~pantr.quad.PointsLattice` stores a tensor-product evaluation grid as one
1D coordinate array per direction.  It is accepted wherever multi-dimensional
tabulation functions expect an array of points:

```python
from pantr.quad import PointsLattice

# 10×8 evaluation grid in 2D
lattice = PointsLattice([np.linspace(0, 1, 10), np.linspace(0, 1, 8)])
lattice.dim       # 2
lattice.pts_per_dir  # tuple of two 1D arrays

# Flatten to (80, 2) array (C order: last index varies fastest)
all_pts = lattice.get_all_points()
all_pts_F = lattice.get_all_points(order="F")
```

`create_lagrange_points_lattice` builds a `PointsLattice` whose nodes follow a
specific Lagrange variant — convenient for interpolation node lattices:

```python
from pantr.quad import create_lagrange_points_lattice
from pantr.basis import LagrangeVariant

lattice = create_lagrange_points_lattice(
    LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, [5, 5]
)
```

### QuadratureRule

{class}`~pantr.quad.QuadratureRule` is an immutable tensor-product rule on the unit
cube $[0,1]^d$.  It is the reference rule consumed by
{func}`pantr.grid.cell_quadrature`, which maps it onto each physical grid cell.

```python
from pantr.quad import gauss_legendre_quadrature, tensor_product_quadrature

# 3×3 Gauss-Legendre rule on [0,1]²
rule = gauss_legendre_quadrature(ndim=2, npts=3)

rule.ndim        # 2
rule.num_points  # 9
rule.points      # (9, 2) read-only array on [0,1]²
rule.weights     # (9,) read-only, sum = 1
```

Mix different 1D rules with `tensor_product_quadrature`:

```python
from pantr.quad import tensor_product_quadrature, get_gauss_legendre_1d, get_gauss_lobatto_legendre_1d

rule_mixed = tensor_product_quadrature([
    get_gauss_legendre_1d(4),          # u-direction: 4-point GL
    get_gauss_lobatto_legendre_1d(5),  # v-direction: 5-point GLL
])
```

To integrate over every cell of a grid, use {func}`pantr.grid.cell_quadrature`
(see the Grids section of the API reference), which returns physical quadrature
points and weights scaled by the Jacobian of the affine cell map.
