# Bézier

A {class}`~pantr.bezier.Bezier` is a polynomial map from $[0,1]^{\text{dim}}$ to
$\mathbb{R}^{\text{rank}}$, stored in the **Bernstein basis**.  The degree in each
parametric direction is inferred from the control-point array shape: a shape
`(p+1, rank)` array gives a degree-$p$ curve; a shape `(p+1, q+1, rank)` array gives
a degree-$(p,q)$ surface.

## Creating a Bézier

```python
import numpy as np
from pantr.bezier import Bezier

# Degree-2 planar curve (3 control points, rank=2)
cp = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
curve = Bezier(cp)
curve.dim     # 1
curve.degree  # (2,)
curve.rank    # 2

# Bilinear surface patch (degree 1 × 1, rank=3)
cp_surf = np.array([
    [[0., 0., 0.], [0., 1., 0.]],
    [[1., 0., 0.], [1., 1., 0.]],
])
surface = Bezier(cp_surf)
surface.dim    # 2
surface.degree # (1, 1)
```

### Rational Bézier (NURBS)

Pass `is_rational=True` and include the homogeneous weight as the last coordinate of
each control point.  The `rank` property returns the *spatial* dimension (weight
excluded):

```python
w = 1.0 / np.sqrt(2)
# Quarter-circle: rational degree-2 curve in 2D
cp = np.array([[1., 0., 1.], [1., 1., w], [0., 1., 1.]])   # (x*w, y*w, w)
arc = Bezier(cp, is_rational=True)
arc.rank   # 2
```

## Evaluation

`evaluate(pts)` expects an `(n, dim)` float64 array of parametric points on
$[0,1]^{\text{dim}}$ and returns an `(n, rank)` array:

```python
u = np.linspace(0, 1, 100).reshape(-1, 1)
pts = curve.evaluate(u)    # (100, 2)
```

Mixed partial derivatives of any order are available through `evaluate_derivatives`.
The `orders` argument gives the derivation order per direction:

```python
# First derivative in the u-direction
d1 = curve.evaluate_derivatives(u, orders=(1,))   # (100, 2)

# Second derivative for a bivariate surface
u2 = np.random.rand(50, 2)
d_uv = surface.evaluate_derivatives(u2, orders=(1, 1))   # (50, 3)
```

`derivative(direction)` returns the hodograph as a new `Bezier` of degree reduced
by one:

```python
dcurve = curve.derivative(0)   # degree-1 Bezier, same rank
```

## Degree operations

Degree elevation is exact; degree reduction is a least-squares approximation:

```python
elevated = curve.elevate_degree((1,))    # degree 2 → 3, exact
reduced  = curve.reduce_degree((1,))     # degree 2 → 1, approximate
minimal  = curve.minimize_degree()       # find minimal degree within tolerance
```

## Domain operations

All operations return new `Bezier` objects; `self` is never modified.

```python
left, right = curve.split(0, 0.5)          # split at u = 0.5 in direction 0
sub          = curve.restrict([(0.2, 0.8)])  # reparametrize to [0.2, 0.8]
edge         = surface.boundary(0, 0)        # u = 0 boundary → curve
section      = surface.slice(1, 0.5)         # fix v = 0.5 → curve
```

## Algebraic operations

`multiply` computes the exact pointwise product of two Béziers with the same domain
dimension; `compose` substitutes one Bézier into another:

```python
# Exact product: (a · b)(u) = a(u) * b(u), component-wise
product = a.multiply(b)

# Composition: outer(inner(u))
composed = outer.compose(inner)
```

## Root-finding

For scalar Béziers (`rank == 1`), two root-finding functions are available:

```python
from pantr.bezier import find_roots, find_monotone_root

# All roots on [0, 1] (general polynomial)
roots = find_roots(scalar_curve)

# Single root of a strictly monotone polynomial (faster)
root = find_monotone_root(mono_curve)
```

## Converting between Bézier and B-spline

```python
# Bézier → B-spline with Bézier-like knot vector (open, single element)
bspline = curve.to_bspline()

# B-spline (single-element) → Bézier
bezier = bspline_curve.to_bezier()

# B-spline → array of Bézier patches, one per element
beziers = bspline_curve.to_beziers()
```

The `to_beziers` result is cached on the `Bspline` object, so repeated calls are
free.
