# Splines

PaNTr provides three spline types that share a common interface for evaluation,
derivatives, degree operations, and domain manipulation.

| Type | Class | Description |
|---|---|---|
| Bézier | {class}`~pantr.bezier.Bezier` | Single-element polynomial in Bernstein form |
| B-spline / NURBS | {class}`~pantr.bspline.Bspline` | Piecewise polynomial on a knot vector |
| THB-spline | {class}`~pantr.bspline.THBSpline` | Locally refined spline on a hierarchical grid |

All three share a consistent interface:

- **Evaluation** — `evaluate(pts)` maps parametric points to physical coordinates.
- **Derivatives** — `evaluate_derivatives(pts, orders)` and `derivative(direction)`.
- **Rank** — the physical-space dimension of the image (1 for scalar fields, 2 for
  2D curves, 3 for 3D surfaces, etc.).
- **Rational geometry** — pass `is_rational=True` to use the last control-point
  coordinate as a homogeneous weight, enabling NURBS.

```{toctree}
:maxdepth: 1

bezier
bsplines
thb-splines
approximation
spanwise-element-extraction
```
