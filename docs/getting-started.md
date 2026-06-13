# Getting Started

## Installation

Requires Python 3.10–3.14.

```bash
pip install pantr
```

For development:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Example

Build a quadratic B-spline curve and evaluate it:

```python
import numpy as np

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# Quadratic univariate space on the knot vector [0, 0, 0, 1, 2, 3, 3, 3]
space = BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)])

# Five 2-D control points
control_points = np.array(
    [[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 1.0], [4.0, 0.0]]
)
curve = Bspline(space, control_points)

# Evaluate at 50 parameters spanning the domain [0, 3]
u = np.linspace(0.0, 3.0, 50)
points = curve.evaluate(u)  # shape (50, 2)
```

To render geometries interactively or export them to VTK, see
[Visualization](visualization.md) (requires the `viz` extra).

## Building the Documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```
