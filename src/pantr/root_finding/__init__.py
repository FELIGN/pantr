"""Root finding for polynomials in the Bernstein basis.

Provides algorithms for finding all real roots of a scalar Bernstein polynomial
on [0, 1]. Two complementary solvers are available:

- **Yuksel** (2022): recursive monotone-decomposition via the derivative chain.
  Scale-invariant, low overhead for moderate degrees.
- **Bezier clipping**: iterative convex-hull narrowing without derivatives.
  Super-linear convergence, preferred for high-degree well-conditioned polynomials.

An automatic dispatcher selects the best algorithm based on polynomial degree and
coefficient dynamic range.

Main exports:

- :func:`find_roots` -- find all roots (single polynomial, auto-dispatch).
- :func:`find_roots_batch` -- find roots of many same-degree polynomials in parallel.
- :func:`solve_monotone_root` -- fast solver for a single root on a monotone polynomial.
- :func:`solve_monotone_root_batch` -- batch-parallel monotone solver.
"""

from pantr.root_finding._root_finding import (
    find_roots,
    find_roots_batch,
    solve_monotone_root,
    solve_monotone_root_batch,
)

__all__ = [
    "find_roots",
    "find_roots_batch",
    "solve_monotone_root",
    "solve_monotone_root_batch",
]
