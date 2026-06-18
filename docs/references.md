# References

The algorithms and mathematical objects in PaNTr come from a well-established
literature. This page collects the works cited throughout the documentation, followed
by encyclopedic background reading for the core concepts.

## Bibliography

```{bibliography}
:style: unsrt
:all:
```

## Background reading

Encyclopedic introductions to the underlying mathematics, useful as a gentle on-ramp
before the primary literature above:

- [B-spline](https://en.wikipedia.org/wiki/B-spline) — basis functions, knot vectors,
  and the Cox–de Boor recurrence.
- [Non-uniform rational B-spline (NURBS)](https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline)
  — rational spline geometry.
- [Bézier curve](https://en.wikipedia.org/wiki/B%C3%A9zier_curve) and
  [Bernstein polynomial](https://en.wikipedia.org/wiki/Bernstein_polynomial) — the
  Bézier toolkit and its polynomial basis.
- [De Boor's algorithm](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm) — stable
  evaluation of B-spline curves.
- [De Casteljau's algorithm](https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm)
  — stable evaluation and subdivision of Bézier curves.
- [Isogeometric analysis](https://en.wikipedia.org/wiki/Isogeometric_analysis) — the
  CAD/analysis bridge PaNTr targets.
- [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature) — the
  Gauss–Legendre rules behind {mod}`pantr.quad`.
- [Coons patch](https://en.wikipedia.org/wiki/Coons_patch) — transfinite interpolation
  used by the constructive-geometry surfaces in {mod}`pantr.cad`.
