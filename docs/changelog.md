# Changelog

## Unreleased

### Added
- `pantr.multipatch`: new subpackage for multi-patch topology. `Interface` and
  `MultiPatch` describe how patches meet; `detect_interfaces` finds the
  interfaces of a patch collection by enumerating candidate orientations and
  verifying them, which resolves the ambiguity a corner-matching rule leaves on
  symmetric faces; `match_face_cps` pairs the control points across an
  interface. Knots, coordinates and NURBS weights are compared against separate
  tolerances, since they are not the same kind of quantity.
- `pantr.bspline.find_roots`: every zero of a scalar univariate B-spline,
  computed on the spline's own knot vector by repeatedly inserting the zero of
  the control polygon (Mørken & Reimers, *Math. Comp.* 76, 2007). No Bézier
  extraction, no stitching of roots at segment boundaries, and about 8 times
  faster than extraction plus the Bernstein solver on a cubic with 1000
  elements.
- `Bspline.locate`: point inversion from physical to parametric coordinates,
  batched over all points, returning the cell and the parameters. The default
  tolerance scales with the coordinate magnitude, not only with the bounding-box
  diagonal, so a geometry far from the origin does not silently lose points.
- `BsplineSpace.boundary_dofs`: the control-point indices of a boundary slab of
  chosen thickness, on a chosen face.
- `BsplineSpace.cell_supports` and `BsplineSpace1D.first_basis_per_interval`:
  which basis functions are supported on which cells, and the index of the first
  one per knot interval. Both are cached and computed in exact integer
  arithmetic, with no basis evaluation.
- `THBSplineSpace.prolongation_to_sparse`: the CSR counterpart of
  `prolongation_to`, assembled column by column so the dense operator is never
  built. Storage grows linearly with the coarse space where the dense form grows
  quadratically (measured over a refinement sweep: 12.9x against 120.6x).
- `THBSplineSpace.max_active_per_cell`, `HierarchicalGrid.export_cells` and
  `Grid.boundary_facets`: the largest number of active functions on any cell, a
  deduplicated vertex/cell export for downstream meshes, and the boundary facets
  of a grid, with an axis-aligned default on the base class.
- `SpanwiseElementExtraction.factors`: per-cell access to the univariate
  extraction factors, as views sharing memory with the stored operators.
- `pantr.change_basis`: `compute_legendre_to_cardinal_1d`,
  `compute_cardinal_to_legendre_1d` and
  `compute_cardinal_dual_legendre_coeffs_1d`, completing the Legendre-cardinal
  pair. The transforms are obtained from an LU solve on the cardinal matrix
  rather than from its Gram matrix, which would square the conditioning; the
  docstrings state the accuracy actually attainable per degree instead of
  promising a fixed bound.
- `pantr.bspline.dof_owner_windowed`: owner lookup restricted to a window of
  degrees of freedom, for the distributed layer.
- `Bezier.degree_reduction_error`: the exact $L^2$ norm of the error
  `Bezier.reduce_degree` would introduce, computed through the Bernstein mass
  matrix rather than by sampling.

### Changed
- Knot insertion runs the Oslo recurrence over the `degree + 1` band each row is
  supported on, instead of over the full row. Refining a spline with 10⁴ control
  points dyadically takes 1.9 ms where it took 1432 ms. The refinement matrix
  returned by the internal dense entry point is unchanged bit for bit, so the
  THB, spline-product and periodic-conversion paths are untouched; inserted
  control points sum the same products in a different order and can move in the
  last bit.
- `dof_owner` and `compute_halo` are compiled. Measured on 10⁶ control points in
  3D at degree 3: 68 ms and 168 ms. They stay single-threaded on purpose — the
  Numba workqueue threading layer is not threadsafe against the warmup thread
  the package starts at import.
- The span search shared by the basis and derivative kernels is fused into the
  parallel loop instead of running as a serial pass before it. It dominated the
  runtime, so multi-core speed-up was capped near 1.3x by Amdahl's law.
- `Bezier.reduce_degree` and `Bspline.reduce_degree` now interpolate the
  endpoints of every segment: among the lower-degree polynomials that reproduce
  the original at both ends of the parametric domain, the result is the closest
  in $L^2$. Reduced control points therefore differ from previous releases on
  any reduction that is not exact. In exchange, adjacent B-spline segments meet
  at breakpoints bit for bit and the previous averaging of the shared control
  point — which moved *both* segments off their own optima — is gone. Without
  the endpoint conditions the $L^2$ error would be a factor 1.1 (degree 16) to
  4.5 (reduction to a straight line) smaller. Reduction to degree 0 keeps no
  endpoint condition and still returns the mean of the control points.

### Fixed
- Knot comparisons went through `np.isclose(a, b, atol=tol)` at 26 sites across
  seven modules. Setting `atol` does not clear `rtol`, which stays at NumPy's
  default `1e-5`, so the effective test was `|a - b| <= tol + 1e-5 * |b|`: on a
  domain of length 1 placed at `1e6` the tolerance was 10, not the `1e-15` the
  space reports. The consequences were a periodic space reporting the wrong
  number of basis functions, the in-domain gate admitting points ten domain
  lengths outside, `remove_knots` removing a different knot than the one asked
  for, `split` and `restrict` returning a piece that no longer interpolated its
  own end control point, and the exact extraction operator being discarded in
  favour of the identity on a knot vector that was near-uniform but not uniform.
  On a *translated* periodic domain it was also memory-unsafe: degree elevation
  indexed out of bounds, unchecked under `nopython`. All 26 now compare
  absolutely.
- Degree operations had no upper bound, while the exact-integer binomial
  recurrence they rely on overflows `int64` at `n = 62` — silently, since
  `nopython` does not trap it. `Bspline.elevate_degree`, `Bezier.elevate_degree`
  and Bézier composition returned corrupted control points from that degree on,
  where degree elevation must leave the curve pointwise unchanged. Two paths
  nobody would associate with degree elevation reach the same recurrence and are
  now capped too: `Bspline.derivative` on a rational spline, and
  `Bezier.minimize_degree`.
- Evaluating a 1D `Bspline` at points shaped `(n, 1)` — the shape the
  `(n_pts, dim)` convention implies, and the one this project's own docstring
  example uses — raised a Numba typing error from inside the kernel instead of
  being accepted. Points are now normalised before they reach any kernel, on all
  four entry points, and array-likes are coerced rather than failing on a
  missing `dtype` attribute.
- `SpanwiseElementExtraction.apply` and `apply_transpose` accepted an `out=`
  array aliasing their input and returned a silently wrong result; the alias
  check existed only for the two bilateral op kinds. `idx_map` was checked
  against its upper bound only, and only for non-identity entries, so a negative
  index reached the kernel and read out of bounds.
- `get_cardinal_intervals` on a degree-0 space read past the end of an empty
  array. `BVH` accepted node arrays deeper than the fixed traversal stack its own
  query kernels assume, and node arrays whose leaf markers and child pointers
  disagreed; both are rejected at construction now.
- `Bspline.reduce_degree` wrote past the end of its output buffers whenever the
  reduced Bézier form needed more control points than `len(knots)` allowed —
  from 13 elements on at degree 4, 8 at degree 5, 5 at degree 8, on the open,
  periodic and tensor-product paths alike. Unchecked under `nopython`, it
  surfaced as a control-point/basis-count mismatch from the `Bspline`
  constructor.
- The Bernstein evaluation recurrence seeded from `(1 - u)^p`, which underflows
  to zero for `u` close enough to 1 from degree 21 in float64 (degree 7 in
  float32); every later term is a positive multiple of the seed, so the basis
  summed to 0 instead of 1. The recurrence now starts from whichever endpoint
  keeps the seed above `0.5^p`. Degrees below the proven-safe bound keep the
  original branch-free loop, so the benchmark-gated tabulation is unaffected.
- The Cox-de Boor denominator guard compared a knot difference against an
  absolute tolerance, which is scale-dependent: on a small enough domain a
  genuinely non-empty knot span fell below it and was zeroed, breaking the
  partition of unity. The guard now tests against exact zero, which is
  scale-invariant by construction because knot vectors already snap
  near-duplicate knots to one bitwise value.
- `find_roots` reported a root at every interior knot of multiplicity
  `degree + 1` whose two straddling coefficients change sign, where the spline
  is C^-1 and jumps across the axis without ever reaching it. The tracking cited
  Mørken-Reimers Lemma 3, whose conclusion `c[a] = 0` rests on the iterate lying
  in the half-open interval between two consecutive Greville abscissae, and that
  interval is empty at precisely that multiplicity: the secant through the two
  coefficients is vertical there, so its zero is the knot for every `lambda` and
  nothing forces the coefficient to vanish. The fabricated root then took a real
  one with it, since reporting it split the spline at the jump and pinned the
  coefficient on the far side to zero as the split barrier, destroying the sign
  change that bracketed the next zero. On a quadratic split once at C^-1 the
  reported set was `[0.40269975, 0.5]` where the zeros are `0.40269975` and
  `0.61721778`. The lemma's hypothesis is now tested on the knot vector, which
  needs no tolerance, and a jump is rejected by the same residual test that
  already separates a tangential zero from a false sign change.
- `get_tanh_sinh_1d` returned `inf` for `1/sqrt(x)` — the integrand a
  double-exponential rule exists for — at every `n_pts >= 45`, and raising the
  point count therefore turned a correct answer into `inf`. A node whose
  distance to the endpoint had underflowed was moved *onto* the endpoint and
  kept, with a nonzero weight, and from `n_pts = 53` a second node landed on the
  same boundary and was returned twice. The rule is now truncated there instead:
  generation stops at the last node whose gap survives the mapping onto `[0, 1]`
  in the requested dtype, which is one machine epsilon. The discarded weight is
  at most `pi * cosh(t) * gap`, measured at `8.2e-15` against a weight sum of
  `2`, so a smooth integrand is unchanged; a singular one now converges to the
  truncation floor, `2e-8` for `x**-0.5` and `4e-15` for `log(x)`, both stated
  in the docstring. Returning fewer nodes than requested was already documented.
  In `float32` the cast onto `[0, 1]` collapsed a node onto `1.0` from
  `n_pts = 19`, which the dtype-aware threshold also closes.
- `QuadratureRule` claimed its factory-built rules integrate the constant `1`
  exactly. They do not, and cannot: dividing by a computed sum leaves the
  rescaled weights summing to `1` only up to rounding. Measured over Gauss-
  Legendre rules of 1 to 40 points per direction, `|sum - 1|` reaches 2 ulp in
  1D, 3 ulp in 2D and 4 ulp in 3D. The docstring now says that.
- Lagrange tabulation was not reproducible between processes. SciPy's
  `BarycentricInterpolator` permutes the nodes before forming the barycentric
  weights, which is Berrut and Trefethen's remedy against the product over- or
  underflowing, and it was built without the `rng` argument, so the permutation
  came from the unseeded global NumPy state. Two runs of the same call differed
  by 1.6e-16 to 1.5e-15 relative at degrees 3 to 12, by 4.18 absolute on a value
  scale of 3.75e7 at degree 62, and by `inf` against 1e16 evaluated outside
  `[0, 1]`. That is an unseeded generator, not floating-point nondeterminism
  with a bound. `tabulate_lagrange`, `compute_lagrange_to_bernstein_1d`,
  `tabulate_Lagrange_extraction_operators` and `SpanwiseElementExtraction` with
  a Lagrange target all inherited it. The permutation is now seeded from a
  recorded constant, which also makes the `degree + 1` cardinal functions share
  one set of weights instead of drawing their own.

## 0.6.0 (2026-06-24)

### Added
- `pantr.mpi.configure_threads`: explicitly set the per-rank Numba thread count
  for hybrid MPI + threads runs (optionally also limiting BLAS/LAPACK thread
  pools via `threadpoolctl`). Calling it disables the default MPI
  thread policy.

### Changed
- MPI-engaging entry points (`pantr.mpi.DistributedSpace`,
  `pantr.mpi.from_dolfinx`) now apply a process-level default on first use: the
  Numba thread pool is limited to one thread per rank (flat MPI), preventing
  `n_ranks x n_cores` thread oversubscription when running under `mpiexec`.
  Explicit configuration always wins (`NUMBA_NUM_THREADS`,
  `pantr.set_num_threads`, `pantr.num_threads`, or
  `pantr.mpi.configure_threads`), and the policy is applied at most once per
  process, so raising the count afterwards sticks.

### Removed
- Dropped the `THIRD_PARTY_NOTICES` file and its `pyproject.toml` `license-files`
  entry. The routines that previously followed algoim — Bézier/B-spline degree
  reduction, Bernstein interpolation, the Bernstein L2 norm and degree
  minimization, and the tanh–sinh rule with its Lambert W step — were
  reimplemented clean-room from public references (The NURBS Book; Farouki &
  Rajan, *CAGD* 1988; Golub & Van Loan; Takahasi & Mori 1974; SciPy), with no
  change in public API or numerical results, so the third-party attribution is
  no longer required.

## 0.5.1 (2026-06-03)

### Added
- `pantr.grid.overlay`: build the coarsest `TensorProductGrid` that refines two
  input tensor-product grids — its per-axis breakpoints are the union of both
  inputs' breakpoints restricted to their domain overlap, so every overlay cell
  lies inside one cell of each input. Defined for any `ndim >= 1`. The
  background-grid bridge for immersed / unfitted quadrature.

## 0.5.0 (2026-06-03)

### Added
- `pantr.grid`: new structured-grid layer. `Grid` is an abstract base class
  defining the grid contract (cell bounds, point location, facet neighbours)
  with axis-aligned box defaults for facets, reference maps, neighbour lists,
  batch point location, and AABB queries. `TensorProductGrid` is a concrete,
  low-footprint tensor-product grid of axis-aligned boxes with per-axis
  breakpoints and row-major (C-order) cell ids matching
  `SpanwiseElementExtraction`. Factories `uniform_grid` and `tensor_product_grid`
  build a uniform grid and a B-spline knot-span grid respectively. `BVH` is a
  bounding-volume hierarchy over cell AABBs (lazily built, backing
  `Grid.query_aabb`), and `CellTags` / `FacetTags` are sparse, dolfinx-style
  named tag registries for cells and facets.
- `pantr.grid.HierarchicalGrid`: hierarchical refinement grid with a fixed
  per-direction subdivision factor (octree = the dyadic case). Active cells are
  stored as rectangular blocks per level (no per-cell storage); supports
  `refine(level, lo, hi)` with union semantics, automatic single-level balance,
  `refine_cells`, and `hanging_neighbors` for hanging-node facets. Built with the
  `hierarchical_grid(root, factor)` factory.
- `pantr.viz.grid_to_pyvista`: export a 1-D/2-D/3-D `Grid` to a pyvista
  `UnstructuredGrid` (lines / quads / hexahedra).
- `pantr.quad.QuadratureRule`: immutable d-dimensional quadrature rule on the
  unit cube `[0, 1]^ndim`, with `tensor_product_quadrature` (tensor product of
  per-axis 1-D rules) and `gauss_legendre_quadrature` (isotropic or anisotropic
  Gauss-Legendre) factories.
- `pantr.grid.cell_quadrature`: map a reference `QuadratureRule` from the unit
  cube onto a grid's cells (or a subset), returning per-cell points
  `(num_cells, num_points, ndim)` and weights `(num_cells, num_points)` via the
  per-cell affine map with volume-scaled weights. The uncut/background-cell
  quadrature bridge for immersed / unfitted discretizations.

## 0.4.0 (2026-06-02)

### Added
- `pantr.geometry`: new module exposing `AABB`, an immutable, general-*d*
  axis-aligned bounding box (#153). Shared domain primitive for spline-space
  parametric domains and grid-cell bounds; decoupled from any concrete affine
  transform via a structural `_AffineMap` protocol.

### Changed
- `pantr.transform.AffineTransform`: stricter input validation (reject
  zero / non-finite scaling factors; validate rotation-axis and mirror-normal
  finiteness and the `center` shape), a cached `inverse`, and C-contiguous
  stored arrays (#154). Enables lepard to adopt pantr's `AffineTransform` and
  drop its local copy.

## 0.3.0 (2026-05-06)

### Added
- `SpanwiseElementExtraction` class providing a unified interface for element-wise
  extraction operators across B-spline spaces (#143).
- Batch apply methods on `SpanwiseElementExtraction` for vectorized evaluation (#145).
- Numba-callable Kronecker kernels backing tensor-product extraction (#140).
- Structural identity predicate for Bezier and Lagrange extraction operators (#147).
- Numba-callable struct-view of `SpanwiseElementExtraction` for downstream JIT code (#149).
- Python 3.13 and 3.14 are now officially supported (#151).
- User guide for `SpanwiseElementExtraction` (#146).

### Changed
- `nD` Bezier extraction is now routed through `SpanwiseElementExtraction`,
  unifying the 1D and multi-dimensional code paths (#144).

### Performance
- Compact storage for identity-heavy extraction spaces (#148).
- CI pipeline and test suite sped up (#150).

## 0.2.0 (2026-04-19)

### Added
- `pantr.bezier`: mask / boolean-array operations (#112), Sylvester and Bezout matrix
  construction (#113), determinant and rank via Givens-rotation QR (#115),
  `Bezier.interpolate` / `Bezier.fit` classmethods (#118, #120),
  resultant / discriminant / `minimize_degree` (#121), pure-Numba implicit quadrature
  module (#128, #130, #131), implicit domain reparameterization with Lagrange cells
  (#132).
- `pantr.bspline`: interpolation, fitting, and L2 projection (#122, #124).
- `pantr.quad`: modified Chebyshev nodes for Bernstein interpolation (#114),
  tanh–sinh quadrature rule (#116).
- `pantr.root_finding`: Bernstein polynomial root-finding module, with a unified
  single / batch API (#111, #123). First contribution by @DavorDobrota.

### Changed
- Public function renames in `bspline`, `bezier`, and `cad` for clarity (#125, #126).
- Conversion classmethods extracted as standalone module functions in `bspline`
  and `bezier` (#127).
- `change_basis`: added `compute_monomial_to_bernstein_1d`, reused across
  `bezier` and `bezier.implicit` (#135).
- `bezier`: `_gauss_legendre_01` now delegates to `pantr.quad` (#133).
- Layer 2 validation helpers consolidated and shared across `bezier` / `bspline`
  (#138, #139).
- `bezier.implicit`: legacy algoim engine moved out into `lepard.algoim`,
  and dead algoim-era modules dropped (#136, #137).

### Documentation
- Added algoim attribution and third-party notice for the implicit quadrature
  module (#134).

## 0.1.0 (2026-03-24)

- Initial release: project scaffolding, tooling configuration, and
  documentation skeleton, plus the core `basis`, `bspline`, `bezier`, `quad`,
  `change_basis`, `cad`, and `viz` modules.
