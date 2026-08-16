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

### Rejected where it used to be accepted
- **A knot vector whose spacing is finer than its dtype resolves at its own coordinate
  magnitude is now refused at construction**, where it used to be accepted and then
  silently behave as if it had no intervals. `BsplineSpace1D` raises a `ValueError`
  naming the dtype, the coordinate magnitude, the merge tolerance in ulp and the closest
  pair of knots you supplied, and suggesting float64, a domain nearer the origin, or a
  coarser mesh. Concretely: float32 on `[1e6, 1e6 + 1]` with more than one interval. The
  ulp there is 0.0625, so a four-interval mesh spaces breakpoints 0.25 apart while
  telling two knots apart at that magnitude needs 0.48 — an interior knot computed by any
  route is uncertain by 6% of the window, and no tolerance both keeps the mesh and merges
  two routes to the same knot. The previous release kept the mesh only by never merging
  anything (its tolerance there was 1.6e-6 ulp), so the knots it stored were noise.
  Reachable in float64 too, but only from `|offset| / span ≈ 1.4e14` at four intervals,
  where the window is 8 ulp wide. Two things are deliberately *not* refused: a knot vector
  that was already a single repeated value, which is what the caller asked for and comes
  back untouched, and anything built with `snap_knots=False`, which bypasses the merging
  and the check together.

### Changed
- **`pantr.tolerance`'s presets are now dimensionless relative tolerances.**
  `get_strict`, `get_default` and `get_conservative` return `K * eps(dtype)` with
  `K` equal to 4, 64 and 4096 — an acknowledged safety factor, stated, a power of
  two, and the *same* in every precision. The previous table was a fixed value per
  dtype, and in units of that dtype's own epsilon it was neither constant nor
  ordered the same way: strict was 0.10 eps in float16, 0.84 in float32 and 4.5 in
  float64, so `get_strict` demanded bitwise equality in the two smaller formats
  while allowing four roundings in the largest. The module docstring called the
  presets absolute while `Bspline.locate` and both root finders were already
  multiplying them by a magnitude; that ambiguity is what this change removes. A
  preset is now unusable on its own — multiply it by the magnitude of the quantity
  being compared, and name that magnitude at the call site. Numerically:
  `get_default(float64)` moves from 1e-12 to 1.42e-14 and `get_conservative`
  from 1e-10 to 9.09e-13, while `get_strict(float32)` moves from 1e-7 to 4.77e-7.
  The per-dtype table and its platform branch on whether `longdouble` aliases
  `float64` are gone: reading the epsilon from the platform makes both unnecessary.
- **`BsplineSpace1D.tolerance` now carries the knot vector's own magnitude.** It is
  `8 * eps(dtype) * max(span, |knots[0]|, |knots[-1]|)` — the strict preset,
  doubled for one extra rounding upstream, scaled by the parametric magnitude every
  comparison in the B-spline layer is relative to. Previously it was the bare
  per-dtype constant, so the same absolute band meant four ulp on a unit domain,
  nothing at all from `|knot| ≈ 5` upward, and half the breakpoints of a
  twenty-interval float32 mesh on `[0, 1e-6]`, which then reported ten. Everything downstream that takes `space.tolerance` — domain membership,
  knot matching, cardinal interval lengths, knot insertion and removal, products,
  degree changes — becomes scale-covariant with it. `detect_interfaces` is the one
  caller that wanted the dimensionless factor instead and now takes it from
  `pantr.tolerance` directly, since none of the three quantities it compares is a
  parametric coordinate of a single patch.
- **The individual sites that still carried a bare constant now derive one.** The
  presets above made a tolerance mean something; these are the places that could not be
  re-derived until it did. Each grades a named quantity, scales by that quantity's own
  magnitude, and states its safety factor in an attribute docstring:
  - **The THB refinement gate** certified a candidate whenever the worst prolongation
    residual stayed under `1e-8 * (1 + max_coarse_value)`. The shape was right — the
    residual is compared against the size of the coarse column it reproduces — but the
    constant was wrong by seven decades, so an impostor whose residual landed anywhere in
    `[1e-14, 1e-8]` was certified and `prolongation_to` returned a matrix that does not
    prolong. It is now `4096 * eps * (1 + max_coarse_value)`, the conservative tier
    standing in for the backward-stability constant of the `gelsd` least-squares solve,
    which has no closed form. Measured over 132 configurations (1D/2D/3D, degrees 2–5,
    truncated and not, one and two refinement levels, three domain scales) the worst
    residual is 18.5 eps, so the gate keeps a safety factor of 443 while tightening
    1.1e4-fold. A `sqrt(M)` growth term was tried and dropped on measurement: `M` spans
    13 to 19683 with no trend in the residual.
  - **THB cell membership** allowed a fixed `1e-12` of slack at a cell boundary, which
    rejected a point one ulp outside once the parametric domain reached 1e6 and accepted
    some 9000 ulp of slop on `[0, 1]`. It is now `8 * eps * max(|lo|, |hi|, hi - lo)` per
    axis: the same rule, and the same number, the knot layer uses for the same question.
  - **The Coons corner check** decided a `ValueError` with `atol=1e-12, rtol=0`. Across
    twelve decades that rejects a one-ulp corner mismatch at model size 1e6 — a
    micron-unit model of a metre-scale part failing on a geometrically perfect patch —
    while accepting a 1e-9 *relative* gap at 1e-6. It is now `4096 * eps` times the
    largest absolute coordinate over **all eight** corner values, so the verdict does not
    depend on which corner sits at the origin.
  - **`Bspline.locate`'s geometric scale** no longer floors at 1.0. The floor held a
    millimetre-scale part to a thousand times the relative accuracy of a metre-scale one;
    the fallback to 1.0 now fires only for a geometry with no extent *at the origin*,
    which is the one case with no scale to read. The docstring's account of the default
    tier is corrected with it: the `60 * eps * scale` it cited as a six percent margin was
    a censored measurement, since the iteration stops at the threshold. Re-measured at six
    thresholds from 2 to 4096 epsilons over 9480 query points, the achieved residual
    tracks whatever threshold is set and nothing is lost even at `4 * eps * scale`, so the
    threshold is an accuracy contract rather than a safety margin, and loosening the tier
    would cost accuracy one for one.
  - **`multipatch`'s joint geometric tolerance** was `relative * diagonal`, which tracks
    how big the geometry is but not where it sits: two unit patches meeting at 1e6 were
    graded at 2.5e-18, eight orders below their own noise floor. It now takes
    `max(diagonal, largest |coordinate|)`.
  - **`Bspline.restrict`** gains no constant, which is its policy and is now written
    down: every comparison it makes is a knot-identity question that
    `BsplineSpace1D.tolerance` already answers, and the module docstring records the three
    properties making one number enough — insertion preserves the endpoints, the
    extraction offsets widen away from the interval, and the snapping invariant stops the
    widening over-collecting.
- **`Bspline.remove_knots` is now lossless by default.** `tol=None` used to mean an
  absolute `1e-10`, a geometric budget wearing no units: the same exactly-removable knot
  came out at geometry scale 1e-6, 1 and 1e3 and was silently refused at 1e6 and 1e9. It
  now means *remove only what is exact to round-off*, `8 * (degree + 1) * eps *
  max(bbox diagonal, largest |P_i|) * sqrt(n)`, which is the one default needing no
  arbitrary constant. Measured over 105 cases (degrees 1–7, ranks 1–3, geometry scales
  1e-6 to 1e9) by bisecting for the smallest budget that still removes a just-inserted
  knot: the largest distance observed is 0.84 eps × scale, and the tightest margin
  anywhere in the set is a factor of 48. The `sqrt(n)` is what the *other* directions
  add: on a surface or a volume the removal test is handed a slice with every other
  direction's control points flattened into it, so its single Euclidean distance spans
  `n` control points rather than one, and stacking `n` blocks each of norm at most `e`
  reaches `sqrt(n) · e`. Without that term the default silently refuses an exactly
  redundant knot once `n` is large enough — measured on a degree-3 volume, from
  `n = 124609` upward. **This is a behaviour change** — a knot whose removal costs up to
  1e-10 of geometry used to come out and no longer does. Trading geometry for a shorter
  knot vector is a price only the caller can set, so it has to be asked for.
- **A rational knot-removal budget is converted before it reaches the kernel.** The kernel
  measures a Euclidean distance over every column of the array it is given, which for a
  NURBS is the homogeneous `[w·x, w·y, w·z, w]`, while the caller's budget is a distance
  in projected space. Eq. (5.30) of Piegl & Tiller (2nd ed., 1997, p. 185),
  `TOL = d · w_min / (1 + |P|_max)`, is now applied. Without it a deviation carried by the
  **weight** column is graded as though it were a coordinate: sweeping a weight
  perturbation over sixteen decades with the coordinates at 1e6, the worst projected
  deviation an accepted removal produced was 7.4e5 times the requested budget; with the
  conversion, 0.32. A circle cannot show this, its weights being structurally tied to its
  coordinates, so the tests vary weights and coordinates independently.
- `pantr.grid.overlay` merges breakpoints closer than the default *relative*
  tolerance times the axis's own magnitude -- the intersection window and the
  coordinates bounding it -- instead of against a bare `float64` constant. A
  window of length 1 sitting at 1e6 resolves its breakpoints no better than
  `eps * 1e6` however short it is, and the merge verdict is now the same on
  `[0, 1]` as on any rescaling of it.
- The minimum supported SciPy is now 1.15, raised from 1.11. Lagrange tabulation
  seeds the node permutation SciPy's `BarycentricInterpolator` applies, and the
  `rng` argument that makes that possible arrived in 1.15. Earlier versions
  cannot be made reproducible through that class at all — 1.11 accepts neither a
  seed nor precomputed weights — so the previous floor was already untrue for
  this function rather than merely untested.
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
- Knot snapping merged the wrong knots, and from `|knot| ≈ 5` upward merged none
  at all. `BsplineSpace1D._snap_knots` rounded onto a grid of width `tolerance`,
  which was an absolute per-dtype constant: at knot 1.0 the grid was 4.5 ulp wide
  and it narrowed relative to the knots as they grew, so two knots one ulp apart
  stopped merging at base 5 and the detected multiplicity dropped — a cubic asked
  for multiplicity 2 (C¹) silently became C². A rounding *grid* is also wrong in
  principle: two values one ulp apart can straddle a grid line and never merge,
  for any spacing. Knots are now grouped by relative gap — a class ends where the
  step to the next knot exceeds `8 * eps * max(span, |knots[0]|, |knots[-1]|)`,
  three roundings of an affine map of the span with 2.7x over it — and every
  member of a class takes the class's *first* knot. Choosing rather than averaging
  makes snapping idempotent and keeps the stored values ones the caller supplied;
  `np.mean` over `degree + 1` identical copies is not even the identity at large
  magnitude, and it was moving the reported domain (`1000001.0` came back as
  `1000000.875` in float32).
- `get_unique_knots_and_multiplicity` reported knots the space does not contain.
  The helper behind it rounded the knots onto the tolerance grid to decide which
  were the same and then returned the *rounded* values, although the indices it
  had built already pointed at the originals. On a float32 space over `[0, 1e-6]`
  with interior knots at 2.5e-7, 5.0e-7 and 7.5e-7 it answered 2.0e-7, 5.0e-7 and
  8.0e-7. `Bspline.multiply` builds the product's breakpoint mesh from this
  accessor, so squaring such a spline returned a curve wrong by 8% relative at
  points nowhere near a knot; `to_beziers`, knot insertion and removal and
  quasi-interpolation all read the same helper. The accessor now returns stored
  knots, and `_snap_knots` is implemented on top of it, so the space and its own
  accessor can no longer disagree about which knots are the same knot.
- `pantr.bspline.find_roots` returned values that are not roots, silently, on
  ordinary input: a clamped cubic on `[0, 1]` with control points alternating
  `+1, -1` gave back `0.375` and `0.625`, where the spline is `0.0208` rather
  than zero. Its residual test only ever ran on one of the four ways the
  tracking of a sign change can stop; on the other three the residual was
  hard-coded to zero and the value accepted unconditionally, and those are the
  branches that fire when the iteration is in trouble. Every exit now evaluates
  `|f(x)|` and the same test decides all of them, so a status records how the
  iteration stopped and never whether the value may be reported. Two further
  changes were needed to keep that from costing genuine roots. The stop on a
  repeated iterate now tests the hypothesis Mørken and Reimers actually state in
  their Corollary 14 — `degree - 1` active knots collapsed onto the iterate —
  rather than the bare repetition, which is also what a nearly horizontal
  control-polygon secant produces at a shallow non-zero minimum. And the
  threshold a tracked iterate is tested against now carries the term the
  parametric tolerance itself allows, `|f'| * 2 * tol * scale`, alongside the
  evaluation error: testing the evaluation error alone rejects the correctly
  located zeros of a steep spline. A side effect worth knowing: from degree 20 a
  double root came back as the two sign changes bracketing it, so a caller
  counting zeros got two where the answer is one. One of the pair was an
  uncertified budget-limited iterate and is now dropped, leaving the count right.
- The merge that collapses several reports of one zero could return a point
  between two distinct zeros. Reports are joined into a run on the *larger* of
  two merge radii, and the radius is capped at
  `domain_length * (degree! * zero_tol / coeff_scale) ** (1 / degree)`, which
  *grows* with degree — 0.114 at degree 9, 0.767 at 15, past the whole domain
  from 17 — so at high degree one radius joined every later root into its run.
  On a degree-15 spline with a C⁰ interior knot the three zeros found, each with
  a residual of 1e-17, came back as their midpoint alone, where the spline is
  `-0.64`. A merged midpoint is a value nobody tracked, so it now takes the same
  residual test as the reports it replaces, and a run whose midpoint fails is
  left as the separate roots it was. The radius policy itself is unchanged.
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
  scale-invariant by construction. What makes the exact test safe is the shape of
  the recurrence, not the knot vector: the denominator is the sum of the two
  non-negative terms the step then multiplies the quotient by, so the two weights
  are a convex combination and a tiny denominator cancels against an equally tiny
  numerator. (An earlier version of this entry justified it by claiming that knot
  snapping stores equal knots bitwise identical. Snapping does that, but the guard
  never needed it: with snapping off and two interior knots separated by 0 to
  64 ulp, `max|N|` is 1.0 exactly and the partition of unity holds to 6.7e-16.)
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
- `Bspline.elevate_degree` and `Bspline.reduce_degree` were unusable on every
  float32 spline. Both kernels allocated the two halves of their return with
  different dtypes: the control points followed the input while the knot vector
  was hardcoded float64. On a clamped space the `Bspline` constructor then
  rejected the mismatched pair with a message about the *caller's* control
  points, at every degree and every knot count; on a periodic space the round
  trip through open form converted the control points as well, so the call
  succeeded and silently discarded the caller's choice of precision, doubling
  memory and halving throughput without a word. Both knot outputs now follow the
  input knot vector's dtype.
- `Bspline.reduce_degree` never returned on a degree-1 periodic spline. Degree
  reduction hands the periodic conversion `m_bdy - decrement` as the target
  boundary multiplicity, and a maximally smooth periodic space has `m_bdy = 1`,
  so the argument is 0 at every degree; the conversion documents
  `1 <= m_bdy <= degree` and nothing checked it. The ghost-knot builder then had
  a per-period tile of total multiplicity 0 and its right-hand loop, which
  appends until it has enough ghost knots, could append nothing and incremented
  its shift forever. Degrees 2 and 3 escaped only because the C^0 seam check a
  few lines earlier rejected them first, so the trigger's narrowness was an
  accident of check ordering rather than anything about degree 1. The
  precondition is now enforced, so every degree refuses in under 10 ms with a
  message naming the actual problem; the loop's termination argument is written
  down beside it. Reducing a periodic spline to degree 0 has no answer in this
  representation, an empty `[1, degree]` meaning no ghost knots and nothing to
  wrap; `to_open_bspline().reduce_degree(1)` does it in the open form.
- `Bspline.multiply` built the product space from `max(m_f + q, m_g + p)` at a
  breakpoint both operands share and from a hardcoded `p + q` at one only a
  single operand carries. The second form assumes the present operand has
  multiplicity `p`, a hypothesis the module never stated, and it is wrong in both
  directions. Against a discontinuous operand (multiplicity `degree + 1`, which
  `subdivide(n, regularity=-1)` produces) the knot vector came out one knot
  short, the intermediate Bézier space did not contain the product, and the
  underdetermined Oslo system returned a least-squares vector unrelated to it:
  multiplying such a spline by the constant 1 gave a different function.
  Exhaustive over `p, q <= 3` with multiplicities up to `degree + 1`, 45 of 81
  configurations were wrong, every one of them with a discontinuous operand.
  Below multiplicity `p` the same line asked for up to `p - m_f` knots more than
  the product needs, so the ordinary continuous case now yields a smaller space
  as well: degree 4 times degree 1 across a C^3 knot builds 8 control points
  where it built 11. The rule that replaces both, minimal by Curry-Schoenberg, is
  `(p + q) - min(s_f, s_g)` with `s = degree - m` the smoothness of an operand
  there and `s = +inf` where it has no knot at all. It now lives in one function
  that the 1D and tensor-product paths both call.
- `Bspline.elevate_degree` returned control points and a knot vector that
  disagreed on how many basis functions the elevated space has, by one per
  interior knot of multiplicity `degree + 1`. Piegl-Tiller A5.9 starts each
  segment at its second elevated coefficient because the segment before it
  already wrote the shared junction value; where the spline is discontinuous the
  two segments share nothing and that coefficient was dropped. At degree 0 a
  second cause compounded it: the segment sweep relied on the closing block of
  `degree + 1` equal knots to reach the end of the knot vector, which at degree 0
  is a single knot, so the last segment and the closing knots were never written
  and the elevated spline came back on a domain collapsed to a point, without an
  error.
- `Bspline.derivative(direction, keep_degree=True)` raised
  `ValueError: The number of control points must be a multiple of the number of
  basis functions` for every spline with a C⁰ interior knot, which is the
  ordinary Bézier-element mesh rather than an exotic input. It differentiates to
  `degree - 1` and elevates back, so a knot of multiplicity `degree` becomes one
  of multiplicity `(degree - 1) + 1` in the space handed to the elevation kernel:
  the discontinuous case fixed above. It now returns the derivative, which is
  itself discontinuous there, carrying multiplicity `degree + 1`.
- `Bspline.reduce_degree` fitted a discontinuous knot into a C^0 space. Reduction
  preserves smoothness, so multiplicity `m` becomes `m - t`; the target was
  clamped at `new_degree`, one short of the `new_degree + 1` a jump needs, and
  the kernel underneath merged the two sides of every junction into one control
  point on the strength of an endpoint-interpolation argument that holds only
  where the original spline is continuous. On input built to be exactly
  reducible, the reduced curve now reproduces it to 4.4e-16 or better at degrees
  2 to 4, where the errors were 5.7e-01, 1.0e+00 and 4.9e-01.
- `Bspline.reduce_degree` on a maximally smooth periodic spline asked the
  periodic conversion for a seam multiplicity of `m_bdy - decrement`, which is 0,
  while interior breakpoints three lines away were already floored at 1. It now
  uses the same floor, which asks for less smoothness than the seam already has
  and is therefore always representable, and a maximally smooth periodic
  quadratic reduces to degree 1. From degree 3 the call still fails, because the
  segment-wise reduction does not preserve C^1 across the seam, but on the
  residual test that says so rather than on the knot vector. The degree-0
  rejection above is unaffected and now fires only where its reasoning applies.

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
