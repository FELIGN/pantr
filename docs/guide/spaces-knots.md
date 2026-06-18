# Spaces, knots & element extraction

This chapter follows one thread end to end: from the **knot vector** that defines a
spline space, through the **representation changes** that rewrite a geometry without
altering it, to the **element extraction** operators that hand each element to a
finite-element / isogeometric assembly loop. It assumes the vocabulary from
{doc}`/guide/concepts`.

## Knot vectors

A {class}`~pantr.bspline.BsplineSpace1D` is a non-decreasing **knot vector** plus a
**degree** ``p``. The knots partition the domain into spans (elements); on each span the
basis is a degree-``p`` polynomial, and the knot multiplicities control how the pieces
join (see [Continuity](#continuity) below). A {class}`~pantr.bspline.BsplineSpace` is a
tensor product of these 1-D spaces.

You rarely type knot vectors by hand. The factories in {mod}`pantr.bspline` build the
standard families:

| Factory | Builds | Notes |
|---|---|---|
| {func}`~pantr.bspline.create_uniform_open_knots` | an **open** (clamped) uniform knot vector | endpoints have multiplicity ``p+1`` so the curve is clamped to its end control points |
| {func}`~pantr.bspline.create_uniform_periodic_knots` | a **periodic** uniform knot vector | wraps smoothly; for closed curves/surfaces |
| {func}`~pantr.bspline.create_cardinal_knots` | a **cardinal** (uniform B-spline) knot vector | shift-invariant interior basis |
| {func}`~pantr.bspline.create_uniform_space` | a whole tensor-product {class}`~pantr.bspline.BsplineSpace` | per-direction ``degree`` and ``num_intervals``; the one-call convenience |

```python
from pantr.bspline import create_uniform_space

# Biquadratic surface space, 8 x 8 elements on [0,1]^2
space = create_uniform_space([2, 2], [8, 8])
space.num_basis            # (10, 10) basis functions per direction
space.num_total_intervals  # 64 elements
```

Each factory takes an optional ``continuity`` to raise interior knot multiplicities, and
{func}`~pantr.bspline.create_uniform_space` also takes ``periodic`` and ``domain``.

(continuity)=
## Continuity

Across an interior knot of multiplicity ``m``, a degree-``p`` spline is
``C^{p-m}`` continuous:

- ``m = 1`` (simple knot): ``C^{p-1}`` — the maximal smoothness of a uniform B-spline.
- ``m = p``: ``C^0`` — a kink is allowed; the curve still connects.
- ``m = p+1``: the geometry splits into independent pieces (this is exactly the clamped
  end of an *open* knot vector, and the basis of [Bézier extraction](#bezier-extraction)).

{meth}`~pantr.bspline.BsplineSpace1D.get_unique_knots_and_multiplicity` reports the
multiplicities; multiplicity comparisons use the space's {attr}`~pantr.bspline.BsplineSpace1D.tolerance`
(from {mod}`pantr.tolerance`).

## Greville abscissae

The **Greville abscissa** of a basis function is the average of its interior knots — the
parameter value naturally "attached" to its control point.
{func}`~pantr.bspline.get_greville_abscissae` returns them for a 1-D space, and
{func}`~pantr.bspline.create_greville_lattice` builds the tensor-product lattice for a
multivariate space. They are the default sites for interpolation
({doc}`/tutorials/05_approximation`).

## Representation-preserving changes

A central property: several operations change a geometry's *representation* — its space
and control points — while leaving the curve/surface/volume it describes **unchanged** (to
round-off). They are the workhorses of refinement and inter-operability. Knot insertion
follows {cite:t}`boehm1980knots` and the Oslo algorithm of {cite:t}`cohen1980oslo`;
degree elevation follows {cite:t}`prautzsch1984degree,prautzsch1991fast`:

| Operation | Method | Effect on representation |
|---|---|---|
| Knot insertion | {meth}`~pantr.bspline.Bspline.insert_knots` | adds knots + control points; refines the net |
| Knot removal | {meth}`~pantr.bspline.Bspline.remove_knots` | removes knots where smoothness permits |
| Degree elevation | {meth}`~pantr.bspline.Bspline.elevate_degree` | raises the degree, adds control points |
| Degree reduction | {meth}`~pantr.bspline.Bspline.reduce_degree` | lowers the degree (approximate if exact reduction is impossible) |
| Split | {meth}`~pantr.bspline.Bspline.split` | cuts into two geometries at a parameter |
| Restrict | {meth}`~pantr.bspline.Bspline.restrict` | extracts the geometry over a sub-box |
| Open / periodic | {meth}`~pantr.bspline.Bspline.to_open_bspline` / {meth}`~pantr.bspline.Bspline.to_periodic` | converts between clamped and periodic forms |

{doc}`/tutorials/03_knot_operations` demonstrates the geometric invariance directly
(evaluate before and after, compare).

(bezier-extraction)=
## Bézier extraction

Insert every interior knot up to multiplicity ``p`` and each element becomes an isolated
**Bézier patch** — a single polynomial in Bernstein form. This *Bézier extraction* is the
bridge between the smooth, globally-coupled spline basis and the element-local basis a
finite-element code assembles against {cite:p}`borden2011bezier,scott2011tsplines`.

{meth}`~pantr.bspline.Bspline.to_beziers` returns the per-element Bézier pieces (as an
array of {class}`~pantr.bezier.Bezier` objects); {meth}`~pantr.bspline.Bspline.to_bezier`
returns a single one when the geometry is already a single element. Conversely a
{class}`~pantr.bezier.Bezier` round-trips back with {func}`~pantr.bspline.create_from_bezier`.
The Bézier patches themselves are the subject of {doc}`/tutorials/07_bezier_and_roots`.

---

## Element extraction operators

```{admonition} Advanced
:class: note
The rest of this page is for code that assembles element matrices/vectors (IGA / FEM).
If you only need geometry, you can stop here.
```

{meth}`~pantr.bspline.Bspline.to_beziers` *materializes* Bézier geometries. When you
instead need the **change-of-basis operator** itself — to convert coefficients or to
pull element matrices between the spline basis and an element-local basis — use
{class}`~pantr.bspline.SpanwiseElementExtraction`. It builds the per-direction 1-D
operators once and applies the ``d``-dimensional Kronecker product matrix-free, never
forming the full tensor product in memory.

### Construction and targets

```python
from pantr.bspline import SpanwiseElementExtraction

ext = SpanwiseElementExtraction(space, "bezier")     # Bernstein/Bézier basis per element
ext = SpanwiseElementExtraction(space, "lagrange")   # Lagrange basis at chosen nodes
ext = SpanwiseElementExtraction(space, "cardinal")   # cardinal B-spline basis
```

| Target | Element-local basis |
|---|---|
| ``"bezier"`` | Bernstein / Bézier on each element |
| ``"lagrange"`` | Lagrange at a {class}`~pantr.basis.LagrangeVariant` node set (``lagrange_variant=…``) |
| ``"cardinal"`` | cardinal (uniform) B-spline |

Construction is ``O(n_elements · p²)`` per direction and happens once; all later applies
reuse the cached operators.

### Applying the operator

Let ``M`` be the extraction operator for one element. Four operation kinds cover the
bilateral pattern of element assembly, each with a single-cell and a batch (``_many``,
parallelized over cells) form:

| Method | Computes | Typical use |
|---|---|---|
| {meth}`~pantr.bspline.SpanwiseElementExtraction.apply` | ``y = M v`` | convert coefficients (spline → target) |
| {meth}`~pantr.bspline.SpanwiseElementExtraction.apply_transpose` | ``y = Mᵀ v`` | adjoint / dual conversion |
| {meth}`~pantr.bspline.SpanwiseElementExtraction.apply_MT_K_M` | ``B = Mᵀ K M`` | pull a target-basis element matrix back to the spline basis |
| {meth}`~pantr.bspline.SpanwiseElementExtraction.apply_M_K_MT` | ``B = M K Mᵀ`` | push a spline-basis element matrix to the target basis |

```python
import numpy as np

n_in  = int(np.prod(ext.input_shape_per_dir))
n_out = int(np.prod(ext.output_shape_per_dir))

v = np.ones(n_in)
y = ext.apply(v, cell=5)                 # one element, y.shape == (n_out,)

cells = np.arange(ext.num_total_intervals)
Y = ext.apply_many(np.ones((cells.size, n_in)), cells)   # all elements at once
```

Pass a pre-allocated ``out=`` (and ``scratch=``) array to avoid per-call allocation in
loops. A cell index may be a flat ``int`` (row-major over the elements) or a
per-direction tuple.

### Identity short-circuit

On many elements an extraction operator is exactly the identity (e.g. an element that is
already a Bézier patch, or a cardinal interval). The operator detects these *structurally*
— no floating-point comparison — and skips the multiply.
{attr}`~pantr.bspline.SpanwiseElementExtraction.is_identity`,
{meth}`~pantr.bspline.SpanwiseElementExtraction.is_identity_at`, and
{attr}`~pantr.bspline.SpanwiseElementExtraction.num_identity_elements` expose the result;
identity-heavy spaces also use a compact storage that saves memory.

### Materializing operators

When you do want the dense matrix, {meth}`~pantr.bspline.SpanwiseElementExtraction.operator`
assembles the ``(n_out, n_in)`` Kronecker product for one element and
{meth}`~pantr.bspline.SpanwiseElementExtraction.tabulate` returns the full
``(num_total_intervals, n_out, n_in)`` stack (this is what backs
{meth}`~pantr.bspline.Bspline.to_beziers`). Prefer the matrix-free applies in hot loops.

### Calling kernels from Numba

For code that is itself ``@njit``-compiled, the operator stacks
({attr}`~pantr.bspline.SpanwiseElementExtraction.ops_1d`,
{attr}`~pantr.bspline.SpanwiseElementExtraction.is_identity_mask_1d`, the compact arrays
and index maps) are plain read-only NumPy arrays, and the underlying Kronecker kernels in
``pantr.bspline._extraction_kernels`` are importable Numba functions. The helper
{func}`~pantr.bspline.make_struct_view` bundles every array + shape into a single
{class}`~pantr.bspline.ExtractionStructView` (a ``NamedTuple`` Numba can unbox) so you can
pass one argument instead of forwarding the pieces. See the
{class}`~pantr.bspline.SpanwiseElementExtraction` and
{class}`~pantr.bspline.ExtractionStructView` API entries for the full field list.

## Where to go next

- Element extraction in action (Bézier pieces of a curve): {doc}`/tutorials/03_knot_operations`.
- The element-local bases themselves: {doc}`/tutorials/06_polynomial_bases`.
- Quadrature and grids to complete an assembly loop: {doc}`/tutorials/09_grids_and_quadrature`.
