# Changelog

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
