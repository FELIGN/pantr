# SpanwiseElementExtraction

{class}`~pantr.bspline.SpanwiseElementExtraction` is the central object for
element-local change-of-basis operations on tensor-product B-spline spaces.
It converts a local patch of a B-spline space into an equivalent Bézier,
Lagrange, or cardinal B-spline representation — one element at a time, or
in parallel over a batch of elements — without ever forming the full
$d$-dimensional Kronecker product in memory.

## Layer architecture

The implementation spans three layers that each have a distinct
responsibility:

| Layer | Module | Role |
|---|---|---|
| 3 — kernels | `pantr.bspline._extraction_kernels` | `@njit(cache=True)` free functions; pure computation, no validation |
| 2 — helpers | `pantr.bspline._extraction_helpers` | shape/dtype/writability validation, scratch allocation, kernel dispatch |
| 1 — class | `pantr.bspline.SpanwiseElementExtraction` | construction, target dispatch, identity detection, public API |

**Layer 3** contains 24 Numba-compiled kernels: four operation kinds
(`apply`, `apply_T`, `MT_K_M`, `M_K_MT`) × three dimensions ($d \in \{1, 2, 3\}$)
× two calling modes (single-cell, batch-over-many-cells).  All kernels accept
plain NumPy arrays, which makes them callable from other `@njit` functions.

**Layer 2** is pure Python.  It validates operand shapes, dtypes, and array
writability; sizes the intermediate scratch buffer; and selects the right
Layer-3 kernel for the given operation kind and dimension.

**Layer 1** (`SpanwiseElementExtraction`) builds the per-direction 1D
operator stacks once at construction time and exposes the high-level API.
It never contains Numba code.

## Construction

```python
import pantr
from pantr.bspline import SpanwiseElementExtraction

space = pantr.BsplineSpace(...)          # any BsplineSpace

ext = SpanwiseElementExtraction(space, "bezier")    # Bézier target
ext = SpanwiseElementExtraction(space, "lagrange")  # Lagrange target
ext = SpanwiseElementExtraction(space, "cardinal")  # cardinal B-spline target
```

Three targets are supported:

| Target | Description | Identity detection |
|---|---|---|
| `"bezier"` | Bernstein / Bézier basis on each element | Structural: both boundary knots have multiplicity $\geq p+1$ |
| `"lagrange"` | Lagrange basis at the chosen point distribution | Structural: Bézier identity + Lagrange-to-Bernstein matrix $= I$ |
| `"cardinal"` | Cardinal B-spline basis on each element | Structural: `BsplineSpace1D.get_cardinal_intervals()` |

Additional keyword arguments:

```python
from pantr.basis import LagrangeVariant

ext = SpanwiseElementExtraction(
    space, "lagrange",
    lagrange_variant=LagrangeVariant.GAUSS_LOBATTO,  # default: EQUISPACES
)
```

Construction is $O(n_{\text{elements}} \cdot p^2)$ per direction and happens
once; all subsequent apply calls reuse the cached operator stacks.

## Apply variants

Four operation kinds are available, matching the bilateral assembly pattern
common in IGA:

| Method | Formula | Typical use |
|---|---|---|
| `apply` | $\mathbf{y} = M \mathbf{v}$ | basis conversion (B-spline → target) |
| `apply_transpose` | $\mathbf{y} = M^T \mathbf{v}$ | adjoint / dual conversion |
| `apply_MT_K_M` | $B = M^T K M$ | pull-back of a target-basis matrix to B-spline basis |
| `apply_M_K_MT` | $B = M K M^T$ | push-forward of a B-spline-basis matrix to target basis |

Here $M = M_0 \otimes \cdots \otimes M_{d-1}$ is the $d$-dimensional
Kronecker operator; identity directions short-circuit and are skipped.

### Single-cell applies

Each method takes a cell index and an operand, and optionally a pre-allocated
`out` array and `scratch` buffer (both allocated internally when omitted):

```python
import numpy as np

# single-cell apply (index as flat int or per-direction tuple)
cell_flat = 5                   # flat index, row-major over num_intervals
cell_nd   = (1, 2)              # per-direction index for a 2D space
cell = cell_flat                # use either form below

N_in  = int(np.prod(ext.input_shape_per_dir))
N_out = int(np.prod(ext.output_shape_per_dir))

v = np.ones(N_in, dtype=np.float64)
y = ext.apply(v, cell)                         # y.shape == (N_out,)
y = ext.apply_transpose(y, cell)               # y.shape == (N_in,)

K_out = np.eye(N_out, dtype=np.float64)
B = ext.apply_MT_K_M(K_out, cell)             # B.shape == (N_in, N_in)

K_in  = np.eye(N_in, dtype=np.float64)
B = ext.apply_M_K_MT(K_in, cell)              # B.shape == (N_out, N_out)
```

Pre-allocated buffers avoid repeated heap allocation in loops over elements:

```python
out = np.empty(N_out, dtype=np.float64)

for cell in range(len(ext)):
    ext.apply(v, cell, out=out)
    # use out ...
```

### Batch applies

The `_many` variants parallelize over a batch of cells via `prange` in the
Layer-3 kernels.  Pass a flat 1-D cell-index array or a per-direction 2-D
array:

```python
n_cells = 200
all_cells = np.arange(n_cells, dtype=np.intp)           # flat indices
# or: all_cells = np.stack([idx_dir0, idx_dir1], axis=1) # per-direction

V = np.random.default_rng(0).random((n_cells, N_in))
Y = ext.apply_many(V, all_cells)                        # (n_cells, N_out)

K_batch = np.eye(N_out)[np.newaxis].repeat(n_cells, axis=0)  # (n_cells, N_out, N_out)
B_batch = ext.apply_MT_K_M_many(K_batch, all_cells)          # (n_cells, N_in, N_in)
```

The full matrix of batch methods mirrors the single-cell API:

| Method | Input shape | Output shape |
|---|---|---|
| `apply_many` | `(n_cells, N_in)` | `(n_cells, N_out)` |
| `apply_transpose_many` | `(n_cells, N_out)` | `(n_cells, N_in)` |
| `apply_MT_K_M_many` | `(n_cells, N_out, N_out)` | `(n_cells, N_in, N_in)` |
| `apply_M_K_MT_many` | `(n_cells, N_in, N_in)` | `(n_cells, N_out, N_out)` |

Passing `all_cells = np.arange(ext.num_total_intervals)` processes every
element.  You can also pass a subset to skip elements or process patches in
a different order.

## Identity detection and short-circuit

When a 1D operator $M_k$ for direction $k$ at element $i$ is the identity
matrix, applying it is a no-op.  The kernels exploit this: an identity flag
is passed alongside the operator pointer, and the kernel skips the matrix
multiply for that direction.

### How flags are determined

**`"cardinal"` target** — flags come from the *structural* mask returned by
{meth}`~pantr.bspline.BsplineSpace1D.get_cardinal_intervals`.  An interval is
cardinal if it has the same span length as the $p - 1$ preceding and $p - 1$
following non-zero spans (where $p$ is the polynomial degree); on such
intervals the cardinal extraction operator is provably the identity.  No
numerical comparison is performed.

**`"bezier"` target** — an element is identity iff both its boundary unique
knots (in the domain) have multiplicity $\geq p+1$, meaning the element is
already a Bézier patch fully isolated from its neighbours.  Knot
multiplicities are computed using the space's own `tolerance`.

**`"lagrange"` target** — the Lagrange extraction is `bezier_op[e] @ lagr_to_bzr`.
This is the identity iff the Bézier extraction is identity *and* the
Lagrange-to-Bernstein matrix `lagr_to_bzr` equals `I`.  The latter holds
when the Lagrange nodes coincide with the Bernstein abscissae `i/p` — for
example, `degree == 1` with equispaced, GLL, or Chebyshev-2nd nodes.  For
`degree == 0`, every element is trivially identity.  No floating-point matrix
comparison is performed; the check is a single `np.array_equal` against `I`.

### Querying identity flags

```python
# global: True only if every element in every direction is identity
if ext.is_identity:
    ...

# per-element
if ext.is_identity_at(cell):
    ...

# per-direction flags for a single element
flags = ext.per_direction_identity_flags(cell)  # tuple[bool, ...] of length d

# count of fully-identity elements
n_trivial = ext.num_identity_elements

# raw masks for programmatic use
masks = ext.is_identity_mask_1d  # tuple of 1-D bool arrays, one per direction
```

## Numba-callability contract

{attr}`~pantr.bspline.SpanwiseElementExtraction.ops_1d` and
{attr}`~pantr.bspline.SpanwiseElementExtraction.is_identity_mask_1d` are
read-only NumPy arrays and can be passed directly into `@njit` functions.
The Layer-3 kernels are importable free functions:

```python
from numba import njit
from pantr.bspline._extraction_kernels import apply_kron_2d, apply_kron_MT_K_M_2d

ops_1d   = ext.ops_1d                    # tuple of float64 arrays (read-only)
masks_1d = ext.is_identity_mask_1d       # tuple of bool arrays (read-only)

M0, M1 = ops_1d[0], ops_1d[1]           # (n_el_0, p+1, p+1), (n_el_1, p+1, p+1)
f0, f1 = masks_1d[0], masks_1d[1]       # (n_el_0,), (n_el_1,)

@njit(cache=True)
def my_kernel(M0, f0, M1, f1, v, out, scratch):
    i0, i1 = 3, 7
    apply_kron_2d(M0[i0], f0[i0], M1[i1], f1[i1], v, out, scratch)
```

Batch kernels follow `apply_kron_{op_kind}_many_{d}d`.  Single-cell kernels
drop the redundant `apply` prefix — as shown in the table below:

| op_kind | Single-cell | Batch |
|---|---|---|
| `apply` / `apply_T` | `apply_kron_{1,2,3}d` / `apply_kron_T_{1,2,3}d` | `apply_kron_apply_many_{1,2,3}d` / `apply_kron_apply_T_many_{1,2,3}d` |
| `MT_K_M` | `apply_kron_MT_K_M_{1,2,3}d` | `apply_kron_MT_K_M_many_{1,2,3}d` |
| `M_K_MT` | `apply_kron_M_K_MT_{1,2,3}d` | `apply_kron_M_K_MT_many_{1,2,3}d` |

All kernels accept only plain NumPy arrays (no Python objects).  Dimensions
above 3 raise `NotImplementedError` from the Layer-2 dispatcher.

### Struct view for `@njit` callers

{class}`~pantr.bspline.ExtractionStructView` bundles the compact storage and
shape metadata of a {class}`~pantr.bspline.SpanwiseElementExtraction` into a
single immutable `typing.NamedTuple` that Numba can unbox.  Pass it as a
single argument instead of forwarding seven separate objects.

```python
from numba import njit
from pantr.bspline import SpanwiseElementExtraction, make_struct_view
from pantr.bspline._extraction_kernels import apply_kron_apply_many_2d

ext  = SpanwiseElementExtraction(space, "bezier")
view = make_struct_view(ext)   # shares arrays; no copies

@njit(cache=True)
def batch_apply(view, cell_indices, v, out, scratch):
    apply_kron_apply_many_2d(
        view.compact_ops_1d[0],       view.compact_ops_1d[1],
        view.idx_maps_1d[0],          view.idx_maps_1d[1],
        view.is_identity_mask_1d[0],  view.is_identity_mask_1d[1],
        cell_indices, v, out, scratch,
    )
```

The struct view exposes the same arrays and integer metadata already
available on the extraction object — see the
{class}`~pantr.bspline.ExtractionStructView` API reference for the full
field list.  Because every field has a uniform Numba type (homogeneous
tuples of arrays, plain ints, and int tuples), the view is a drop-in
argument for any `@njit` function that previously took the individual
pieces.

## Materializing operators

Use {meth}`~pantr.bspline.SpanwiseElementExtraction.operator` to assemble the
full $(N_{\text{out}} \times N_{\text{in}})$ Kronecker product for a single
element, or {meth}`~pantr.bspline.SpanwiseElementExtraction.tabulate` to
produce the complete `(num_total_intervals, N_out, N_in)` stack:

```python
M_cell = ext.operator(cell)   # (N_out, N_in), assembled via np.kron
M_all  = ext.tabulate()       # (num_total_intervals, N_out, N_in)
```

Both use `numpy.kron` and allocate a fresh matrix.  Prefer the matrix-free
apply methods in performance-critical code.

{meth}`~pantr.bspline.SpanwiseElementExtraction.tabulate` is the canonical way
to obtain the Bézier extraction operators used by
{meth}`~pantr.bspline.Bspline.to_beziers` — in fact the `to_beziers`
implementation sources its operators from `ops_1d` directly.

## Iteration and indexing

`SpanwiseElementExtraction` supports `len`, index access, and iteration:

```python
len(ext)          # == ext.num_total_intervals

ops, flags = ext[cell]    # per-direction ops and identity flags at one element

for ops, flags in ext:    # row-major over num_intervals
    ...
```

Each `ops` is a tuple of 2D views into `ops_1d`; each `flags` is a tuple of
bools from `is_identity_mask_1d`.
