# Porting the extraction machinery: what moves, in what order, and what the bound is

**Status:** proposed, 2026-08-31. Written while building #399's first slice, for the
remaining slices of #399 and for whoever sequences them against #396.
**Date:** 2026-08-31.
**Scope:** which parts of `pantr.bspline`'s extraction cluster can move to C++ today and
which cannot, what shape the C++ takes, what parity claim its kernels can support, and
whether `ExtractionStructView` is a type that moves at all. Not what an accessor hands
back, which is `design/bspline_ownership_lifetime.md`. Not where a derived cache lives,
which is `design/bspline_derived_caches.md`.
**Companions:** `design/cross_backend_types.md` (the ownership ruling and the kernel-seam
table this obeys), `design/backend_parity.md` (Rules 3, 9 and 10, which the bound below is
built from), `design/bspline_ownership_lifetime.md` (F5, which classifies
`ExtractionStructView`), `design/bspline_derived_caches.md` (which assigns
`SpanwiseElementExtraction`'s two memos to #399 and the knot scans to #396).

**Validated against:** originally `proto/cpp` at `43453c0` plus #399's first slice, which has
since merged; **re-checked against `proto/cpp` at `7ba20d3`**, where that slice is `7ba20d3`
itself. The Python locators are still the shifted ones -- the slice added about seventy lines
near the top of `src/pantr/bspline/spanwise_element_extraction.py`, so those line numbers differ
from the two companion notes by that much, and now agree with the trunk rather than diverging
from it. C++ locators are this branch's. nanobind **2.14.0**, CPython **3.14.6**,
g++ **14.4.0**.

**One finding below has been overtaken by the trunk and is corrected in place rather than
rewritten**: see the amendment under F1. The rest was re-read at `7ba20d3` and stands.

## The decision in one paragraph

**Only one of #399's four pieces could be built when this was written, and it was not the one
the ticket names.** `cpp/include/pantr/` had no `bspline/` directory at all, and every part of
the extraction cluster except the Kronecker apply kernels reaches through a B-spline space:
the operator builders need the unique-knot-and-multiplicity scan and the cardinal-interval
scan, and the domain type holds a `BsplineSpace`. The apply kernels do not -- they take
operator matrices, identity flags and buffers, and nothing else -- so they were the slice to
build, and they are what the C++ `SpanwiseElementExtraction` will call when it exists.
**Since then #428 has landed two of the three missing pieces**; the amendment under F1 says
which, and what is still owed. Their parity claim turned out to be **bitwise rather than
bounded** -- the port reproduces the oracle's stage order and its within-stage summation order,
so only a fusing build separates them, and the stage count remains per cell rather than per
kernel because the identity short-circuit removes whole contraction stages. And
`ExtractionStructView` **does not become a C++ type**: it exists to be unboxed by Numba,
which a bound class cannot be, and `design/cross_backend_types.md` already puts that kind of
type on the Python side. What moves is the storage it views, not the view.

## Findings against the ticket as written

### F1 (critical). #399 is blocked on #396 for three of its four pieces, and the DAG does not record it

`gh issue view 399` lists `blocked-by: #394, #393` -- the two design tickets, both closed.
Nothing records a dependency on #396 (`BsplineSpace1D` and `BsplineSpace` in C++). Reading
the tree, three of the four pieces have one:

- **The 1D extraction operator builders** (`src/pantr/bspline/_bspline_extraction.py`) open
  with `_get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)`
  (`:71-73`) and, for the cardinal target, `_get_Bspline_cardinal_intervals_1D_impl`
  (`:340`). Both live in `_bspline_knots.py`, and `design/bspline_derived_caches.md`'s
  per-ticket table assigns the unique-knot scan to **#396** (the row deleting
  `_cached_unique_knots_and_multiplicity`, and the paragraph on the `tobytes()` key going to
  zero "because the data is owned"). A C++ extraction that carried its own scan would be the
  second implementation of one computation that `design/cross_backend_types.md` exists to
  forbid.
- **The two structural identity-mask predicates** have the same dependency, through
  multiplicities (`spanwise_element_extraction.py:1129`) and through
  `get_cardinal_intervals`.
- **`SpanwiseElementExtraction` itself** holds a `BsplineSpace`
  (`spanwise_element_extraction.py:278-286`), which is the class-H accessor
  `design/bspline_ownership_lifetime.md` assigns it. There is no C++ type to hold.

**What is left, and it is a real slice:** the Kronecker apply kernels
(`_extraction_kernels.py`, 1934 lines, 24 kernels). Their arguments are per-direction
operator matrices, identity flags, an operand and two buffers. No knots, no space, no
tolerance. They are on the kernel seam exactly as `design/cross_backend_types.md`'s table
describes it, and they are what the ported type will call.

**The consequence for sequencing.** #399 cannot be worked to completion in parallel with
#396; it can be worked to *one PR* in parallel with it. Either the DAG gains the edge
`#399 blocked-by #396`, or #399 is split so that the kernel slice keeps its independence and
the rest inherits the edge. The second is better, because the kernel slice is the largest
single piece and blocking it on #396 wastes the parallel window.

#### Amendment, 2026-09-01: #428 landed two of the three, and the cardinal scan is what is left

Re-read at `proto/cpp` `7ba20d3`. `cpp/include/pantr/bspline/` now holds `knots.hpp` and
`space_1d.hpp`, so the finding above is half overtaken and must not be quoted as it stands.

**Available now:**

- `pantr::bspline::unique_knots_and_multiplicity` (`cpp/include/pantr/bspline/knots.hpp:285`),
  returning representatives and multiplicities together;
- the in-domain views on the type, `unique_knots_in_domain()` and `multiplicity_in_domain()`
  (`cpp/include/pantr/bspline/space_1d.hpp:286`, `:295`). The boundary multiplicity the Bézier
  builder opens with is the first entry of the second, so
  `_get_multiplicity_of_first_knot_in_domain_impl` needs no separate port.

**Still missing, and it is exactly one thing:** `get_cardinal_intervals`. Not an oversight --
#428 excluded it deliberately and says why, in its file comment on what the type owns: it is
*"a computation over the knots rather than a property of them"*, so it belongs with the
operations rather than with the type.

**Corrected 2026-09-03, while building the Bézier half of S3: the sentence above about the
boundary multiplicity is wrong.** It read that "the boundary multiplicity the Bézier builder
opens with is the first entry of [`multiplicity_in_domain()`], so
`_get_multiplicity_of_first_knot_in_domain_impl` needs no separate port". The two are
different computations over different index ranges and they disagree:

- the oracle's helper counts how many of `knots[0 .. degree]` lie within `tol` of
  `knots[degree]`, so it never looks past index `degree` and never chains;
- `multiplicity_in_domain().front()` is the size of the whole gap-chained **class** holding
  `knots[degree]`, which may reach either side of that index.

The class count is therefore always at least the helper's, and strictly more whenever the
first in-domain knot is repeated. Measured on `[0, 0.4, 0.5, 0.5, 1, 1.5, 2, 2.5]` at degree 2,
`snap_knots=False`: the helper returns 1 and the class holds 2, and the two produce different
operators. The port carries the helper, as
`pantr::bspline::multiplicity_of_first_knot_in_domain`, factored out of `num_basis`'s periodic
branch, which was already computing exactly it inline.
`tests/parity/test_bspline_bezier_extraction.py` keeps that vector as a parity case for this
reason and a mutation confirmed it is the only case in the table that separates the two.

**What that does to the slices.** S3 splits. The **Bézier** operator builder and the Bézier
identity-mask predicate are unblocked today; the **Lagrange** ones follow immediately, since
they are the Bézier operator post-multiplied by a matrix `lagrange_to_bernstein_1d` already
provides. The **cardinal** target still waits on the interval scan, which is a small port in
its own right and would sit beside `knots.hpp` rather than inside the type. S4 is unchanged:
it needs `BsplineSpace` (the nD one), not `BsplineSpace1D`.

**The rule the original finding rests on has not changed**, and it is why the cardinal half
still waits rather than being carried here: a second implementation of one computation is what
`design/cross_backend_types.md` forbids, so the interval scan is ported once, wherever it lands,
and not copied into the extraction slice.

### F2 (critical). `ExtractionStructView` should not become a C++ type, and the lifetime note does not say it should

`design/bspline_ownership_lifetime.md` F5 classifies it as class **A** and says it "needs
nothing from the aggregation rule". That settles its *lifetime*. It does not settle whether
the type moves, and three things say it should not:

1. **Its only purpose is Numba unboxing.** Its own docstring
   (`spanwise_element_extraction.py:1190`) says "for `@njit` callers ... an object that
   Numba can unbox"; `docs/guide/spaces-knots.md:198-202` says the same to users; and
   `tests/test_spanwise_element_extraction.py:1555-1712` passes one into four `@njit`
   functions and drives the batch kernels through it. A nanobind-bound class cannot be
   unboxed by Numba, so binding it breaks that outright rather than changing it.
2. **`design/cross_backend_types.md` already puts this kind of type on the Python side**:
   "a type that exists only to implement the Python binding or its dispatch stays in Python,
   because it could not move even in principle". `DegreeKernels` is the example given; this
   is the same shape with data instead of function references.
3. **In the end state it does not exist.** With no interpreter there is no Numba and no
   unboxing, so the `NamedTuple` dies with the binding layer. Its C++ analogue -- a
   non-owning aggregate of spans handed to a kernel -- is a different object with a different
   job, and it is not the thing the Python name refers to.

**What moves instead is the storage it views.** `make_struct_view` bundles
`extraction.compact_ops_1d`, `.idx_maps_1d` and `.is_identity_mask_1d` by reference
(`:1258-1266`), so once `SpanwiseElementExtraction` is C++-owned those are the class-A
`nb::ndarray` views its accessors hand out, each carrying the owner as keep-alive. They are
real read-only `numpy.ndarray`s, which is already what Numba unboxes today, so the
`NamedTuple`, its seven fields and their order are unchanged, and
`tests/test_spanwise_element_extraction.py:1785`'s `np.shares_memory` assertion is what pins
that the bundling is still a view rather than a copy.

### F3 (important). The accumulation width is the storage dtype in all twelve kernels, and Rule 9 makes that worth writing down

`design/backend_parity.md` Rule 9 records that an oracle's accumulation width is a
per-kernel fact and that `_bezier_core.py` uses three different ones without announcing any
of them. This module is the opposite case and the fact is just as load-bearing: **every one
of the twelve per-cell kernels opens its contraction with `zero = M_0.dtype.type(0.0)`** --
verified, twelve occurrences, at `_extraction_kernels.py:80, 131, 223, 323, 373, 461, 554, 608,
673, 798, 925, 1113` -- so the accumulator is the operator's dtype, which Layer 2 has
already made equal to the operand's and the output's.

Two consequences:

- **A C++ port must accumulate in `T`, not in `double`.** Accumulating a `float` chain in
  `double` would be more accurate and would not be the same function; the difference would
  surface as a `float32` parity failure attributed to the wrong cause. This is
  `change_basis.hpp`'s "arithmetic width is the output dtype, not the accumulator" paragraph,
  and it applies here for the same reason.
- **`storage_per_stage` is zero in the parity claim**, both dtypes, because the accumulator
  *is* the storage and nothing narrows on the store.

The dtype is taken from **`M_0`** specifically, in every kernel, including branches where
`M_0` is flagged identity and its values are never read. Only its `dtype` is touched, so
this is not a bug; it is a constraint on the binding, which must key the instantiation on
the operator array rather than on the operand.

### F4 (important). The stage count is a property of the cell, not of the kernel

Every kernel short-circuits per direction on `is_id_k`, and a short-circuited direction
performs **no contraction at all** -- the axis is passed through and `M_k` is not read
(`_extraction_kernels.py:23-28` states it; the branches at `:133` and `:144` are the 2D case). When
every direction is identity the kernel degenerates to a copy, which is the one case where
`out` may alias the input.

So the length of the dependency chain from an input element to an output element is
`sum over the non-identity directions k of n_in_k`, and that varies cell by cell within one
call. A claim written as `Roundings(stages=f(d), ...)` would be wrong on any cell with an
identity direction -- too loose there, which `design/backend_parity.md` Rule 3 refuses once
it reaches the values being compared, and an all-identity cell is **bitwise** rather than
bounded, because a copy commits no roundings at all.

**The claim therefore has to be built from the cell's own identity flags**, and the
all-identity cell has to be claimed `bitwise_parity` rather than clamped to one stage.
Rule 10 records the same mistake being made and caught in the Bézier port: "a zero-stage
claim -- degree 0, where every one of these kernels short-circuits -- was being clamped to
one stage instead of being what it is, which is bitwise".

### F5 (critical). A `StrEnum` in a `nopython` kernel is a silent wrong answer, which is why the target is an `IntEnum`

The project rule already says a closed set of choices is an `IntEnum` "which is also
Numba-legal". Measured, that phrasing is too kind to the alternative: a `StrEnum` is not
*rejected* by Numba, it is **accepted and gives the wrong answer**.

Measured 2026-09-01, numba **0.65.1**, CPython 3.14.6, with a two-member `StrEnum` `SE`:

| expression inside `@nb_jit(nopython=True)` | `SE.A` | `SE.B` | `"a"` | `"b"` |
|---|---|---|---|---|
| `x == SE.B` (compare against the captured member) | False | **False** | False | False |
| `x == "b"` (compare against a literal) | False | True | False | True |

`numba.typeof(SE.B)` is `unicode_type`, and a member round-trips through a kernel as its
bare string (`echo(SE.B)` returns `'b'`, not the member). So the natural spelling -- capture
the member, compare against it -- compiles cleanly and is **dead in every branch**, with
nothing raised and no diagnostic. The same code written against an `IntEnum` is correct, which
`tests/test_spanwise_element_extraction.py::test_target_members_are_numba_holdable` pins.

This is the "a passing test is consistent with a false theorem" class: no test of a kernel
written that way would fail, because the kernel would simply never take the branch.

**The one live consequence in the tree, and it is latent rather than a bug.**
`pantr.basis.LagrangeVariant` **is** a `StrEnum` (`src/pantr/basis/_basis_tabulate.py:30`).
Verified by reading: it reaches no `nopython` function today -- the only module holding both
it and a kernel is `_bspline_extraction.py`, where it appears at `:252` and `:261-262`, inside
the Layer-2 `_tabulate_Bspline_Lagrange_1D_extraction_impl` and not inside either of the two
jitted functions (`:33`, `:133`). `design/cross_backend_types.md` records the other half of
the same containment: the variant is resolved to nodes on the Python side and never crosses
into `change_basis`'s kernels either.

So nothing is wrong now. What is wrong is that nothing *stops* it: the first author to pass a
`LagrangeVariant` into a kernel and branch on a member gets a silently dead branch. Flagged
below rather than fixed here.

**Re-checked 2026-09-03, by the slice that passes closest to this edge.** S3's Lagrange half is
the first dispatched code path that has a `LagrangeVariant` in scope *and* a kernel to call, so
it is where the containment would have broken. It did not: the variant is resolved to a matrix
in Layer 2 and the **matrix** crosses the seam, never the tag. That was a deliberate choice and
not an accident of the shape -- building the matrix inside the kernel would have needed the
variant there -- so the containment is now one decision wide rather than zero, and it is still
a fact about the call graph rather than a guard.

### F6 (recommended). `OpKind` is the same rule's second instance, in the same cluster, and is deliberately left alone

`OpKind = Literal["apply", "apply_T", "MT_K_M", "M_K_MT"]`
(`src/pantr/bspline/_extraction_helpers.py:57`) is a closed set of choices, stringly typed,
`==`-dispatched at `:228-235` and `:309-316`, validated at `:107`, and used as half of the key of
the two dispatch tables at `:68` and `:83`. It is exactly what the target was before #399's first
slice.

It is left alone there, and the reason is scope rather than merit: the ticket names the
target, `OpKind` reaches no public signature, and the integer never has to cross the seam
under the binding shape below (each op kind gets its own entry point, so the *selection*
stays Python-side). It is worth its own small change, and it should be taken together with
whatever moves the dispatch tables into the catalogue, since that is the code that reads it.
`OpKind` is private and the not-yet-public downstream consumer imports private symbols, so
it owes the same census the target owed.

### F7 (important). Two facts about the oracle that a generic port has to get right, and one it does not

Read out of the oracle in full, because each would be a silent divergence.

**The `d = 2` unilateral kernels are branched by identity *combination*, while every
other multi-directional kernel is branched *per stage*.** `apply_kron_2d` and
`apply_kron_T_2d` write four independent code blocks, one per `(is_id_0, is_id_1)`
pattern (`_extraction_kernels.py:125`, `:133`, `:144`, `:155`), and the two
single-identity blocks contract straight from the operand into `out` with no scratch at
all. `apply_kron_3d` and both bilateral families instead run one linear stage sequence
in which an identity stage is a `pass`.

A generic implementation that wrote `out` at the **last direction** would not reproduce
that: at `d = 3` with the last direction identity it would contract into scratch and
then copy. Writing `out` at the **last contracting** direction reproduces it exactly,
for every dimension and every pattern, and is one pass shorter than the oracle in the
one case where they differ. That is the rule the port follows, and it is why the
combination branching does not have to be transliterated.

**The bilateral stage order is direction-major, `(row 0, col 0, row 1, col 1, ...)`,
`2d` stages in one pass** -- not `d` stages twice. Confirmed twice over: from the
kernels (`_extraction_kernels.py:682`, `:697`, `:713`, `:731` for `d = 2`, six slots for
`d = 3`) and independently from `_bilateral_scratch_size`, whose simulation computes
`axis = k if stage % 2 == 0 else d + k` (`_extraction_helpers.py:181-190`). Following it
is what keeps the caller's scratch large enough, so it is a contract rather than a
convention.

**And one thing that looked like a hazard and is not.** `_required_scratch_size`'s
docstring claims sufficiency "for any identity-flag pattern" while its two component
functions simulate only the fully non-identity sequence, which is an unproved claim as
written. It is nevertheless true, and the argument is short: the intermediate after
stage `k` has extents `out_0..out_k, in_{k+1}..in_{d-1}` whatever the pattern, because
an identity direction has `in == out`; a pattern with identities therefore realises a
**subset** of the same products the formula maximises over, and a subset's maximum
cannot exceed the whole set's. The port relies on the same sizes and inherits the same
guarantee. Worth stating in the docstring on the Python side, which is a one-line
`docs` change for whoever next opens that file.

## The C++ shape

### One implementation, not twenty-four

The oracle has twelve per-cell kernels because Numba needs a separate specialisation per
dimension: a `nopython` function cannot loop over a variable-length tuple of arrays. C++ has
no such constraint, and `design/backend_parity.md` accepts a **bounded** claim rather than
bit-identity precisely so the C++ can be written as C++.

So: one mode-wise contraction primitive, four op-kind drivers over it, and the dimension a
runtime argument.

```cpp
/// A per-direction 1D operator with its identity flag.
///
/// `matrix` is unread when `is_identity` -- the direction is passed through -- which is
/// what makes an all-identity cell a copy rather than a chain of identity products.
template <Real T>
struct ModeOperator {
    span2d<const T> matrix;  ///< `(n_out, n_in)`; unread when `is_identity`.
    bool is_identity;
};

/// `out = kron(ops[0], ..., ops[d-1]) @ v`, by mode-wise contraction.
template <Real T>
void apply_kron(std::span<const ModeOperator<T>> ops, std::span<const T> v,
                std::span<T> out, std::span<T> scratch);
```

and the three siblings `apply_kron_transpose`, `apply_kron_mt_k_m`, `apply_kron_m_k_mt`.

**What must be preserved exactly, and what is free.** The *stage structure* is not free: the
set of modes contracted, the order they are contracted in, and which are skipped all have to
match the oracle, because they are what fixes the number of roundings the bound is built
from, and because a different mode order would change the intermediate magnitudes rather
than only the last bits. The *summation order within one contraction* is free -- that is what
the bound covers -- and so is the buffer layout, the blocking and the vectorisation.

**The accumulator is `T`.** See F3. Not `Acc`, not `double`.

### The bindings stay flat and stay one per (op kind, dimension)

Twenty-four entry points, each a three-line adapter that packs its `M_0 … M_{d-1}` and
`is_id_0 … is_id_{d-1}` arguments into a small `std::array<ModeOperator<T>, d>` and calls the
generic implementation. The flat form is the seam's, not a preference: an argument list of
separate arrays and scalars is what `design/cross_backend_types.md`'s table admits, and it is
what keeps the Python catalogue selecting between two functions with the same signature
rather than between two conventions -- which is the property `_change_basis_core.py`'s
docstring names as the reason its kernels take their nodes as arguments.

`.noconvert()` on every array argument, per `test_bezier_binding_contract.py`'s fourth
paragraph: a silent dtype cast here would change the accumulation width, which is exactly
the fact F3 says a reader cannot recover.

### The catalogue is bare callables keyed by (op kind, dimension)

`design/cross_backend_types.md`: a record when the consumer needs more than one kernel at
once, a bare callable when it does not. `_prepare_apply_call` selects exactly one kernel per
call from `_KERNELS[(op_kind, d)]` (`_extraction_helpers.py:68`, `:83`), so it is the bare-callable
branch. The two dispatch tables move from `_extraction_helpers.py` into a new
`_extraction_backend.py` and become the catalogue, with the same `_select` shape the other
four ported packages use.

Mirroring: every public `apply*` method takes `out`, so the kernels fill the caller's buffer
and return nothing. That is already both backends' convention and needs no change.

### What is *not* dispatched, and why that is not an omission

The per-cell kernels are documented as callable from other `@njit` code
(`_extraction_kernels.py:29-32`), and `ExtractionStructView` exists so that downstream
`@njit` code can drive the *batch* kernels directly. Those remain Numba functions and remain
importable under their current names: a C++ kernel is not callable from inside a `nopython`
function without a flat C ABI, which is a separate piece of work with its own reasons
(`~/claude-config/user-CLAUDE.md`'s note on the two adapters over one core). The dispatch is
added at the Layer-2 entry points, above them.

## The parity claim

### The shape

For a single cell, let `K` be the set of directions with `is_id_k` false, in the order the
kernel contracts them. Both backends perform, for each `k` in `K`, one contraction whose
inner products have length `n_in_k` (unilateral) and whose accumulator is `T`.

- **`K` empty:** the kernel is a copy. `bitwise_parity`, with the reason being that no
  arithmetic is performed. Not a one-stage bounded claim (F4).
- **`K` non-empty:**

  ```
  Roundings(stages=sum(n_in_k for k in K), accumulator_per_stage=1, storage_per_stage=0)
  ```

  with `accumulator = storage = dtype` and the amplification below. `storage_per_stage` is
  zero by F3. `accumulator_per_stage` is one because a contraction's dependency chain commits
  one rounding per accumulation step and the step count is already in `stages` -- the same
  reading `test_bezier_degree.py:467` and `test_change_basis.py:556` use.
  On a build where the contraction may fuse, `design/backend_parity.md` Rule 10's budget
  applies instead and `contraction_may_fuse()` is the switch, exactly as
  `test_bezier_degree.py:451-458` does it.

For the bilateral kinds the chain is the two passes, so `stages` is the sum over both.

### The amplification, and why it is the absolute-value companion rather than `max|M|`

Running the same kernel on `|M_k|` and `|v|`, elementwise, is the correct elementwise
magnitude reachable at each output element, and it is what Rule 10 prescribes for the one
non-convex entry in its own table (`|R| @ |c|`, "since `R` has negative entries").

It has to be the companion here rather than `max|M| * max|v|`, and the reason is specific to
this cluster: **the operators are not all convex.** A Bézier extraction operator is, being a
product of knot-insertion convex combinations -- its **columns** sum to one, which is the
partition-of-unity invariant below. The Lagrange and cardinal operators are that matrix
post-multiplied by a Lagrange-to-Bernstein or cardinal-to-Bernstein matrix, in
`_tabulate_Bspline_Lagrange_1D_extraction_impl` and its cardinal sibling.

**Corrected 2026-09-03, while building the Lagrange half of S3: this said "those have negative
entries ... false for two of the three targets", and it is one of the three.** The
Lagrange-to-Bernstein matrix is `L[j, k] = B_j(x_k)`, the Bernstein basis tabulated at the
Lagrange nodes, and every node of every family here lies in `[0, 1]` where the Bernstein basis
is non-negative and sums to one. So `L` is column-stochastic, and the Lagrange extraction
operator is a product of two column-stochastic matrices: entrywise non-negative, columns
summing to one, exactly like its Bézier parent. Measured over all five node families at
degrees 1, 2, 3, 5, 8 and 12: the smallest entry is never negative and the largest column-sum
deviation from one is 1.2e-15. Only **cardinal-to-Bernstein** has negative entries.

That does not change what the code does -- the absolute-value companion is correct either way,
and is what both parity files use -- but it does change what the amplification is worth: for
the Lagrange target the companion is the answer's own magnitude, bounded by one, so the bound
is tight rather than merely valid. `tests/parity/test_bspline_lagrange_extraction.py` asserts
the non-negativity rather than assuming it, so a family whose nodes left `[0, 1]` would be a
failure here rather than a silently loose bound.

The sentence above also said the Bézier operator's *rows* sum to one. They do not; its columns
do, which the 2026-09-03 correction further down already records for the same reason.

### The independent accuracy check

Parity says the two backends agree, not that either is right, and a shared index-order error
is invisible to it. Two independent oracles, both cheap:

1. **Exact integer arithmetic.** With small-integer operator entries and an integer operand,
   `kron(M_0, …, M_{d-1}) @ v` is exactly representable in binary64, so the kernel's float64
   output must equal a Python-integer `numpy.kron` computation **exactly**, and
   `assert_accuracy` gets a zero bound. This is the check that catches a transposed index or
   a wrong mode order, which is the error class a bound cannot see because both backends
   would make it. It is also a check on the *kernel*, independent of anything B-spline.
2. **Partition of unity, for the Bézier target only.** Each Bézier extraction operator's
   **columns** sum to one, so `kron` of them is column-stochastic and `ones^T M = ones^T` to
   within the contraction's own rounding. `tests/test_thb_validation_identities.py` already
   carries the hierarchical form of this identity, which the ticket names as the natural
   oracle for the THB half.

   **Corrected 2026-09-03: this paragraph said *rows*, and it is columns.** The identity
   follows from `sum_i N_i = 1` on the element and `sum_j B_j = 1` on the reference interval:
   `sum_i sum_j C_ij B_j = 1 = sum_j B_j`, and the Bernstein basis being independent forces
   `sum_i C_ij = 1` for each column `j`. Measured on the quadratic three-element open spline,
   whose first operator is `[[1,0,0],[0,1,1/2],[0,0,1/2]]`: its columns sum to `1, 1, 1` and
   its rows to `1, 3/2, 1/2`. The distinction is not cosmetic -- it is exactly what makes the
   check able to catch a transposed operator, which a row-sum check on a matrix whose row sums
   are all one could not. Nothing built on the wrong version: the apply kernels' claim in
   `tests/parity/test_extraction_kernels.py` is bitwise and rests on no stochasticity, and the
   amplification argument two sections above uses the absolute-value companion rather than
   convexity.

**The rule this milestone learned the hard way applies to both**: a bound compared only
against zero has not been checked. Case 1 is exact by construction and is the one at risk --
its assertion must be that the observed disagreement is exactly zero *and* that the sweep
included cases where the float64 result is not itself exactly the integer answer, or it is
only ever comparing zero against zero. Case 2 must assert the observed error is nonzero.

## The slices

- **S1 (landed).** `ExtractionTarget`, the target enum. Python only, no C++.
- **S2.** The Kronecker apply kernels in C++: the generic implementation, twenty-four
  bindings, `_extraction_backend.py`, the Layer-2 rewiring, C++ unit tests, and
  `tests/parity/test_extraction_kernels.py` carrying the claim above. **Not blocked on #396.**
- **S3.** The 1D extraction operator builders and the two identity-mask predicates.
  **Blocked on #396** for the knot scans (F1). *Split, and the Bézier half landed
  2026-09-03*: `pantr::bspline::bezier_extraction_1d` and
  `bezier_structural_identity_mask` in `cpp/include/pantr/bspline/extraction.hpp`, bound by
  `cpp/bindings/bspline_extraction_operators.cpp` and dispatched from
  `pantr.bspline._extraction_backend`. *The **Lagrange** half landed 2026-09-03*:
  `pantr::bspline::lagrange_extraction_1d` and `lagrange_structural_identity_mask` in the same
  header, bound in the same file and dispatched from the same catalogue. It was the small slice
  this note predicted, with two things it did not:

  - **The change-of-basis matrix is an argument, not something the C++ builds.**
    `change_basis.hpp` already owns `lagrange_to_bernstein_1d`, `pantr.change_basis` caches the
    finished matrix per `(degree, variant, dtype)`, and `LagrangeVariant` must not approach a
    kernel (F5). Passing the matrix keeps one implementation, keeps the cache, keeps the enum
    off the seam, and makes the matrix **common mode** between the backends, so the parity
    claim is about the extraction and `test_change_basis.py`'s is about the change of basis.
  - **The claim is bounded rather than bitwise, and it is the first builder in this cluster
    that is.** The oracle's post-multiplication is `numpy.matmul`, a BLAS `gemm` whose
    summation order is unspecified and blocked; the C++ runs an ascending loop. Both accumulate
    in the storage format, so the budget is `Roundings(degree + 1, 1, 0)` with the
    absolute-value companion as amplification. Measured: the two differ by one unit of roundoff
    at degree 3 and above and not at all below, so the bound is live rather than nominal. The
    identity change of basis -- degree 1 with equispaced, Gauss-Lobatto-Legendre or
    second-kind Chebyshev nodes -- is the exception and falls back to the Bézier claim.

  **Cardinal still waits** on the interval scan.
- **S4.** `SpanwiseElementExtraction` itself: the class-H `space` accessor, the two memos
  `design/bspline_derived_caches.md` assigns here (`ops_1d` DCLP-lazy returning a view of the
  memo, `num_identity_elements` eager), the wrapper, `__reduce__`, and the binding-contract
  file `design/bspline_ownership_lifetime.md` asks every type in the milestone for.
  **Blocked on #396**, hard.

S2 is the one to build while #396 is in flight. S3 and S4 are one PR each afterwards, and
S4's shape is already fully settled by the two design notes -- it needs building, not
deciding.

## Alternatives rejected

**Transliterate the twelve Numba kernels into twelve C++ functions, for bit-identity.**
Tempting because a bitwise claim is stronger than a bounded one and asserts more. Rejected
because it optimises the C++ for the oracle rather than for the consumer it is being written
for: the end state has no Numba, and a dimension-specialised transliteration is code nobody
would write for C++ and that nothing would later un-write. It also does not buy bit-identity
on a fusing build, which is what Rule 10 exists to say. **What would change it:** a
measurement showing the generic form is materially slower at `d <= 3`, which would be an
argument for specialising the *hot* shapes rather than all of them.

**Bind one dimension-generic entry point taking a list of operator arrays.** Fewer bindings,
and closer to the C++ API. Rejected because a list of arrays is not on
`design/cross_backend_types.md`'s table of what crosses, and because it would put a
per-call Python-list construction in front of a kernel measured in microseconds.

**Port the extraction operator builders now, carrying their own C++ knot scan.** It would
unblock S3 immediately. Rejected because it is the two-implementations-of-one-computation
failure `design/cross_backend_types.md` was written to prevent, and #396 is in flight rather
than hypothetical. **What would change it:** #396 being abandoned or deferred past this
milestone.

**Bind `ExtractionStructView` as a C++ type.** Rejected in F2.

**Make the target a `StrEnum`, matching `LagrangeVariant`.** It would be the consistent choice
against the nearest sibling, and it is the only mechanism that keeps `ext.target == "bezier"`
true, so it would need no census at all. Rejected on two grounds, the first measured and
decisive: a `StrEnum` compared against its own member inside a `nopython` kernel is silently
always False (F5), so it forfeits the property the enum exists for; and
`design/cross_backend_types.md`'s kernel-seam table admits "an `IntEnum` value, as an integer"
and refuses "enums as strings", so it could not cross the seam either. **What would change it:**
nothing available. Both objections are about mechanisms outside this repository.

**Keep the `IntEnum` but give it string equality**, so that `ext.target == "bezier"` stays
true. Rejected because it breaks equality itself, measured: with `__eq__` extended,
`T.BEZIER == 0` and `T.BEZIER == "bezier"` are both true while `0 == "bezier"` is false, so
equality stops being transitive; and `hash(T.BEZIER) != hash("bezier")`, so a dict keyed by
the legacy string raises `KeyError` on a member that compares equal to its key. That relocates
the silent failure rather than removing it, into a place with no `__eq__` to blame. Note also
that the obvious variant -- an enum deriving from both `str` and `int` -- **does not exist**:
CPython refuses it outright (`too many data types for 'Both': {<class 'str'>, <class 'int'>}`).

## Bad practices flagged, for the user's inspection rather than for fixing here

- **`LagrangeVariant` is a `StrEnum`, and nothing stops it reaching a kernel** (F5). It does not
  today, and that is a fact about the current call graph rather than a guard. **Cost of keeping:**
  the first kernel that takes a variant and branches on a member is silently wrong, and no test of
  it can fail. **Cost of fixing:** it is a public name whose members' *values* are strings that
  appear in user code (`lagrange_variant="gauss_legendre"` works today through `StrEnum`'s own
  coercion), so converting it is the same census the target's conversion needed, on a wider
  surface. Worth its own ticket rather than a side effect; and if it stays a `StrEnum`, the reason
  should be written next to it, because "the sibling is a `StrEnum`" is exactly the argument that
  produced this note's rejected alternative.

## Epistemic status

- **Verified by reading `feat/399-extraction-cpp` at `1d2682c`**, independently re-checked
  claim by claim by a second pass: that `cpp/include/pantr/` had no `bspline/` directory
  before this branch added one. **That reading is now historical** -- see F1's amendment, which
  was verified against `proto/cpp` `7ba20d3` by reading `knots.hpp`, `space_1d.hpp` and
  grepping the tree for `cardinal_interval`, which appears only in a comment; that all twelve per-cell kernels set their accumulator from `M_0.dtype`, at the
  twelve lines cited in F3; that the identity branches skip the contraction entirely and that
  the all-identity branch is a copy; that `make_struct_view` bundles the three array tuples by
  reference; that `_KERNELS` is keyed by `(op_kind, dimension)`; that
  `lagrange_to_bernstein_1d` and `cardinal_to_bernstein_1d` are already bound in C++, so S3's
  post-multiplication is available and only the Boehm insertion and the knot scans are not.
- **Verified by execution:** that `pantr._pantr_cpp` exports both of those names; every cell of
  F5's table, and the `numba.typeof` and round-trip results beside it; that CPython refuses a
  `str`-and-`int` enum; that an `IntEnum`-with-string-equality loses transitivity and
  hash/equality consistency. The two scripts are throwaway and are not committed -- what they
  establish is a property of numba and CPython at the versions named, not of this repository,
  and the durable half is the `IntEnum` test that ships.
- **Measured 2026-09-03, by the Lagrange half of S3**, and each figure is reproduced by a test
  rather than quoted here: that `compute_lagrange_to_bernstein_1d` is entrywise non-negative
  and column-stochastic for every node family and every degree tried, which refutes the
  sentence this note used to carry; that the two backends' Lagrange operators differ by one
  unit of roundoff at degree 3 and above and agree exactly below; that the two backends'
  change-of-basis matrices themselves differ only at `float32` for the second-kind Chebyshev
  family, whose nodes dispatch. The last one is why the end-to-end parity test measures the
  matrix gap instead of assuming it away.
- **Deliberately not claimed:** an accuracy bound for the Lagrange operator against the
  B-spline basis tabulated at the nodes, above degree 2. That identity is the target's defining
  property and it is checked, but only on the dyadic family where both routes are exact.
  Composing a bound above it needs the Cox-de Boor evaluation error, the error of the
  `pow`-seeded Bernstein ratio recurrence inside the matrix, and the affine map's rounding
  amplified by `max|N'|` -- and `design/backend_parity.md`'s open question 2 records the
  transcendental category as having no vocabulary in the harness and no source consulted. Three
  unsharp terms would give a bound satisfied by any result, which Rule 3 refuses.
- **Derived, not measured:** the rounding budget in "The parity claim". It is the standard
  inner-product bound composed over stages, in the vocabulary
  `tests/_parity_harness.py` already uses; nothing has been built to compare it against, and
  the first thing S2 should do is check it over a sweep ten times the shipped one, per the
  milestone's own rule.
- **Asserted, not measured:** that the generic C++ form is fast enough at `d <= 3` that
  specialising per dimension is not worth it. No benchmark exists. If S2 finds otherwise, the
  first rejected alternative is where to look.
- **Not investigated:** whether the not-yet-public downstream consumer imports
  `_extraction_kernels` or `_extraction_helpers` directly. It matters for S2, because the
  Layer-2 rewiring changes where the dispatch happens, and the module docstring advertises the
  per-cell kernels as callable from downstream `@njit` code.
- **Not investigated:** whether `SpanwiseElementExtraction`'s `ops_1d` decompression is on
  any consumer's hot path. `design/bspline_ownership_lifetime.md` flags the same gap; it
  changes S4's laziness answer, not its lifetime answer.
