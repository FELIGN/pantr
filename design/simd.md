# SIMD: AVX and NEON

**Status:** design note for the C++ port. Nothing here is implemented.
**Date:** 2026-08-17.
**Scope:** where explicit vectorization pays in pantr, what enables it, and the three
hazards that bite. Not GPU, which was assessed separately and declined.
**Companions:** `design/bezier_extraction_api.md` and `design/large_data_fitting.md`, whose
shared banded-operator kernel is the main target here.

**Validated against:** pantr **0.7.0** (`main`, tag `v0.7.0`), 2026-08-19. Line numbers
below refer to that tree.

## Why this is a different question from the GPU

The GPU was declined because of transport: 537 MB over PCIe costs about 54 ms round trip
against 20 to 30 ms of host computation. That is a fixed cost with nothing to amortize it
against.

**SIMD has no transport cost.** The data is already where the vector unit needs it. So the
only question is whether a given kernel is limited by arithmetic or by memory, and there is
no threshold to clear before it can pay at all.

## But SIMD does not help a bandwidth-bound kernel

If a loop is waiting on memory, widening the arithmetic changes nothing. The same low
arithmetic intensity that killed the GPU case also bounds what SIMD can deliver.

The central kernel, applying a banded 1D operator along axis `d`, does `p + 1` fused
multiply-adds per output element while reading `p + 1` inputs and writing one. At `p = 3`
in `float64` that is about 8 flops per 40 bytes, so **0.2 flops per byte**. Streaming from
RAM, that is firmly bandwidth-bound and vectorizing it buys perhaps 1.0 to 1.3×.

## Blocking is what makes SIMD pay

This is the point to take away.

With the tiling proposed in `design/bezier_extraction_api.md`, the working set fits in L1
or L2. The same kernel then reads from cache rather than RAM, becomes **arithmetic-bound**,
and SIMD delivers close to the vector width.

So blocking and vectorization are not two independent optimizations to be ranked. Blocking
without SIMD leaves the vector units idle; SIMD without blocking leaves them waiting on
memory. Either one alone is worth little, and the pair is worth several times.

Practical consequence for sequencing: **implement the blocked kernel first, then vectorize
it.** Vectorizing the unblocked kernel would measure as a disappointment and could easily
be mistaken for evidence that SIMD does not help here.

## Candidates

| kernel | vectorizes? | expected gain |
|---|---|---|
| banded contraction, **blocked** | ideally: trailing index contiguous, no dependencies, no gather | 2 to 4× on AVX2 `float64` |
| banded contraction, unblocked | equally well, but waits on memory | 1.0 to 1.3× |
| Cox-de Boor / de Boor **across points** | yes, once the points in a block share a span | 2 to 4× |
| Cox-de Boor **within one point** | no: triangular recurrence, dependent chain of length `p` | nothing |
| root finding, packet-with-masking | yes, and a sibling project already did it | adopt the approach rather than reinvent |
| `locate` / point inversion across points | yes, masking for unequal iteration counts | moderate |
| BVH traversal | divergent; only with batched queries | weak |
| spline derivative | 3 flops per output, memory-bound | nothing |

The pattern is consistent: **vectorize across independent items, never inside one item.**
Every recurrence in this library (Cox-de Boor, de Boor, de Casteljau, knot insertion) has a
dependent inner chain, so the lane axis is always the point index, the element index or the
polynomial index.

## The existing design is right for threads and it blocks SIMD

This is a real tension and it deserves stating properly rather than being papered over.

`_compute_basis_nurbs_book_impl` in `src/pantr/bspline/_bspline_basis_kernels.py` documents a
deliberate decision:

> *Each point's span search and Cox-de Boor evaluation are independent, so both are fused
> into a single `prange` loop over evaluation points (span search alone does not parallelize
> well enough on its own to be worth a separate pass ...)*

and the span-search helper is `inline="always"` into the kernel (`:126-132`).

That is the **correct** choice for thread parallelism. A separate span-search pass would be
a memory-bound pass over all points that parallelizes poorly, so fusing removes it.

It is also, unavoidably, the choice that blocks SIMD: one `prange` iteration holds one
point, and one point's recurrence does not vectorize.

### The resolution, which does not require unfusing

Keep one pass, but have each iteration process a **block of `W` consecutive points**, `W`
being the vector width. Inside the block: span search for the `W` points, then Cox-de Boor
across them.

The obvious objection is that the `W` points may lie in different spans, which would force
a gather. It mostly does not arise, and the reason is worth knowing:

- **For a lattice** (image fitting, tabulation on a grid) the points are already sorted by
  parameter, so `W` consecutive points share a span except at span boundaries.
- **For per-element quadrature** the points are generated element by element, so a block is
  span-homogeneous by construction and the span is known analytically, with no search at all.
- The block that straddles a boundary is handled as a **scalar remainder**, which costs
  `O(number of spans)` scalar points in total rather than `O(number of points)`.

So no sort, no extra pass, no layout change, and no gather. For genuinely scattered points
the gather is unavoidable, and there the asymmetry matters: AVX2 gather costs roughly 5 to
12 cycles, and **NEON has no gather instruction at all**. That is an argument for treating
scattered points as a separate, slower path rather than as the general case.

The infrastructure for this dispatch already exists in spirit: the kernels already choose
between a parallel and a serial twin at `_PARALLEL_MIN_NUM_PTS`. A block-width dispatch is
the same shape.

Also already available: `first_basis_per_interval()` is computed and **cached**
(`_bspline_space_1d.py:341-378`, with `writeable = False`), and `tabulate_basis` exposes
`out_first_basis` (`:525`). So the per-span first-basis index does not need recomputing to
drive span-homogeneous blocking.

## Auto-vectorize first

The blocked contraction inner loop is `for j: out[i,j] += a * in[k,j]`. GCC and Clang
vectorize that unaided at `-O3`, and **they do not need to prove the arrays do not
alias**: GCC compiles both a vector and a scalar version of the loop and chooses between
them with a runtime check. Writing intrinsics for it would be effort spent reproducing
what the compiler already does, and it would have to be written twice.

`__restrict` still buys something, and where it has to go is not obvious. On locals
initialised from `std::span::data()` it changes no generated code at all; only on a
helper's **parameters** does it remove the versioning.

What removing it is worth is **not a per-compiler constant, and reading it as one would
be wrong**: measured, it is a modest gain under GCC at every level, and under Clang it
is a wash at the baseline, a small **regression** at `x86-64-v3`, and a gain of GCC's
order at `native`. So it is an ISA-level and code-generation effect rather than a
property of the compiler, and anything that adopts it has to be measured at each level
it ships rather than adopted once. Both halves are measured in
`scripts/measure_bezier_degree_templating.py`, which prints the vectorizer's own verdict
beside what removing the versioning bought.

What does **not** auto-vectorize, and is where a batch abstraction earns its place:

- the span-homogeneous point block, which needs the blocking transformation first;
- packet root finding, which needs masking;
- anything with data-dependent control flow.

So the sequence is: auto-vectorize, **measure**, and reach for an explicit batch type only
where the compiler demonstrably fails. A `-fopt-info-vec-missed` (GCC) or
`-Rpass-missed=loop-vectorize` (Clang) report is the evidence to act on, not intuition.

If a batch type is needed, **xsimd** fits the constraints: header-only, no dependencies,
covers SSE / AVX / AVX-512 / NEON / SVE, one `FetchContent` entry, and compatible with the
C++20 baseline of `D4`. `std::simd` is targeted at C++26 and is not available.
Hand-rolling one is the alternative and is roughly a few hundred lines.

## Shipping several ISA variants, if it ever proves worth it

The dispatch question is settled in shape: compile the extension module several times
(baseline, `x86-64-v3`, `x86-64-v4`), pick one at import, and keep arm64 as a single module
because NEON is the AArch64 architectural baseline and needs no probe.

**But do not build any of it until the measurement in open question 1 says the gap is
real.** Tripling the shipped extension is paid by every user on every install, including the
ones on baseline hardware who gain nothing, and it is justified only by a measured
difference on pantr's own kernels.

When the time comes, four things about this mechanism are non-obvious enough to be worth
stating in advance, because each of them is a way to get it subtly wrong.

**The feature probe must be a separate module compiled at the toolchain baseline.** This is
the heart of the design and the reason it is not simply a function inside the extension: you
cannot ask an AVX-512 module whether AVX-512 is available, because *importing it* may already
fault. The probe must therefore link no library code, carry no `-march`, and be safe to import
on any CPU of the target architecture.

**A CPUID feature bit is not sufficient evidence.** The operating system must also have
enabled the matching vector state in `XCR0`, or the wider registers are not preserved across
a context switch. So the probe has to check `OSXSAVE` and then `XGETBV` before believing any
feature bit, and it must **fail closed**: anything it cannot confirm counts as absent. On GCC
and Clang, `__builtin_cpu_supports` folds this check in already; a hand-rolled MSVC path does
not and must do it explicitly. Related: a feature *level* like `x86-64-v3` is a bundle, not
one flag, so testing AVX2 alone would select a module the CPU does not fully implement.

**An explicit override must not fall through to the next candidate.** If an environment
variable names a variant and that variant is missing, the import should fail rather than
quietly load a different one. A silent downgrade makes every A/B measurement untrustworthy,
which is precisely what the override exists to enable.

**The variant set follows the *target* architecture, not the build host.** This bites when
cross-compiling: on Apple, `CMAKE_SYSTEM_PROCESSOR` still reports the host, so a macOS x86_64
wheel built on an arm64 runner would otherwise get the arm64 ladder. Whatever the wheel
builder actually sets for the target architecture has to take precedence, and a universal
binary cannot carry two ladders at all.

One further rule, which belongs in `design/toolchain_requirements.md` and is repeated here
because it is easy to violate exactly at this point: any flag participating in a numerical
claim (floating-point contraction, for instance) must be set on the **interface target** so
that it reaches every variant. Set per-variant, the variants will disagree numerically with
one another, and the resulting bug will look like a dispatch bug.

## Lane widths are not equal, and it changes the expected payoff

| ISA | width | `float64` lanes | `float32` lanes |
|---|---|---|---|
| NEON (Apple Silicon, AArch64 baseline) | 128 bit | **2** | 4 |
| AVX2 | 256 bit | 4 | 8 |
| AVX-512 | 512 bit | 8 | 16 |

So on Apple Silicon the SIMD ceiling for `float64` is **2×**, not 4× or 8×. Apple's cores
compensate with several NEON pipelines, so real throughput is obtained through instruction
level parallelism (multiple independent accumulators) rather than through width. A kernel
written with one accumulator will underperform there even when perfectly vectorized. SVE is
not available on Apple Silicon; it appears only on server-class ARM.

**Corollary that reinforces the fitting note:** `float32` doubles the lane count everywhere.
For the image-fitting case that is a second, independent reason to support `float32`, on top
of halving the memory. Two lanes become four on NEON, four become eight on AVX2.

## Three hazards

### AVX-512 can be slower

On many Intel parts, sustained AVX-512 use lowers the clock, so a 512-bit kernel can lose to
a 256-bit one in a mixed workload. This has to be measured per kernel, not assumed from the
width. A sibling project ships the `v4` variant but cannot execute it on standard CI runners
and tests it under the Intel SDE emulator instead, which is worth knowing before planning to
validate it in the cloud.

### FMA breaks bit-level parity, and that touches `D1`

**Measured caveat first:** at the *baseline* target there is no FMA instruction at all, so
nothing fuses and parity can be exact and provably so. This hazard applies from
`-march=x86-64-v3` upward. See `design/build_findings.md` for the disassembly evidence
and for the two corrections the resulting bound needed.

`a * b + c` fused and unfused differ in the last bits. The project's own rule already says
bit-exactness is the wrong target, but the **parity tolerance against the Numba backend has
to absorb the FMA difference explicitly**, and it is derivable: on the order of one ulp per
fused operation, accumulating with the operation count of the kernel.

If that is not derived up front, the first parity test on a vectorized kernel fails and the
cause is not obvious from the failure. This should be written into the parity harness as a
derived bound with its derivation, not discovered.

### Never `-ffast-math`

A sibling project draws the line in a place worth copying: it offers
`-funsafe-math-optimizations` behind an explicit option but **never `-ffast-math`, and never
`-ffinite-math-only`**, on the grounds that outward-rounded interval arithmetic depends on
signed infinities surviving.

pantr's reason differs but the conclusion is the same: `-ffast-math` permits reassociation of
sums, and reassociation invalidates the error bounds that every derived tolerance in the
library assumes. A tolerance derived under one association order is not a tolerance under
another. It is not a performance option, it is a silent correctness change.

## Epistemic status

- **Verified by reading the code:** that span search is deliberately fused into the
  per-point `prange` loop, with the stated reasoning (the docstrings of
  `_compute_basis_nurbs_book_impl` and `_find_span_and_first_basis_point`, both in
  `_bspline_basis_kernels.py`); that a serial twin is selected below
  `_PARALLEL_MIN_NUM_PTS`; that
  `first_basis_per_interval` is cached and `tabulate_basis` exposes `out_first_basis`
  (`_bspline_space_1d.py:341-378`, `:598`); and that a sibling project has a SIMD batch
  abstraction, packet kernels, and a fast-math option drawn where described above.
- **Derived:** the 0.2 flops/byte figure for the contraction kernel, and the arithmetic
  behind the expected gains.
- **Standard platform facts, stated from knowledge and worth a check:** the lane widths
  table; that AVX2 gather costs roughly 5 to 12 cycles and NEON has none; that sustained
  AVX-512 use downclocks on many Intel parts; that Apple Silicon has several NEON pipelines
  and no SVE. None of these were measured here.
- **Asserted, not measured:** every expected-gain figure in the candidates table. They are
  upper bounds from lane width discounted for loop overhead and dependencies, not
  benchmarks. The purpose of the table is to rank candidates, not to predict outcomes.
- **Measured, and it did change the priority order** (issue #379, measured by
  `scripts/measure_bezier_degree_templating.py`): the n-d Bézier contraction's inner loop
  already auto-vectorizes under `-O3` on both compilers at every ISA level tried, so no
  batch abstraction is needed for it. What the measurement also found is that this settles
  less than the question expected, because the loop that vectorizes is not the one that
  costs. See open questions 1 and 2.

## Open questions

1. ~~Does the blocked contraction auto-vectorize?~~ **Answered, measured.** Yes, unaided,
   on GCC and Clang and at every ISA level tried, with the width rising as the level does.
   It is the only loop of `contract_leading_axis`
   (`cpp/include/pantr/bezier/evaluate.hpp`) that vectorizes, and correctly so: the
   zeroing loop becomes a `memset` and the loop over terms is the accumulation, which must
   keep its order. So no batch abstraction is needed for this kernel and xsimd stays out
   of the dependency list on its account. The evidence is the compilers' own reports, in
   section 1 of `scripts/measure_bezier_degree_templating.py`.

   **The answer settles less than the question assumed, and that is the finding.** The
   loop that vectorizes is not where the time goes. Both n-d schedules contract their
   *last* direction against a trailing block of `cp_size` values, and that direction
   carries nearly all of the calls and nearly all of the element-operations, so in pantr's
   own workload the vector body is usually never entered and the call is prologue plus
   scalar remainder. Widening the arithmetic cannot help a loop that does not reach its
   vector body. Fixing that loop's trip count at compile time can, which is question 2.
0. ~~Does split mode compose with several ISA variants?~~ **Answered, measured.** Two
   frontends at different `-march` against one backend both import in the same process
   and both work. See `design/build_findings.md`.
2. ~~Is the block width `W` a compile-time constant per ISA variant, or a runtime
   parameter?~~ **Answered, measured: compile-time, and by a wide margin at the widths
   pantr actually produces.** Fixing the contraction's *inner* trip count is worth several
   times over at a trailing block of a few elements, and the margin decays towards nothing
   as the block grows past the vector width times the unroll factor the compiler chose,
   which is the crossing to state it at rather than a number of elements. Stating it that
   way is what keeps it from being a fitted constant: it tracks a code-generation
   decision, which the compiler makes from the ISA and not from the host. **That last
   step is an inference and not a measurement** -- the sweep ran on one machine across
   three ISA levels and two compilers, so a second CPU of the same ISA level was never
   checked, and it is the cheap check to run before anything is built on it.

   Fixing the *outer* trip count, the degree, **does not pay at any degree** on either
   compiler: its median stays close to unity everywhere -- every block size, both scalar
   types, every ISA level -- with no trend in the degree, and the individual cells that
   do rise above it are a small fraction of what fixing the block width gives at the very
   same shape. That was issue #379's own question and the answer is no. So the thing to
   specialize is the trailing block width and the key is `cp_size`, not `p` -- which also
   makes the closed set much smaller than a degree ladder would have been.

   Three things this does **not** settle, and they belong to the follow-up rather than
   here:

   - ~~whether the win survives at the level of `evaluate`.~~ **Measured by issue #481:
     on a lattice, yes; on a point array, no.** `scripts/measure_bezier_cp_size_dispatch.py`
     times `Bezier.evaluate` end to end against a null arm. With a switch on the block
     size inside `contract_leading_axis`, built on the tree before #481's change, a
     `PointsLattice` at two and
     three dimensions gains outside the setup's spread in nearly every cell at eight and
     thirty-two points per direction, more at the finer grid and the higher dimension, and
     in almost none at two. A point array gains a few percent at most, and some of its
     cells got slower. That slowdown is **code placement, not the dispatch's arithmetic**.
     The switch made GCC stop inlining the kernel into `evaluate`, which moved its hot
     loops, and the measuring CPU is a Skylake-SP, the family whose jump-conditional-code
     erratum
     mitigation stops a branch that crosses or ends on a 32-byte boundary from being
     cached as decoded micro-ops. In a C++ driver linked against the binding object,
     assembling both builds with `-Wa,-mbranches-within-32B-boundaries`, which pads such
     branches off the boundary, removed the slowdown. That the mitigation is active is
     inferred from the microcode revision rather than read from Intel's notes, and it is
     one CPU and one experiment: the mechanism for this host, not a rule.

     So the specialization shipped for the lattice only, in the pull request for #481:
     `evaluate_on_lattice` switches once per direction into
     `detail::contract_lattice_direction`, and `evaluate`
     calls the runtime kernel as before. The machine code of `evaluate` is
     instruction-identical to the tree before, on GCC and Clang at the baseline,
     `x86-64-v3` and `x86-64-v4`; results are bit-identical over the sweep with a
     reversed-order control that disagrees; and `--against` re-times the two commits. The
     figures come from `scripts/measure_bezier_cp_size_dispatch.py` and are recorded in the
     pull request for #481; `cpp/tests/test_bezier_evaluate.cpp` keeps the bitwise
     agreement as a test.

     The erratum finding reaches past this ticket. The padding also made the shipped
     point-array path itself clearly faster in that driver, and any change to that
     translation unit can move `evaluate`'s branches across a boundary, so a
     before-and-after timing of it on this CPU family should check that the timed
     function's code and alignment moved before reading a few percent either way. Whether
     the build should pad branches is not decided here and has not been measured on
     another CPU.
   - what to do about the cost, which is real. Instantiating a grid of widths multiplies
     compile time and emitted code for one translation unit, and then multiplies again by
     the ISA-variant count this note plans for above. For the lattice-only closed set of
     four, #481 reports compile time and emitted text of `cpp/bindings/bezier.cpp` before
     and after at the three ISA levels.
   - whether `__restrict` on the kernel's parameters is the cheaper half of the same win.
     It needs no instantiation at all, and it helps under GCC roughly where the
     specialization stops helping. See "Auto-vectorize first".
3. Should scattered-point evaluation be a documented slow path, given that it is the only
   case needing a gather and NEON cannot do one? This is the same
   tensor-product-versus-scattered question that `design/large_data_fitting.md` and
   `design/user_functions_across_the_boundary.md` both raise, and the answer should be the
   same in all three.
4. `_PARALLEL_MIN_NUM_PTS` is a threshold. Is it derived or measured, and does a SIMD
   variant need its own? A block-width dispatch adds a second threshold, and two unjustified
   thresholds interacting is exactly the kind of thing that becomes untraceable.
