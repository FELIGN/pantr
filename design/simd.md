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

`src/pantr/bspline/_bspline_basis_core.py:262-269` documents a deliberate decision:

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
vectorize that unaided **if** they can prove the arrays do not alias: `__restrict` on the
pointers, contiguous spans, `-O3`. Writing intrinsics for it would be effort spent
reproducing what the compiler already does, and it would have to be written twice.

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
  per-point `prange` loop, with the stated reasoning (`_bspline_basis_core.py:262-269`,
  `:126-132`); that a serial twin is selected below `_PARALLEL_MIN_NUM_PTS`; that
  `first_basis_per_interval` is cached and `tabulate_basis` exposes `out_first_basis`
  (`_bspline_space_1d.py:341-378`, `:598`); and that a sibling project has a SIMD batch
  abstraction, packet kernels, and a fast-math option drawn where described above.
- **Derived:** the 0.2 flops/byte figure for the contraction kernel, and the arithmetic
  behind the expected gains.
- **Standard platform facts, stated from knowledge and worth a check:** the lane widths
  table; that AVX2 gather costs roughly 5 to 12 cycles and NEON has none; that sustained
  AVX-512 use downclocks on many Intel parts; that Apple Silicon has several NEON pipelines
  and no SVE. None of these were measured *here*, but see the next item for the third.
- **Measured 2026-08-19 on the C++ prototype's cardinal B-spline kernel** (GCC 14.4.0,
  `-O3`, 10^6 points, minimum over repeats), which is the measurement the ISA-variant
  section gates itself on:

  | target | degree 3 | degree 8 | fused sites |
  |---|---:|---:|---:|
  | baseline x86-64 | 10.47 ms | 50.73 ms | 0 |
  | `-march=x86-64-v3` | 8.69 ms | 40.04 ms | 1 |
  | `-march=native` (AVX-512 here) | 9.20 ms | 43.69 ms | 1 |

  Three things follow. The gap is **real but modest**, 1.20x to 1.27x, so the ladder is a
  decision rather than a free win. **`-march=native` is slower than `-march=x86-64-v3`** on
  this machine, which measures the "AVX-512 can be slower" hazard above on one of pantr's own
  kernels rather than asserting it. And the ISA introduces **exactly one** fused site, the
  one the parity bound is derived around, confirming by disassembly that the fused set is
  determinable from the source -- which is what the `-ffp-contract=on` choice rests on.
- **Measured, and it corrects nothing here but is worth recording next to the flag
  discussion:** with no `-march` the target has no FMA instruction at all, so
  `-ffp-contract=on` fuses nothing and the C++ and numba backends agree **bit for bit**. The
  parity tolerance this note calls for is therefore dormant on the shipped build and becomes
  live exactly when the ladder is turned on.
- **Asserted, not measured:** every expected-gain figure in the candidates table. They are
  upper bounds from lane width discounted for loop overhead and dependencies, not
  benchmarks. The purpose of the table is to rank candidates, not to predict outcomes.
- **Not investigated:** whether the current kernels already auto-vectorize under `-O3`. That
  is the first measurement to take, and it could change the priority order substantially: if
  the blocked contraction vectorizes unaided, most of this note reduces to "add `__restrict`
  and block the loop".

## Open questions

1. Does the blocked contraction auto-vectorize? If yes, no batch abstraction is needed at
   all for the main kernel and xsimd never enters the dependency list.
2. Is the block width `W` a compile-time constant per ISA variant, or a runtime parameter?
   Compile-time composes with the multi-ISA build below, which already compiles the module
   once per ISA, and lets the remainder handling be resolved at compile time.
3. Should scattered-point evaluation be a documented slow path, given that it is the only
   case needing a gather and NEON cannot do one? This is the same
   tensor-product-versus-scattered question that `design/large_data_fitting.md` and
   `design/user_functions_across_the_boundary.md` both raise, and the answer should be the
   same in all three.
4. `_PARALLEL_MIN_NUM_PTS` is a threshold. Is it derived or measured, and does a SIMD
   variant need its own? A block-width dispatch adds a second threshold, and two unjustified
   thresholds interacting is exactly the kind of thing that becomes untraceable.
