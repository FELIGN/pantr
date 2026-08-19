# Deep review: the C++ infrastructure prototype

**Target:** branch `feat/cpp-infrastructure`, to be merged into `proto/cpp`. The C++ port's
skeleton: CMake with configure-time gates, FetchContent for Kokkos mdspan and Eigen, a
nanobind extension, one ported kernel, a dual-backend Python seam, and a parity harness.

**Date:** 2026-08-19.
**Machine:** shared Linux server, 20-CPU cap. conda-forge GCC 14.4.0 and Clang 18.1.8,
CMake 4.4.2, Ninja 1.13.2, CPython 3.14.6, numba 0.65.1, numpy 2.4.6, nanobind 2.14.0.

## Lenses

**Run:** API and contracts (`api-auditor`) · architecture, adversarial (`architect`) ·
mathematical claims, attack (`mathematician`) · tolerance policy (`tolerance-hunter`) ·
dependency assumptions (`surveyor`, on nanobind) · test coverage (`test-writer`) · style
conformance (`stylist`) · three regional `reviewer`s over the build files, the C++ tree, and
the parity harness.

**Skipped, with reasons:** `numerics-shakedown`, because parity against the oracle is
bit-exact across degrees 0-16 in both dtypes and in and out of domain, the kernel is a
verbatim port introducing no new algorithm, and the protocol writes to the repo.
`optimizer`, because the performance question was measured directly during the work and its
one finding (`-Os`) was fixed; the suite runs in 1.5 s. `doc-writer`, because the prototype
has no docs-tree entry and `cpp/README.md` is its documentation, covered by the regional
reviewers.

## Findings

Ranked. `C` critical, `I` important, `R` recommended, `S` suggested, `N` nitpick,
`U` unconfirmed. Round 1.

### Critical

**C1-1 · `pantr::value_type_t` does not compile for any type it exists for.**
`cpp/include/pantr/core/scalar.hpp`. The alias spells its lookup `detail::value_of(...)`, a
*qualified* call, which suppresses ADL and so never finds a scalar type's hidden-friend
`value_of` — the exact mistake the file spends ten lines warning about immediately above.
Confirmed independently by three lenses, each compiling it: `value_type_t<Dual1>`, against
the project's own test fixture, fails to compile. It survived because the alias has zero
uses in the tree. **Failure scenario:** the first person to write the `Dual<T,N>` of
`design/automatic_differentiation.md` reaches for the trait the design rests on and gets a
template error from a header they did not write. **Direction:** the two-step ADL pattern the
neighbouring `has_value_of` concept already gets right; four lines, and a `static_assert` so
it cannot regress unused.

**C1-2 · The binding turns a bad argument into memory corruption, reachable from one line
of Python.** `cpp/bindings/pantr_cpp.cpp`. Reproduced here and by three lenses:
`_pantr_cpp.tabulate_cardinal_bspline_1d(-1, pts, out)` exits with SIGSEGV, as does a call
whose `out` is smaller than `(points.size, degree+1)`. `static_cast<std::size_t>(degree)` in
the kernel turns `-1` into `SIZE_MAX`. The numba oracle returns normally on the same input.
**Failure scenario:** CLAUDE.md records that a downstream consumer imports pantr's private
symbols; the extension is importable and its `.pyi` invites a positional call with an
unconstrained `degree: int`. **Direction:** validate at the binding, which is Layer 2's C++
half — not in the kernel, which stays Layer 3. Three comparisons against an O(n·p²) kernel.

**C1-3 · The B1 regression test does not test the fix it claims to pin.**
`tests/test_cpp_parity.py`. Its docstring claims to pin both halves of the non-contiguous-`out`
fix. A lens stripped `nb::arg("out").noconvert()` from the binding and the suite still passed
119/119: the test goes through the public API, and the adapter in `pantr._backend` buffers a
strided `out` before nanobind ever sees one. **Failure scenario:** the `.noconvert()` is
removed in a later refactor and nothing notices, restoring a silent wrong-answer bug.
**Direction:** call the raw binding directly and assert `TypeError`; correct the docstring to
claim only what it tests.

**C1-4 · The accuracy bound is violated by six orders of magnitude, one degree-list entry
away from turning the suite red against correct code.** `tests/test_cpp_parity.py`,
`_companion_bounds`. Every rounding is modelled as purely relative (`u·|x|`); in the gradual
underflow range a rounding commits an *absolute* error floored at the smallest subnormal.
Measured on the module's own oracle points, float32, error over bound: 0.353 at degree 8,
**459 645 at degree 12**, **1 048 576 at degree 16**. `test_matches_the_exact_rational_oracle`
escapes only because its degree list stops at 8 while `DEGREES` runs to 16. **Failure
scenario:** the next contributor adds degree 12 — the natural thing to do — and the suite
fails against a correct kernel, which is the most expensive kind of red. **Direction:**
Higham's model with the `η` term, `η_fmt = smallest_subnormal`, added per accumulator
rounding and per store.

**C1-5 · `absolute_tolerance` has the same missing floor, at harness level.**
`tests/_parity_harness.py`. Returns `2·γ·amplification`, purely relative. At float32,
degree 11, `u = 1e-3`, the tolerance is 42.7× smaller than one subnormal ulp there, so a
one-ulp disagreement would be rejected as a bound violation. Dormant only because parity is
currently bitwise. **Failure scenario:** it becomes live the day `-march=x86-64-v3` lands,
and **all five following PRs inherit it** regardless of what their kernels do.
**Direction:** the same floor, expressed with the counts `Roundings` already carries — no new
API surface. Reported inflation on every previously valid case is exactly 1.000000×.

### Important

**I1-1 · `use_backend` is not thread-safe, in a design that invites threading.**
`src/pantr/_backend.py` mutates a module global under a context manager. Demonstrated
deterministically: two overlapping blocks in two threads leave the process permanently on
`CPP`; a thread reads `NUMBA` while the main thread is inside a `CPP` block. The binding
releases the GIL specifically so callers can thread at the Python level, and `pytest -n` uses
processes so the suite cannot see it. **Direction:** `contextvars.ContextVar`, six lines,
same shape; state the semantic change (a spawned thread inherits the process default).

**I1-2 · A masked array makes the two backends disagree at the public API.** Measured:
numba raises `NumbaTypeError`, the C++ path accepts it through the buffer protocol, silently
ignores the mask, and returns numbers. That is exactly what `_backend.py`'s docstring
promises cannot happen. **Direction:** reject it in Layer 2; the numba behaviour is correct.

**I1-3 · A wrong derivation in a C++ test, and the test does not discriminate what it
claims.** `cpp/tests/test_cardinal_bspline.cpp`. The stated store count is `degree + 1`; the
kernel performs `n(n+3)/2` — 65 at degree 10, not 11. The bound holds by a different argument
(each stage is a stochastic map over non-negative values, so it adds at most `eps32` in the
1-norm) and the measured error is *flat* in degree while the stated bound grows linearly.
Separately, a mutant with `accumulator<float>::type = float` — disabling the whole point of
the trait — still passes this test at every degree 0-20; the real protection is a
`static_assert` in a different file.

**I1-4 · nanobind strips the module, so nothing can be profiled or backtraced.**
`nm` on the shipped `.so` reports "no symbols". In a prototype whose stated job is
measurement and provenance, `perf` cannot name the kernel and C1-2's segfault gives no
backtrace. **Direction:** `NOSTRIP` beside `NOMINSIZE`.

**I1-5 · The `<mdspan>` toggle is structurally unreachable.** The probe body uses
`view[1, 2]`, a C++23 *language* feature, while the probe is pinned to `-std=c++20` where
that is the comma operator. `PANTR_HAS_STD_MDSPAN` can therefore only ever be OFF, even on a
libstdc++ that does ship the header. **Direction:** probe `__cpp_lib_mdspan`, or use the
`std::array` subscript form.

**I1-6 · The offline-escape gate is vacuous on every from-scratch run.**
`scripts/ci_local.sh` runs `gates` before `cxx`, and the offline check needs
`build/gcc/_deps` populated, which only `cxx` does. So the one workflow meant to prove the
escape works always SKIPs it. Honest, not a lying pass, but it proves nothing.

**I1-7 · Four files misattribute the validation site.** `cpp/bindings/pantr_cpp.cpp` says the
extent check "is checked in `src/pantr/_backend.py`". It is in
`src/pantr/basis/_basis_1D.py`; `_backend.py`'s adapter explicitly disclaims validation.
C1-2 makes this material — a reader chasing the guarantee looks in the wrong file.

**I1-8 · `cardinal_bspline.hpp` calls the nanobind wrapper "the validating wrapper".**
False, and it points a reader at the path that segfaults.

**I1-9 · `requires_cpp_backend` is dead code that reintroduces the named silent-skip trap.**
`tests/_parity_harness.py` exports it, nothing uses it, and it implements a plain skip with
no `PANTR_REQUIRE_CPP` escalation. It is the obvious symbol for the author of PR 2 to reach
for, and CLAUDE.md names that failure mode explicitly.

**I1-10 · Two "measured" docstring claims do not hold.** The "96 unit roundoffs" figure is
off by 2× in the file's own `u = eps/2` units (~192), and `OUTSIDE_DEGREES`'s stated overflow
rationale is falsified by direct computation — the amplification *decreases* with degree at
fixed `|u|`.

### Recommended

**R1-1 · `matrix.cc` is dead in the workflow**, so `CMAKE_C_COMPILER` is never pinned while
Eigen's `project()` enables C implicitly. The Clang leg does not build entirely under Clang.

**R1-2 · The `-ffinite-math-only` gate is weaker than its `-ffast-math` sibling** — it
asserts only that configure failed, so an unrelated failure reads as the gate firing.

**R1-3 · `ci_local.sh` writes to fixed `/tmp` paths** that collide between concurrent runs on
this explicitly shared machine.

**R1-4 · Build policy leaks into the library's usage requirements.** `pantr_core` links an
interface target carrying `-Werror` and the whole warning set, so anything linking
`pantr::core` inherits pantr's warning discipline. Latent today; a support burden the day the
core is installable.

**R1-5 · `CMakePresets.json` pins `jobs: 20`**, a machine-specific constant in a committed
file, which already forced the workflow to bypass the presets — the exact drift presets exist
to prevent.

**R1-6 · `pyproject.toml` uses an already-deprecated scikit-build-core metadata API**
(`tool.scikit-build.metadata`, which warns at build time) with no upper bound on the
requirement, so a future release could silently break version single-sourcing.

**R1-7 · Three spellings of "evaluation points"** across one vertical slice (`pts`, `t`,
`points`); the adapter uses two of them in eight lines.

**R1-8 · The seam's type was cut to fit a kernel with no serial twin.**
`_tabulate_basis_1D_impl_helper` takes a *pair* (parallel plus serial) and dispatches on
`_PARALLEL_MIN_NUM_PTS`. Porting Bernstein next forces a second return shape for the same
concept.

*Corrected 2026-08-20.* This finding originally read "cardinal B-spline is the only basis
kernel with no twin", and that is backwards: **Bernstein is the only 1D basis tabulation
that has one.** Cardinal B-spline, Lagrange and Legendre all lack a twin, and
`_basis_1D.py` passes `core_func_serial=` at exactly one of its four call sites. The
conclusion is unaffected and better supported -- the bare callable fits three of the four
kernels by coincidence, not one.

### Suggested and nitpick

**S1-1** No `.clang-format`/`.clang-tidy` in the tree, and no C++ format check anywhere; the
four new headers are hand-formatted and consistent, which is luck rather than mechanism.
**S1-2** `pantr::at` takes two adjacent `std::size_t` — Core Guidelines I.24, and a swap here
silently transposes output. **S1-3** The Layer 3 kernel is not `noexcept`. **S1-4** The
binding's header comment claims `nb::arg` is a compile-time tag in 2.x; it is a runtime
struct there, and only a tag in 3.x. The conclusion holds, the reason does not.
**N1-1** `accumulator_t` has no standalone docstring. **N1-2** `_ACTIVE` and `_CPP_AVAILABLE`
document the identical "resolved once at import" behaviour asymmetrically.

### Unconfirmed — worth a look

**U1-1** The claim that GCC implements `-ffp-contract=on` distinctly from `fast` only since
GCC 14 could not be settled here (no GCC 12/13 toolchain). It is load-bearing: the parity
bound's derivation rests on the fused set being readable from the source. Cheapest check: a
two-statement fusion probe on GCC 13 versus 14, diff the assembly.
**U1-2** Compile time for one TU carrying every binding, times N ISA variants, times two
scalar types. Not measurable at one kernel; the escape (explicit instantiation in a `.cpp`)
stays available.
**U1-3** Intermittent parity failures observed by one lens (`sums=0`, then `109 skipped`) that
vanished on rerun, attributed to a concurrent editable rebuild racing the test process — the
extension is a symlink into `build/skbuild`. Almost certainly environmental, and I caused at
least one instance of it myself, but worth confirming no real reentrancy defect hides behind
it before dismissing.

## Where the code refuted a design note

Recorded separately because the notes are sealed and the user rules on them.

1. **`toolchain_requirements.md` open question 1 is answered, uncomfortably.** GCC 10 and
   Clang 10 both pass the concepts probe *and* compile and correctly run the kernel. The
   note's own rule is that the known-broken list "should only grow from observed failures,
   never from speculation", and there is no observed failure behind the Clang < 14 entry.
2. **The `<mdspan>` toggle, as implemented, is a permanent OFF** (I1-5). The note's
   classification is right; the probe cannot express it. Separately, `mdspan.hpp`'s stated
   reason for the absence was wrong (it is a libstdc++ implementation gap, not a
   standard-version gate) and has been corrected in this branch.
3. **`automatic_differentiation.md`'s "templates are enough" is not yet earned by the code.**
   `Real` certifies a legitimate forward-mode `Dual` that the kernel then rejects with a
   template dump — the concept is a strict subset of the kernel's actual requirements — and
   `value_type_t` does not compile for it at all (C1-1).
4. **`large_data_fitting.md`'s float64-accumulation rationale does not apply to this
   kernel.** The state vector round-trips through float32 storage every stage, and the
   measured float32 error is flat in degree, not `sqrt(m)·eps32`. The rule is right for the
   fitting kernels it was written for; `accumulator_t` imports its justification into a
   kernel where it does not hold.
5. **`simd.md`'s AVX-512 hazard is now measured** on a pantr kernel rather than asserted:
   `-march=native` (AVX-512 here) is slower than `-march=x86-64-v3`.
6. **`design/isa_dispatch.md` is cited by two notes and does not exist.** Its content appears
   to have been folded into `simd.md`.

## Design questions: ruled on

Decided 2026-08-19, after the review. Four were applied in this PR and four are recorded
here with their ruling.

**Applied.**

- **Rank-generic `span_nd<T, Rank>`**, with `span2d = span_nd<T, 2>` so no call site moved.
  The layout stays `layout_right` until a kernel needs a stride.
- **`std::constructible_from<T, double>` on `Real`**, so the concept is the kernel's contract
  rather than a subset of it, pinned by a fixture the concept must reject.
- **The version floor**, set to 10 for GCC and Clang alike from a measurement, and covered by
  `scripts/ci_local.sh` on every run rather than by a one-off.
- **A `pull_request` trigger scoped to `proto/cpp`**, after the workflow turned out to be
  undispatchable, with the run cut to one compiler and cached.

**Approved, not yet applied.** Each is cheaper now, with one kernel, than after the second:

- **A · Move `accumulator_t` to `core/scalar.hpp`** and key it on `value_type_t` rather than
  on the storage type. It encodes a project-wide policy from a basis-specific header, and
  the same rule currently answers differently depending on whether the scalar is a built-in.
- **B · A `CoreKernels(parallel, serial)` record** for the seam, `serial=None` allowed. The
  consumer already takes a pair and dispatches on `_PARALLEL_MIN_NUM_PTS`, and Bernstein is
  the only 1D basis tabulation that has a serial twin -- so the bare callable fits the
  kernel ported so far by coincidence. Porting Bernstein forces a second return shape for
  one concept. (See the correction under R1-8: this was first written the other way round.)
- **C · Invert the `_backend` import cycle**: policy in `_backend`, catalogue beside each
  kernel. Removes two `noqa: PLC0415`, the `TYPE_CHECKING` import and the cycle itself, and
  lets `lint-imports` hold it with a one-line contract.
- **D · Make `pantr::at` variadic and drop its `#if`.** Its stated premise is false: Kokkos
  0.6.0 provides `operator[](const std::array<I, rank()>&)` unconditionally, as C++23 does,
  so one spelling serves both branches. Removing the branch also makes `at` rank-generic,
  which is what `span_nd` now needs.

- **E · Split `Backend` from an `IsaVariant` axis.** Two orthogonal questions -- which family
  and which build of that family -- and folding the second into the first would multiply
  `available_backends()`, the parse and every `if chosen is Backend.NUMBA` by their product.
  Deciding it now is free; deciding it later changes the accepted values of an environment
  variable, which is user-facing surface the moment anyone puts it in a script. The
  measurement that gated stage 2 has now been taken, so that day is nearer than it was.
- **F · Mark `pantr._backend` explicitly unstable** in its module docstring: scaffolding for
  the duration of the port, removable when the port ends. Not made public like
  `pantr._parallel`, despite being maintained to the same standard -- going public commits
  to maintaining it past the point where its reason to exist has gone, and CLAUDE.md warns
  that a downstream consumer already imports pantr private symbols where CI cannot see it.

**Nothing from this review is left undecided.** Every finding was applied, and every design
question was ruled on.

## Design questions left open for the user

Not findings. Each is cheap now and expensive per kernel added.

- Rank-generic `span_nd<T, Rank>` now, or the current rank-2 `span2d`? The notes already
  specify rank 3 for extraction and strided reads for fitting. Three lines, zero call-site
  churn, and it decides the signature every later kernel copies.
- `std::constructible_from<T, double>` on `Real`, so the concept is the kernel's actual
  contract.
- Move `accumulator_t` to `core/` and key it on the value type rather than the storage type.
- A `CoreKernels(parallel, serial)` record for the seam (R1-8).
- Invert the `_backend` import cycle: policy in `_backend`, catalogue next to each kernel.
  Two `noqa: PLC0415` comments currently document the cycle and no contract prevents its
  growth.
- Separate `Backend` (which family) from an `IsaVariant` axis (which build), before
  `PANTR_BACKEND`'s accepted values are user-facing.
- Make `_backend`'s surface public like `_parallel`'s, or mark it explicitly unstable.

## What holds up

The kernel is a faithful port: read line by line against the oracle by two lenses, and
bit-exact against it across degrees 0-16 in both dtypes, inside and outside the domain. Every
configure-time gate genuinely fires — the concepts hard gate, the version filter and its
override, both math-flag refusals were each triggered by hand. The FetchContent SHAs match
their claimed tags. A real wheel and a real sdist were built and inspected: `py.typed`, the
stub and the extension present, version single-sourcing working, no nanobind runtime
dependency. The `ParityClaim`/`Roundings` type design — a rounding *count* rather than a
magnitude, with an underived tolerance made inexpressible — is the best-designed thing in the
PR and should be the model for the rest. Six mutants of the kernel were each caught by 11 to
89 of the 119 tests, and a bug injected into *both* backends simultaneously — the one case
parity cannot see — was caught by the exact rational oracle at a ratio of 5×10¹⁴.
