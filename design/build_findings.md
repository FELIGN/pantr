# Findings from the first build, and what they cost to rediscover

**Status:** measured on the build server, 2026-08-19/20, during the infrastructure PR.
**Validated against:** pantr **0.7.0** plus the port skeleton on `proto/cpp`.
**Scope:** things that were discovered by colliding with them. Each one silently degrades
the build or the numbers rather than failing loudly, which is why they are written down.

## `-Os` silently defeats the whole port

`nanobind_add_module()` appends **`-Os` after** the build type's `-O3`, and the last flag
wins. The numerical kernel therefore compiles optimized for size inside the extension.

**Measured: 31.4 ms against 10.3 ms, same source.** A factor of three, invisible, and it
lands precisely on the number the port is judged by.

The fix is two settings, and both are needed:

- `NOMINSIZE` on `nanobind_add_module()`.
- `install.strip = false` in `pyproject.toml`, because scikit-build-core strips again at
  install time.

This one deserves its own regression check: a build that quietly loses 3x is worse than a
build that fails.

## `NB_SUPPRESS_WARNINGS` is mandatory in split mode

In split mode nanobind adds its own includes directly to the extension target, and marks
them `SYSTEM` **only** when that option is given. Without it, `nb_attr.h:78`
(`using arg = arg_t<>`) trips `-Wshadow`, and with `-Werror` the build dies in a
third-party header.

## Split mode works, and it composes with multi-ISA

Verified with nanobind `3.0.0.dev2` and `nanobind-backend 1.0.0.dev2`: the extension
builds, imports and runs, and the frontend links as `.abi3.so`, so it does target the
stable ABI as advertised.

The composition question that `design/simd.md` left open is **answered**: two frontends
compiled from one source at different `-march` (baseline and `x86-64-v3`) against a single
backend both import in the same process and both work.

Two consequences:

- **`nanobind-backend` is a runtime dependency of the wheel.** Without it the import
  fails, though with a clear and actionable message.
- **Split mode and `NB_STATIC` are indistinguishable in behaviour.** Identical acceptance
  matrix over 17 cases, performance within run-to-run spread. The difference is packaging
  and nothing else, so the choice can be made on distribution grounds alone.

## Parity can be exact, and the reason is checkable

`design/simd.md` warns that FMA breaks bit-level parity and that the tolerance must absorb
it. At the **baseline** target that warning does not apply, and the reason is worth
keeping:

base `x86-64` has no FMA instruction, so `-ffp-contract=on` cannot fuse anything. Verified
by disassembly on g++ 14.4.0 and clang++ 18.1.8 (`grep -c vfmadd` = 0 across all objects;
the site compiles to `mulsd` + `addsd`), and Numba emits the same site unfused in its IR
(`fmul` then `fadd`, never `fmuladd`). Both backends execute the same IEEE-754 sequence in
the same order.

Measured: degrees 0 to 16, `float32` and `float64`, 306 306 elements over `[-0.25, 1.25]`
plus both signed zeros, **zero bit discrepancies**.

The lesson generalizes: **exact parity is achievable wherever both sides evaluate the same
expression in the same order and no fusion is available**, and it is checkable by
disassembly rather than assumed. Where either condition fails, a bound is required. A
closed form ported verbatim satisfies both; two different algorithms computing the same
mathematical object satisfy neither.

The extension reports its own `__fp_contract__`, and the parity harness selects the exact
or the bounded branch from it rather than from a build-time guess.

## Two corrections a rounding bound needed, both found by attacking it

With `-march=x86-64-v3` exactly one fused site appears, and the bounded branch becomes
live. A naive `degree · eps/2` bound was **refuted by measurement** at degrees 3, 4 and 5,
with observed/bound ratios up to 1.29. What replaced it is a per-stage rounding budget, and
it needed two terms that only surfaced under attack:

- **An underflow floor.** In the subnormal range, rounding commits *absolute* error, not
  relative. Without that term the bound was violated by **six orders of magnitude** at
  degree 12 in `float32`.
- **Amplification propagated as a hull, not as a computed coefficient.** When the computed
  `1 - term` rounds to exact zero, a bound built on that computed value collapses to zero
  and bounds nothing.

Both are reusable, and both are likely to be needed again the first time a bound is written
for a module whose two implementations use different algorithms. The general lesson is the
one the project already states and this confirms: a bound is a claim, and the cheapest way
to find out it is false is to try to break it with measurement before trusting it.

## The editable worktree flow is cheap, given two decisions

Measured: **7.5 s cold** without build isolation, **9.3 s** with it, **2.5 s incremental**.

What makes it cheap:

- a **persistent `build-dir`** in `pyproject.toml`;
- **Eigen fetched only when the C++ tests are requested**, not on the pip path.

Three things to know before setting the same flow up elsewhere:

- The venv must be created with `--system-site-packages`.
- The editable install must go into a **per-worktree venv**. `CLAUDE.md` forbids installing
  editable from a worktree against the shared conda environment, because it leaves the
  install pointing at a worktree that is later deleted.
- `sphinx` is not in the conda environment, so a docs build needs it installed in that venv.

## The oracle is not always Numba, and that changes what the numbers mean

Measured across the stage-1 modules:

| module | Numba kernels |
|---|---|
| `geometry`, `transform`, `quad`, `change_basis`, `tolerance` | **0** |
| `basis` | 18 |
| `grid` | 15 |
| `bezier` | 41 |

So the dual-backend design's "Numba implementation as the parity oracle" is **only half
true**. Five of the stage-1 modules are pure NumPy, which is why `Backend.NUMBA` was
correctly renamed to `Backend.PYTHON`.

Two consequences, and both are easy to get wrong in opposite directions.

**Speed figures do not transfer.** The `quad` PR measured 60.6x, 38.9x and similar against
*interpreted NumPy*. `basis`, `grid` and `bezier` have real Numba kernels, which compile to
machine code exactly as the C++ does. Those ratios will look nothing like these, and
quoting the `quad` numbers as the port's baseline would be misleading. The first honest
C++-against-compiled-code comparison is still ahead.

**Parity evidence differs.** Against a NumPy oracle, exactness is established by
disassembling the extension alone: NumPy's own operation order is what it is. Against a
Numba oracle it takes both sides — the disassembly *and* the Numba IR, to show neither
fuses. `quad` could skip the IR step because there was no Numba to inspect; `basis`, `grid`
and `bezier` cannot.

## Exactness is a property of the build, not of the kernel

The `quad` PR established this by trying to break it, and the result is worth stating
plainly because it changes how an exactness claim must be made.

Three builds of one source:

| build | FMA sites | values differing |
|---|---|---|
| `-O3 -ffp-contract=on` (shipped) | 0 | 0 / 2143 |
| `-O3 -march=native` | 6 | **1994 / 2143** |
| `-O3 -march=native -ffp-contract=off` | 0 | 0 / 2143 |

With FMA available, **93% of the values move**. Turning off contraction alone restores
exactness.

So a claim of bit-exact parity is never a property of the algorithm. It must be selected at
**runtime**, from what the extension reports about its own build, which is what the
`__fp_contract__` mechanism does. A claim hard-coded at design time would be false on the
first machine that enables the ISA ladder.

## Two ways a rounding bound silently degenerates

Both were found in the parity harness itself, and both had passed the first consumer.

**A bound that dwarfs the values it compares was accepted.** `1.0` against `-1e250` passed.

**And the instructive one:** `(1 + u)^stages - 1` evaluates to **exactly zero** in float64
for any budget of one rounding per stage, because `1 + eps/2` lands on the midpoint and
round-to-even returns it to 1. A claim reporting `BOUNDED` was therefore asserting bit
equality.

Why it survived design: the first consumer's budget was two roundings per stage, so
`per_stage = eps`, which *is* representable added to 1. One rounding per stage — the
ordinary shape of a kernel that does not narrow — is what `quad` asked for first. And the
guard written for exactly this class tested `per_stage == 0`, which here is `1.11e-16` and
passes. **A budget can be non-zero and still produce a zero bound.**

The general lesson, which the project already asserts and this confirms twice: a bound is a
claim, and a test that passes cannot distinguish a correct bound from a degenerate one. The
cheapest way to find out is to try to break it by measurement before trusting it.

## A factor of two, propagated through four places

The notes cited a measured displacement in units of `eps` while calling it `u`. Since
`u = eps/2`, every margin built on top was inflated twofold: the provable constant is 3.5x
the observation, not 7x.

Worth recording as a class of error rather than an incident: `eps` and `u` differ by two,
both are called "machine epsilon" in prose, and a bound stated in the wrong one is
conservative in one direction and wrong in the other. Any derivation should name which it
uses at the point of use.

## The check suite had a hole where the check already existed

The keyword-only argument change broke `scripts/bench_quad.py`, which called
positionally. **mypy flags all four sites statically**, because the stub carries the
PEP 3102 marker. But `Makefile:73` runs `mypy --config-file mypy.ini src tests`, and
`scripts/` is outside that.

So the tooling to catch it was present, correct, and pointed somewhere else. This is worth
fixing at the Makefile rather than by remembering: `scripts/` is exactly the kind of
directory whose contents are run rarely and break silently.

## Two more corrections the compiler made

- `std::pow(c, 2.0)` never reaches the binary: gcc folds it from `-O1` and `nm -D` finds no
  `pow`. Fifteen lines were written justifying a choice the compiler undoes.
- A C++ test's `n <= 75` range was fitted to this machine's `cos`. One ulp of difference
  moves it to 57, inside the asserted range. A threshold measured on one libm is not a
  threshold.

## What the first kernel did *not* establish

Speed. Kernel against kernel, single-threaded, 10^6 points: **1.42x at degree 1, 1.30x at
degree 3, 0.98x at degree 8**. At degree 8 the C++ is marginally *slower*.

This is one closed-form kernel with no loop worth vectorizing, so it says little about the
port as a whole, but it does mean **"the port makes pantr faster" is not established** and
should not be claimed. The case for the port rests on the standalone C++ library, the
scalar-generic design and `float32`, none of which this kernel exercises. The first honest
speed evidence will come from the banded-contraction kernel, which is stage 2.

Note also that this number was wrong twice before it was right, and the second cause was
the `-Os` above. Any future speed claim should state that `NOMINSIZE` was in effect.
