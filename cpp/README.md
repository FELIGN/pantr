# The pantr C++ prototype

**This is a prototype and it is meant to be abandonable.** It exists to answer a
short list of questions about whether a C++ port of pantr is worth doing, with
measurements rather than opinions. It ports two modules so far, one per pull
request: `pantr.basis`'s cardinal B-spline tabulation, and the rule generation of
`pantr.quad`. If the answers had come out badly the right response would have
been to delete the branch, and nothing here is shaped to make that hard.

The decisions it implements were taken beforehand and live in `design/*.md` at
the repository root. This directory implements them; it does not revisit them.

## What is here

```
cpp/include/pantr/core/scalar.hpp        the Real concept and value_of (Tier A/B)
cpp/include/pantr/core/mdspan.hpp        the <mdspan> / Kokkos switch
cpp/include/pantr/basis/cardinal_bspline.hpp   the first ported kernel
cpp/include/pantr/quad/legendre.hpp      Gauss-Legendre by Newton on P_n
cpp/include/pantr/quad/lambert_w.hpp     the principal branch, by Halley
cpp/include/pantr/quad/tanh_sinh.hpp     the double-exponential rule
cpp/include/pantr/quad/simple_rules.hpp  trapezoidal and modified Chebyshev nodes
cpp/bindings/pantr_cpp.cpp               the module shell and its provenance
cpp/bindings/basis.cpp, quad.cpp         one register_*(m) per ported package
cpp/tests/                               ctest: dependencies, the concept, the kernels
cpp/benchmark/                           the kernel alone, no Python
```

The Python side of the same prototype:

```
src/pantr/_backend.py                PANTR_BACKEND and PANTR_ISA_VARIANT: which
                                     implementation, and the rule it never breaks
src/pantr/basis/_basis_backend.py    the basis kernels of each backend, and the
                                     adapter that calls the extension
src/pantr/quad/_quad_backend.py      the same, for the five quadrature kernels
src/pantr/quad/_rules_core.py        the Python they are transliterated from, which
                                     stays the parity oracle
tests/parity/                        each C++ result against its Python oracle
scripts/bench_parity.py              both kernels and both entry points, timed
scripts/bench_quad.py                the same, for the quadrature rules
scripts/ci_local.sh                  the whole check, and the gates asserted to fire
```

## Building it

Two entry points, both supported and both exercised by `scripts/ci_local.sh`.

```bash
# Standalone: C++ tests and the benchmark, no Python involved.
cmake --preset gcc && cmake --build --preset gcc && ctest --preset gcc
cmake --preset clang && cmake --build --preset clang && ctest --preset clang

# Through Python: the extension only.
python -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -e . --no-build-isolation
```

Each compiler gets its own build directory on purpose. Configure-time probe
results are cached in `CMakeCache.txt`, so reusing one directory across
compilers answers for whichever configured it first.

## What it measured

All figures from the development server: conda-forge GCC 14.4.0 and Clang
18.1.8, CMake 4.4.2, numba 0.65.1, 20 CPUs of a shared 160-CPU machine.

**Both compilers build it** with `-Werror` and all of
`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wdouble-promotion`, and the
three ctest binaries pass under both.

**`<mdspan>` is absent**, so the Kokkos reference implementation is the normal
path rather than a fallback. The reason is an implementation gap in GCC 14's
libstdc++, not the C++20 baseline: `__cpp_lib_mdspan` is undefined even under
`-std=gnu++23`. Clang 18 here uses that same libstdc++ and inherits it.

**Kokkos mdspan and Eigen both survive** `SYSTEM` plus `-Werror`, and both
declare a `cmake_minimum_required` new enough for CMake 4. Eigen 5.0.1 emits one
`CMP0146` deprecation warning at configure time, which is a policy warning
rather than a compile warning and so does not reach `-Werror`.

**GCC 10 and Clang 10 pass the concepts probe** and compile and correctly run
the ported kernel. This matters because it means the probe alone does not keep
them out; the version filter in `cmake/PantrCompilerProbes.cmake` does. See the
open question at the end.

**No FMA is emitted.** The build sets `-ffp-contract=on` but no `-march`, so the
target is baseline x86-64, which has no FMA instruction to fuse into. Verified
by disassembly: no `vfmadd` in any object either preset produces, and the one
`a*b+c` site compiles to `mulsd` + `addsd` on both compilers. Adding `-mfma`
fuses that site and only that site. numba emits it unfused in both cases. So the
two backends currently execute an identical sequence of IEEE-754 operations, and
`tests/parity/test_basis_cardinal_bspline.py` asserts exactness rather than a
tolerance, switching to the derived FMA bound when the extension reports
`__fp_contract__ == "available"`.

**Speed, kernel against kernel, single-threaded** (minimum of 5 runs, ms):

| degree | points | numba, 1 thread | C++ | ratio |
|---:|---:|---:|---:|---:|
| 1 | 10^6 | 4.04 | 2.78 | 1.45x |
| 2 | 10^6 | 7.94 | 5.93 | 1.34x |
| 3 | 10^6 | 13.29 | 10.22 | 1.30x |
| 5 | 10^6 | 23.22 | 23.62 | 0.98x |
| 8 | 10^6 | 49.62 | 52.49 | 0.95x |

So the port is somewhat faster at low degree and level at high degree. That is
the honest headline, and it is a modest one: both compilers are generating
reasonable code for the same recurrence, and there is no free factor of ten
waiting in a straight transliteration.

At the *entry point* the picture differs at small sizes, and not because of the
kernel: the numba cardinal B-spline kernel is `parallel=True` unconditionally
and has no serial twin, so it pays a fork/join even for 100 points. The C++
backend is 4x to 7x faster there and 0.2x as fast at 10^6 points, where numba's
20 threads do what 20 threads do.

**One defect this measurement found.** `nanobind_add_module()` appends `-Os`
after the build type's `-O3`, and the last optimisation flag wins, so the kernel
was being compiled for size inside the extension while the standalone benchmark
compiled it for speed. Same source, 31.4 ms against 10.3 ms at degree 3. Fixed
with `NOMINSIZE` in `cpp/bindings/CMakeLists.txt`, where the numbers are
recorded. Without the fix the prototype reported the port as 3x slower than
numba instead of 1.3x faster, which is the opposite conclusion.

## What it deliberately does not do

- **One module per PR.** `pantr.basis` was the first, `pantr.quad` the second.
  The rule is about the size of a reviewable change, not about a ceiling on the
  port: what makes a second module cheap is that the infrastructure question was
  settled by the first, and what makes it worth reviewing separately is that each
  one brings its own numerical content. `quad` needed two corrections to the
  parity harness that one consumer could not have revealed.
- **No `-march`, no SIMD, no ISA variants.** `design/simd.md` makes that stage 2
  and gates it on first measuring the baseline gap against `-march=x86-64-v3`.
  **That measurement is now taken**, at `-O3` on the standalone kernel, 10^6
  points, minimum of repeats:

  | target | degree 3 | degree 8 | `vfmadd` sites |
  |---|---:|---:|---:|
  | baseline x86-64 | 10.47 ms | 50.73 ms | 0 |
  | `-march=x86-64-v3` | 8.69 ms | 40.04 ms | 1 |
  | `-march=native` (AVX-512) | 9.20 ms | 43.69 ms | 1 |

  So the gap is real but modest, 1.20x to 1.27x, and it is a decision for
  whoever opens stage 2 rather than a free win. Two things in that table are
  worth more than the ratios. The ISA introduces **exactly one** fused site,
  which is the one the parity bound is derived around. And **`-march=native` is
  slower than `-march=x86-64-v3`** on this machine, whose AVX-512 the wider
  target enables: `design/simd.md` lists "AVX-512 can be slower" as a hazard
  stated from knowledge and not measured, and this measures it on one of pantr's
  own kernels.
- **No threading on the C++ side.** A single-threaded kernel is what makes the
  comparison above readable, and the threading model is constrained by decisions
  in `design/user_functions_across_the_boundary.md` that this PR does not spend.
- **No MPI, no CUDA, no AD type.** `scalar.hpp` keeps the door open for a
  forward-mode `Dual`; nothing instantiates one except `cpp/tests/test_scalar_generic.cpp`.
- **No wheel strategy.** The stable ABI is deliberately off; split mode is
  probed by `scripts/ci_local.sh` and not adopted.

## The version floor, and how it was set

Measured here rather than guessed, against this tree rather than a snippet:

| compiler | probe | build under `-Werror` | ctest |
|---|---|---|---|
| g++ 9.5 | fails (no `-std=c++20`) | -- | -- |
| g++ 10 | passes | yes | 3/3 |
| clang++ 10 | passes | yes | 3/3 |
| g++ 14.4, clang++ 18.1.8 | passes | yes | 3/3 |

So the floor is **10 for GCC and Clang alike**, and it means *the lowest version anyone has
actually exercised*. It replaces a floor of 14 for Clang, inherited from
`design/toolchain_requirements.md` as an explicit guess, together with no floor at all for
GCC -- an asymmetry that let a GCC 10 configure silently while a Clang 10 of the same year
was refused outright, and that nobody had measured in either direction.

AppleClang stays exempt: its version numbers do not map to LLVM versions, so any threshold
applied to it is a row that lies. `-DPANTR_ALLOW_UNTESTED_COMPILER=ON` still opens the gate
for anyone who knows better, and the floor should rise only from an observed failure.

`scripts/ci_local.sh` builds and tests with both floor compilers on every run, so the table
above is re-established rather than remembered. **The GitHub workflow does not**: it runs
GCC 14 and Clang 18, and `ubuntu-24.04` does not package GCC 10. So the floor is guaranteed
by one machine, and that is a deliberate trade rather than an oversight -- covering it in CI
needs an older runner image or a container, which is more than this prototype should carry.
