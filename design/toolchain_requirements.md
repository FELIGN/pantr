# Toolchain requirements and configure-time gating

**Status:** decided, and implemented in `cmake/PantrCompilerProbes.cmake`.
**Date:** 2026-08-19, amended 2026-09-14 by FELIGN/pantr#376 -- see *The floor is the
standard library, not the compiler* below, which supersedes the floor this note originally
recorded.
**Scope:** what a compiler must provide to build pantr, how that is checked, and the flag
decisions that live in the same CMake file.
**Companions:** `design/simd.md` (which flags earn their place) and `design/isa_dispatch.md`
(how variants are shipped). This note is about whether the build may proceed at all.

**Validated against:** pantr **0.7.0** (`main`, tag `v0.7.0`), 2026-08-19. Line numbers
below refer to that tree.

## The decision

**Probe features at configure time. Do not gate on compiler version numbers.**

Version numbers are a proxy for features and they fail in both directions: a vendor may
backport a feature to a lower version, and a version may nominally have a feature that is
broken. AppleClang does not map to LLVM versions at all, so any version table has a row
that lies.

The trigger for settling this: the development server runs GCC 10 and Clang 10, both from
2020, and `D4` fixes a C++20 baseline whose concepts requirement (`D5`, `D6`) they may not
meet.

## `cxx_std_20` alone does not check what it appears to

`target_compile_features(tgt PRIVATE cxx_std_20)` verifies only that the compiler **accepts
the flag**. GCC 10 accepts `-std=c++20` and passes that check while lacking parts of the
standard library. It is therefore not a gate, and a build system that relies on it will fail
later, in a template error, instead of at configure time with a useful message.

## Hard gates versus feature toggles

Not every requirement is a rejection. The question for each is whether a cheap fallback
exists.

| requirement | fallback? | treatment |
|---|---|---|
| C++20 mode | none | **hard gate** |
| working concepts | none: the scalar-generic design rests entirely on them | **hard gate** |
| `std::span` | trivial to write, not worth it | hard gate in practice |
| `<mdspan>` | yes, the Kokkos reference implementation | **detect and adapt** |
| `<expected>` | yes, a small `Result` type or `tl::expected` | detect and adapt |

The mechanism the mdspan decision already needed (`#if __cpp_lib_mdspan`, else Kokkos)
generalizes: **adapt where adapting is cheap, reject only where it is not.**

## Probe the constructs pantr actually uses

A generic C++20 test proves little. The concepts probe should be the shape of the scalar
concept the library defines, not a toy `requires` clause.

```cmake
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++20")

check_cxx_source_compiles("
  #include <concepts>
  template <class T> concept Scalar = requires(T a, T b) {
      { a + b } -> std::convertible_to<T>;
  };
  template <Scalar T> T twice(T x) { return x + x; }
  int main() { return twice(1.0) == 2.0 ? 0 : 1; }
" PANTR_HAS_CONCEPTS)

if(NOT PANTR_HAS_CONCEPTS)
  message(FATAL_ERROR
    "pantr requires working C++20 concepts.\n"
    "  Detected: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n"
    "  Fix (no root needed): conda install -c conda-forge gxx=14")
endif()

check_cxx_source_compiles("#include <mdspan>\nint main(){}" PANTR_HAS_STD_MDSPAN)
```

Each probe should be kept in sync with what the code relies on. A probe that tests something
the library no longer uses is a gate that rejects for no reason; a construct used without a
probe is a template error waiting for a user.

## The version floor survives, in a different role

A probe reports whether something **compiles**, not whether it is **correct**. Clang 10's
concepts support was partial and buggy, so a small probe could plausibly compile on an
implementation that then miscompiles the real library.

So a version check stays, but it is no longer the criterion. It is a **filter for
implementations known to be broken**, kept short, each entry carrying its reason, and with an
escape for someone who knows better:

```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14
   AND NOT PANTR_ALLOW_UNTESTED_COMPILER)
  message(FATAL_ERROR "Clang < 14 has known-incomplete concepts support. "
                      "Override with -DPANTR_ALLOW_UNTESTED_COMPILER=ON at your own risk.")
endif()
```

The list should only grow from observed failures, never from speculation, or it becomes the
version table this note exists to avoid.

## The floor is the standard library, not the compiler

**Amendment, 2026-09-14, from FELIGN/pantr#376.** Implemented.

The floor set on 2026-08-19 was a pair of compiler versions, GCC 10 and Clang 10, and it was
true the day it was measured. It stopped being true when `cpp/include/pantr/core/format.hpp`
began calling the **floating-point** overloads of `std::to_chars`. libstdc++ 10 implements
`std::to_chars` for integers only and defines no `__cpp_lib_to_chars`; the floating-point
overloads arrived in libstdc++ 11.

What that exposed is that a compiler version was never the quantity being bounded:

| toolchain | front end | standard library | builds pantr |
|---|---|---|---|
| `g++-10` | GCC 10 | libstdc++ 10 | **no** |
| `clang++-10` | Clang 10 | libstdc++ 12, the system's | **yes** |

Same year, same language level, opposite answers, and the front-end version predicts neither.
A compiler does not carry a standard library, it is *paired* with one, and on Linux a Clang
picks up whichever libstdc++ is installed. So the binding bound is the library:

> **libstdc++ 11 or newer** -- more precisely, any standard library that provides
> `__cpp_lib_to_chars` and the floating-point `std::to_chars` overloads.

That is a raise, and it is the raise this ticket's own Non-goal forbade before its owner's
ruling amended it. The three constraints cannot all hold: `AC1` demands that a library without
the facility be refused, `AC5` demands that the local gate pass in full, and refusing `g++-10`
*is* moving the declared floor. Two ways out were considered and rejected. Warning instead of
failing makes the breakage legible without making libstdc++ 10 compile, so the floor build
stays red and nothing is fixed. Reimplementing `format_repr` without `std::to_chars` was
already rejected by the tree: `format.hpp`'s file comment records that a second copy of the
repr rule is exactly how the first one went wrong, and the file is shaped so that a probe can
name one function -- `format_general` and `format_fixed` beside it go through `snprintf`,
which the old library does have.

**It is enforced as a probe, not as a version number**, per the decision at the top of this
note. `cmake/PantrCompilerProbes.cmake` gains a second hard gate beside concepts. It compiles
the two `std::to_chars` calls `format_repr` actually makes and fails with a message naming the
facility, so the refusal arrives at configure time rather than as an overload-resolution error
inside a header part way through a build -- which is how #376 was found.

**And a second time in the header, for the consumer.** A configure-time probe measures the
toolchain that *built* the package. pantr is header-only and installable, `pantrConfig.cmake.in`
runs no probes, and someone who installs the package and compiles against it with libstdc++ 10
never meets the CMake gate at all -- they meet the overload-resolution error, which is the exact
outcome the ticket's item 1 says the gate exists to prevent. So
`cpp/include/pantr/core/format.hpp` carries an `#error` on `__cpp_lib_to_chars`, on the pattern
and for the reason `cpp/include/pantr/core/mdspan.hpp` already states: *the header decides, not
the build*. It differs from the mdspan case in having no second branch to select -- there is
nothing to adapt to, so it refuses.

This is one step beyond the ticket's literal Specification, which names a configure-time probe
and stops there. It is recorded here rather than folded in quietly.

No version comparison was added, and the existing one did not move. The GNU row cannot become
11 without asserting a *library* fact through a *front-end* number, and `clang++-10` is the
standing counterexample sitting on the machine that would have made the assertion. The version
check keeps its unchanged meaning: nothing below 10 in either family has ever been tried.

The gate is deliberately outside `PANTR_ALLOW_UNTESTED_COMPILER`. That flag says "I know this
version is untested"; an absent facility is a different claim, and opening the gate for it
would return exactly the template error the gate exists to replace.

## Keeping it true: an absent floor compiler is a failure, not a skip

The second half of #376 was not the floor but the report. `scripts/ci_local.sh` was the only
thing checking the floor anywhere, and it `record SKIP`ped when a floor compiler was absent --
so on any machine without one the guarantee was vacuous and read as passing. A check that
reports absence and success identically is worse than no check, because it is believed.

Of the three ways to close that, the one taken is **the floor stays local-only, and its
section becomes mandatory**: an absent floor compiler is now a `FAIL`. The two alternatives,
and why not:

- *Run the floor in GitHub Actions.* This reverses a decision recorded in
  `.github/workflows/cpp.yaml` whose reason still holds -- `ubuntu-24.04` packages neither
  Clang 10 nor GCC 10, so it needs an older image or a container. That reason was about cost,
  and #376 did not change the cost. It only changed how much the local check is worth.
- *Keep it local-only and write down that the claim is best-effort.* Honest, and strictly
  weaker: it records the vacuity instead of removing it, and nothing then obliges the one
  machine to run the check at all.

Promoting it costs nothing on the machine that makes the claim, where both compilers are
present, and it turns "guaranteed by one machine" from a conditional into something that
machine's own gate enforces. `cpp.yaml`'s trade is not reversed; its premise is made true, and
its header now says so.

The floor is therefore established by these rows, on every run of `scripts/ci_local.sh`:

| row | what it asserts |
|---|---|
| `clang++-10 is accepted` | the gates do not stand in the way of a toolchain that works |
| `the to_chars gate refuses g++-10` | the gate fires, and the refusal carries its own message |
| `PANTR_ALLOW_UNTESTED_COMPILER does not open the to_chars gate` | the two are not wired together |
| `format.hpp refuses libstdc++ 10 directly` | the consumer's half fires too, compiling one TU outside CMake |
| `floor: clang++-10` configure / build (`-Werror`) / ctest | the tree is built whole and tested at the floor |

Every one of them is a `FAIL` rather than a `SKIP` when its compiler is missing, and the
detail line says which compiler and what it would have proved.

## The message is the deliverable

A `FATAL_ERROR` naming the detected compiler, its version, the missing capability and a
command that fixes it is worth far more than one reading "C++20 required". The person who
hits it is about to either give up or write to the maintainer, and that line decides which.

## Flags that belong in the same file

Two decisions from elsewhere are enforced here, because CMake is the only place they can be
enforced once:

- **Never `-ffast-math`, and never `-ffinite-math-only`.** Reassociation of sums invalidates
  the error bounds every derived tolerance in the library assumes. This is not a performance
  option, it is a silent correctness change. `-funsafe-math-optimizations` alone may be
  offered behind an explicit option.
- **The floating-point contraction flag goes on the interface target**, not on individual
  targets. Any flag that participates in a numerical claim must reach every variant, or the
  ISA variants of `design/isa_dispatch.md` will disagree numerically with each other.

## CMake 4 is strict about what dependencies declare

Verified on the build server (2026-08-19): it runs **CMake 4.4.2**.

CMake 4 **removed compatibility with `cmake_minimum_required(VERSION < 3.5)`**. A project
declaring an older minimum fails hard at configure time rather than warning. This matters
here specifically because of `FetchContent`: the build pulls third-party `CMakeLists.txt`
files that nobody in this project controls, and one of them declaring an ancient minimum
takes the build down with an error that reads as if it came from pantr's own CMake.

Eigen and the Kokkos mdspan reference implementation declare modern minimums and should be
fine. Any dependency added later is a coin flip until checked.

The escape hatch exists for exactly this and should be documented next to the dependency
declarations rather than discovered: `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.

On the useful side of the same version: CMake 4.4 has `FetchContent_Declare(... SYSTEM)`
(needs 3.25 or newer), which is what keeps a dependency's own warnings from tripping
`-Werror` in pantr's build.

## Two nanobind flags that are not optional

Both were found by colliding with them, and both are recorded in full in
`design/build_findings.md`:

- **`NOMINSIZE`** on `nanobind_add_module()`, plus **`install.strip = false`** in
  `pyproject.toml`. Without them nanobind's own `-Os` lands after the build type's `-O3`
  and wins, compiling the numerical kernels for size. Measured at **31.4 ms against
  10.3 ms** on the same source. This is a silent 3x, aimed squarely at the number the port
  is judged by.
- **`NB_SUPPRESS_WARNINGS`**. In split mode nanobind adds its includes to the extension
  target and marks them `SYSTEM` only under that option; without it a `-Wshadow` inside
  `nb_attr.h` meets `-Werror` and the build dies in a third-party header.

## The offline escape

`FetchContent` needs network access at configure time. The development server has it; a
cluster compute node typically does not. Expose `FETCHCONTENT_SOURCE_DIR_<NAME>` for each
dependency and document `FETCHCONTENT_FULLY_DISCONNECTED` with a pre-populated cache. Five
lines of CMake, and the day it is needed is the day nobody wants to be writing them.

## Two caveats worth knowing before they cost an afternoon

- **Probe results are cached** in `CMakeCache.txt`. Switching compilers inside an existing
  build directory yields stale answers. The fix is to delete the build directory, but the
  symptom is confusing enough to be worth documenting next to the presets.
- **Each probe is a compiler invocation**, so a dozen of them cost a few seconds of configure
  time, once. Negligible against the build, but it is why the probe set should stay small and
  targeted rather than exhaustive.

## Epistemic status

- **Verified:** that GCC 15.2 lacks `<mdspan>` while the laptop's Clang has it, which is what
  makes mdspan a toggle rather than a gate; that the sibling project gates only on
  `cxx_std_20`, so copying its CMake would not have caught GCC 10.
- **Measured on the build server (2026-08-19):** the conda environment shadows the system
  GCC 10 and Clang 10 with conda-forge GCC 14 and Clang 18.1.8, and **all three of `g++`,
  `clang++` and `x86_64-conda-linux-gnu-g++` pass a C++20 concepts probe**. `<mdspan>` is
  **absent** there, confirming on the actual build machine that the Kokkos fallback is
  required and not merely a precaution. CMake 4.4.2, Ninja 1.13.2, ccache 4.13.6.
- **Measured 2026-09-14, on the system compilers the 2026-08-19 pass could not reach** (they
  are shadowed by the conda environment, so each was named explicitly): `g++-10` passes the
  concepts gate and **fails** the floating-point `std::to_chars` gate; `clang++-10` passes
  both, against the system libstdc++ 12. Re-established by `scripts/ci_local.sh` on every run
  rather than held as a dated result.
- **Stated from knowledge and explicitly uncertain:** the exact C++20 feature matrix of GCC 10
  and Clang 10, and the version at which Clang's concepts support became reliable. The
  Clang 14 floor above is a starting guess and should be replaced by whatever the probe run on
  real compilers shows.
- **Not investigated:** what compiler the `manylinux_2_28` image ships, which sets the real
  floor for the Linux wheels regardless of what this note prefers.

## Open questions

1. ~~What does the probe report on the development server's GCC 10 and Clang 10?~~
   **Answered**, in two passes. 2026-08-19: both pass the concepts probe, which replaced the
   guessed Clang 14 floor with a measured 10 for both families. 2026-09-14 (#376): `g++-10`
   fails the `std::to_chars` gate and `clang++-10` passes it, which is what moved the binding
   bound off the compiler and onto the standard library.
2. Which compiler does `manylinux_2_28` provide? If it is older than the gate, the wheel build
   fails and either the image or the gate has to move.
3. Should the gate run in the Python build path too, or only for a direct CMake configure?
   scikit-build-core invokes CMake, so it inherits the gate, but the error surfaces inside a
   `pip install` log where it is much easier to miss.
