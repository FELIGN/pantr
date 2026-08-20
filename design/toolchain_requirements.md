# Toolchain requirements and configure-time gating

**Status:** decided. Not implemented; belongs in the infrastructure PR.
**Date:** 2026-08-19.
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
- **Consequently unanswered and now mostly moot:** what the probe reports on the *system*
  GCC 10 and Clang 10, since the environment shadows them and the build will not use them.
  What that would have told us is where the real floor sits.
- **Stated from knowledge and explicitly uncertain:** the exact C++20 feature matrix of GCC 10
  and Clang 10, and the version at which Clang's concepts support became reliable. The
  Clang 14 floor above is a starting guess and should be replaced by whatever the probe run on
  real compilers shows.
- **Not investigated:** what compiler the `manylinux_2_28` image ships, which sets the real
  floor for the Linux wheels regardless of what this note prefers.

## Open questions

1. What does the probe report on the development server's GCC 10 and Clang 10? That result
   replaces the guessed Clang 14 floor with a measured one.
2. Which compiler does `manylinux_2_28` provide? If it is older than the gate, the wheel build
   fails and either the image or the gate has to move.
3. Should the gate run in the Python build path too, or only for a direct CMake configure?
   scikit-build-core invokes CMake, so it inherits the gate, but the error surfaces inside a
   `pip install` log where it is much easier to miss.
