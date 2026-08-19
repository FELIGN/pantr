# Configure-time capability probes for the pantr C++ core.
#
# The policy this file implements is design/toolchain_requirements.md: probe the
# features the library actually uses, never gate on a compiler version number.
# Version numbers are a proxy for features and they fail in both directions -- a
# vendor may backport a feature to a lower version, and a version may nominally
# carry a feature that is broken. AppleClang does not map to LLVM versions at
# all, so any version table has a row that lies.
#
# Two kinds of requirement, distinguished by whether a cheap fallback exists:
#
#   hard gate      concepts. The scalar-generic design rests entirely on them and
#                  there is no fallback, so a compiler without them is rejected.
#   feature toggle <mdspan>. The Kokkos reference implementation is a drop-in, so
#                  its absence selects a fallback rather than ending the build.
#
# Caveat worth knowing before it costs an afternoon: probe results are cached in
# CMakeCache.txt, so switching compilers inside an existing build directory
# yields stale answers. Delete the build directory; the presets give each
# compiler its own so this does not arise in the normal flow.

include_guard(GLOBAL)
include(CheckCXXSourceCompiles)

# The probes must run under the same standard the library is built with, or they
# answer a question nobody asked.
set(CMAKE_REQUIRED_FLAGS "-std=c++20")

# One line, reused by every failure below, so the person who hits any of them is
# told what they are running and what fixes it. That line decides whether they
# give up or write to the maintainer.
set(PANTR_TOOLCHAIN_REPORT
    "  Detected: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n"
    "  Compiler: ${CMAKE_CXX_COMPILER}\n"
    "  Fix (no root needed): conda install -c conda-forge gxx=14\n"
    "  Then configure a FRESH build directory -- probe results are cached.")

# --------------------------------------------------------------------------
# Hard gate: C++20 concepts, probed in the shape pantr::Real actually has.
# --------------------------------------------------------------------------
#
# target_compile_features(tgt PRIVATE cxx_std_20) verifies only that the
# compiler ACCEPTS the flag. GCC 10 accepts -std=c++20 and passes that check
# while lacking parts of the standard library, so it is not a gate: it defers
# the failure to a template error deep in a build.
#
# A generic C++20 test proves little either. This probe is the shape of the
# scalar concept in cpp/include/pantr/core/scalar.hpp -- the arithmetic closure,
# the two-step `using pantr::value_of; value_of(x)` that Tier B operations go
# through, and a constrained template instantiated on double. Keep the two in
# sync: a probe testing something the library no longer uses is a gate that
# rejects for no reason, and a construct used without a probe is a template
# error waiting for a user.
check_cxx_source_compiles("
#include <concepts>
#include <type_traits>

namespace pantr {

template <std::floating_point T>
constexpr T value_of(T x) noexcept { return x; }

namespace detail {
using pantr::value_of;
template <class T>
concept has_value_of = requires(const T& a) {
    { value_of(a) } -> std::floating_point;
};
}  // namespace detail

template <class T>
concept Real = std::copyable<T> && detail::has_value_of<T> &&
    requires(T a, T b) {
        { -a } -> std::convertible_to<T>;
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { a / b } -> std::convertible_to<T>;
    };

template <Real T>
constexpr T twice(T x) { return x + x; }

}  // namespace pantr

int main() {
    static_assert(pantr::Real<double>);
    static_assert(pantr::Real<float>);
    static_assert(!pantr::Real<int>);
    return pantr::twice(1.0) == 2.0 ? 0 : 1;
}
" PANTR_HAS_CONCEPTS)

if(NOT PANTR_HAS_CONCEPTS)
  message(FATAL_ERROR
      "pantr requires working C++20 concepts, and this compiler does not provide "
      "them. The scalar-generic core rests entirely on concepts; there is no "
      "fallback path.\n"
      ${PANTR_TOOLCHAIN_REPORT})
endif()

# --------------------------------------------------------------------------
# Feature toggle: <mdspan>.
# --------------------------------------------------------------------------
#
# Absent from GCC 14, which is this project's own build compiler, so the Kokkos
# fallback is the normal path here rather than a precaution.
#
# The probe asks TWO questions, because cpp/include/pantr/core/mdspan.hpp needs
# both answers and they come from different parts of the standard:
#
#   __cpp_lib_mdspan   does the standard LIBRARY implement P0009? That macro is
#                      this question's own feature-test macro, so it answers it
#                      directly. The probe used not to consult it and inferred the
#                      answer from whether the body compiled, which conflates a
#                      missing header with a body that does not parse -- and the
#                      body did not parse, for the reason immediately below.
#   view[1, 2]         does the LANGUAGE have the multidimensional subscript?
#                      `pantr::at` uses exactly this spelling in its
#                      PANTR_HAS_STD_MDSPAN branch, so a standard library that
#                      shipped <mdspan> in a mode where this does not parse would
#                      be of no use: the toggle would turn ON and the build would
#                      then fail. Under C++20 `view[1, 2]` is the comma operator,
#                      which indexes a rank-2 mdspan with a single index and does
#                      not compile.
#
# Both are C++23 and are gated together, so under the C++20 baseline this probe is
# a foregone OFF. That is the correct answer -- but it is now reached for a stated
# reason rather than by accident, it says so in the CMake log, and it will start
# answering ON by itself on the day the baseline moves.
#
# CMAKE_REQUIRED_FLAGS at the top of this file pins the probe to -std=c++20, and
# that must be kept in step with the cxx_std_20 in cmake/PantrCompileOptions.cmake.
# The pin is the safe direction of the two: a probe run at a LOWER standard than
# the build can only under-report a feature, which selects the fallback, and the
# fallback compiles in every mode. A probe run at a HIGHER standard could turn the
# toggle on for a build that cannot compile it.
check_cxx_source_compiles("
#include <version>
#if !defined(__cpp_lib_mdspan)
#  error \"the standard library does not implement P0009 in this language mode\"
#endif
#include <mdspan>
#include <cstddef>
int main() {
    double storage[6]{};
    std::mdspan<double, std::dextents<std::size_t, 2>> view(storage, 2, 3);
    view[1, 2] = 1.0;
    return view[1, 2] == 1.0 ? 0 : 1;
}
" PANTR_HAS_STD_MDSPAN)

if(PANTR_HAS_STD_MDSPAN)
  message(STATUS "pantr: <mdspan> is available; the Kokkos fallback is not used")
else()
  message(STATUS "pantr: <mdspan> is absent; using the Kokkos reference implementation")
endif()

# --------------------------------------------------------------------------
# The version floor survives, in a different role.
# --------------------------------------------------------------------------
#
# A probe reports whether something COMPILES, not whether it is CORRECT, so a
# version check stays alongside it. What that check MEANS is the part worth
# stating, because it changed once already.
#
# It used to read "Clang before 14 has known-incomplete concepts support", which
# is a claim about Clang that nobody here had evidence for. It came from
# design/toolchain_requirements.md, which records it as a starting guess: the
# development server's system Clang 10 is shadowed by the conda environment, so
# it was never actually tried.
#
# Measured 2026-08-19, against this tree rather than a toy:
#
#   g++ 9.5      rejected, and early -- it does not accept -std=c++20 at all, so
#                the concepts probe above stops it before this check runs.
#   g++ 10       configures, builds the whole tree under -Werror with the full
#                warning set, and passes 3/3 ctest.
#   clang++ 10   the same, once the override this block used to demand was given.
#   g++ 14.4, clang++ 18.1.8   the development toolchains, likewise.
#
# So the floor is 10 for both families, and it now means THE LOWEST VERSION
# ACTUALLY EXERCISED rather than a guess about anyone's concepts implementation.
# "Untested below this" is a claim about us and we can support it; "broken below
# this" was a claim about the compiler and we could not.
#
# The check covers GNU as well as Clang, and that symmetry is the other half of
# the correction. It was Clang-only because the guess was about Clang's concepts,
# so a GCC 10 walked in with nothing said while a Clang 10 hit a hard stop --
# same year, same standard-library era, opposite treatment, neither measured.
#
# AppleClang stays excluded deliberately: its version numbers do not map to LLVM
# versions, so any threshold applied to it is a row that lies.
#
# Raise this floor only from an OBSERVED failure, never from speculation, or it
# becomes the version table this file exists to avoid.
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"
        AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10
   AND NOT PANTR_ALLOW_UNTESTED_COMPILER)
  message(FATAL_ERROR
      "pantr has never been built with ${CMAKE_CXX_COMPILER_ID} below version 10. "
      "Version 10 of both GCC and Clang is measured to build this tree and pass "
      "its tests; nothing older has been tried, and the probe above cannot tell a "
      "compiler that miscompiles from one that does not.\n"
      ${PANTR_TOOLCHAIN_REPORT}
      "\n  Override with -DPANTR_ALLOW_UNTESTED_COMPILER=ON at your own risk.")
endif()

unset(CMAKE_REQUIRED_FLAGS)
