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
#   hard gate      the floating-point overloads of std::to_chars. Exactly one
#                  function needs them, and a second spelling of what it does is
#                  the thing that must not happen, so a standard library without
#                  them is rejected.
#   feature toggle <mdspan>. The Kokkos reference implementation is a drop-in, so
#                  its absence selects a fallback rather than ending the build.
#
# The two hard gates bound different things, and that is why both exist. Concepts
# are a property of the compiler front end; std::to_chars is a property of the
# standard LIBRARY it happens to be paired with, and a toolchain can pass either
# gate while failing the other. Measured here: clang++ 10 passes both, because it
# resolves to the system libstdc++ 12, while g++ 10 passes the first and fails
# the second against its own libstdc++ 10.
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
# Hard gate: the floating-point overloads of std::to_chars.
# --------------------------------------------------------------------------
#
# pantr::detail::format_repr (cpp/include/pantr/core/format.hpp) reproduces
# Python's repr() of a float for the port's exception messages, which a parity
# test compares character for character. It gets the shortest round-tripping
# digits from std::to_chars and then picks the notation by Python's positional
# rule, and the digits are the half no other facility in the standard library
# provides.
#
# libstdc++ 10 implements std::to_chars for INTEGERS only. The floating-point
# overloads arrived in libstdc++ 11, and 10 defines no __cpp_lib_to_chars at all.
# Without this gate the refusal arrives as an overload-resolution error inside a
# header, part way through a build -- which is how FELIGN/pantr#376 was found,
# and the failure mode the rest of this file exists to convert into a message.
#
# This gate is deliberately NOT behind PANTR_ALLOW_UNTESTED_COMPILER. That flag
# says "I know this version is untested"; this is a facility that is absent, so
# opening the gate would buy nothing but the template error back.
#
# Reimplementing format_repr instead was considered and rejected in #376: the
# repr rule went wrong once as a local copy, and a second copy is how that
# happens again. format_general and format_fixed alongside it go through
# snprintf deliberately, which the old library does have, so this gate names one
# function rather than a file.
#
# The probe compiles the two calls format_repr actually makes -- the shortest-form
# overload at std::chars_format::scientific and at std::chars_format::fixed, neither
# taking a precision. Keep those two in step with format_repr, for the reason the
# concepts probe above gives.
#
# It does NOT consult __cpp_lib_to_chars, and that is the whole point. An earlier
# version of this probe tested the macro first and short-circuited before the calls,
# which refuses a working toolchain: libc++ implements the floating-point overloads
# and leaves the macro undefined -- measured on libc++ 19.1.7, whose <version> has
# every __cpp_lib_to_chars line commented out while the call below compiles clean.
# That would have refused every Clang paired with libc++, which on macOS is every
# AppleClang there is. Compiling the call is the only test that is right in both
# directions, and it is also what design/toolchain_requirements.md already asks for:
# probe the feature, never a version or a stand-in for one.
check_cxx_source_compiles("
#include <array>
#include <charconv>
#include <system_error>
int main() {
    std::array<char, 64> buffer{};
    const auto sci = std::to_chars(buffer.data(), buffer.data() + buffer.size(), 1.0,
                                   std::chars_format::scientific);
    if (sci.ec != std::errc{}) {
        return 1;
    }
    const auto fixed = std::to_chars(buffer.data(), buffer.data() + buffer.size(), 1.0,
                                     std::chars_format::fixed);
    return fixed.ec == std::errc{} ? 0 : 1;
}
" PANTR_HAS_FP_TO_CHARS)

if(NOT PANTR_HAS_FP_TO_CHARS)
  message(FATAL_ERROR
      "pantr requires the FLOATING-POINT overloads of std::to_chars (<charconv>), "
      "and this standard library does not provide them. The bound is the standard "
      "LIBRARY, not the compiler: libstdc++ 10 implements std::to_chars for "
      "integers only, while libstdc++ 11 and newer provide the floating-point "
      "overloads. A clang++ paired with a newer libstdc++, or with libc++, "
      "satisfies this gate at any front-end version.\n"
      "  Needed by: pantr::detail::format_repr, in "
      "cpp/include/pantr/core/format.hpp, which reproduces Python's repr() of a "
      "float exactly. There is no fallback: a second implementation of that rule "
      "is what this arrangement exists to prevent.\n"
      ${PANTR_TOOLCHAIN_REPORT})
endif()

# --------------------------------------------------------------------------
# Feature toggle: <mdspan>.
# --------------------------------------------------------------------------
#
# Absent from GCC 14, which is this project's own build compiler, so the Kokkos
# fallback is the normal path here rather than a precaution.
#
# The probe asks the question its own feature-test macro answers -- does the
# standard LIBRARY implement P0009? -- and then compiles the one expression the
# library actually depends on. It used to consult no macro at all and infer the
# answer from whether the body compiled, which conflates a missing header with a
# body that does not parse.
#
# The body indexes through `std::array`, because that is what `pantr::at` does.
# It used to index with `view[1, 2]`, the C++23 multidimensional subscript, which
# added a second question -- does the LANGUAGE have that operator? -- that the
# library no longer needs anyone to answer: the array overload is specified
# unconditionally in C++23 and provided unconditionally by Kokkos, so `at` uses
# it in both branches of the switch and parses under C++20 either way. Asking it
# anyway made the toggle answer OFF under a C++20 baseline no matter what the
# standard library shipped, which is the correct answer today for the wrong
# reason and the wrong answer the day one of them changes.
#
# Under the C++20 baseline this probe is still a foregone OFF, because libstdc++
# defines __cpp_lib_mdspan only in C++23 mode. That is now the single reason, it
# says so in the CMake log, and it will start answering ON by itself on the day
# the baseline moves.
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
#include <array>
#include <cstddef>
int main() {
    double storage[6]{};
    std::mdspan<double, std::dextents<std::size_t, 2>> view(storage, 2, 3);
    const std::array<std::size_t, 2> ij{1, 2};
    view[ij] = 1.0;
    return view[ij] == 1.0 ? 0 : 1;
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
# What backs it is a measurement rather than a date. scripts/ci_local.sh
# re-establishes the whole of it on EVERY run, and an absent toolchain is a
# failure there rather than a skip, so the claim cannot quietly become vacuous:
#
#   clang++-10   configured, built whole under -Werror with the full warning set,
#                and ctested. The lowest toolchain the tree is actually exercised
#                on, and therefore what "floor" names.
#   g++-10       asserted to be REFUSED at configure time, by the std::to_chars
#                gate above and by that gate's own message. Below the floor, and
#                checked as being below it.
#   g++ 9.5      asserted to be refused by the concepts gate, which stops it
#                before this check is ever reached.
#   g++ 14.4, clang++ 18.1.8   the development toolchains, built and ctested by
#                the same script on the same run.
#
# The floor used to be 10 for both families, from a hand measurement taken
# 2026-08-19 which recorded g++ 10 building the whole tree and passing 3/3 ctest.
# That row stopped being true the day cpp/include/pantr/core/format.hpp began
# needing floating-point std::to_chars, and nothing noticed, because the check
# that would have said so reported an absent compiler and a passing one
# identically. Both halves of that are FELIGN/pantr#376.
#
# So the floor is now two bounds, and they are not the same bound:
#
#   the standard library   binding, and raised: libstdc++ 11 or newer. Enforced by
#                          the std::to_chars probe above, which is a feature test
#                          rather than a version comparison.
#   the compiler version   10 for GCC and Clang alike, unchanged, and meaning
#                          UNTESTED BELOW THIS rather than a guess about anyone's
#                          concepts implementation. That is a claim about us and we
#                          can support it; "broken below this" was a claim about
#                          the compiler and we could not.
#
# That second line used to read "the lowest version ACTUALLY EXERCISED", and the
# two readings were the same thing while both families' 10 built the tree. They
# are not any more. The lowest GNU actually exercised is now the environment's 14
# -- g++ 10 is refused by the gate above and nothing between the two is installed
# here -- while clang++ 10 still makes both readings true at once. Only the
# weaker one is supportable for both families, so only the weaker one is claimed.
#
# clang++ 10 is what holds those two apart, and is why the GNU row of the check
# below did not simply move to 11: a 2020 front end that satisfies the raised
# floor, because it resolves to the system libstdc++ 12. Writing 11 here instead
# would be a version number standing in for a library fact -- precisely the row
# that lies.
#
# The check covers GNU as well as Clang, and that symmetry is the other half of
# the correction. It was Clang-only because the guess was about Clang's concepts,
# so a GCC 10 walked in with nothing said while a Clang 10 hit a hard stop --
# same year, opposite treatment, neither measured. (Not the same standard library,
# as it turned out, and that is the whole of FELIGN/pantr#376: on this machine the
# two resolve to libstdc++ 10 and libstdc++ 12 respectively. The symmetry being
# corrected here was the right correction to make; the assumption underneath it,
# that a shared year implies a shared library, was not.)
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
      "Clang 10 is measured to build this tree and pass its tests, against a "
      "libstdc++ new enough for the gate above; nothing older than 10 has been "
      "tried in either family, and the probes above cannot tell a compiler that "
      "miscompiles from one that does not.\n"
      ${PANTR_TOOLCHAIN_REPORT}
      "\n  Override with -DPANTR_ALLOW_UNTESTED_COMPILER=ON at your own risk.")
endif()

unset(CMAKE_REQUIRED_FLAGS)
