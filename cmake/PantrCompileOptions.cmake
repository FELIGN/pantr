# The interface target every pantr C++ target links against.
#
# Two decisions from design/toolchain_requirements.md and design/simd.md are
# enforced here, because CMake is the only place they can be enforced once.
#
# ---------------------------------------------------------------------------
# 1. Never -ffast-math, and never -ffinite-math-only
# ---------------------------------------------------------------------------
#
# -ffast-math permits REASSOCIATION of sums, and reassociation invalidates the
# error bounds that every derived tolerance in this library assumes: a tolerance
# derived under one association order is not a tolerance under another. It is
# not a performance option, it is a silent correctness change, and the failure it
# produces is a wrong number rather than a crash.
#
# -ffinite-math-only is banned for a second reason on top of that one: it tells
# the compiler no infinity or NaN can occur, which licenses it to delete the
# very checks that detect a degenerate input.
#
# -funsafe-math-optimizations alone may be offered behind an explicit option one
# day. It is not offered today.
#
# The check below covers flags reaching the build through CMAKE_CXX_FLAGS and
# the per-configuration variants, which is how they actually arrive (an
# environment CXXFLAGS, a conda activation script, a well-meaning preset).
#
# ---------------------------------------------------------------------------
# 2. The floating-point contraction flag goes on the INTERFACE target
# ---------------------------------------------------------------------------
#
# Any flag that participates in a numerical claim must reach every variant, or
# the ISA variants of design/simd.md will disagree numerically with each other
# and the resulting bug will look like a dispatch bug. Setting it per target is
# how that happens; setting it here is how it is prevented.
#
# The value is `on` rather than `fast`, and the choice is load-bearing for the
# parity tolerance rather than a matter of taste:
#
#   `on`   is the standard's FP_CONTRACT behaviour: a multiply may be fused into
#          an add only within a single expression. The set of fused sites is
#          then determinable by reading the source, which is what makes the
#          parity bound in tests/parity/test_basis_cardinal_bspline.py DERIVED
#          rather than fitted.
#   `fast` allows contraction across statements, so the fused set depends on
#          what the optimiser chose to hoist, and no bound written by hand can
#          be checked against it.
#
# It is also set explicitly because the compilers disagree on their default:
# GCC defaults to `fast` and Clang to `on`. Leaving it implicit would mean the
# two compilers compute different numbers, which is precisely the disagreement
# the interface target exists to prevent.
#
# Caveat: GCC only implements `on` distinctly from `fast` since GCC 14. On
# GCC 13 and older the flag is accepted and silently means `fast`, so the fused
# set is wider than the bound assumes there. The concepts gate already rejects
# compilers that old in practice; noted so it is not a surprise.

include_guard(GLOBAL)

set(_pantr_forbidden_math_flags "-ffast-math" "-ffinite-math-only" "/fp:fast")

foreach(_flag_var
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL)
  foreach(_forbidden IN LISTS _pantr_forbidden_math_flags)
    if("${${_flag_var}}" MATCHES "(^| )${_forbidden}( |$)")
      message(FATAL_ERROR
          "pantr refuses to build with ${_forbidden}, found in ${_flag_var}:\n"
          "    ${${_flag_var}}\n"
          "It permits reassociation of floating-point sums, which invalidates "
          "the error bound behind every derived tolerance in this library. "
          "This is a correctness setting, not a performance one, and its "
          "failure mode is a wrong number rather than a crash.")
    endif()
  endforeach()
endforeach()

add_library(pantr_cxx_flags INTERFACE)
add_library(pantr::cxx_flags ALIAS pantr_cxx_flags)

target_compile_features(pantr_cxx_flags INTERFACE cxx_std_20)

# -ffp-contract=on used to live here and now sits on `pantr_core` in the
# top-level CMakeLists, deliberately. This target is the project's warning
# policy, which is internal and must not reach a consumer; the contraction
# setting is part of the numerical contract and must. Keeping it in both places
# would give a load-bearing flag two sources of truth. Every target that links
# this one also links pantr::core, so nothing here lost the flag.

# -Wsign-conversion is named explicitly because -Wconversion does NOT mean the
# same thing in the two compilers: Clang implies it, GCC does not. Measured on an
# `std::int64_t` indexing an `std::array` -- `g++ -Wconversion` warns 0 times,
# `g++ -Wconversion -Wsign-conversion` and `clang++ -Wconversion` warn once each.
# The asymmetry let six sites in grid/bvh.hpp break the Clang build for two days
# while the GCC-only CI stayed green, and only scripts/ci_local.sh runs Clang.
target_compile_options(pantr_cxx_flags INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wdouble-promotion>)

if(PANTR_WERROR)
  target_compile_options(pantr_cxx_flags INTERFACE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>)
endif()

# Deliberately absent: any -march or -mtune. Shipping several ISA variants is
# stage 2 in design/simd.md, and it is gated there on first MEASURING the gap
# between the baseline and x86-64-v3 on pantr's own kernels. Adding the flag
# here would spend that decision before the measurement that justifies it.
