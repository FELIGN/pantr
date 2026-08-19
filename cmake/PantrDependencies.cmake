# Third-party dependencies for the pantr C++ core, fetched at configure time.
#
# Three rules govern every entry here, and each exists because of a failure that
# is confusing when it happens:
#
#   GIT_TAG is a full 40-character SHA. A tag is a mutable pointer: it can be
#   moved or deleted upstream, which turns a reproducible build into one that
#   silently changes. The human-readable tag each SHA corresponds to is recorded
#   in the comment beside it, since a SHA alone tells a reader nothing.
#
#   SYSTEM marks the dependency's headers as system headers, so warnings raised
#   inside them do not reach pantr's -Werror. Without it, upstream's warning
#   discipline decides whether pantr builds. Needs CMake 3.25 or newer.
#
#   EXCLUDE_FROM_ALL keeps the dependency's own targets (its tests, its
#   benchmarks, its install rules) out of pantr's default build and out of
#   pantr's install tree.
#
# ---------------------------------------------------------------------------
# CMake 4 is strict about what dependencies declare
# ---------------------------------------------------------------------------
#
# CMake 4 REMOVED compatibility with cmake_minimum_required(VERSION < 3.5). A
# project declaring an older minimum fails hard at configure time rather than
# warning. That matters here specifically because FetchContent pulls
# CMakeLists.txt files nobody in this project controls, and one of them
# declaring an ancient minimum takes the build down with an error that reads as
# if it came from pantr's own CMake.
#
# The escape hatch exists for exactly this, and is documented here rather than
# discovered:
#
#     cmake --preset gcc -DCMAKE_POLICY_VERSION_MINIMUM=3.5
#
# ---------------------------------------------------------------------------
# The offline escape
# ---------------------------------------------------------------------------
#
# FetchContent needs network access at configure time. The development server
# has it; a cluster compute node typically does not. Point each dependency at a
# pre-populated checkout instead:
#
#     cmake --preset gcc \
#         -DFETCHCONTENT_SOURCE_DIR_MDSPAN=/path/to/mdspan \
#         -DFETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/eigen
#
# or, with a fully populated FetchContent cache, forbid network access outright
# so a missing entry fails loudly instead of hanging on a clone:
#
#     cmake --preset gcc -DFETCHCONTENT_FULLY_DISCONNECTED=ON
#
# The variable name is FETCHCONTENT_SOURCE_DIR_<NAME> with <NAME> upper-cased
# from the FetchContent_Declare name, hence MDSPAN and EIGEN below.

include_guard(GLOBAL)
include(FetchContent)

# --------------------------------------------------------------------------
# Kokkos mdspan -- the reference implementation of std::mdspan.
# --------------------------------------------------------------------------
#
# Fetched only when the standard library does not provide <mdspan>. GCC 14, the
# build compiler on the development server, does not, so this is the normal path
# here rather than a fallback for exotic toolchains.
if(NOT PANTR_HAS_STD_MDSPAN)
  FetchContent_Declare(
      mdspan
      GIT_REPOSITORY https://github.com/kokkos/mdspan.git
      GIT_TAG        9ceface91483775a6c74d06ebf717bbb2768452f  # tag mdspan-0.6.0
      GIT_SHALLOW    FALSE
      SYSTEM
      EXCLUDE_FROM_ALL)

  # Ask Kokkos to place its symbols in the namespace the C++23 header would use,
  # so cpp/include/pantr/core/mdspan.hpp aliases one name in both branches of
  # the switch rather than papering over two different spellings.
  set(MDSPAN_CXX_STANDARD 20 CACHE STRING "" FORCE)
  set(MDSPAN_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(MDSPAN_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MDSPAN_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(mdspan)
endif()

# --------------------------------------------------------------------------
# Eigen -- dense and sparse linear algebra.
# --------------------------------------------------------------------------
#
# Not used by the one kernel this prototype ships. It is fetched and compiled
# against anyway, by cpp/tests/test_dependency_smoke.cpp, because the question
# this prototype exists to answer is whether the dependency survives pantr's own
# warning flags and this CMake version -- and a header that is never included
# answers nothing. design/large_data_fitting.md and
# design/adaptive_thb_approximation.md both settle on Eigen::SparseMatrix with
# SimplicialLDLT for the fitting solves, which is what it is here for.
#
# Eigen's own CMakeLists is run rather than bypassed, deliberately: bypassing it
# (SOURCE_SUBDIR pointing at a directory with no CMakeLists) would hide exactly
# the CMake 4 incompatibility this prototype is meant to measure.
# Fetched only when the tests are built. Nothing in the installed extension
# includes an Eigen header, so a `pip install` has no business cloning it -- and
# a dependency a wheel build does not need is a dependency a wheel build cannot
# fail on.
if(PANTR_BUILD_TESTS)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        bc3b39870ecb690a623a3f49149a358b95c5781d  # tag 5.0.1
    GIT_SHALLOW    FALSE
    SYSTEM
    EXCLUDE_FROM_ALL)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_CMAKE_PACKAGE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen)

endif()
