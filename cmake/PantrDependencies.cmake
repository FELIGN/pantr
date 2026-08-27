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
#
# ---------------------------------------------------------------------------
# An installed copy is preferred over a fetched one
# ---------------------------------------------------------------------------
#
# Both entries carry FIND_PACKAGE_ARGS, so FetchContent_MakeAvailable tries
# find_package() first and clones only when that fails. Point it at your own
# build and nothing is fetched:
#
#     cmake --preset gcc -DEigen3_ROOT=/path/to/eigen -Dmdspan_ROOT=/path/to/mdspan
#
# or set CMAKE_PREFIX_PATH once for both. This is what makes the tree installable
# by a distributor who already has these: the exported package requires Eigen3
# from the system, and building against a fetched copy while installing against a
# system one would be two different Eigens.
#
# To force the fetch even when a system copy exists, for a reproducible build
# pinned to the SHAs below:
#
#     cmake --preset gcc -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER

include_guard(GLOBAL)
include(FetchContent)

# OFF by default, and the default is the load-bearing half. FIND_PACKAGE_ARGS on
# its own makes find_package() the FIRST choice, which quietly defeats the SHA
# pinning above: measured on the development server, the build silently moved
# from the pinned Eigen to /usr/include/eigen3 -- two major versions apart -- and
# every check passed without anything reporting which Eigen it had verified.
#
# So an installed copy is used only when asked for. Reproducible by default,
# distributor-friendly on request.
if(NOT PANTR_USE_SYSTEM_DEPS)
  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
else()
  message(STATUS "pantr: accepting installed Eigen3/mdspan; the pinned SHAs are "
                 "a fallback, so the versions in use are not this project's")
endif()

# A root passed without the option would otherwise be ignored in silence, which
# is the same class of failure the option exists to prevent.
foreach(_dep Eigen3 mdspan)
  if(NOT PANTR_USE_SYSTEM_DEPS AND (DEFINED ${_dep}_ROOT OR DEFINED ENV{${_dep}_ROOT}))
    message(WARNING "${_dep}_ROOT is set but PANTR_USE_SYSTEM_DEPS is OFF, so it "
                    "is ignored and the pinned SHA is fetched. Pass "
                    "-DPANTR_USE_SYSTEM_DEPS=ON to use it.")
  endif()
endforeach()

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
      EXCLUDE_FROM_ALL
      FIND_PACKAGE_ARGS NAMES mdspan)

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
# A dependency of the shipped extension, fetched unconditionally.
#
# It was test-only until the change-of-basis port, on the reasoning that "a
# dependency a wheel build does not need is a dependency a wheel build cannot
# fail on". That reasoning held only while nothing shipped needed a solve. Two
# stage-1 modules do: change_basis inverts four matrices per builder pair
# (Eigen::PartialPivLU, standing in for numpy's LAPACK gesv), and bezier's
# interpolation takes an SVD pseudo-inverse of a Bernstein Vandermonde with rank
# truncation, which is not a thing to hand-roll. Outside stage 1, bspline's
# lstsq/pinv/eigh sites and the SimplicialLDLT solves that
# design/large_data_fitting.md and design/adaptive_thb_approximation.md settle
# on are all waiting behind the same header.
#
# Eigen's own CMakeLists is run rather than bypassed, deliberately: bypassing it
# (SOURCE_SUBDIR pointing at a directory with no CMakeLists) would hide exactly
# the CMake 4 incompatibility this prototype is meant to measure.
#
# GIT_SHALLOW is TRUE here and FALSE for mdspan above, which is a measurement
# rather than a preference. Eigen is now on the pip path, so every cold editable
# install pays its clone: measured on the development server, a cold
# `pip install -e . --no-build-isolation` costs 7.56 s without Eigen, 21.48 s
# with it cloned deep, and 12.19 s cloned shallow, with the build tree at
# 6.4 MB / 145 MB / 50 MB respectively. The incremental rebuild is unaffected
# (2.96 s). Shallow against a pinned commit needs the server to serve an
# arbitrary SHA; GitLab does, and the checkout was verified to be exactly the
# SHA below rather than a silent fallback.

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        bc3b39870ecb690a623a3f49149a358b95c5781d  # tag 5.0.1
    GIT_SHALLOW    TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS NAMES Eigen3)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_CMAKE_PACKAGE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen)
