# Build for the pantr C++ core.
#
# PROTOTYPE. This tree exists to answer whether the C++ port is worth pursuing,
# not to ship it: one kernel, measured against the Numba implementation that
# stays as the parity oracle. It is meant to be abandonable. See design/*.md for
# the decisions it implements and cpp/README.md for what it does and does not
# cover.
#
# Two entry points, both supported:
#
#   standalone   cmake --preset gcc && cmake --build --preset gcc && ctest --preset gcc
#                Builds the C++ tests and the benchmark. No Python involved.
#
#   Python       pip install -e .    (through scikit-build-core)
#                Builds the nanobind extension only.

cmake_minimum_required(VERSION 3.25)  # SYSTEM on FetchContent_Declare

# The version is read rather than repeated. pyproject.toml calls `__version__` in
# src/pantr/__init__.py the single source of truth, and it was true of the Python
# package only: this line carried its own copy. Exporting a package version makes
# the drift consequential, since the installed pantrConfigVersion.cmake would
# claim a version the Python package disagrees with.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/pantr/__init__.py" _pantr_init)
string(REGEX MATCH "__version__[^=]*=[ \t]*\"([^\"]+)\"" _ "${_pantr_init}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "could not read __version__ from src/pantr/__init__.py")
endif()
set(PANTR_VERSION "${CMAKE_MATCH_1}")
unset(_pantr_init)

project(pantr_cpp
        VERSION ${PANTR_VERSION}
        DESCRIPTION "C++ core for pantr (prototype)"
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PANTR_BUILD_TESTS "Build the C++ tests" ON)
option(PANTR_BUILD_BENCHMARK "Build the C++ benchmark" ON)
# OFF by default and switched on explicitly by pyproject.toml's
# cmake.define, rather than inferred from a scikit-build-core variable: the
# name and truthiness of those variables are the build backend's to change, and
# a silently-OFF extension produces a wheel that installs cleanly and then
# fails at import.
option(PANTR_BUILD_PYTHON "Build the nanobind extension module" OFF)
option(PANTR_WERROR "Treat compiler warnings as errors" ON)
option(PANTR_INSTALL "Generate install rules for the C++ core" "${PROJECT_IS_TOP_LEVEL}")
option(PANTR_USE_SYSTEM_DEPS
       "Accept an installed Eigen3 / mdspan instead of the pinned SHAs" OFF)
option(PANTR_ALLOW_UNTESTED_COMPILER
       "Bypass the known-broken-compiler filter (see cmake/PantrCompilerProbes.cmake)" OFF)

# Order matters: the probes set PANTR_HAS_STD_MDSPAN, which decides whether the
# dependency step fetches Kokkos at all.
include(PantrCompilerProbes)
include(PantrDependencies)
include(PantrCompileOptions)

# --------------------------------------------------------------------------
# The core: header-only, scalar-generic.
# --------------------------------------------------------------------------
add_library(pantr_core INTERFACE)
add_library(pantr::core ALIAS pantr_core)

target_include_directories(pantr_core INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# `pantr::cxx_flags` is deliberately NOT linked here. It carries this project's
# warning policy and, under PANTR_WERROR, `-Werror`; exporting either would decide
# a consumer's build for them, and a new warning in their compiler would break it
# over our headers. Our own tests, benchmark and bindings link it explicitly.
#
# What DOES cross is the floating-point semantics, because the parity bounds in
# design/backend_parity.md are derived by counting roundings per stage and
# `-ffp-contract` decides which sites fuse. A consumer compiling these headers
# under different contraction gets different numbers than the bounds describe.
target_compile_options(pantr_core INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=on>)

# BUILD_INTERFACE, and CMake enforces it rather than merely preferring it: a
# fetched Eigen is not in any export set, so exporting a target that links it
# fails outright. That refusal is correct and worth understanding, because it
# separates two different things -- which Eigen WE compiled and verified
# against, and which one the consumer will compile these headers with.
#
# pantrConfig.cmake supplies the consumer's, through find_dependency(Eigen3).
# The consequence is real and belongs in the record: this project's parity
# results were measured against the pinned Eigen, and anything that goes through
# Eigen under a consumer's different version is outside what was measured.
target_link_libraries(pantr_core INTERFACE $<BUILD_INTERFACE:Eigen3::Eigen>)

# Threads, and unlike Eigen this one DOES cross the install boundary.
# `pantr/core/lazy.hpp` puts an `std::mutex` and an `std::atomic<bool>` inside a
# header, so the thread-safety contract the domain types promise is compiled into
# every consumer's translation unit rather than into a library they link. On a
# glibc new enough to have folded the pthread symbols into libc this changes
# nothing; on anything older it is the difference between linking and not, and a
# consumer has no way to know they owe it. `find_dependency(Threads)` in
# cmake/pantrConfig.cmake.in is the other half.
# `FindThreads` recommends this: without it a platform may be linked with a bare
# `-lpthread` instead of the `-pthread` compiler flag, and some libc versions need
# the flag's preprocessor defines for thread-safe behaviour rather than merely the
# library. Inert on every configuration exercised today -- glibc has folded the
# pthread symbols into libc, and CMAKE_HAVE_LIBC_PTHREAD comes back true -- which
# is exactly why it would go unnoticed on the first platform where it is not.
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
target_link_libraries(pantr_core INTERFACE Threads::Threads)

# BUILD_INTERFACE only, deliberately. cpp/include/pantr/core/mdspan.hpp decides
# for itself from `__cpp_lib_mdspan`, so this definition is redundant in-tree and
# would be actively wrong once installed: the probe measures the compiler that
# built the package, and a header-only library is compiled by the consumer's.
if(PANTR_HAS_STD_MDSPAN)
  target_compile_definitions(pantr_core INTERFACE
      $<BUILD_INTERFACE:PANTR_HAS_STD_MDSPAN=1>)
else()
  # BUILD_INTERFACE: whether the consumer needs Kokkos depends on THEIR standard
  # library, which mdspan.hpp now decides for itself. pantrConfig.cmake attaches
  # it when it is both found and needed.
  target_link_libraries(pantr_core INTERFACE $<BUILD_INTERFACE:std::mdspan>)
endif()

# --------------------------------------------------------------------------
# Install and export
# --------------------------------------------------------------------------
#
# Header-only, so this installs headers and a target description and nothing
# else. What the exported target carries, and why, is argued at its definition
# above: the include directory, `-ffp-contract=on`, and Eigen. Not the warning
# flags, and not the mdspan choice.
if(PANTR_INSTALL)
  include(CMakePackageConfigHelpers)

  # EXPORT_NAME, or the namespace is applied to the target's own name and the
  # exported target is `pantr::pantr_core` while the in-tree alias is
  # `pantr::core`. Caught by the out-of-tree consumer in cpp/consumer, which is
  # the only thing that exercises the installed package rather than this tree.
  set_target_properties(pantr_core PROPERTIES EXPORT_NAME core)

  install(TARGETS pantr_core EXPORT pantrTargets)

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/pantr"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  install(EXPORT pantrTargets
          FILE pantrTargets.cmake
          NAMESPACE pantr::
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pantr")

  configure_package_config_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pantrConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/pantrConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pantr")

  # SameMajorVersion is wrong below 1.0, where a minor bump may break the API.
  # ExactVersion would be unusable, so SameMinorVersion is the honest middle:
  # 0.8.1 satisfies a request for 0.8, and 0.9.0 does not.
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/pantrConfigVersion.cmake"
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMinorVersion)

  install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/pantrConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/pantrConfigVersion.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pantr")
endif()

if(PANTR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(cpp/tests)
endif()

if(PANTR_BUILD_BENCHMARK)
  add_subdirectory(cpp/benchmark)
endif()

if(PANTR_BUILD_PYTHON)
  add_subdirectory(cpp/bindings)
endif()

# Spelled out rather than interpolated. PANTR_HAS_STD_MDSPAN is EMPTY when the
# probe fails, so "std::mdspan ${PANTR_HAS_STD_MDSPAN}" rendered as a bare
# "std::mdspan" on exactly the builds that are not using it -- a summary line
# asserting the opposite of the truth about the only toggle in this build.
if(PANTR_HAS_STD_MDSPAN)
  set(_pantr_mdspan "std::mdspan")
else()
  set(_pantr_mdspan "Kokkos mdspan")
endif()

message(STATUS "pantr: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, "
               "build type ${CMAKE_BUILD_TYPE}, "
               "${_pantr_mdspan}, "
               "tests ${PANTR_BUILD_TESTS}, python ${PANTR_BUILD_PYTHON}")
