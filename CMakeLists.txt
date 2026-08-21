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

project(pantr_cpp
        VERSION 0.7.0
        DESCRIPTION "C++ core for pantr (prototype)"
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>")

target_link_libraries(pantr_core INTERFACE pantr::cxx_flags Eigen3::Eigen)

if(PANTR_HAS_STD_MDSPAN)
  target_compile_definitions(pantr_core INTERFACE PANTR_HAS_STD_MDSPAN=1)
else()
  target_link_libraries(pantr_core INTERFACE std::mdspan)
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
