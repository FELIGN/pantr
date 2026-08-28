# The nanobind extension module.
#
# nanobind is located through the interpreter rather than through a system
# package, because the version that matters is the one in the environment doing
# the build -- which under scikit-build-core is the isolated build environment,
# not the developer's shell.

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE _nanobind_probe)

if(NOT _nanobind_probe EQUAL 0)
  message(FATAL_ERROR
      "nanobind was not importable by ${Python_EXECUTABLE}.\n"
      "  Fix: pip install nanobind")
endif()

find_package(nanobind CONFIG REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.__version__)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PANTR_NANOBIND_VERSION)
message(STATUS "pantr: nanobind ${PANTR_NANOBIND_VERSION}")

# Deliberately NOT built with STABLE_ABI.
#
# The stable ABI buys one wheel per platform instead of one per Python version,
# at the cost of routing every call through the limited API's slower dispatch.
# This prototype's whole purpose includes measuring how fast the C++ kernel is
# against numba, and a measurement taken through the stable-ABI dispatcher would
# understate it while telling nobody which part of the gap was the port and which
# was the packaging.
#
# The wheel strategy is a separate decision and is not this PR's to make.
# design/_memory/nanobind-status.md records that nanobind's split mode supersedes
# the stable-ABI approach on both axes that mattered, and that it was still
# unreleased (3.0.0.dev2, nanobind-backend 1.0.0.dev2) when this was written.
# scripts/ci_local.sh exercises split mode against the dev release so the answer
# is measured rather than assumed.
# NOMINSIZE is load-bearing, and its absence cost this prototype its headline
# measurement once already.
#
# nanobind_add_module() optimises for SIZE by default: it appends `-Os
# -ffunction-sections -fdata-sections` AFTER the build type's `-O3`, and the
# last optimisation flag on the command line is the one that takes effect. That
# is the right default for nanobind -- a binding module is mostly dispatch glue,
# where size is what costs -- and the wrong one here, because the numerical
# kernel is compiled into this same translation unit and silently inherits it.
#
# Measured on this machine, cardinal B-spline tabulation, degree 3, 10^6 points,
# GCC 14.4.0, against the numba kernel on one thread:
#
#     -Os (nanobind's default)   31.4 ms   0.44x
#     -O3 (NOMINSIZE)            10.3 ms   1.33x
#
# So this one word decides whether the prototype reports the port as three times
# slower than numba or a third faster, on identical source. The standalone
# benchmark in cpp/benchmark/ never saw it, because it is a plain executable and
# nanobind's CMake never touches it -- which is exactly how the discrepancy was
# noticed, and an argument for keeping both benchmarks rather than one.
#
# The cost is a larger extension. That is the trade nanobind's default declines
# to make on your behalf, and it is obviously the right way round once a kernel
# lives in the module.
#
# NOSTRIP is the same argument applied to the symbol table.
#
# nanobind_add_module() strips the module by default, which is right for a binding
# that is only ever imported and wrong for one whose stated purpose is to be
# MEASURED. Verified on the shipped build before this was added: `nm` on the
# installed .so reported "no symbols", so `perf` could attribute time to the module
# but not to any function inside it, and a segfault produced a backtrace with
# nothing in it but an address. Both are exactly what this prototype exists to
# look at, and the second one was needed while these bindings were being fixed.
#
# The cost is disk and nothing else: a symbol table does not change the code that
# is generated, so no measurement is affected by its presence.
#
# ---------------------------------------------------------------------------
# PANTR_NANOBIND_SPLIT: the wheel-strategy probe, off by default
# ---------------------------------------------------------------------------
#
# Split mode (nanobind 3) replaces the stable-ABI approach: the extension splits
# into a frontend targeting the Python 3.10 stable ABI, one wheel per platform
# covering every version, plus a small version-specific backend installed from
# PyPI. design/_memory/nanobind-status.md records why that is better on both
# axes that mattered, and that its composition with a multi-ISA build is
# UNVERIFIED and must be tested by this skeleton.
#
# It is an option rather than the default because nanobind 3 is a pre-release
# and because it makes `nanobind-backend` a RUNTIME dependency of the wheel,
# which is a packaging decision this PR does not get to make. scripts/ci_local.sh
# turns it on against the dev release so the answer is measured.
#
# NB_STATIC and split mode are mutually exclusive: NB_STATIC links libnanobind
# into the extension, which is precisely what split mode moves out of it.
option(PANTR_NANOBIND_SPLIT "Build the extension in nanobind split mode (needs nanobind 3)" OFF)

if(PANTR_NANOBIND_SPLIT)
  if(PANTR_NANOBIND_VERSION VERSION_LESS 3)
    message(FATAL_ERROR
        "PANTR_NANOBIND_SPLIT needs nanobind 3 or newer; found ${PANTR_NANOBIND_VERSION}.\n"
        "  Fix: pip install --pre 'nanobind>=3.0.0.dev0'")
  endif()
  # NB_SUPPRESS_WARNINGS is required here and has no equivalent below.
  #
  # In split mode nanobind adds its own include directory straight onto the
  # extension target, and marks it SYSTEM only when this option is passed. The
  # foreach further down cannot help: it retags the include directories of
  # `nanobind-static`, and split mode creates no such target -- the whole point
  # is that libnanobind is not linked in. Measured with nanobind 3.0.0.dev2 and
  # GCC 14: without this, `nb_attr.h:78: using arg = arg_t<>` trips -Wshadow and
  # -Werror ends the build.
  nanobind_add_module(_pantr_cpp NOMINSIZE NOSTRIP NB_SUPPRESS_WARNINGS
                      BACKEND_MODULE nanobind_backend
                      pantr_cpp.cpp basis.cpp quad.cpp change_basis.cpp bezier.cpp
                      bezier_root_finding.cpp grid.cpp geometry.cpp transform.cpp
                      grid_types.cpp grid_tags.cpp grid_bvh.cpp grid_partition.cpp
                      quad_types.cpp bezier_type.cpp)
else()
  nanobind_add_module(_pantr_cpp NB_STATIC NOMINSIZE NOSTRIP
                      pantr_cpp.cpp basis.cpp quad.cpp change_basis.cpp bezier.cpp
                      bezier_root_finding.cpp grid.cpp geometry.cpp transform.cpp
                      grid_types.cpp grid_tags.cpp grid_bvh.cpp grid_partition.cpp
                      quad_types.cpp bezier_type.cpp)
endif()

# nanobind's own headers are not -Wshadow clean (measured: 2.14.0 on GCC 14
# raises `declaration of 'value' shadows a member` in nb_types.h, nb_attr.h and
# several others), and pantr builds with -Werror. Marking them SYSTEM is the same
# decision cmake/PantrDependencies.cmake takes for the fetched dependencies: a
# third party's warning discipline must not decide whether pantr builds.
#
# It has to be done by hand here, and the reason is worth recording because the
# asymmetry is not obvious. CMake treats an IMPORTED target's interface includes
# as system by default, but `nanobind-static` is BUILT by nanobind_add_module
# rather than imported, so its includes arrive as -I. Re-adding them as -isystem
# on the consuming target does not work either: GCC drops an -isystem path that
# already appeared as -I. The property has to move on the producing target.
foreach(_nb_target IN ITEMS nanobind-static nanobind-static-abi3 nanobind)
  if(TARGET ${_nb_target})
    get_target_property(_nb_includes ${_nb_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_nb_includes)
      set_target_properties(${_nb_target} PROPERTIES
          INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_nb_includes}")
    endif()
  endif()
endforeach()

target_link_libraries(_pantr_cpp PRIVATE pantr::core pantr::cxx_flags)

# scikit-build-core installs whatever lands in the install tree. The extension
# belongs next to the pure-Python package so `pantr._pantr_cpp` resolves.
install(TARGETS _pantr_cpp LIBRARY DESTINATION pantr)
