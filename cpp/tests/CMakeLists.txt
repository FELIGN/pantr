# C++ tests. Plain executables registered with ctest -- see cpp/tests/check.hpp
# for why there is no test framework here.

set(PANTR_TEST_SOURCES
    test_dependency_smoke.cpp
    test_cardinal_bspline.cpp
    test_scalar_generic.cpp
    test_bernstein.cpp
    test_legendre.cpp
    test_change_basis.cpp
    test_quad.cpp
    test_binomial.cpp
    test_bezier.cpp
    test_bezier_root_finding.cpp
    test_grid_locate.cpp
    test_grid_bvh.cpp
    test_grid_hierarchical.cpp
    test_grid_blocks.cpp
    test_grid_hierarchical_type.cpp
    test_grid_hierarchical_refine.cpp
    test_aabb.cpp
    test_affine.cpp
    test_bezier_type.cpp
    test_bezier_evaluate.cpp
    test_bezier_degree.cpp
    test_bezier_shape.cpp
    test_quad_rule.cpp
    test_grid_tags.cpp
    test_grid_partition.cpp
    test_grid_bvh_type.cpp
    test_grid_defaults.cpp
    test_grid_tensor_product.cpp
    test_bezier_product.cpp
    test_format.cpp
    test_lazy.cpp
    test_bspline_knots.cpp
    test_bspline_space_1d.cpp
    test_bspline_space_nd.cpp
    test_extraction_kernels.cpp
    test_bspline_extraction.cpp
    test_bspline_knot_insertion.cpp
    test_thb_space.cpp
    test_bspline_type.cpp)

# Reserved slots (FELIGN/pantr#380): registered ahead of the port so the thirteen
# tickets this milestone unblocks do not collide on this file.
#
# EMPTY. `test_grid_tensor_product.cpp` was the last one and FELIGN/pantr#387 filled
# it, so `ctest -L placeholder` now reports zero tests and every green line in this
# directory comes from a file that asserts something.
#
# The list is kept rather than deleted because the mechanism is still owed: the
# milestone's remaining tickets reserve slots the same way, and the label is what
# tells a stub apart from a real test where the number is actually read. A stub
# prints PLACEHOLDER on stdout, but ctest captures stdout and shows it only on -V or
# on failure -- never on the pass path -- so without the label a reserved slot's
# summary row is indistinguishable from a real one and "N/N tests passed" reads as
# more evidence than it is.
#
#     ctest --preset gcc -N -L placeholder    # list what is not written yet
#     ctest --preset gcc -LE placeholder      # run only the tests that assert something
#
# Add an entry when a slot is reserved; delete it as its ticket fills the slot.
set(PANTR_PLACEHOLDER_TEST_SOURCES)

foreach(_src IN LISTS PANTR_TEST_SOURCES PANTR_PLACEHOLDER_TEST_SOURCES)
  cmake_path(GET _src STEM _name)
  add_executable(${_name} ${_src})
  target_link_libraries(${_name} PRIVATE pantr::core pantr::cxx_flags)
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${_name} COMMAND ${_name})
endforeach()

foreach(_src IN LISTS PANTR_PLACEHOLDER_TEST_SOURCES)
  cmake_path(GET _src STEM _name)
  set_tests_properties(${_name} PROPERTIES LABELS "placeholder")
endforeach()

list(LENGTH PANTR_PLACEHOLDER_TEST_SOURCES _pantr_placeholder_count)
message(STATUS
    "pantr: ${_pantr_placeholder_count} reserved test slot(s) assert nothing yet "
    "(ctest -L placeholder)")

# ---------------------------------------------------------------------------
# Negative translation units (FELIGN/pantr#386)
# ---------------------------------------------------------------------------
#
# Each of these is a program that MUST NOT COMPILE. The test builds it, and the pass
# criterion is the DIAGNOSTIC'S OWN TEXT rather than a non-zero exit -- the same
# distinction the precondition tests below make, and for the same reason. A negative
# translation unit rejected for the wrong reason (a typo, a moved header, a stale
# include path) also fails to build, so an exit-code test would pass while asserting
# nothing. Matching the message is what ties each file to the mechanism it exists for.
#
# Most of these are rejected by a `static_assert` inside PANTR_GRID_CENSUS, so their
# expected text is the assertion's own and is identical on every compiler. Where the
# rejection is a language rule instead, the pattern is the substring all four compilers
# were MEASURED to print; the measurement is in the file's own header.
#
# No pattern may contain a QUOTE around a compiler-quoted identifier, and that rule was
# bought the hard way. GCC renders its `%<...%>` quoting as U+2018/U+2019 in a UTF-8
# locale, so a pattern written with ASCII apostrophes matches on one GCC build and not
# on another -- measured: g++ 10.5 prints `as âthisâ argument`, where the conda g++ 14
# prints `as 'this' argument`. Match text the compiler writes as prose instead.
#
# The targets are OBJECT libraries -- nothing to link, no `main` to write -- and
# EXCLUDE_FROM_ALL, so `cmake --build` never tries to build them and only ctest does.
# They share a RESOURCE_LOCK because each drives a build of this same tree, and two
# concurrent ninja invocations in one directory contend.

function(pantr_add_negative_test _name _message)
  add_library(${_name} OBJECT EXCLUDE_FROM_ALL "negative/${_name}.cpp")
  target_link_libraries(${_name} PRIVATE pantr::core pantr::cxx_flags)
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/negative")
  add_test(NAME ${_name}
           COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
                   --target ${_name} --config $<CONFIG>)
  set_tests_properties(${_name} PROPERTIES
      PASS_REGULAR_EXPRESSION "${_message}"
      RESOURCE_LOCK pantr_negative_build
      LABELS "negative")
endfunction()

# The ticket's six, with two departures the design records and five additions.
#
#   * "a const grid invalidating its cache" became "nothing hands out the cache slot".
#     This ticket ships no `invalidate_caches()`: FELIGN/pantr#378 makes refinement
#     return a new grid, so there is nothing to invalidate, and the property the
#     mechanism existed to protect is that the slot is private and `cell_bvh()` yields
#     only a const reference to it.
#   * "a grid defined without its traits" is rejected at the grid's own definition,
#     which is why the pattern names the traits template rather than a census message.
#
# Five are additions: `num_local_facets` -- a `2 * ndim` invariant of the base's own
# `facet_tags_`, which nothing else asserts negatively -- the two directions of a hook
# disagreeing with its declaration, which are what make the bitmask trustworthy and
# which the acceptance criteria did not list, and the const-output-span trap on the two
# primitives the ticket did not name.
pantr_add_negative_test(neg_cache_handed_out_mutably "binding reference of type")
pantr_add_negative_test(neg_const_grid_mutates_tags "discards qualifiers|not marked const")
pantr_add_negative_test(neg_grid_without_traits "grid_traits<TraitlessGrid>")
pantr_add_negative_test(neg_primitive_const_output_span "not a grid")
pantr_add_negative_test(neg_locate_const_ref_point "not a grid")
pantr_add_negative_test(neg_neighbor_wrong_return "not a grid")
pantr_add_negative_test(neg_hook_wrong_return_type
                        "does not have the signature of the default")
pantr_add_negative_test(neg_hook_hardcodes_double_at_float "FloatGrid, locate_many")
pantr_add_negative_test(neg_hook_written_not_declared
                        "UndeclaredHookGrid, boundary_facets.: the class and its grid_traits")
pantr_add_negative_test(neg_hook_declared_not_written
                        "UnwrittenHookGrid, boundary_facets.: the class and its grid_traits")
pantr_add_negative_test(neg_num_local_facets_specialised
                        "num_local_facets is not specialisable")

# ---------------------------------------------------------------------------
# Precondition tests (FELIGN/pantr#359)
# ---------------------------------------------------------------------------
#
# Each of these expresses a documented precondition violation and is expected to
# abort. They are registered ONLY where the preconditions are checked: PANTR_PRECONDITION
# is `assert`, so a release build compiles it out and the executable would run the
# undefined behaviour instead of reporting it.
#
# The pass criterion is the assertion's own message rather than a non-zero exit.
# That distinction is the whole test: against the unfixed headers these programs also
# exit non-zero, because AddressSanitizer reports the overflow and aborts. Matching the
# message is what makes them fail against the code they were written for.

function(pantr_add_precondition_test _name _message)
  add_executable(${_name} ${_name}.cpp)
  target_link_libraries(${_name} PRIVATE pantr::core pantr::cxx_flags)
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${_name} COMMAND ${_name})
  set_tests_properties(${_name} PROPERTIES PASS_REGULAR_EXPRESSION "${_message}")
endfunction()

# An allow-list, not a deny-list, and the difference was measured. Denying Release and
# MinSizeRel still registered these under RelWithDebInfo, where CMake adds -DNDEBUG by
# default: the assertions are compiled out, so the executables run the undefined behaviour
# they exist to report. They never passed falsely, but one of the three ran for over five
# minutes without terminating, and ctest's computed timeout here is effectively unbounded.
# A multi-config generator leaves CMAKE_BUILD_TYPE empty, which the deny-list also let
# through.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  pantr_add_precondition_test(precondition_decode_flat_id
                              "cid must be at or above the first block's base")
  pantr_add_precondition_test(precondition_bvh_stack
                              "tree depth must fit the traversal stack")
  pantr_add_precondition_test(precondition_negative_degree
                              "degree must be non-negative")
  pantr_add_precondition_test(precondition_zero_extent
                              "every block must have a positive extent per axis")
  pantr_add_precondition_test(precondition_bezier_n_deriv
                              "n_deriv must be non-negative")
  pantr_add_precondition_test(precondition_block_of_midx
                              "level must be non-negative")
  message(STATUS "pantr: precondition tests enabled (build type ${CMAKE_BUILD_TYPE})")
else()
  message(STATUS "pantr: precondition tests skipped, assertions are compiled out here")
endif()
