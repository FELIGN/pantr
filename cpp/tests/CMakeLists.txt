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
    test_aabb.cpp
    test_affine.cpp
    test_bezier_type.cpp
    test_quad_rule.cpp
    test_grid_tags.cpp
    test_grid_partition.cpp)

# Reserved slots (FELIGN/pantr#380): registered ahead of the port so the thirteen
# tickets this milestone unblocks do not collide on this file. Each asserts nothing
# yet; see the source file for what it is reserved for.
#
# They are a SEPARATE list because they are labelled `placeholder`, and the label is
# the only thing that tells them apart where the number is actually read. A stub
# prints PLACEHOLDER on stdout, but ctest captures stdout and prints it on -V or on
# failure -- never on the pass path, and these pass by construction. So the summary
# row for a stub is indistinguishable from a real one, and "N/N tests passed" reads
# as more evidence than it is. With the label:
#
#     ctest --preset gcc -N -L placeholder    # list what is not written yet
#     ctest --preset gcc -LE placeholder      # run only the tests that assert something
#
# Delete each entry from this list as its ticket fills the slot; when the list is
# empty, delete the list.
set(PANTR_PLACEHOLDER_TEST_SOURCES
    test_grid_bvh_type.cpp
    test_grid_defaults.cpp
    test_grid_tensor_product.cpp
    test_bezier_evaluate.cpp
    test_bezier_degree.cpp
    test_bezier_shape.cpp
    test_bezier_product.cpp)

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
