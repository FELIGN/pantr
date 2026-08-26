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
    test_grid_hierarchical.cpp)

foreach(_src IN LISTS PANTR_TEST_SOURCES)
  cmake_path(GET _src STEM _name)
  add_executable(${_name} ${_src})
  target_link_libraries(${_name} PRIVATE pantr::core)
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${_name} COMMAND ${_name})
endforeach()

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
  target_link_libraries(${_name} PRIVATE pantr::core)
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${_name} COMMAND ${_name})
  set_tests_properties(${_name} PROPERTIES PASS_REGULAR_EXPRESSION "${_message}")
endfunction()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  pantr_add_precondition_test(precondition_decode_flat_id
                              "cid must be at or above the first block's base")
  pantr_add_precondition_test(precondition_bvh_stack
                              "tree depth must fit the traversal stack")
  pantr_add_precondition_test(precondition_negative_degree
                              "degree must be non-negative")
  message(STATUS "pantr: precondition tests enabled (build type ${CMAKE_BUILD_TYPE})")
else()
  message(STATUS "pantr: precondition tests skipped, assertions are compiled out here")
endif()
