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
    test_binomial.cpp)

foreach(_src IN LISTS PANTR_TEST_SOURCES)
  cmake_path(GET _src STEM _name)
  add_executable(${_name} ${_src})
  target_link_libraries(${_name} PRIVATE pantr::core)
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${_name} COMMAND ${_name})
endforeach()
