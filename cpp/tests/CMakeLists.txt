# C++ tests. Plain executables registered with ctest -- see cpp/tests/check.hpp
# for why there is no test framework here.

set(PANTR_TEST_SOURCES
    test_dependency_smoke.cpp
    test_cardinal_bspline.cpp
    test_scalar_generic.cpp
    test_quad.cpp)

foreach(_src IN LISTS PANTR_TEST_SOURCES)
  cmake_path(GET _src STEM _name)
  add_executable(${_name} ${_src})
  target_link_libraries(${_name} PRIVATE pantr::core)
  target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${_name} COMMAND ${_name})
endforeach()

# Eigen is only reached by the dependency smoke test. Linking it to pantr::core
# would put it on the include path of the kernel and the bindings too, which
# would let a stray #include compile without anyone deciding to take the
# dependency on.
target_link_libraries(test_dependency_smoke PRIVATE Eigen3::Eigen)
