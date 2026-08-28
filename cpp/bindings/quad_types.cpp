/// \file
/// nanobind bindings for `pantr::quad::QuadratureRule` -- reserved, empty.
///
/// The module's three assembly points (`pantr_cpp.cpp`, `register.hpp` and
/// `CMakeLists.txt`) are edited once, here, so the ticket that ports the type
/// edits this file and nothing else. Thirteen tickets in this milestone would
/// otherwise collide on those three.

#include <nanobind/nanobind.h>

#include "register.hpp"

void register_quad_types(nanobind::module_&) {
    // Nothing yet: the port adds the `nanobind::class_` for the type here.
}
