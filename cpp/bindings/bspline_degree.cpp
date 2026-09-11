/// \file
/// nanobind bindings for changing a `pantr::bspline::Bspline`'s degree.
///
/// Two entry points, `differentiate_bspline` and `elevate_bspline_degree`, each
/// registered twice -- once per storage format -- exactly as
/// `bspline_refinement.cpp` registers `insert_bspline_knots` and
/// `subdivide_bspline`. Overload resolution separates the two instantiations on
/// the field handle's own class, so neither needs a dtype argument and neither
/// can be reached with a mismatched one.
///
/// ## What crosses, and what does not
///
/// **The field itself crosses, not its arrays**, for the same reason
/// `bspline_refinement.cpp` gives: these are operations on a domain type C++ has
/// owned since the 2026-08-27 amendment to `design/cross_backend_types.md`, so
/// the binding is handed the `Bspline32`/`Bspline64` handle the Python wrapper
/// already holds and hands back a new one.
///
/// What does cross as plain data is the *arguments*: `direction` is a scalar
/// axis index, and `increments` is one degree increment per direction. Neither
/// carries an invariant to lose, and neither is a numpy array -- unlike
/// `bspline_refinement.cpp`'s knot lists, so there is no `const_vec<T>` alias
/// here and no `.noconvert()` to reason about: nanobind's `std::vector<T>`
/// caster converts a Python sequence of integers regardless, and there is no
/// narrowing it could silently perform on an integer count.
///
/// ## What this file validates: nothing
///
/// `pantr/bspline/degree.hpp` validates its own arguments and throws
/// `std::invalid_argument`, which nanobind maps to `ValueError` preserving
/// `what()`. The messages are the oracle's character for character;
/// `tests/parity/test_bspline_degree.py` compares the two texts rather than
/// just the exception type.
///
/// ## The GIL is held through the computation, unlike the kernels
///
/// `pantr_cpp.cpp` records that the GIL is released around a kernel, because a
/// kernel touches no Python object. These two are not kernels and the reason is
/// specific rather than stylistic: the space handles reachable from the field
/// were created by `nanobind/stl/shared_ptr.h`, so their deleter is
/// `py_deleter`, which touches a Python refcount. Dropping the last reference
/// to one with the GIL released would be a crash rather than a slowdown, and
/// while no path here *can* drop a last reference -- the caller's field keeps
/// every handle alive for the whole call -- that is an argument about the
/// current body rather than a property of the signature.
/// `bspline_refinement.cpp` holds the GIL through its own two entry points for
/// the identical reason.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <span>
#include <vector>

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/degree.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// Differentiate a field along one direction, returning a new one.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to differentiate.
/// \param direction The direction to differentiate, in `[0, field.dim())`.
/// \return The hodograph.
template <class T>
pantr::bspline::Bspline<T> bind_differentiate(const pantr::bspline::Bspline<T>& field,
                                              std::int64_t direction) {
    return pantr::bspline::derivative<T>(field, direction);
}

/// Elevate a field's degree per direction, returning a new one.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to elevate.
/// \param increments Degrees to add per direction, in axis order.
/// \return The elevated field.
template <class T>
pantr::bspline::Bspline<T> bind_elevate_degree(const pantr::bspline::Bspline<T>& field,
                                               const std::vector<std::int64_t>& increments) {
    return pantr::bspline::elevate_degree<T>(field, std::span<const std::int64_t>(increments));
}

}  // namespace

void register_bspline_degree(nb::module_& m) {
    m.def("differentiate_bspline", &bind_differentiate<double>, nb::arg("bspline"),
          nb::arg("direction"));
    m.def("differentiate_bspline", &bind_differentiate<float>, nb::arg("bspline"),
          nb::arg("direction"));

    m.def("elevate_bspline_degree", &bind_elevate_degree<double>, nb::arg("bspline"),
          nb::arg("increments"));
    m.def("elevate_bspline_degree", &bind_elevate_degree<float>, nb::arg("bspline"),
          nb::arg("increments"));
}
