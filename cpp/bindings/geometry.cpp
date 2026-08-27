/// \file
/// nanobind bindings for `pantr::geometry::AABB`.
///
/// ## The first binding that exposes a type rather than kernels
///
/// Every sibling file here binds free functions over arrays, because
/// `design/cross_backend_types.md` used to forbid a C++ counterpart of any pantr
/// class. Under the 2026-08-27 amendment the domain types are owned by C++ and
/// Python wraps them, so this file exposes `AABB` itself and the Python class
/// holds a handle to it.
///
/// **Ownership moves; it is not duplicated.** There is exactly one implementation
/// of a box, and it is the one in `pantr/geometry/aabb.hpp`. The Python class is
/// a wrapper, not a second box, which is what keeps the failure the superseded
/// note was written to prevent from coming back.
///
/// ## `double` only
///
/// The oracle, `pantr.geometry.AABB`, coerces its corners to `float64` and has no
/// `float32` path at all. Registering a `float` box would create a surface with no
/// oracle behind it, which `design/backend_parity.md` Rule 8 forbids. The header
/// stays templated on `Real`, as the core's convention requires and as
/// `test_aabb.cpp` exercises for `float`; only the binding is narrow.
///
/// ## What this file validates, and what it does not
///
/// Nothing. That is the change worth noticing: every sibling binding re-checks
/// its arguments because the kernel below it asserts rather than throws. `AABB`
/// validates its own arguments and throws `std::invalid_argument`, which nanobind
/// surfaces as `ValueError` -- the exception the oracle raises for the same
/// inputs. Duplicating the checks here would put a second copy of the contract in
/// a second language, which is the shape this port exists to avoid.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/geometry/aabb.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using Box = pantr::geometry::AABB<double>;

/// A read-only, contiguous 1D `float64` array as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2D `float64` array as nanobind sees it.
template <class T>
using const_mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// View a 1D nanobind array as a span.
///
/// \param a The array.
/// \return A span over its storage, valid while the array is alive.
std::span<const double> as_span(const_vec<double> a) {
    return {a.data(), a.shape(0)};
}

/// Copy a corner into a freshly allocated, read-only numpy array.
///
/// The copy is deliberate: handing out a view of the box's own `std::vector`
/// would let the array outlive the box, and the box is immutable precisely so
/// that nobody has to reason about when its storage changes.
///
/// \param corner The corner to expose.
/// \return An owning `float64` array of shape `(ndim,)`, flagged read-only.
nb::object corner_to_numpy(std::span<const double> corner) {
    auto* data = new std::vector<double>(corner.begin(), corner.end());
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t n = data->size();
    auto arr = nb::ndarray<nb::numpy, const double, nb::ndim<1>>(data->data(), {n}, owner);
    return nb::cast(arr, nb::rv_policy::reference_internal);
}

}  // namespace

void register_geometry(nb::module_& m) {
    nb::class_<Box>(m, "AABB")
        .def(
            "__init__",
            [](Box* self, const_vec<double> lo, const_vec<double> hi) {
                new (self) Box(as_span(lo), as_span(hi));
            },
            nb::arg("lo"), nb::arg("hi"))
        .def_static("unbounded", &Box::unbounded, nb::arg("ndim"))
        .def_static("empty", &Box::empty, nb::arg("ndim"))
        .def_prop_ro("ndim", &Box::ndim)
        .def_prop_ro("lo", [](const Box& b) { return corner_to_numpy(b.lo()); })
        .def_prop_ro("hi", [](const Box& b) { return corner_to_numpy(b.hi()); })
        .def("is_empty", &Box::is_empty)
        .def(
            "contains_point",
            [](const Box& b, const_vec<double> x) { return b.contains_point(as_span(x)); },
            nb::arg("x"))
        .def("overlaps", &Box::overlaps, nb::arg("other"))
        // Named `union` on the Python side, where it is not a keyword. The C++
        // name is `merge` for exactly that reason, and the two are one function.
        .def("union", &Box::merge, nb::arg("other"))
        .def("intersect", &Box::intersect, nb::arg("other"))
        .def(
            "pad",
            [](const Box& b, nb::object r) {
                if (nb::isinstance<const_vec<double>>(r)) {
                    return b.pad(as_span(nb::cast<const_vec<double>>(r)));
                }
                return b.pad(nb::cast<double>(r));
            },
            nb::arg("r"))
        .def(
            "transform",
            [](const Box& b, const_mat<double> matrix, const_vec<double> offset) {
                const auto view = pantr::span2d<const double>(matrix.data(), matrix.shape(0),
                                                              matrix.shape(1));
                return b.transform(view, as_span(offset));
            },
            nb::arg("matrix"), nb::arg("offset"))
        .def_static(
            "from_bounds",
            [](const_mat<double> bounds) {
                return Box::from_bounds(pantr::span2d<const double>(bounds.data(),
                                                                    bounds.shape(0),
                                                                    bounds.shape(1)));
            },
            nb::arg("bounds"))
        .def("as_bounds",
             [](const Box& b) {
                 auto* data = new std::vector<double>(b.ndim() * 2);
                 b.as_bounds(pantr::span2d<double>(data->data(), b.ndim(), 2));
                 nb::capsule owner(data, [](void* p) noexcept {
                     delete static_cast<std::vector<double>*>(p);
                 });
                 // Writeable and freshly allocated, matching the oracle, whose
                 // docstring promises exactly that.
                 return nb::ndarray<nb::numpy, double, nb::ndim<2>>(data->data(),
                                                                    {b.ndim(), std::size_t{2}},
                                                                    owner);
             })
        .def("__repr__", &Box::to_string)
        .def("__eq__", [](const Box& a, const Box& b) { return a == b; }, nb::is_operator())
        .def("__hash__",
             [](const Box& b) { return static_cast<Py_ssize_t>(std::hash<Box>{}(b)); });
}
