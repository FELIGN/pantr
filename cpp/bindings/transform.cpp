/// \file
/// nanobind bindings for `pantr::transform::AffineTransform`.
///
/// The second type this extension exposes, after `AABB`, and it follows that
/// file's reasoning: ownership moved to C++ under the 2026-08-27 amendment, the
/// Python class wraps one, and the binding validates nothing because the type
/// validates itself and throws `std::invalid_argument`, which nanobind surfaces
/// as the `ValueError` the oracle raises.
///
/// ## `double` only
///
/// The oracle coerces to `float64` throughout and has no `float32` path, so a
/// `float` map would be a surface with no oracle behind it, which
/// `design/backend_parity.md` Rule 8 forbids. The header stays templated on
/// `Real` and `cpp/tests/test_affine.cpp` instantiates both.
///
/// ## No `__eq__`, deliberately
///
/// `pantr.transform.AffineTransform` has neither `__eq__` nor `__hash__`: two
/// maps built from the same numbers are distinct objects and compare by
/// identity. The C++ type does define `operator==`, because its own test needs
/// to compare maps, but binding it would give the Python class an equality it
/// has never had and would change behaviour that callers may rely on. A port
/// reproduces its oracle, including the operations it does not offer.
///
/// ## `center` stays on the Python side
///
/// The oracle's five re-centred factories take an optional `center` and
/// conjugate by a translation. In C++ that is one method, `about_center`,
/// applied after the fact. The wrapper reassembles the oracle's signatures from
/// it, exactly as `AABB`'s `union` sits over the C++ `merge`.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <span>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/transform/affine.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using Affine = pantr::transform::AffineTransform<double>;

/// A read-only, contiguous 1D `float64` array as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2D `float64` array as nanobind sees it.
template <class T>
using const_mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A writeable, contiguous 2D `float64` array as nanobind sees it.
template <class T>
using out_mat = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// View a 1D nanobind array as a span.
///
/// \param a The array.
/// \return A span over its storage.
std::span<const double> as_span(const_vec<double> a) {
    return {a.data(), a.shape(0)};
}

/// View a 2D nanobind array as a row-major 2D span.
///
/// \param a The array.
/// \return A view over its storage.
pantr::span2d<const double> as_span2d(const_mat<double> a) {
    return pantr::span2d<const double>(a.data(), a.shape(0), a.shape(1));
}

/// Copy a contiguous block into a freshly allocated, read-only numpy array.
///
/// Copied rather than viewed, for the same reason as `AABB`'s corners: a view
/// could outlive the map, and the map is immutable so that nobody has to reason
/// about when its storage changes.
///
/// \param data The block to copy.
/// \param rows Row count; zero means a 1D result.
/// \param cols Column count, or the length when `rows` is zero.
/// \return An owning `float64` array, flagged read-only.
nb::object to_numpy(const double* data, std::size_t rows, std::size_t cols) {
    const std::size_t total = (rows == 0 ? cols : rows * cols);
    auto* owned = new std::vector<double>(data, data + total);
    nb::capsule owner(owned,
                      [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    if (rows == 0) {
        return nb::cast(nb::ndarray<nb::numpy, const double, nb::ndim<1>>(owned->data(),
                                                                          {cols}, owner));
    }
    return nb::cast(
        nb::ndarray<nb::numpy, const double, nb::ndim<2>>(owned->data(), {rows, cols}, owner));
}

}  // namespace

void register_transform(nb::module_& m) {
    nb::class_<Affine>(m, "AffineTransform")
        .def(
            "__init__",
            [](Affine* self, const_mat<double> matrix, const_vec<double> offset) {
                new (self) Affine(as_span2d(matrix), as_span(offset));
            },
            nb::arg("matrix"), nb::arg("offset"))
        .def_static("identity", &Affine::identity, nb::arg("n"))
        .def_static(
            "translation",
            [](const_vec<double> offset) { return Affine::translation(as_span(offset)); },
            nb::arg("offset"))
        .def_static(
            "scaling",
            [](const_vec<double> factors) { return Affine::scaling(as_span(factors)); },
            nb::arg("factors"))
        .def_static("rotation_2d", &Affine::rotation_2d, nb::arg("angle"))
        .def_static(
            "rotation_3d",
            [](double angle, const_vec<double> axis) {
                return Affine::rotation_3d(angle, as_span(axis));
            },
            nb::arg("angle"), nb::arg("axis"))
        .def_static(
            "mirror",
            [](const_vec<double> normal) { return Affine::mirror(as_span(normal)); },
            nb::arg("normal"))
        .def_static("shear", &Affine::shear, nb::arg("n"), nb::arg("component"),
                    nb::arg("direction"), nb::arg("factor"))
        .def_prop_ro("dim", &Affine::dim)
        .def_prop_ro("matrix",
                     [](const Affine& t) {
                         return to_numpy(t.matrix().data_handle(), t.dim(), t.dim());
                     })
        .def_prop_ro("offset",
                     [](const Affine& t) { return to_numpy(t.offset().data(), 0, t.dim()); })
        .def("inverse", &Affine::inverse)
        .def("compose", &Affine::compose, nb::arg("other"))
        .def(
            "about_center",
            [](const Affine& t, const_vec<double> center) {
                return t.about_center(as_span(center));
            },
            nb::arg("center"))
        .def(
            "apply",
            [](const Affine& t, const_mat<double> points, out_mat<double> out) {
                t.apply(as_span2d(points),
                        pantr::span2d<double>(out.data(), out.shape(0), out.shape(1)));
            },
            nb::arg("points"), nb::arg("out").noconvert());
}
