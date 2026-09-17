/// \file
/// nanobind bindings for the shape operations on a `pantr::bspline::Bspline`.
///
/// Three entry points -- `reverse_bspline`, `permute_bspline_directions` and
/// `transform_bspline` -- each registered twice, once per storage format, exactly as
/// `bspline_structural.cpp` and `bezier.cpp` register theirs. Overload resolution
/// separates the two instantiations on the field handle's own class, so none needs a
/// dtype argument and none can be reached with a mismatched one. The names are the
/// Bézier ones with the type swapped: `reverse_bezier`, `permute_bezier_directions`
/// and `transform_bezier` are already bound under exactly this shape.
///
/// ## What crosses, and what does not
///
/// **The field itself crosses, not its arrays**, for `bspline_structural.cpp`'s
/// reason: these are operations on a domain type C++ owns since the 2026-08-27
/// amendment to `design/cross_backend_types.md`, and unpacking a field into a knot
/// vector and a control net on one side and reassembling it on the other is the shape
/// that amendment forbids.
///
/// **`transform_bspline` takes a matrix and an offset, never an `AffineTransform`.**
/// `pantr.bspline.Bspline.transform` reads `affine.matrix` and `affine.offset`, which
/// are numpy arrays on either backend, so no affine implementation is ever converted
/// into the other -- the shape `design/cross_backend_types.md` forbids for the same
/// reason. `bezier.cpp`'s `transform_bezier` crosses this way already. Both arrive as
/// `double` whatever the field stores, because that is what an `AffineTransform`
/// holds; `pantr/bspline/shape.hpp` casts them to the storage format before
/// multiplying, which is the oracle's `matrix.astype(dtype)`.
///
/// The permutation crosses as a list of integers rather than as an array, which is
/// what `permute_bezier_directions` does and what the oracle's own signature takes.
///
/// ## What this file validates: nothing
///
/// `pantr/bspline/shape.hpp` validates its own arguments and throws
/// `std::invalid_argument`, which nanobind maps to `ValueError` preserving `what()`.
/// The messages are the oracle's character for character;
/// `tests/parity/test_bspline_shape.py` compares the two texts rather than just the
/// exception type.
///
/// Every one of those refusals is also unreachable from Python, because
/// `pantr.bspline.Bspline`'s three methods validate in Layer 1 before dispatching --
/// which is what makes the texts agree on both backends rather than merely on one.
/// They are still transcribed, because a caller with no Python reads them and
/// `cpp/tests/test_bspline_shape.cpp` is what pins them there.
///
/// ## The GIL is held through the computation
///
/// For `bspline_structural.cpp`'s reason, unchanged: the space handles reachable from
/// the field were created by `nanobind/stl/shared_ptr.h`, so their deleter is
/// `py_deleter`, which touches a Python refcount. Dropping the last reference to one
/// with the GIL released would be a crash rather than a slowdown. That reason bites
/// here more than there, since all three of these operations move space handles
/// around by construction.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <span>
#include <vector>

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/shape.hpp"
#include "pantr/core/mdspan.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A read-only, contiguous 1-D array as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2-D array as nanobind sees it.
template <class T>
using const_mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// Reverse one parametric direction, reflecting its knot vector.
///
/// \tparam T The scalar type the field stores.
/// \param bspline The field to reverse.
/// \param direction The direction to reverse.
/// \return The reversed field.
template <class T>
pantr::bspline::Bspline<T> bind_reverse(const pantr::bspline::Bspline<T>& bspline,
                                        std::int64_t direction) {
    return pantr::bspline::reverse<T>(bspline, direction);
}

/// Reorder the parametric directions.
///
/// \tparam T The scalar type the field stores.
/// \param bspline The field to permute.
/// \param permutation A permutation of `range(dim)`.
/// \return The permuted field.
template <class T>
pantr::bspline::Bspline<T> bind_permute(const pantr::bspline::Bspline<T>& bspline,
                                        const std::vector<std::int64_t>& permutation) {
    return pantr::bspline::permute_directions<T>(
        bspline, std::span<const std::int64_t>(permutation));
}

/// Apply an affine map to the geometric coordinates.
///
/// See the file comment for why the map arrives as a matrix and an offset rather than
/// as an `AffineTransform`.
///
/// \tparam T The scalar type the field stores.
/// \param bspline The field to transform.
/// \param matrix The linear part, `(n, n)` and always `double`.
/// \param offset The translation, `n` values and always `double`.
/// \return The transformed field, over this field's own space handle.
template <class T>
pantr::bspline::Bspline<T> bind_transform(const pantr::bspline::Bspline<T>& bspline,
                                          const_mat<double> matrix, const_vec<double> offset) {
    const pantr::span2d<const double> linear(matrix.data(), matrix.shape(0), matrix.shape(1));
    return pantr::bspline::transform<T>(
        bspline, linear, std::span<const double>(offset.data(), offset.shape(0)));
}

}  // namespace

void register_bspline_shape(nb::module_& m) {
    m.def("reverse_bspline", &bind_reverse<double>, nb::arg("bspline"), nb::arg("direction"));
    m.def("reverse_bspline", &bind_reverse<float>, nb::arg("bspline"), nb::arg("direction"));

    m.def("permute_bspline_directions", &bind_permute<double>, nb::arg("bspline"),
          nb::arg("permutation"));
    m.def("permute_bspline_directions", &bind_permute<float>, nb::arg("bspline"),
          nb::arg("permutation"));

    m.def("transform_bspline", &bind_transform<double>, nb::arg("bspline"), nb::arg("matrix"),
          nb::arg("offset"));
    m.def("transform_bspline", &bind_transform<float>, nb::arg("bspline"), nb::arg("matrix"),
          nb::arg("offset"));
}
