/// \file
/// nanobind bindings for the structural operations on a `pantr::bspline::Bspline`.
///
/// Four entry points -- `open_bspline`, `split_bspline`, `slice_bspline` and
/// `slice_bspline_point` -- each registered twice, once per storage format, exactly as
/// `bspline_refinement.cpp` and `bezier.cpp` register theirs. Overload resolution
/// separates the two instantiations on the field handle's own class, so none needs a
/// dtype argument and none can be reached with a mismatched one.
///
/// ## What crosses, and what does not
///
/// **The field itself crosses, not its arrays**, for the reason
/// `bspline_refinement.cpp` states in full: these are operations on a domain type C++
/// owns since the 2026-08-27 amendment to `design/cross_backend_types.md`, and
/// unpacking a field into a knot vector and a control net on one side and reassembling
/// it on the other is the shape that amendment forbids. The arguments that cross are
/// plain scalars -- an axis, a side, a parameter -- with no invariant to lose.
///
/// `split_bspline` returns a **pair**, which `nanobind/stl/pair.h` converts to a
/// 2-tuple; `split_bezier` already crosses that way and this is the same shape.
///
/// ## Why `slice_bspline_point` fills a buffer while the others return a value
///
/// Because its result is a plain array of known length rather than a field, which is
/// the same line `bezier.cpp` draws between `slice_bezier` and `slice_bezier_point`.
/// The length is the net's component count, which the caller already knows, so letting
/// Python own the allocation keeps the dtype and the read-only policy where the rest of
/// `pantr`'s `out=` convention keeps them, and this file allocates nothing.
///
/// **The weight column is still on.** `pantr::bspline::slice_point` returns homogeneous
/// components and `pantr.bspline._structural_backend` divides, because the oracle's
/// division is a numpy expression and reproducing it here would be a second spelling of
/// it. `pantr/bspline/structural.hpp` records the same for its own reason.
///
/// ## `boundary` is deliberately not bound
///
/// `pantr::bspline::boundary` exists for a C++ caller, and
/// `pantr.bspline.Bspline.boundary` is defined as a `slice` at a domain endpoint -- so
/// it reaches C++ through `slice_bspline` and a binding of its own would be public
/// surface with no caller. `pantr._pantr_cpp.bezier_boundary` is exactly that today,
/// bound and stubbed and reached by nothing in `src/` or `tests/`; one such precedent
/// is enough.
///
/// ## What this file validates: nothing
///
/// `pantr/bspline/structural.hpp` validates its own arguments and throws
/// `std::invalid_argument`, which nanobind maps to `ValueError` preserving `what()`.
/// The messages are the oracle's character for character;
/// `tests/parity/test_bspline_structural.py` compares the two texts rather than just
/// the exception type.
///
/// Two refusals cannot be reached from Python and stay documented rather than tested
/// against the oracle: `slice_bspline`'s "dimension at least two", which
/// `pantr.bspline.Bspline.slice` routes past by calling `slice_bspline_point` instead,
/// and `slice_bspline`'s out-of-domain message at `float32`, whose oracle interpolates
/// a numpy scalar rather than a Python float. The header's file comment carries that
/// second one in full.
///
/// ## The GIL is held through the computation
///
/// For `bspline_refinement.cpp`'s reason, unchanged: the space handles reachable from
/// the field were created by `nanobind/stl/shared_ptr.h`, so their deleter is
/// `py_deleter`, which touches a Python refcount. Dropping the last reference to one
/// with the GIL released would be a crash rather than a slowdown.
///
/// `slice_bspline_point` is the one entry point whose whole body could run without it
/// -- it reads `T` and writes `T` -- but it is also the cheapest of the four, so
/// releasing the GIL around it would buy a handful of nanoseconds and add a rule a
/// reader has to check. If a slice sweep ever shows up in a profile, the release
/// belongs around `corner_cut_along_axis` alone.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/structural.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A writeable, contiguous 1-D array as nanobind sees it.
template <class T>
using out_vec = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// The field re-expressed over clamped, non-periodic knot vectors.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to convert.
/// \return The open field.
template <class T>
pantr::bspline::Bspline<T> bind_to_open(const pantr::bspline::Bspline<T>& field) {
    return pantr::bspline::to_open<T>(field);
}

/// The field cut in two at a parameter of one direction.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to split.
/// \param direction The direction to split.
/// \param value The parameter to split at.
/// \return The left and right halves, as a 2-tuple on the Python side.
template <class T>
std::pair<pantr::bspline::Bspline<T>, pantr::bspline::Bspline<T>>
bind_split(const pantr::bspline::Bspline<T>& field, std::int64_t direction, double value) {
    return pantr::bspline::split<T>(field, direction, value);
}

/// The field with one direction fixed at a value, so one dimension fewer.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to slice, of dimension at least two.
/// \param axis The direction to fix.
/// \param value The parameter to fix it at.
/// \return The sliced field.
template <class T>
pantr::bspline::Bspline<T> bind_slice(const pantr::bspline::Bspline<T>& field,
                                      std::int64_t axis, double value) {
    return pantr::bspline::slice<T>(field, axis, value);
}

/// The point a one-dimensional field takes at a parameter, in homogeneous components.
///
/// \tparam T The scalar type the field stores.
/// \param field A one-dimensional field.
/// \param value The parameter.
/// \param out The destination, of length `field.net().num_components()`.
/// \throws nanobind::value_error If `out` is not that long.
template <class T>
void bind_slice_point(const pantr::bspline::Bspline<T>& field, double value, out_vec<T> out) {
    const std::vector<T> point = pantr::bspline::slice_point<T>(field, value);
    if (out.shape(0) != point.size()) {
        throw nb::value_error(("out has length " + std::to_string(out.shape(0))
                               + ", but this call needs " + std::to_string(point.size()))
                                  .c_str());
    }
    for (std::size_t i = 0; i < point.size(); ++i) {
        out(i) = point[i];
    }
}

}  // namespace

void register_bspline_structural(nb::module_& m) {
    m.def("open_bspline", &bind_to_open<double>, nb::arg("bspline"));
    m.def("open_bspline", &bind_to_open<float>, nb::arg("bspline"));

    m.def("split_bspline", &bind_split<double>, nb::arg("bspline"), nb::arg("direction"),
          nb::arg("value"));
    m.def("split_bspline", &bind_split<float>, nb::arg("bspline"), nb::arg("direction"),
          nb::arg("value"));

    m.def("slice_bspline", &bind_slice<double>, nb::arg("bspline"), nb::arg("axis"),
          nb::arg("value"));
    m.def("slice_bspline", &bind_slice<float>, nb::arg("bspline"), nb::arg("axis"),
          nb::arg("value"));

    m.def("slice_bspline_point", &bind_slice_point<double>, nb::arg("bspline"),
          nb::arg("value"), nb::kw_only(), nb::arg("out").noconvert());
    m.def("slice_bspline_point", &bind_slice_point<float>, nb::arg("bspline"), nb::arg("value"),
          nb::kw_only(), nb::arg("out").noconvert());
}
