/// \file
/// nanobind bindings for refining a `pantr::bspline::Bspline`.
///
/// Two entry points, `insert_bspline_knots` and `subdivide_bspline`, each registered
/// twice -- once per storage format -- exactly as `bezier.cpp` registers its
/// n-dimensional degree and shape operations. Overload resolution separates the two
/// instantiations on the field handle's own class, so neither needs a dtype argument
/// and neither can be reached with a mismatched one.
///
/// ## What crosses, and what does not
///
/// **The field itself crosses, not its arrays.** These are operations on a domain type
/// that C++ owns since the 2026-08-27 amendment to `design/cross_backend_types.md`, so
/// the binding is handed the `Bspline32`/`Bspline64` handle the Python wrapper already
/// holds and hands back a new one. Unpacking a field into a knot vector and a control
/// net on one side and reassembling it on the other is the shape that amendment
/// forbids, and the rationality flag is exactly the invariant a reassembly drops.
/// `pantr.bezier._bezier_backend` says the same of its own n-dimensional entry points.
///
/// What does cross as arrays is the *arguments*: one 1-D knot array per direction, and
/// the per-direction subdivision counts. Those are plain data with no invariant to
/// lose.
///
/// The dtype is not converted, and that *is* a check: without `.noconvert()` nanobind
/// silently casts, so `insert_bspline_knots(a_float32_field, a_float64_list)` would
/// narrow the caller's knots without a word and the refined vector would not be the
/// one the caller asked for. Refusing the cast makes a wrapper that forgot to cast a
/// loud failure instead of a quiet change of precision.
///
/// An **empty array skips its direction**, which is how the oracle reads `None` or an
/// empty array in the same position. `pantr.bspline.Bspline.insert_knots` normalises
/// `None` to an empty array before it gets here, so there is no `std::optional` in
/// these signatures and no second spelling of "skip". `subdivide_bspline` reads a count
/// of 1 the same way, which is again the oracle's reading of `None` there: every branch
/// in `Bspline.subdivide` treats `None` and `1` identically.
///
/// ## What this file validates: nothing
///
/// `pantr/bspline/refinement.hpp` validates its own arguments and throws
/// `std::invalid_argument`, which nanobind maps to `ValueError` preserving `what()`.
/// The messages are the oracle's character for character;
/// `tests/parity/test_bspline_refinement.py` compares the two texts rather than just
/// the exception type.
///
/// One refusal of the oracle's cannot be made on either side of this seam and stays the
/// Python wrapper's: `f"new_knots must be a 1D array-like, got shape ..."` quotes a
/// numpy shape, and `nb::ndim<1>` would refuse a rank-2 array with nanobind's own
/// message about C++ types instead. `pantr.bspline._knot_insertion_backend` raises it,
/// which also keeps its text common mode between the backends.
///
/// ## The GIL is held through the computation, unlike the kernels
///
/// `pantr_cpp.cpp` records that the GIL is released around a kernel, because a kernel
/// touches no Python object. These two are not kernels and the reason is specific
/// rather than stylistic: the space handles reachable from the field were created by
/// `nanobind/stl/shared_ptr.h`, so their deleter is `py_deleter`, which touches a
/// Python refcount. Dropping the last reference to one with the GIL released would be
/// a crash rather than a slowdown, and while no path here *can* drop a last reference
/// -- the caller's field keeps every handle alive for the whole call -- that is an
/// argument about the current body rather than a property of the signature.
///
/// `bezier.cpp`'s returning entry points (`elevate_bezier_degree`, `restrict_bezier`,
/// `split_bezier`) hold the GIL for the same shape of reason. If refinement ever shows
/// up in a profile as a serialisation point, the fix is to release it around
/// `refine_along_axis` alone, which touches nothing but `T`.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/refinement.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A read-only, contiguous 1-D array as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Reinterpret a bound list of 1-D arrays as the spans the header takes.
///
/// \tparam T The scalar type the arrays hold.
/// \param arrays The arrays, borrowed for the duration of the call.
/// \return One span per entry, in the same order. An empty array yields an empty span,
///         which is how the header spells "skip this direction".
template <class T>
std::vector<std::span<const T>> as_spans(const std::vector<const_vec<T>>& arrays) {
    std::vector<std::span<const T>> spans;
    spans.reserve(arrays.size());
    for (const const_vec<T>& array : arrays) {
        spans.emplace_back(array.data(), array.shape(0));
    }
    return spans;
}

/// Insert knots into a field, returning a new one.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to refine.
/// \param new_knots One array of knot values per direction, in axis order.
/// \return The refined field.
template <class T>
pantr::bspline::Bspline<T> bind_insert_knots(const pantr::bspline::Bspline<T>& field,
                                             const std::vector<const_vec<T>>& new_knots) {
    const std::vector<std::span<const T>> lists = as_spans<T>(new_knots);
    return pantr::bspline::insert_knots<T>(field,
                                           std::span<const std::span<const T>>(lists));
}

/// Subdivide a field's knot spans uniformly, returning a new one.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to refine.
/// \param n_subdivisions Equal sub-spans per existing span, one per direction; 1 skips.
/// \param regularity The continuity at each inserted knot, or `None` for
///        `degree - 1` per direction.
/// \return The refined field.
template <class T>
pantr::bspline::Bspline<T> bind_subdivide(const pantr::bspline::Bspline<T>& field,
                                          const std::vector<std::int64_t>& n_subdivisions,
                                          std::optional<std::int64_t> regularity) {
    return pantr::bspline::subdivide<T>(
        field, std::span<const std::int64_t>(n_subdivisions), regularity);
}

}  // namespace

void register_bspline_refinement(nb::module_& m) {
    m.def("insert_bspline_knots", &bind_insert_knots<double>, nb::arg("bspline"),
          nb::arg("new_knots").noconvert());
    m.def("insert_bspline_knots", &bind_insert_knots<float>, nb::arg("bspline"),
          nb::arg("new_knots").noconvert());

    m.def("subdivide_bspline", &bind_subdivide<double>, nb::arg("bspline"),
          nb::arg("n_subdivisions"), nb::arg("regularity"));
    m.def("subdivide_bspline", &bind_subdivide<float>, nb::arg("bspline"),
          nb::arg("n_subdivisions"), nb::arg("regularity"));
}
