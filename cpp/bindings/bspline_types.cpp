/// \file
/// nanobind bindings for `pantr::bspline::BsplineSpace1D`.
///
/// ## Two registrations, because the storage format is part of the value
///
/// `pantr.bspline.BsplineSpace1D` stores whatever float dtype it is handed, its
/// `dtype` property is public, and `float32` knot vectors are exercised across the
/// suite. So the same choice `cpp/bindings/bezier_type.cpp` made applies here:
/// `BsplineSpace1D32` and `BsplineSpace1D64` are registered separately and the
/// class of the handle *is* the dtype, because a single name would have nothing
/// left to carry it. `pantr.bspline._bspline_space_1d._impl_class` picks between
/// them per instance.
///
/// The dtype is not converted, and that *is* a check. Without `.noconvert()`
/// nanobind silently casts, so `BsplineSpace1D32(a_float64_vector)` would narrow a
/// caller's knots and `BsplineSpace1D64(a_float32_vector)` would widen them --
/// changing the space's tolerance by four orders in the second case, since the
/// tolerance is `8 * eps(T) * scale`. Both would happen without a word.
///
/// ## What this file validates: nothing
///
/// The type validates its own arguments and throws `std::invalid_argument`, which
/// nanobind maps to `ValueError` preserving `what()` -- and the messages are the
/// oracle's character for character, which is what
/// `tests/parity/test_bspline_space_1d.py` compares. Two things the wrapper still
/// owes are Python's calling convention rather than validation: coercing an
/// `ArrayLike` to a contiguous array of a supported dtype, and the `TypeError` for
/// a knot vector that is not one, for which `pantr/core/error.hpp` records there is
/// no `std::exception` nanobind turns into one.
///
/// ## Arrays out: views, never copies
///
/// Every array here is storage the space owns and computed once -- the knot vector
/// at construction, the derived knot classes behind the memo in
/// `pantr/core/lazy.hpp`. They go out as `nb::ndarray` views with the space as the
/// array's owner, which is `bezier_type.cpp`'s idiom and for its reasons: the owner
/// is what keeps the storage alive when the array outlives the handle it came from,
/// and `const T` is what nanobind turns into the read-only flag, which is what stops
/// a caller mutating a validated space's knots from outside.
///
/// Copying instead would be worse than wasteful. The whole point of the derived
/// block is that a loop over intervals reads it rather than recomputing it, and a
/// property that copied would make the natural spelling of every such loop
/// quadratic in nothing.
///
/// **None of these arrays can be empty**, so the null-`data()` trap that
/// `grid_types.cpp` guards against with a stand-in address does not arise here: a
/// knot vector has at least `2 * degree + 2` entries, a vector has at least one
/// knot class, the constructor refuses a space with no interval so the in-domain
/// range has at least two classes, and `first_basis_per_interval` has exactly
/// `num_intervals` entries -- checked over 2850 accepted cases and asserted in
/// `cpp/tests/test_bspline_space_1d.cpp` so it stays true.
///
/// ## `_ref` accessors are not bound, and there are none to bind
///
/// `design/bspline_ownership_lifetime.md` requires that no borrowing accessor
/// reaches Python. `BsplineSpace1D` grew none -- it hands out spans of its own
/// storage rather than references to nested objects -- so the rule is satisfied
/// vacuously here. `tests/parity/test_bspline_binding_contract.py` asserts it over
/// the bound surface anyway, because the next type in this front will have one.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "pantr/bspline/space_1d.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A read-only, contiguous 1D array of the given type as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Hand out a span of the owner's storage as a read-only numpy view.
///
/// \param self The Python object that owns the storage, kept alive by the array.
/// \param data The span; must not be empty, see the file comment.
/// \return A read-only 1D `numpy` array viewing `data`.
template <class T>
nb::object view_of(nb::handle self, std::span<const T> data) {
    return nb::cast(nb::ndarray<nb::numpy, const T, nb::ndim<1>>(data.data(), {data.size()}, self));
}

/// Register one scalar type's `BsplineSpace1D`.
///
/// \param m The extension module.
/// \param name The Python-visible class name, `BsplineSpace1D32` or `BsplineSpace1D64`.
template <class T>
void bind_space_1d(nb::module_& m, const char* name) {
    using Space = pantr::bspline::BsplineSpace1D<T>;
    using pantr::bspline::KnotSnapping;

    nb::class_<Space>(m, name)
        .def(
            "__init__",
            [](Space* self, const_vec<T> knots, std::int64_t degree, bool periodic,
               bool snap_knots) {
                new (self) Space(std::span<const T>(knots.data(), knots.size()), degree, periodic,
                                 snap_knots ? KnotSnapping::merge_near_duplicates
                                            : KnotSnapping::as_given);
            },
            nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("periodic") = false,
            nb::arg("snap_knots") = true)
        .def_prop_ro("degree", [](const Space& s) { return s.degree(); })
        .def_prop_ro("periodic", &Space::periodic)
        .def_prop_ro("tolerance", &Space::tolerance)
        .def_prop_ro("num_basis", [](const Space& s) { return s.num_basis(); })
        .def_prop_ro("num_intervals", [](const Space& s) { return s.num_intervals(); })
        .def_prop_ro("knots",
                     [](nb::handle self) {
                         return view_of<T>(self, nb::cast<const Space&>(self).knots());
                     })
        // A tuple of two scalars, because the oracle's `domain` is a tuple and a
        // caller unpacking `lo, hi = space.domain` must keep working. The wrapper
        // is what turns them into numpy scalars of the space's own dtype, since
        // that is a presentation decision rather than a property of the value.
        .def_prop_ro("domain",
                     [](const Space& s) {
                         const auto ends = s.domain();
                         return nb::make_tuple(ends[0], ends[1]);
                     })
        .def(
            "get_unique_knots_and_multiplicity",
            [](nb::handle self, bool in_domain) {
                const Space& s = nb::cast<const Space&>(self);
                if (in_domain) {
                    return nb::make_tuple(view_of<T>(self, s.unique_knots_in_domain()),
                                          view_of<std::int64_t>(self, s.multiplicity_in_domain()));
                }
                return nb::make_tuple(view_of<T>(self, s.unique_knots()),
                                      view_of<std::int64_t>(self, s.multiplicity()));
            },
            nb::arg("in_domain") = false)
        .def("first_basis_per_interval",
             [](nb::handle self) {
                 return view_of<std::int64_t>(self,
                                              nb::cast<const Space&>(self).first_basis_per_interval());
             })
        .def("has_left_end_open", &Space::has_left_end_open)
        .def("has_right_end_open", &Space::has_right_end_open)
        .def("has_open_knots", &Space::has_open_knots)
        // Spelled the oracle's way. The C++ name is `has_bezier_like_knots`,
        // because a capital inside an identifier is not this codebase's C++ style;
        // the Python name is what a caller has always written.
        .def("has_Bezier_like_knots", &Space::has_bezier_like_knots);
}

}  // namespace

void register_bspline_types(nb::module_& m) {
    bind_space_1d<float>(m, "BsplineSpace1D32");
    bind_space_1d<double>(m, "BsplineSpace1D64");
}
