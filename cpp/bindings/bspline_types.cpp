/// \file
/// nanobind bindings for `pantr::bspline::BsplineSpace1D` and the tensor-product
/// `pantr::bspline::BsplineSpace` built over it.
///
/// ## Two registrations per type, because the storage format is part of the value
///
/// `pantr.bspline.BsplineSpace1D` stores whatever float dtype it is handed, its
/// `dtype` property is public, and `float32` knot vectors are exercised across the
/// suite. So the same choice `cpp/bindings/bezier_type.cpp` made applies here:
/// `BsplineSpace1D32` and `BsplineSpace1D64` are registered separately and the
/// class of the handle *is* the dtype, because a single name would have nothing
/// left to carry it. `pantr.bspline._bspline_space_1d._impl_class` picks between
/// them per instance.
///
/// The nD type inherits that split, with a second reason on top of the first:
/// `BsplineSpace<T>` can hold only `BsplineSpace1D<T>`, so the width is already a
/// property of the C++ type and one Python name could not front both.
/// `BsplineSpace32` and `BsplineSpace64` are the two, and
/// `pantr.bspline._bspline_space_nd._impl_class` picks between them.
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
/// **No array a 1D space hands out can be empty**, and that was once the whole of
/// this paragraph: a knot vector has at least `2 * degree + 2` entries, a vector has
/// at least one knot class, the constructor refuses a space with no interval so the
/// in-domain range has at least two classes, and `first_basis_per_interval` has
/// exactly `num_intervals` entries, which `cpp/tests/test_bspline_space_1d.cpp`
/// asserts so that it stays true.
///
/// **The nD type breaks that, so the guard is here after all.** A dimensionless
/// `BsplineSpace` is legal -- the oracle admits `BsplineSpace([])`, and
/// `tests/test_bspline_space.py::test_empty_spaces_list` pins it -- and its domain
/// block has no rows. An empty `std::vector`'s `data()` may be null, and nanobind
/// reads a null pointer as "no array" rather than as an empty one, so `view_of`
/// substitutes a never-dereferenced stand-in address exactly as
/// `grid_types.cpp:96-104` does.
///
/// ## `_ref` accessors are not bound
///
/// `design/bspline_ownership_lifetime.md` requires that no borrowing accessor
/// reaches Python. `BsplineSpace1D` grew none -- it hands out spans of its own
/// storage rather than references to nested objects. `BsplineSpace` is the first
/// type in this front that *has* one: `space_ref(d)` borrows a direction, costs a
/// measured 5.83 ns against `space(d)`'s 14.92 ns, and is deliberately absent from
/// the surface below. `tests/parity/test_bspline_binding_contract.py` asserts over
/// the bound surface that no method name ends in `_ref`, which is the only available
/// check: there is no `static_assert` for absence.
///
/// ## Handing out a direction: a `shared_ptr`, and no policy
///
/// `spaces` is `design/bspline_ownership_lifetime.md`'s class **H** -- an accessor
/// handing out a subobject the owner keeps -- so the C++ type stores
/// `std::shared_ptr<const BsplineSpace1D<T>>` and the binding returns a copy of the
/// handle. No `rv_policy` question arises, because ownership travels in the return
/// value rather than in an annotation. Three consequences worth stating, all of them
/// measured in that note and each the failure of an alternative:
///
///  - The returned Python object is **identity-stable**: `nb_type_put` looks the
///    pointer up in `inst_c2p` before creating an instance, so
///    `nd.spaces[0] is one_d` holds for the handle a caller passed in. That is what
///    makes `tests/test_bspline_space.py:89` reachable, and it is why the C++
///    constructor shares rather than copies.
///  - `sys.getrefcount` on the nD handle is **unchanged** by the access, because no
///    keep-alive is installed. A non-zero delta means somebody reverted to
///    `reference_internal`, and the contract test asserts the zero.
///  - The direction **outlives its owner**. Passing a handle in takes a reference on
///    its Python object (`nanobind/stl/shared_ptr.h`'s `py_deleter`), so the nD space
///    keeps its directions alive, and a direction taken back out keeps its own value
///    alive after the nD space is dropped.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A read-only, contiguous 1D array of the given type as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A never-dereferenced address for a zero-length array.
///
/// An empty `std::vector`'s `data()` may be null, and nanobind reads a null pointer
/// as "no array" rather than as an empty one. A dimensionless nD space has an empty
/// domain block, which is legal, so a valid address stands in.
/// `grid_types.cpp:96-104` carries the same sentinel for the same reason.
///
/// \tparam T The element type.
template <class T>
T kEmptyStorage{};

/// Hand out a span of the owner's storage as a read-only numpy view.
///
/// \param self The Python object that owns the storage, kept alive by the array.
/// \param data The span; may be empty only for the nD domain block, see the file
///        comment.
/// \return A read-only 1D `numpy` array viewing `data`.
template <class T>
nb::object view_of(nb::handle self, std::span<const T> data) {
    return nb::cast(nb::ndarray<nb::numpy, const T, nb::ndim<1>>(data.data(), {data.size()}, self));
}

/// Hand out a span of the owner's storage as a read-only `(rows, cols)` numpy view.
///
/// \param self The Python object that owns the storage, kept alive by the array.
/// \param data The span, row-major; its size must be `rows * cols`. May be empty.
/// \param rows The row count.
/// \param cols The column count.
/// \return A read-only 2D `numpy` array viewing `data`.
template <class T>
nb::object view_2d_of(nb::handle self, std::span<const T> data, std::size_t rows,
                      std::size_t cols) {
    const T* base = data.empty() ? &kEmptyStorage<T> : data.data();
    return nb::cast(nb::ndarray<nb::numpy, const T, nb::ndim<2>>(base, {rows, cols}, self));
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

/// Hand out a span of counts as a tuple of Python integers.
///
/// A tuple rather than an array, because the oracle's `degrees`, `num_basis` and
/// `num_intervals` are tuples and the suite compares them with `==` against one:
/// `space.num_basis == (3, 2)` on an array is elementwise and then has no truth
/// value. A tuple rather than a list, because the oracle's are tuples and
/// `tests/test_bspline_space.py` asserts `isinstance(..., tuple)`.
///
/// This is the one place the nD binding copies rather than views, and it is a
/// presentation rather than a copy of state: `dim` is at most 3 everywhere in the
/// tree, so three `PyLong`s per access is not a shape change.
///
/// \param counts The per-direction counts.
/// \return A tuple of `counts.size()` integers.
nb::tuple counts_tuple(std::span<const std::int64_t> counts) {
    nb::list values;
    for (const std::int64_t count : counts) {
        values.append(count);
    }
    return nb::tuple(values);
}

/// Register one scalar type's tensor-product `BsplineSpace`.
///
/// \param m The extension module.
/// \param name The Python-visible class name, `BsplineSpace32` or `BsplineSpace64`.
template <class T>
void bind_space_nd(nb::module_& m, const char* name) {
    using Space = pantr::bspline::BsplineSpace<T>;
    using Space1D = pantr::bspline::BsplineSpace1D<T>;
    using Handles = std::vector<std::shared_ptr<const Space1D>>;

    nb::class_<Space>(m, name)
        // Takes the directions as handles, which is what makes the C++ space SHARE
        // them; see the file comment. There is no dtype check to make here and none
        // to write: each `BsplineSpace1D<T>` class is distinct, so a `float32`
        // direction handed to `BsplineSpace64` is a `TypeError` from nanobind rather
        // than a silent widening -- and the oracle's own `ValueError` about mixed
        // dtypes is the wrapper's, which is where a type-kind check belongs.
        .def(
            "__init__",
            [](Space* self, Handles spaces) { new (self) Space(std::move(spaces)); },
            nb::arg("spaces"))
        .def_prop_ro("dim", [](const Space& s) { return s.dim(); })
        // Class H. No policy: the value is a `shared_ptr`, so ownership travels in
        // the return value. A tuple, matching the oracle's `spaces`.
        .def_prop_ro("spaces",
                     [](const Space& s) {
                         nb::list handles;
                         for (const auto& one_d : s.spaces()) {
                             handles.append(nb::cast(one_d));
                         }
                         return nb::tuple(handles);
                     })
        .def_prop_ro("degrees", [](const Space& s) { return counts_tuple(s.degrees()); })
        .def_prop_ro("tolerance", &Space::tolerance)
        .def_prop_ro("num_basis", [](const Space& s) { return counts_tuple(s.num_basis()); })
        .def_prop_ro("num_total_basis", [](const Space& s) { return s.num_total_basis(); })
        .def_prop_ro("num_intervals",
                     [](const Space& s) { return counts_tuple(s.num_intervals()); })
        .def_prop_ro("num_total_intervals",
                     [](const Space& s) { return s.num_total_intervals(); })
        // Class A: a read-only `(dim, 2)` view of the space's own storage, with the
        // space as the array's owner. `const T` is what sets the read-only flag and
        // `self` is what keeps the storage alive. That the oracle's counterpart is a
        // writable cached array is the defect recorded in
        // `design/bspline_derived_caches.md`; the wrapper is what reconciles the two,
        // by copying on the way out under both backends.
        .def_prop_ro("domain",
                     [](nb::handle self) {
                         const Space& s = nb::cast<const Space&>(self);
                         return view_2d_of<T>(self, s.domain_flat(),
                                              static_cast<std::size_t>(s.dim()), 2);
                     })
        // Spelled the oracle's way, as the 1D twin above is.
        .def("has_Bezier_like_knots", &Space::has_bezier_like_knots);
}

}  // namespace

void register_bspline_types(nb::module_& m) {
    bind_space_1d<float>(m, "BsplineSpace1D32");
    bind_space_1d<double>(m, "BsplineSpace1D64");
    bind_space_nd<float>(m, "BsplineSpace32");
    bind_space_nd<double>(m, "BsplineSpace64");
}
