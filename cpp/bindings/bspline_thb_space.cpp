/// \file
/// nanobind bindings for `pantr::bspline::THBSplineSpace`.
///
/// ## Two registrations, and one grid under both
///
/// `THBSplineSpace32` and `THBSplineSpace64` are separate classes for the reason
/// `bspline_types.cpp` gives: the root space's storage format is part of its value,
/// `float32` root spaces are exercised in the suite, and a single Python name would
/// have nothing left to carry the width.
///
/// The **grid is `double` under both**, which is not an oversight and is not the
/// dtype axis leaking. `pantr.grid` is `float64`-only by its own port's ruling and
/// `cpp/bindings/grid_hierarchical.cpp` registers no `float`, while a root B-spline
/// space stores whatever it was handed;
/// `tests/test_thb_spline_space.py::TestCellMembershipTolerance` ships a `float32`
/// root over a `float64` grid and pins that the grading stays `float64`. So the two
/// scalars are genuinely independent and the type says so.
///
/// ## What this file validates: nothing
///
/// The type validates its own arguments and throws `std::invalid_argument`, which
/// nanobind maps to `ValueError` preserving `what()`, and the messages are the
/// oracle's. What the wrapper still owes is Python's calling convention rather than
/// validation, and there are three pieces of it:
///
///  - coercing an `ArrayLike` `cell_ids` into a contiguous `int64` array, and the
///    `TypeError` for a non-integer one, for which `pantr/core/error.hpp` records
///    there is no `std::exception` nanobind turns into one;
///  - broadcasting a scalar `regularity` to one entry per direction, and refusing a
///    non-integer entry;
///  - the `IndexError` the oracle raises for an out-of-range cell id. It is
///    `std::out_of_range` here, which nanobind maps to `IndexError`, so this one is
///    already the right *kind* -- what the wrapper owes is the message's wording,
///    which lists every offending id where this names the first.
///
/// ## Arrays out: views, never copies
///
/// Every array below is storage the space owns and computed once -- the per-level
/// active sets at construction, the contribution table behind the memo in
/// `pantr/core/lazy.hpp`, a truncated function's coefficient box. They go out as
/// `nb::ndarray` views with the space as the array's owner, which is
/// `bspline_types.cpp`'s idiom and for its reasons: the owner keeps the storage alive
/// when the array outlives the handle it came from, and `const T` is what nanobind
/// turns into the read-only flag.
///
/// That is a **deliberate difference from the oracle**, which copies:
/// `active_function_indices` ends in `.copy()` and `active_basis` builds a fresh
/// array. `design/bspline_derived_caches.md` asks for exactly this change for the
/// contribution table -- a `span`-returning accessor retiring the oracle's unenforced
/// *"callers must not mutate it"* convention -- and the same argument reaches the
/// active sets. The wrapper is what reconciles the two: it copies on the way out under
/// both backends, so a caller that mutates the result of `active_function_indices`
/// keeps working and cannot corrupt a C++-owned array.
///
/// ## `_ref` accessors are not bound
///
/// `design/bspline_ownership_lifetime.md` requires that no borrowing accessor reaches
/// Python. This type has three -- `root_space_ref`, `grid_ref`, `level_space_ref` --
/// and none is below.
/// `tests/parity/test_bspline_binding_contract.py` asserts over the bound surface that
/// no method name ends in `_ref`, which is the only available check.
///
/// ## `grid` is the one handle that is not `const`
///
/// `design/bspline_ownership_lifetime.md`'s single exception, and `pantr/grid/tags.hpp`
/// carries the argument: a grid holds tag registries that are the reasoned
/// accumulating-container exception to construct-then-freeze, and a `const` handle
/// would reach only the `const` overload of `cell_tags()`, silently removing the
/// ability to tag cells through a THB space's grid.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "pantr/bspline/space_nd.hpp"
#include "pantr/bspline/thb_space.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using Grid = pantr::grid::HierarchicalGrid<double>;

/// A read-only, contiguous 1D `int64` array as nanobind sees it.
using const_ids = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A never-dereferenced address for a zero-length array.
///
/// An empty `std::vector`'s `data()` may be null and nanobind reads a null pointer as
/// "no array" rather than as an empty one. A cell with no active function and a
/// dimensionless space both produce empty results here, so a valid address stands in;
/// `bspline_types.cpp` and `grid_types.cpp` carry the same sentinel.
///
/// \tparam T The element type.
template <class T>
T kEmptyStorage{};

/// Hand out a span of the owner's storage as a read-only 1D numpy view.
///
/// \tparam T The element type.
/// \param self The Python object that owns the storage, kept alive by the array.
/// \param data The span; may be empty.
/// \return A read-only 1D `numpy` array viewing `data`.
template <class T>
nb::object view_of(nb::handle self, std::span<const T> data) {
    const T* base = data.empty() ? &kEmptyStorage<T> : data.data();
    return nb::cast(nb::ndarray<nb::numpy, const T, nb::ndim<1>>(base, {data.size()}, self));
}

/// A tuple of integers, as the oracle's per-direction properties are.
///
/// A near-copy of `bspline_types.cpp`'s, and deliberately so: `register.hpp`'s
/// reserved slots exist precisely so a ticket touches one `.cpp`, and hoisting a
/// six-line helper would put two tickets back in one file.
///
/// \param counts The values.
/// \return A Python tuple of them.
nb::tuple counts_tuple(std::span<const std::int64_t> counts) {
    nb::list values;
    for (const std::int64_t count : counts) {
        values.append(count);
    }
    return nb::tuple(values);
}

/// A `std::span` over a 1D nanobind array of ids.
///
/// \param ids The array.
/// \return A view of its elements.
std::span<const std::int64_t> ids_span(const const_ids& ids) {
    return {ids.data(), ids.size()};
}

/// Register one scalar type's `THBSplineSpace`.
///
/// \tparam T The root space's scalar type.
/// \param m The extension module.
/// \param name The Python-visible class name, `THBSplineSpace32` or `THBSplineSpace64`.
template <class T>
void bind_thb_space(nb::module_& m, const char* name) {
    using Space = pantr::bspline::THBSplineSpace<T>;
    using Root = pantr::bspline::BsplineSpace<T>;

    nb::class_<Space>(m, name)
        // Takes both nested objects as handles, which is what makes the space SHARE
        // them rather than copy them; `level_space(0)` then hands back the very handle
        // it was built from, which is the oracle's own identity contract.
        .def(
            "__init__",
            [](Space* self, std::shared_ptr<const Root> root_space, std::shared_ptr<Grid> grid,
               bool truncate, std::vector<std::optional<std::int64_t>> regularity) {
                new (self) Space(std::move(root_space), std::move(grid), truncate,
                                 std::move(regularity));
            },
            nb::arg("root_space"), nb::arg("grid"), nb::arg("truncate"),
            nb::arg("regularity"))
        // Class H throughout: the value is a `shared_ptr`, so ownership travels in the
        // return value and no policy is needed.
        .def_prop_ro("root_space", &Space::root_space)
        .def_prop_ro("grid", &Space::grid)
        .def_prop_ro("dim", [](const Space& s) { return s.dim(); })
        .def_prop_ro("degrees",
                     [](const Space& s) { return counts_tuple(s.degrees()); })
        .def_prop_ro("num_levels", [](const Space& s) { return s.num_levels(); })
        .def_prop_ro("truncate", &Space::truncate)
        .def_prop_ro("regularity",
                     [](const Space& s) {
                         nb::list values;
                         for (const std::optional<std::int64_t>& r : s.regularity()) {
                             values.append(r.has_value() ? nb::cast(*r) : nb::none());
                         }
                         return nb::tuple(values);
                     })
        .def_prop_ro("num_total_basis", [](const Space& s) { return s.num_total_basis(); })
        .def_prop_ro("num_basis_per_level",
                     [](const Space& s) { return counts_tuple(s.num_basis_per_level()); })
        .def_prop_ro("level_offsets",
                     [](nb::handle self) {
                         const Space& s = nb::cast<const Space&>(self);
                         return view_of<std::int64_t>(self, s.level_offsets());
                     })
        // Class A: a read-only `(dim, 2)` view of the root space's storage, with THIS
        // space as the array's owner -- which is correct because the root space is
        // shared and outlives it either way.
        .def_prop_ro("domain",
                     [](nb::handle self) {
                         const Space& s = nb::cast<const Space&>(self);
                         const std::span<const T> flat = s.domain();
                         const T* base = flat.empty() ? &kEmptyStorage<T> : flat.data();
                         return nb::cast(nb::ndarray<nb::numpy, const T, nb::ndim<2>>(
                             base, {static_cast<std::size_t>(s.dim()), std::size_t{2}}, self));
                     })
        .def_prop_ro("tolerance", &Space::tolerance)
        .def_prop_ro("num_truncated", [](const Space& s) { return s.num_truncated(); })
        .def("level_space", &Space::level_space, nb::arg("level"))
        .def("active_function_indices",
             [](nb::handle self, std::int64_t level) {
                 const Space& s = nb::cast<const Space&>(self);
                 return view_of<std::int64_t>(self, s.active_function_indices(level));
             },
             nb::arg("level"))
        .def("active_basis",
             [](nb::handle self, std::int64_t cid) {
                 const Space& s = nb::cast<const Space&>(self);
                 return view_of<std::int64_t>(self, s.active_basis(cid));
             },
             nb::arg("cid"))
        // The three parallel views of one cell's slice of the contribution table.
        // A tuple rather than three calls, because a caller wanting the multi-indices
        // wants the dofs beside them and a second lookup would re-check the cid.
        .def("contributions",
             [](nb::handle self, std::int64_t cid) {
                 const Space& s = nb::cast<const Space&>(self);
                 const pantr::bspline::CellContributions c = s.contributions(cid);
                 const std::size_t rows = c.multi_indices.size() == 0
                                              ? 0
                                              : static_cast<std::size_t>(c.size());
                 const std::int64_t* multi_base = c.multi_indices.empty()
                                                      ? &kEmptyStorage<std::int64_t>
                                                      : c.multi_indices.data();
                 return nb::make_tuple(
                     view_of<std::int64_t>(self, c.dofs),
                     view_of<std::int64_t>(self, c.levels),
                     nb::cast(nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<2>>(
                         multi_base, {rows, static_cast<std::size_t>(s.dim())}, self)));
             },
             nb::arg("cid"))
        .def("max_active_per_cell", &Space::max_active_per_cell)
        .def("dof_level", &Space::dof_level, nb::arg("dof"))
        // `None` for a function the truncation left alone, which is what the oracle's
        // `_trunc.get(dof)` gives and what distinguishes a plain tensor-product
        // B-spline from a truncated one carrying an all-ones box.
        .def("truncated",
             [](nb::handle self, std::int64_t dof) -> nb::object {
                 const Space& s = nb::cast<const Space&>(self);
                 const std::optional<pantr::bspline::TruncatedView> view = s.truncated(dof);
                 if (!view.has_value()) {
                     return nb::none();
                 }
                 std::vector<std::size_t> shape;
                 shape.reserve(view->shape.size());
                 for (const std::int64_t n : view->shape) {
                     shape.push_back(static_cast<std::size_t>(n));
                 }
                 const double* base =
                     view->coeffs.empty() ? &kEmptyStorage<double> : view->coeffs.data();
                 nb::object coeffs = nb::cast(nb::ndarray<nb::numpy, const double>(
                     base, shape.size(), shape.data(), self));
                 return nb::make_tuple(view->rep_level, counts_tuple(view->box_lo), coeffs);
             },
             nb::arg("dof"))
        .def("refine",
             [](const Space& s, const_ids cell_ids,
                std::optional<std::int64_t> admissible_class) {
                 return s.refine(ids_span(cell_ids), admissible_class);
             },
             nb::arg("cell_ids").noconvert(), nb::arg("admissible_class"))
        .def("refine_region",
             [](const Space& s, std::int64_t level, const_ids lo, const_ids hi,
                std::optional<std::int64_t> admissible_class) {
                 return s.refine_region(level, ids_span(lo), ids_span(hi), admissible_class);
             },
             nb::arg("level"), nb::arg("lo").noconvert(), nb::arg("hi").noconvert(),
             nb::arg("admissible_class"))
        .def("coarsen",
             [](const Space& s, const_ids cell_ids,
                std::optional<std::int64_t> admissible_class) {
                 return s.coarsen(ids_span(cell_ids), admissible_class);
             },
             nb::arg("cell_ids").noconvert(), nb::arg("admissible_class"))
        .def("__repr__", &Space::to_string);
}

}  // namespace

void register_bspline_thb_space(nb::module_& m) {
    bind_thb_space<float>(m, "THBSplineSpace32");
    bind_thb_space<double>(m, "THBSplineSpace64");
}
