/// \file
/// nanobind bindings for `pantr::grid::TensorProductGrid`.
///
/// The fifth type this extension exposes, and the first that is a GRID: it derives
/// from the CRTP mixin in `pantr/grid/grid.hpp`, so most of what is bound below is a
/// generic default the grid inherits rather than something the grid wrote.
///
/// ## `double` only
///
/// `src/pantr/grid/_grid_backend.py` records that `pantr.grid`'s oracle is
/// `float64`-only, so a `float` grid would be a surface with no oracle behind it,
/// which `design/backend_parity.md` Rule 8 forbids. The header stays templated and
/// `cpp/tests/test_grid_tensor_product.cpp` censuses it at `float` as a compile-time
/// device; only the binding is narrow, and no name here carries a width suffix.
///
/// ## Returning a reference into the grid: which binding shape needs a policy, and why
///
/// All three of `cell_tags`, `facet_tags` and `cell_bvh` return a REFERENCE to a member
/// of the grid, so none of them may be copied on the way out. The two binding shapes
/// differ, and an earlier version of this comment got that wrong by measuring one and
/// writing about the other:
///
///  - **A `.def()` method returning `T&` defaults to a COPY.** `nb_cast.h:459-464`
///    resolves both `automatic` and `automatic_reference` to `rv_policy::copy` for an
///    lvalue reference. Measured on this tree: with the policy dropped from a
///    `.def()`-bound `cell_tags`, `g.cell_tags().set(...)` mutated a temporary, the tag
///    did not appear in `names`, and nothing raised. `cell_bvh` is bound this way --
///    the Python contract for it is a call -- so its `reference_internal` is
///    load-bearing.
///  - **A `def_prop_ro` getter already defaults to `reference_internal`.**
///    nanobind hardcodes it when it builds the getter (`nb_class.h:701`), independent
///    of the extras the caller passes. Measured: with the annotation removed from
///    `cell_tags`, the write still survived and the identity still held. So on the two
///    registries the annotation is redundant. It is kept anyway, and stated to be
///    redundant rather than removed: it is a hardcoded default in a third-party header
///    rather than a documented guarantee, and one word here is cheaper than the silent
///    data loss its absence would cause if that default ever moved.
///
/// `tests/parity/test_grid_types.py` asserts that a tag written through the property is
/// visible on the next read, which is what stops any of this regressing.
///
/// ## What this file validates: nothing
///
/// The grid validates its own arguments and throws, and nanobind's translator maps
/// `std::invalid_argument` to `ValueError` and `std::out_of_range` to `IndexError`
/// preserving `what()`, which are the exceptions and the messages the oracle raises for
/// the same inputs. Two things the wrapper still owes are Python's own calling
/// convention rather than validation: coercing an `ArrayLike` to a contiguous `float64`
/// array, and the `TypeError` for a non-integer `cell_ids`, for which there is no
/// `std::exception` nanobind turns into one.
///
/// ## Arrays out: owned copies, except the breakpoints
///
/// Everything computed on demand -- located ids, collected bounds, boundary facets --
/// is handed over as an array that OWNS a moved-from vector, because the vector is a
/// temporary and a view of it would dangle. The breakpoints are the exception: they are
/// the grid's own storage, they are what the type exists to hold once, and copying them
/// per access would introduce a performance difference the backend switch must not
/// introduce. They go out as read-only views owning a reference to the grid.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/grid/tensor_product_grid.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using Grid = pantr::grid::TensorProductGrid<double>;

/// A read-only, contiguous 1D array of the given type as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2D `float64` array as nanobind sees it.
using const_points = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A never-dereferenced address for a zero-length array.
///
/// An empty `std::vector`'s `data()` may be null, and nanobind reads a null pointer as
/// "no array" rather than as an empty one. An empty result is legal here -- a query box
/// overlapping nothing, a batch of no points -- so a valid address stands in.
///
/// \tparam T The element type.
template <class T>
T kEmptyStorage{};

/// Hand a computed vector to numpy as a fresh, owning, writeable array.
///
/// The vector is moved onto the heap and a capsule deletes it, so the array outlives
/// the call that produced it. Writeable, and the freezing is the wrapper's job rather
/// than this file's: `pantr.grid.TensorProductGrid` clears the write flag on every array
/// its contract calls read-only -- `boundary_facets`, `bounds`, and both of
/// `restrict`'s index arrays -- so that the flag cannot depend on which backend
/// produced the array. Anything added here inherits that rule rather than an
/// exception list.
///
/// \tparam T The element type.
/// \tparam Rank The array's rank.
/// \param values The values to hand over.
/// \param shape The shape, whose product must be `values.size()`.
/// \return The numpy array.
template <class T, std::size_t Rank>
nb::object owned_array(std::vector<T>&& values, std::array<std::size_t, Rank> shape) {
    auto* data = new std::vector<T>(std::move(values));
    // One reserved slot rather than the shared `kEmptyStorage` sentinel, and the reason
    // is `grid_bvh.cpp`'s: that sentinel is `const` and is shared by every zero-size
    // result of its type, which is fine for a read-only view and wrong in principle for
    // an array handed back WRITEABLE. Reserving gives this array an address of its own,
    // owned by its own capsule, at the cost of one element for a result that has none.
    if (data->empty()) {
        data->reserve(1);
    }
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, T, nb::ndim<Rank>>(data->data(), Rank, shape.data(), owner));
}

/// Expose part of a grid's own storage as a read-only numpy view.
///
/// The owner is the grid's Python object, so the view keeps the grid alive rather than
/// the other way round, and it is also what makes the array read-only: an ownerless
/// `nb::ndarray` return silently COPIES and comes back writeable, which would pass a
/// value assertion and fail the oracle's read-only contract on `breakpoints`.
///
/// \param self The grid's Python object.
/// \param data The span into its storage.
/// \return A read-only `float64` array aliasing `data`.
nb::object view_of(nb::handle self, std::span<const double> data) {
    const std::size_t n = data.size();
    const double* base = n == 0 ? &kEmptyStorage<double> : data.data();
    return nb::cast(nb::ndarray<nb::numpy, const double, nb::ndim<1>>(base, {n}, self));
}

/// A `std::span` over a 1D nanobind array.
///
/// \tparam T The element type.
/// \param a The array.
/// \return A span over its storage, valid while the array is alive.
template <class T>
std::span<const T> as_span(const_vec<T> a) {
    return {a.data(), a.shape(0)};
}

/// A row-major 2D view over a nanobind array.
///
/// \param a The array.
/// \return A view over its storage.
pantr::span2d<const double> as_points(const_points a) {
    return pantr::span2d<const double>(a.data(), a.shape(0), a.shape(1));
}

/// Build a grid from a sequence of per-axis breakpoint arrays.
///
/// Each element is cast to a contiguous `float64` array rather than converted element
/// by element, so a wrongly typed axis is rejected here instead of being silently
/// widened. The grid then packs them into its single flat buffer, once.
///
/// \param self Uninitialised storage for the grid.
/// \param breakpoints A sequence of one array per axis.
void init_grid(Grid* self, nb::sequence breakpoints) {
    std::vector<std::vector<double>> axes;
    std::size_t axis = 0;
    for (nb::handle item : breakpoints) {
        // `try_cast` rather than `cast`, because a bare `nb::cast` that fails throws
        // `nb::cast_error`, which IS `std::bad_cast` (`nb_error.h`), which the default
        // translator maps to `RuntimeError: std::bad_cast` -- an exception of the wrong
        // type whose message names neither the argument nor the reason. nanobind gives
        // a declared `nb::ndarray` PARAMETER a graceful `TypeError` for free; a manual
        // cast inside a lambda body is outside that path and has to raise its own.
        const_vec<double> arr;
        if (!nb::try_cast(item, arr)) {
            throw nb::type_error(
                ("breakpoints[" + std::to_string(axis)
                 + "] must be a contiguous float64 array")
                    .c_str());
        }
        axes.emplace_back(arr.data(), arr.data() + arr.shape(0));
        ++axis;
    }
    new (self) Grid(axes);
}

/// The per-axis `[lo, hi]` extremes as an `(ndim, 2)` array.
///
/// \param g The grid.
/// \return A fresh `(ndim, 2)` `float64` array.
nb::object grid_bounds(const Grid& g) {
    const auto n = static_cast<std::size_t>(g.ndim());
    std::vector<double> values(n * 2);
    for (std::size_t d = 0; d < n; ++d) {
        const std::span<const double> bp = g.breakpoints(static_cast<std::int64_t>(d));
        values[d * 2] = bp.front();
        values[(d * 2) + 1] = bp.back();
    }
    return owned_array<double, 2>(std::move(values), {n, std::size_t{2}});
}

/// The per-axis breakpoint arrays, as a tuple of read-only views.
///
/// A tuple rather than a list, because the oracle's `breakpoints` is a tuple and a
/// caller unpacking or comparing it must not start failing under the C++ backend.
///
/// \param self The grid's Python object.
/// \return A length-`ndim` tuple of views into the grid's flat buffer.
nb::tuple grid_breakpoints(nb::handle self) {
    const Grid& g = nb::cast<const Grid&>(self);
    nb::list axes;
    for (std::int64_t d = 0; d < g.ndim(); ++d) {
        axes.append(view_of(self, g.breakpoints(d)));
    }
    return nb::tuple(axes);
}

/// Cell `cid`'s corners, as two fresh arrays.
///
/// \param g The grid.
/// \param cid Cell identifier.
/// \return `(lo, hi)`, each of shape `(ndim,)`.
nb::tuple cell_bounds(const Grid& g, std::int64_t cid) {
    const auto n = static_cast<std::size_t>(g.ndim());
    std::vector<double> lo(n);
    std::vector<double> hi(n);
    g.cell_bounds(cid, lo, hi);
    return nb::make_tuple(owned_array<double, 1>(std::move(lo), {n}),
                          owned_array<double, 1>(std::move(hi), {n}));
}

/// Local facet `lfid` of cell `cid`'s degenerate corners, as two fresh arrays.
///
/// \param g The grid.
/// \param cid Cell identifier.
/// \param lfid Local facet identifier.
/// \return `(lo, hi)`, each of shape `(ndim,)`.
nb::tuple local_facet_bounds(const Grid& g, std::int64_t cid, std::int64_t lfid) {
    const auto n = static_cast<std::size_t>(g.ndim());
    std::vector<double> lo(n);
    std::vector<double> hi(n);
    g.local_facet_bounds(cid, lfid, lo, hi);
    return nb::make_tuple(owned_array<double, 1>(std::move(lo), {n}),
                          owned_array<double, 1>(std::move(hi), {n}));
}

/// Per-cell corners in cell-id order, as two fresh `(num_cells, ndim)` arrays.
///
/// \param g The grid.
/// \return `(cell_lo, cell_hi)`.
nb::tuple collect_cell_bounds(const Grid& g) {
    const auto rows = static_cast<std::size_t>(g.num_cells());
    const auto cols = static_cast<std::size_t>(g.ndim());
    std::vector<double> lo(rows * cols);
    std::vector<double> hi(rows * cols);
    g.collect_cell_bounds(pantr::span2d<double>(lo.data(), rows, cols),
                          pantr::span2d<double>(hi.data(), rows, cols));
    return nb::make_tuple(owned_array<double, 2>(std::move(lo), {rows, cols}),
                          owned_array<double, 2>(std::move(hi), {rows, cols}));
}

/// The windowed sub-grid and its two index arrays.
///
/// A plain tuple rather than a bound `GridRestriction`: the public type is the Python
/// `NamedTuple` of that name, and the wrapper assembles it. Binding a second one would
/// put two spellings of one record in front of the caller.
///
/// `in_subset` leaves here as `uint8` because that is what the C++ struct holds --
/// `std::vector<bool>` is a bit-packed proxy with no `data()`. The wrapper converts it
/// to the `bool` array the oracle returns, which is Python's calling convention rather
/// than a reassembly of anything.
///
/// \param g The grid.
/// \param cell_ids Flat cell identifiers to span.
/// \return `(sub_grid, local_to_global_cell, in_subset)`.
nb::tuple restrict(const Grid& g, const_vec<std::int64_t> cell_ids) {
    auto r = g.restrict(as_span(cell_ids));
    const std::size_t n = r.local_to_global_cell.size();
    return nb::make_tuple(nb::cast(std::move(r.grid)),
                          owned_array<std::int64_t, 1>(std::move(r.local_to_global_cell), {n}),
                          owned_array<std::uint8_t, 1>(std::move(r.in_subset), {n}));
}

}  // namespace

void register_grid_types(nb::module_& m) {
    nb::class_<Grid>(m, "TensorProductGrid")
        .def("__init__", &init_grid, nb::arg("breakpoints"))

        // ------------------------------------------------------------------
        // Read-only geometry
        // ------------------------------------------------------------------
        .def_prop_ro("ndim", &Grid::ndim)
        .def_prop_ro("num_cells", &Grid::num_cells)
        // A tuple, for the reason `grid_breakpoints` gives about `breakpoints`.
        .def_prop_ro("cells_per_axis",
                     [](const Grid& g) {
                         nb::list counts;
                         for (const std::int64_t n : g.cells_per_axis()) {
                             counts.append(n);
                         }
                         return nb::tuple(counts);
                     })
        .def_prop_ro("breakpoints", &grid_breakpoints)
        .def_prop_ro("bounds", &grid_bounds)
        .def_prop_ro("is_uniform", &Grid::is_uniform)

        // ------------------------------------------------------------------
        // Index helpers
        // ------------------------------------------------------------------
        .def(
            "cell_multi_index",
            [](const Grid& g, std::int64_t cid) {
                std::vector<std::int64_t> multi(static_cast<std::size_t>(g.ndim()));
                g.cell_multi_index(cid, multi);
                nb::list indices;
                for (const std::int64_t i : multi) {
                    indices.append(i);
                }
                return nb::tuple(indices);
            },
            nb::arg("cid"))
        .def(
            "flat_cell_index",
            [](const Grid& g, const_vec<std::int64_t> multi) {
                return g.flat_cell_index(as_span(multi));
            },
            // `.noconvert()` on every integer argument in this file, for the reason its
            // siblings give: without it nanobind converts a float array into an int64
            // temporary, so `flat_cell_index([1.9, 1.1])` truncates to `[1, 1]` and
            // returns a plausible cell id instead of refusing. The public wrapper checks
            // the dtype itself, so this is defence for a direct caller of the handle.
            nb::arg("multi").noconvert())

        // ------------------------------------------------------------------
        // The primitives
        // ------------------------------------------------------------------
        .def("cell_bounds", &cell_bounds, nb::arg("cid"))
        .def(
            "locate",
            [](const Grid& g, const_vec<double> pt) { return g.locate(as_span(pt)); },
            nb::arg("pt"))
        .def("neighbor_across_facet", &Grid::neighbor_across_facet, nb::arg("cid"),
             nb::arg("lfid"))

        // ------------------------------------------------------------------
        // The hooks
        // ------------------------------------------------------------------
        .def(
            "locate_many",
            [](const Grid& g, const_points points) {
                std::vector<std::int64_t> out = g.locate_many(as_points(points));
                const std::size_t n = out.size();
                return owned_array<std::int64_t, 1>(std::move(out), {n});
            },
            nb::arg("points"))
        .def("collect_cell_bounds", &collect_cell_bounds)
        .def("restrict", &restrict, nb::arg("cell_ids").noconvert())

        // ------------------------------------------------------------------
        // The generic defaults the grid inherits
        // ------------------------------------------------------------------
        .def("cell_aabb", &Grid::cell_aabb, nb::arg("cid"))
        .def("cell_level", &Grid::cell_level, nb::arg("cid"))
        .def("child_cells", &Grid::child_cells, nb::arg("cid"))
        .def("reference_map", &Grid::reference_map, nb::arg("cid"))
        .def("neighbors", &Grid::neighbors, nb::arg("cid"))
        .def("num_local_facets", &Grid::num_local_facets, nb::arg("cid"))
        .def("local_facet_axis_side", &Grid::local_facet_axis_side, nb::arg("cid"),
             nb::arg("lfid"))
        .def("local_facet_bounds", &local_facet_bounds, nb::arg("cid"), nb::arg("lfid"))
        .def("is_mesh_boundary_facet", &Grid::is_mesh_boundary_facet, nb::arg("cid"),
             nb::arg("lfid"))
        .def("boundary_facets",
             [](const Grid& g) {
                 std::vector<std::int64_t> rows = g.boundary_facets();
                 const std::size_t n = rows.size() / 2;
                 return owned_array<std::int64_t, 2>(std::move(rows), {n, std::size_t{2}});
             })
        .def("hanging_neighbors", &Grid::hanging_neighbors, nb::arg("cid"), nb::arg("lfid"))
        //
        // `query_aabb` is deliberately NOT bound. The C++ grid has it, but
        // `_GridWrapper.query_aabb` routes through `cell_bvh()` instead, which is what
        // the oracle's own default does and which keeps a single unpacking of an `AABB`
        // into corner arrays -- `pantr.grid.BVH`'s. Binding it would add the only member
        // here with no Python caller.

        // ------------------------------------------------------------------
        // Base-owned state: the three that must alias rather than copy
        // ------------------------------------------------------------------
        //
        // See the file header. Without `reference_internal` a write through the
        // returned registry is silently lost, and the returned BVH is a copy of a
        // cache rather than the cache.
        //
        // `cell_tags` and `facet_tags` are PROPERTIES here and methods in C++, which
        // is the one place this file departs from mirroring the header. The Python
        // contract for both is an attribute -- `g.cell_tags.set(...)` -- and matching
        // it here is what lets the wrapper reach the two implementations through one
        // expression instead of testing which backend it holds. `cell_bvh` stays a
        // method, because the Python contract for that one is a call.
        .def_prop_ro("cell_tags", nb::overload_cast<>(&Grid::cell_tags),
                     nb::rv_policy::reference_internal)
        .def_prop_ro("facet_tags", nb::overload_cast<>(&Grid::facet_tags),
                     nb::rv_policy::reference_internal)
        .def("cell_bvh", &Grid::cell_bvh, nb::rv_policy::reference_internal)

        .def("__repr__", &Grid::to_string);

    // The factory is a free function in C++ and a free function here, matching
    // `pantr.grid.uniform_grid`. `cells` arrives as an array rather than as a scalar or
    // a sequence: broadcasting a scalar over the axes is the wrapper's job, being part
    // of Python's calling convention rather than of the grid.
    m.def(
        "uniform_grid",
        [](const_points bounds, const_vec<std::int64_t> cells) {
            return pantr::grid::uniform_grid<double>(as_points(bounds), as_span(cells));
        },
        nb::arg("bounds"), nb::arg("cells").noconvert());
}
