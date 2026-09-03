/// \file
/// nanobind bindings for `pantr::grid::HierarchicalGrid`.
///
/// The second grid the extension exposes, after `grid_types.cpp`'s tensor-product one,
/// and the last piece of FELIGN/pantr#395: the C++ type was complete before this file
/// existed, and what is added here is the seam Python reaches it through.
///
/// ## `double` only
///
/// Same reason as its sibling's, and it is not a preference. `src/pantr/grid/_grid_backend.py`
/// records that `pantr.grid`'s oracle is `float64`-only, so a `float` hierarchy would be a
/// surface with no oracle behind it, which `design/backend_parity.md` Rule 8 forbids --
/// a parity claim is only defined where the comparison can say something. The header
/// stays templated and `cpp/tests/test_grid_hierarchical_type.cpp` censuses it at
/// `float`; only the binding is narrow, and no name here carries a width suffix.
///
/// ## What this file validates: nothing
///
/// The grid validates its own arguments and throws, and nanobind maps
/// `std::invalid_argument` to `ValueError` and `std::out_of_range` to `IndexError`
/// preserving `what()`. What the wrapper still owes is Python's calling convention, not
/// validation, and there are exactly three pieces of it:
///
///  - coercing an `ArrayLike` or a `Sequence[int]` into a contiguous typed array;
///  - the `TypeError` for a non-integer `cell_ids`, for which no `std::exception` maps;
///  - **`cell_id` on a wrongly sized `midx`.** It throws here -- `pantr/core/error.hpp`
///    puts value and range checks in C++ and leaves shape coercion to the wrapper --
///    while the oracle folds a wrong length into its "not an active leaf" answer and
///    returns `None`. That is a difference in the *type* of the answer rather than in
///    its wording, so the wrapper owes the `None` and catches the length case before
///    forwarding. `tests/parity/test_grid_hierarchical.py` pins it, and it is why
///    `is_active_leaf` is not bound at all (see below).
///
/// ## Arrays out: owned copies throughout
///
/// Unlike `grid_types.cpp`, which hands out the breakpoints as a view into the grid's own
/// storage, nothing here is storage a caller reads repeatedly: every array below is
/// computed on demand -- a mask, a lattice, a connectivity table -- so it owns a moved-from
/// vector and there is no view to dangle. The one thing that *is* stored, the root grid,
/// goes out by value (see `root` below).
///
/// `owned_array` is a near-copy of `grid_types.cpp`'s, and deliberately so: the reserved
/// slots in `register.hpp` exist precisely so a ticket touches one `.cpp`, and hoisting a
/// shared helper would put every grid ticket back in one file. It is twenty lines with a
/// single caller pattern; the duplication is cheaper than the coupling.

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
#include <utility>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/grid/blocks.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/grid/tensor_product_grid.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using Grid = pantr::grid::HierarchicalGrid<double>;
using Root = pantr::grid::TensorProductGrid<double>;

/// A read-only, contiguous 1D array of the given type as nanobind sees it.
template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2D array of the given type as nanobind sees it.
template <class T>
using const_mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A never-dereferenced address for a zero-length array.
///
/// An empty `std::vector`'s `data()` may be null, and nanobind reads a null pointer as
/// "no array" rather than as an empty one. An empty result is legal here -- a batch of no
/// points, a boundary with no facets -- so a valid address stands in.
///
/// \tparam T The element type.
template <class T>
T kEmptyStorage{};

/// Hand a computed vector to numpy as a fresh, owning, writeable array.
///
/// The vector is moved onto the heap and a capsule deletes it, so the array outlives the
/// call that produced it. Writeable, and the freezing is the wrapper's job rather than
/// this file's: `pantr.grid.HierarchicalGrid` clears the write flag on every array its
/// contract calls read-only. Anything added here inherits that rule rather than an
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
    // One reserved slot rather than the shared `kEmptyStorage` sentinel: that one is
    // `const` and shared by every zero-size result of its type, which is fine for a
    // read-only view and wrong in principle for an array handed back WRITEABLE.
    if (data->empty()) {
        data->reserve(1);
    }
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, T, nb::ndim<Rank>>(data->data(), Rank, shape.data(), owner));
}

/// A `std::span` over a 1D nanobind array.
///
/// \tparam T The element type.
/// \param a The array.
/// \return A span over its storage, valid while the array is alive.
template <class T>
std::span<const T> as_span(const_vec<T> a) {
    return {a.shape(0) == 0 ? &kEmptyStorage<T> : a.data(), a.shape(0)};
}

/// A row-major 2D view over a nanobind array.
///
/// \param a The array.
/// \return A view over its storage.
pantr::span2d<const double> as_points(const_mat<double> a) {
    return pantr::span2d<const double>(a.data(), a.shape(0), a.shape(1));
}

/// The per-axis subdivision factor, as a tuple.
///
/// A tuple rather than a list, because the oracle's `factor` is a tuple and a caller
/// unpacking or comparing it must not start failing under the C++ backend.
///
/// \param g The grid.
/// \return A length-`ndim` tuple of `int`.
nb::tuple grid_factor(const Grid& g) {
    nb::list entries;
    for (const std::int64_t f : g.factor()) {
        entries.append(f);
    }
    return nb::tuple(entries);
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

/// Cell `cid`'s per-axis index at its own level, as a tuple.
///
/// \param g The grid.
/// \param cid Cell identifier.
/// \return A length-`ndim` tuple of `int`.
nb::tuple cell_multi_index(const Grid& g, std::int64_t cid) {
    std::vector<std::int64_t> midx(static_cast<std::size_t>(g.ndim()));
    g.cell_multi_index(cid, midx);
    nb::list entries;
    for (const std::int64_t i : midx) {
        entries.append(i);
    }
    return nb::tuple(entries);
}

/// The per-axis cell count of the level-`level` grid, as a tuple.
///
/// A tuple over every axis, where the header takes an `axis` argument and returns one
/// count. That is the one place this file departs from mirroring the header, and the
/// reason is `grid_types.cpp`'s for binding `cell_tags` as a property: the Python
/// contract for this name is a length-`ndim` tuple, and matching it here is what lets the
/// wrapper reach the two implementations through one expression instead of testing which
/// backend it holds.
///
/// \param g The grid.
/// \param level Hierarchy level; a pure formula, so it need not exist.
/// \return A length-`ndim` tuple of `int`.
/// \throws std::invalid_argument If `level` is negative.
nb::tuple level_cells_per_axis(const Grid& g, std::int64_t level) {
    nb::list counts;
    for (std::int64_t k = 0; k < g.ndim(); ++k) {
        counts.append(g.level_cells_per_axis(level, k));
    }
    return nb::tuple(counts);
}

/// The active-leaf rectangles at `level`, as a tuple of `(lo, hi)` tuples.
///
/// Tuples of `int` rather than arrays, matching the oracle: the blocks are a small,
/// hashable description of the active set that the suite compares with `==`, and two
/// arrays would compare elementwise instead.
///
/// \param g The grid.
/// \param level Hierarchy level.
/// \return `((lo, hi), ...)` in stored order.
nb::tuple active_blocks(const Grid& g, std::int64_t level) {
    const auto [lo, hi] = g.active_blocks(level);
    const std::size_t d = lo.extent(1);
    nb::list blocks;
    for (std::size_t b = 0; b < lo.extent(0); ++b) {
        nb::list block_lo;
        nb::list block_hi;
        for (std::size_t k = 0; k < d; ++k) {
            block_lo.append(lo(b, k));
            block_hi.append(hi(b, k));
        }
        blocks.append(nb::make_tuple(nb::tuple(block_lo), nb::tuple(block_hi)));
    }
    return nb::tuple(blocks);
}

/// One of the two level masks, flat.
///
/// Flat rather than shaped, and `uint8` rather than `bool`: the C++ side holds
/// `std::vector<std::uint8_t>` because `std::vector<bool>` is a bit-packed proxy with no
/// `data()`, and the wrapper has to make a `bool` copy in any case. Reshaping it there,
/// where `level_cells_per_axis` is already a tuple, costs nothing and keeps this file
/// free of a rank it would have to compute at run time.
///
/// \param values The mask.
/// \return A fresh `uint8` array of shape `(prod(level_cells_per_axis),)`.
nb::object flat_mask(std::vector<std::uint8_t>&& values) {
    const std::size_t n = values.size();
    return owned_array<std::uint8_t, 1>(std::move(values), {n});
}

/// The windowed sub-grid and its two index arrays.
///
/// A plain tuple rather than a bound `GridRestriction`, for `grid_types.cpp`'s reason: the
/// public type is the Python `NamedTuple` of that name and the wrapper assembles it.
/// `in_subset` leaves as `uint8` for the reason `flat_mask` gives.
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

/// The deduplicated leaf-cell mesh.
///
/// \param g The grid.
/// \return `(points, conn)` of shapes `(num_vertices, ndim)` and `(num_cells, 2 ** ndim)`.
nb::tuple export_cells(const Grid& g) {
    auto mesh = g.export_cells();
    const auto d = static_cast<std::size_t>(g.ndim());
    const auto vertices = static_cast<std::size_t>(mesh.num_vertices);
    const auto cells = static_cast<std::size_t>(g.num_cells());
    const std::size_t corners = cells == 0 ? 0 : mesh.conn.size() / cells;
    return nb::make_tuple(owned_array<double, 2>(std::move(mesh.points), {vertices, d}),
                          owned_array<std::int64_t, 2>(std::move(mesh.conn), {cells, corners}));
}

/// Rebuild a hierarchy from the packed active-set descriptor.
///
/// The entry point `HierarchicalGrid.__reduce__` names. The blocks arrive as the packed
/// form the type already stores -- two `(n_blocks, ndim)` corner arrays plus the per-level
/// block index range -- rather than as nested sequences, because that is the shape the
/// grid holds and the shape `active_blocks` reads out, so a round trip converts nothing.
///
/// Every level is normalized (`std::nullopt`), which the header's `\warning` records as
/// always safe: normalization is a fixed point, so a level holding another grid's already
/// normalized list normalizes to itself, in the same order. Naming levels clean would be a
/// cost switch with a silent wrong answer as its failure mode, and a pickle load is not
/// where to spend that.
///
/// \param root The level-0 grid.
/// \param factor Per-axis subdivision factor, length `root.ndim()`.
/// \param block_lo Lower corners, `(n_blocks, ndim)`, in level order.
/// \param block_hi Upper corners, same shape.
/// \param level_start Block index range per level, length `n_levels + 1`.
/// \return The grid.
/// \throws std::invalid_argument If the corner arrays disagree in shape, `level_start` is
///         not non-decreasing from zero to `n_blocks`, or the grid itself refuses.
Grid from_blocks(const Root& root, const_vec<std::int64_t> factor,
                 const_mat<std::int64_t> block_lo, const_mat<std::int64_t> block_hi,
                 const_vec<std::int64_t> level_start) {
    const std::int64_t ndim = root.ndim();
    if (block_lo.shape(0) != block_hi.shape(0) || block_lo.shape(1) != block_hi.shape(1)) {
        throw std::invalid_argument("block_lo and block_hi must have the same shape.");
    }
    if (block_lo.shape(1) != static_cast<std::size_t>(ndim)) {
        throw std::invalid_argument("block_lo and block_hi must have root.ndim() columns.");
    }
    if (level_start.shape(0) < 2) {
        throw std::invalid_argument("level_start must have at least two entries.");
    }
    const std::span<const std::int64_t> starts = as_span(level_start);
    const auto n_blocks = static_cast<std::int64_t>(block_lo.shape(0));
    if (starts.front() != 0 || starts.back() != n_blocks) {
        throw std::invalid_argument("level_start must run from 0 to the block count.");
    }
    const auto d = static_cast<std::size_t>(ndim);
    std::vector<pantr::grid::BlockList> blocks;
    blocks.reserve(starts.size() - 1);
    for (std::size_t l = 0; l + 1 < starts.size(); ++l) {
        if (starts[l + 1] < starts[l]) {
            throw std::invalid_argument("level_start must be non-decreasing.");
        }
        pantr::grid::BlockList level(ndim);
        level.reserve(static_cast<std::size_t>(starts[l + 1] - starts[l]));
        for (auto b = static_cast<std::size_t>(starts[l]);
             b < static_cast<std::size_t>(starts[l + 1]); ++b) {
            level.push_back(pantr::grid::BlockView{
                std::span<const std::int64_t>(block_lo.data() + (b * d), d),
                std::span<const std::int64_t>(block_hi.data() + (b * d), d)});
        }
        blocks.push_back(std::move(level));
    }
    return Grid::from_blocks(root, as_span(factor), std::move(blocks), std::nullopt);
}

}  // namespace

void register_grid_hierarchical(nb::module_& m) {
    nb::class_<Grid>(m, "HierarchicalGrid")
        .def(
            "__init__",
            [](Grid* self, const Root& root, const_vec<std::int64_t> factor) {
                new (self) Grid(root, as_span(factor));
            },
            nb::arg("root"),
            // `.noconvert()` on every integer argument in this file, for the reason its
            // siblings give: without it nanobind converts a float array into an int64
            // temporary, so a factor of `[1.9]` truncates to `[1]` and builds a grid that
            // never subdivides instead of refusing. The wrapper checks the dtype itself,
            // so this is defence for a direct caller of the handle.
            nb::arg("factor").noconvert())
        .def_static("from_blocks", &from_blocks, nb::arg("root"),
                    // Keyword-only from here: four adjacent, same-typed, mutually
                    // unordered `int64` descriptor arrays, every transposition of which
                    // type-checks and runs. `cpp/bindings/grid.cpp` argues this at length.
                    nb::kw_only(), nb::arg("factor").noconvert(),
                    nb::arg("block_lo").noconvert(), nb::arg("block_hi").noconvert(),
                    nb::arg("level_start").noconvert())

        // ------------------------------------------------------------------
        // This grid's own surface
        // ------------------------------------------------------------------
        //
        // `root` hands back a COPY, by value, and that is a deliberate choice rather than
        // an oversight. `design/bspline_ownership_lifetime.md` classifies the accessor
        // rather than the type: an accessor that constructs its result is class V, bound
        // with no policy -- nanobind resolves a by-value return to `move` even where a
        // property's built-in `reference_internal` is passed ahead of the extras, so there
        // is nothing to name and nothing to get wrong. The grid stores its root by value,
        // so class H's `shared_ptr<const T>` would be a change to the C++ type for the
        // binding's convenience, which is the wrong direction of causation. What keeps
        // `g.root is g.root` true is the wrapper's memo slot, not this.
        .def_prop_ro("root", [](const Grid& g) { return g.root(); })
        .def_prop_ro("factor", &grid_factor)
        .def_prop_ro("max_level", &Grid::max_level)
        .def("level_cells_per_axis", &level_cells_per_axis, nb::arg("level"))
        .def("active_blocks", &active_blocks, nb::arg("level"))
        .def(
            "active_leaf_mask",
            [](const Grid& g, std::int64_t level) { return flat_mask(g.active_leaf_mask(level)); },
            nb::arg("level"))
        .def(
            "subdomain_mask",
            [](const Grid& g, std::int64_t level) { return flat_mask(g.subdomain_mask(level)); },
            nb::arg("level"))
        .def("cell_multi_index", &cell_multi_index, nb::arg("cid"))
        .def(
            "cell_id",
            [](const Grid& g, std::int64_t level, const_vec<std::int64_t> midx) {
                return g.cell_id(level, as_span(midx));
            },
            nb::arg("level"), nb::arg("midx").noconvert())
        //
        // `is_active_leaf` is deliberately NOT bound, on `query_aabb`'s reasoning below.
        // The wrapper owes a `None` for a wrongly sized `midx` (see the file header), so
        // it must route through `cell_id` in any case, and its `is_active_leaf` is then
        // the oracle's own one-line body -- one definition, which cannot drift from the
        // one the oracle runs. Binding this would add a member with no Python caller.
        .def("export_cells", &export_cells)

        // ------------------------------------------------------------------
        // Refinement and coarsening: each returns a new grid
        // ------------------------------------------------------------------
        .def(
            "refine",
            [](const Grid& g, std::int64_t level, const_vec<std::int64_t> lo,
               const_vec<std::int64_t> hi) { return g.refine(level, as_span(lo), as_span(hi)); },
            nb::arg("level"), nb::arg("lo").noconvert(), nb::arg("hi").noconvert())
        .def(
            "refine_cells",
            [](const Grid& g, const_vec<std::int64_t> cell_ids) {
                return g.refine_cells(as_span(cell_ids));
            },
            nb::arg("cell_ids").noconvert())
        .def(
            "coarsen",
            [](const Grid& g, std::int64_t level, const_vec<std::int64_t> lo,
               const_vec<std::int64_t> hi) { return g.coarsen(level, as_span(lo), as_span(hi)); },
            nb::arg("level"), nb::arg("lo").noconvert(), nb::arg("hi").noconvert())
        .def(
            "coarsen_cells",
            [](const Grid& g, const_vec<std::int64_t> cell_ids) {
                return g.coarsen_cells(as_span(cell_ids));
            },
            nb::arg("cell_ids").noconvert())

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
        .def("cell_level", &Grid::cell_level, nb::arg("cid"))
        .def("hanging_neighbors", &Grid::hanging_neighbors, nb::arg("cid"), nb::arg("lfid"))
        .def("boundary_facets",
             [](const Grid& g) {
                 std::vector<std::int64_t> rows = g.boundary_facets();
                 const std::size_t n = rows.size() / 2;
                 return owned_array<std::int64_t, 2>(std::move(rows), {n, std::size_t{2}});
             })
        .def(
            "locate_many",
            [](const Grid& g, const_mat<double> points) {
                std::vector<std::int64_t> out = g.locate_many(as_points(points));
                const std::size_t n = out.size();
                return owned_array<std::int64_t, 1>(std::move(out), {n});
            },
            nb::arg("points"))
        .def("collect_cell_bounds",
             [](const Grid& g) {
                 const auto rows = static_cast<std::size_t>(g.num_cells());
                 const auto cols = static_cast<std::size_t>(g.ndim());
                 std::vector<double> lo(rows * cols);
                 std::vector<double> hi(rows * cols);
                 g.collect_cell_bounds(pantr::span2d<double>(lo.data(), rows, cols),
                                       pantr::span2d<double>(hi.data(), rows, cols));
                 return nb::make_tuple(owned_array<double, 2>(std::move(lo), {rows, cols}),
                                       owned_array<double, 2>(std::move(hi), {rows, cols}));
             })
        .def("restrict", &restrict, nb::arg("cell_ids").noconvert())

        // ------------------------------------------------------------------
        // The generic defaults the grid inherits
        // ------------------------------------------------------------------
        .def_prop_ro("ndim", &Grid::ndim)
        .def_prop_ro("num_cells", &Grid::num_cells)
        .def("cell_aabb", &Grid::cell_aabb, nb::arg("cid"))
        .def("child_cells", &Grid::child_cells, nb::arg("cid"))
        .def("reference_map", &Grid::reference_map, nb::arg("cid"))
        .def("neighbors", &Grid::neighbors, nb::arg("cid"))
        .def("num_local_facets", &Grid::num_local_facets, nb::arg("cid"))
        .def("local_facet_axis_side", &Grid::local_facet_axis_side, nb::arg("cid"),
             nb::arg("lfid"))
        .def("local_facet_bounds", &local_facet_bounds, nb::arg("cid"), nb::arg("lfid"))
        .def("is_mesh_boundary_facet", &Grid::is_mesh_boundary_facet, nb::arg("cid"),
             nb::arg("lfid"))
        //
        // `query_aabb` is deliberately NOT bound, for `grid_types.cpp`'s reason:
        // `_GridWrapper.query_aabb` routes through `cell_bvh()`, which is what the oracle's
        // own default does and which keeps a single unpacking of an `AABB` into corner
        // arrays -- `pantr.grid.BVH`'s.

        // ------------------------------------------------------------------
        // Base-owned state: the three that must alias rather than copy
        // ------------------------------------------------------------------
        //
        // Same shapes and same reasons as `grid_types.cpp`. `cell_bvh` is a `.def()`
        // returning `T&`, where `automatic` resolves to `rv_policy::copy` and the returned
        // BVH would be a copy of the cache rather than the cache, so its
        // `reference_internal` is load-bearing. On the two registries the annotation is
        // redundant -- `def_prop_ro` already builds its getter with that policy -- and it
        // is kept and said to be redundant rather than dropped, because that is a
        // hardcoded default in a third-party header rather than a documented guarantee.
        .def_prop_ro("cell_tags", nb::overload_cast<>(&Grid::cell_tags),
                     nb::rv_policy::reference_internal)
        .def_prop_ro("facet_tags", nb::overload_cast<>(&Grid::facet_tags),
                     nb::rv_policy::reference_internal)
        .def("cell_bvh", &Grid::cell_bvh, nb::rv_policy::reference_internal)

        .def("__repr__", &Grid::to_string);
}
