/// \file
/// nanobind bindings for `pantr::grid::BVH`.
///
/// Follows `geometry.cpp`, `grid_tags.cpp` and `grid_partition.cpp`: ownership
/// moved to C++ under the 2026-08-27 amendment to
/// `design/cross_backend_types.md`, the Python class wraps one, and this file
/// validates nothing because the type validates itself.
///
/// ## `double` only
///
/// The oracle is `float64` throughout -- none of the three grid kernel modules
/// mentions `float32` -- so registering a `float` tree would create a surface with
/// no oracle behind it, which `design/backend_parity.md` Rule 8 forbids. The header
/// stays templated on `Real` and `cpp/tests/test_grid_bvh_type.cpp` instantiates
/// both.
///
/// ## What the wrapper still does
///
/// The **dtype** checks. `pantr.grid.BVH.__init__` promises `TypeError` for a
/// non-`float64` corner array or a non-`int64` index array, and nanobind has no
/// path producing a `TypeError` from an exception -- it would raise its own, naming
/// this C++ signature rather than the argument. The wrapper also owns the array
/// **shape** messages, because the oracle words them in terms of a numpy shape
/// tuple that is gone once the binding has a span, and it translates
/// `CapacityError` back to the `ValueError` the pre-port class raised, so that
/// `PANTR_BACKEND` does not decide which exception a caller catches.
///
/// ## The five arrays are views, and the owner is load-bearing
///
/// `design/bvh.md` establishes the node arrays as public API, and the oracle
/// returns its stored arrays directly with `flags.writeable` cleared. So these are
/// zero-copy views whose owner is the Python object holding the tree -- copying
/// would make the backend switch a performance switch on the one surface a
/// consumer is expected to traverse itself.
///
/// Two details are easy to get wrong and both are load-bearing. An **ownerless**
/// `nb::ndarray` return silently *copies* and comes back **writable**, so the owner
/// is what buys read-only-ness and not just lifetime; `node_lo.base is not None` is
/// the half of that a test can see. And read-only is enforced against accident, not
/// against malice -- `ctypes` clears the flag in two lines -- so nothing here
/// claims that C++-owned memory cannot be written. `bvh_tree.hpp` records what that
/// costs under issue #359: on this backend a corrupted index is undefined
/// behaviour where the numba kernel merely returns a defined wrong answer.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/grid/bvh_tree.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using Tree = pantr::grid::BVH<double>;

/// A read-only, contiguous 1D `float64` array as nanobind sees it.
using const_vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2D `float64` array as nanobind sees it.
using const_mat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 1D `int64` array as nanobind sees it.
using const_idx = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A valid, never-dereferenced address for a zero-length view's data pointer.
///
/// `std::vector::data()` on an empty vector may return `nullptr`, which nanobind
/// reads as "no array" rather than as an empty one. A zero-cell tree is legal and
/// has empty node arrays, so the views need an address.
constexpr double kEmptyCoord = 0.0;

/// The same, for the three index arrays.
constexpr std::int64_t kEmptyIndex = 0;

/// View a 1D nanobind array as a span.
///
/// \param a The array.
/// \return A span over its storage.
std::span<const double> as_span(const_vec a) {
    return {a.data(), a.shape(0)};
}

/// View a 2D nanobind array as a row-major 2D span.
///
/// \param a The array.
/// \return A view over its storage.
pantr::span2d<const double> as_span2d(const_mat a) {
    return pantr::span2d<const double>(a.data(), a.shape(0), a.shape(1));
}

/// Expose one of the two `(n_nodes, ndim)` coordinate arrays as a read-only view.
///
/// \param self The Python object holding the tree; becomes the array's owner.
/// \param block The tree's own storage for that array.
/// \return A `float64` array of shape `(n_nodes, ndim)`, read-only, aliasing.
nb::object coordinate_view(nb::object self, pantr::span2d<const double> block) {
    const double* base = block.size() == 0 ? &kEmptyCoord : block.data_handle();
    return nb::cast(nb::ndarray<nb::numpy, const double, nb::ndim<2>>(
        base, {block.extent(0), block.extent(1)}, self));
}

/// Expose one of the three per-node index arrays as a read-only view.
///
/// \param self The Python object holding the tree; becomes the array's owner.
/// \param block The tree's own storage for that array.
/// \return An `int64` array of shape `(n_nodes,)`, read-only, aliasing.
nb::object index_view(nb::object self, std::span<const std::int64_t> block) {
    const std::int64_t* base = block.empty() ? &kEmptyIndex : block.data();
    return nb::cast(
        nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>>(base, {block.size()}, self));
}

}  // namespace

void register_grid_bvh(nb::module_& m) {
    nb::class_<Tree>(m, "BVH")
        .def(
            "__init__",
            [](Tree* self, const_mat node_lo, const_mat node_hi, const_idx node_left,
               const_idx node_right, const_idx node_cell, std::int64_t n_cells) {
                new (self) Tree(as_span2d(node_lo), as_span2d(node_hi),
                                std::span<const std::int64_t>(node_left.data(),
                                                              node_left.shape(0)),
                                std::span<const std::int64_t>(node_right.data(),
                                                              node_right.shape(0)),
                                std::span<const std::int64_t>(node_cell.data(),
                                                              node_cell.shape(0)),
                                n_cells);
            },
            nb::arg("node_lo"), nb::arg("node_hi"), nb::arg("node_left"), nb::arg("node_right"),
            nb::arg("node_cell"), nb::arg("n_cells"))
        .def_static(
            "from_cell_bounds",
            [](const_mat cell_lo, const_mat cell_hi) {
                return Tree::from_cell_bounds(as_span2d(cell_lo), as_span2d(cell_hi));
            },
            nb::arg("cell_lo"), nb::arg("cell_hi"))
        .def_prop_ro("ndim", &Tree::ndim)
        .def_prop_ro("n_cells", &Tree::n_cells)
        .def_prop_ro("n_nodes", &Tree::n_nodes)
        .def_prop_ro("node_lo",
                     [](nb::object self) {
                         return coordinate_view(self, nb::cast<const Tree&>(self).node_lo());
                     })
        .def_prop_ro("node_hi",
                     [](nb::object self) {
                         return coordinate_view(self, nb::cast<const Tree&>(self).node_hi());
                     })
        .def_prop_ro("node_left",
                     [](nb::object self) {
                         return index_view(self, nb::cast<const Tree&>(self).node_left());
                     })
        .def_prop_ro("node_right",
                     [](nb::object self) {
                         return index_view(self, nb::cast<const Tree&>(self).node_right());
                     })
        .def_prop_ro("node_cell",
                     [](nb::object self) {
                         return index_view(self, nb::cast<const Tree&>(self).node_cell());
                     })
        .def(
            "query_aabb",
            [](const Tree& tree, const_vec qlo, const_vec qhi) {
                // Moved rather than copied: `query_aabb` returns a fresh vector, and
                // a query on a large tree can match many cells.
                auto* owned =
                    new std::vector<std::int64_t>(tree.query_aabb(as_span(qlo), as_span(qhi)));
                // A zero-capacity vector's `data()` may be null; one slot of
                // capacity gives the empty array a valid address without changing
                // its size. The named `kEmpty*` sentinels the read-only views use
                // are `const` and cannot serve here, because this array is handed
                // back writeable.
                if (owned->empty()) {
                    owned->reserve(1);
                }
                nb::capsule owner(owned, [](void* p) noexcept {
                    delete static_cast<std::vector<std::int64_t>*>(p);
                });
                // Freshly allocated and writeable, matching the oracle, which
                // returns a `numpy.empty` it filled.
                return nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>(
                    owned->data(), {owned->size()}, owner);
            },
            nb::arg("qlo"), nb::arg("qhi"));
}
