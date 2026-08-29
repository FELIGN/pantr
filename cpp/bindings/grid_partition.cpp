/// \file
/// nanobind bindings for `pantr::grid::Partition`.
///
/// Follows `geometry.cpp` and `grid_tags.cpp`: ownership moved to C++ under the
/// 2026-08-27 amendment to `design/cross_backend_types.md`, the Python class wraps
/// one, and this file validates nothing because the type validates itself and
/// throws `std::invalid_argument`, which nanobind surfaces as the `ValueError` the
/// oracle raises.
///
/// ## What the wrapper still does
///
/// One thing, and it is the same reason as everywhere else in this port: the
/// "``cell_owner`` must be a 1D integer array" check is a **dtype and rank** check,
/// and by the time the binding hands C++ a `std::span<const std::int32_t>` both
/// facts are gone. So the wrapper coerces -- including the narrowing cast to
/// `int32`, which is part of the contract rather than a detail -- and this file
/// receives an array that is already the right shape and type.
///
/// ## Two returns, two different lifetimes
///
/// `cell_owner` is a **zero-copy read-only view** whose owner is the Python object
/// holding the partition, so the array cannot outlive the storage it aliases. A
/// partition has no mutator, so nothing can move that storage under a live view.
/// The owner argument is load-bearing twice: without it nanobind silently *copies*
/// and returns a **writable** array, which would pass every value assertion and
/// quietly drop the oracle's read-only contract.
///
/// `owned_cells` is a **fresh owning array**, because the ids are computed rather
/// than stored, and the oracle returns a freshly allocated writeable array from
/// `numpy.flatnonzero`. Both halves of that -- fresh and writeable -- are
/// reproduced.
///
/// Read-only here means read-only against accident, not against malice: `ctypes`
/// clears the flag in two lines. What the flag buys is that ordinary code cannot
/// write into C++-owned storage by mistake.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pantr/grid/partition.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using pantr::grid::Partition;

/// A read-only, contiguous 1D `int32` array as nanobind sees it.
using const_owners = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A valid, never-dereferenced address for a zero-length view's data pointer.
///
/// `std::vector::data()` on an empty vector may return `nullptr`, which nanobind
/// reads as "no array" rather than as an empty one. A partition over zero cells is
/// legal -- the oracle's own tests build one -- so the view needs an address.
constexpr std::int32_t kEmptyOwner = 0;

}  // namespace

void register_grid_partition(nb::module_& m) {
    nb::class_<Partition>(m, "Partition")
        .def(
            "__init__",
            [](Partition* self, const_owners cell_owner, std::int64_t n_parts) {
                new (self) Partition(std::span<const std::int32_t>(cell_owner.data(),
                                                                   cell_owner.shape(0)),
                                     n_parts);
            },
            nb::arg("cell_owner"), nb::arg("n_parts"))
        .def_prop_ro(
            "cell_owner",
            [](nb::object self) {
                const std::span<const std::int32_t> owners =
                    nb::cast<const Partition&>(self).cell_owner();
                const std::int32_t* base = owners.empty() ? &kEmptyOwner : owners.data();
                return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                    base, {owners.size()}, self);
            })
        .def_prop_ro("n_parts", &Partition::n_parts)
        .def_prop_ro("n_cells", &Partition::n_cells)
        .def(
            "owned_cells",
            [](const Partition& p, std::int64_t rank) {
                auto* owned = new std::vector<std::int64_t>(p.owned_cells(rank));
                // A zero-capacity vector's `data()` may be null; one slot of
                // capacity gives the empty array a valid address without changing
                // its size. `kEmptyOwner` cannot serve here: it is `const`, and this
                // array is handed back writeable to match the oracle.
                if (owned->empty()) {
                    owned->reserve(1);
                }
                nb::capsule owner(owned, [](void* p_owned) noexcept {
                    delete static_cast<std::vector<std::int64_t>*>(p_owned);
                });
                return nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>(
                    owned->data(), {owned->size()}, owner);
            },
            nb::arg("rank"))
        .def("__repr__", &Partition::to_string);
}
