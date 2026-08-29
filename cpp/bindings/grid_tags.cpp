/// \file
/// nanobind bindings for `pantr::grid::CellTags` and `pantr::grid::FacetTags`.
///
/// The third and fourth types this extension exposes, after `AABB` and
/// `AffineTransform`, and they follow those files' reasoning: ownership moved to
/// C++ under the 2026-08-27 amendment to `design/cross_backend_types.md`, the
/// Python classes wrap one, and this file validates nothing because the types
/// validate themselves.
///
/// ## What the wrapper still does, and it is not laziness
///
/// Three things stay on the Python side, each because nanobind cannot express
/// them rather than because they were easier there:
///
///  - **`KeyError` for an unregistered name.** `pantr/core/error.hpp` records the
///    rule: nanobind 2.14.0 maps `std::out_of_range` to `IndexError` and has no
///    path to `KeyError`. The wrapper checks `__contains__` and raises it itself,
///    which keeps the message and costs one dictionary-sized lookup.
///  - **`TypeError` for a non-integer `ids`, `values` or destination `dtype`.**
///    Same reason: there is no `std::exception` nanobind turns into a `TypeError`.
///  - **Allocating and filling `to_dense`'s destination.** `numpy.full` is what
///    decides what an out-of-range `fill` does, and reproducing its `OverflowError`
///    message by hand would be a second spelling of numpy's contract. So the
///    wrapper allocates and fills, and `scatter` below writes only the tagged
///    entries -- the project's ordinary `out` convention, reached for a reason.
///
/// ## Zero-copy views, and what keeps them alive
///
/// `get` hands back views into the registry's own storage rather than copies,
/// unlike `geometry.cpp`, whose corners are `ndim` doubles and cheap to copy. A
/// tag is as long as the tagged subset of the grid, and the oracle hands back its
/// stored arrays without copying, so copying here would be a performance
/// difference the backend switch must not introduce.
///
/// That makes lifetime the whole question, and `tags.hpp` answers it: each tag is
/// held behind a `std::shared_ptr<const Tag>`, and every view below carries a
/// capsule holding its own copy of that pointer. A later `set` on the same name
/// replaces the registry's pointer and leaves the view's storage alive.
///
/// The owner is load-bearing twice over. It is what keeps the storage alive, and
/// it is also what makes the array read-only: an ownerless `nb::ndarray` return
/// silently *copies* and comes back writable, which would pass a value assertion
/// and fail the oracle's read-only contract. `node_lo.base is not None` is the
/// half of that pair a test can see.
///
/// ## Eight `scatter` overloads
///
/// The oracle's `to_dense(dtype=...)` accepts any of numpy's eight integer
/// dtypes, so the destination type is genuinely part of the surface. Each overload
/// carries `.noconvert()`, without which nanobind would happily convert an `int32`
/// destination into an `int64` temporary, write the scatter into it and discard it
/// -- a silent no-op, which is the failure `.noconvert()` exists to prevent
/// throughout this directory.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/grid/tags.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using pantr::grid::CellTags;
using pantr::grid::FacetTags;

/// A read-only, contiguous 1D `int64` array as nanobind sees it.
using const_ids = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2D `int64` array as nanobind sees it.
using const_keys = nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A writeable, contiguous 1D destination as nanobind sees it.
template <class T>
using out_vec = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A writeable, contiguous 2D destination as nanobind sees it.
template <class T>
using out_mat = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// View a 1D nanobind array as a span.
///
/// \param a The array.
/// \return A span over its storage, valid while the array is alive.
std::span<const std::int64_t> as_span(const_ids a) {
    return {a.data(), a.shape(0)};
}

/// View a 2D nanobind array as a row-major 2D span.
///
/// \param a The array.
/// \return A view over its storage.
pantr::span2d<const std::int64_t> as_span2d(const_keys a) {
    return pantr::span2d<const std::int64_t>(a.data(), a.shape(0), a.shape(1));
}

/// Expose one of a tag's arrays as a read-only numpy view that owns its lifetime.
///
/// \tparam Tag The registry's tag type.
/// \param tag The shared handle to keep alive; a copy is stored in the capsule.
/// \param values The array inside `tag` to view.
/// \param shape The view's shape, one entry per dimension.
/// \return An `int64` array, read-only, aliasing the tag's storage.
template <class Tag, std::size_t Rank>
nb::object tag_view(std::shared_ptr<const Tag> tag, const std::vector<std::int64_t>& values,
                    std::array<std::size_t, Rank> shape) {
    // An empty vector's `data()` may be null, and nanobind reads a null pointer as
    // "no array" rather than as an empty one. An empty tag is legal -- the oracle's
    // own tests set one -- so a valid, never-dereferenced address stands in.
    const std::int64_t* base =
        values.empty() ? &pantr::grid::detail::kEmptyStorage : values.data();
    auto* keep = new std::shared_ptr<const Tag>(std::move(tag));
    nb::capsule owner(keep, [](void* p) noexcept {
        delete static_cast<std::shared_ptr<const Tag>*>(p);
    });
    // The pointer-and-rank constructor rather than the brace-list one: the shape
    // arrives as a `std::array` because the rank is a template parameter here, and
    // nanobind's brace-list overload takes an `initializer_list` a runtime array
    // cannot become.
    return nb::cast(nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<Rank>>(
        base, Rank, shape.data(), owner));
}

/// Register one `scatter` overload on the cell registry.
///
/// \tparam IntT The destination integer type.
/// \param cls The class being built.
template <class IntT>
void bind_cell_scatter(nb::class_<CellTags>& cls) {
    cls.def(
        "scatter",
        [](const CellTags& tags, const std::string& name, out_vec<IntT> out) {
            tags.scatter<IntT>(name, std::span<IntT>(out.data(), out.shape(0)));
        },
        nb::arg("name"), nb::arg("out").noconvert());
}

/// Register one `scatter` overload on the facet registry.
///
/// \tparam IntT The destination integer type.
/// \param cls The class being built.
template <class IntT>
void bind_facet_scatter(nb::class_<FacetTags>& cls) {
    cls.def(
        "scatter",
        [](const FacetTags& tags, const std::string& name, out_mat<IntT> out) {
            tags.scatter<IntT>(name,
                               pantr::span2d<IntT>(out.data(), out.shape(0), out.shape(1)));
        },
        nb::arg("name"), nb::arg("out").noconvert());
}

/// Register the `scatter` overload set on both registries.
///
/// One fold rather than sixteen `def` lines, so that adding a destination type
/// cannot be done for one registry and forgotten for the other.
///
/// \param cells The cell registry class.
/// \param facets The facet registry class.
void bind_scatter_overloads(nb::class_<CellTags>& cells, nb::class_<FacetTags>& facets) {
    const auto bind_one = [&cells, &facets]<class IntT>() {
        bind_cell_scatter<IntT>(cells);
        bind_facet_scatter<IntT>(facets);
    };
    bind_one.template operator()<std::int8_t>();
    bind_one.template operator()<std::int16_t>();
    bind_one.template operator()<std::int32_t>();
    bind_one.template operator()<std::int64_t>();
    bind_one.template operator()<std::uint8_t>();
    bind_one.template operator()<std::uint16_t>();
    bind_one.template operator()<std::uint32_t>();
    bind_one.template operator()<std::uint64_t>();
}

}  // namespace

void register_grid_tags(nb::module_& m) {
    nb::class_<CellTags> cells(m, "CellTags");
    cells.def(nb::init<std::int64_t>(), nb::arg("num_cells"))
        .def_prop_ro("num_cells", &CellTags::num_cells)
        .def_prop_ro("names", &CellTags::names)
        .def("__len__", &CellTags::size)
        .def(
            "__contains__",
            [](const CellTags& tags, const std::string& name) { return tags.contains(name); },
            nb::arg("name"))
        .def(
            "set",
            [](CellTags& tags, const std::string& name, const_ids ids, const_ids values) {
                tags.set(name, as_span(ids), as_span(values));
            },
            nb::arg("name"), nb::arg("ids"), nb::arg("values"))
        .def(
            "get",
            [](const CellTags& tags, const std::string& name) {
                auto tag = tags.get(name);
                const std::size_t n = tag->ids.size();
                return nb::make_tuple(tag_view<CellTags::Tag, 1>(tag, tag->ids, {n}),
                                      tag_view<CellTags::Tag, 1>(tag, tag->values, {n}));
            },
            nb::arg("name"))
        .def(
            "remove", [](CellTags& tags, const std::string& name) { tags.remove(name); },
            nb::arg("name"))
        .def("__repr__", &CellTags::to_string);

    nb::class_<FacetTags> facets(m, "FacetTags");
    facets.def(nb::init<std::int64_t, std::int64_t>(), nb::arg("num_cells"),
               nb::arg("facets_per_cell"))
        .def_prop_ro("num_cells", &FacetTags::num_cells)
        .def_prop_ro("facets_per_cell", &FacetTags::facets_per_cell)
        .def_prop_ro("names", &FacetTags::names)
        .def("__len__", &FacetTags::size)
        .def(
            "__contains__",
            [](const FacetTags& tags, const std::string& name) { return tags.contains(name); },
            nb::arg("name"))
        .def(
            "set",
            [](FacetTags& tags, const std::string& name, const_keys keys, const_ids values) {
                tags.set(name, as_span2d(keys), as_span(values));
            },
            nb::arg("name"), nb::arg("keys"), nb::arg("values"))
        .def(
            "get",
            [](const FacetTags& tags, const std::string& name) {
                auto tag = tags.get(name);
                const std::size_t rows = tag->rows();
                return nb::make_tuple(
                    tag_view<FacetTags::Tag, 2>(tag, tag->keys,
                                                {rows, pantr::grid::detail::kFacetKeyWidth}),
                    tag_view<FacetTags::Tag, 1>(tag, tag->values, {rows}));
            },
            nb::arg("name"))
        .def(
            "remove", [](FacetTags& tags, const std::string& name) { tags.remove(name); },
            nb::arg("name"))
        .def("__repr__", &FacetTags::to_string);

    bind_scatter_overloads(cells, facets);
}
