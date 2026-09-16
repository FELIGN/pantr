/// \file
/// nanobind bindings for `pantr::bspline::SpanwiseElementExtraction`.
///
/// ## Two registrations per type, for the reason `bspline_types.cpp` gives
///
/// `SpanwiseElementExtraction<T>` can hold only a `BsplineSpace<T>` and stores its
/// operators in `T`, so the width is already a property of the C++ type and one
/// Python name could not front both. `SpanwiseElementExtraction32` and
/// `SpanwiseElementExtraction64` are the two, and
/// `pantr.bspline.spanwise_element_extraction._impl_class` picks between them.
///
/// The dtype is not converted, and that *is* a check: without `.noconvert()` nanobind
/// would silently cast a `float64` operator block into a `float32` extraction, which
/// changes every value the type hands out and would surface as a parity failure
/// attributed to the builders. `nb::arg(...).noconvert()` propagates through the
/// `std::vector` caster to the element casters, which
/// `tests/parity/test_spanwise_binding_contract.py` asserts rather than assumes.
///
/// ## The operators arrive built
///
/// `__init__` takes the per-direction *dense* operator blocks and identity masks
/// rather than building them, which is the type's own seam --
/// `pantr/bspline/spanwise_extraction.hpp` states why, and the short version is that
/// slice S3 owns the builders and the cardinal target has none in C++. So this file
/// binds a compaction, not an extraction.
///
/// ## What this file validates: the array *shapes* it cannot express in a signature
///
/// The type validates its own arguments and throws `std::invalid_argument`, which
/// nanobind maps to `ValueError` preserving `what()`. What the type cannot see is
/// the rank and contiguity of the incoming arrays, so those are in the signature:
/// `nb::ndim<3>` and `nb::c_contig` are what turn a strided or mis-shaped operator
/// block into a `TypeError` at the call rather than a silent reinterpretation.
///
/// Three refusals the oracle makes stay in the Python wrapper, for the reasons
/// `spanwise_extraction.hpp` lists: the mixed-dtype `ValueError`, the
/// `NotImplementedError` for a periodic direction, and the `ValueError` naming an
/// unrecognised target spelling. `pantr/core/error.hpp` records that nanobind has no
/// path from a `std::exception` to `NotImplementedError`, and the other two are
/// type-kind checks.
///
/// ## Arrays out: views, never copies
///
/// Every array here is storage the extraction owns and computed once -- the compact
/// operators and the index maps at construction, the dense block behind the memo in
/// `pantr/core/lazy.hpp`. They go out as `nb::ndarray` views with the extraction as
/// the array's owner, which is `bezier_type.cpp`'s idiom and for its reasons: the
/// owner is what keeps the storage alive when the array outlives the handle it came
/// from, and `const T` is what nanobind turns into the read-only flag.
///
/// Copying would not merely be wasteful here, it would break a contract.
/// `pantr.bspline.make_struct_view` bundles these three arrays by reference into the
/// `NamedTuple` a downstream `@njit` function unboxes, and
/// `tests/test_spanwise_element_extraction.py::test_factors_zero_copy_and_readonly`
/// asserts `np.shares_memory` against them.
///
/// **No array this type hands out can be empty.** A compact block has at least one
/// row -- the sentinel the oracle keeps for exactly this -- and at least one row and
/// column per operator, and `BsplineSpace1D` refuses a space with no interval, so an
/// index map and a mask have at least one entry. A dimensionless space has no
/// directions at all, so the tuples are empty and no array is built. That is why
/// there is no `kEmptyStorage` sentinel here and `bspline_types.cpp` needs one.
///
/// ## `_ref` accessors are not bound
///
/// `space_ref` borrows the space rather than copying its handle, which is what saves
/// the atomic pair inside a C++ loop, and it is deliberately absent from the surface
/// below. `tests/parity/test_spanwise_binding_contract.py` asserts over the bound
/// surface that no method name ends in `_ref`.
///
/// ## Handing out the space: a `shared_ptr`, and no policy
///
/// `space` is `design/bspline_ownership_lifetime.md`'s class **H**, so the C++ type
/// stores `std::shared_ptr<const BsplineSpace<T>>` and the binding returns a copy of
/// the handle. No `rv_policy` question arises, because ownership travels in the
/// return value. The returned Python object is identity-stable for a space that came
/// *from* Python, since `nb_type_put` finds its existing instance in `inst_c2p` --
/// which is what lets the wrapper's `extraction.space is space` contract hold under
/// both backends.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bspline/space_nd.hpp"
#include "pantr/bspline/spanwise_extraction.hpp"
#include "pantr/core/mdspan.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using pantr::span_nd;
using pantr::bspline::BsplineSpace;
using pantr::bspline::ExtractionTarget;
using pantr::bspline::SpanwiseElementExtraction;

/// A read-only, contiguous `(n_elements, n_out, n_in)` operator block as nanobind
/// sees it.
template <class T>
using const_ops = nb::ndarray<const T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous per-element identity mask as nanobind sees it.
///
/// `numpy.bool_` is one byte and so is C++ `bool`, which is what makes a view of the
/// type's own flags a `bool` array on the Python side rather than a reinterpretation.
using const_mask = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Hand out a 3D span of the owner's storage as a read-only numpy view.
///
/// Local to this file rather than shared, which is the convention `bezier_type.cpp`
/// and `bspline_types.cpp` already follow: each binding file carries the view helper
/// its own arrays need.
///
/// \tparam T The element type.
/// \param self The Python object that owns the storage, kept alive by the array.
/// \param data The span; never empty, see the file comment.
/// \return A read-only 3D `numpy` array viewing `data`.
template <class T>
nb::object view_3d_of(nb::handle self, span_nd<const T, 3> data) {
    return nb::cast(nb::ndarray<nb::numpy, const T, nb::ndim<3>>(
        data.data_handle(), {data.extent(0), data.extent(1), data.extent(2)}, self));
}

/// Hand out a 1D span of the owner's storage as a read-only numpy view.
///
/// \tparam T The element type.
/// \param self The Python object that owns the storage, kept alive by the array.
/// \param data The span; never empty, see the file comment.
/// \return A read-only 1D `numpy` array viewing `data`.
template <class T>
nb::object view_of(nb::handle self, std::span<const T> data) {
    return nb::cast(nb::ndarray<nb::numpy, const T, nb::ndim<1>>(data.data(), {data.size()}, self));
}

/// Hand out a span of counts as a tuple of Python integers.
///
/// A tuple rather than an array, for `bspline_types.cpp`'s reason: the oracle's
/// `num_intervals`, `input_shape_per_dir` and `output_shape_per_dir` are tuples and
/// the suite compares them with `==` against one.
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

/// Build the constructor's per-direction bundles from the two incoming sequences.
///
/// \tparam T The scalar type.
/// \param operators One dense `(n_elements, n_out, n_in)` block per direction.
/// \param masks One `(n_elements,)` identity mask per direction.
/// \return One bundle per direction, viewing the caller's arrays. The type copies
///         what it keeps, so the views need only outlive the constructor.
/// \throws std::invalid_argument If the two sequences differ in length.
template <class T>
std::vector<typename SpanwiseElementExtraction<T>::DirectionInput> bundles(
    const std::vector<const_ops<T>>& operators, const std::vector<const_mask>& masks) {
    using Input = typename SpanwiseElementExtraction<T>::DirectionInput;
    if (operators.size() != masks.size()) {
        throw std::invalid_argument("got " + std::to_string(operators.size())
                                    + " operator blocks and " + std::to_string(masks.size())
                                    + " identity masks; there must be one of each per direction");
    }
    std::vector<Input> inputs;
    inputs.reserve(operators.size());
    for (std::size_t d = 0; d < operators.size(); ++d) {
        inputs.push_back(Input{span_nd<const T, 3>(operators[d].data(), operators[d].shape(0),
                                                   operators[d].shape(1), operators[d].shape(2)),
                               std::span<const bool>(masks[d].data(), masks[d].shape(0))});
    }
    return inputs;
}

/// Register one scalar type's `SpanwiseElementExtraction`.
///
/// \tparam T The scalar type.
/// \param m The extension module.
/// \param name The Python-visible class name.
template <class T>
void bind_spanwise_extraction(nb::module_& m, const char* name) {
    using Extraction = SpanwiseElementExtraction<T>;
    using Space = BsplineSpace<T>;

    nb::class_<Extraction>(m, name)
        // The space arrives as a handle, which is what makes the C++ extraction SHARE
        // it; see the file comment. The target arrives as the `ExtractionTarget`
        // IntEnum's own integer and the variant as the `LagrangeVariant` StrEnum's own
        // string: neither is a bound enum, because there is no bound enum anywhere in
        // this extension.
        .def(
            "__init__",
            [](Extraction* self, std::shared_ptr<const Space> space, std::int64_t target,
               std::string lagrange_variant, const std::vector<const_ops<T>>& operators,
               const std::vector<const_mask>& masks) {
                if (!pantr::bspline::is_extraction_target(target)) {
                    throw std::invalid_argument("target must be one of 0, 1, 2; got "
                                                + std::to_string(target));
                }
                const std::vector<typename Extraction::DirectionInput> inputs =
                    bundles<T>(operators, masks);
                new (self) Extraction(std::move(space), static_cast<ExtractionTarget>(target),
                                      std::move(lagrange_variant),
                                      std::span<const typename Extraction::DirectionInput>(inputs));
            },
            nb::arg("space"), nb::arg("target"), nb::arg("lagrange_variant"),
            nb::arg("operators").noconvert(), nb::arg("masks").noconvert())
        // Class H. No policy: the value is a `shared_ptr`, so ownership travels in the
        // return value.
        .def_prop_ro("space", [](const Extraction& e) { return e.space(); })
        .def_prop_ro("target",
                     [](const Extraction& e) { return static_cast<std::int64_t>(e.target()); })
        .def_prop_ro("lagrange_variant", [](const Extraction& e) { return e.lagrange_variant(); })
        .def_prop_ro("dim", [](const Extraction& e) { return e.dim(); })
        .def_prop_ro("num_intervals",
                     [](const Extraction& e) { return counts_tuple(e.num_intervals()); })
        .def_prop_ro("num_total_intervals",
                     [](const Extraction& e) { return e.num_total_intervals(); })
        // Class A, three times over: a read-only view of the extraction's own storage
        // with the extraction as the array's owner. `const T` sets the read-only flag,
        // `self` keeps the storage alive, and the tuple is the shape the oracle's
        // same-named properties have.
        .def_prop_ro("compact_ops_1d",
                     [](nb::handle self) {
                         const Extraction& e = nb::cast<const Extraction&>(self);
                         nb::list blocks;
                         for (std::int64_t d = 0; d < e.dim(); ++d) {
                             blocks.append(view_3d_of<T>(self, e.compact_ops(d)));
                         }
                         return nb::tuple(blocks);
                     })
        .def_prop_ro("idx_maps_1d",
                     [](nb::handle self) {
                         const Extraction& e = nb::cast<const Extraction&>(self);
                         nb::list maps;
                         for (std::int64_t d = 0; d < e.dim(); ++d) {
                             maps.append(view_of<std::int64_t>(self, e.idx_map(d)));
                         }
                         return nb::tuple(maps);
                     })
        .def_prop_ro("is_identity_mask_1d",
                     [](nb::handle self) {
                         const Extraction& e = nb::cast<const Extraction&>(self);
                         nb::list masks;
                         for (std::int64_t d = 0; d < e.dim(); ++d) {
                             masks.append(view_of<bool>(self, e.is_identity_mask(d)));
                         }
                         return nb::tuple(masks);
                     })
        // Class A over the memo rather than over a constructed block: the first read
        // fills it, every read views it, which is what makes the oracle's
        // `np.shares_memory` assertion hold here too.
        .def_prop_ro("ops_1d",
                     [](nb::handle self) {
                         const Extraction& e = nb::cast<const Extraction&>(self);
                         nb::list blocks;
                         for (std::int64_t d = 0; d < e.dim(); ++d) {
                             blocks.append(view_3d_of<T>(self, e.ops_1d(d)));
                         }
                         return nb::tuple(blocks);
                     })
        .def_prop_ro("input_shape_per_dir",
                     [](const Extraction& e) { return counts_tuple(e.input_shape_per_dir()); })
        .def_prop_ro("output_shape_per_dir",
                     [](const Extraction& e) { return counts_tuple(e.output_shape_per_dir()); })
        .def_prop_ro("num_identity_elements",
                     [](const Extraction& e) { return e.num_identity_elements(); })
        .def_prop_ro("is_identity", [](const Extraction& e) { return e.is_identity(); });
}

}  // namespace

void register_bspline_spanwise_extraction(nb::module_& m) {
    bind_spanwise_extraction<float>(m, "SpanwiseElementExtraction32");
    bind_spanwise_extraction<double>(m, "SpanwiseElementExtraction64");
}
