/// \file
/// nanobind bindings for `pantr::bspline::Bspline`, the B-spline field.
///
/// ## Two registrations, because the storage format is part of the value
///
/// `Bspline32` and `Bspline64`, for the reason `bezier_type.cpp` and
/// `bspline_types.cpp` register two of each: the field stores whatever float dtype
/// it is handed, `pantr.bspline.Bspline.dtype` is public, `float32` fields are
/// exercised across the suite, and the class of the handle is the only thing left
/// to carry the format. `pantr.bspline._bspline._impl_class` picks between them.
///
/// The split is doubly forced here. `Bspline<T>` can hold only a
/// `BsplineSpace<T>`, so the width is already a property of the C++ type and one
/// Python name could not front both.
///
/// The dtype is not converted, and that *is* a check: without `.noconvert()`
/// nanobind silently casts, so `Bspline32(space32, a_float64_net)` would narrow the
/// caller's geometry and `Bspline64(space64, a_float32_net)` would widen it, both
/// without a word. Refusing the cast makes `_impl_class` picking the wrong class a
/// loud failure instead of a quiet change of precision.
///
/// ## What this file validates: nothing
///
/// `Bspline` and `ControlNet` validate their own arguments and throw
/// `std::invalid_argument`, which nanobind maps to `ValueError` preserving
/// `what()`. The messages are the oracle's character for character where the
/// oracle has one, which `tests/parity/test_bspline_type.py` compares.
///
/// Two refusals the wrapper still owes, and both are Python's calling convention
/// rather than validation: coercing an `ArrayLike` to a contiguous array, and the
/// `ValueError` for control points whose dtype is not the space's -- a type-kind
/// fact that `Bspline<T>` cannot even represent, since it holds only a
/// `BsplineSpace<T>`. The wrapper also owes the *order* of those two against the
/// coefficient-count check, because the oracle raises the count one first and only
/// the order decides which message a caller reads.
///
/// The rank of the control-point array is deliberately unconstrained: a field is a
/// curve, a surface or a volume, and a mis-ranked array is refused by `Bspline`
/// against the space's basis counts rather than by a caster talking about C++
/// types.
///
/// ## The space goes in as a handle and comes out as one
///
/// `space` is `design/bspline_ownership_lifetime.md`'s class **H**, so the C++ type
/// stores `std::shared_ptr<const BsplineSpace<T>>` and this binding both takes and
/// returns a copy of the handle. No `rv_policy` question arises, because ownership
/// travels in the return value rather than in an annotation. Three consequences,
/// each of them the failure of an alternative that note measured:
///
///  - The returned Python object is **identity-stable**: `nb_type_put` looks the
///    pointer up in `inst_c2p` before creating an instance, so
///    `field.space is the_space_handed_in` holds. That is what lets the Python
///    wrapper's `Bspline(space, cp).space is space` contract rest on something
///    rather than on the wrapper alone.
///  - `sys.getrefcount` on the field handle is **unchanged** by the access, and it stays
///    unchanged under a reversion to `reference_internal` as well, so the delta is
///    **not** the detector `design/bspline_ownership_lifetime.md` M2 offers for this
///    accessor. Measured on this binding: rebinding it to return a reference with
///    `rv_policy::reference_internal` leaves the delta at zero, the handed-out object
///    identical to the one passed in, and its value readable after the owner dies.
///    The reason is specific and worth carrying: a field's space always arrives *from
///    Python*, so it already has a live instance that `nb_type_put`'s `inst_c2p`
///    lookup finds, and no new instance is created for a keep-alive to be installed
///    on. M2's detector is live only where the nested object is built in C++ and has
///    no instance of its own -- `THBSplineSpace`'s `level_space` is such an accessor;
///    this one is not. **What decides this accessor is the C++ test**, which compares
///    addresses and outlives the owner: `cpp/tests/test_bspline_type.cpp`.
///  - The space **outlives its field**. Passing a handle in takes a reference on
///    its Python object (`nanobind/stl/shared_ptr.h`'s `py_deleter`), and a space
///    taken back out keeps its own value alive after the field is dropped.
///
/// ## `control_points` is a view, and read-only
///
/// A control net is the whole geometry, read on every evaluation, so copying it per
/// property access would make the natural spelling of every operation quadratic in
/// nothing. The array returned views the field's own storage with the field as its
/// owner, and is read-only because the scalar type is `const T` -- nanobind passes
/// `std::is_const_v<Scalar>` straight into the writeable flag. The `const` is what
/// stops a caller mutating a validated geometry from outside, which
/// `design/bspline_ownership_lifetime.md` records as a live defect of the oracle
/// rather than a hypothetical.
///
/// **The owner argument is belt-and-braces here rather than the repair**, which is F1
/// of that note applied to an array: `def_prop_ro` passes
/// `rv_policy::reference_internal` positionally ahead of the caller's arguments, so a
/// property getter already ties its return to `self`. Measured on this binding --
/// dropping the `self` argument leaves the array aliasing the same storage, still
/// read-only, and still valid after the field is dropped, so the two spellings are
/// indistinguishable from Python. It is written anyway, because it is the only
/// spelling that survives someone changing `def_prop_ro` to a plain `.def`, where the
/// default policy is `automatic` and the omission is a dangling view.
///
/// No sentinel address is needed for an empty net, unlike `bspline_types.cpp`'s
/// dimensionless domain block: a field has at least one basis function per
/// direction and at least rank 1, so `values()` is never empty and its `data()` is
/// never null.
///
/// ## `_ref` accessors are not bound
///
/// `space_ref()` borrows the space rather than copying its handle, which is what
/// saves the atomic pair inside a C++ loop, and it is deliberately absent below.
/// `tests/parity/test_bspline_binding_contract.py` asserts over the bound surface
/// that no method name ends in `_ref`, which is the only available check: there is
/// no `static_assert` for absence.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pantr/bspline/bspline.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A read-only, contiguous array of any rank as nanobind sees it.
template <class T>
using const_nd = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

/// Register one scalar type's `Bspline`.
///
/// \param m The extension module.
/// \param name The Python-visible class name, `Bspline32` or `Bspline64`.
template <class T>
void bind_bspline(nb::module_& m, const char* name) {
    using Field = pantr::bspline::Bspline<T>;
    using Space = pantr::bspline::BsplineSpace<T>;
    using Net = typename Field::net_type;

    nb::class_<Field>(m, name)
        .def(
            "__init__",
            [](Field* self, std::shared_ptr<const Space> space, const_nd<T> control_points,
               bool is_rational) {
                std::vector<std::size_t> shape(control_points.ndim());
                for (std::size_t d = 0; d < shape.size(); ++d) {
                    shape[d] = control_points.shape(d);
                }
                new (self) Field(
                    std::move(space),
                    Net(std::span<const T>(control_points.data(), control_points.size()),
                        std::span<const std::size_t>(shape)),
                    is_rational);
            },
            nb::arg("space"), nb::arg("control_points").noconvert(),
            nb::arg("is_rational") = false)
        // Class H. No policy: the value is a `shared_ptr`, so ownership travels in
        // the return value.
        .def_prop_ro("space", &Field::space)
        // Class A: a read-only view of the field's own storage, with the field as
        // the array's owner.
        .def_prop_ro("control_points",
                     [](nb::handle self) {
                         const Field& b = nb::cast<const Field&>(self);
                         const std::span<const std::size_t> shape = b.net().shape();
                         return nb::ndarray<nb::numpy, const T>(b.net().values().data(),
                                                                shape.size(), shape.data(), self);
                     })
        .def_prop_ro("is_rational", &Field::is_rational)
        .def_prop_ro("dim", &Field::dim)
        .def_prop_ro("rank", &Field::rank)
        // A tuple rather than a list or an array, because the oracle's `degree` is
        // a tuple and a caller comparing `b.degree == (2, 3)` must not start
        // failing under the C++ backend -- on an array the comparison is
        // elementwise and then has no truth value.
        .def_prop_ro("degree", [](const Field& b) {
            nb::list degrees;
            for (const std::int64_t p : b.degree()) {
                degrees.append(p);
            }
            return nb::tuple(degrees);
        });
}

}  // namespace

void register_bspline_type(nb::module_& m) {
    bind_bspline<float>(m, "Bspline32");
    bind_bspline<double>(m, "Bspline64");
}
