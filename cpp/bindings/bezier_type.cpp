/// \file
/// nanobind bindings for `pantr::bezier::Bezier`.
///
/// ## Two registrations, not one
///
/// `AABB` and `AffineTransform` are bound at `double` only, because their Python
/// oracles coerce everything to `float64` and a `float` registration would be a
/// surface with no oracle behind it. `pantr.bezier.Bezier` is different in exactly
/// that respect: it stores whatever float dtype it is handed, its `dtype` property
/// is public, `tests/test_bezier.py::TestBezierInit::test_float32` pins the
/// behaviour, and `tools/adversarial_sweep/_probes_bezier.py` sweeps both formats.
/// So both are registered, as `Bezier32` and `Bezier64`, and
/// `pantr.bezier._impl_class` picks between them by dtype.
///
/// The two names carry the storage format because there is nothing else to carry
/// it: the class of the handle *is* the dtype, and a single `Bezier` name would
/// have to smuggle the format through a constructor argument that could then
/// disagree with the array it was given.
///
/// ## What this file validates, and what it does not
///
/// Nothing, beyond what nanobind's typed parameters settle for it -- dtype,
/// C-contiguity and device. `Bezier` and `ControlNet` validate their own arguments
/// and throw `std::invalid_argument`, which nanobind surfaces as `ValueError`, the
/// exception the oracle raises for the same inputs. Duplicating the checks here
/// would put a second copy of the contract in a second language, which is the
/// shape this port exists to avoid.
///
/// The rank of the array is deliberately unconstrained. A Bézier is a curve, a
/// surface or a volume, so its control points have any rank from 1 upward, and a
/// rank-0 array is rejected by `ControlNet` with the oracle's own message rather
/// than by a caster with a message about C++ types.
///
/// The dtype is not converted, and that *is* a check. Without `.noconvert()`
/// nanobind silently casts, so `Bezier32(a_float64_array)` would narrow the
/// caller's geometry and `Bezier64(a_float32_array)` would widen it, both without
/// a word -- and the class name is the only thing carrying the storage format, so
/// there would be nothing left to notice with. Refusing the cast makes
/// `pantr.bezier._impl_class` picking the wrong class a loud failure instead of a
/// quiet change of precision. The cost is that the caller owes a C-contiguous
/// array of the right dtype, which the wrapper already produces.
///
/// ## `control_points` is a view, and read-only
///
/// The two merged ports copy their corner arrays out, because a corner is a
/// handful of doubles. A control net is not: it is the whole geometry, read on
/// every evaluation, and copying it per property access would make the natural
/// spelling of every operation quadratic in nothing.
///
/// So the array returned here views the Bézier's own storage, with the Bézier as
/// its owner, and is read-only because the scalar type is `const T` -- nanobind
/// passes `std::is_const_v<Scalar>` straight into the array's writeable flag. Both
/// halves are load-bearing. The owner is what keeps the storage alive when the
/// array outlives the handle it came from; the read-only flag is what stops a
/// caller mutating a validated geometry from the outside, which is the half of
/// FELIGN/pantr#338's defect that lives on the way *out*.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <span>
#include <vector>

#include "pantr/bezier/bezier.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A read-only, contiguous array of any rank as nanobind sees it.
template <class T>
using const_nd = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

/// Register one scalar type's `Bezier`.
///
/// \param m The extension module.
/// \param name The Python-visible class name, `Bezier32` or `Bezier64`.
template <class T>
void bind_bezier(nb::module_& m, const char* name) {
    using Bez = pantr::bezier::Bezier<T>;
    using Net = pantr::bezier::ControlNet<T>;

    nb::class_<Bez>(m, name)
        .def(
            "__init__",
            [](Bez* self, const_nd<T> control_points, bool is_rational) {
                std::vector<std::size_t> shape(control_points.ndim());
                for (std::size_t d = 0; d < shape.size(); ++d) {
                    shape[d] = control_points.shape(d);
                }
                new (self) Bez(Net(std::span<const T>(control_points.data(),
                                                      control_points.size()),
                                   std::span<const std::size_t>(shape)),
                               is_rational);
            },
            nb::arg("control_points").noconvert(), nb::arg("is_rational") = false)
        .def_prop_ro("control_points",
                     [](nb::handle self) {
                         const Bez& b = nb::cast<const Bez&>(self);
                         const std::span<const std::size_t> shape = b.net().shape();
                         return nb::ndarray<nb::numpy, const T>(b.net().values().data(),
                                                                shape.size(), shape.data(), self);
                     })
        .def_prop_ro("is_rational", &Bez::is_rational)
        .def_prop_ro("dim", &Bez::dim)
        .def_prop_ro("rank", &Bez::rank)
        // A tuple rather than a list, because the oracle's `degree` is a tuple and
        // a caller comparing `b.degree == (2, 3)` must not start failing under the
        // C++ backend.
        .def_prop_ro("degree", [](const Bez& b) {
            nb::list degrees;
            for (const std::size_t p : b.degree()) {
                degrees.append(p);
            }
            return nb::tuple(degrees);
        });
}

}  // namespace

void register_bezier_type(nb::module_& m) {
    bind_bezier<float>(m, "Bezier32");
    bind_bezier<double>(m, "Bezier64");
}
