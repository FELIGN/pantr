/// \file
/// nanobind bindings for `pantr::quad::QuadratureRule`.
///
/// The third binding that exposes a type rather than kernels, after
/// `geometry.cpp` and `transform.cpp`. Under the 2026-08-27 amendment to
/// `design/cross_backend_types.md` the rule is owned by C++ and
/// `src/pantr/quad/_rule_nd.py` holds one.
///
/// ## `double` only, and here it is not a choice
///
/// `pantr.quad.QuadratureRule` casts points and weights to `float64`
/// unconditionally, so there is no `float32` oracle a second overload could be
/// parity against -- `design/backend_parity.md` Rule 8. The header is
/// `double`-only for that reason and for the measured one `scripts/ci_local.sh`
/// enforces, so unlike `AABB` there is no generic version to narrow *from*.
///
/// ## What this file validates, and what it does not
///
/// Almost nothing, and the exceptions are the point. The type validates its own
/// arguments and throws `std::invalid_argument`, which nanobind surfaces as
/// `ValueError` -- the exception the oracle raises for the same inputs. What
/// stays on this side is only what the C++ signature cannot express:
///
///  - **Rank and dtype**, refused by nanobind's typed signature before the body
///    runs. A rank error therefore reaches Python as `TypeError`, where the
///    oracle raises `ValueError`, so `_rule_nd.py` checks rank *before* calling
///    and this layer never sees a wrong-rank array from the ordinary path.
///  - **The `(ndim, npts)` calling convention** of `gauss_legendre_quadrature`,
///    which exists only in Python: the C++ factory takes one span and reads the
///    dimension off it, so there is no second argument for the two to disagree
///    about. `_rule_nd.py` owns those two checks for the same reason.
///
/// ## The pair, and why it is not a `Rule1D` here
///
/// The header takes `std::span<const Rule1D>` so that a C++ caller cannot pair
/// one axis's nodes with another's weights (FELIGN/pantr#358). That guarantee
/// stops at this boundary on purpose: `tensor_product_quadrature`'s **public
/// Python signature** already takes `Sequence[tuple[nodes, weights]]`, it is not
/// this ticket's to change, and binding `Rule1D` as a Python class would expose a
/// non-owning view of two numpy buffers whose lifetime nothing here controls. So
/// the pairs are unpacked into named locals immediately below and turned into
/// `Rule1D`s before anything else happens, which is the narrowest place the
/// transposition can still be made and the one line that has to be read.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/quad/rule.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using Rule = pantr::quad::QuadratureRule;

/// A read-only, contiguous 1D `float64` array as nanobind sees it.
using const_vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A read-only, contiguous 2D `float64` array as nanobind sees it.
using const_mat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// One axis's `(nodes, weights)` pair as it crosses the boundary.
using Pair1D = std::pair<const_vec, const_vec>;

/// View a 1D nanobind array as a span.
///
/// \param a The array.
/// \return A span over its storage, valid while the array is alive.
std::span<const double> as_span(const const_vec& a) {
    return {a.data(), a.shape(0)};
}

/// Copy a contiguous block into a freshly allocated, read-only numpy array.
///
/// Copied rather than viewed, for the same reason as `AABB`'s corners and the
/// affine map's matrix: a view could outlive the rule, and the rule is immutable
/// precisely so that nobody has to reason about when its storage changes. The
/// Python wrapper caches the result, so a rule is copied out once rather than
/// once per attribute read.
///
/// \param data The block to copy.
/// \param rows Row count; zero means a 1D result.
/// \param cols Column count, or the length when `rows` is zero.
/// \return An owning `float64` array, flagged read-only.
nb::object to_numpy(const double* data, std::size_t rows, std::size_t cols) {
    const std::size_t total = (rows == 0 ? cols : rows * cols);
    auto* owned = new std::vector<double>(data, data + total);
    nb::capsule owner(owned,
                      [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    if (rows == 0) {
        return nb::cast(
            nb::ndarray<nb::numpy, const double, nb::ndim<1>>(owned->data(), {cols}, owner));
    }
    return nb::cast(
        nb::ndarray<nb::numpy, const double, nb::ndim<2>>(owned->data(), {rows, cols}, owner));
}

/// Build the tensor product from per-axis `(nodes, weights)` pairs.
///
/// \param axes One pair per axis.
/// \return The rule.
Rule tensor_product(const std::vector<Pair1D>& axes) {
    std::vector<pantr::quad::Rule1D> rules;
    rules.reserve(axes.size());
    for (const auto& [nodes, weights] : axes) {
        rules.push_back(pantr::quad::Rule1D{as_span(nodes), as_span(weights)});
    }
    return Rule::tensor_product(rules);
}

}  // namespace

void register_quad_types(nb::module_& m) {
    nb::class_<Rule>(m, "QuadratureRule")
        .def(
            "__init__",
            [](Rule* self, const_mat points, const_vec weights) {
                new (self) Rule(
                    pantr::span2d<const double>(points.data(), points.shape(0), points.shape(1)),
                    as_span(weights));
            },
            nb::arg("points"), nb::arg("weights"))
        .def_static("tensor_product", &tensor_product, nb::arg("rules"))
        .def_static(
            "gauss_legendre",
            [](const std::vector<int>& npts) { return Rule::gauss_legendre(npts); },
            nb::arg("npts"))
        .def_prop_ro("ndim", &Rule::ndim)
        .def_prop_ro("num_points", &Rule::num_points)
        .def_prop_ro("points",
                     [](const Rule& r) {
                         return to_numpy(r.points().data_handle(), r.num_points(), r.ndim());
                     })
        .def_prop_ro("weights",
                     [](const Rule& r) { return to_numpy(r.weights().data(), 0, r.num_points()); })
        .def("__repr__", &Rule::to_string);
}
