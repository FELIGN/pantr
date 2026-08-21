/// \file
/// nanobind bindings for the `pantr.change_basis` kernels.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <limits>
#include <span>
#include <string>

#include "pantr/change_basis/change_basis.hpp"
#include "pantr/core/mdspan.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// See the equivalent aliases in basis.cpp: nanobind checks dtype, rank,
/// C-contiguity and device before the body runs, so these are the input
/// validation for everything a type can express. What they cannot express are
/// the relations between arguments, checked below.
template <class T>
using const_vector = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_matrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// The largest degree the kernels can express, as an `unsigned`.
constexpr unsigned max_degree = static_cast<unsigned>(std::numeric_limits<int>::max());

/// Reject a degree that cannot round-trip through the kernels' `int`.
///
/// `degree` is `unsigned` in every binding here for the reason basis.cpp gives:
/// nanobind's caster then refuses a negative value before any pantr code runs,
/// and the one range `unsigned` admits that `int` does not is refused here rather
/// than wrapped around by the cast.
///
/// \param degree The requested degree.
void check_degree(unsigned degree) {
    if (degree > max_degree) {
        throw nb::value_error(("degree " + std::to_string(degree) +
                               " exceeds the largest degree the kernel can express (" +
                               std::to_string(max_degree) + ")")
                                  .c_str());
    }
}

/// Reject an `out` that is not the square matrix the degree calls for.
///
/// \param out The caller's output array.
/// \param degree The requested degree.
template <class T>
void check_square_out(const out_matrix<T>& out, unsigned degree) {
    const std::size_t n = static_cast<std::size_t>(degree) + 1;
    if (out.shape(0) != n || out.shape(1) != n) {
        throw nb::value_error(("out has shape (" + std::to_string(out.shape(0)) + ", " +
                               std::to_string(out.shape(1)) + "), but degree " +
                               std::to_string(degree) + " needs (" + std::to_string(n) +
                               ", " + std::to_string(n) + ")")
                                  .c_str());
    }
}

/// Reject a node array whose length is not `degree + 1`.
///
/// Every builder bound here integrates or interpolates with exactly `degree + 1`
/// nodes, so a mismatch is a caller error rather than a supported shape. Checking
/// it is not optional: the Gram products would otherwise contract over a length
/// the output was not sized for.
///
/// \param nodes The node array.
/// \param degree The requested degree.
/// \param name The parameter's name, for the message.
template <class T>
void check_node_count(const const_vector<T>& nodes, unsigned degree, const char* name) {
    const std::size_t n = static_cast<std::size_t>(degree) + 1;
    if (nodes.size() != n) {
        throw nb::value_error((std::string(name) + " has length " +
                               std::to_string(nodes.size()) + ", but degree " +
                               std::to_string(degree) + " needs " + std::to_string(n))
                                  .c_str());
    }
}

/// Adapter for the two builders that take Lagrange nodes and no weights.
template <class T, void (*Kernel)(int, std::span<const T>, pantr::span2d<T>)>
void from_nodes(unsigned degree, const_vector<T> nodes, out_matrix<T> out) {
    check_degree(degree);
    check_node_count(nodes, degree, "nodes");
    check_square_out(out, degree);

    const std::size_t n = static_cast<std::size_t>(degree) + 1;
    const std::span<const T> node_span(nodes.data(), nodes.size());
    const pantr::span2d<T> view(out.data(), n, n);

    const nb::gil_scoped_release release;
    Kernel(static_cast<int>(degree), node_span, view);
}

/// Adapter for the five builders that take a quadrature rule.
///
/// `nodes` and `weights` are separate arrays rather than one `(n, 2)` block
/// because that is the shape `pantr.quad.get_gauss_legendre_1d` returns, and
/// re-packing them in Layer 2 would put a copy in front of every call.
template <class T, void (*Kernel)(int, std::span<const T>, std::span<const T>, pantr::span2d<T>)>
void from_rule(unsigned degree, const_vector<T> nodes, const_vector<T> weights,
               out_matrix<T> out) {
    check_degree(degree);
    check_node_count(nodes, degree, "nodes");
    check_node_count(weights, degree, "weights");
    check_square_out(out, degree);

    const std::size_t n = static_cast<std::size_t>(degree) + 1;
    const std::span<const T> node_span(nodes.data(), nodes.size());
    const std::span<const T> weight_span(weights.data(), weights.size());
    const pantr::span2d<T> view(out.data(), n, n);

    const nb::gil_scoped_release release;
    Kernel(static_cast<int>(degree), node_span, weight_span, view);
}

/// Adapter for the one builder that needs nothing but a degree.
template <class T>
void monomial(unsigned degree, out_matrix<T> out) {
    check_degree(degree);
    check_square_out(out, degree);

    const std::size_t n = static_cast<std::size_t>(degree) + 1;
    const pantr::span2d<T> view(out.data(), n, n);

    const nb::gil_scoped_release release;
    pantr::monomial_to_bernstein_1d<T>(static_cast<int>(degree), view);
}

}  // namespace

void register_change_basis(nb::module_& m) {
    // `.noconvert()` everywhere, for the reason basis.cpp sets out at length: on
    // an output array a silent conversion fills a temporary and returns the
    // caller's buffer untouched, with no error anywhere. On the inputs it refuses
    // a hidden copy inside a timed region.
    //
    // `nb::kw_only()` before `out` matters more here than it did there. Five of
    // these take two same-dtype, same-length arrays in a row, and a positional
    // call that swapped `nodes` and `weights` would type-check, run, and return a
    // plausible matrix.
    const auto bind_nodes = [&m](const char* name, auto f64, auto f32) {
        m.def(name, f64, nb::arg("degree"), nb::arg("nodes").noconvert(), nb::kw_only(),
              nb::arg("out").noconvert());
        m.def(name, f32, nb::arg("degree"), nb::arg("nodes").noconvert(), nb::kw_only(),
              nb::arg("out").noconvert());
    };
    const auto bind_rule = [&m](const char* name, auto f64, auto f32) {
        m.def(name, f64, nb::arg("degree"), nb::arg("nodes").noconvert(),
              nb::arg("weights").noconvert(), nb::kw_only(), nb::arg("out").noconvert());
        m.def(name, f32, nb::arg("degree"), nb::arg("nodes").noconvert(),
              nb::arg("weights").noconvert(), nb::kw_only(), nb::arg("out").noconvert());
    };

    bind_nodes("lagrange_to_bernstein_1d",
               &from_nodes<double, &pantr::lagrange_to_bernstein_1d<double>>,
               &from_nodes<float, &pantr::lagrange_to_bernstein_1d<float>>);
    bind_nodes("bernstein_to_lagrange_1d",
               &from_nodes<double, &pantr::bernstein_to_lagrange_1d<double>>,
               &from_nodes<float, &pantr::bernstein_to_lagrange_1d<float>>);

    bind_rule("bernstein_to_cardinal_1d",
              &from_rule<double, &pantr::bernstein_to_cardinal_1d<double>>,
              &from_rule<float, &pantr::bernstein_to_cardinal_1d<float>>);
    bind_rule("cardinal_to_bernstein_1d",
              &from_rule<double, &pantr::cardinal_to_bernstein_1d<double>>,
              &from_rule<float, &pantr::cardinal_to_bernstein_1d<float>>);
    bind_rule("legendre_to_cardinal_1d",
              &from_rule<double, &pantr::legendre_to_cardinal_1d<double>>,
              &from_rule<float, &pantr::legendre_to_cardinal_1d<float>>);
    bind_rule("cardinal_to_legendre_1d",
              &from_rule<double, &pantr::cardinal_to_legendre_1d<double>>,
              &from_rule<float, &pantr::cardinal_to_legendre_1d<float>>);
    bind_rule("cardinal_dual_legendre_coeffs_1d",
              &from_rule<double, &pantr::cardinal_dual_legendre_coeffs_1d<double>>,
              &from_rule<float, &pantr::cardinal_dual_legendre_coeffs_1d<float>>);

    m.def("monomial_to_bernstein_1d", &monomial<double>, nb::arg("degree"), nb::kw_only(),
          nb::arg("out").noconvert());
    m.def("monomial_to_bernstein_1d", &monomial<float>, nb::arg("degree"), nb::kw_only(),
          nb::arg("out").noconvert());
}
