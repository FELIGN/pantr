/// \file
/// nanobind bindings for the `pantr.quad` kernels.
///
/// See `basis.cpp`'s file comment for the two rules that apply here without
/// repeating: no docstrings in C++, and the GIL is released around every
/// kernel call because none of them touches a Python object.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>

#include "pantr/quad/lambert_w.hpp"
#include "pantr/quad/legendre.hpp"
#include "pantr/quad/simple_rules.hpp"
#include "pantr/quad/tanh_sinh.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

template <class T>
using out_vector = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// The largest count every kernel here can express: its parameter is `int`,
/// so a value beyond `INT_MAX` cannot be passed through the `static_cast`
/// below without wrapping. `n` arrives as `unsigned`, which already refuses a
/// negative value at the nanobind caster, so this is the other half of the
/// same guard `tabulate` in `basis.cpp` applies to `degree`.
constexpr unsigned max_count = static_cast<unsigned>(std::numeric_limits<int>::max());

/// Refuse two output buffers that overlap in memory.
///
/// **This is a correctness requirement, not tidiness, and the failure it prevents
/// is silent.** Every kernel here writes its two outputs independently, so handing
/// it the same array twice is accepted by nanobind (the two parameters have the
/// same type, rank, dtype and contiguity) and the second write simply overwrites
/// the first. Measured before this guard existed:
/// `gauss_legendre_symmetric(5, same, same)` returned the WEIGHTS in `same`, with
/// no exception anywhere -- a plausible-looking array of the right shape holding
/// the wrong quantity.
///
/// The Python side could not do this: `pantr.quad`'s Layer 3 returns freshly
/// allocated arrays and has no `out` parameters at all, so aliasing was
/// structurally impossible there and became reachable only when the port
/// introduced out-parameters. The sanctioned callers, the `_cpp_*` adapters in
/// `pantr.quad._quad_backend`, allocate two fresh arrays per call and cannot
/// trigger it; this guards the direct-import path that `basis.cpp` documents as a
/// real one.
///
/// Overlap rather than equality: two different views onto one buffer alias just as
/// destructively as the same object passed twice.
void refuse_overlapping_outputs(const out_vector<double>& first, const char* first_name,
                                const out_vector<double>& second, const char* second_name) {
    const auto* first_begin = first.data();
    const auto* first_end = first_begin + first.size();
    const auto* second_begin = second.data();
    const auto* second_end = second_begin + second.size();
    if (first_begin < second_end && second_begin < first_end) {
        throw nb::value_error(
            (std::string(first_name) + " and " + second_name +
             " overlap in memory. Each is written independently, so one would "
             "silently overwrite the other and the result would be neither. Pass "
             "two separate arrays.")
                .c_str());
    }
}

void gauss_legendre_symmetric(unsigned n, out_vector<double> out_nodes,
                               out_vector<double> out_weights) {
    if (n < 1) {
        throw nb::value_error(("n must be at least 1, got " + std::to_string(n)).c_str());
    }
    if (n > max_count) {
        throw nb::value_error(("n " + std::to_string(n) +
                               " exceeds the largest value the kernel can express (" +
                               std::to_string(max_count) + ")")
                                  .c_str());
    }
    if (out_nodes.size() != n) {
        throw nb::value_error(("out_nodes has size " + std::to_string(out_nodes.size()) +
                               " but n = " + std::to_string(n) + " requires size " +
                               std::to_string(n))
                                  .c_str());
    }
    if (out_weights.size() != n) {
        throw nb::value_error(("out_weights has size " + std::to_string(out_weights.size()) +
                               " but n = " + std::to_string(n) + " requires size " +
                               std::to_string(n))
                                  .c_str());
    }

    refuse_overlapping_outputs(out_nodes, "out_nodes", out_weights, "out_weights");

    const std::span<double> nodes(out_nodes.data(), n);
    const std::span<double> weights(out_weights.data(), n);

    const nb::gil_scoped_release release;
    pantr::gauss_legendre_symmetric(static_cast<int>(n), nodes, weights);
}

/// Validate `x` and solve, unlike the kernel behind it.
///
/// This function is Layer 2's C++ half, in the sense `basis.cpp`'s `tabulate`
/// comment gives: `pantr::lambert_w_principal` is a Layer 3 kernel and its
/// docstring states plainly that it checks nothing, while this wrapper is the
/// caller that has to establish what the kernel assumes. That is a genuine
/// asymmetry with the Python side, not an oversight to reconcile:
/// `pantr.quad._rules._lambert_w_principal` is a Layer 3 kernel in Python too,
/// and *its* caller is `_generate_tanh_sinh`, which only ever passes an
/// argument at or above 1.885 (smallest at n = 2) and so never exercises the
/// check. Below about 1.61 the branch-free asymptotic start lands on the
/// wrong branch and no number of Halley steps recovers it -- measured on the
/// Python original, an argument corresponding to a decay factor of 0.50 leaves
/// 2.8e4 units of roundoff, 0.40 leaves 2.4e16, and below about 0.31 the inner
/// logarithm is not real. This binding is a public attribute of a public
/// module, though, with no such guarantee on its caller, so it checks.
double lambert_w_principal(double x) {
    if (!std::isfinite(x) || x < 1.61) {
        throw nb::value_error(
            ("x must be finite and at least 1.61, got " + std::to_string(x)).c_str());
    }

    const nb::gil_scoped_release release;
    return pantr::lambert_w_principal(x);
}

int generate_tanh_sinh(unsigned n, double min_gap, out_vector<double> out_nodes,
                        out_vector<double> out_weights) {
    if (n < 1) {
        throw nb::value_error(("n must be at least 1, got " + std::to_string(n)).c_str());
    }
    if (n > max_count) {
        throw nb::value_error(("n " + std::to_string(n) +
                               " exceeds the largest value the kernel can express (" +
                               std::to_string(max_count) + ")")
                                  .c_str());
    }
    if (!std::isfinite(min_gap) || min_gap <= 0.0) {
        throw nb::value_error(
            ("min_gap must be finite and strictly positive, got " + std::to_string(min_gap))
                .c_str());
    }
    if (out_nodes.size() < n) {
        throw nb::value_error(("out_nodes has size " + std::to_string(out_nodes.size()) +
                               " but n = " + std::to_string(n) + " requires size at least " +
                               std::to_string(n))
                                  .c_str());
    }
    if (out_weights.size() < n) {
        throw nb::value_error(("out_weights has size " + std::to_string(out_weights.size()) +
                               " but n = " + std::to_string(n) + " requires size at least " +
                               std::to_string(n))
                                  .c_str());
    }

    refuse_overlapping_outputs(out_nodes, "out_nodes", out_weights, "out_weights");

    const std::span<double> nodes(out_nodes.data(), out_nodes.size());
    const std::span<double> weights(out_weights.data(), out_weights.size());

    const nb::gil_scoped_release release;
    const std::size_t count = pantr::generate_tanh_sinh(static_cast<int>(n), min_gap, nodes, weights);
    // count <= n <= max_count <= INT_MAX, so the narrowing below cannot lose.
    return static_cast<int>(count);
}

void trapezoidal(unsigned n, out_vector<double> out_nodes, out_vector<double> out_weights) {
    if (n < 1) {
        throw nb::value_error(("n must be at least 1, got " + std::to_string(n)).c_str());
    }
    if (n > max_count) {
        throw nb::value_error(("n " + std::to_string(n) +
                               " exceeds the largest value the kernel can express (" +
                               std::to_string(max_count) + ")")
                                  .c_str());
    }
    if (out_nodes.size() != n) {
        throw nb::value_error(("out_nodes has size " + std::to_string(out_nodes.size()) +
                               " but n = " + std::to_string(n) + " requires size " +
                               std::to_string(n))
                                  .c_str());
    }
    if (out_weights.size() != n) {
        throw nb::value_error(("out_weights has size " + std::to_string(out_weights.size()) +
                               " but n = " + std::to_string(n) + " requires size " +
                               std::to_string(n))
                                  .c_str());
    }

    refuse_overlapping_outputs(out_nodes, "out_nodes", out_weights, "out_weights");

    const std::span<double> nodes(out_nodes.data(), n);
    const std::span<double> weights(out_weights.data(), n);

    const nb::gil_scoped_release release;
    pantr::trapezoidal(static_cast<int>(n), nodes, weights);
}

/// Computed in `T`, matching `simple_rules.hpp`'s file comment: unlike every
/// other kernel in this directory, the arithmetic is not narrowed at the end
/// but carried out in the storage format throughout, because a
/// double-then-narrow port was measured to disagree with a `float32` `cos` on
/// 62% of arguments. So this wrapper is templated too, and bound twice below.
template <class T>
void modified_chebyshev_nodes(unsigned n, out_vector<T> out) {
    if (n < 2) {
        throw nb::value_error(("n must be at least 2, got " + std::to_string(n)).c_str());
    }
    if (n > max_count) {
        throw nb::value_error(("n " + std::to_string(n) +
                               " exceeds the largest value the kernel can express (" +
                               std::to_string(max_count) + ")")
                                  .c_str());
    }
    if (out.size() != n) {
        throw nb::value_error(("out has size " + std::to_string(out.size()) + " but n = " +
                               std::to_string(n) + " requires size " + std::to_string(n))
                                  .c_str());
    }

    const std::span<T> view(out.data(), n);

    const nb::gil_scoped_release release;
    pantr::modified_chebyshev_nodes<T>(static_cast<int>(n), view);
}

}  // namespace

void register_quad(nb::module_& m) {
    // Every array argument below is `.noconvert()`, for the reason `basis.cpp`
    // records at length: an `nb::ndarray<T, c_contig>` parameter that fails the
    // constraint is silently satisfied by a CONVERTED temporary rather than
    // refused, and for an `out` array that means the kernel fills a temporary
    // that is then discarded while the caller's array comes back untouched, no
    // exception raised anywhere.
    m.def("gauss_legendre_symmetric", &gauss_legendre_symmetric, nb::arg("n"),
          nb::arg("out_nodes").noconvert(), nb::arg("out_weights").noconvert());

    m.def("lambert_w_principal", &lambert_w_principal, nb::arg("x"));

    m.def("generate_tanh_sinh", &generate_tanh_sinh, nb::arg("n"), nb::arg("min_gap"),
          nb::arg("out_nodes").noconvert(), nb::arg("out_weights").noconvert());

    m.def("trapezoidal", &trapezoidal, nb::arg("n"), nb::arg("out_nodes").noconvert(),
          nb::arg("out_weights").noconvert());

    m.def("modified_chebyshev_nodes", &modified_chebyshev_nodes<double>, nb::arg("n"),
          nb::arg("out").noconvert());
    m.def("modified_chebyshev_nodes", &modified_chebyshev_nodes<float>, nb::arg("n"),
          nb::arg("out").noconvert());
}
