/// \file
/// nanobind bindings for the `pantr.basis` kernels.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <limits>
#include <span>
#include <string>

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/basis/legendre.hpp"
#include "pantr/core/mdspan.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// The array shapes the kernel accepts, expressed in the type system.
///
/// nanobind checks dtype, rank, C-contiguity and device before the function body
/// runs and raises a `TypeError` otherwise, so these aliases *are* the input
/// validation for everything they can express. What they cannot express is a
/// relation BETWEEN two arguments -- `out.shape(0) == points.size()` and
/// `out.shape(1) == degree + 1` -- and that is checked in the function body
/// below.
template <class T>
using const_points = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_matrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// Tabulate into `out`, after checking what the kernel assumes and never checks.
///
/// **This function is Layer 2's C++ half**, and the checks below belong here for
/// the reason CLAUDE.md gives: a Layer 3 kernel validates nothing, so every
/// guarantee it relies on is established by its caller. The adapter in
/// `pantr.basis._basis_backend` is the Python half of the same layer, but it is
/// not the only caller -- the extension
/// is importable, and `pantr._pantr_cpp.tabulate_cardinal_bspline_1d` is a public
/// attribute of a public module. Measured before these checks existed, one line
/// of Python reached undefined behaviour twice over: a negative `degree` became
/// `SIZE_MAX` at `static_cast<std::size_t>(degree)` and exited with SIGSEGV, and
/// an `out` too small for the degree overran the allocation and corrupted the
/// heap. Neither raised anything a caller could catch.
///
/// `degree` is `unsigned` rather than `int` so that the negative case never
/// reaches this body at all: nanobind's own caster rejects it, raising a
/// `TypeError` before any of pantr's code runs. The kernel's parameter stays
/// `int` -- it is a discrete-structure parameter, and design/automatic
/// _differentiation.md's fourth discipline keeps those as plain integers -- so
/// the one range that `unsigned` admits and `int` does not is refused explicitly
/// rather than wrapped around by the cast.
///
/// The cost is three integer comparisons in front of an
/// `O(points.size() * degree^2)` kernel.
///
/// The kernel is a template parameter rather than a runtime argument so that the
/// three tabulations share this body without sharing a dispatch: each `m.def`
/// below instantiates its own copy, and adding a fourth basis cannot accidentally
/// route to the wrong one.
template <class T, void (*Kernel)(int, std::span<const T>, pantr::span2d<T>)>
void tabulate(unsigned degree, const_points<T> points, out_matrix<T> out) {
    constexpr unsigned max_degree = static_cast<unsigned>(std::numeric_limits<int>::max());
    if (degree > max_degree) {
        throw nb::value_error(("degree " + std::to_string(degree) +
                               " exceeds the largest degree the kernel can express (" +
                               std::to_string(max_degree) + ")")
                                  .c_str());
    }

    const std::size_t num_pts = points.size();
    const std::size_t num_basis = static_cast<std::size_t>(degree) + 1;
    if (out.shape(0) != num_pts || out.shape(1) != num_basis) {
        throw nb::value_error(("out has shape (" + std::to_string(out.shape(0)) + ", " +
                               std::to_string(out.shape(1)) + "), but degree " +
                               std::to_string(degree) + " at " + std::to_string(num_pts) +
                               " points needs (" + std::to_string(num_pts) + ", " +
                               std::to_string(num_basis) + ")")
                                  .c_str());
    }

    const std::span<const T> pts(points.data(), num_pts);
    const pantr::span2d<T> view(out.data(), num_pts, num_basis);

    // The kernel touches no Python object, so the GIL buys nothing while it
    // runs. Releasing it is what lets a caller thread at the Python level.
    const nb::gil_scoped_release release;
    Kernel(static_cast<int>(degree), pts, view);
}

}  // namespace

void register_basis(nb::module_& m) {
    // `.noconvert()` on `out` is a correctness requirement, not a tuning knob.
    //
    // An `nb::ndarray<T, c_contig>` parameter does NOT reject an argument that
    // fails the constraint: nanobind satisfies the constraint by CONVERTING it
    // into a contiguous temporary. For an input that is merely wasteful. For an
    // output it is silently wrong -- the kernel fills the temporary, the
    // temporary is discarded, and the caller's array comes back untouched.
    //
    // Measured before the fix: with PANTR_BACKEND=cpp,
    // `tabulate_cardinal_bspline_1d(2, pts, out=non_contiguous)` returned all
    // zeros while the numba backend returned the right answer, with no error
    // anywhere. An out-parameter that quietly does nothing is the worst failure
    // shape available: no exception, no warning, a plausible-looking array of
    // the right shape and dtype.
    //
    // `points` gets it too. There a conversion is correct, so the argument is
    // weaker -- but this prototype exists to measure, and a silent copy of a
    // 10^6-element array inside the timed region would be attributed to the
    // kernel. Better to refuse and make the caller fix the layout.
    //
    // `nb::kw_only()` before `out` is a second, independent guard: with only
    // one output buffer here there is nothing to transpose within this
    // binding, but the convention has to be uniform before `quad.cpp`'s
    // two-output kernels are bound, where a positional call silently accepts
    // `out_nodes` and `out_weights` swapped.
    //
    // The Bernstein and Legendre tabulations were added for the change-of-basis
    // builders, which evaluate one or both of them before every Gram solve. They
    // are bound here rather than in change_basis.cpp because they are
    // `pantr.basis` kernels and the split follows the Python package, not the
    // consumer.
    const auto bind = [&m](const char* name, auto f64, auto f32) {
        m.def(name, f64, nb::arg("degree"), nb::arg("points").noconvert(), nb::kw_only(),
              nb::arg("out").noconvert());
        m.def(name, f32, nb::arg("degree"), nb::arg("points").noconvert(), nb::kw_only(),
              nb::arg("out").noconvert());
    };

    bind("tabulate_cardinal_bspline_1d",
         &tabulate<double, &pantr::tabulate_cardinal_bspline_1d<double>>,
         &tabulate<float, &pantr::tabulate_cardinal_bspline_1d<float>>);
    bind("tabulate_bernstein_1d", &tabulate<double, &pantr::tabulate_bernstein_1d<double>>,
         &tabulate<float, &pantr::tabulate_bernstein_1d<float>>);
    bind("tabulate_legendre_1d", &tabulate<double, &pantr::tabulate_legendre_1d<double>>,
         &tabulate<float, &pantr::tabulate_legendre_1d<float>>);
}
