/// \file
/// nanobind bindings for the `pantr.bezier` arithmetic kernels.
///
/// Eight entry points: the seven of `_bezier_core.py` and the reduction-operator
/// apply that `pantr.bezier` reaches for through `pantr.bspline`. The checks in
/// this file are Layer 2's C++ half, for the reason `basis.cpp` states at length:
/// a Layer 3 kernel validates nothing, the extension is importable, and every
/// bound name here is a public attribute of a public module.
///
/// ## What the shape checks are for
///
/// nanobind's typed parameters settle dtype, rank, C-contiguity and device before
/// any body runs. What they cannot express is a relation between arguments, and
/// every kernel here has one: `out` must have `degree + 1` rows, or
/// `points.size()` of them, or `a.size() + b.size() - 1` entries. Those are the
/// checks below, and each one is in front of a kernel that would otherwise write
/// past the allocation.
///
/// One is not a shape at all. `degree_elevate` and `scalar_bernstein_product`
/// form binomial coefficients, and past `core::kBincoeffMaxN` the exact-integer
/// recurrence overflows -- undefined behaviour in C++, where the numba original
/// merely wraps. Python's Layer 2 checks it too, through
/// `_check_bincoeff_envelope`, but a direct call on the extension does not go
/// through Python's Layer 2.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <limits>
#include <span>
#include <string>

#include "pantr/bezier/bezier.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/reduction_operator.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using const_mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_vec = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_mat = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_cube = nb::ndarray<T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

/// Raise unless `out` has exactly `rows` by `cols`.
template <class Arr>
void require_shape(const Arr& out, std::size_t rows, std::size_t cols, const char* what) {
    if (out.shape(0) != rows || out.shape(1) != cols) {
        throw nb::value_error((std::string(what) + " has shape (" +
                               std::to_string(out.shape(0)) + ", " +
                               std::to_string(out.shape(1)) + "), but this call needs (" +
                               std::to_string(rows) + ", " + std::to_string(cols) + ")")
                                  .c_str());
    }
}

/// Raise unless `out` has exactly `n` entries.
template <class Arr>
void require_length(const Arr& out, std::size_t n, const char* what) {
    if (out.shape(0) != n) {
        throw nb::value_error((std::string(what) + " has length " +
                               std::to_string(out.shape(0)) + ", but this call needs " +
                               std::to_string(n))
                                  .c_str());
    }
}

/// Raise unless the exact-integer binomial recurrence stays inside its envelope.
void require_bincoeff_envelope(std::size_t n, const char* what) {
    if (n > static_cast<std::size_t>(pantr::core::kBincoeffMaxN)) {
        throw nb::value_error(
            (std::string(what) + " needs binomial coefficients up to C(" + std::to_string(n) +
             ", k), beyond the largest upper index " +
             std::to_string(pantr::core::kBincoeffMaxN) +
             " the exact-integer recurrence can compute without an int64 overflow")
                .c_str());
    }
}

template <class T>
void bind_evaluate(const_mat<T> ctrl, const_vec<T> points, out_mat<T> out) {
    const std::size_t num_pts = points.size();
    const std::size_t rank = ctrl.shape(1);
    require_shape(out, num_pts, rank, "out");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), ctrl.shape(0), rank);
    const std::span<const T> pts(points.data(), num_pts);
    const pantr::span2d<T> out_view(out.data(), num_pts, rank);

    const nb::gil_scoped_release release;
    pantr::evaluate_bezier_1d<T>(ctrl_view, pts, out_view);
}

template <class T>
void bind_evaluate_deriv(const_mat<T> ctrl, const_vec<T> points, unsigned n_deriv,
                         out_cube<T> out) {
    constexpr unsigned max_deriv = static_cast<unsigned>(std::numeric_limits<int>::max());
    if (n_deriv > max_deriv) {
        throw nb::value_error("n_deriv exceeds the largest order the kernel can express");
    }
    const std::size_t num_pts = points.size();
    const std::size_t rank = ctrl.shape(1);
    const std::size_t orders = static_cast<std::size_t>(n_deriv) + 1;
    if (out.shape(0) != num_pts || out.shape(1) != orders || out.shape(2) != rank) {
        throw nb::value_error(("out has shape (" + std::to_string(out.shape(0)) + ", " +
                               std::to_string(out.shape(1)) + ", " +
                               std::to_string(out.shape(2)) + "), but this call needs (" +
                               std::to_string(num_pts) + ", " + std::to_string(orders) + ", " +
                               std::to_string(rank) + ")")
                                  .c_str());
    }

    const pantr::span2d<const T> ctrl_view(ctrl.data(), ctrl.shape(0), rank);
    const std::span<const T> pts(points.data(), num_pts);
    const pantr::span_nd<T, 3> out_view(out.data(), num_pts, orders, rank);

    const nb::gil_scoped_release release;
    pantr::evaluate_bezier_deriv_1d<T>(ctrl_view, pts, static_cast<int>(n_deriv), out_view);
}

template <class T>
void bind_degree_elevate(unsigned degree, const_mat<T> ctrl, unsigned degree_increment,
                         out_mat<T> out) {
    constexpr unsigned max_degree = static_cast<unsigned>(std::numeric_limits<int>::max()) / 2;
    if (degree > max_degree || degree_increment > max_degree) {
        throw nb::value_error("degree or degree_increment exceeds what the kernel can express");
    }
    const std::size_t rank = ctrl.shape(1);
    const std::size_t elevated = static_cast<std::size_t>(degree) +
                                 static_cast<std::size_t>(degree_increment) + 1;
    require_bincoeff_envelope(elevated - 1, "degree elevation");
    if (ctrl.shape(0) != static_cast<std::size_t>(degree) + 1) {
        throw nb::value_error(("ctrl has " + std::to_string(ctrl.shape(0)) +
                               " rows, but degree " + std::to_string(degree) + " needs " +
                               std::to_string(degree + 1))
                                  .c_str());
    }
    require_shape(out, elevated, rank, "out");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), ctrl.shape(0), rank);
    const pantr::span2d<T> out_view(out.data(), elevated, rank);

    const nb::gil_scoped_release release;
    pantr::degree_elevate_bezier_1d<T>(static_cast<int>(degree), ctrl_view,
                                       static_cast<int>(degree_increment), out_view);
}

template <class T>
void bind_slice(const_mat<T> ctrl, double value, out_vec<T> out) {
    const std::size_t n_cols = ctrl.shape(1);
    require_length(out, n_cols, "out");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), ctrl.shape(0), n_cols);
    const std::span<T> out_view(out.data(), n_cols);

    const nb::gil_scoped_release release;
    pantr::slice_bezier_1d<T>(ctrl_view, value, out_view);
}

template <class T>
void bind_split(const_mat<T> ctrl, double value, out_mat<T> out_left, out_mat<T> out_right) {
    const std::size_t rows = ctrl.shape(0);
    const std::size_t n_cols = ctrl.shape(1);
    require_shape(out_left, rows, n_cols, "out_left");
    require_shape(out_right, rows, n_cols, "out_right");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), rows, n_cols);
    const pantr::span2d<T> left_view(out_left.data(), rows, n_cols);
    const pantr::span2d<T> right_view(out_right.data(), rows, n_cols);

    const nb::gil_scoped_release release;
    pantr::split_bezier_1d<T>(ctrl_view, value, left_view, right_view);
}

template <class T>
void bind_restrict(const_mat<T> ctrl, double lower, double upper, out_mat<T> out) {
    const std::size_t rows = ctrl.shape(0);
    const std::size_t n_cols = ctrl.shape(1);
    require_shape(out, rows, n_cols, "out");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), rows, n_cols);
    const pantr::span2d<T> out_view(out.data(), rows, n_cols);

    const nb::gil_scoped_release release;
    pantr::restrict_bezier_1d<T>(ctrl_view, lower, upper, out_view);
}

template <class T>
void bind_product(const_vec<T> a, const_vec<T> b, out_vec<T> out) {
    if (a.size() == 0 || b.size() == 0) {
        throw nb::value_error("a and b must each hold at least one control point");
    }
    const std::size_t degree_sum = a.size() + b.size() - 2;
    require_bincoeff_envelope(degree_sum, "the Bernstein product");
    require_length(out, degree_sum + 1, "out");

    const std::span<const T> a_view(a.data(), a.size());
    const std::span<const T> b_view(b.data(), b.size());
    const std::span<T> out_view(out.data(), degree_sum + 1);

    const nb::gil_scoped_release release;
    pantr::scalar_bernstein_product_1d<T>(a_view, b_view, out_view);
}

template <class T>
void bind_apply_reduction_operator(const_mat<double> op, const_mat<T> ctrl, out_mat<T> out) {
    const std::size_t n_out = op.shape(0);
    const std::size_t n_in = op.shape(1);
    const std::size_t rank = ctrl.shape(1);
    if (ctrl.shape(0) != n_in) {
        throw nb::value_error(("ctrl has " + std::to_string(ctrl.shape(0)) +
                               " rows, but the operator's " + std::to_string(n_in) +
                               " columns need that many")
                                  .c_str());
    }
    require_shape(out, n_out, rank, "out");

    const pantr::span2d<const double> op_view(op.data(), n_out, n_in);
    const pantr::span2d<const T> ctrl_view(ctrl.data(), n_in, rank);
    const pantr::span2d<T> out_view(out.data(), n_out, rank);

    const nb::gil_scoped_release release;
    pantr::core::apply_reduction_operator<T>(op_view, ctrl_view, out_view);
}

}  // namespace

void register_bezier(nb::module_& m) {
    // `.noconvert()` everywhere and `nb::kw_only()` before the outputs, for the
    // two reasons `basis.cpp` sets out in full: without `.noconvert()` nanobind
    // satisfies a contiguity constraint by filling a temporary and discarding it,
    // so an out-parameter silently does nothing; and a positional call is how
    // two same-dtype same-shape output buffers get swapped.
    //
    // `split_bezier_1d` is the first kernel in the tree where that second risk is
    // real rather than anticipatory. Its `out_left` and `out_right` have the same
    // dtype and the same shape, so nothing in the type system separates them, and
    // a caller that passed them the wrong way round would get a plausible answer
    // with the two halves exchanged. Keyword-only is what makes that unsayable.
    m.def("evaluate_bezier_1d", &bind_evaluate<double>, nb::arg("ctrl").noconvert(),
          nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out").noconvert());
    m.def("evaluate_bezier_1d", &bind_evaluate<float>, nb::arg("ctrl").noconvert(),
          nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out").noconvert());

    m.def("evaluate_bezier_deriv_1d", &bind_evaluate_deriv<double>, nb::arg("ctrl").noconvert(),
          nb::arg("points").noconvert(), nb::arg("n_deriv"), nb::kw_only(),
          nb::arg("out").noconvert());
    m.def("evaluate_bezier_deriv_1d", &bind_evaluate_deriv<float>, nb::arg("ctrl").noconvert(),
          nb::arg("points").noconvert(), nb::arg("n_deriv"), nb::kw_only(),
          nb::arg("out").noconvert());

    m.def("degree_elevate_bezier_1d", &bind_degree_elevate<double>, nb::arg("degree"),
          nb::arg("ctrl").noconvert(), nb::arg("degree_increment"), nb::kw_only(),
          nb::arg("out").noconvert());
    m.def("degree_elevate_bezier_1d", &bind_degree_elevate<float>, nb::arg("degree"),
          nb::arg("ctrl").noconvert(), nb::arg("degree_increment"), nb::kw_only(),
          nb::arg("out").noconvert());

    m.def("slice_bezier_1d", &bind_slice<double>, nb::arg("ctrl").noconvert(), nb::arg("value"),
          nb::kw_only(), nb::arg("out").noconvert());
    m.def("slice_bezier_1d", &bind_slice<float>, nb::arg("ctrl").noconvert(), nb::arg("value"),
          nb::kw_only(), nb::arg("out").noconvert());

    m.def("split_bezier_1d", &bind_split<double>, nb::arg("ctrl").noconvert(), nb::arg("value"),
          nb::kw_only(), nb::arg("out_left").noconvert(), nb::arg("out_right").noconvert());
    m.def("split_bezier_1d", &bind_split<float>, nb::arg("ctrl").noconvert(), nb::arg("value"),
          nb::kw_only(), nb::arg("out_left").noconvert(), nb::arg("out_right").noconvert());

    m.def("restrict_bezier_1d", &bind_restrict<double>, nb::arg("ctrl").noconvert(),
          nb::arg("lower"), nb::arg("upper"), nb::kw_only(), nb::arg("out").noconvert());
    m.def("restrict_bezier_1d", &bind_restrict<float>, nb::arg("ctrl").noconvert(),
          nb::arg("lower"), nb::arg("upper"), nb::kw_only(), nb::arg("out").noconvert());

    m.def("scalar_bernstein_product_1d", &bind_product<double>, nb::arg("a").noconvert(),
          nb::arg("b").noconvert(), nb::kw_only(), nb::arg("out").noconvert());
    m.def("scalar_bernstein_product_1d", &bind_product<float>, nb::arg("a").noconvert(),
          nb::arg("b").noconvert(), nb::kw_only(), nb::arg("out").noconvert());

    m.def("apply_reduction_operator", &bind_apply_reduction_operator<double>,
          nb::arg("operator").noconvert(), nb::arg("ctrl").noconvert(), nb::kw_only(),
          nb::arg("out").noconvert());
    m.def("apply_reduction_operator", &bind_apply_reduction_operator<float>,
          nb::arg("operator").noconvert(), nb::arg("ctrl").noconvert(), nb::kw_only(),
          nb::arg("out").noconvert());
}
