/// \file
/// nanobind bindings for the `pantr.bezier` arithmetic kernels.
///
/// Thirteen entry points: the seven of `_bezier_core.py`, the reduction-operator apply
/// that `pantr.bezier` reaches for through `pantr.bspline`, the two n-dimensional
/// evaluation entry points of `pantr/bezier/evaluate.hpp`, and the three degree
/// operations of `pantr/bezier/degree.hpp`. The checks in
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
/// Two are not shapes at all.
///
/// **A degree inferred from a shape has a floor that nothing else states.** Five
/// of these kernels take no `degree` argument and compute `ctrl.extent(0) - 1` in
/// `std::size_t`, so a zero-row `ctrl` underflows to `SIZE_MAX` and the kernel
/// walks off the allocation. Measured before `require_control_points` existed:
/// `slice_bezier_1d(np.empty((0, 2)), 0.5, out=...)` exited with SIGSEGV, no
/// exception. `basis.cpp` avoids this class by taking `degree` as an `unsigned`
/// the caster refuses to make negative; there is no such argument here, so the
/// floor has to be checked explicitly.
///
/// **A parameter domain the docstring states is a promise.** `value`, `lower` and
/// `upper` are documented as living in the unit interval, and outside it the
/// two-pass restriction stops being a sequence of convex combinations. Unchecked,
/// `slice_bezier_1d(ctrl, 2.5, ...)` extrapolated silently and
/// `restrict_bezier_1d(ctrl, 0.8, 0.2, ...)` -- the bounds transposed -- returned a
/// different, plausible Bézier. `lower` and `upper` are adjacent same-typed
/// positional parameters, so ordering them is what closes that transposition trap.
///
/// And `degree_elevate` and `scalar_bernstein_product`
/// form binomial coefficients, and past `core::kBincoeffMaxN` the exact-integer
/// recurrence overflows -- undefined behaviour in C++, where the numba original
/// merely wraps. Python's Layer 2 checks it too, through
/// `_check_bincoeff_envelope`, but a direct call on the extension does not go
/// through Python's Layer 2.
///
/// ## The two evaluation entry points take the value, not its arrays
///
/// Every other binding here is a Layer 3 kernel over raw buffers, which is what
/// `design/cross_backend_types.md`'s kernel-seam table describes. The two added by
/// FELIGN/pantr#389 are not kernels: they are operations on a domain type, and
/// since the 2026-08-27 amendment that type is owned by C++. So they take a
/// `Bezier32`/`Bezier64` handle and read the net through it, rather than being
/// handed a control-point array the wrapper unpacked. Unpacking and reassembling
/// is what that amendment forbids, and the rationality flag is exactly the kind of
/// invariant that gets dropped in a reassembly.
///
/// ## The degree operations take their operators as data
///
/// `reduce_bezier_degree` and `bezier_degree_reduction_error` are handed the
/// reduction operators and Bernstein Gram matrices rather than assembling them, which
/// is the ruling `core/reduction_operator.hpp` records for the operator and which
/// `pantr/bezier/degree.hpp` extends to the Gram matrix and argues for. Both are
/// `float64` whatever the Bézier stores, and both are indexed by parametric
/// direction, so a direction the caller is not acting on passes an empty array rather
/// than being absent -- an absent entry would make the list's index stop meaning the
/// direction, which is the one thing that must not become positional by accident.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/degree.hpp"
#include "pantr/bezier/evaluate.hpp"
#include "pantr/bezier/kernels_1d.hpp"
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

/// A writable output of any rank: the lattice result's rank is `dim + 1`, which is
/// a runtime quantity, so no `nb::ndim<N>` can state it.
template <class T>
using out_any = nb::ndarray<T, nb::c_contig, nb::device::cpu>;

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

/// Raise unless `ctrl` holds at least one control point.
///
/// The floor for every kernel that infers its degree from a shape. See the file
/// comment: below it the subtraction underflows and the kernel is not merely
/// wrong but undefined.
template <class Arr>
void require_control_points(const Arr& ctrl, const char* what) {
    if (ctrl.shape(0) == 0) {
        throw nb::value_error(
            (std::string(what) + " has no rows, but its degree is inferred as "
             "rows - 1, which needs at least one")
                .c_str());
    }
}

/// Raise unless `value` lies in the closed unit interval.
///
/// Written as a conjunction of two comparisons rather than a negated range so a
/// NaN, which compares false against everything, is refused rather than admitted.
void require_unit_interval(double value, const char* what) {
    if (!(value >= 0.0 && value <= 1.0)) {
        throw nb::value_error((std::string(what) + " must lie in [0, 1], got " +
                               std::to_string(value))
                                  .c_str());
    }
}

/// Raise unless `0 <= lower <= upper <= 1`.
///
/// The ordering is the half that matters: the two bounds are adjacent same-typed
/// positional parameters, so nothing in the type system stops a transposed call,
/// and a transposed call returns a different restriction rather than an error.
void require_ordered_bounds(double lower, double upper) {
    require_unit_interval(lower, "lower");
    require_unit_interval(upper, "upper");
    if (lower > upper) {
        throw nb::value_error(("lower (" + std::to_string(lower) + ") must not exceed upper (" +
                               std::to_string(upper) +
                               "); transposing them returns a different restriction rather "
                               "than an error")
                                  .c_str());
    }
}

/// Refuse two output buffers that overlap in memory.
///
/// The same guard `cpp/bindings/quad.cpp` applies to its two-output rules, and for
/// the same reason: the two buffers have the same dtype, rank, shape and
/// contiguity, so nanobind accepts one array passed twice and the second write
/// overwrites the first. Measured before this existed:
/// `split_bezier_1d(ctrl, 0.5, out_left=same, out_right=same)` returned the RIGHT
/// half under both names, with no exception. `nb::kw_only()` closes transposition;
/// only this closes aliasing.
///
/// Overlap rather than equality: two views onto one buffer alias just as
/// destructively as the same object twice.
template <class Arr>
void refuse_overlapping_outputs(const Arr& first, const char* first_name, const Arr& second,
                                const char* second_name) {
    const auto* first_begin = first.data();
    const auto* first_end = first_begin + first.size();
    const auto* second_begin = second.data();
    const auto* second_end = second_begin + second.size();
    if (first_begin < second_end && second_begin < first_end) {
        throw nb::value_error(
            (std::string(first_name) + " and " + second_name +
             " overlap in memory. Each half is written independently, so one would "
             "silently overwrite the other and the result would be neither. Pass "
             "two separate arrays.")
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
    require_control_points(ctrl, "ctrl");
    const std::size_t num_pts = points.size();
    const std::size_t rank = ctrl.shape(1);
    require_shape(out, num_pts, rank, "out");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), ctrl.shape(0), rank);
    const std::span<const T> pts(points.data(), num_pts);
    const pantr::span2d<T> out_view(out.data(), num_pts, rank);

    const nb::gil_scoped_release release;
    pantr::bezier::evaluate_bezier_1d<T>(ctrl_view, pts, out_view);
}

template <class T>
void bind_evaluate_deriv(const_mat<T> ctrl, const_vec<T> points, unsigned n_deriv,
                         out_cube<T> out) {
    constexpr unsigned max_deriv = static_cast<unsigned>(std::numeric_limits<int>::max());
    if (n_deriv > max_deriv) {
        throw nb::value_error("n_deriv exceeds the largest order the kernel can express");
    }
    require_control_points(ctrl, "ctrl");
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
    pantr::bezier::evaluate_bezier_deriv_1d<T>(ctrl_view, pts, static_cast<int>(n_deriv), out_view);
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
    pantr::bezier::degree_elevate_bezier_1d<T>(static_cast<int>(degree), ctrl_view,
                                               static_cast<int>(degree_increment), out_view);
}

template <class T>
void bind_slice(const_mat<T> ctrl, double value, out_vec<T> out) {
    require_control_points(ctrl, "ctrl");
    require_unit_interval(value, "value");
    const std::size_t n_cols = ctrl.shape(1);
    require_length(out, n_cols, "out");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), ctrl.shape(0), n_cols);
    const std::span<T> out_view(out.data(), n_cols);

    const nb::gil_scoped_release release;
    pantr::bezier::slice_bezier_1d<T>(ctrl_view, value, out_view);
}

template <class T>
void bind_split(const_mat<T> ctrl, double value, out_mat<T> out_left, out_mat<T> out_right) {
    require_control_points(ctrl, "ctrl");
    require_unit_interval(value, "value");
    const std::size_t rows = ctrl.shape(0);
    const std::size_t n_cols = ctrl.shape(1);
    require_shape(out_left, rows, n_cols, "out_left");
    require_shape(out_right, rows, n_cols, "out_right");
    refuse_overlapping_outputs(out_left, "out_left", out_right, "out_right");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), rows, n_cols);
    const pantr::span2d<T> left_view(out_left.data(), rows, n_cols);
    const pantr::span2d<T> right_view(out_right.data(), rows, n_cols);

    const nb::gil_scoped_release release;
    pantr::bezier::split_bezier_1d<T>(ctrl_view, value, left_view, right_view);
}

template <class T>
void bind_restrict(const_mat<T> ctrl, double lower, double upper, out_mat<T> out) {
    require_control_points(ctrl, "ctrl");
    require_ordered_bounds(lower, upper);
    const std::size_t rows = ctrl.shape(0);
    const std::size_t n_cols = ctrl.shape(1);
    require_shape(out, rows, n_cols, "out");

    const pantr::span2d<const T> ctrl_view(ctrl.data(), rows, n_cols);
    const pantr::span2d<T> out_view(out.data(), rows, n_cols);

    const nb::gil_scoped_release release;
    pantr::bezier::restrict_bezier_1d<T>(ctrl_view, lower, upper, out_view);
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
    pantr::bezier::scalar_bernstein_product_1d<T>(a_view, b_view, out_view);
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

/// Evaluate a Bézier at an explicit array of points.
///
/// The shape relations `evaluate` itself checks are left to it: it throws
/// `std::invalid_argument`, which nanobind maps to `ValueError`, and duplicating
/// them here would give two spellings of one message for the parity tests to
/// disagree about.
template <class T>
void bind_evaluate_bezier(const pantr::bezier::Bezier<T>& bezier, const_mat<T> points,
                          out_mat<T> out) {
    const pantr::span2d<const T> points_view(points.data(), points.shape(0), points.shape(1));
    const pantr::span2d<T> out_view(out.data(), out.shape(0), out.shape(1));

    const nb::gil_scoped_release release;
    pantr::bezier::evaluate<T>(bezier, points_view, out_view);
}

/// Evaluate a Bézier on a tensor-product lattice of points.
///
/// `out` is flat and of any rank: its logical shape is
/// `(m_0, ..., m_{dim-1}, rank)`, which no `nb::ndim<N>` can state because `dim` is
/// a runtime quantity. The total size is checked by `evaluate_on_lattice`, and a
/// C-contiguous array of the right size is the right array whatever its rank.
template <class T>
void bind_evaluate_bezier_on_lattice(const pantr::bezier::Bezier<T>& bezier,
                                     const std::vector<const_vec<T>>& points_per_dir,
                                     out_any<T> out) {
    std::vector<std::span<const T>> columns;
    columns.reserve(points_per_dir.size());
    for (const const_vec<T>& column : points_per_dir) {
        columns.emplace_back(column.data(), column.shape(0));
    }
    const std::span<T> out_view(out.data(), out.size());

    const nb::gil_scoped_release release;
    pantr::bezier::evaluate_on_lattice<T>(
        bezier, std::span<const std::span<const T>>(columns), out_view);
}

/// A list of dense `float64` matrices, one per parametric direction.
///
/// Directions the caller is not acting on pass an empty array; the header reads only
/// the entries whose direction has a nonzero decrement.
using matrix_list = std::vector<nb::ndarray<const double, nb::ndim<2>, nb::c_contig,
                                            nb::device::cpu>>;

/// Reinterpret a bound matrix list as the spans the header takes.
///
/// \param matrices The arrays, borrowed for the duration of the call.
/// \return One `span2d` per entry, in the same order.
std::vector<pantr::span2d<const double>> as_spans(const matrix_list& matrices) {
    std::vector<pantr::span2d<const double>> spans;
    spans.reserve(matrices.size());
    for (const auto& matrix : matrices) {
        spans.emplace_back(matrix.data(), matrix.shape(0), matrix.shape(1));
    }
    return spans;
}

/// Degree-elevate a Bézier, returning a new one.
template <class T>
pantr::bezier::Bezier<T> bind_elevate_degree(const pantr::bezier::Bezier<T>& bezier,
                                             const std::vector<std::size_t>& increments) {
    return pantr::bezier::elevate_degree<T>(bezier, std::span<const std::size_t>(increments));
}

/// Degree-reduce a Bézier with caller-supplied operators, returning a new one.
template <class T>
pantr::bezier::Bezier<T> bind_reduce_degree(const pantr::bezier::Bezier<T>& bezier,
                                            const std::vector<std::size_t>& decrements,
                                            const matrix_list& operators) {
    const std::vector<pantr::span2d<const double>> ops = as_spans(operators);
    return pantr::bezier::reduce_degree<T>(bezier, std::span<const std::size_t>(decrements),
                                           std::span<const pantr::span2d<const double>>(ops));
}

/// The `L2` error a degree reduction would introduce.
template <class T>
double bind_degree_reduction_error(const pantr::bezier::Bezier<T>& bezier,
                                   const std::vector<std::size_t>& decrements,
                                   const matrix_list& operators, const matrix_list& grams) {
    const std::vector<pantr::span2d<const double>> ops = as_spans(operators);
    const std::vector<pantr::span2d<const double>> gram_spans = as_spans(grams);
    return pantr::bezier::degree_reduction_error<T>(
        bezier, std::span<const std::size_t>(decrements),
        std::span<const pantr::span2d<const double>>(ops),
        std::span<const pantr::span2d<const double>>(gram_spans));
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

    // The two n-dimensional evaluation entry points. Their first argument is the
    // Bézier itself rather than its control points; see the file comment. Overload
    // resolution separates the two instantiations on the handle's own class, so
    // these two need no dtype argument and cannot be reached with a mismatched one.
    m.def("evaluate_bezier", &bind_evaluate_bezier<double>, nb::arg("bezier"),
          nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out").noconvert());
    m.def("evaluate_bezier", &bind_evaluate_bezier<float>, nb::arg("bezier"),
          nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out").noconvert());

    m.def("evaluate_bezier_on_lattice", &bind_evaluate_bezier_on_lattice<double>,
          nb::arg("bezier"), nb::arg("points_per_dir").noconvert(), nb::kw_only(),
          nb::arg("out").noconvert());
    m.def("evaluate_bezier_on_lattice", &bind_evaluate_bezier_on_lattice<float>,
          nb::arg("bezier"), nb::arg("points_per_dir").noconvert(), nb::kw_only(),
          nb::arg("out").noconvert());

    // The three degree operations. Unlike every kernel above these RETURN a value --
    // a new Bézier, or a number -- rather than filling a caller's buffer, because
    // what they produce is a value of a C++-owned type rather than an array whose
    // shape the caller already knows. The mirroring rule of
    // design/cross_backend_types.md is about the kernel seam and does not reach here.
    m.def("elevate_bezier_degree", &bind_elevate_degree<double>, nb::arg("bezier"),
          nb::arg("increments"));
    m.def("elevate_bezier_degree", &bind_elevate_degree<float>, nb::arg("bezier"),
          nb::arg("increments"));

    m.def("reduce_bezier_degree", &bind_reduce_degree<double>, nb::arg("bezier"),
          nb::arg("decrements"), nb::arg("operators").noconvert());
    m.def("reduce_bezier_degree", &bind_reduce_degree<float>, nb::arg("bezier"),
          nb::arg("decrements"), nb::arg("operators").noconvert());

    m.def("bezier_degree_reduction_error", &bind_degree_reduction_error<double>,
          nb::arg("bezier"), nb::arg("decrements"), nb::arg("operators").noconvert(),
          nb::arg("grams").noconvert());
    m.def("bezier_degree_reduction_error", &bind_degree_reduction_error<float>,
          nb::arg("bezier"), nb::arg("decrements"), nb::arg("operators").noconvert(),
          nb::arg("grams").noconvert());
}
