/// \file
/// nanobind bindings for the general-knot B-spline basis tabulation kernels of
/// `pantr/bspline/tabulate.hpp`.
///
/// ## These call Layer 3, not the space-level Layer 2 equivalent
///
/// `tabulate.hpp` offers two pairs of free functions: `basis_funcs_1d` /
/// `basis_derivs_1d`, which take a raw knot vector and are Layer 3, and
/// `tabulate_basis_1d` / `tabulate_basis_derivatives_1d`, which take a
/// `BsplineSpace1D` and choose between the general recurrence and the
/// Bézier-like fast path -- the Layer 2 equivalent for a C++ caller with no
/// interpreter. **Both pairs are bound, and they are bound for different
/// callers.**
///
/// `pantr.bspline._basis_backend` calls the *raw-knot* pair, and
/// `tabulate.hpp`'s "Why the dispatch stays on the Python side" is why: building
/// a space per call would re-validate and copy the knot vector in front of a
/// kernel the oracle calls with two or three points, and the space's constructor
/// snaps by default, so it could silently evaluate on a different knot vector
/// than the oracle did. That dispatch stays where it is.
///
/// The *space-level* pair is bound so that the parity suite can exercise it
/// against the oracle's own dispatch. It is the code a C++ consumer writes, and
/// until now `ctest` was the only thing that ran it -- which tests it against
/// itself rather than against the oracle, and so cannot see the two disagreeing
/// about which path a space takes or about the change of variable on it. It
/// takes an already-built space, so none of the per-call cost above applies.
///
/// ## What nanobind checks, and what is checked here
///
/// The aliases below pin dtype, rank, C-contiguity and device, so those are
/// validated before either function body runs -- the same split `basis.cpp`
/// argues. What a type cannot express is the relation between arguments:
/// `knots.size() >= 2 * degree + 2`, and that every output array has the shape
/// its degree, `n_deriv` and point count call for.
///
/// **None of these checks has an oracle counterpart, and that is deliberate
/// rather than an omission.** The oracle's Layer 2 (`_validate_out_array`, the
/// domain check, `n_deriv < 0`) is Python, lives above the backend dispatch, and
/// runs before either backend's kernel -- so it is shared by both and cannot
/// itself diverge between them. What is checked below exists for a caller who
/// reaches `pantr._pantr_cpp` directly, which is possible because it is a
/// public attribute of a public module: `basis.cpp` records a measured SIGSEGV
/// (a negative degree wrapping to `SIZE_MAX`) and a measured heap corruption (an
/// undersized `out`) from exactly that path, before its own checks existed.
///
/// `degree` and `n_deriv` are `unsigned` for the reason `basis.cpp` gives:
/// nanobind's own caster rejects a negative value with `TypeError` before this
/// body runs, while the kernels' parameters stay `std::int64_t`.
///
/// ## `.noconvert()` everywhere
///
/// The kernels accumulate in the knots' own scalar type -- `tabulate.hpp`'s file
/// comment measures this at length for the factorial-scaling site -- so a
/// silent `float32` to `float64` promotion on `knots` or `points` would not
/// merely copy, it would change the accumulation width and make the result
/// disagree with the oracle for a reason no caller could see. For every output
/// array `.noconvert()` is the correctness argument `basis.cpp` makes: a
/// converting cast would fill a discarded temporary and hand the caller back an
/// untouched array, with no exception anywhere.
///
/// ## `out_first_basis` is `std::int64_t`, not a platform-width type
///
/// `bspline_extraction.cpp:88-94` gives the reason this port repeats at every
/// seam that crosses an index array: `np.intp` is `int64` on every platform this
/// is built for, so `.noconvert()` accepts the oracle's own array unchanged,
/// and a JIT cannot reliably infer a platform-width type -- `size_t` must not
/// cross the seam either way.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "pantr/bspline/tabulate.hpp"
#include "pantr/core/mdspan.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using pantr::span2d;
using pantr::span_nd;

template <class T>
using const_knots = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using const_points = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_matrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_tensor = nb::ndarray<T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

/// The first-basis output, `np.intp` on the Python side; see the file comment.
using out_first_basis_t = nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Refuse a `degree` or `n_deriv` too large to fit the kernels' `std::int64_t`
/// parameter once widened from `unsigned`, and refuse a knot vector too short
/// for `degree` -- the one check here that is a memory-safety obligation of the
/// kernel rather than a shape mismatch, so it is checked rather than asserted.
///
/// \param name Which parameter to name in the "too large" message: "degree" or
///        "n_deriv".
/// \param value The parameter's value.
/// \throws nb::value_error If `value` exceeds what a C `int` can express.
void check_fits_int(const char* name, unsigned value) {
    constexpr unsigned max_value = static_cast<unsigned>(std::numeric_limits<int>::max());
    if (value > max_value) {
        throw nb::value_error((std::string(name) + " " + std::to_string(value) +
                               " exceeds the largest value the kernel can express (" +
                               std::to_string(max_value) + ")")
                                  .c_str());
    }
}

/// Refuse a knot vector too short for `degree`, the kernels' memory-safety
/// obligation.
///
/// Divided rather than multiplied, as `bspline_extraction_operators.cpp`'s
/// `check_spline_info` is: `2 * degree + 2` is signed overflow for a `degree`
/// near the top of the range a `std::int64_t` can hold, while the division
/// form agrees with it everywhere else. `degree` is already known non-negative
/// here, being `unsigned`.
///
/// \param knot_count Number of knots, as a signed count.
/// \param degree The polynomial degree.
/// \throws nb::value_error If `knot_count < 2 * degree + 2`.
void check_knots_long_enough(std::int64_t knot_count, std::int64_t degree) {
    if (knot_count < 2 || (knot_count - 2) / 2 < degree) {
        throw nb::value_error(("knots has " + std::to_string(knot_count) +
                               " elements, but degree " + std::to_string(degree) +
                               " needs at least " + std::to_string(2 * degree + 2))
                                  .c_str());
    }
}

/// Tabulate the B-spline basis over a general knot vector, checked and dispatched.
///
/// \tparam T Scalar type of the knots, the points and the output.
/// \tparam Kernel `pantr::bspline::basis_funcs_1d<T>`.
/// \param knots The knot vector, non-decreasing, at least `2 * degree + 2` entries.
/// \param degree The polynomial degree.
/// \param periodic Whether the space is periodic.
/// \param points The evaluation points.
/// \param out_basis Shape `(points.size(), degree + 1)`, written in full.
/// \param out_first_basis One entry per point, written in full.
/// \throws nb::value_error If `degree` is too large to fit a C `int`, if `knots`
///         is too short for `degree`, or if either output has the wrong shape.
template <class T, void (*Kernel)(std::span<const T>, std::int64_t, bool, std::span<const T>,
                                  span2d<T>, std::span<std::int64_t>)>
void tabulate_basis(const_knots<T> knots, unsigned degree, bool periodic, const_points<T> points,
                    out_matrix<T> out_basis, out_first_basis_t out_first_basis) {
    check_fits_int("degree", degree);
    const auto degree_i64 = static_cast<std::int64_t>(degree);
    check_knots_long_enough(static_cast<std::int64_t>(knots.size()), degree_i64);

    const std::size_t num_pts = points.size();
    const std::size_t num_basis = static_cast<std::size_t>(degree) + 1;
    if (out_basis.shape(0) != num_pts || out_basis.shape(1) != num_basis) {
        throw nb::value_error(("out_basis has shape (" + std::to_string(out_basis.shape(0)) +
                               ", " + std::to_string(out_basis.shape(1)) + "), but degree " +
                               std::to_string(degree) + " at " + std::to_string(num_pts) +
                               " points needs (" + std::to_string(num_pts) + ", " +
                               std::to_string(num_basis) + ")")
                                  .c_str());
    }
    if (out_first_basis.size() != num_pts) {
        throw nb::value_error(("out_first_basis has " + std::to_string(out_first_basis.size()) +
                               " elements, but " + std::to_string(num_pts) +
                               " points need one each")
                                  .c_str());
    }

    const std::span<const T> knot_span(knots.data(), knots.size());
    const std::span<const T> pts(points.data(), num_pts);
    const span2d<T> basis_view(out_basis.data(), num_pts, num_basis);
    const std::span<std::int64_t> first_basis_view(out_first_basis.data(), num_pts);

    // Neither kernel touches a Python object, so the GIL buys nothing while
    // either runs; releasing it is what lets a caller thread at the Python
    // level, as `basis.cpp` does.
    const nb::gil_scoped_release release;
    Kernel(knot_span, degree_i64, periodic, pts, basis_view, first_basis_view);
}

/// Tabulate the B-spline basis derivatives over a general knot vector, checked
/// and dispatched.
///
/// \tparam T Scalar type of the knots, the points and the output.
/// \tparam Kernel `pantr::bspline::basis_derivs_1d<T>`.
/// \param knots The knot vector, non-decreasing, at least `2 * degree + 2` entries.
/// \param degree The polynomial degree.
/// \param periodic Whether the space is periodic.
/// \param n_deriv Highest derivative order.
/// \param points The evaluation points.
/// \param out_deriv Shape `(points.size(), n_deriv + 1, degree + 1)`, written in full.
/// \param out_first_basis One entry per point, written in full.
/// \throws nb::value_error If `degree` or `n_deriv` is too large to fit a C
///         `int`, if `knots` is too short for `degree`, or if either output has
///         the wrong shape.
template <class T, void (*Kernel)(std::span<const T>, std::int64_t, bool, std::int64_t,
                                  std::span<const T>, span_nd<T, 3>, std::span<std::int64_t>)>
void tabulate_basis_derivatives(const_knots<T> knots, unsigned degree, bool periodic,
                                unsigned n_deriv, const_points<T> points,
                                out_tensor<T> out_deriv, out_first_basis_t out_first_basis) {
    check_fits_int("degree", degree);
    check_fits_int("n_deriv", n_deriv);
    const auto degree_i64 = static_cast<std::int64_t>(degree);
    const auto n_deriv_i64 = static_cast<std::int64_t>(n_deriv);
    check_knots_long_enough(static_cast<std::int64_t>(knots.size()), degree_i64);

    const std::size_t num_pts = points.size();
    const std::size_t num_basis = static_cast<std::size_t>(degree) + 1;
    const std::size_t num_rows = static_cast<std::size_t>(n_deriv) + 1;
    if (out_deriv.shape(0) != num_pts || out_deriv.shape(1) != num_rows ||
        out_deriv.shape(2) != num_basis) {
        throw nb::value_error(("out_deriv has shape (" + std::to_string(out_deriv.shape(0)) +
                               ", " + std::to_string(out_deriv.shape(1)) + ", " +
                               std::to_string(out_deriv.shape(2)) + "), but degree " +
                               std::to_string(degree) + " and n_deriv " +
                               std::to_string(n_deriv) + " at " + std::to_string(num_pts) +
                               " points needs (" + std::to_string(num_pts) + ", " +
                               std::to_string(num_rows) + ", " + std::to_string(num_basis) +
                               ")")
                                  .c_str());
    }
    if (out_first_basis.size() != num_pts) {
        throw nb::value_error(("out_first_basis has " + std::to_string(out_first_basis.size()) +
                               " elements, but " + std::to_string(num_pts) +
                               " points need one each")
                                  .c_str());
    }

    const std::span<const T> knot_span(knots.data(), knots.size());
    const std::span<const T> pts(points.data(), num_pts);
    const span_nd<T, 3> deriv_view(out_deriv.data(), num_pts, num_rows, num_basis);
    const std::span<std::int64_t> first_basis_view(out_first_basis.data(), num_pts);

    const nb::gil_scoped_release release;
    Kernel(knot_span, degree_i64, periodic, n_deriv_i64, pts, deriv_view, first_basis_view);
}

/// Tabulate a space's basis, taking its Bézier-like fast path.
///
/// The space-level counterpart of `tabulate_basis`, and the reason it exists is
/// in the file comment: it is what a C++ consumer calls, and binding it is what
/// lets the parity suite compare that dispatch against the oracle's own rather
/// than only against `ctest`.
///
/// The space arrives already built and already validated, so the knot-length
/// check `tabulate_basis` performs has no counterpart here -- its constructor
/// made it impossible. Only the output shapes, which the space cannot know, are
/// checked.
///
/// \tparam T Scalar type of the space, the points and the output.
/// \param space The space to tabulate.
/// \param points The evaluation points.
/// \param out_basis Shape `(points.size(), space.degree() + 1)`, written in full.
/// \param out_first_basis One entry per point, written in full.
/// \throws nb::value_error If either output has the wrong shape.
template <class T>
void tabulate_space_basis(const pantr::bspline::BsplineSpace1D<T>& space,
                          const_points<T> points, out_matrix<T> out_basis,
                          out_first_basis_t out_first_basis) {
    const std::size_t num_pts = points.size();
    const auto num_basis = static_cast<std::size_t>(space.degree()) + 1;
    if (out_basis.shape(0) != num_pts || out_basis.shape(1) != num_basis) {
        throw nb::value_error(("out_basis has shape (" + std::to_string(out_basis.shape(0)) +
                               ", " + std::to_string(out_basis.shape(1)) + "), but degree " +
                               std::to_string(space.degree()) + " at " +
                               std::to_string(num_pts) + " points needs (" +
                               std::to_string(num_pts) + ", " + std::to_string(num_basis) + ")")
                                  .c_str());
    }
    if (out_first_basis.size() != num_pts) {
        throw nb::value_error(("out_first_basis has " + std::to_string(out_first_basis.size()) +
                               " elements, but " + std::to_string(num_pts) +
                               " points need one each")
                                  .c_str());
    }

    const std::span<const T> pts(points.data(), num_pts);
    const span2d<T> basis_view(out_basis.data(), num_pts, num_basis);
    const std::span<std::int64_t> first_basis_view(out_first_basis.data(), num_pts);

    const nb::gil_scoped_release release;
    pantr::bspline::tabulate_basis_1d<T>(space, pts, basis_view, first_basis_view);
}

/// Tabulate a space's basis derivatives, taking its Bézier-like fast path.
///
/// See `tabulate_space_basis` for why the space-level pair is bound at all, and
/// why no knot-length check appears here.
///
/// \tparam T Scalar type of the space, the points and the output.
/// \param space The space to tabulate.
/// \param n_deriv Highest derivative order.
/// \param points The evaluation points.
/// \param out_deriv Shape `(points.size(), n_deriv + 1, space.degree() + 1)`.
/// \param out_first_basis One entry per point, written in full.
/// \throws nb::value_error If `n_deriv` is too large to fit a C `int`, or if
///         either output has the wrong shape.
template <class T>
void tabulate_space_basis_derivatives(const pantr::bspline::BsplineSpace1D<T>& space,
                                      unsigned n_deriv, const_points<T> points,
                                      out_tensor<T> out_deriv,
                                      out_first_basis_t out_first_basis) {
    check_fits_int("n_deriv", n_deriv);
    const auto n_deriv_i64 = static_cast<std::int64_t>(n_deriv);

    const std::size_t num_pts = points.size();
    const auto num_basis = static_cast<std::size_t>(space.degree()) + 1;
    const std::size_t num_rows = static_cast<std::size_t>(n_deriv) + 1;
    if (out_deriv.shape(0) != num_pts || out_deriv.shape(1) != num_rows ||
        out_deriv.shape(2) != num_basis) {
        throw nb::value_error(("out_deriv has shape (" + std::to_string(out_deriv.shape(0)) +
                               ", " + std::to_string(out_deriv.shape(1)) + ", " +
                               std::to_string(out_deriv.shape(2)) + "), but degree " +
                               std::to_string(space.degree()) + " and n_deriv " +
                               std::to_string(n_deriv) + " at " + std::to_string(num_pts) +
                               " points needs (" + std::to_string(num_pts) + ", " +
                               std::to_string(num_rows) + ", " + std::to_string(num_basis) +
                               ")")
                                  .c_str());
    }
    if (out_first_basis.size() != num_pts) {
        throw nb::value_error(("out_first_basis has " + std::to_string(out_first_basis.size()) +
                               " elements, but " + std::to_string(num_pts) +
                               " points need one each")
                                  .c_str());
    }

    const std::span<const T> pts(points.data(), num_pts);
    const span_nd<T, 3> deriv_view(out_deriv.data(), num_pts, num_rows, num_basis);
    const std::span<std::int64_t> first_basis_view(out_first_basis.data(), num_pts);

    const nb::gil_scoped_release release;
    pantr::bspline::tabulate_basis_derivatives_1d<T>(space, n_deriv_i64, pts, deriv_view,
                                                     first_basis_view);
}

}  // namespace

void register_bspline_basis(nb::module_& m) {
    // `nb::kw_only()` before the outputs and `.noconvert()` on every array, for
    // the same reasons `basis.cpp` gives: an output silently filled into a
    // discarded temporary is the worst failure shape available, and two
    // same-shaped outputs in a row make a transposed positional call type-check.
    const auto bind_basis = [&m](const char* name, auto f64, auto f32) {
        m.def(name, f64, nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("periodic"),
              nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out_basis").noconvert(),
              nb::arg("out_first_basis").noconvert());
        m.def(name, f32, nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("periodic"),
              nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out_basis").noconvert(),
              nb::arg("out_first_basis").noconvert());
    };

    bind_basis("tabulate_bspline_basis_1d",
              &tabulate_basis<double, &pantr::bspline::basis_funcs_1d<double>>,
              &tabulate_basis<float, &pantr::bspline::basis_funcs_1d<float>>);

    const auto bind_derivatives = [&m](const char* name, auto f64, auto f32) {
        m.def(name, f64, nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("periodic"),
              nb::arg("n_deriv"), nb::arg("points").noconvert(), nb::kw_only(),
              nb::arg("out_deriv").noconvert(), nb::arg("out_first_basis").noconvert());
        m.def(name, f32, nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("periodic"),
              nb::arg("n_deriv"), nb::arg("points").noconvert(), nb::kw_only(),
              nb::arg("out_deriv").noconvert(), nb::arg("out_first_basis").noconvert());
    };

    bind_derivatives(
        "tabulate_bspline_basis_derivatives_1d",
        &tabulate_basis_derivatives<double, &pantr::bspline::basis_derivs_1d<double>>,
        &tabulate_basis_derivatives<float, &pantr::bspline::basis_derivs_1d<float>>);

    // The space-level pair. The space carries its own degree and periodicity, so
    // those parameters are gone; the outputs keep `nb::kw_only()` and
    // `.noconvert()` for the reasons above, which the space does not change.
    m.def("tabulate_bspline_space_basis_1d", &tabulate_space_basis<double>, nb::arg("space"),
          nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out_basis").noconvert(),
          nb::arg("out_first_basis").noconvert());
    m.def("tabulate_bspline_space_basis_1d", &tabulate_space_basis<float>, nb::arg("space"),
          nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out_basis").noconvert(),
          nb::arg("out_first_basis").noconvert());

    m.def("tabulate_bspline_space_basis_derivatives_1d",
          &tabulate_space_basis_derivatives<double>, nb::arg("space"), nb::arg("n_deriv"),
          nb::arg("points").noconvert(), nb::kw_only(), nb::arg("out_deriv").noconvert(),
          nb::arg("out_first_basis").noconvert());
    m.def("tabulate_bspline_space_basis_derivatives_1d", &tabulate_space_basis_derivatives<float>,
          nb::arg("space"), nb::arg("n_deriv"), nb::arg("points").noconvert(), nb::kw_only(),
          nb::arg("out_deriv").noconvert(), nb::arg("out_first_basis").noconvert());
}
