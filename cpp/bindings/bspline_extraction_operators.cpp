/// \file
/// Bindings for the Bézier extraction operator builder and its structural
/// identity mask, from `pantr/bspline/extraction.hpp`.
///
/// Separate from `bspline_extraction.cpp` because the two are different ports with
/// different parity claims: that file binds the tensor-product *apply* kernels,
/// which take operators as arguments and know nothing about knots, while these
/// *build* the operators from a knot vector. `register.hpp` gives each its own
/// entry point for the same reason.
///
/// ## What is validated here, and why it is here
///
/// The header validates nothing, as a Layer 3 kernel must not, so this file is the
/// C++ half of Layer 2 and owns every refusal. The three that mirror the oracle's
/// `_check_spline_info` carry **its messages character for character**, and the
/// `tol` refusal carries `_prepare_extraction_out`'s, so that the two backends are
/// indistinguishable to a caller who reads the text and `tests/parity/` can compare
/// them. The order is the oracle's too: degree, then length, then monotonicity.
/// The 1D-ness check that opens the oracle's version is nanobind's `nb::ndim<1>`,
/// because nanobind has no path to `TypeError`, which is what
/// `pantr/core/error.hpp` sets as the split for the whole port.
///
/// One refusal has **no counterpart in the oracle**, and it is a deliberate
/// divergence rather than an omission on either side. A knot vector spanning no
/// in-domain interval gives an empty `(0, p+1, p+1)` result, and the oracle's core
/// then indexes `out[0]` on it when the boundary multiplicity is short -- out of
/// bounds on an empty array. Refusing the vector is the only thing this side can do
/// that is not undefined behaviour, and the message is
/// `check_space_has_an_interval`'s, which is what `BsplineSpace1D` already raises
/// for the same knot vector. `pantr.bspline._extraction_backend` records that the
/// two backends differ here.
///
/// ## `.noconvert()` on every array
///
/// The builder's whole arithmetic runs in the knots' own scalar type -- the
/// insertion weight, its complement and the two products that combine two columns
/// -- so a silent `float32` to `float64` promotion on `knots` would not merely copy,
/// it would change the accumulation width and make the result disagree with the
/// oracle for a reason no caller could see. `design/backend_parity.md` Rule 9 is the
/// rule; this is where it is enforced at the seam.
///
/// The mask entry point takes `int64` and `bool` arrays and no float at all, so its
/// `.noconvert()` is about a silent `int32`-to-`int64` copy of a caller's index
/// array rather than about arithmetic.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/core/mdspan.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using pantr::span_nd;

template <class T>
using const_knots = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_operators = nb::ndarray<T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

/// A per-unique-knot multiplicity array, `np.intp` on the Python side.
///
/// `std::int64_t` rather than a platform-width type, for the reason
/// `bspline_extraction.cpp` gives of its own index maps: `np.intp` is `int64` on
/// every platform this is built for, so `.noconvert()` accepts the oracle's own
/// arrays unchanged, and a stub has to name a fixed type.
using const_multiplicity = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig,
                                       nb::device::cpu>;

/// A per-interval boolean mask.
using out_mask = nb::ndarray<bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Refuse the arguments the oracle's `_check_spline_info` refuses, in its order.
///
/// \param knots The knot vector as supplied.
/// \param degree The requested degree.
/// \throws nb::value_error With the oracle's message, on any of the three.
template <class T>
void check_spline_info(const_knots<T> knots, std::int64_t degree) {
    if (degree < 0) {
        throw nb::value_error("degree must be non-negative");
    }
    // Divided rather than multiplied, as `BsplineSpace1D::check_arguments` is and
    // for the same reason: `2 * degree + 2` is signed overflow for a degree near
    // the top of the range, while the oracle's Python integers are exact for any
    // degree. The `size < 2` guard is what makes the two spellings agree on every
    // input rather than nearly every one; `space_1d.hpp` carries the argument.
    const auto count = static_cast<std::int64_t>(knots.size());
    if (count < 2 || (count - 2) / 2 < degree) {
        throw nb::value_error("knots must have at least 2*degree+2 elements");
    }
    // `np.all(np.diff(knots) >= 0)`: the difference is formed first and only then
    // compared, which is not the same predicate as `knots[i] >= knots[i - 1]` -- a
    // vector of two infinities differences to NaN and is refused, while the direct
    // comparison would accept it.
    for (std::size_t i = 1; i < knots.size(); ++i) {
        const T step = knots(i) - knots(i - 1);
        if (!(static_cast<double>(step) >= 0.0)) {
            throw nb::value_error("knots must be non-decreasing");
        }
    }
}

/// Build the Bézier extraction operators, validated and dispatched.
///
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \param tol The absolute parametric tolerance.
/// \param out The `(n_intervals, degree + 1, degree + 1)` result.
/// \throws nb::value_error If an argument fails the oracle's checks, if the vector
///         spans no interval, or if `out` has the wrong shape.
template <class T>
void bezier_extraction(const_knots<T> knots, std::int64_t degree, double tol,
                       out_operators<T> out) {
    if (tol < 0.0) {
        throw nb::value_error("tol must be non-negative");
    }
    check_spline_info<T>(knots, degree);

    const std::span<const T> knot_span(knots.data(), knots.size());
    const std::int64_t n_intervals =
        pantr::bspline::classify_knots<T>(knot_span, degree, tol).num_intervals();
    // No counterpart in the oracle; see the file comment. `std::invalid_argument`
    // reaches Python as a `ValueError` through nanobind's default translator, which
    // is what the caller sees from `BsplineSpace1D` for the same vector.
    pantr::bspline::check_space_has_an_interval<T>(knot_span, degree, n_intervals, tol);

    const auto side = static_cast<std::size_t>(degree) + 1;
    if (out.shape(0) != static_cast<std::size_t>(n_intervals) || out.shape(1) != side
        || out.shape(2) != side) {
        throw nb::value_error(("out must have shape (" + std::to_string(n_intervals) + ", "
                              + std::to_string(side) + ", " + std::to_string(side) + ")")
                                 .c_str());
    }

    const span_nd<T, 3> result(out.data(), out.shape(0), side, side);

    const nb::gil_scoped_release release;
    pantr::bspline::bezier_extraction_1d<T>(knot_span, degree, tol, result);
}

/// Mark the intervals whose Bézier operator is the identity, validated and dispatched.
///
/// \param multiplicity The in-domain knot multiplicities, `n_intervals + 1` of them.
/// \param degree The polynomial degree.
/// \param out One flag per interval.
/// \throws nb::value_error If `multiplicity` is empty, `degree` is negative, or
///         `out` is not one shorter than `multiplicity`.
void bezier_identity_mask(const_multiplicity multiplicity, std::int64_t degree, out_mask out) {
    if (degree < 0) {
        throw nb::value_error("degree must be non-negative");
    }
    if (multiplicity.size() == 0) {
        throw nb::value_error("multiplicity must hold at least one knot class");
    }
    const std::size_t n_intervals = multiplicity.size() - 1;
    if (out.size() != n_intervals) {
        throw nb::value_error(
            ("out must have " + std::to_string(n_intervals) + " elements").c_str());
    }

    const std::span<const std::int64_t> counts(multiplicity.data(), multiplicity.size());
    const std::span<bool> flags(out.data(), out.size());

    const nb::gil_scoped_release release;
    pantr::bspline::bezier_structural_identity_mask(counts, degree, flags);
}

}  // namespace

void register_bspline_extraction_operators(nb::module_& m) {
    // Argument names mirror the oracle's Layer 2 helpers, and there is no
    // `nb::kw_only()`: the dispatcher in `pantr.bspline._extraction_backend` calls
    // these positionally, as it calls the Numba cores.
    m.def("bezier_extraction_1d", &bezier_extraction<double>, nb::arg("knots").noconvert(),
          nb::arg("degree"), nb::arg("tol"), nb::arg("out").noconvert());
    m.def("bezier_extraction_1d", &bezier_extraction<float>, nb::arg("knots").noconvert(),
          nb::arg("degree"), nb::arg("tol"), nb::arg("out").noconvert());
    m.def("bezier_structural_identity_mask", &bezier_identity_mask,
          nb::arg("multiplicities").noconvert(), nb::arg("degree"), nb::arg("out").noconvert());
}
