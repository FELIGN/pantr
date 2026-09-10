/// \file
/// Tests for `pantr/bspline/structural.hpp`: the open conversion, the split, the
/// slice and the boundary.
///
/// ## What is checked against what
///
/// The three operations divide cleanly into one that needs an **independent oracle**
/// and two that need a **consistency** check, and the file is arranged that way.
///
/// **`corner_cut_along_axis` is the one new kernel, so it is checked against closed
/// forms rather than against another implementation of itself.** Two identities pin
/// it, both exact in exact arithmetic for *any* knot vector and any degree:
///
///  - **Partition of unity.** A field whose every coefficient is 1 is the constant 1,
///    because the B-spline basis sums to one on the domain. So the corner cut must
///    return 1 at every parameter.
///  - **Linear precision.** A field whose coefficient `i` is the Greville abscissa
///    `g_i = (t_{i+1} + ... + t_{i+p}) / p` is the identity map `S(u) = u`. This is
///    the degree-one case of Marsden's identity, and it is a *stronger* test than
///    partition of unity: it fails if the span search picks the wrong span, if the
///    multiplicity count is off by one, or if a knot index in the recurrence is
///    shifted -- none of which partition of unity can see, since every coefficient
///    there is the same number.
///
/// Neither identity is a second computation of the corner cut, which is what makes
/// them oracles rather than mirrors. `test_bspline_refinement.cpp` uses Marsden's
/// identity in coefficient space for the same reason.
///
/// **`to_open` and `split` add no numerics** -- they are `inserted_knot_vector`,
/// `oslo_bands_1d` and `refine_along_axis`, which have their own tests and their own
/// derivations, wrapped in bookkeeping. So what is checked here is that the
/// bookkeeping is right: **the geometry does not move.** Both are sampled with the
/// corner cut, at parameters the identities above have just certified it on, and
/// compared against the original field sampled the same way. A wrong trim, a wrong
/// partition index, a wrong end multiplicity or a wrong wrap all displace the map,
/// and none of them is visible in the knot vector alone.
///
/// The structural facts each also guarantees -- the end multiplicities `to_open`
/// reaches, the domains the two halves span, the axis the slice drops, the handles it
/// shares -- are checked exactly, since they are integers and identities.
///
/// ## The accuracy bound, derived
///
/// `gamma_K = K u / (1 - K u)` with `u` the unit roundoff of the storage format, times
/// the magnitude of the quantity, plus `K` denormal floors. The floor is the absolute
/// half of Higham's model and it is load-bearing rather than decoration: linear
/// precision at `u = 0` has an exact-zero closed form, and a purely relative bound
/// there would assert bit-identity that nothing here has grounds for.
///
/// `K` for one corner cut at degree `p`:
///
///  - **`p` for the Greville abscissa itself** -- `p - 1` additions and one division.
///    Charged only where the coefficients are Greville abscissae; partition of unity
///    stores an exact 1.
///  - **`5p` for the cut**, being at most `p` passes at 5 roundings each: the two
///    products and the sum are 3 roundings of the coefficients, and the weight pair
///    `(fl_T(alpha), fl_T(1 - alpha))` departs from summing to one by at most 2 more.
///    The weights are in `[0, 1]` for a parameter inside its span, so no pass
///    amplifies the magnitude and the passes compose additively.
///
/// So `K = 6p` for linear precision and `K = 5p` for partition of unity. The
/// geometry-preservation checks compare two corner cuts of the *same* field over
/// different knot vectors, so they charge two cuts plus the insertion the refined side
/// went through, whose own count `test_bspline_refinement.cpp` derives as
/// `2p + 3 + (7p + 2)` per refined direction; `insertion_roundings` below assembles
/// the sum rather than restating it.
///
/// The magnitude is `max(|lo|, |hi|)` for linear precision -- a Greville abscissa is a
/// convex combination of knots, so it cannot leave the knot vector's own range -- and
/// 1 for partition of unity.
///
/// ## Both storage formats
///
/// Every templated check runs at `double` and at `float`, because the file comment of
/// `pantr/bspline/structural.hpp` records three sites where the oracle's arithmetic
/// width differs between them and a `double`-only suite would exercise none of the
/// three. The refusal checks are `double`-only: a message's text does not depend on
/// the format except in the one case that file records as unreachable.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/bspline/structural.hpp"

using pantr::bspline::Bspline;
using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::boundary;
using pantr::bspline::corner_cut_along_axis;
using pantr::bspline::KnotSnapping;
using pantr::bspline::open_along_axis;
using pantr::bspline::select_rows_along_axis;
using pantr::bspline::slice;
using pantr::bspline::slice_point;
using pantr::bspline::split;
using pantr::bspline::to_open;

namespace {

/// `gamma_m = m u / (1 - m u)`, the standard accumulation constant.
///
/// Defined here rather than shared, which is the convention the three test files that
/// already need it follow; collecting them is a cleanup this file does not take.
///
/// \tparam T The format the roundings are charged in.
/// \param m The number of roundings.
/// \return The constant, in units of the value being bounded.
template <class T>
double gamma_of(std::int64_t m) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    const double mu = static_cast<double>(m) * u;
    return mu / (1.0 - mu);
}

/// The absolute bound on a quantity of the given magnitude after `K` roundings.
///
/// \tparam T The storage format.
/// \param roundings `K`.
/// \param magnitude The magnitude of the quantity being bounded.
/// \return The bound, strictly positive even at magnitude zero.
template <class T>
double bound_for(std::int64_t roundings, double magnitude) {
    return gamma_of<T>(roundings) * magnitude
           + static_cast<double>(roundings) * static_cast<double>(std::numeric_limits<T>::min());
}

/// The roundings one knot insertion of a `p`-degree direction is charged.
///
/// `test_bspline_refinement.cpp` derives `2p + 3 + D(7p + 2)` for `D` refined
/// directions; here `D` is always 1, because every insertion below touches one
/// direction.
///
/// \param degree The direction's degree.
/// \return The count.
std::int64_t insertion_roundings(std::int64_t degree) {
    return 2 * degree + 3 + (7 * degree + 2);
}

/// A space over a knot vector, shared.
///
/// \tparam T The storage format.
/// \param knots The knot vector.
/// \param degree The degree.
/// \param periodic Whether the space is periodic.
/// \return The space, in a handle a field can take.
template <class T>
std::shared_ptr<const BsplineSpace1D<T>> direction(const std::vector<T>& knots,
                                                   std::int64_t degree, bool periodic) {
    return std::make_shared<const BsplineSpace1D<T>>(
        std::span<const T>(knots), degree, periodic, KnotSnapping::merge_near_duplicates);
}

/// A field over the given directions, with the given coefficients.
///
/// \tparam T The storage format.
/// \param directions One space handle per direction.
/// \param values The coefficients, row-major under `(*num_basis, num_components)`.
/// \param num_components How many components each coefficient has.
/// \param is_rational Whether the last component is a homogeneous weight.
/// \return The field.
template <class T>
Bspline<T> field_of(std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions,
                    const std::vector<T>& values, std::size_t num_components,
                    bool is_rational) {
    auto space = std::make_shared<const BsplineSpace<T>>(std::move(directions));
    std::vector<std::size_t> shape;
    for (const std::int64_t count : space->num_basis()) {
        shape.push_back(static_cast<std::size_t>(count));
    }
    shape.push_back(num_components);
    return Bspline<T>(space,
                      typename Bspline<T>::net_type(std::span<const T>(values),
                                                    std::span<const std::size_t>(shape)),
                      is_rational);
}

/// The Greville abscissae of a knot vector.
///
/// `g_i = (t_{i+1} + ... + t_{i+p}) / p`, which at `p = 0` is an average over an empty
/// range: the knot `t_i` is returned instead. **That branch is written rather than
/// merely documented**, because `vectors()` now carries a degree-0 entry and every
/// caller but `check_corner_cut_is_linearly_precise` reaches this unguarded --
/// `sum / T(0)` is a NaN that would propagate into a geometry comparison and read
/// there as a moved curve. Linear precision itself does not hold at degree 0, which is
/// why that one check skips it: a piecewise constant cannot reproduce the identity map.
///
/// \tparam T The storage format.
/// \param knots The knot vector.
/// \param degree The degree, non-negative.
/// \return One abscissa per basis function.
template <class T>
std::vector<T> greville(const std::vector<T>& knots, std::int64_t degree) {
    const auto count = static_cast<std::int64_t>(knots.size()) - degree - 1;
    std::vector<T> out(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        if (degree == 0) {
            out[static_cast<std::size_t>(i)] = knots[static_cast<std::size_t>(i)];
            continue;
        }
        T sum = T(0);
        for (std::int64_t j = 1; j <= degree; ++j) {
            sum = sum + knots[static_cast<std::size_t>(i + j)];
        }
        out[static_cast<std::size_t>(i)] = sum / static_cast<T>(degree);
    }
    return out;
}

/// The parameters a check samples across one direction's domain.
///
/// The two endpoints, every distinct interior knot, and a handful of points that are
/// neither -- the endpoints and the knots because they are where the span search and
/// the multiplicity count branch, and the rest because a wrong branch there would be
/// invisible.
///
/// \tparam T The storage format.
/// \param space The direction.
/// \return The parameters, as `double`.
template <class T>
std::vector<double> sample_parameters(const BsplineSpace1D<T>& space) {
    const std::array<T, 2> domain = space.domain();
    const double lo = static_cast<double>(domain[0]);
    const double hi = static_cast<double>(domain[1]);
    std::vector<double> out{lo, hi};
    for (const T& knot : space.unique_knots_in_domain()) {
        out.push_back(static_cast<double>(knot));
    }
    for (int step = 1; step < 8; ++step) {
        out.push_back(lo + (hi - lo) * static_cast<double>(step) / 8.0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The corner cut, against two closed forms
// ---------------------------------------------------------------------------

/// One knot vector's label, degree and entries.
struct Vector1D {
    const char* label;        ///< What to name it in a failure.
    std::int64_t degree;      ///< The degree.
    bool periodic;            ///< Whether the entries describe a periodic space.
    std::vector<double> raw;  ///< The entries, in `double`.
};

/// The knot vectors every templated check below runs over.
///
/// Clamped at both ends, clamped at one, clamped at neither, and periodic; degrees 0
/// through 4; interior multiplicities both 1 and above; one vector with **no** interior
/// knot, where the corner cut's span is the whole domain; and one periodic vector at
/// each end of the `[1, degree]` boundary-multiplicity range a periodic knot vector
/// admits, since the two trim a different number of ghost entries.
///
/// The vector clamped at one end only is named for the end it is **not** clamped at,
/// which is the reading `to_open` and `has_open_knots()` use: "open" here means
/// clamped.
///
/// \return The vectors.
std::vector<Vector1D> vectors() {
    return {
        {"p0-open", 0, false, {0.0, 0.4, 1.0}},
        {"p1-open", 1, false, {0.0, 0.0, 0.3, 0.7, 1.0, 1.0}},
        {"p2-open", 2, false, {0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0}},
        {"p2-single-span", 2, false, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}},
        {"p3-open", 3, false, {0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.0}},
        {"p4-open", 4, false, {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}},
        {"p2-double-interior", 2, false, {0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}},
        {"p3-c0-interior", 3, false, {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0}},
        {"p2-unclamped", 2, false, {-0.2, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2}},
        {"p2-unclamped-left-only", 2, false,
         {-0.2, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0}},
        {"p2-periodic-m1", 2, true, {-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}},
        {"p2-periodic-m2", 2, true, {-0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 1.5}},
        {"p2-shifted", 2, false, {2.0, 2.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.0}},
    };
}

/// The vector's entries at the storage format.
///
/// \tparam T The storage format.
/// \param raw The entries in `double`.
/// \return The entries in `T`.
template <class T>
std::vector<T> at_format(const std::vector<double>& raw) {
    std::vector<T> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = static_cast<T>(raw[i]);
    }
    return out;
}

/// The corner cut returns 1 wherever every coefficient is 1.
///
/// \tparam T The storage format.
/// \param format The format's name, for a failure note.
template <class T>
void check_corner_cut_sums_to_one(const char* format) {
    for (const Vector1D& vector : vectors()) {
        const std::vector<T> knots = at_format<T>(vector.raw);
        const auto n_full = static_cast<std::int64_t>(knots.size()) - vector.degree - 1;
        const std::vector<T> ones(static_cast<std::size_t>(n_full), T(1));
        const BsplineSpace1D<T> space(std::span<const T>(knots), vector.degree, vector.periodic,
                                      KnotSnapping::merge_near_duplicates);
        const double bound = bound_for<T>(5 * vector.degree, 1.0);
        for (const double u : sample_parameters<T>(space)) {
            const std::vector<T> got =
                corner_cut_along_axis<T>(std::span<const T>(knots), vector.degree,
                                         space.tolerance(), std::span<const T>(ones), u, 1, 1);
            const double deviation = std::abs(static_cast<double>(got[0]) - 1.0);
            PANTR_CHECK_MSG(got.size() == 1 && deviation <= bound,
                            std::string("partition of unity, ") + format + " "
                                + vector.label + " at u=" + std::to_string(u) + ": off by "
                                + std::to_string(deviation) + ", bound "
                                + std::to_string(bound));
        }
    }
}

/// The corner cut over the Greville abscissae is the identity map.
///
/// \tparam T The storage format.
/// \param format The format's name, for a failure note.
template <class T>
void check_corner_cut_is_linearly_precise(const char* format) {
    for (const Vector1D& vector : vectors()) {
        if (vector.degree < 1) {
            continue;
        }
        const std::vector<T> knots = at_format<T>(vector.raw);
        const std::vector<T> net = greville<T>(knots, vector.degree);
        const BsplineSpace1D<T> space(std::span<const T>(knots), vector.degree, vector.periodic,
                                      KnotSnapping::merge_near_duplicates);
        const std::array<T, 2> domain = space.domain();
        const double magnitude = std::max(std::abs(static_cast<double>(domain[0])),
                                          std::abs(static_cast<double>(domain[1])));
        const double bound = bound_for<T>(6 * vector.degree, magnitude);
        for (const double u : sample_parameters<T>(space)) {
            const std::vector<T> got =
                corner_cut_along_axis<T>(std::span<const T>(knots), vector.degree,
                                         space.tolerance(), std::span<const T>(net), u, 1, 1);
            const double deviation = std::abs(static_cast<double>(got[0]) - u);
            PANTR_CHECK_MSG(deviation <= bound,
                            std::string("linear precision, ") + format + " " + vector.label
                                + " at u=" + std::to_string(u) + ": off by "
                                + std::to_string(deviation) + ", bound "
                                + std::to_string(bound));
        }
    }
}

/// The corner cut of an interior block equals the corner cut of that block alone.
///
/// The strided sweep's own claim: output `(o, k)` depends on `values(o, ., k)` and
/// nothing else, so running a three-dimensional net through one call must agree
/// **bitwise** with running each `(o, k)` line through its own call. That is the
/// argument `pantr/bspline/structural.hpp` makes for replacing the oracle's transpose,
/// and it is the one claim in this file that is exact rather than bounded, because
/// both sides execute the same operations on the same operands.
///
/// \tparam T The storage format.
/// \param format The format's name, for a failure note.
template <class T>
void check_the_strided_sweep_is_line_by_line(const char* format) {
    const std::vector<T> knots =
        at_format<T>({0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0});
    const std::int64_t degree = 2;
    const std::int64_t num_basis = 6;
    const std::int64_t outer = 3;
    const std::int64_t inner = 4;
    std::vector<T> values(static_cast<std::size_t>(outer * num_basis * inner));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<T>(0.5) + static_cast<T>(i) * static_cast<T>(0.125);
    }
    const BsplineSpace1D<T> space(std::span<const T>(knots), degree, false,
                                  KnotSnapping::merge_near_duplicates);
    const double u = 0.4;
    const std::vector<T> together =
        corner_cut_along_axis<T>(std::span<const T>(knots), degree, space.tolerance(),
                                 std::span<const T>(values), u, outer, inner);
    PANTR_CHECK(together.size() == static_cast<std::size_t>(outer * inner));

    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t k = 0; k < inner; ++k) {
            std::vector<T> line(static_cast<std::size_t>(num_basis));
            for (std::int64_t i = 0; i < num_basis; ++i) {
                line[static_cast<std::size_t>(i)] =
                    values[static_cast<std::size_t>(o * num_basis * inner + i * inner + k)];
            }
            const std::vector<T> alone =
                corner_cut_along_axis<T>(std::span<const T>(knots), degree, space.tolerance(),
                                         std::span<const T>(line), u, 1, 1);
            PANTR_CHECK_MSG(alone[0] == together[static_cast<std::size_t>(o * inner + k)],
                            std::string("the strided sweep is not line by line at ")
                                + format + " (" + std::to_string(o) + ", "
                                + std::to_string(k) + ")");
        }
    }
}

// ---------------------------------------------------------------------------
// The open conversion
// ---------------------------------------------------------------------------

/// The `k`-th coefficient line of a one-dimensional field, sampled at `u`.
///
/// \tparam T The storage format.
/// \param field A one-dimensional field.
/// \param u The parameter.
/// \return The homogeneous components at `u`.
template <class T>
std::vector<T> sample(const Bspline<T>& field, double u) {
    return slice_point<T>(field, u);
}

/// `to_open` reaches full end multiplicity, drops periodicity, and keeps the map.
///
/// \tparam T The storage format.
/// \param format The format's name, for a failure note.
template <class T>
void check_to_open_keeps_the_map(const char* format) {
    for (const Vector1D& vector : vectors()) {
        if (vector.degree == 0) {
            // A degree-0 spline is a piecewise constant, so it is discontinuous at
            // every breakpoint -- and `sample_parameters` is mostly breakpoints. "The
            // two representations describe the same map" is then undefined exactly
            // where this would test it, since the two knot vectors disagree about
            // which side of the jump a breakpoint belongs to. Degree 0 is covered by
            // partition of unity above and, cross-backend, by
            // `tests/parity/test_bspline_structural.py`'s `p0` cases.
            continue;
        }
        const std::vector<T> knots = at_format<T>(vector.raw);
        auto space = direction<T>(knots, vector.degree, vector.periodic);
        if (space->has_open_knots() && !space->periodic()) {
            continue;  // Nothing to convert; the refusal is checked separately.
        }
        const std::vector<T> net = greville<T>(knots, vector.degree);
        // A periodic space stores one period, so the field's net is the leading
        // `num_basis` Greville abscissae of the ghost-extended vector. The map that
        // net describes is *not* the identity -- the wrap makes it periodic -- which
        // is fine: what is compared is the two representations of one map, not a
        // closed form.
        const auto stored = static_cast<std::size_t>(space->num_basis());
        const std::vector<T> values(net.begin(), net.begin() + static_cast<std::ptrdiff_t>(stored));
        const Bspline<T> original = field_of<T>({space}, values, 1, false);
        const Bspline<T> opened = to_open<T>(original);

        const BsplineSpace1D<T>& got = opened.space_ref().space_ref(0);
        PANTR_CHECK_MSG(!got.periodic(), std::string("to_open left ") + vector.label
                                             + " periodic at " + format);
        PANTR_CHECK_MSG(got.has_open_knots(), std::string("to_open left ") + vector.label
                                                  + " unclamped at " + format);
        PANTR_CHECK_MSG(got.multiplicity_in_domain().front() == vector.degree + 1
                            && got.multiplicity_in_domain().back() == vector.degree + 1,
                        std::string("to_open did not reach multiplicity degree + 1 on ")
                            + vector.label + " at " + format);

        const std::int64_t roundings =
            insertion_roundings(vector.degree) + 10 * vector.degree;
        const double magnitude = std::max(std::abs(static_cast<double>(net.front())),
                                          std::abs(static_cast<double>(net.back())));
        const double bound = bound_for<T>(roundings, magnitude);
        for (const double u : sample_parameters<T>(*space)) {
            const std::vector<T> before = sample<T>(original, u);
            const std::vector<T> after = sample<T>(opened, u);
            const double deviation = std::abs(static_cast<double>(before[0])
                                              - static_cast<double>(after[0]));
            PANTR_CHECK_MSG(deviation <= bound,
                            std::string("to_open moved the map, ") + format + " "
                                + vector.label + " at u=" + std::to_string(u) + ": by "
                                + std::to_string(deviation) + ", bound "
                                + std::to_string(bound));
        }
    }
}

/// `to_open` refuses a field that is already open in every direction.
void check_to_open_refuses_an_open_field() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const std::vector<double> values{0.0, 0.25, 0.75, 1.0};
    const Bspline<double> field = field_of<double>({direction<double>(knots, 2, false)},
                                                   values, 1, false);
    std::string message;
    try {
        static_cast<void>(to_open<double>(field));
    } catch (const std::invalid_argument& error) {
        message = error.what();
    }
    PANTR_CHECK_MSG(message == "B-spline is already open in every direction.",
                    "to_open's refusal text is \"" + message + "\"");
}

/// `to_open` leaves an already-open direction's handle alone.
///
/// The property the Python wrapper's `opened.space.spaces[d] is field.space.spaces[d]`
/// rests on, and the reason `to_open` carries the handle rather than rebuilding an
/// equal-valued space.
void check_to_open_shares_an_untouched_direction() {
    const std::vector<double> open{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const std::vector<double> unclamped{-0.2, -0.1, 0.0, 0.5, 1.0, 1.1, 1.2};
    auto first = direction<double>(open, 2, false);
    auto second = direction<double>(unclamped, 2, false);
    const std::vector<double> values(16, 1.0);
    const Bspline<double> field = field_of<double>({first, second}, values, 1, false);
    const Bspline<double> opened = to_open<double>(field);
    PANTR_CHECK_MSG(opened.space_ref().space(0) == first,
                    "to_open rebuilt a direction it did not have to touch");
    PANTR_CHECK_MSG(opened.space_ref().space(1) != second,
                    "to_open kept the direction it was supposed to convert");
}

// ---------------------------------------------------------------------------
// The split
// ---------------------------------------------------------------------------

/// `split` cuts the domain in two and keeps the map on each half.
///
/// \tparam T The storage format.
/// \param format The format's name, for a failure note.
template <class T>
void check_split_keeps_the_map(const char* format) {
    for (const Vector1D& vector : vectors()) {
        if (vector.degree == 0) {
            continue;  // Discontinuous at a breakpoint; see `check_to_open_keeps_the_map`.
        }
        const std::vector<T> knots = at_format<T>(vector.raw);
        auto space = direction<T>(knots, vector.degree, vector.periodic);
        const std::vector<T> net = greville<T>(knots, vector.degree);
        const auto stored = static_cast<std::size_t>(space->num_basis());
        const std::vector<T> values(net.begin(), net.begin() + static_cast<std::ptrdiff_t>(stored));
        const Bspline<T> original = field_of<T>({space}, values, 1, false);

        const std::array<T, 2> domain = space->domain();
        const double lo = static_cast<double>(domain[0]);
        const double hi = static_cast<double>(domain[1]);
        // Two cut points: one that is no knot at all, and one that is an interior knot
        // already -- the second exercises the `deficit` path with a head start, and a
        // knot of full multiplicity already would exercise `deficit == 0`.
        for (const double at : {lo + 0.4 * (hi - lo), lo + 0.5 * (hi - lo)}) {
            const std::pair<Bspline<T>, Bspline<T>> halves = split<T>(original, 0, at);
            const BsplineSpace1D<T>& left = halves.first.space_ref().space_ref(0);
            const BsplineSpace1D<T>& right = halves.second.space_ref().space_ref(0);

            PANTR_CHECK_MSG(!left.periodic() && !right.periodic(),
                            std::string("a half of ") + vector.label + " is periodic at "
                                + format);
            const double left_hi = static_cast<double>(left.domain()[1]);
            const double right_lo = static_cast<double>(right.domain()[0]);
            const double seam = bound_for<T>(2, std::max(std::abs(at), 1.0));
            PANTR_CHECK_MSG(std::abs(left_hi - at) <= seam && std::abs(right_lo - at) <= seam,
                            std::string("the halves of ") + vector.label
                                + " do not meet at the cut at " + format);

            const std::int64_t roundings =
                insertion_roundings(vector.degree) + 10 * vector.degree;
            const double magnitude = std::max(std::abs(lo), std::abs(hi));
            const double bound = bound_for<T>(roundings, magnitude);
            for (const double u : sample_parameters<T>(*space)) {
                const Bspline<T>& half = u <= at ? halves.first : halves.second;
                const double clamped = u <= at ? std::min(u, left_hi) : std::max(u, right_lo);
                const std::vector<T> before = sample<T>(original, clamped);
                const std::vector<T> after = sample<T>(half, clamped);
                const double deviation = std::abs(static_cast<double>(before[0])
                                                  - static_cast<double>(after[0]));
                PANTR_CHECK_MSG(deviation <= bound,
                                std::string("split moved the map, ") + format + " "
                                    + vector.label + " cut at " + std::to_string(at)
                                    + ", u=" + std::to_string(clamped) + ": by "
                                    + std::to_string(deviation) + ", bound "
                                    + std::to_string(bound));
            }
        }
    }
}

/// `split` leaves every other direction's handle alone, in both halves.
void check_split_shares_the_untouched_directions() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    auto first = direction<double>(knots, 2, false);
    auto second = direction<double>(knots, 2, false);
    const std::vector<double> values(16, 1.0);
    const Bspline<double> field = field_of<double>({first, second}, values, 1, false);
    const std::pair<Bspline<double>, Bspline<double>> halves = split<double>(field, 0, 0.25);
    PANTR_CHECK_MSG(halves.first.space_ref().space(1) == second
                        && halves.second.space_ref().space(1) == second,
                    "split rebuilt a direction it did not cut");
    PANTR_CHECK_MSG(halves.first.space_ref().space(0) != first
                        && halves.second.space_ref().space(0) != first,
                    "split kept the direction it was supposed to cut");
    PANTR_CHECK_MSG(halves.first.dim() == 2 && halves.second.dim() == 2,
                    "split changed the dimension");
}

// ---------------------------------------------------------------------------
// The slice and the boundary
// ---------------------------------------------------------------------------

/// `slice` drops the axis, keeps the other directions' handles, and agrees with a
/// direct corner cut of the same field.
void check_slice_drops_the_axis() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const std::vector<double> other{0.0, 0.0, 0.4, 1.0, 1.0};
    auto first = direction<double>(knots, 2, false);
    auto second = direction<double>(other, 1, false);
    auto third = direction<double>(knots, 2, false);
    std::vector<double> values(4 * 3 * 4 * 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.5 + static_cast<double>(i) * 0.125;
    }
    const Bspline<double> field = field_of<double>({first, second, third}, values, 2, false);

    const Bspline<double> sliced = slice<double>(field, 1, 0.4);
    PANTR_CHECK(sliced.dim() == 2);
    PANTR_CHECK_MSG(sliced.space_ref().space(0) == first
                        && sliced.space_ref().space(1) == third,
                    "slice rebuilt a surviving direction or reordered them");
    PANTR_CHECK(sliced.net().extent(0) == 4 && sliced.net().extent(1) == 4
                && sliced.net().num_components() == 2);

    // The same computation the other way round: the axis's own corner cut, strided.
    const std::vector<double> direct = corner_cut_along_axis<double>(
        std::span<const double>(other), 1, second->tolerance(), field.net().values(), 0.4, 4,
        4 * 2);
    PANTR_CHECK(direct.size() == sliced.net().size());
    bool identical = true;
    for (std::size_t i = 0; i < direct.size(); ++i) {
        identical = identical && direct[i] == sliced.net().values()[i];
    }
    PANTR_CHECK_MSG(identical, "slice and a direct strided corner cut disagree");
}

/// `slice` refuses a one-dimensional field and `slice_point` refuses a wider one.
void check_the_slice_pair_refuses_each_other_s_case() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    auto one = direction<double>(knots, 2, false);
    const Bspline<double> curve =
        field_of<double>({one}, {0.0, 0.25, 0.75, 1.0}, 1, false);
    const std::vector<double> flat(16, 1.0);
    const Bspline<double> surface = field_of<double>({one, one}, flat, 1, false);

    std::string first;
    try {
        static_cast<void>(slice<double>(curve, 0, 0.5));
    } catch (const std::invalid_argument& error) {
        first = error.what();
    }
    PANTR_CHECK_MSG(first.find("dimension at least two") != std::string::npos,
                    "slice's one-dimensional refusal reads \"" + first + "\"");

    std::string second;
    try {
        static_cast<void>(slice_point<double>(surface, 0.5));
    } catch (const std::invalid_argument& error) {
        second = error.what();
    }
    PANTR_CHECK_MSG(second == "slice_point needs a one-dimensional B-spline, got dimension 2.",
                    "slice_point's refusal reads \"" + second + "\"");
}

/// `boundary` is `slice` at a domain endpoint, and refuses a side that is neither.
void check_boundary_is_a_slice_at_an_end() {
    const std::vector<double> knots{2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0};
    auto one = direction<double>(knots, 2, false);
    std::vector<double> values(16);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 1.0 + static_cast<double>(i);
    }
    const Bspline<double> surface = field_of<double>({one, one}, values, 1, false);

    for (const int side : {0, 1}) {
        const Bspline<double> face = boundary<double>(surface, 0, side);
        const Bspline<double> want = slice<double>(surface, 0, side == 0 ? 2.0 : 4.0);
        bool identical = face.net().size() == want.net().size();
        for (std::size_t i = 0; identical && i < want.net().size(); ++i) {
            identical = face.net().values()[i] == want.net().values()[i];
        }
        PANTR_CHECK_MSG(identical, "boundary and the slice it is defined as disagree on side "
                                       + std::to_string(side));
    }

    std::string message;
    try {
        static_cast<void>(boundary<double>(surface, 0, 2));
    } catch (const std::invalid_argument& error) {
        message = error.what();
    }
    PANTR_CHECK_MSG(message == "side must be 0 or 1, got 2.",
                    "boundary's side refusal reads \"" + message + "\"");

    // Bad in both ways. The oracle checks the side first, so that is the message a
    // caller reads; the claim is in `boundary`'s own comment and had no test until now.
    std::string both;
    try {
        static_cast<void>(boundary<double>(surface, 7, 2));
    } catch (const std::invalid_argument& error) {
        both = error.what();
    }
    PANTR_CHECK_MSG(both == "side must be 0 or 1, got 2.",
                    "a bad side together with a bad axis reports \"" + both + "\"");
}

/// `slice_point` hands back the weight column of a rational field unprojected.
///
/// The division is the caller's, which is what the header and the binding both say and
/// what keeps the oracle's own numpy division in one place.
void check_slice_point_leaves_a_rational_unprojected() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    // Four coefficients of (x, w) with every weight 2, so the projection is x / 2 and
    // an accidental projection here would be visible.
    const std::vector<double> values{0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 2.0};
    const Bspline<double> curve =
        field_of<double>({direction<double>(knots, 2, false)}, values, 2, true);
    const std::vector<double> point = slice_point<double>(curve, 0.5);
    PANTR_CHECK(point.size() == 2);
    PANTR_CHECK_MSG(std::abs(point[1] - 2.0) <= bound_for<double>(10, 2.0),
                    "slice_point projected a rational field, or lost its weight");
}

// ---------------------------------------------------------------------------
// The primitives' own refusals
// ---------------------------------------------------------------------------

/// The rationality flag and the weight column survive all three operations.
///
/// The C++-only gap the parity suite covers cross-backend but this file did not: a
/// defect in how `detail::assemble` carries `is_rational`, or in a sweep that dropped
/// the last component, would show here and nowhere else in this file.
void check_the_weight_column_survives() {
    const std::vector<double> knots{-0.2, -0.1, 0.0, 0.5, 1.0, 1.1, 1.2};
    const std::vector<double> other{0.0, 0.0, 0.4, 1.0, 1.0};
    auto first = direction<double>(knots, 2, false);
    auto second = direction<double>(other, 1, false);
    // Coefficients of (x, y, w), every weight 2 so an accidental projection or a
    // dropped column shows in the value as well as in the component count.
    std::vector<double> values(4 * 3 * 3);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = (i % 3 == 2) ? 2.0 : 1.0 + static_cast<double>(i);
    }
    const Bspline<double> field = field_of<double>({first, second}, values, 3, true);
    PANTR_CHECK(field.rank() == 2);

    const Bspline<double> opened = to_open<double>(field);
    const std::pair<Bspline<double>, Bspline<double>> halves = split<double>(field, 0, 0.25);
    const Bspline<double> sliced = slice<double>(field, 1, 0.4);
    for (const Bspline<double>* result : {&opened, &halves.first, &halves.second, &sliced}) {
        PANTR_CHECK_MSG(result->is_rational(), "an operation dropped the rationality flag");
        PANTR_CHECK_MSG(result->rank() == 2, "an operation changed the rank");
        PANTR_CHECK_MSG(result->net().num_components() == 3,
                        "an operation dropped the weight column");
        // Every input weight is 2 and every operation recombines weights with
        // coefficients summing to one, so each output weight is 2 to within its own
        // rounding. A projection would give 1; a dropped column changes the count
        // above.
        const std::span<const double> got = result->net().values();
        for (std::size_t i = 2; i < got.size(); i += 3) {
            PANTR_CHECK_MSG(std::abs(got[i] - 2.0) <= bound_for<double>(30, 2.0),
                            "an operation moved a weight to " + std::to_string(got[i]));
        }
    }
}

/// The two strided primitives refuse a shape they cannot address.
void check_the_primitives_refuse_a_bad_shape() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const std::vector<double> values(4, 1.0);
    const std::vector<std::int64_t> rows{0, 1};
    const std::vector<std::int64_t> bad_row{4};
    const std::vector<double> too_short{0.0, 1.0};

    // One assertion per refusal rather than one counter for six: a counter reading
    // "accepted 1 shape(s)" does not say which of the six stopped firing.
    const auto attempt = [](const char* what, const auto& call) {
        bool refused = false;
        try {
            call();
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        PANTR_CHECK_MSG(refused, std::string("accepted ") + what);
    };

    attempt("a select_rows buffer that is not outer * num_cols * inner", [&] {
        static_cast<void>(select_rows_along_axis<double>(
            std::span<const double>(values), 4, std::span<const std::int64_t>(rows), 2, 1));
    });
    attempt("a negative extent in select_rows", [&] {
        static_cast<void>(select_rows_along_axis<double>(
            std::span<const double>(values), 4, std::span<const std::int64_t>(rows), -1, 1));
    });
    attempt("a select_rows row outside [0, num_cols)", [&] {
        static_cast<void>(select_rows_along_axis<double>(
            std::span<const double>(values), 4, std::span<const std::int64_t>(bad_row), 1, 1));
    });
    attempt("a corner-cut buffer that is not outer * num_basis * inner", [&] {
        static_cast<void>(corner_cut_along_axis<double>(std::span<const double>(knots), 2, 1e-15,
                                                        std::span<const double>(values), 0.5, 1,
                                                        2));
    });
    attempt("a negative degree in the corner cut", [&] {
        static_cast<void>(corner_cut_along_axis<double>(std::span<const double>(knots), -1,
                                                        1e-15, std::span<const double>(values),
                                                        0.5, 1, 1));
    });
    attempt("a knot vector too short for its degree in the corner cut", [&] {
        static_cast<void>(corner_cut_along_axis<double>(std::span<const double>(too_short), 2,
                                                        1e-15, std::span<const double>(values),
                                                        0.5, 1, 1));
    });
}

/// `split`'s two Layer 1 refusals carry the oracle's texts.
void check_split_refusals() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const Bspline<double> curve = field_of<double>({direction<double>(knots, 2, false)},
                                                   {0.0, 0.25, 0.75, 1.0}, 1, false);
    std::string direction_text;
    try {
        static_cast<void>(split<double>(curve, 1, 0.5));
    } catch (const std::invalid_argument& error) {
        direction_text = error.what();
    }
    PANTR_CHECK_MSG(direction_text == "direction must be in [0, 1), got 1.",
                    "split's direction refusal reads \"" + direction_text + "\"");

    for (const double at : {0.0, 1.0, -0.5}) {
        std::string value_text;
        try {
            static_cast<void>(split<double>(curve, 0, at));
        } catch (const std::invalid_argument& error) {
            value_text = error.what();
        }
        PANTR_CHECK_MSG(value_text
                            == "value must be strictly inside the domain (0.0, 1.0), got "
                                   + std::string(at == 0.0 ? "0.0" : (at == 1.0 ? "1.0" : "-0.5"))
                                   + ".",
                        "split's value refusal reads \"" + value_text + "\"");
    }
}

/// `slice`'s two Layer 1 refusals carry the oracle's texts.
void check_slice_refusals() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    auto one = direction<double>(knots, 2, false);
    const std::vector<double> flat(16, 1.0);
    const Bspline<double> surface = field_of<double>({one, one}, flat, 1, false);

    std::string axis_text;
    try {
        static_cast<void>(slice<double>(surface, 2, 0.5));
    } catch (const std::invalid_argument& error) {
        axis_text = error.what();
    }
    PANTR_CHECK_MSG(axis_text == "axis must be in [0, 2), got 2.",
                    "slice's axis refusal reads \"" + axis_text + "\"");

    std::string domain_text;
    try {
        static_cast<void>(slice<double>(surface, 1, 1.7));
    } catch (const std::invalid_argument& error) {
        domain_text = error.what();
    }
    PANTR_CHECK_MSG(domain_text == "value 1.7 is outside the domain [0.0, 1.0] of direction 1.",
                    "slice's domain refusal reads \"" + domain_text + "\"");
}

/// `open_along_axis` refuses an axis that is already clamped and not periodic.
void check_open_along_axis_refuses_a_clamped_axis() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const std::vector<double> values{0.0, 0.25, 0.75, 1.0};
    std::string message;
    try {
        static_cast<void>(open_along_axis<double>(std::span<const double>(knots), 2, false,
                                                  1e-15, std::span<const double>(values), 4, 1,
                                                  1));
    } catch (const std::invalid_argument& error) {
        message = error.what();
    }
    PANTR_CHECK_MSG(
        message
            == "B-spline is already open (both boundary knots have multiplicity degree + 1).",
        "open_along_axis's refusal reads \"" + message + "\"");
}

}  // namespace

int main() {
    check_corner_cut_sums_to_one<double>("float64");
    check_corner_cut_sums_to_one<float>("float32");
    check_corner_cut_is_linearly_precise<double>("float64");
    check_corner_cut_is_linearly_precise<float>("float32");
    check_the_strided_sweep_is_line_by_line<double>("float64");
    check_the_strided_sweep_is_line_by_line<float>("float32");

    check_to_open_keeps_the_map<double>("float64");
    check_to_open_keeps_the_map<float>("float32");
    check_to_open_refuses_an_open_field();
    check_to_open_shares_an_untouched_direction();

    check_split_keeps_the_map<double>("float64");
    check_split_keeps_the_map<float>("float32");
    check_split_shares_the_untouched_directions();

    check_slice_drops_the_axis();
    check_the_slice_pair_refuses_each_other_s_case();
    check_boundary_is_a_slice_at_an_end();
    check_slice_point_leaves_a_rational_unprojected();
    check_the_weight_column_survives();

    check_the_primitives_refuse_a_bad_shape();
    check_split_refusals();
    check_slice_refusals();
    check_open_along_axis_refuses_a_clamped_axis();

    return pantr::test::summary("test_bspline_structural");
}
