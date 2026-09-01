#pragma once

/// \file
/// The knot vector: its tolerance, its classes, and the refusals a space owes.
///
/// Everything here is a free function over a knot span and a degree, deliberately:
/// `BsplineSpace1D` calls all of it from its constructor, and a free function
/// taking the frozen state can be tested without an object and cannot read a
/// half-initialised `this`. `design/bspline_derived_caches.md` asks for exactly
/// that shape for the derived block.
///
/// ## One tolerance, and where it comes from
///
/// `knot_tolerance` is `8 * eps(T) * knot_scale(knots)`, and every parametric
/// comparison the B-spline layer makes is a comparison of quantities of that
/// magnitude: are these two knots one knot, is this point inside the domain, are
/// these two knot intervals the same length. The derivation is the oracle's and is
/// written out in `src/pantr/bspline/_bspline_knots.py`: two knots reached by
/// different routes differ by at most `4 * eps * scale` by the triangle
/// inequality, and the factor is that bound doubled, for a contraction, a
/// different vector width and one further rounding upstream. The scale carries the
/// coordinate magnitudes as well as the span, because on a knot vector based at
/// 1e6 a knot difference is formed from two coordinates of that size.
///
/// **The tolerance is a `double` at every width, and that is not an oversight.**
/// The oracle computes it in Python floats and passes it into a numba kernel typed
/// `float64` even when the knots are `float32`, so the comparison there widens the
/// `float32` difference and tests it against a `double` threshold. Storing it as
/// `T` would round the threshold to `float` and change the predicate in the last
/// bits, which is a knot-classification verdict rather than a rounding.
///
/// ## The two arithmetics, which are not the same arithmetic
///
/// This is the trap in porting this module, so it is stated once here and again at
/// each site. The oracle uses **two** ways of differencing two knots, and they
/// differ at `float32`:
///
///  - **In `T`, then widened.** The class scan and the boundary multiplicity count
///    are numba expressions over a `float32[:]` array, so the subtraction is
///    `float32` and only the comparison against `tol` widens. Reproduced here as
///    `value_of(T(a - b))` cast to `double`. Verified rather than assumed, by
///    reading numba's own typing: `njit(lambda k: k[1] - k[0])` on a `float32`
///    array reports a `float32` return type, and the comparison against a Python
///    float takes `(float32[:], float64) -> bool`.
///  - **Widened, then in `double`.** `_left_end_open`, `_right_end_open` and
///    `_knot_scale` are Python expressions written `float(a) - float(b)`, so both
///    operands widen first and the subtraction happens in `double`. Reproduced
///    here as `as_double(a) - as_double(b)`.
///
/// **They disagree about a class boundary only inside a hairline band at the
/// threshold itself, and the argument is worth having rather than trusting.** An
/// earlier draft of this comment opened with "cannot disagree", which its own
/// derivation already contradicted and which a review refuted: fuzzing the two
/// spellings against each other found real `float32` disagreements, all within the
/// last ulp of `tol`, at about 0.05% of trials drawn deliberately onto the
/// threshold and none at `float64`. A verdict flips only if the two differences
/// straddle `tol`. Two knots whose difference is near `tol = 8 * eps * scale` and whose own
/// magnitudes are near `scale` are within a relative `8 * eps` of each other, so
/// Sterbenz makes the storage-width subtraction *exact* and the two agree
/// identically. Where the operands are small against `scale` -- a pair straddling
/// zero on a vector whose scale comes from elsewhere -- the subtraction is an
/// addition and its error is at most half an ulp of the result, so a flip needs the
/// exact difference to lie within a relative `2^-24` of the threshold. That is
/// derived, and then measured twice with results that agree. A differential sweep of
/// 3000 *randomly drawn* knot vectors against the oracle, with the class scan
/// written both ways, reported 0 mismatches for either spelling -- random draws do
/// not land in a band of relative width `2^-24`. Fuzzing drawn deliberately onto the
/// threshold does land in it, and finds the disagreements above.
///
/// So the site is a faithfulness question rather than a correctness one, but not a
/// vacuous one: the arithmetic here is the oracle's, and a test that pinned the
/// difference would be pinning which of two defensible answers the last ulp gets.
///
/// ## Messages
///
/// Every refusal below carries the oracle's message character for character, so
/// that the two backends are indistinguishable to a caller who reads the text and
/// so that `tests/parity/` can compare them. `pantr/core/format.hpp` carries the
/// three Python format specifiers they interpolate.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

namespace detail {

/// The floating-point value of a scalar, as a `double`.
///
/// Tier B, so it goes through the two-step `value_of` lookup rather than casting
/// the scalar. Used wherever the oracle wrote `float(x)`.
///
/// \param x The scalar.
/// \return Its value, widened.
template <Real T>
[[nodiscard]] constexpr double as_double(const T& x) noexcept {
    using pantr::value_of;
    return static_cast<double>(value_of(x));
}

/// The numpy name of `T`'s storage format, as the oracle's messages spell it.
///
/// Local to this header because exactly one message interpolates it. Promote it
/// to the core when a second one does.
///
/// \return `"float32"` or `"float64"`.
template <Real T>
[[nodiscard]] constexpr const char* dtype_name() noexcept {
    if constexpr (std::same_as<value_type_t<T>, float>) {
        return "float32";
    } else {
        return "float64";
    }
}

/// Machine epsilon of `T`'s value component.
///
/// \return `eps`, as a `double`.
template <Real T>
[[nodiscard]] constexpr double epsilon() noexcept {
    return static_cast<double>(std::numeric_limits<value_type_t<T>>::epsilon());
}

/// Machine epsilons a strict parametric comparison allows for.
///
/// The oracle's `pantr.tolerance._STRICT_EPS_FACTOR`, times its
/// `_bspline_knots._KNOT_MERGE_SAFETY`: four for a value reached by one convex
/// combination and differenced against another such value, doubled for a
/// contraction, a different vector width and one rounding upstream. Both are
/// powers of two, so the product is exact and the only rounding in
/// `knot_tolerance` is its final multiply by the scale -- which is what lets the
/// two backends agree on the threshold to the last bit rather than merely to
/// within it.
inline constexpr double kKnotToleranceEpsilons = 8.0;

}  // namespace detail

/// The magnitude a parametric comparison on a knot vector is relative to.
///
/// The span alone is not enough: on a knot vector based at 1e6 a knot difference
/// is formed from two coordinates of that size, so it carries an absolute error of
/// `eps * 1e6` however short the span is. The coordinate magnitudes are therefore
/// taken alongside the span, and the largest wins.
///
/// No floor is applied. A floor of 1 would be a physical choice this layer is not
/// entitled to make, and it would destroy scale covariance on a domain smaller
/// than one unit -- exactly the case where an absolute tolerance merged every knot
/// in the vector.
///
/// \param knots A non-decreasing, non-empty knot vector.
/// \return `max(span, |knots[0]|, |knots[n-1]|)`, zero only for a vector that is
///         identically zero.
template <Real T>
[[nodiscard]] double knot_scale(std::span<const T> knots) {
    // Unqualified, with the using-declaration, per the rule in
    // `pantr/core/scalar.hpp`; these arguments are already `double`, and the
    // discipline guard in `scripts/ci_local.sh` is spelling-based by design.
    using std::abs;
    // `float(hi) - float(lo)`: the oracle widens both ends before subtracting.
    const double lo = detail::as_double(knots.front());
    const double hi = detail::as_double(knots.back());
    return std::max({hi - lo, abs(lo), abs(hi)});
}

/// The absolute tolerance for comparing parametric coordinates of a knot vector.
///
/// See the file comment for the derivation and for why the result is a `double` at
/// every width.
///
/// \param knots A non-decreasing, non-empty knot vector.
/// \return `8 * eps(T) * knot_scale(knots)`, in the units of `knots`.
template <Real T>
[[nodiscard]] double knot_tolerance(std::span<const T> knots) {
    return detail::kKnotToleranceEpsilons * detail::epsilon<T>() * knot_scale(knots);
}

/// Where the knot classes of a vector begin and end, without allocating.
///
/// A class is a maximal run of knots no two consecutive members of which differ by
/// more than the tolerance. The domain range names the classes holding the two
/// domain ends, whose whole class is reported -- clamped copies outside the domain
/// included, which is what the multiplicity of a boundary knot means.
struct KnotClassRange {
    std::int64_t count = 0;         ///< How many classes the whole vector has.
    std::int64_t domain_begin = 0;  ///< The class holding `knots[degree]`.
    std::int64_t domain_end = 0;    ///< One past the class holding `knots[n-degree-1]`.

    /// The number of in-domain knot intervals the range describes.
    ///
    /// \return One less than the number of in-domain classes; never negative,
    ///         because the domain always spans at least one class.
    [[nodiscard]] constexpr std::int64_t num_intervals() const noexcept {
        return domain_end - domain_begin - 1;
    }
};

/// Locate the knot classes of a non-decreasing knot vector.
///
/// Knots are grouped by *gap*: the vector is non-decreasing, so one pass suffices,
/// and a class ends where the step to the next knot exceeds `tol`. Grouping by gap
/// rather than by a rounding grid is what makes the answer depend on the knots and
/// not on where they sit relative to an origin: two values one ulp apart can
/// straddle a grid line and never merge, for any grid spacing.
///
/// The cost of the gap rule is chaining -- `n` knots each within `tol` of the next
/// collapse into one class however far apart the ends are. That is accepted rather
/// than capped: a class-width cap would break idempotence, since a class split by
/// the cap leaves two representatives that may be within `tol` of each other and
/// would merge on the next pass.
///
/// \param knots A non-decreasing knot vector of at least `2 * degree + 2` entries.
/// \param degree The polynomial degree, which fixes where the domain starts and ends.
/// \param tol The absolute parametric tolerance, from `knot_tolerance`.
/// \return The class count and the in-domain class range.
template <Real T>
[[nodiscard]] KnotClassRange classify_knots(std::span<const T> knots, std::int64_t degree,
                                            double tol) {
    const auto n = static_cast<std::int64_t>(knots.size());
    const std::int64_t lo_index = degree;
    const std::int64_t hi_index = n - degree - 1;

    KnotClassRange range;
    std::int64_t klass = 0;
    std::int64_t lo_class = 0;
    std::int64_t hi_class = 0;
    for (std::int64_t i = 1; i < n; ++i) {
        // Differenced in `T` and only then widened: the oracle's scan is a numba
        // expression over the knot array. See the file comment.
        const T step = knots[static_cast<std::size_t>(i)] - knots[static_cast<std::size_t>(i - 1)];
        if (detail::as_double(step) > tol) {
            ++klass;
        }
        if (i == lo_index) {
            lo_class = klass;
        }
        if (i == hi_index) {
            hi_class = klass;
        }
    }
    range.count = klass + 1;
    range.domain_begin = lo_class;
    range.domain_end = hi_class + 1;
    return range;
}

/// The distinct knots of a vector and their multiplicities.
///
/// Every member of a class is represented by the class's **first** knot, chosen
/// rather than averaged, so the values returned are knots the vector actually
/// contains and the grouping is idempotent: regrouping the represented vector
/// reproduces it. An average would be a fresh rounding, and could place two
/// adjacent classes' representatives a single ulp apart.
///
/// The multiplicities sum to `knots.size()`, so repeating each representative by
/// its multiplicity rebuilds the whole vector with each class collapsed onto it.
template <Real T>
struct KnotClasses {
    std::vector<T> unique;                  ///< One representative per class.
    std::vector<std::int64_t> multiplicity;  ///< How many knots each class holds.
    std::size_t domain_begin = 0;           ///< First in-domain class.
    std::size_t domain_end = 0;             ///< One past the last in-domain class.
};

/// Build the distinct knots and their multiplicities.
///
/// The in-domain form the oracle offers is the contiguous subrange
/// `[domain_begin, domain_end)` of these arrays rather than a second computation,
/// which is why there is one memo here and not two.
///
/// \param knots A non-decreasing knot vector of at least `2 * degree + 2` entries.
/// \param degree The polynomial degree.
/// \param tol The absolute parametric tolerance, from `knot_tolerance`.
/// \return The classes, with the in-domain subrange marked.
template <Real T>
[[nodiscard]] KnotClasses<T> unique_knots_and_multiplicity(std::span<const T> knots,
                                                           std::int64_t degree, double tol) {
    const KnotClassRange range = classify_knots(knots, degree, tol);

    KnotClasses<T> classes;
    classes.unique.reserve(static_cast<std::size_t>(range.count));
    classes.multiplicity.assign(static_cast<std::size_t>(range.count), 0);
    classes.domain_begin = static_cast<std::size_t>(range.domain_begin);
    classes.domain_end = static_cast<std::size_t>(range.domain_end);

    std::size_t klass = 0;
    classes.unique.push_back(knots.front());
    classes.multiplicity[0] = 1;
    for (std::size_t i = 1; i < knots.size(); ++i) {
        const T step = knots[i] - knots[i - 1];
        if (detail::as_double(step) > tol) {
            ++klass;
            classes.unique.push_back(knots[i]);
        }
        ++classes.multiplicity[klass];
    }
    return classes;
}

/// The number of basis functions a knot vector and degree define.
///
/// In the non-periodic case it is the knot count less the degree less one. In the
/// periodic case that count is reduced by the regularity at the domain's start,
/// which is the degree less the multiplicity of the first in-domain knot.
///
/// \param knots A non-decreasing knot vector of at least `2 * degree + 2` entries.
/// \param degree The polynomial degree.
/// \param periodic Whether the space is periodic.
/// \param tol The absolute parametric tolerance, from `knot_tolerance`.
/// \return The number of basis functions, which the caller must still check is at
///         least `degree + 1`.
template <Real T>
[[nodiscard]] std::int64_t num_basis(std::span<const T> knots, std::int64_t degree, bool periodic,
                                     double tol) {
    auto count = static_cast<std::int64_t>(knots.size()) - degree - 1;
    if (!periodic) {
        return count;
    }

    using std::abs;
    // The multiplicity of `knots[degree]` among the first `degree + 1` knots.
    // Differenced in `T`, as the oracle's numba expression is.
    const T first = knots[static_cast<std::size_t>(degree)];
    std::int64_t multiplicity = 0;
    for (std::int64_t i = 0; i <= degree; ++i) {
        const T offset = knots[static_cast<std::size_t>(i)] - first;
        if (abs(detail::as_double(offset)) <= tol) {
            ++multiplicity;
        }
    }
    const std::int64_t regularity = degree - multiplicity;
    return count - regularity - 1;
}

/// Collapse each group of knots meant to be one knot onto a single stored value.
///
/// The grouping is the one `classify_knots` reports, so a space and its own
/// accessor cannot disagree about which knots are the same knot -- two
/// implementations of one idea is how they came to disagree before.
///
/// \param knots A non-decreasing knot vector.
/// \param tol The absolute parametric tolerance, from `knot_tolerance`.
/// \return A vector of the same length, dtype and ordering, with each class
///         collapsed onto its first knot.
template <Real T>
[[nodiscard]] std::vector<T> snap_knots(std::span<const T> knots, double tol) {
    std::vector<T> snapped(knots.begin(), knots.end());
    T representative = knots.front();
    for (std::size_t i = 1; i < knots.size(); ++i) {
        // The class boundaries are decided on the ORIGINAL vector, so the step is
        // taken from `knots` and never from the partly-rewritten `snapped`.
        const T step = knots[i] - knots[i - 1];
        if (detail::as_double(step) > tol) {
            representative = knots[i];
        }
        snapped[i] = representative;
    }
    return snapped;
}

/// Refuse a knot vector that snapping collapsed onto a single point.
///
/// This refuses one specific cause -- the knots *were* distinct, and the merge rule
/// found none of them distinguishable at their own magnitude -- so that the caller
/// is told the thing they could not see coming. A vector that arrived already flat
/// is not this function's case and passes through untouched; it is refused a step
/// later by `check_space_has_an_interval`, which owns the *consequence* both causes
/// share. Keeping the two apart is what lets each message be true.
///
/// This is reachable, and the arithmetic that makes it so is not a defect. A mesh
/// of `n` intervals over a span `s` at offset `x` survives only while
/// `s / n > 8 * eps * max(s, |x|)`. When `|x|` dominates, that fails once `|x| / s`
/// reaches `1 / (8 * eps * n)` -- about `5.2e5 / n` at `float32` and `5.6e14 / n` at
/// `float64`. Past it no threshold satisfies both requirements at once, because an
/// interior knot is then uncertain by a sizeable fraction of the span.
///
/// \param raw The knot vector as supplied, non-decreasing, before snapping.
/// \param snapped The same vector afterwards.
/// \param tol The absolute tolerance snapping used, from `knot_tolerance`.
/// \throws std::invalid_argument If `raw` held more than one distinct knot while
///         `snapped` holds only one.
template <Real T>
void check_snapping_kept_an_interval(std::span<const T> raw, std::span<const T> snapped,
                                     double tol) {
    using pantr::value_of;
    // Both vectors are non-decreasing, so "all knots identical" is exactly "first
    // equals last", tested bitwise and needing no tolerance of its own.
    if (value_of(raw.front()) == value_of(raw.back())
        || value_of(snapped.front()) != value_of(snapped.back())) {
        return;
    }

    const double scale = knot_scale(raw);
    // `np.spacing(raw.dtype.type(scale))`: the scale is rounded INTO the knot
    // format first, and the ulp is that format's, not the accumulator's.
    const auto scale_in_storage = static_cast<value_type_t<T>>(scale);
    const double ulp = static_cast<double>(
        std::nextafter(scale_in_storage, std::numeric_limits<value_type_t<T>>::infinity())
        - scale_in_storage);

    // `np.diff(raw.astype(float64))`: widened first, so this difference is the
    // `double` one and not the storage-width one. A positive gap exists because the
    // vector is non-decreasing and its ends differ.
    double closest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < raw.size(); ++i) {
        const double gap = detail::as_double(raw[i]) - detail::as_double(raw[i - 1]);
        if (gap > 0.0) {
            closest = std::min(closest, gap);
        }
    }

    const std::string name = detail::dtype_name<T>();
    // Widening the format is only a remedy if there is a wider one to move to.
    const std::string remedy =
        std::same_as<value_type_t<T>, float>
            ? "Use float64, move the domain nearer the origin, or coarsen the mesh."
            : "Move the domain nearer the origin, or coarsen the mesh.";

    throw std::invalid_argument(
        "knot snapping collapsed every knot onto "
        + pantr::detail::format_scalar(snapped.front()) + ": in " + name + " at |coordinate| ~ "
        + pantr::detail::format_general(scale, 3) + " two knots are the same knot unless they "
        + "differ by more than " + pantr::detail::format_general(tol, 3) + " ("
        + pantr::detail::format_fixed(tol / ulp, 0) + " ulp there), and the closest pair in ["
        + pantr::detail::format_scalar(raw.front()) + ", "
        + pantr::detail::format_scalar(raw.back()) + "] is "
        + pantr::detail::format_general(closest, 3) + " apart. This mesh is finer than " + name
        + " resolves at that magnitude. " + remedy);
}

/// Refuse a knot vector whose domain is a single point.
///
/// A space with no interval has no cell, and every operation defined over its cells
/// is therefore undefined on it. The rule lives at the one place a space comes into
/// being rather than at each consumer, because three consumers met the degeneracy
/// by *returning a value* rather than raising, and a survey of what raised would
/// not have found them.
///
/// The predicate is the interval count, not "every knot is the same value". The
/// family is wider than it looks: `[0, 1, 1, 1, 2]` at degree 1 holds three
/// distinct values and still has a domain of one point, because an interior knot of
/// high enough multiplicity swallows it whole.
///
/// \param knots The knot vector, non-decreasing and already snapped if snapping was
///        requested.
/// \param degree The polynomial degree, which fixes where the domain starts and ends.
/// \param num_intervals The space's interval count, from `KnotClassRange`.
/// \param tol The absolute tolerance the count was taken at.
/// \throws std::invalid_argument If `num_intervals` is zero.
template <Real T>
void check_space_has_an_interval(std::span<const T> knots, std::int64_t degree,
                                 std::int64_t num_intervals, double tol) {
    if (num_intervals > 0) {
        return;
    }

    const std::int64_t last = static_cast<std::int64_t>(knots.size()) - degree - 1;
    const std::string tol_text = pantr::detail::format_general(tol, 3);
    throw std::invalid_argument(
        "knot vector spans no interval: at degree " + std::to_string(degree)
        + " the domain runs from knots[" + std::to_string(degree) + "] = "
        + pantr::detail::format_scalar(knots[static_cast<std::size_t>(degree)]) + " to knots["
        + std::to_string(last) + "] = "
        + pantr::detail::format_scalar(knots[static_cast<std::size_t>(last)])
        + ", and every step between them is at most this vector's tolerance of " + tol_text
        + ", so its in-domain knots are all one knot, the space has no cell, and nothing can be "
          "evaluated, tabulated or located on it. The domain needs two consecutive knots more "
          "than "
        + tol_text + " apart.");
}

}  // namespace pantr::bspline
