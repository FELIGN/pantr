#pragma once

/// \file
/// The 1D B-spline space: a knot vector, a degree, and the quantities they fix.
///
/// ## What this type owns, and what it does not
///
/// It owns the *value* -- the knot vector, the degree, the periodicity flag and
/// the parametric tolerance those imply -- plus every quantity that is a function
/// of them: how many basis functions there are, how many intervals, where the
/// domain runs, whether the ends are clamped, the distinct knots and their
/// multiplicities, and the first supported basis index of each interval.
///
/// It owns no *operations*. Basis tabulation, the three families of extraction
/// operator, knot insertion, subdivision and restriction are separate ports over
/// free functions taking a `const BsplineSpace1D&`, exactly as
/// `pantr/bezier/bezier.hpp` splits the Bézier value from evaluation, elevation
/// and the product. That is what lets them proceed independently of one another
/// once this type exists, and it is why `get_cardinal_intervals` -- a computation
/// over the knots rather than a property of them -- is not here.
///
/// ## Validating rather than asserting
///
/// This is not a kernel but the C++ counterpart of Layer 2, so it validates and
/// throws in a release build as much as a debug one. A caller with no Python
/// cannot be protected by `cpp/bindings/`. `pantr/core/error.hpp` sets the split
/// the whole port uses: type-kind checks stay in the Python wrapper, because
/// nanobind has no path to `TypeError`; value and range checks live here.
///
/// ## Parity notes for the Python oracle
///
/// `pantr.bspline.BsplineSpace1D` is the oracle. What this type reproduces, and
/// the two places it deliberately differs:
///
///  - **The order of the checks is the oracle's**, and it is load-bearing rather
///    than incidental: two simultaneously bad arguments must produce the same
///    message on both sides, and only the order decides which one they get.
///  - **The messages are the oracle's, character for character**, including the
///    two that interpolate a formatted float. `pantr/core/format.hpp` carries the
///    three Python format specifiers involved.
///  - **The tolerance is derived from the knots as supplied, before snapping**,
///    and is never recomputed afterwards -- which matters because snapping can
///    move the last knot onto its class's first one and so change the scale. The
///    oracle computes it once in `__init__` and stores it; so does this.
///  - **`num_basis` is computed from the snapped knots and the raw tolerance.**
///    The oracle's `_validate_input` computes a *different* basis count, from the
///    raw knots, purely to refuse a vector that cannot support the degree; the
///    stored count comes from the snapped vector. The two can differ, because
///    snapping moves knots onto class representatives and so can change the
///    multiplicity of the first in-domain knot. Both are reproduced, in that
///    order.
///  - **The derived arrays are built at most once and shared, not copied.** The
///    oracle memoises the same quantities per object with `functools.cached_property`
///    and an `lru_cache`; the latter was a process-global cache and is gone.
///  - **Both sides copy the knot vector, and there is no divergence here.** An
///    earlier draft of this comment said the oracle stores a floating-point array
///    in place and that only this type copies. That is false:
///    `_bspline_space_1d.py:99-106` copies too, and says why -- it freezes the
///    vector read-only afterwards, and freezing a caller's array in place would
///    mutate caller state. Recorded because the claim was written down and
///    believed.
///
///    `pantr/bezier/bezier.hpp:60-65` makes the structurally similar claim about a
///    control net, and **that one is true** -- checked by execution rather than
///    assumed, because the resemblance is exactly what made this one look safe.
///    `_BezierPython.__init__` stores what `numpy.asarray` returns, which does not
///    copy an array that already has the right dtype, and a write through the
///    caller's array is visible in the Bezier afterwards. The two oracles genuinely
///    differ: a space copies because it freezes its knots read-only, and a Bezier
///    does not, which is the defect FELIGN/pantr#375 fixes.
///
/// ## Thread safety
///
/// Every accessor below is safe to call concurrently on the same object with no
/// external locking, including the ones that fill the derived block: it is a
/// `pantr::LazySlot`, whose file comment carries the measurement behind that
/// choice. In a sweep, hoist `unique_knots()` or `first_basis_per_interval()` into
/// a local span before the loop rather than calling it per element.

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "pantr/bspline/knots.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// What the constructor does with knots that are the same knot.
///
/// A `bool` here would read `BsplineSpace1D(knots, 2, false, true)` at the call
/// site, where neither flag is recoverable without opening the header.
enum class KnotSnapping : std::uint8_t {
    /// Collapse each class of near-duplicate knots onto its first member, and
    /// refuse a vector that this collapses onto a single point. The oracle's
    /// `snap_knots=True`, and its default.
    merge_near_duplicates = 0,
    /// Store the knots as supplied. The interval requirement still applies; only
    /// the merging and the snapping diagnosis are skipped. The oracle's
    /// `snap_knots=False`.
    as_given = 1,
};

/// A 1D B-spline space: a knot vector and a polynomial degree.
///
/// An instance always has at least one interval. A knot vector whose in-domain
/// knots all fall in one class is refused at construction, since such a space has
/// no cell and nothing can be evaluated, tabulated or located on it.
///
/// Instances are immutable: no operation changes an existing space, and every
/// derived one is returned by value. The one mutable member is the derived block,
/// whose fill no caller can observe -- it yields the same values and the same
/// spans however many times it runs -- which is the test `CLAUDE.md` sets for a
/// reasoned exception to construct-then-freeze.
template <Real T>
class BsplineSpace1D {
  public:
    /// Build a space from a knot vector and a degree.
    ///
    /// \param knots The knot vector: non-decreasing, at least `2 * degree + 2`
    ///        entries. Copied; the caller keeps its own storage.
    /// \param degree The polynomial degree, non-negative.
    /// \param periodic Whether the space is periodic.
    /// \param snapping What to do with knots that are the same knot.
    /// \throws std::invalid_argument If the degree is negative, the vector is too
    ///         short or not non-decreasing, the vector cannot support the degree,
    ///         snapping collapsed a vector that had more than one distinct knot,
    ///         or the resulting space spans no interval.
    BsplineSpace1D(std::span<const T> knots, std::int64_t degree, bool periodic,
                   KnotSnapping snapping)
        : degree_(degree), periodic_(periodic) {
        check_arguments(knots, degree, periodic);

        // Derived from the knots as supplied, and never recomputed: snapping can
        // move the last knot onto its class's first one, which would move the
        // scale. See the parity note in the file comment.
        tol_ = knot_tolerance(knots);

        if (snapping == KnotSnapping::merge_near_duplicates) {
            knots_ = snap_knots(knots, tol_);
            // Runs first: when snapping is what destroyed the mesh, its diagnosis
            // is the specific one and names the remedy. The interval check below
            // owns the same symptom from any other cause.
            check_snapping_kept_an_interval<T>(knots, std::span<const T>(knots_), tol_);
        } else {
            knots_.assign(knots.begin(), knots.end());
        }

        const KnotClassRange range = classify_knots<T>(knots_, degree_, tol_);
        num_intervals_ = range.num_intervals();
        check_space_has_an_interval<T>(knots_, degree_, num_intervals_, tol_);
        num_basis_ = pantr::bspline::num_basis<T>(knots_, degree_, periodic_, tol_);
    }

    /// The knot vector, as stored.
    ///
    /// \return A view of this space's own storage, valid while it lives.
    [[nodiscard]] std::span<const T> knots() const noexcept {
        return std::span<const T>(knots_);
    }

    /// The polynomial degree.
    ///
    /// \return The degree, non-negative.
    [[nodiscard]] std::int64_t degree() const noexcept { return degree_; }

    /// Whether the space is periodic.
    ///
    /// \return `true` for a periodic space.
    [[nodiscard]] bool periodic() const noexcept { return periodic_; }

    /// The absolute tolerance for parametric comparisons on this space.
    ///
    /// In the units of the knot vector, being a dimensionless strict tier scaled by
    /// the knot vector's own magnitude. Two knots closer than this are the same
    /// knot, a point within this of an end is inside the domain, and two knot
    /// intervals differing by less than this are the same length.
    ///
    /// Being an absolute *parametric* length, it is not the factor to scale a
    /// physical coordinate or a spline coefficient by.
    ///
    /// \return The tolerance, a `double` at every knot width. See
    ///         `pantr/bspline/knots.hpp` for why.
    [[nodiscard]] double tolerance() const noexcept { return tol_; }

    /// The number of basis functions.
    ///
    /// \return The knot count less the degree less one, reduced in the periodic
    ///         case by the regularity at the domain's start.
    [[nodiscard]] std::int64_t num_basis() const noexcept { return num_basis_; }

    /// The number of intervals in the domain.
    ///
    /// \return One less than the number of distinct in-domain knots; at least 1,
    ///         because the constructor refuses a space with none.
    [[nodiscard]] std::int64_t num_intervals() const noexcept { return num_intervals_; }

    /// The knot vector's domain.
    ///
    /// Two indexed reads rather than a stored pair: an eager field and this are the
    /// same computation, and not duplicating state is the tiebreak.
    ///
    /// \return `{knots[degree], knots[n - degree - 1]}`.
    [[nodiscard]] std::array<T, 2> domain() const noexcept {
        const auto first = static_cast<std::size_t>(degree_);
        const std::size_t last = knots_.size() - first - 1;
        return {knots_[first], knots_[last]};
    }

    /// Whether the left end is clamped.
    ///
    /// The comparison is **absolute and in `double`**: the oracle writes
    /// `abs(float(a) - float(b)) <= tol`, so both operands widen before the
    /// subtraction. A relative comparison would read a gap of up to
    /// `1e-5 * |knots[degree]|` as clamped -- half the domain, on a knot vector
    /// based at 1e6.
    ///
    /// \return `true` if the first `degree + 1` knots are equal to within the
    ///         tolerance; always `false` for a periodic space, whatever its knots.
    [[nodiscard]] bool has_left_end_open() const noexcept {
        using std::abs;
        if (periodic_) {
            return false;
        }
        const double gap = detail::as_double(knots_.front())
                           - detail::as_double(knots_[static_cast<std::size_t>(degree_)]);
        return abs(gap) <= tol_;
    }

    /// Whether the right end is clamped.
    ///
    /// Absolute and in `double`, as `has_left_end_open` is.
    ///
    /// \return `true` if the last `degree + 1` knots are equal to within the
    ///         tolerance; always `false` for a periodic space.
    [[nodiscard]] bool has_right_end_open() const noexcept {
        using std::abs;
        if (periodic_) {
            return false;
        }
        const std::size_t first = knots_.size() - static_cast<std::size_t>(degree_) - 1;
        const double gap = detail::as_double(knots_[first]) - detail::as_double(knots_.back());
        return abs(gap) <= tol_;
    }

    /// Whether both ends are clamped.
    ///
    /// \return `true` if the space is open at both ends.
    [[nodiscard]] bool has_open_knots() const noexcept {
        return has_left_end_open() && has_right_end_open();
    }

    /// Whether the knots describe a single Bézier segment.
    ///
    /// \return `true` iff the space is non-periodic, clamped at both ends, and has
    ///         exactly `degree + 1` basis functions.
    [[nodiscard]] bool has_bezier_like_knots() const noexcept {
        return !periodic_ && has_left_end_open() && has_right_end_open()
               && num_basis_ == degree_ + 1;
    }

    /// The distinct knots of the whole vector.
    ///
    /// \return One representative per class, in order; a view of the derived block.
    [[nodiscard]] std::span<const T> unique_knots() const {
        return std::span<const T>(derived().classes.unique);
    }

    /// The multiplicities of the distinct knots of the whole vector.
    ///
    /// \return One count per class, summing to `knots().size()`.
    [[nodiscard]] std::span<const std::int64_t> multiplicity() const {
        return std::span<const std::int64_t>(derived().classes.multiplicity);
    }

    /// The distinct knots inside the domain.
    ///
    /// A subrange of `unique_knots()` rather than a second computation, which is
    /// why there is one memo here and not two.
    ///
    /// \return The in-domain representatives, `num_intervals() + 1` of them.
    [[nodiscard]] std::span<const T> unique_knots_in_domain() const {
        const KnotClasses<T>& classes = derived().classes;
        return std::span<const T>(classes.unique)
            .subspan(classes.domain_begin, classes.domain_end - classes.domain_begin);
    }

    /// The multiplicities of the distinct knots inside the domain.
    ///
    /// The whole class of each boundary knot is reported, clamped copies outside
    /// the domain included, which is what the multiplicity of a boundary knot
    /// means.
    ///
    /// \return The in-domain counts, `num_intervals() + 1` of them.
    [[nodiscard]] std::span<const std::int64_t> multiplicity_in_domain() const {
        const KnotClasses<T>& classes = derived().classes;
        return std::span<const std::int64_t>(classes.multiplicity)
            .subspan(classes.domain_begin, classes.domain_end - classes.domain_begin);
    }

    /// The index of the first basis function supported on each interval.
    ///
    /// Entry `j` is the smallest global function index `i` whose function is
    /// non-zero on the open interval between in-domain unique knots `j` and
    /// `j + 1`. The functions non-zero there are exactly `i` through `i + degree`,
    /// so this one index describes the whole support of the interval.
    ///
    /// It is exact integer arithmetic and no basis evaluation: the functions
    /// non-zero on the span starting at knot index `k` are `k - degree` through
    /// `k`, where `k` is the *last* position at which that unique knot occurs.
    /// Selecting the unique knots whose last position lies in `[degree, num_basis)`
    /// picks out exactly the knots that start an in-domain interval, which makes
    /// the first interval no different from the rest and needs no special case for
    /// a non-clamped knot vector.
    ///
    /// \return A view of `num_intervals()` non-decreasing indices, whose successive
    ///         differences are the interior knot multiplicities.
    /// \throws std::invalid_argument If the space is periodic.
    [[nodiscard]] std::span<const std::int64_t> first_basis_per_interval() const {
        if (periodic_) {
            throw std::invalid_argument(
                "first_basis_per_interval: periodic B-spline spaces are not supported.");
        }
        return std::span<const std::int64_t>(derived().first_basis);
    }

  private:
    /// Everything derived from the knots that allocates. One struct, one flag, one
    /// fill: grouping by "computed by the same scan" is what keeps this to a single
    /// mutex rather than one per quantity.
    struct Derived {
        KnotClasses<T> classes;                 ///< The distinct knots and their counts.
        std::vector<std::int64_t> first_basis;  ///< Empty for a periodic space.
    };

    /// Refuse the constructor's arguments before anything is stored.
    ///
    /// The order is the oracle's `_validate_input`: degree, then length, then
    /// monotonicity, then whether the vector can support the degree at all. The
    /// type-kind checks that sit between the first two there belong to the Python
    /// wrapper, because nanobind has no path to `TypeError`.
    ///
    /// \param knots The knot vector as supplied.
    /// \param degree The requested degree.
    /// \param periodic Whether the space is periodic, which changes the basis count.
    /// \throws std::invalid_argument On any of the four refusals.
    static void check_arguments(std::span<const T> knots, std::int64_t degree, bool periodic) {
        if (degree < 0) {
            throw std::invalid_argument("degree must be non-negative");
        }
        // Divided rather than multiplied: `2 * degree + 2` is signed overflow, and
        // so undefined, for a degree near the top of the range. Python's integers
        // are arbitrary-precision, so the oracle's spelling is exact for any degree
        // and this one has to be too.
        //
        // The `size < 2` guard is what makes the division equivalent to the
        // oracle's multiplication rather than nearly so. C++ integer division
        // truncates toward zero where the rewrite needs a floor, and the two differ
        // on exactly one input: at one knot and degree 0, `(1 - 2) / 2` truncates to
        // 0 instead of -1, so the refusal is skipped and the caller is told
        // something else further down. Any vector shorter than two knots is refused
        // for any non-negative degree, since `2 * degree + 2` is at least 2, so the
        // guard costs no case and closes that one.
        const auto count = static_cast<std::int64_t>(knots.size());
        if (count < 2 || (count - 2) / 2 < degree) {
            throw std::invalid_argument("knots must have at least 2*degree+2 elements");
        }
        // `np.all(np.diff(knots) >= 0)`: the difference is formed first and only
        // then compared, which is not the same predicate as `knots[i] >= knots[i-1]`
        // -- a vector of two infinities differences to NaN and is refused, while the
        // direct comparison would accept it.
        for (std::size_t i = 1; i < knots.size(); ++i) {
            const T step = knots[i] - knots[i - 1];
            if (!(detail::as_double(step) >= 0.0)) {
                throw std::invalid_argument("knots must be non-decreasing");
            }
        }
        // Derived only here, after the shape and ordering checks: the tolerance
        // reads the endpoints, so it needs a vector that has them.
        const double tol = knot_tolerance(knots);
        if (pantr::bspline::num_basis<T>(knots, degree, periodic, tol) < degree + 1) {
            throw std::invalid_argument("Not enough knots for the specified degree");
        }
    }

    /// Build the derived block from the frozen state.
    ///
    /// A static function over the state rather than a member, so that it can be
    /// tested without an object and cannot read a half-initialised `this`.
    ///
    /// \param knots The stored knot vector.
    /// \param degree The degree.
    /// \param periodic Whether the space is periodic.
    /// \param tol The parametric tolerance.
    /// \param num_basis The stored basis count.
    /// \return The classes, and the per-interval first basis index for a
    ///         non-periodic space.
    static Derived build_derived(std::span<const T> knots, std::int64_t degree, bool periodic,
                                 double tol, std::int64_t num_basis) {
        Derived block{unique_knots_and_multiplicity(knots, degree, tol), {}};
        if (periodic) {
            return block;
        }
        std::int64_t last_position = -1;
        for (const std::int64_t multiplicity : block.classes.multiplicity) {
            last_position += multiplicity;
            if (last_position >= degree && last_position < num_basis) {
                block.first_basis.push_back(last_position - degree);
            }
        }
        return block;
    }

    /// The derived block, building it on first use.
    ///
    /// \return The block, valid while this space lives.
    [[nodiscard]] const Derived& derived() const {
        return derived_.get([this] {
            return build_derived(std::span<const T>(knots_), degree_, periodic_, tol_, num_basis_);
        });
    }

    std::vector<T> knots_;              ///< The knot vector, snapped if requested.
    std::int64_t degree_ = 0;           ///< The polynomial degree.
    bool periodic_ = false;             ///< Whether the space is periodic.
    double tol_ = 0.0;                  ///< The parametric tolerance, from the raw knots.
    std::int64_t num_basis_ = 0;        ///< The number of basis functions.
    std::int64_t num_intervals_ = 0;    ///< The number of in-domain intervals.
    LazySlot<Derived> derived_;         ///< The derived arrays, built at most once.
};

}  // namespace pantr::bspline
