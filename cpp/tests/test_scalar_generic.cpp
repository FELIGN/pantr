/// \file
/// The Tier A discipline, checked by compiling the kernel on a type that breaks
/// if the discipline is broken.
///
/// design/automatic_differentiation.md concludes that pantr does not need
/// automatic differentiation, and that templating the scalar is enough to keep
/// the option open at four disciplines' cost. The risk with a conclusion of that
/// shape is that the four disciplines quietly rot: nothing in a `double`-only
/// build notices when a comparison starts bypassing `value_of`, when a `floor`
/// or an `int` cast lands on the scalar, or when a math call is written
/// `std::sqrt(x)` and so blocks argument-dependent lookup.
///
/// So this file defines a minimal forward-mode dual number and instantiates the
/// kernel on it. Every one of those four violations is a compile error against
/// this type, which turns the disciplines from a convention into a build
/// failure. The type is a test fixture and nothing more -- design/automatic
/// _differentiation.md's `Dual<T, N>` is a later decision, not this.
///
/// It also checks the derivative it produces, because a dual number that
/// compiles but propagates the wrong derivative would pass a compile-only test
/// while proving nothing.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace {

/// Forward-mode dual number carrying one derivative component.
///
/// Deliberately minimal, and deliberately missing things:
///
///   - **no `operator==` and no ordering.** This is the point of the fixture.
///     `pantr::Real` does not require `std::regular` precisely so that a scalar
///     may omit them, which makes `if (a < b)` on the scalar a compile error and
///     forces the Tier B spelling `value_of(a) < value_of(b)`.
///   - **no conversion to `double`.** So an integer cast or a `std::floor` on
///     the scalar cannot compile either.
///
/// `value_of` is a hidden friend, so it is found by argument-dependent lookup
/// through the two-step pattern the library uses and *only* through it.
class Dual1 {
  public:
    constexpr Dual1() noexcept = default;
    constexpr explicit Dual1(double value) noexcept : value_(value) {}
    constexpr Dual1(double value, double derivative) noexcept
        : value_(value), derivative_(derivative) {}

    [[nodiscard]] constexpr double derivative() const noexcept { return derivative_; }

    constexpr Dual1 operator-() const noexcept { return {-value_, -derivative_}; }

    friend constexpr Dual1 operator+(Dual1 a, Dual1 b) noexcept {
        return {a.value_ + b.value_, a.derivative_ + b.derivative_};
    }
    friend constexpr Dual1 operator-(Dual1 a, Dual1 b) noexcept {
        return {a.value_ - b.value_, a.derivative_ - b.derivative_};
    }
    friend constexpr Dual1 operator*(Dual1 a, Dual1 b) noexcept {
        return {a.value_ * b.value_, a.derivative_ * b.value_ + a.value_ * b.derivative_};
    }
    friend constexpr Dual1 operator/(Dual1 a, Dual1 b) noexcept {
        const double q = a.value_ / b.value_;
        return {q, (a.derivative_ - q * b.derivative_) / b.value_};
    }

    /// The Tier B access point. A hidden friend, so `pantr::value_of(x)` does
    /// *not* find it and only the `using pantr::value_of; value_of(x)` spelling
    /// works -- which is the behaviour the library's convention relies on.
    friend constexpr double value_of(Dual1 x) noexcept { return x.value_; }

  private:
    double value_ = 0.0;
    double derivative_ = 0.0;
};

// The design claims, as static assertions rather than as prose.
static_assert(pantr::Real<Dual1>, "the fixture must satisfy the scalar concept");
static_assert(!std::equality_comparable<Dual1>,
              "Real must not require operator==; requiring it would make the Tier B "
              "discipline unenforceable at the comparison sites it exists to protect");
static_assert(!std::convertible_to<Dual1, double>,
              "a scalar convertible to double would let an integer cast or a floor "
              "compile silently");
static_assert(std::same_as<pantr::accumulator_t<Dual1>, Dual1>,
              "a differentiable scalar must not be narrowed to double for accumulation");
static_assert(std::same_as<pantr::accumulator_t<float>, double>);
static_assert(std::same_as<pantr::accumulator_t<double>, double>);

static_assert(std::constructible_from<Dual1, double>,
              "the kernels build constants -- T(1) seeds the recurrence -- so a scalar "
              "that cannot be constructed from a double is not usable by them");
static_assert(!std::convertible_to<double, Dual1>,
              "constructible_from must be satisfied by an EXPLICIT constructor: an "
              "implicit conversion the other way would let a stray double enter the "
              "computation silently");

/// A scalar the concept must REJECT, and the reason it exists.
///
/// A forward-mode dual built by named factories rather than by a numeric
/// constructor is a common AD shape: it keeps `Dual d = 3;` from compiling, which
/// is usually what its author wants. It has the five operators and a `value_of`,
/// so before `std::constructible_from<T, double>` joined `Real` it satisfied the
/// concept -- and then failed to compile inside the kernel at `T(1)`, producing a
/// template error from a header its author did not write.
///
/// Pinned here so `Real` is checked to be the kernel's actual contract rather
/// than a subset of it: it must reject this type *at the constraint*.
class DualByFactory {
  public:
    static constexpr DualByFactory constant(double v) noexcept { return {v, 0.0}; }
    static constexpr DualByFactory seed(double v) noexcept { return {v, 1.0}; }

    constexpr DualByFactory() noexcept = default;

    constexpr DualByFactory operator-() const noexcept { return {-value_, -derivative_}; }
    friend constexpr DualByFactory operator+(DualByFactory a, DualByFactory b) noexcept {
        return {a.value_ + b.value_, a.derivative_ + b.derivative_};
    }
    friend constexpr DualByFactory operator-(DualByFactory a, DualByFactory b) noexcept {
        return {a.value_ - b.value_, a.derivative_ - b.derivative_};
    }
    friend constexpr DualByFactory operator*(DualByFactory a, DualByFactory b) noexcept {
        return {a.value_ * b.value_, a.derivative_ * b.value_ + a.value_ * b.derivative_};
    }
    friend constexpr DualByFactory operator/(DualByFactory a, DualByFactory b) noexcept {
        const double q = a.value_ / b.value_;
        return {q, (a.derivative_ - q * b.derivative_) / b.value_};
    }
    friend constexpr double value_of(DualByFactory x) noexcept { return x.value_; }

  private:
    constexpr DualByFactory(double value, double derivative) noexcept
        : value_(value), derivative_(derivative) {}

    double value_ = 0.0;
    double derivative_ = 0.0;
};

/// Exercise every operator of the fixture, so that its rejection is known to be
/// about the missing constructor and nothing else.
///
/// This is not decoration. The `static_assert`s below short-circuit on
/// `constructible_from`, so without a real use the operators are never
/// instantiated -- Clang says so and refuses the build under `-Werror`
/// (`-Wunused-function`), which is the correct complaint: a fixture whose
/// arithmetic is never compiled cannot be claimed to differ from `Real` by one
/// requirement. Evaluating it here compiles all five, and the expected value is
/// worked out by hand rather than read off a run.
///
/// With `a = (2, 1)` and `b = (3, 0)`: `a + b = (5, 1)`, `a * b = (6, 3)`,
/// `(a * b) / b = (2, 1)`, so `a + b - (a * b) / b = (3, 0)` and negating gives
/// `(-3, 0)`.
constexpr DualByFactory fixture_arithmetic() noexcept {
    const DualByFactory a = DualByFactory::seed(2.0);
    const DualByFactory b = DualByFactory::constant(3.0);
    return -(a + b - a * b / b);
}

static_assert(value_of(fixture_arithmetic()) == -3.0,
              "the fixture's operators must be real arithmetic, not placeholders");

static_assert(!pantr::Real<DualByFactory>,
              "Real must reject a scalar the kernel cannot build a constant of, at the "
              "constraint rather than thirty lines into a template instantiation");
static_assert(!std::constructible_from<DualByFactory, double>,
              "the fixture is only meaningful while this is the reason it is rejected");

// `value_type_t` is an alias template, so nothing checks it until something
// instantiates it, and it shipped with no instantiation anywhere in the tree.
//
// It named `value_of` through a QUALIFIED call. That is the failure the header's
// own comment spends ten lines warning about, and it failed in the way that
// warning describes rather than uniformly: measured with this project's GCC
// 14.4.0, the qualified spelling still compiled for `double` and for `float` --
// the `using pantr::value_of;` in `detail` makes the built-in overload visible to
// qualified lookup too -- and failed only for `Dual1`, whose `value_of` is a
// hidden friend that only ADL can reach.
//
// So a `double`-only build could never have noticed, which is precisely why all
// three are pinned here and not just the interesting one.
static_assert(std::same_as<pantr::value_type_t<double>, double>);
static_assert(std::same_as<pantr::value_type_t<float>, float>);
static_assert(std::same_as<pantr::value_type_t<Dual1>, double>,
              "a differentiable scalar's underlying value type must be reachable through "
              "the two-step lookup, which is the only way a hidden-friend value_of is found");

constexpr double kEps = std::numeric_limits<double>::epsilon();

std::vector<Dual1> seeded_points(const std::vector<double>& pts) {
    std::vector<Dual1> seeded;
    seeded.reserve(pts.size());
    for (const double t : pts) {
        seeded.emplace_back(t, 1.0);  // d/du seeded to 1
    }
    return seeded;
}

std::vector<double> sample_points() {
    std::vector<double> pts;
    for (int i = 0; i <= 32; ++i) {
        pts.push_back(static_cast<double>(i) / 32.0);
    }
    return pts;
}

std::vector<Dual1> tabulate_dual(int degree, const std::vector<Dual1>& pts) {
    const auto stride = static_cast<std::size_t>(degree + 1);
    std::vector<Dual1> out(pts.size() * stride);
    const pantr::span2d<Dual1> view(out.data(), pts.size(), stride);
    pantr::tabulate_cardinal_bspline_1d<Dual1>(degree, std::span<const Dual1>(pts), view);
    return out;
}

constexpr double kOneSidedToTwoSided = 2.0;
/// Neither side is the exact answer, so a parity bound is twice a one-sided forward-error
/// bound. `ONE_SIDED_TO_TWO_SIDED` in `tests/_parity_harness.py` is the same factor.

constexpr int kAccumulatorRoundingsPerStage = 2;
/// Roundings each stage commits along the dominant path, in the accumulator format: one
/// multiply and one add. Taken from the `Roundings` this site's Python parity claim
/// already carries (`tests/parity/test_basis_cardinal_bspline.py`), unchanged.
///
/// It is **conservative here by one rounding per stage**, deliberately rather than by
/// oversight. That claim compares two backends that each commit both roundings; this one
/// compares a `double` path that fuses them into one against a `Dual1` path that cannot,
/// so the true count is one on one side and two on the other. Charging two to each bounds
/// three by four, and it keeps the two sites spelling the same budget the same way, which
/// is worth more than the factor.

/// Get the relative half of the bound separating the two scalar types at one degree.
///
/// Higham's `gamma_m = m u / (1 - m u)` over `m = degree * kAccumulatorRoundingsPerStage`,
/// the closed form `_relative_growth` uses in `tests/_parity_harness.py` and for the reason
/// recorded there: the algebraically equivalent power form evaluates to exactly zero in
/// float64 for a budget of one rounding per stage, which would turn this bound into an
/// assertion of bit-identity while claiming to be a bound.
///
/// The storage format is the accumulator format here, so a store rounds nothing and
/// contributes no term.
///
/// \param degree Polynomial degree, one recurrence stage each. Zero gives zero, which is
///        correct: at degree 0 the kernel writes the value with no arithmetic at all.
/// \return The relative growth, to be multiplied by the amplification.
double dual_double_relative_bound(int degree) {
    const double u = 0.5 * kEps;
    const double total = static_cast<double>(degree * kAccumulatorRoundingsPerStage) * u;
    return total / (1.0 - total);
}

/// Get the absolute half of the same bound, the half a purely relative one omits.
///
/// Each rounding contributes at most one smallest-positive-subnormal absolutely, on top of
/// its relative contribution, and inside the span the stage maps are convex combinations of
/// non-negative values, so an absolute perturbation passes through unamplified and the
/// floors simply add. `_underflow_budget` in `tests/_parity_harness.py` derives this and
/// records why omitting it is not a small error: a bound written as `u |x|` alone goes to
/// zero with `x` while the true error does not.
///
/// \param degree Polynomial degree, one recurrence stage each.
/// \return The absolute floor accumulated over the stages.
double dual_double_underflow_budget(int degree) {
    const double eta = std::numeric_limits<double>::denorm_min();
    return static_cast<double>(degree * kAccumulatorRoundingsPerStage) * eta;
}

/// The bound above must reject a perturbation at its own size.
///
/// A bound is only a check while something can fail it, and the one next door is
/// derived rather than measured, so nothing in the comparison itself would notice if a
/// future edit widened it past usefulness. Its Python counterpart carries the same
/// guard for the same reason
/// (`test_bounded_branch_admits_a_perturbation_at_the_bound`).
///
/// Two failure modes, at the two ends. A bound that collapses to zero asserts
/// bit-identity while claiming to be a bound -- the trap `_relative_growth` records
/// hitting in the Python harness, where the algebraically equivalent power form
/// evaluated to exactly zero. A bound that grows past round-off scale stops refusing
/// anything a defect would produce. Both are checked against the *size* of the bound;
/// checking that a perturbation twice its size is refused would be arithmetic rather
/// than a property of the bound, and would pass however wide it grew.
void the_parity_bound_stays_a_check_at_both_ends() {
    for (int degree = 1; degree <= 8; ++degree) {
        const double value = 0.7;
        const double relative = dual_double_relative_bound(degree) * std::abs(value);
        const double bound =
            kOneSidedToTwoSided * (relative + dual_double_underflow_budget(degree));

        PANTR_CHECK_MSG(bound > 0.0, "degree " + std::to_string(degree)
                                         + ": the bound collapsed to zero, so the "
                                           "comparison asserts bit-identity in disguise");

        // The ceiling is a fixed number of epsilons, and it must not be derived from
        // `kAccumulatorRoundingsPerStage`: a ceiling computed from the budget widens
        // with it and cannot catch it widening, which is the whole point of being here.
        // (Measured: an earlier version of this check did exactly that and passed with
        // the budget quadrupled.) Sixty-four epsilons is the project's `get_default`
        // tier -- a short algorithm plus build slack -- and the bound at the largest
        // degree tested sits about four times below it.
        const double ceiling = 64.0 * kEps * std::abs(value);
        PANTR_CHECK_MSG(bound <= ceiling,
                        "degree " + std::to_string(degree) + ": the bound is "
                            + std::to_string(bound) + ", past the round-off ceiling "
                            + std::to_string(ceiling)
                            + "; it no longer refuses what it exists to refuse");
    }
}

/// The kernel's values must not depend on the scalar type carrying a derivative:
/// a dual number's value component obeys the same `double` recurrence.
///
/// This used to demand bit-identity, on the argument that "the arithmetic performed
/// on the value component is operation for operation the same". That argument is
/// true of the IEEE operations *named in the source* and false of the number of
/// roundings *executed*, which is a property of the build -- and the unstated
/// hypothesis it needs is "and the target has no FMA instruction".
///
/// At `cardinal_bspline.hpp`'s `saved + nr_old * term` the `double` path is one
/// expression and contracts to a single `vfmadd` on any FMA target, one rounding.
/// The `Dual1` path cannot: the multiply and the add are two expressions inside two
/// operator functions, and contraction is a within-one-expression permission, so no
/// conforming compiler may fuse across them. Two roundings. Measured on GCC 14.4.0
/// and Clang 18.1.8 alike at `-mavx2 -mfma`, so this is not a compiler quirk; the
/// same recurrence in plain `double` with the multiply and the add forced apart
/// reproduces the `Dual1` value component bit for bit at every index and degree.
///
/// So the bound below, and it is the project's existing one rather than a new
/// constant: `tests/parity/test_basis_cardinal_bspline.py` carries a parity claim for
/// this same site, and `dual_double_relative_bound` and
/// `dual_double_underflow_budget` transcribe its budget, its closed form and its
/// underflow floor unchanged, doubled once for a two-sided comparison exactly as
/// `absolute_tolerance` does. Where the two situations differ the transcription stays
/// conservative rather than being retuned; the constants say where.
///
/// The amplification factor is the companion recurrence run on the absolute
/// coefficients, which inside `[0, 1]` -- where every sample point here lies -- equals
/// the value itself, because the stage weights are then a convex combination and
/// nothing cancels. Hence a relative bound, plus the absolute floor.
///
/// At degree 0 there are no stages, both halves are zero, and it collapses to exact
/// equality -- which is right, because the kernel writes the value there with no
/// arithmetic at all. Measured worst case on this data: 3 ulps, against a bound of
/// about 16.
void dual_value_component_matches_double_within_the_parity_bound() {
    using pantr::value_of;
    const auto pts = sample_points();
    const auto seeded = seeded_points(pts);

    for (int degree = 0; degree <= 8; ++degree) {
        const auto stride = static_cast<std::size_t>(degree + 1);
        const auto dual = tabulate_dual(degree, seeded);

        std::vector<double> plain(pts.size() * stride, 0.0);
        const pantr::span2d<double> view(plain.data(), pts.size(), stride);
        pantr::tabulate_cardinal_bspline_1d<double>(degree, std::span<const double>(pts), view);

        const double bound_relative = dual_double_relative_bound(degree);
        const double bound_floor = dual_double_underflow_budget(degree);

        for (std::size_t i = 0; i < plain.size(); ++i) {
            const double bound =
                kOneSidedToTwoSided * (bound_relative * std::abs(plain[i]) + bound_floor);
            PANTR_CHECK_MSG(std::abs(value_of(dual[i]) - plain[i]) <= bound,
                            "degree " + std::to_string(degree) + " index " +
                                std::to_string(i) + ": dual " +
                                std::to_string(value_of(dual[i])) + " vs double " +
                                std::to_string(plain[i]) + ", bound " +
                                std::to_string(bound));
        }
    }
}

/// Differentiating partition of unity: `sum_r N_r(u) = 1` for all `u`, so the
/// derivatives must sum to zero. An independent constraint on the propagated
/// derivative -- it does not recompute it.
void derivatives_sum_to_zero() {
    const auto pts = sample_points();
    const auto seeded = seeded_points(pts);

    for (int degree = 0; degree <= 12; ++degree) {
        const auto stride = static_cast<std::size_t>(degree + 1);
        const auto dual = tabulate_dual(degree, seeded);

        // The derivative components are NOT sign-definite, unlike the values, so
        // the convex-combination argument of test_cardinal_bspline.cpp does not
        // apply and cancellation is possible. The scale is set by the largest
        // derivative in the row, which for the cardinal basis is O(degree), so
        // the bound is the summation bound `stride * eps` times that scale.
        double scale = 0.0;
        for (std::size_t j = 0; j < pts.size(); ++j) {
            for (std::size_t r = 0; r < stride; ++r) {
                scale = std::max(scale, std::abs(dual[j * stride + r].derivative()));
            }
        }
        const double bound = 8.0 * static_cast<double>(stride) * kEps * std::max(scale, 1.0);

        for (std::size_t j = 0; j < pts.size(); ++j) {
            double sum = 0.0;
            for (std::size_t r = 0; r < stride; ++r) {
                sum += dual[j * stride + r].derivative();
            }
            PANTR_CHECK_MSG(std::abs(sum) <= bound,
                            "degree " + std::to_string(degree) + " at u=" +
                                std::to_string(pts[j]) + ": sum of derivatives " +
                                std::to_string(sum) + " > " + std::to_string(bound));
        }
    }
}

/// Closed-form derivatives at low degree, derived on paper from the closed forms
/// in test_cardinal_bspline.cpp. The oracle for the derivative, as opposed to the
/// constraint above.
void low_degree_derivatives_match_closed_forms() {
    const auto pts = sample_points();
    const auto seeded = seeded_points(pts);
    const double bound = 16.0 * kEps;

    {  // degree 1: d/du (1 - u, u) = (-1, 1)
        const auto v = tabulate_dual(1, seeded);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            PANTR_CHECK(std::abs(v[2 * j + 0].derivative() + 1.0) <= bound);
            PANTR_CHECK(std::abs(v[2 * j + 1].derivative() - 1.0) <= bound);
        }
    }

    {  // degree 2: d/du ((1-u)^2/2, (1 + 2u - 2u^2)/2, u^2/2) = (u - 1, 1 - 2u, u)
        const auto v = tabulate_dual(2, seeded);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            const double t = pts[j];
            PANTR_CHECK(std::abs(v[3 * j + 0].derivative() - (t - 1.0)) <= bound);
            PANTR_CHECK(std::abs(v[3 * j + 1].derivative() - (1.0 - 2.0 * t)) <= bound);
            PANTR_CHECK(std::abs(v[3 * j + 2].derivative() - t) <= bound);
        }
    }
}

}  // namespace

/// The three sign predicates are exactly the product forms they replace.
///
/// They exist to remove a product from a sign test, because the product is the
/// defect: two operands of magnitude `1e-23` are perfectly representable at
/// `float32` while their product falls under that format's minimum subnormal and
/// flushes to zero, so the test answers as though one operand had vanished
/// (FELIGN/pantr#351). Substituting them is only sound if they agree with the
/// product everywhere the product is right.
///
/// Every pair drawn from the value list has an exact product -- `inf * 0 = NaN`
/// included -- so agreement here must be total. Both floating-point widths are
/// checked, since the substituted sites span `T` and `accumulator_t<T>`.
template <std::floating_point V>
void sign_predicates_match_the_product_forms() {
    const V values[] = {-std::numeric_limits<V>::infinity(),
                        V{-3},
                        V{-1},
                        V{-0.0},
                        V{0},
                        V{1},
                        V{3},
                        std::numeric_limits<V>::infinity(),
                        std::numeric_limits<V>::quiet_NaN()};

    for (const V a : values) {
        for (const V b : values) {
            const V product = a * b;
            PANTR_CHECK(pantr::have_same_sign(a, b) == (product > V{0}));
            PANTR_CHECK(pantr::have_opposite_signs(a, b) == (product < V{0}));
            PANTR_CHECK(pantr::spans_zero(a, b) == (product <= V{0}));
        }
    }
}

/// The predicates are right in the regime where the product flushes to zero.
///
/// This is the defect itself, reached at `float` where it bites the shipped code
/// and again at `double` so that no dtype conversion is involved. Both operands
/// stay perfectly representable; only their product cannot be formed.
template <std::floating_point V>
void sign_predicates_survive_underflow(V tiny) {
    PANTR_CHECK(tiny > V{0});
    PANTR_CHECK(tiny * tiny == V{0});

    PANTR_CHECK(pantr::have_same_sign(tiny, tiny));
    PANTR_CHECK(!(tiny * tiny > V{0}));

    PANTR_CHECK(pantr::have_opposite_signs(tiny, -tiny));
    PANTR_CHECK(!(tiny * -tiny < V{0}));

    PANTR_CHECK(!pantr::spans_zero(tiny, tiny));
    PANTR_CHECK(tiny * tiny <= V{0});
}

/// A zero paired with an infinity is not a spanned bracket.
///
/// The single pair that separates the exact predicate from the naive one: `0 *
/// inf` is NaN, which is not less than or equal to zero, so the product form
/// reports no span and `spans_zero` must agree. A regression here is invisible
/// in the matrix above, which is why it is called out.
void zero_with_infinity_is_not_spanned() {
    for (const double zero : {0.0, -0.0}) {
        for (const double infinity : {HUGE_VAL, -HUGE_VAL}) {
            PANTR_CHECK(!pantr::spans_zero(zero, infinity));
            PANTR_CHECK(!pantr::spans_zero(infinity, zero));
        }
    }
}

/// The predicates are reachable from a scalar that has no ordering of its own.
///
/// This is the Tier B claim, and `Dual1` is what makes it a build failure rather
/// than a convention: the fixture deliberately supplies no `operator==` and no
/// ordering, so `a * b < T{0}` -- the form these predicates replaced -- cannot
/// compile for it at all. Going through `value_of` is what makes them work here,
/// and this function failing to compile is the regression signal.
void sign_predicates_reach_a_scalar_without_ordering() {
    static_assert(pantr::Real<Dual1>, "the fixture must still satisfy Real");

    const Dual1 positive{2.0, 1.0};
    const Dual1 negative{-2.0, 1.0};
    const Dual1 zero{0.0, 1.0};

    PANTR_CHECK(pantr::have_same_sign(positive, positive));
    PANTR_CHECK(!pantr::have_same_sign(positive, negative));
    PANTR_CHECK(pantr::have_opposite_signs(positive, negative));
    PANTR_CHECK(!pantr::have_opposite_signs(positive, positive));
    PANTR_CHECK(pantr::spans_zero(positive, negative));
    PANTR_CHECK(pantr::spans_zero(positive, zero));
    PANTR_CHECK(!pantr::spans_zero(positive, positive));
}

int main() {
    dual_value_component_matches_double_within_the_parity_bound();
    the_parity_bound_stays_a_check_at_both_ends();
    derivatives_sum_to_zero();
    low_degree_derivatives_match_closed_forms();
    sign_predicates_match_the_product_forms<float>();
    sign_predicates_match_the_product_forms<double>();
    sign_predicates_survive_underflow(1e-25F);
    sign_predicates_survive_underflow(1e-200);
    zero_with_infinity_is_not_spanned();
    sign_predicates_reach_a_scalar_without_ordering();
    return pantr::test::summary("test_scalar_generic");
}
