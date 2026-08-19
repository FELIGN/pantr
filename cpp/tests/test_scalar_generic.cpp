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

/// The kernel's values must not depend on the scalar type carrying a derivative:
/// a dual number's value component obeys exactly the `double` recurrence, so the
/// two agree to the last bit. This is a stronger statement than a tolerance and
/// it is the correct one -- the arithmetic performed on the value component is
/// operation for operation the same.
void dual_value_component_matches_double_exactly() {
    using pantr::value_of;
    const auto pts = sample_points();
    const auto seeded = seeded_points(pts);

    for (int degree = 0; degree <= 8; ++degree) {
        const auto stride = static_cast<std::size_t>(degree + 1);
        const auto dual = tabulate_dual(degree, seeded);

        std::vector<double> plain(pts.size() * stride, 0.0);
        const pantr::span2d<double> view(plain.data(), pts.size(), stride);
        pantr::tabulate_cardinal_bspline_1d<double>(degree, std::span<const double>(pts), view);

        for (std::size_t i = 0; i < plain.size(); ++i) {
            PANTR_CHECK_MSG(value_of(dual[i]) == plain[i],
                            "degree " + std::to_string(degree) + " index " +
                                std::to_string(i) + ": dual " +
                                std::to_string(value_of(dual[i])) + " vs double " +
                                std::to_string(plain[i]));
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

int main() {
    dual_value_component_matches_double_exactly();
    derivatives_sum_to_zero();
    low_degree_derivatives_match_closed_forms();
    return pantr::test::summary("test_scalar_generic");
}
