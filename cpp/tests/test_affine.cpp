/// \file
/// Properties of the affine map.
///
/// ## Where the tolerances come from
///
/// Nothing here is compared with a bare constant. Three bounds appear, each
/// derived from what the operation does:
///
///  - **Exact.** A permutation, a sign flip, or a product of exactly
///    representable entries introduces no rounding, so identity, `shear`,
///    integer `scaling` and the structure of the factories are compared with
///    `==`. Using a tolerance there would hide a defect rather than allow for
///    one.
///  - **A few eps for trigonometry.** `cos` and `sin` are correctly rounded to
///    within one ulp by any libm worth the name, and the Rodrigues combination
///    applies at most four roundings on top. Eight eps of the largest entry is
///    that with a factor of two of slack, stated as slack rather than derived.
///  - **`kappa * eps` for anything through the inverse.** Both the LU and the
///    subsequent product are backward stable, so a round trip through
///    `inverse()` returns the input perturbed by roughly the condition number
///    times the working precision. The matrices below are well conditioned by
///    construction and their condition numbers are computed in the test rather
///    than assumed.
///
/// ## What each group would catch
///
/// The factories are checked against *analytic* answers -- rotating the x axis a
/// quarter turn must give the y axis -- rather than against a second copy of the
/// same formula, which would pass however wrong the formula was. `inverse` and
/// `compose` are checked by the identities that define them, not by recomputing
/// them.

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "pantr/transform/affine.hpp"

namespace {

using pantr::transform::AffineTransform;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kTrigSlack = 8.0;

/// Build a map from a row-major matrix and an offset.
///
/// \param n The dimension.
/// \param matrix The linear part, row-major.
/// \param offset The translation.
/// \return The map.
AffineTransform<double> make(std::size_t n, std::vector<double> matrix,
                             std::vector<double> offset) {
    return AffineTransform<double>(pantr::span2d<const double>(matrix.data(), n, n),
                                   std::span<const double>(offset));
}

/// Whether calling `fn` throws `std::invalid_argument`.
///
/// \param fn The call to attempt.
/// \return `true` when it threw.
template <class F>
bool throws_invalid(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

/// Apply a map to one point.
///
/// \param t The map.
/// \param x The point.
/// \return The image.
std::vector<double> apply_one(const AffineTransform<double>& t, std::vector<double> x) {
    std::vector<double> out(t.dim());
    t.apply(pantr::span2d<const double>(x.data(), 1, t.dim()),
            pantr::span2d<double>(out.data(), 1, t.dim()));
    return out;
}

void test_construction_validates() {
    const std::vector<double> two{1.0, 0.0, 0.0, 1.0};
    PANTR_CHECK(throws_invalid([&two] {
        (void)AffineTransform<double>(pantr::span2d<const double>(two.data(), 1, 4),
                                      std::span<const double>(two));
    }));
    PANTR_CHECK_MSG(throws_invalid([&two] {
        const std::vector<double> wrong(3, 0.0);
        (void)AffineTransform<double>(pantr::span2d<const double>(two.data(), 2, 2),
                                      std::span<const double>(wrong));
    }), "translation length must match");
    PANTR_CHECK(throws_invalid([] { (void)AffineTransform<double>::identity(0); }));
}

void test_identity_is_exact() {
    const auto id = AffineTransform<double>::identity(3);
    // The identity applied to anything must return it BITWISE, not nearly: every
    // product is by an exact 0 or 1 and every sum adds an exact zero.
    const std::vector<double> x{0.1, -1e300, 3.0};
    const auto got = apply_one(id, x);
    for (std::size_t i = 0; i < 3; ++i) {
        PANTR_CHECK(got[i] == x[i]);
    }
    PANTR_CHECK(id.compose(id) == id);
    PANTR_CHECK(id.inverse() == id);
}

void test_rotation_2d_against_the_analytic_answer() {
    const double quarter = std::numbers::pi / 2.0;
    const auto r = AffineTransform<double>::rotation_2d(quarter);
    // The x axis must land on the y axis. This is the analytic answer, not a
    // second evaluation of the same formula.
    const auto image = apply_one(r, {1.0, 0.0});
    PANTR_CHECK(std::abs(image[0] - 0.0) <= kTrigSlack * kEps);
    PANTR_CHECK(std::abs(image[1] - 1.0) <= kTrigSlack * kEps);

    // Four quarter turns are the identity, up to the accumulated trigonometry.
    const auto full = r.compose(r).compose(r).compose(r);
    const auto back = apply_one(full, {1.0, 0.0});
    PANTR_CHECK(std::abs(back[0] - 1.0) <= 4.0 * kTrigSlack * kEps);
    PANTR_CHECK(std::abs(back[1] - 0.0) <= 4.0 * kTrigSlack * kEps);

    PANTR_CHECK(throws_invalid([] {
        (void)AffineTransform<double>::rotation_2d(std::numeric_limits<double>::infinity());
    }));
}

void test_rotation_3d_about_a_coordinate_axis() {
    const double quarter = std::numbers::pi / 2.0;
    const std::vector<double> z{0.0, 0.0, 1.0};
    const auto r = AffineTransform<double>::rotation_3d(quarter, std::span<const double>(z));
    const auto image = apply_one(r, {1.0, 0.0, 0.0});
    PANTR_CHECK(std::abs(image[0]) <= kTrigSlack * kEps);
    PANTR_CHECK(std::abs(image[1] - 1.0) <= kTrigSlack * kEps);
    PANTR_CHECK(std::abs(image[2]) <= kTrigSlack * kEps);

    // A point ON the axis is fixed, exactly the property Rodrigues guarantees.
    const auto fixed = apply_one(r, {0.0, 0.0, 5.0});
    PANTR_CHECK(std::abs(fixed[2] - 5.0) <= kTrigSlack * kEps * 5.0);
    PANTR_CHECK(std::abs(fixed[0]) <= kTrigSlack * kEps * 5.0);

    // The whole matrix against the analytic quarter turn about z, entry by entry.
    // Checking a couple of images instead leaves entries untouched: rotating
    // (1,0,0) and (0,0,5) never reads column 1, so a flipped sign there survived
    // an earlier version of this test. Verified by mutation that it no longer does.
    const double expected[9] = {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            PANTR_CHECK_MSG(
                std::abs(pantr::at(r.matrix(), i, j) - expected[i * 3 + j]) <= kTrigSlack * kEps,
                "every entry of the rotation matrix");
        }
    }

    // The axis is normalized internally, so a non-unit axis is the same rotation.
    const std::vector<double> long_z{0.0, 0.0, 7.0};
    const auto r2 = AffineTransform<double>::rotation_3d(quarter,
                                                         std::span<const double>(long_z));
    PANTR_CHECK(r2 == r);

    const std::vector<double> zero{0.0, 0.0, 0.0};
    PANTR_CHECK(throws_invalid([&zero] {
        (void)AffineTransform<double>::rotation_3d(1.0, std::span<const double>(zero));
    }));
    const std::vector<double> wrong_len{1.0, 0.0};
    PANTR_CHECK(throws_invalid([&wrong_len] {
        (void)AffineTransform<double>::rotation_3d(1.0, std::span<const double>(wrong_len));
    }));
}

void test_mirror_is_an_involution_and_fixes_its_plane() {
    const std::vector<double> normal{0.0, 3.0, 0.0};  // deliberately not unit
    const auto m = AffineTransform<double>::mirror(std::span<const double>(normal));
    // A unit normal makes every entry exactly representable, so the involution is
    // EXACT here rather than approximate.
    PANTR_CHECK(m.compose(m) == AffineTransform<double>::identity(3));
    const auto in_plane = apply_one(m, {1.0, 0.0, -2.0});
    PANTR_CHECK(in_plane[0] == 1.0 && in_plane[1] == 0.0 && in_plane[2] == -2.0);
    const auto flipped = apply_one(m, {0.0, 4.0, 0.0});
    PANTR_CHECK(flipped[1] == -4.0);

    const std::vector<double> zero{0.0, 0.0};
    PANTR_CHECK(throws_invalid([&zero] {
        (void)AffineTransform<double>::mirror(std::span<const double>(zero));
    }));
}

void test_scaling_and_shear_are_exact() {
    const std::vector<double> factors{2.0, 0.5, -4.0};
    const auto s = AffineTransform<double>::scaling(std::span<const double>(factors));
    const auto got = apply_one(s, {1.0, 8.0, 0.25});
    PANTR_CHECK(got[0] == 2.0 && got[1] == 4.0 && got[2] == -1.0);

    const std::vector<double> singular{1.0, 0.0};
    PANTR_CHECK(throws_invalid([&singular] {
        (void)AffineTransform<double>::scaling(std::span<const double>(singular));
    }));

    const auto sh = AffineTransform<double>::shear(3, 0, 2, 3.0);
    const auto sheared = apply_one(sh, {1.0, 1.0, 2.0});
    PANTR_CHECK(sheared[0] == 7.0 && sheared[1] == 1.0 && sheared[2] == 2.0);
    PANTR_CHECK(throws_invalid([] { (void)AffineTransform<double>::shear(3, 1, 1, 2.0); }));
    PANTR_CHECK(throws_invalid([] { (void)AffineTransform<double>::shear(3, 0, 3, 2.0); }));
}

void test_inverse_round_trips_within_its_condition_number() {
    // Well conditioned by construction: the entries are small integers and the
    // matrix is diagonally dominant, so kappa_inf is a handful. It is computed
    // below rather than assumed.
    const auto t = make(3, {4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 5.0}, {1.0, -2.0, 0.5});
    const auto inv = t.inverse();

    double norm_a = 0.0;
    double norm_inv = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        double row_a = 0.0;
        double row_inv = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
            row_a += std::abs(pantr::at(t.matrix(), i, j));
            row_inv += std::abs(pantr::at(inv.matrix(), i, j));
        }
        norm_a = std::max(norm_a, row_a);
        norm_inv = std::max(norm_inv, row_inv);
    }
    const double kappa = norm_a * norm_inv;
    // One LU and one product, each backward stable; 4 n kappa eps is that with
    // the usual small constant, stated rather than fitted.
    const double budget = 4.0 * 3.0 * kappa * kEps;

    const std::vector<double> x{0.3, -1.25, 7.0};
    const auto there = apply_one(t, x);
    const auto back = apply_one(inv, there);
    for (std::size_t i = 0; i < 3; ++i) {
        PANTR_CHECK_MSG(std::abs(back[i] - x[i]) <= budget * std::max(1.0, std::abs(x[i])),
                        "round trip within kappa * eps");
    }

    const auto singular = make(2, {1.0, 2.0, 2.0, 4.0}, {0.0, 0.0});
    PANTR_CHECK_MSG(throws_invalid([&singular] { (void)singular.inverse(); }),
                    "a singular linear part is refused");
}

void test_compose_is_application_in_order() {
    const auto a = make(2, {0.0, -1.0, 1.0, 0.0}, {1.0, 0.0});
    const auto b = make(2, {2.0, 0.0, 0.0, 2.0}, {0.0, 3.0});
    // compose is defined as self(other(x)); checking that identity is the test,
    // rather than recomputing the matrix product a second way.
    const std::vector<double> x{1.0, 1.0};
    const auto composed = apply_one(a.compose(b), x);
    const auto stepwise = apply_one(a, apply_one(b, x));
    PANTR_CHECK(composed[0] == stepwise[0] && composed[1] == stepwise[1]);

    const auto c3 = AffineTransform<double>::identity(3);
    PANTR_CHECK(throws_invalid([&a, &c3] { (void)a.compose(c3); }));
}

void test_about_center_fixes_its_center() {
    const double quarter = std::numbers::pi / 2.0;
    const std::vector<double> center{2.0, -1.0};
    const auto r = AffineTransform<double>::rotation_2d(quarter)
                       .about_center(std::span<const double>(center));
    // The defining property: the centre is a fixed point. A conjugation written
    // the wrong way round moves it, and no shape check would notice.
    const auto fixed = apply_one(r, {2.0, -1.0});
    PANTR_CHECK(std::abs(fixed[0] - 2.0) <= kTrigSlack * kEps * 4.0);
    PANTR_CHECK(std::abs(fixed[1] + 1.0) <= kTrigSlack * kEps * 4.0);

    const std::vector<double> wrong{0.0, 0.0, 0.0};
    PANTR_CHECK(throws_invalid([&r, &wrong] {
        (void)r.about_center(std::span<const double>(wrong));
    }));
}

void test_apply_validates_its_shapes() {
    const auto t = AffineTransform<double>::identity(2);
    std::vector<double> pts{1.0, 2.0, 3.0};
    std::vector<double> out(4);
    PANTR_CHECK(throws_invalid([&t, &pts, &out] {
        t.apply(pantr::span2d<const double>(pts.data(), 1, 3),
                pantr::span2d<double>(out.data(), 1, 3));
    }));
    PANTR_CHECK(throws_invalid([&t, &pts, &out] {
        t.apply(pantr::span2d<const double>(pts.data(), 1, 2),
                pantr::span2d<double>(out.data(), 2, 2));
    }));
}

void test_the_type_is_generic_in_its_scalar() {
    const std::vector<float> factors{2.0F, 4.0F};
    const auto s = AffineTransform<float>::scaling(std::span<const float>(factors));
    PANTR_CHECK(s.dim() == 2);
    PANTR_CHECK(AffineTransform<float>::identity(2).compose(s) == s);
}

}  // namespace

// Forces every member of both instantiations to be compiled.
template class pantr::transform::AffineTransform<double>;
template class pantr::transform::AffineTransform<float>;

int main() {
    test_construction_validates();
    test_identity_is_exact();
    test_rotation_2d_against_the_analytic_answer();
    test_rotation_3d_about_a_coordinate_axis();
    test_mirror_is_an_involution_and_fixes_its_plane();
    test_scaling_and_shear_are_exact();
    test_inverse_round_trips_within_its_condition_number();
    test_compose_is_application_in_order();
    test_about_center_fixes_its_center();
    test_apply_validates_its_shapes();
    test_the_type_is_generic_in_its_scalar();
    return pantr::test::summary("test_affine");
}
