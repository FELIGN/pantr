/// \file
/// The Bézier extraction operators and the structural identity mask.
///
/// ## Where the expected values come from, and none of them from the oracle
///
/// This file is the C++ side's own check, so nothing below is a number read off
/// `pantr.bspline`. Four provenances, kept apart because they carry different
/// weight.
///
///  - **Derived by hand from the definition.** The quadratic three-element open
///    spline `[0,0,0,1,2,3,3,3]` is worked out in `check_quadratic_open` from
///    `N = C @ B` directly: each row is the Bernstein form of one B-spline
///    function on the element, obtained from its value at both ends and its
///    derivative at the left end. The derivation is written out there. Every entry
///    is a half, so the table is exact in `float32` as well as `float64` and the
///    comparison carries no tolerance.
///  - **Analytic invariants.** Each operator's **columns** sum to one, because
///    `sum_i N_i = 1` and `sum_j B_j = 1` and the Bernstein basis is independent.
///    Every entry is non-negative, being a product of convex combinations. Both
///    are checked over every case here, and the column sum is what catches a
///    transposition: the quadratic table above has row sums `1, 1.5, 0.5`, so a
///    transposed implementation fails it.
///  - **Degenerate families with a forced answer.** Degree 0, a Bézier-like knot
///    vector, an interior knot at full multiplicity, and degree 1 at any knots all
///    force the identity -- for degree 1 because the linear B-spline basis on an
///    element *is* the Bernstein basis of degree 1.
///  - **Symmetries.** A knot vector symmetric about its domain's midpoint has
///    operators that are index-reversals of each other, and the interior operator
///    of a *uniform* spline does not depend on whether the ends are clamped,
///    because a function's restriction to an interior element depends only on the
///    local knot pattern. The second one is what exercises the non-clamped
///    boundary-insertion branch against a value derived elsewhere.
///
/// ## What is not checked here
///
/// Agreement with the Python oracle, which is
/// `tests/parity/test_bspline_bezier_extraction.py`'s job, and the reproduction
/// identity `N_e = C_e @ B` evaluated at points -- there is no general B-spline
/// basis tabulation in C++ yet, so that check lives on the Python side where
/// `BsplineSpace1D.tabulate_basis` exists.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/knots.hpp"

namespace {

using pantr::at;
using pantr::span_nd;
using pantr::bspline::bezier_extraction_1d;
using pantr::bspline::bezier_structural_identity_mask;
using pantr::bspline::classify_knots;
using pantr::bspline::KnotClassRange;
using pantr::bspline::multiplicity_of_first_knot_in_domain;
using pantr::bspline::unique_knots_and_multiplicity;

/// One case's operators, kept alive alongside the view over them.
///
/// \tparam T Scalar type.
template <class T>
struct Operators {
    std::vector<T> storage;  ///< `(n_intervals, p+1, p+1)` row-major.
    std::size_t count = 0;   ///< How many operators there are.
    std::size_t side = 0;    ///< `degree + 1`.

    /// The view the builder wrote into.
    ///
    /// \return A rank-3 view over `storage`.
    [[nodiscard]] span_nd<const T, 3> view() const {
        return span_nd<const T, 3>(storage.data(), count, side, side);
    }
};

/// Build every operator of a knot vector.
///
/// \tparam T Scalar type.
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \param tol The absolute parametric tolerance; `knot_tolerance`'s when zero is
///        not wanted, but every case here passes one explicitly.
/// \return The operators.
template <class T>
Operators<T> build(const std::vector<T>& knots, std::int64_t degree, double tol) {
    const std::span<const T> span(knots);
    const KnotClassRange range = classify_knots<T>(span, degree, tol);
    const auto count = static_cast<std::size_t>(range.num_intervals());
    const auto side = static_cast<std::size_t>(degree) + 1;
    Operators<T> ops{std::vector<T>(count * side * side, T(0.0)), count, side};
    const span_nd<T, 3> view(ops.storage.data(), count, side, side);
    bezier_extraction_1d<T>(span, degree, tol, view);
    return ops;
}

/// The tolerance a column sum may miss one by.
///
/// Each entry is a chain of at most `degree` updates `alpha * x + beta * y`, each
/// committing four roundings -- `beta = 1 - alpha`, two products and their sum --
/// and the exact chain is a convex combination of values in `[0, 1]`, so an entry
/// carries at most `gamma_{4 degree}` of absolute error. Summing `degree + 1` of
/// them adds `gamma_{degree}` against a total of one. First order in `u`, that is
/// `(4 degree (degree + 1) + degree) u`.
///
/// \tparam T Scalar type.
/// \param degree The polynomial degree.
/// \return The absolute bound, zero for degree 0 where nothing is computed.
template <class T>
[[nodiscard]] double column_sum_tolerance(std::int64_t degree) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    const auto p = static_cast<double>(degree);
    return (4.0 * p * (p + 1.0) + p) * u;
}

/// Check the two analytic invariants every operator has.
///
/// \tparam T Scalar type.
/// \param ops The operators.
/// \param degree The polynomial degree.
/// \param label What case this is, for the failure message.
template <class T>
void check_invariants(const Operators<T>& ops, std::int64_t degree, const std::string& label) {
    const double bound = column_sum_tolerance<T>(degree);
    for (std::size_t e = 0; e < ops.count; ++e) {
        for (std::size_t j = 0; j < ops.side; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < ops.side; ++i) {
                const double entry = static_cast<double>(at(ops.view(), e, i, j));
                PANTR_CHECK_MSG(entry >= 0.0, label + ": a negative entry");
                sum += entry;
            }
            PANTR_CHECK_MSG(std::abs(sum - 1.0) <= bound, label + ": column " + std::to_string(j)
                                                              + " of operator "
                                                              + std::to_string(e) + " sums to "
                                                              + std::to_string(sum));
        }
    }
}

/// Compare against an expected table, bit for bit.
///
/// \tparam T Scalar type.
/// \param ops The operators.
/// \param expected One flat row-major table per operator.
/// \param label What case this is, for the failure message.
template <class T>
void same(const Operators<T>& ops, const std::vector<std::vector<double>>& expected,
          const std::string& label) {
    PANTR_CHECK_MSG(ops.count == expected.size(), label + ": operator count");
    if (ops.count != expected.size()) {
        return;
    }
    for (std::size_t e = 0; e < ops.count; ++e) {
        for (std::size_t i = 0; i < ops.side; ++i) {
            for (std::size_t j = 0; j < ops.side; ++j) {
                const T want = static_cast<T>(expected[e][i * ops.side + j]);
                PANTR_CHECK_MSG(at(ops.view(), e, i, j) == want,
                                label + ": entry (" + std::to_string(e) + ", "
                                    + std::to_string(i) + ", " + std::to_string(j) + ")");
            }
        }
    }
}

/// Every operator is the identity.
///
/// \tparam T Scalar type.
/// \param ops The operators.
/// \param label What case this is, for the failure message.
template <class T>
void all_identity(const Operators<T>& ops, const std::string& label) {
    PANTR_CHECK_MSG(ops.count >= 1, label + ": no operator was built");
    for (std::size_t e = 0; e < ops.count; ++e) {
        for (std::size_t i = 0; i < ops.side; ++i) {
            for (std::size_t j = 0; j < ops.side; ++j) {
                const T want = (i == j) ? T(1.0) : T(0.0);
                PANTR_CHECK_MSG(at(ops.view(), e, i, j) == want,
                                label + ": entry (" + std::to_string(e) + ", "
                                    + std::to_string(i) + ", " + std::to_string(j)
                                    + ") is not the identity's");
            }
        }
    }
}

/// The quadratic three-element open spline, derived by hand from `N = C @ B`.
///
/// Knots `[0,0,0,1,2,3,3,3]`, degree 2, five basis functions, elements `[0,1]`,
/// `[1,2]`, `[2,3]`. On the reference interval `B_0 = (1-t)^2`, `B_1 = 2t(1-t)`,
/// `B_2 = t^2`, and a quadratic is fixed by its two end values and its slope at
/// the left end, where a Bernstein form has slope `2 (c_1 - c_0)`.
///
///  - **Element `[0,1]`**, functions `N_0, N_1, N_2`. `N_0` has local knots
///    `[0,0,0,1]` so `N_0 = (1-t)^2 = B_0`, giving the row `[1, 0, 0]`. `N_2` has
///    local knots `[0,1,2,3]` so `N_2 = t^2 / 2`, giving `[0, 0, 1/2]`. `N_1` is
///    what partition of unity leaves: `1 - (1-t)^2 - t^2/2 = 2t - 3t^2/2`, whose
///    ends are `0` and `1/2` and whose left slope is `2`, so `c_1 = 1` and the row
///    is `[0, 1, 1/2]`.
///  - **Element `[1,2]`**, functions `N_1, N_2, N_3`. `N_1` restricted there is
///    `(1-t)^2 / 2`, giving `[1/2, 0, 0]`; `N_3` is the mirror image, `t^2 / 2`,
///    giving `[0, 0, 1/2]`; and `N_2 = 1 - (1-t)^2/2 - t^2/2` has ends `1/2`,
///    `1/2` and left slope `1`, so `c_1 = 1` and the row is `[1/2, 1, 1/2]`.
///  - **Element `[2,3]`** is element `[0,1]` reflected, since the knot vector is
///    symmetric about `1.5`, so its operator is the first one with both indices
///    reversed.
const std::vector<std::vector<double>> kQuadraticOpen = {
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5},
    {0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5},
    {0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0},
};

/// The hand-derived quadratic table, at both widths.
template <class T>
void check_quadratic_open() {
    const std::vector<T> knots = {T(0.0), T(0.0), T(0.0), T(1.0),
                                  T(2.0), T(3.0), T(3.0), T(3.0)};
    const Operators<T> ops = build<T>(knots, 2, 0.0);
    same(ops, kQuadraticOpen, "quadratic open");
    check_invariants(ops, 2, "quadratic open");
}

/// The forced-identity families.
template <class T>
void check_identity_families() {
    // Degree 0: the one-by-one operator is [[1]] and nothing is computed.
    all_identity(build<T>({T(0.0), T(1.0)}, 0, 0.0), "degree 0");

    // A Bézier-like vector already is one patch.
    all_identity(build<T>({T(0.0), T(0.0), T(0.0), T(1.0), T(1.0), T(1.0)}, 2, 0.0),
                 "bezier-like");

    // Interior knots at full multiplicity decouple every element.
    all_identity(build<T>({T(0.0), T(0.0), T(0.0), T(0.5), T(0.5), T(0.5), T(1.0), T(1.0),
                           T(1.0)},
                          2, 0.0),
                 "interior full multiplicity");

    // Degree 1: the linear B-spline basis on an element IS the linear Bernstein
    // basis, whatever the knots, so no insertion can change anything.
    all_identity(build<T>({T(0.0), T(0.0), T(1.0), T(2.0), T(3.0), T(3.0)}, 1, 0.0),
                 "degree 1 open");
    all_identity(build<T>({T(0.0), T(0.25), T(0.75), T(1.5), T(2.0), T(3.0)}, 1, 0.0),
                 "degree 1 non-uniform, unclamped");
}

/// The uniform interior operator does not depend on the clamping.
///
/// This is what exercises the boundary-insertion branch: an unclamped vector has a
/// boundary multiplicity of one, so the first interval is short of `degree - 1`
/// insertions. The value it must reach is the interior operator of the clamped
/// spline above, which was derived by hand there.
template <class T>
void check_unclamped_uniform_matches_the_interior() {
    const std::vector<T> knots = {T(0.0), T(0.5), T(1.0), T(1.5), T(2.0), T(2.5), T(3.0)};
    const Operators<T> ops = build<T>(knots, 2, 0.0);
    same(ops, {kQuadraticOpen[1], kQuadraticOpen[1]}, "unclamped uniform");
    check_invariants(ops, 2, "unclamped uniform");
}

/// A symmetric knot vector's operators are each other's index reversals.
template <class T>
void check_mirror_symmetry() {
    // Cubic, two uniform elements, clamped: symmetric about 1.
    const std::vector<T> knots = {T(0.0), T(0.0), T(0.0), T(0.0), T(1.0),
                                  T(2.0), T(2.0), T(2.0), T(2.0)};
    const Operators<T> ops = build<T>(knots, 3, 0.0);
    PANTR_CHECK_MSG(ops.count == 2, "mirror symmetry: two elements");
    if (ops.count != 2) {
        return;
    }
    const std::size_t n = ops.side;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            PANTR_CHECK_MSG(at(ops.view(), 0, i, j)
                                == at(ops.view(), 1, n - 1 - i, n - 1 - j),
                            "mirror symmetry: entry (" + std::to_string(i) + ", "
                                + std::to_string(j) + ")");
        }
    }
    check_invariants(ops, 3, "cubic two-element");

    // Three uniform elements at degree 3: the entries are sixths, so this is the
    // one case here whose invariants need the derived tolerance rather than
    // exactness.
    const std::vector<T> three = {T(0.0), T(0.0), T(0.0), T(0.0), T(1.0), T(2.0),
                                  T(3.0), T(3.0), T(3.0), T(3.0)};
    const Operators<T> wider = build<T>(three, 3, 0.0);
    PANTR_CHECK_MSG(wider.count == 3, "cubic three-element: element count");
    check_invariants(wider, 3, "cubic three-element");
    PANTR_CHECK_MSG(column_sum_tolerance<T>(3) > 0.0,
                    "the derived column-sum tolerance must be positive above degree 0");
}

/// The boundary count is not the class multiplicity, and the builder wants the first.
///
/// `design/extraction_port.md`'s 2026-09-01 amendment says the boundary
/// multiplicity is `multiplicity_in_domain().front()`. It is not: the two are
/// different computations over different index ranges, and this vector separates
/// them. Pinned here because a port that read the class multiplicity instead would
/// still pass every invariant above -- what catches it is agreement with the
/// oracle, in the parity suite.
void check_the_boundary_count_is_not_the_class_multiplicity() {
    const std::vector<double> repeated = {0.0, 0.4, 0.5, 0.5, 1.0, 1.5, 2.0, 2.5};
    const std::span<const double> span(repeated);
    const auto classes = unique_knots_and_multiplicity<double>(span, 2, 0.0);
    PANTR_CHECK_MSG(classes.multiplicity[classes.domain_begin] == 2,
                    "the first in-domain class of the repeated vector holds two knots");
    PANTR_CHECK_MSG(multiplicity_of_first_knot_in_domain<double>(span, 2, 0.0) == 1,
                    "only one of the first degree+1 knots is the domain's first knot");

    // A clamped vector is where the two agree, which is why the confusion survived.
    const std::vector<double> clamped = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::span<const double> clamped_span(clamped);
    const auto clamped_classes = unique_knots_and_multiplicity<double>(clamped_span, 2, 0.0);
    PANTR_CHECK_MSG(clamped_classes.multiplicity[clamped_classes.domain_begin] == 3,
                    "the clamped vector's first in-domain class holds three knots");
    PANTR_CHECK_MSG(multiplicity_of_first_knot_in_domain<double>(clamped_span, 2, 0.0) == 3,
                    "and all three are the domain's first knot");
}

/// The structural identity mask, read off the multiplicities by eye.
void check_identity_mask() {
    std::array<bool, 2> flags{};
    const std::span<bool> out(flags);

    // Clamped quadratic, one interior simple knot: neither element is decoupled.
    bezier_structural_identity_mask(std::vector<std::int64_t>{3, 1, 3}, 2, out);
    PANTR_CHECK_MSG(!flags[0] && !flags[1], "a simple interior knot decouples nothing");

    // The interior knot at full multiplicity decouples both.
    bezier_structural_identity_mask(std::vector<std::int64_t>{3, 3, 3}, 2, out);
    PANTR_CHECK_MSG(flags[0] && flags[1], "a full interior knot decouples both");

    // Only the right element is a patch: the middle and right knots are full, the
    // left is not.
    bezier_structural_identity_mask(std::vector<std::int64_t>{1, 3, 3}, 2, out);
    PANTR_CHECK_MSG(!flags[0] && flags[1], "only the right element is a patch");

    // Degree 0: every class holds at least one knot and the threshold is one, so
    // every element is a patch.
    bezier_structural_identity_mask(std::vector<std::int64_t>{1, 1, 1}, 0, out);
    PANTR_CHECK_MSG(flags[0] && flags[1], "at degree 0 every element is a patch");
}

/// The mask agrees with the operators it describes.
///
/// The two are computed by different routes -- one counts multiplicities, the other
/// runs the insertions -- so agreeing is a real check on both. Only one direction
/// is asserted: a flagged element must have the identity operator. The converse is
/// false by design, since degree 0 and degree 1 have identity operators everywhere
/// while the mask reports what the *structure* forces.
template <class T>
void check_the_mask_agrees_with_the_operators() {
    struct Case {
        std::vector<T> knots;
        std::int64_t degree;
    };
    const std::vector<Case> cases = {
        {{T(0.0), T(0.0), T(0.0), T(1.0), T(2.0), T(3.0), T(3.0), T(3.0)}, 2},
        {{T(0.0), T(0.0), T(0.0), T(0.5), T(0.5), T(0.5), T(1.0), T(1.0), T(1.0)}, 2},
        {{T(0.0), T(0.0), T(0.0), T(0.0), T(1.0), T(2.0), T(2.0), T(2.0), T(2.0)}, 3},
        {{T(0.0), T(0.5), T(1.0), T(1.5), T(2.0), T(2.5), T(3.0)}, 2},
    };
    for (const Case& one : cases) {
        const std::span<const T> span(one.knots);
        const auto classes = unique_knots_and_multiplicity<T>(span, one.degree, 0.0);
        const std::span<const std::int64_t> in_domain =
            std::span<const std::int64_t>(classes.multiplicity)
                .subspan(classes.domain_begin, classes.domain_end - classes.domain_begin);
        // A fixed array rather than a vector: `std::vector<bool>` is a bitset and
        // yields no `bool*` for a span to point at.
        std::array<bool, 8> flags{};
        PANTR_CHECK_MSG(in_domain.size() - 1 <= flags.size(), "mask agreement: case too wide");
        bezier_structural_identity_mask(in_domain, one.degree,
                                        std::span<bool>(flags.data(), in_domain.size() - 1));

        const Operators<T> ops = build<T>(one.knots, one.degree, 0.0);
        check_invariants(ops, one.degree, "mask agreement");
        for (std::size_t e = 0; e < ops.count; ++e) {
            if (!flags[e]) {
                continue;
            }
            for (std::size_t i = 0; i < ops.side; ++i) {
                for (std::size_t j = 0; j < ops.side; ++j) {
                    const T want = (i == j) ? T(1.0) : T(0.0);
                    PANTR_CHECK_MSG(at(ops.view(), e, i, j) == want,
                                    "a flagged element must carry the identity operator");
                }
            }
        }
    }
}

}  // namespace

int main() {
    check_quadratic_open<double>();
    check_quadratic_open<float>();
    check_identity_families<double>();
    check_identity_families<float>();
    check_unclamped_uniform_matches_the_interior<double>();
    check_unclamped_uniform_matches_the_interior<float>();
    check_mirror_symmetry<double>();
    check_mirror_symmetry<float>();
    check_the_boundary_count_is_not_the_class_multiplicity();
    check_identity_mask();
    check_the_mask_agrees_with_the_operators<double>();
    check_the_mask_agrees_with_the_operators<float>();
    return pantr::test::summary("test_bspline_extraction");
}
