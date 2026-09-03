/// \file
/// The Bézier and Lagrange extraction operators and the structural identity masks.
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
/// ## The Lagrange half
///
/// `A_e = C_e L`, and the change-of-basis matrix is an argument rather than something
/// this file builds, so the checks are about the product:
///
///  - **`L = I` reproduces the Bézier operator exactly.** Every term of the
///    contraction is `C[i,k] * 0` or `C[i,j] * 1`, so nothing rounds and the two
///    builders must agree bit for bit. That is what pins the index order: a
///    transposed product would give `C^T` here.
///  - **A hand-derived exact table.** At degree 2 with equispaced nodes,
///    `L = [[1, 1/4, 0], [0, 1/2, 0], [0, 1/4, 1]]` from `L[j,k] = B_j(x_k)` at
///    `x = 0, 1/2, 1`, and the quadratic three-element open spline's Bézier table is
///    halves, so every entry of the product is a binary rational and the comparison
///    carries no tolerance. Written out in `check_lagrange_quadratic_open`.
///  - **The columns still sum to one.** `L` is column-stochastic -- its columns are
///    the Bernstein basis at a node, non-negative and summing to one -- so a product
///    of two column-stochastic matrices is column-stochastic. This also says the
///    Lagrange operator is entrywise non-negative, which
///    `design/extraction_port.md` denied and which is checked here.
///
/// ## What is not checked here
///
/// Agreement with the Python oracle, which is
/// `tests/parity/test_bspline_bezier_extraction.py`'s job, and the reproduction
/// identity `N_e = C_e @ B` evaluated at points -- there is no general B-spline
/// basis tabulation in C++ yet, so that check lives on the Python side where
/// `BsplineSpace1D.tabulate_basis` exists.

#include <algorithm>
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
using pantr::span2d;
using pantr::span_nd;
using pantr::bspline::bezier_extraction_1d;
using pantr::bspline::bezier_structural_identity_mask;
using pantr::bspline::lagrange_extraction_1d;
using pantr::bspline::lagrange_structural_identity_mask;
using pantr::bspline::classify_knots;
using pantr::bspline::KnotClassRange;
using pantr::bspline::KnotClasses;
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

/// The number of insertions on the longest dependency chain of a knot vector.
///
/// One outer iteration of an insertion sequence is one stage: the value of column
/// `k` after iteration `r` depends on columns `k` and `k - 1` after `r - 1`. An
/// element's own sequence is therefore `degree - multiplicity` stages long, the
/// boundary sequence adds `degree - boundary` on top of element 0, and the
/// inter-element **copies** carry the chain forward -- a copy commits no rounding
/// but inherits the error accumulated so far. The sum over the whole vector is thus
/// an upper bound on any single entry's chain.
///
/// **It is not `degree`, and an earlier version of this file said it was.** Element
/// 0 alone can reach `2 (degree - 1)` stages -- both its terms are at most
/// `degree - 1`, since a knot class has multiplicity at least one and `boundary` is
/// at least one because index `degree` matches itself -- and a vector of `n`
/// elements reaches `O(n degree)`. The bound below was therefore too tight by that factor. It never
/// failed, because the observed error is orders below either version, which is
/// exactly the shape of claim nothing in a suite can distinguish.
///
/// \tparam T Scalar type.
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \param tol The absolute parametric tolerance.
/// \return The stage count, zero when no insertion runs at all.
template <class T>
[[nodiscard]] std::int64_t insertion_stages(const std::vector<T>& knots, std::int64_t degree,
                                            double tol) {
    const std::span<const T> span(knots);
    const KnotClasses<T> classes = unique_knots_and_multiplicity<T>(span, degree, tol);
    const std::span<const std::int64_t> in_domain =
        std::span<const std::int64_t>(classes.multiplicity)
            .subspan(classes.domain_begin, classes.domain_end - classes.domain_begin);

    std::int64_t stages =
        std::max<std::int64_t>(degree - multiplicity_of_first_knot_in_domain<T>(span, degree, tol),
                               0);
    for (std::size_t e = 1; e < in_domain.size(); ++e) {
        stages += std::max<std::int64_t>(degree - in_domain[e], 0);
    }
    return stages;
}

/// The tolerance a column sum may miss one by.
///
/// Each stage of an entry's chain forms an exact convex combination of values in
/// `[0, 1]`, whose weights are non-negative and sum to one, so the propagated error
/// carries forward with weight at most one. **The rounding of `alpha` itself
/// cancels**: `alpha` and `beta = 1 - alpha` enter as a pair, and what the row bound
/// sees is `|A + B - 1|`, whose defect is at most `u/2` however `alpha` was
/// computed. That leaves **2.5 roundings per stage**, not four, so an entry ends up
/// within `gamma_{2.5 S}` of its exact value, where `S` is the chain length
/// `insertion_stages` reports. Summing `degree + 1` of them against an exact total
/// of one adds `gamma_{degree}`; the two compose by Higham, *Accuracy and Stability
/// of Numerical Algorithms*, 2nd ed., Lemma 3.3.
///
/// `gamma_{2.5 S + degree}` is sharp and holds exactly, not to first order. What
/// ships is `gamma_{3 S + degree}`: half a rounding per stage of declared slack over
/// a proved bound, so the coefficient is an integer. The closed form
/// `m u / (1 - m u)` is used rather than the truncation `m u`, because `S` grows
/// with the element count rather than with the degree.
///
/// **Stated hypothesis: no underflow in the operator entries**, which a purely
/// relative rounding model requires and this derivation does not establish.
/// Subnormal entries *are* reachable at `float32` with mixed per-gap knot ratios;
/// the bound was observed to hold on them by a wide margin, but observing is not
/// covering.
///
/// The refusal below is `PANTR_PRECONDITION`, which is `assert` and **compiles to
/// nothing under `NDEBUG`** -- and the `gcc` preset that runs this test builds with
/// `-DNDEBUG`. Past the runaway point this returns a negative or infinite bound
/// rather than refusing. Reaching it takes on the order of a million knots at
/// `float32`, which nothing here approaches, so it is dormant; the Python
/// counterpart raises for real.
///
/// \tparam T Scalar type.
/// \param degree The polynomial degree.
/// \param stages The chain length, from `insertion_stages`.
/// \return The absolute bound, zero when nothing is computed.
template <class T>
[[nodiscard]] double column_sum_tolerance(std::int64_t degree, std::int64_t stages) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    const double m = 3.0 * static_cast<double>(stages) + static_cast<double>(degree);
    PANTR_PRECONDITION(m * u < 1.0, "gamma runs away to one past this many roundings");
    return m * u / (1.0 - m * u);
}

/// Check the two analytic invariants every operator has.
///
/// \tparam T Scalar type.
/// \param ops The operators.
/// \param knots The knot vector they were built from, which sizes the bound.
/// \param degree The polynomial degree.
/// \param tol The absolute parametric tolerance the operators were built at.
/// \param label What case this is, for the failure message.
template <class T>
void check_invariants(const Operators<T>& ops, const std::vector<T>& knots, std::int64_t degree,
                      double tol, const std::string& label) {
    const double bound = column_sum_tolerance<T>(degree, insertion_stages<T>(knots, degree, tol));
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
    check_invariants(ops, knots, 2, 0.0, "quadratic open");
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
/// This is what exercises the boundary-insertion branch here: an unclamped vector
/// has a boundary multiplicity of one, so the first interval is short of
/// `degree - 1` insertions. The value it must reach is the interior operator of the
/// clamped spline above, which was derived by hand there.
///
/// **It reaches the branch in its degenerate instance, and two things follow.**
/// Uniform knots make `alpha` exactly one half, so `alpha == beta` and this case
/// cannot tell which of the two multiplies which column; and `reg` is one, so no
/// `r`-loop chaining happens. The chained, `alpha != beta` instance is covered by
/// the parity suite's unclamped non-uniform cubic and by its sweep, not here. Do
/// not read this case as pinning the branch in general.
template <class T>
void check_unclamped_uniform_matches_the_interior() {
    const std::vector<T> knots = {T(0.0), T(0.5), T(1.0), T(1.5), T(2.0), T(2.5), T(3.0)};
    const Operators<T> ops = build<T>(knots, 2, 0.0);
    same(ops, {kQuadraticOpen[1], kQuadraticOpen[1]}, "unclamped uniform");
    check_invariants(ops, knots, 2, 0.0, "unclamped uniform");
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
    check_invariants(ops, knots, 3, 0.0, "cubic two-element");

    // Three uniform elements at degree 3: the entries are sixths, so this is the
    // one case here whose invariants need the derived tolerance rather than
    // exactness.
    const std::vector<T> three = {T(0.0), T(0.0), T(0.0), T(0.0), T(1.0), T(2.0),
                                  T(3.0), T(3.0), T(3.0), T(3.0)};
    const Operators<T> wider = build<T>(three, 3, 0.0);
    PANTR_CHECK_MSG(wider.count == 3, "cubic three-element: element count");
    check_invariants(wider, three, 3, 0.0, "cubic three-element");
    PANTR_CHECK_MSG(insertion_stages<T>(three, 3, 0.0) > 3,
                    "three cubic elements chain more insertions than the degree, which is "
                    "what the corrected bound accounts for");
}

/// Push the sliding knot window as close to the end of the vector as it goes.
///
/// The header derives that the window `knots[w .. w + degree]` always fits, with
/// `w` at most the last knot index of the element's own class and therefore at most
/// `n - degree - 2`. Nothing else here comes near that: the hand-picked cases are
/// low degree with simple interior knots. This one is built to sit on it -- degree
/// 5, an interior class of multiplicity `degree - 1` immediately before the last
/// in-domain class, so the last contracting element's window ends one knot short of
/// the vector -- and its value is checked only through the invariants, since the
/// point is the addressing rather than the answer.
///
/// Its real assertion is the sanitizer's: run under the `gcc-debug` preset, an
/// off-by-one in the window is an AddressSanitizer report rather than a wrong
/// number.
///
/// \tparam T Scalar type.
template <class T>
void check_the_window_reaches_the_end_of_the_vector() {
    const std::int64_t degree = 5;
    // Breakpoints 0, 1, 2, 3, with 2 carrying multiplicity `degree - 1` = 4, so the
    // element [2, 3] is the last contracting one and its window starts as late as
    // the recurrence lets it.
    std::vector<T> knots;
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(T(0.0));
    }
    knots.push_back(T(1.0));
    for (std::int64_t i = 0; i < degree - 1; ++i) {
        knots.push_back(T(2.0));
    }
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(T(3.0));
    }
    const Operators<T> ops = build<T>(knots, degree, 0.0);
    PANTR_CHECK_MSG(ops.count == 3, "the adversarial vector should span three elements");
    check_invariants(ops, knots, degree, 0.0, "window at the end of the vector");

    // And the same vector with the high-multiplicity class at the far end instead,
    // where the window's last read is the vector's last knot but one.
    std::vector<T> shifted;
    for (std::int64_t i = 0; i <= degree; ++i) {
        shifted.push_back(T(0.0));
    }
    shifted.push_back(T(1.0));
    shifted.push_back(T(2.0));
    for (std::int64_t i = 0; i < degree - 1; ++i) {
        shifted.push_back(T(2.5));
    }
    for (std::int64_t i = 0; i <= degree; ++i) {
        shifted.push_back(T(3.0));
    }
    const Operators<T> late = build<T>(shifted, degree, 0.0);
    PANTR_CHECK_MSG(late.count == 4, "the shifted vector should span four elements");
    check_invariants(late, shifted, degree, 0.0, "window with a late high-multiplicity class");
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
        check_invariants(ops, one.knots, one.degree, 0.0, "mask agreement");
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

/// Build every Lagrange operator of a knot vector.
///
/// \tparam T Scalar type.
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \param tol The absolute parametric tolerance.
/// \param matrix The `(degree + 1, degree + 1)` change-of-basis matrix, row-major.
/// \return The operators.
template <class T>
Operators<T> build_lagrange(const std::vector<T>& knots, std::int64_t degree, double tol,
                            const std::vector<T>& matrix) {
    const std::span<const T> span(knots);
    const KnotClassRange range = classify_knots<T>(span, degree, tol);
    const auto count = static_cast<std::size_t>(range.num_intervals());
    const auto side = static_cast<std::size_t>(degree) + 1;
    Operators<T> ops{std::vector<T>(count * side * side, T(0.0)), count, side};
    const span_nd<T, 3> view(ops.storage.data(), count, side, side);
    lagrange_extraction_1d<T>(span, degree, tol, span2d<const T>(matrix.data(), side, side),
                              view);
    return ops;
}

/// The identity matrix of a given side, row-major.
///
/// \tparam T Scalar type.
/// \param side The number of rows and columns.
/// \return The matrix.
template <class T>
std::vector<T> identity_matrix(std::size_t side) {
    std::vector<T> matrix(side * side, T(0.0));
    for (std::size_t i = 0; i < side; ++i) {
        matrix[i * side + i] = T(1.0);
    }
    return matrix;
}

/// With the identity change of basis, the Lagrange builder is the Bézier one.
///
/// Bitwise, not within a tolerance: every term of the contraction is `C[i,k] * 0` or
/// `C[i,j] * 1`, and a sum of exact zeros with one exact term rounds nothing. That
/// is what makes this able to catch a transposed product, which would return `C^T`.
template <class T>
void check_lagrange_with_the_identity_reproduces_bezier() {
    const std::vector<std::vector<T>> vectors = {
        {T(0.0), T(0.0), T(0.0), T(1.0), T(2.0), T(3.0), T(3.0), T(3.0)},
        {T(0.0), T(0.5), T(1.0), T(1.5), T(2.0), T(2.5), T(3.0)},
        {T(0.0), T(0.0), T(0.0), T(0.0), T(1.0), T(2.0), T(3.0), T(4.0), T(4.0), T(4.0), T(4.0)},
    };
    const std::vector<std::int64_t> degrees = {2, 2, 3};
    for (std::size_t c = 0; c < vectors.size(); ++c) {
        const std::size_t side = static_cast<std::size_t>(degrees[c]) + 1;
        const Operators<T> bezier = build<T>(vectors[c], degrees[c], 0.0);
        const Operators<T> lagrange =
            build_lagrange<T>(vectors[c], degrees[c], 0.0, identity_matrix<T>(side));
        PANTR_CHECK_MSG(bezier.storage.size() == lagrange.storage.size(),
                        "the two builders disagree on how many operators there are");
        for (std::size_t n = 0; n < bezier.storage.size(); ++n) {
            PANTR_CHECK_MSG(bezier.storage[n] == lagrange.storage[n],
                            "the identity change of basis moved a bit");
        }
    }
}

/// The quadratic three-element open spline against a hand-derived exact table.
///
/// The Bézier table is `check_quadratic_open`'s. The equispaced degree-2
/// Lagrange-to-Bernstein matrix is `L[j,k] = B_j(x_k)` at `x = 0, 1/2, 1`, i.e.
/// `[[1, 1/4, 0], [0, 1/2, 0], [0, 1/4, 1]]`. Multiplying out gives, per element,
///
///     e = 0: [[1, 1/4, 0], [0, 5/8, 1/2], [0, 1/8, 1/2]]
///     e = 1: [[1/2, 1/8, 0], [1/2, 3/4, 1/2], [0, 1/8, 1/2]]
///     e = 2: [[1/2, 1/8, 0], [1/2, 5/8, 0], [0, 1/4, 1]]
///
/// Every entry is a binary rational, and so is every partial sum of the contraction,
/// so the comparison is exact in `float32` as well as `float64`.
template <class T>
void check_lagrange_quadratic_open() {
    const std::vector<T> knots = {T(0.0), T(0.0), T(0.0), T(1.0),
                                  T(2.0), T(3.0), T(3.0), T(3.0)};
    const std::vector<T> matrix = {T(1.0), T(0.25), T(0.0), T(0.0), T(0.5),
                                   T(0.0), T(0.0),  T(0.25), T(1.0)};
    const std::vector<T> expected = {
        T(1.0),  T(0.25),  T(0.0), T(0.0), T(0.625), T(0.5), T(0.0), T(0.125), T(0.5),
        T(0.5),  T(0.125), T(0.0), T(0.5), T(0.75),  T(0.5), T(0.0), T(0.125), T(0.5),
        T(0.5),  T(0.125), T(0.0), T(0.5), T(0.625), T(0.0), T(0.0), T(0.25),  T(1.0),
    };
    const Operators<T> ops = build_lagrange<T>(knots, 2, 0.0, matrix);
    PANTR_CHECK_MSG(ops.storage.size() == expected.size(),
                    "the quadratic open spline should have three 3x3 operators");
    for (std::size_t n = 0; n < expected.size(); ++n) {
        PANTR_CHECK_MSG(ops.storage[n] == expected[n],
                        "a Lagrange operator entry misses its exact binary rational");
    }
}

/// Each Lagrange operator's column sums reproduce the matrix's own, and none is negative.
///
/// The invariant is `sum_i A[i,j] = sum_k (sum_i C[i,k]) L[k,j] = sum_k L[k,j]`, using
/// the Bézier column sums. It is compared against `sum_k L[k,j]` **as the supplied
/// matrix actually sums**, not against one: `L`'s columns sum to one in exact
/// arithmetic, being the Bernstein basis at a node, but the thirds of the cubic case
/// below do not sum to one in binary, and folding that defect in exactly is better
/// than bounding it. The bound then covers only the Bézier chain plus the
/// contraction's own `gamma_{degree + 1}`, and the amplification is one because every
/// entry stays in `[0, 1]`.
///
/// Non-negativity comes with it: `L` is entrywise non-negative because every Lagrange
/// node lies in `[0, 1]` where the Bernstein basis is, and `C_e` is a product of
/// convex combinations, so the product cannot be negative.
template <class T>
void check_lagrange_columns_are_a_partition_of_unity() {
    // Equispaced degree 2 and degree 3, `L[j,k] = B_j(x_k)`. Degree 3's nodes are
    // 0, 1/3, 2/3, 1, so its entries are ninths and twenty-sevenths.
    const std::vector<T> matrix2 = {T(1.0), T(0.25), T(0.0), T(0.0), T(0.5),
                                    T(0.0), T(0.0),  T(0.25), T(1.0)};
    const std::vector<T> matrix3 = {
        T(1.0), T(8.0 / 27.0), T(1.0 / 27.0), T(0.0), T(0.0), T(12.0 / 27.0), T(6.0 / 27.0),
        T(0.0), T(0.0),        T(6.0 / 27.0), T(12.0 / 27.0), T(0.0), T(0.0), T(1.0 / 27.0),
        T(8.0 / 27.0), T(1.0),
    };
    struct Case {
        std::vector<T> knots;
        std::int64_t degree;
        const std::vector<T>* matrix;
    };
    const std::vector<Case> cases = {
        {{T(0.0), T(0.0), T(0.0), T(1.0), T(2.0), T(3.0), T(3.0), T(3.0)}, 2, &matrix2},
        {{T(0.0), T(0.0), T(0.0), T(1.0), T(1.0), T(2.0), T(3.0), T(3.0), T(3.0)}, 2, &matrix2},
        {{T(0.0), T(0.5), T(1.0), T(1.5), T(2.0), T(2.5), T(3.0)}, 2, &matrix2},
        {{T(0.0), T(0.0), T(0.0), T(0.0), T(0.3), T(0.7), T(1.0), T(1.0), T(1.0), T(1.0)}, 3,
         &matrix3},
    };
    for (const Case& one : cases) {
        const Operators<T> ops = build_lagrange<T>(one.knots, one.degree, 0.0, *one.matrix);
        // The Bézier chain, plus `degree + 1` roundings for the contraction, plus the
        // `degree` additions of the column sum itself. `column_sum_tolerance` already
        // charges the last of those, so only the contraction is added here.
        const std::int64_t stages = insertion_stages<T>(one.knots, one.degree, 0.0);
        const double bound =
            column_sum_tolerance<T>(one.degree, stages + one.degree + 1);
        for (std::size_t j = 0; j < ops.side; ++j) {
            double target = 0.0;
            for (std::size_t k = 0; k < ops.side; ++k) {
                target += static_cast<double>((*one.matrix)[k * ops.side + j]);
            }
            PANTR_CHECK_MSG(std::abs(target - 1.0) <= bound,
                            "the supplied change-of-basis matrix is not column-stochastic");
            for (std::size_t e = 0; e < ops.count; ++e) {
                double total = 0.0;
                for (std::size_t i = 0; i < ops.side; ++i) {
                    const double entry = static_cast<double>(at(ops.view(), e, i, j));
                    PANTR_CHECK_MSG(entry >= 0.0,
                                    "a Lagrange operator entry is negative, so neither factor "
                                    "of the product is column-stochastic after all");
                    total += entry;
                }
                PANTR_CHECK_MSG(std::abs(total - target) <= bound,
                                "a Lagrange operator column does not sum to the matrix's");
            }
        }
    }
}

/// The Lagrange mask is the Bézier mask under the identity and all-false otherwise.
template <class T>
void check_lagrange_identity_mask() {
    const std::vector<std::int64_t> multiplicity = {3, 1, 3, 3};
    std::array<bool, 3> lagrange{};
    std::array<bool, 3> bezier{};
    const std::span<bool> lagrange_flags(lagrange.data(), lagrange.size());
    const std::span<bool> bezier_flags(bezier.data(), bezier.size());

    const std::vector<T> identity = identity_matrix<T>(3);
    bezier_structural_identity_mask(multiplicity, 2, bezier_flags);
    lagrange_structural_identity_mask<T>(multiplicity, 2,
                                         span2d<const T>(identity.data(), 3, 3), lagrange_flags);
    PANTR_CHECK_MSG(bezier[2] && !bezier[0] && !bezier[1],
                    "the Bézier mask should flag only the element between two full knots");
    for (std::size_t e = 0; e < bezier.size(); ++e) {
        PANTR_CHECK_MSG(lagrange[e] == bezier[e],
                        "under the identity change of basis the two masks must agree");
    }

    // One entry off the identity, by the smallest amount there is: the predicate is
    // exact, so this must flip every flag to false.
    std::vector<T> nudged = identity;
    nudged[1] = std::numeric_limits<T>::denorm_min();
    lagrange_structural_identity_mask<T>(multiplicity, 2,
                                         span2d<const T>(nudged.data(), 3, 3), lagrange_flags);
    for (const bool flag : lagrange) {
        PANTR_CHECK_MSG(!flag, "a change of basis that is not the identity forbids every flag");
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
    check_the_window_reaches_the_end_of_the_vector<double>();
    check_the_window_reaches_the_end_of_the_vector<float>();
    check_the_boundary_count_is_not_the_class_multiplicity();
    check_identity_mask();
    check_the_mask_agrees_with_the_operators<double>();
    check_the_mask_agrees_with_the_operators<float>();
    check_lagrange_with_the_identity_reproduces_bezier<double>();
    check_lagrange_with_the_identity_reproduces_bezier<float>();
    check_lagrange_quadratic_open<double>();
    check_lagrange_quadratic_open<float>();
    check_lagrange_columns_are_a_partition_of_unity<double>();
    check_lagrange_columns_are_a_partition_of_unity<float>();
    check_lagrange_identity_mask<double>();
    check_lagrange_identity_mask<float>();
    return pantr::test::summary("test_bspline_extraction");
}
