/// \file
/// Knot insertion and the two-scale refinement matrix.
///
/// ## The two oracles, and why neither is a rerun of the code
///
/// A test that recomputes the Oslo recurrence and compares is a mirror, not a test.
/// Both checks here are closed forms established outside this file.
///
/// **The cardinal refinement identity.** On a uniform knot vector, halving every span
/// gives the classical two-scale relation of the cardinal B-spline,
///
///     B_i^{(m)} = 2^{-p} * sum_{j=0}^{p+1} C(p+1, j) B_{2i+j}^{(m+1)},
///
/// so an interior column of the matrix is the binomial stencil `C(p+1, j) / 2^p` and
/// nothing else. It is a statement about binomial coefficients, and the recurrence
/// under test knows nothing about it. It holds only where the vector really is
/// uniform, so the columns touched by the clamped ends are excluded -- and the check
/// counts how many columns it verified and fails if that count is zero, because a
/// filter that silently matched nothing is the way this kind of test goes vacuous.
/// Measured against the Python oracle at degrees 1 to 4: the agreement is **exact**,
/// deviation `0.0`, which is why this asserts equality rather than a bound. That is
/// not a general claim about the recurrence; it is that this stencil's values are
/// dyadic and every operation reaching them is exact.
///
/// **Partition of unity survives refinement.** `sum_i B_i^{(m)} = 1` and
/// `sum_j B_j^{(m+1)} = 1` together force every *row* of the matrix to sum to one,
/// for any knot vector and any refinement. It is not dyadic, so it is graded against
/// a bound rather than asserted exactly: `p + 2` roundings act on the row's `p + 1`
/// non-zero entries, each in `[0, 1]` and summing to one, so the accumulated error is
/// at most `gamma_{p+2} = (p+2) u / (1 - (p+2) u)` and the constant carries no
/// problem-size growth. `2 * gamma_{p+2}` is used, the factor of two covering the
/// entries' own formation, and it is a stated pad rather than a derivation.
///
/// ## What the refusals pin
///
/// `inserted_knot_vector`'s two refusals are the ones a THB space can actually reach:
/// a subdivision that would push a knot past multiplicity `degree + 1` is what
/// `regularity = -1` does at degree 1, and it must be a message rather than a
/// silently degenerate space. The domain refusal is unreachable from `subdivide`,
/// whose knots are interior by construction, so it is exercised directly.
///
/// ## A factor of one is refused, not silently a copy
///
/// `HierarchicalGrid`'s per-direction factor may be 1, and the THB space skips those
/// axes rather than asking for a subdivision by one -- so the bound here is the oracle's,
/// `n_subdivisions >= 2`, and `check_refusals` pins both entry points' messages. An
/// earlier version accepted 1 and returned a copy; nothing exercised it, and a widened
/// contract with no exerciser is how an unexamined precedent starts.

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/knot_insertion.hpp"

namespace {

using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;
using pantr::bspline::inserted_knot_vector;
using pantr::bspline::oslo_bands_1d;
using pantr::bspline::oslo_matrix_1d;
using pantr::bspline::subdivide;
using pantr::bspline::uniform_subdivision_knots;

/// A clamped uniform knot vector on `[0, 1]`.
///
/// \param degree The polynomial degree.
/// \param num_elements The number of equal spans.
/// \return `degree` copies of 0, the `num_elements + 1` breakpoints, `degree` copies
///         of 1 -- the open vector with an extra copy at each end supplied by the
///         breakpoints themselves.
std::vector<double> uniform_knots(std::int64_t degree, std::int64_t num_elements) {
    std::vector<double> knots;
    for (std::int64_t i = 0; i < degree; ++i) {
        knots.push_back(0.0);
    }
    for (std::int64_t i = 0; i <= num_elements; ++i) {
        knots.push_back(static_cast<double>(i) / static_cast<double>(num_elements));
    }
    for (std::int64_t i = 0; i < degree; ++i) {
        knots.push_back(1.0);
    }
    return knots;
}

/// A space over `uniform_knots`.
///
/// \param degree The polynomial degree.
/// \param num_elements The number of equal spans.
/// \return The space, snapping near-duplicates as the oracle's default does.
BsplineSpace1D<double> uniform_space(std::int64_t degree, std::int64_t num_elements) {
    const std::vector<double> knots = uniform_knots(degree, num_elements);
    return BsplineSpace1D<double>(std::span<const double>(knots), degree, false,
                                  KnotSnapping::merge_near_duplicates);
}

/// The binomial coefficient `C(n, k)`, by the multiplicative form.
///
/// Exact in `std::int64_t` over the degrees this file reaches, and formed here rather
/// than tabulated so the stencil the test compares against is visibly a binomial.
///
/// \param n The upper index, non-negative.
/// \param k The lower index, in `[0, n]`.
/// \return `C(n, k)`.
std::int64_t binomial(std::int64_t n, std::int64_t k) {
    std::int64_t result = 1;
    for (std::int64_t i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

/// `gamma_m = m u / (1 - m u)`, the standard accumulation constant.
///
/// \param m The number of roundings.
/// \return The constant, in units of the value being bounded.
double gamma_of(std::int64_t m) {
    const double u = 0.5 * std::numeric_limits<double>::epsilon();
    const double mu = static_cast<double>(m) * u;
    return mu / (1.0 - mu);
}

/// The two-scale matrix of a dyadic refinement is the binomial stencil, exactly.
///
/// See the file comment for the identity and for why the check counts what it
/// verified.
void check_the_cardinal_refinement_identity() {
    for (std::int64_t degree = 1; degree <= 4; ++degree) {
        const BsplineSpace1D<double> coarse = uniform_space(degree, 8);
        const BsplineSpace1D<double> fine = subdivide<double>(coarse, 2, std::nullopt);

        const std::int64_t num_cols = coarse.num_basis();
        const std::int64_t num_rows = fine.num_basis();
        const std::vector<double> matrix =
            oslo_matrix_1d<double>(degree, coarse.knots(), fine.knots());
        PANTR_CHECK(static_cast<std::int64_t>(matrix.size()) == num_rows * num_cols);

        std::vector<double> stencil(static_cast<std::size_t>(degree) + 2, 0.0);
        for (std::int64_t j = 0; j <= degree + 1; ++j) {
            stencil[static_cast<std::size_t>(j)] =
                static_cast<double>(binomial(degree + 1, j)) / std::exp2(degree);
        }

        std::int64_t verified = 0;
        for (std::int64_t col = 0; col < num_cols; ++col) {
            std::vector<std::int64_t> support;
            for (std::int64_t row = 0; row < num_rows; ++row) {
                if (matrix[static_cast<std::size_t>(row * num_cols + col)] != 0.0) {
                    support.push_back(row);
                }
            }
            // An interior column has exactly `degree + 2` contiguous non-zeros. The
            // columns the clamped ends touch have fewer, or the same count over a
            // different stencil, and the identity does not describe them.
            if (static_cast<std::int64_t>(support.size()) != degree + 2
                || support.back() - support.front() != degree + 1) {
                continue;
            }
            bool matched = true;
            for (std::int64_t j = 0; j <= degree + 1; ++j) {
                const double value = matrix[static_cast<std::size_t>(
                    (support.front() + j) * num_cols + col)];
                matched = matched && value == stencil[static_cast<std::size_t>(j)];
            }
            if (matched) {
                ++verified;
            } else {
                PANTR_CHECK_MSG(false, "an interior column at degree " + std::to_string(degree)
                                           + " is not the binomial stencil");
            }
        }
        PANTR_CHECK_MSG(verified > 0, "the vacuity guard: at degree " + std::to_string(degree)
                                          + " the filter matched no interior column, so the "
                                            "identity was never actually compared");
    }
}

/// Every row of a two-scale matrix sums to one, on any refinement.
///
/// The bound is `2 * gamma_{degree + 2}`; see the file comment for the count and for
/// what the factor of two is and is not.
void check_refinement_preserves_the_partition_of_unity() {
    struct Case {
        std::int64_t degree;
        std::vector<double> knots;
        std::int64_t subdivisions;
        std::int64_t regularity;
    };
    // Non-uniform spans, a non-dyadic factor and a reduced regularity are each
    // present, because on a uniform dyadic refinement the entries are dyadic and the
    // sum is exact -- which would make the bound untested.
    const std::vector<Case> cases = {
        {2, {0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0}, 2, 1},
        {2, {0.0, 0.0, 0.0, 0.1, 0.37, 0.62, 1.0, 1.0, 1.0}, 3, 1},
        {3, {0.0, 0.0, 0.0, 0.0, 0.1, 0.37, 0.62, 1.0, 1.0, 1.0, 1.0}, 2, 0},
        {3, {1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6 + 0.1, 1.0e6 + 0.37, 1.0e6 + 0.62, 1.0e6 + 1.0,
             1.0e6 + 1.0, 1.0e6 + 1.0, 1.0e6 + 1.0},
         2,
         2},
    };

    for (const Case& c : cases) {
        const BsplineSpace1D<double> coarse(std::span<const double>(c.knots), c.degree, false,
                                            KnotSnapping::merge_near_duplicates);
        const BsplineSpace1D<double> fine = subdivide<double>(coarse, c.subdivisions, c.regularity);
        const std::int64_t num_cols = coarse.num_basis();
        const std::int64_t num_rows = fine.num_basis();
        const std::vector<double> matrix =
            oslo_matrix_1d<double>(c.degree, coarse.knots(), fine.knots());

        const double bound = 2.0 * gamma_of(c.degree + 2);
        double worst = 0.0;
        for (std::int64_t row = 0; row < num_rows; ++row) {
            double sum = 0.0;
            for (std::int64_t col = 0; col < num_cols; ++col) {
                sum += matrix[static_cast<std::size_t>(row * num_cols + col)];
            }
            worst = std::max(worst, std::abs(sum - 1.0));
        }
        PANTR_CHECK_MSG(worst <= bound, "a row of the degree-" + std::to_string(c.degree)
                                            + " two-scale matrix does not sum to one");
    }
}

/// The dense matrix is the bands scattered, and the bands are where the values are.
void check_the_dense_matrix_is_the_bands_scattered() {
    const BsplineSpace1D<double> coarse = uniform_space(3, 5);
    const BsplineSpace1D<double> fine = subdivide<double>(coarse, 2, std::nullopt);
    const auto bands = oslo_bands_1d<double>(3, coarse.knots(), fine.knots());
    const std::vector<double> dense = oslo_matrix_1d<double>(3, coarse.knots(), fine.knots());

    const std::int64_t num_cols = coarse.num_basis();
    PANTR_CHECK(bands.num_rows == fine.num_basis());
    PANTR_CHECK(bands.width == 4);

    std::int64_t placed = 0;
    for (std::int64_t row = 0; row < bands.num_rows; ++row) {
        for (std::int64_t l = 0; l < bands.width; ++l) {
            const std::int64_t col = bands.first_col[static_cast<std::size_t>(row)] + l;
            if (col < 0 || col >= num_cols) {
                continue;
            }
            ++placed;
            PANTR_CHECK(dense[static_cast<std::size_t>(row * num_cols + col)]
                        == bands.alphas[static_cast<std::size_t>(row * bands.width + l)]);
        }
    }
    // Everything the dense form holds came from a band: no entry outside one is
    // non-zero, so the two really do describe the same matrix.
    std::int64_t non_zero = 0;
    for (const double value : dense) {
        non_zero += static_cast<std::int64_t>(value != 0.0);
    }
    PANTR_CHECK_MSG(non_zero <= placed, "the dense form holds a value no band placed");
}

/// The subdivided knot vector: the values, the multiplicities and the counts.
void check_subdivision_knots() {
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const BsplineSpace1D<double> space(std::span<const double>(knots), 2, false,
                                       KnotSnapping::merge_near_duplicates);

    // Maximal smoothness: one copy of each interior point of each span.
    const std::vector<double> maximal =
        uniform_subdivision_knots<double>(space.knots(), 2, space.tolerance(), 2, 1);
    PANTR_CHECK(maximal.size() == 2);
    PANTR_CHECK(maximal[0] == 0.25);
    PANTR_CHECK(maximal[1] == 0.75);

    // `C^0`: multiplicity `degree - 0 = 2` at each inserted point.
    const std::vector<double> c_zero =
        uniform_subdivision_knots<double>(space.knots(), 2, space.tolerance(), 2, 0);
    PANTR_CHECK(c_zero.size() == 4);
    PANTR_CHECK(c_zero[0] == 0.25 && c_zero[1] == 0.25);
    PANTR_CHECK(c_zero[2] == 0.75 && c_zero[3] == 0.75);

    // A factor of three puts two interior points in each span.
    const std::vector<double> thirds =
        uniform_subdivision_knots<double>(space.knots(), 2, space.tolerance(), 3, 1);
    PANTR_CHECK(thirds.size() == 4);

    // A factor of one is refused rather than treated as a copy; `check_refusals` pins
    // both messages. See the file comment.
}

/// Subdividing multiplies the interval count and keeps the domain.
void check_subdivide_counts_and_domain() {
    for (std::int64_t degree = 1; degree <= 3; ++degree) {
        const BsplineSpace1D<double> coarse = uniform_space(degree, 3);
        for (std::int64_t factor = 2; factor <= 4; ++factor) {
            const BsplineSpace1D<double> fine = subdivide<double>(coarse, factor, std::nullopt);
            PANTR_CHECK(fine.num_intervals() == coarse.num_intervals() * factor);
            PANTR_CHECK(fine.degree() == degree);
            // Maximal smoothness inserts one knot per new interior breakpoint, so the
            // basis grows by exactly the number of inserted knots.
            PANTR_CHECK(fine.num_basis()
                        == coarse.num_basis() + coarse.num_intervals() * (factor - 1));
            PANTR_CHECK(fine.domain()[0] == coarse.domain()[0]);
            PANTR_CHECK(fine.domain()[1] == coarse.domain()[1]);
        }
    }
}

/// `float` storage computes in `double` and stores in `float`.
///
/// The two spellings are not the same: `float(double(j) * ((double(hi) - double(lo)) /
/// n) + double(lo))` and `float(j) * ((hi - lo) / float(n)) + lo` differ on the span
/// `[1/3, 1]` at `n = 2`, by one `float` ulp. The check is that the stored value is
/// the narrowing of the `double` computation, which is what the oracle does and what
/// `design/cross_backend_types.md` requires -- and the guard below fails if the case
/// chosen stops distinguishing them, since then this would pass over a narrowing that
/// never happened.
void check_float_storage_computes_in_double() {
    const float split = 1.0F / 3.0F;
    const std::vector<float> knots = {0.0F, 0.0F, 0.0F, split, 1.0F, 1.0F, 1.0F};
    const BsplineSpace1D<float> space(std::span<const float>(knots), 2, false,
                                      KnotSnapping::merge_near_duplicates);
    const std::vector<float> inserted =
        uniform_subdivision_knots<float>(space.knots(), 2, space.tolerance(), 2, 1);
    PANTR_CHECK(inserted.size() == 2);

    const double lo = static_cast<double>(split);
    const double step_low = (lo - 0.0) / 2.0;
    const double step_high = (1.0 - lo) / 2.0;
    const float in_double_low = static_cast<float>(1.0 * step_low + 0.0);
    const float in_double_high = static_cast<float>(1.0 * step_high + lo);
    PANTR_CHECK(inserted[0] == in_double_low);
    PANTR_CHECK(inserted[1] == in_double_high);

    const float in_float_high = 1.0F * ((1.0F - split) / 2.0F) + split;
    PANTR_CHECK_MSG(in_double_high != in_float_high,
                    "the vacuity guard: this span no longer distinguishes double-then-narrow "
                    "from float arithmetic, so the case cannot fail on the thing it names");
}

/// The merge is order-preserving on equal values, so a signed zero keeps its place.
///
/// The domain has `-0.0` as an interior knot and a `+0.0` is inserted onto it, so the
/// two are one class of multiplicity two and the degree admits it. Asserting the
/// *sign* of the tie is what `CLAUDE.md` forbids -- the two compare equal and no
/// implementation owes an order -- so what is asserted is the ORDER the concatenation
/// fixed, which stability is what preserves.
void check_the_merge_is_stable() {
    const std::vector<double> knots = {-1.0, -1.0, -1.0, -0.0, 1.0, 1.0, 1.0};
    const std::vector<double> insert = {0.0};
    const BsplineSpace1D<double> space(std::span<const double>(knots), 2, false,
                                       KnotSnapping::merge_near_duplicates);
    const std::vector<double> merged =
        inserted_knot_vector<double>(space.knots(), 2, std::span<const double>(insert),
                                     space.tolerance());
    PANTR_CHECK(merged.size() == 8);
    PANTR_CHECK(merged[2] == -1.0);
    PANTR_CHECK_MSG(std::signbit(merged[3]),
                    "the knot vector's own -0.0 came first in the concatenation and must stay "
                    "first");
    PANTR_CHECK_MSG(!std::signbit(merged[4]), "and the inserted +0.0 must follow it");
    PANTR_CHECK(merged[5] == 1.0);
}

/// What the two entry points refuse, and with which message.
void check_refusals() {
    const BsplineSpace1D<double> space = uniform_space(2, 4);

    bool threw = false;
    try {
        static_cast<void>(subdivide<double>(space, 1, std::nullopt));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "n_subdivisions must be >= 2, got 1";
    }
    PANTR_CHECK_MSG(threw, "a subdivision factor below the oracle's bound must be refused "
                           "with the oracle's message");

    threw = false;
    try {
        static_cast<void>(
            uniform_subdivision_knots<double>(space.knots(), 2, space.tolerance(), 1, 1));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "n_subdivisions must be >= 2, got 1";
    }
    PANTR_CHECK_MSG(threw, "and the knot generator refuses it too, not only its caller");

    threw = false;
    try {
        static_cast<void>(subdivide<double>(space, 2, 2));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "regularity must be in [-1, degree - 1] = [-1, 1], got 2";
    }
    PANTR_CHECK_MSG(threw, "a regularity at or above the degree must be refused");

    threw = false;
    try {
        const std::vector<double> nothing;
        static_cast<void>(inserted_knot_vector<double>(space.knots(), 2,
                                                       std::span<const double>(nothing),
                                                       space.tolerance()));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "new_knots_to_insert must not be empty.";
    }
    PANTR_CHECK_MSG(threw, "an empty insertion must be refused");

    threw = false;
    try {
        const std::vector<double> outside = {1.5};
        static_cast<void>(inserted_knot_vector<double>(space.knots(), 2,
                                                       std::span<const double>(outside),
                                                       space.tolerance()));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).starts_with(
            "new_knots contains values outside the domain [0.0, 1.0]:");
    }
    PANTR_CHECK_MSG(threw, "a knot outside the domain must be refused, naming the domain");

    // Multiplicity: at degree 2 a knot may appear three times. The vector already
    // holds 0.5 once, so inserting it three more times is one too many.
    threw = false;
    try {
        const std::vector<double> too_many = {0.5, 0.5, 0.5};
        static_cast<void>(inserted_knot_vector<double>(space.knots(), 2,
                                                       std::span<const double>(too_many),
                                                       space.tolerance()));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what())
                == "Inserting these knots would exceed the maximum multiplicity of 3. "
                   "Maximum multiplicity found: 4.";
    }
    PANTR_CHECK_MSG(threw, "exceeding the maximum multiplicity must be refused by its own message");

    threw = false;
    try {
        const std::vector<double> knots = {0.0, 1.0};
        static_cast<void>(oslo_bands_1d<double>(-1, std::span<const double>(knots),
                                                std::span<const double>(knots)));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a negative degree must be refused");
}

/// Reduced regularity really reduces the smoothness, and the counts say so.
void check_regularity_changes_the_basis_count() {
    const BsplineSpace1D<double> coarse = uniform_space(3, 2);
    const BsplineSpace1D<double> smooth = subdivide<double>(coarse, 2, std::nullopt);
    const BsplineSpace1D<double> c_one = subdivide<double>(coarse, 2, 1);
    const BsplineSpace1D<double> c_minus = subdivide<double>(coarse, 2, -1);

    // Two new interior breakpoints, at multiplicity `degree - regularity`.
    PANTR_CHECK(smooth.num_basis() == coarse.num_basis() + 2 * 1);
    PANTR_CHECK(c_one.num_basis() == coarse.num_basis() + 2 * 2);
    PANTR_CHECK(c_minus.num_basis() == coarse.num_basis() + 2 * 4);
    // Every one of them still has the same interval count: multiplicity does not
    // create spans.
    PANTR_CHECK(smooth.num_intervals() == 4 && c_one.num_intervals() == 4
                && c_minus.num_intervals() == 4);
}

}  // namespace

int main() {
    check_the_cardinal_refinement_identity();
    check_refinement_preserves_the_partition_of_unity();
    check_the_dense_matrix_is_the_bands_scattered();
    check_subdivision_knots();
    check_subdivide_counts_and_domain();
    check_float_storage_computes_in_double();
    check_the_merge_is_stable();
    check_refusals();
    check_regularity_changes_the_basis_count();
    return pantr::test::summary("test_bspline_knot_insertion");
}
