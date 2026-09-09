/// \file
/// Refining a B-spline field: `insert_knots`, `subdivide`, and the axis sweep.
///
/// ## The independent oracle, and why it is not a rerun of the code
///
/// The obvious invariant -- *inserting knots does not move the curve* -- is checked by
/// evaluating before and after, and evaluation needs basis tabulation, which
/// `pantr/bspline/space_1d.hpp` keeps off this milestone. So the oracle here is the
/// same statement written in **coefficients** instead of in values, which needs no
/// basis at all.
///
/// **Marsden's identity.** For a knot vector `t` and degree `p`,
///
///     (u - y)^p = sum_i prod_{k=1}^{p} (t_{i+k} - y) * N_{i,p}(u),
///
/// and expanding both sides in `y` and matching the coefficient of `(-y)^{p-r}` gives,
/// for every `r` in `[0, p]`,
///
///     u^r = sum_i [ e_r(t_{i+1}, ..., t_{i+p}) / C(p, r) ] * N_{i,p}(u),
///
/// with `e_r` the elementary symmetric polynomial. So the B-spline coefficients of the
/// monomial `u^r` are a closed form in the knots. A field whose control points are
/// those numbers *is* the map `u -> u^r`; refinement must not move it; therefore the
/// refined coefficients must be the same closed form evaluated on the **refined** knot
/// vector.
///
/// That is a complete oracle rather than a partial one, and the difference matters.
/// `r = 1` alone is the Greville abscissa and says the refinement is affine-invariant;
/// `r = 0` alone is the partition of unity. Either on its own leaves a row of `p + 1`
/// entries pinned by one linear functional, so a wrong matrix that happens to preserve
/// affine maps would pass. Taking **all** `r` in `[0, p]` gives `p + 1` independent
/// functionals per row, which determines the row completely: the Vandermonde-like
/// system `sum_l alpha_l * g_l^{(r)} = ghat^{(r)}`, `r = 0..p`, has the monomial
/// moments of `p + 1` distinct knot windows as its matrix.
///
/// **That the system is nonsingular is not proved here, and the qualifier matters.**
/// It is *false* in general: a window over a collapsed (zero-width) span makes the
/// matrix singular. It holds for every window a refinement can actually produce, which
/// is Curry-Schoenberg local polynomial reproduction (Schoenberg's B-spline basis
/// reproduces polynomials of degree <= p exactly on each nonempty span). Checked in
/// exact rational arithmetic over the reachable bands rather than assumed, and the
/// unreachable singular windows were found and confirmed unreachable -- **observed
/// with an argument, not proved in this file.**
///
/// Nothing in it consults the Oslo recurrence: elementary symmetric polynomials of
/// knots and a binomial coefficient, formed in this file.
///
/// **In several directions at once it stays a closed form.** The refinement is a
/// tensor product of the per-direction matrices, so the field
/// `prod_d u_d^{r_d}` has control points `prod_d A^d_{i_d, r_d}`, and each component
/// of the net below carries one power multi-index. Refining direction `d` must move
/// that direction's factor to its refined value and leave every other factor alone --
/// which is the check that catches an axis sweep applied to the wrong axis, a
/// transposed stride, or a component axis mistaken for a parametric one.
///
/// ## Where the oracle does not hold, and the vacuity guard that says so
///
/// On a **clamped** vector every row's band lies inside `[0, num_cols)`: the leading
/// knots are the left endpoint repeated `p + 1` times, so `mu >= p` for every refined
/// knot and `first_col >= 0`. On an **unclamped** one the leading rows address columns
/// that do not exist, their bands are truncated, and their coefficients are not the
/// closed form. Such a vector is legal -- `BsplineSpace1D` accepts it with
/// `periodic = false` -- so it is exercised, with the identity asserted only on the
/// rows whose band is whole and a **count of what was verified** that fails if the
/// filter matched nothing. That is the same discipline
/// `test_bspline_knot_insertion.cpp` applies to its binomial-stencil columns, and for
/// the same reason: a filter that silently matched nothing is how this kind of test
/// goes vacuous.
///
/// ## The bound, and why it is not `1e-12`
///
/// Every quantity compared is a relative one on non-negative data, so the bound is
/// `gamma_K = K u / (1 - K u)` times the magnitude the value is built from, with `K`
/// counted rather than fitted:
///
///  - **The oracle's own formation**: `e_r` by the recurrence `E_j <- E_j + x E_{j-1}`
///    commits two roundings per knot along the dominant chain, then one division by
///    `C(p, r)`. `2p + 1`.
///  - **The cast into the field's storage.** `marsden_field` forms the closed form in
///    `double` and stores it in `T`, which at `float32` is a real rounding on the
///    *input* the sweep then propagates. It costs one, because the sweep's convex
///    combination does not amplify a relative perturbation. `1`.
///  - **One direction's band recurrence**: `p` levels, and each level's dominant path
///    is two subtractions (`t[j+k] - x` and the denominator), a division, a
///    multiplication and an addition. `5p`.
///  - **One direction's control-point sweep**: `p + 1` terms, each a multiplication
///    and an addition. `2p + 2`.
///  - Relative errors compose sub-additively and `gamma_a + gamma_b <= gamma_{a+b}`,
///    so `D` refined directions cost `D (7p + 2)`.
///
///  - **The store into the result**: the sweep accumulates in `double` and writes the
///    coefficient back into `T`, which at `float32` is a real rounding on the *output*,
///    symmetric to the input cast above and derived the same way. `1`.
///
/// `K = 2p + 3 + D (7p + 2)`, so the `2p + 3` is the oracle's `e_r` recurrence plus one
/// rounding on each end -- the cast in and the store out. Every term above names the
/// operations it charges; none is a pad. The
/// magnitude it multiplies is `prod_d max_i A^d_{i, r_d}`, per component: the discrete
/// B-splines of a refinement are non-negative, so each output coefficient is a convex
/// combination of coarse ones and no partial sum exceeds the largest of them. That
/// non-negativity is a classical property (Cohen, Lyche & Riesenfeld 1980) and it is
/// **asserted here rather than assumed**, because it is the premise the whole bound
/// rests on and an assumed premise is exactly the kind of claim nothing else in the
/// suite would notice.
///
/// A magnitude can be an exact zero -- `e_p` of a window containing the knot 0 -- and a
/// relative bound of zero asserts bit-identity nothing here has grounds for, so `K`
/// underflow floors are added. That is the absolute half of Higham's model, and it is
/// what lets `accuracy_bound` return a strictly positive number for every entry
/// instead of the comparison carrying a special case for zero.
///
/// The counts are charged at `u(T)` throughout, including the oracle's, which is
/// formed in `double` whatever `T` is. That over-states the `float` case, where
/// `double` roundings are 2^29 times smaller, and it keeps one derivation for both.
///
/// A dyadic refinement of a dyadic vector rounds nothing, so the bound would go
/// untested on halvings alone. Every group below carries a non-dyadic case -- a factor
/// of three, and knot vectors with thirds and sevenths in them.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/control_net.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/refinement.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"

namespace {

using pantr::bspline::Bspline;
using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;
using pantr::bspline::insert_knots;
using pantr::bspline::oslo_bands_1d;
using pantr::bspline::oslo_matrix_1d;
using pantr::bspline::refine_along_axis;
using pantr::bspline::subdivide;

/// `gamma_m = m u / (1 - m u)`, the standard accumulation constant.
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

/// The binomial coefficient `C(n, k)`, by the multiplicative form.
///
/// \param n The upper index, non-negative.
/// \param k The lower index, in `[0, n]`.
/// \return `C(n, k)`, exact in `std::int64_t` over the degrees this file reaches.
std::int64_t binomial(std::int64_t n, std::int64_t k) {
    std::int64_t result = 1;
    for (std::int64_t i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

/// The Marsden coefficients of `u^r` in one direction's B-spline basis.
///
/// `A_i = e_r(t_{i+1}, ..., t_{i+degree}) / C(degree, r)`; see the file comment for the
/// identity. Formed in `double` whatever the knots are stored in, by the elementary
/// symmetric recurrence, so nothing here re-runs the code under test.
///
/// \tparam T The scalar type the knots are stored in.
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \param r The monomial power, in `[0, degree]`.
/// \return One coefficient per basis function, `knots.size() - degree - 1` of them.
template <class T>
std::vector<double> marsden_coefficients(std::span<const T> knots, std::int64_t degree,
                                         std::int64_t r) {
    const std::int64_t count = static_cast<std::int64_t>(knots.size()) - degree - 1;
    std::vector<double> out(static_cast<std::size_t>(count), 0.0);
    for (std::int64_t i = 0; i < count; ++i) {
        // e[j] over the window `knots[i+1 .. i+degree]`, built one knot at a time.
        std::vector<double> e(static_cast<std::size_t>(degree) + 1, 0.0);
        e[0] = 1.0;
        for (std::int64_t k = 1; k <= degree; ++k) {
            const double x = static_cast<double>(knots[static_cast<std::size_t>(i + k)]);
            for (std::int64_t j = k; j >= 1; --j) {
                e[static_cast<std::size_t>(j)] += x * e[static_cast<std::size_t>(j - 1)];
            }
        }
        out[static_cast<std::size_t>(i)] =
            e[static_cast<std::size_t>(r)] / static_cast<double>(binomial(degree, r));
    }
    return out;
}

/// One field to refine, its expected refinement, and the bound between them.
///
/// The net's component axis enumerates power multi-indices row-major: component `c`
/// decodes to `(r_0, ..., r_{dim-1})` with `r_d` in `[0, degree_d]`, and holds
/// `prod_d A^d_{i_d, r_d}`. So one net carries every monomial the degrees admit, and
/// one refinement checks all of them at once.
///
/// \tparam T The scalar type the field stores.
struct Expectation {
    /// The net values, row-major under `(*num_basis, num_components)`.
    std::vector<double> values;
    /// The elementwise magnitude the bound is proportional to.
    std::vector<double> magnitude;
    /// The shape, `(*num_basis, num_components)`.
    std::vector<std::size_t> shape;
};

/// The Marsden net over a set of directions, with its per-element magnitude.
///
/// \tparam T The scalar type the knots are stored in.
/// \param directions One knot vector and degree per parametric direction.
/// \return The net, its shape and the magnitude each entry is a convex combination of.
template <class T>
Expectation marsden_net(const std::vector<const BsplineSpace1D<T>*>& directions) {
    const std::size_t dim = directions.size();
    std::vector<std::size_t> counts(dim);
    std::vector<std::size_t> powers(dim);
    // Per direction and per power, the coefficients and their largest magnitude.
    std::vector<std::vector<std::vector<double>>> tables(dim);
    std::vector<std::vector<double>> largest(dim);
    for (std::size_t d = 0; d < dim; ++d) {
        const BsplineSpace1D<T>& space = *directions[d];
        counts[d] = static_cast<std::size_t>(space.num_basis());
        powers[d] = static_cast<std::size_t>(space.degree()) + 1;
        for (std::int64_t r = 0; r <= space.degree(); ++r) {
            std::vector<double> column = marsden_coefficients<T>(space.knots(), space.degree(), r);
            double biggest = 0.0;
            for (const double value : column) {
                biggest = std::max(biggest, std::abs(value));
            }
            tables[d].push_back(std::move(column));
            largest[d].push_back(biggest);
        }
    }

    std::size_t num_components = 1;
    for (const std::size_t p : powers) {
        num_components *= p;
    }

    Expectation out;
    out.shape.assign(counts.begin(), counts.end());
    out.shape.push_back(num_components);
    std::size_t total = num_components;
    for (const std::size_t count : counts) {
        total *= count;
    }
    out.values.assign(total, 0.0);
    out.magnitude.assign(total, 0.0);

    std::vector<std::size_t> index(dim, 0);
    std::size_t flat = 0;
    while (true) {
        for (std::size_t c = 0; c < num_components; ++c) {
            std::size_t rest = c;
            double value = 1.0;
            double bound = 1.0;
            for (std::size_t d = dim; d-- > 0;) {
                const std::size_t r = rest % powers[d];
                rest /= powers[d];
                value *= tables[d][r][index[d]];
                bound *= largest[d][r];
            }
            out.values[flat * num_components + c] = value;
            out.magnitude[flat * num_components + c] = bound;
        }
        ++flat;
        // Odometer over the parametric multi-index, last axis fastest, which is the
        // net's own row-major order.
        std::size_t axis = dim;
        while (axis-- > 0) {
            if (++index[axis] < counts[axis]) {
                break;
            }
            index[axis] = 0;
            if (axis == 0) {
                return out;
            }
        }
    }
}

/// A space over the given knot vectors and degrees, snapping as the oracle's default.
///
/// \tparam T The scalar type the knots are stored in.
/// \param knots One knot vector per direction.
/// \param degrees One degree per direction.
/// \return The handles, in axis order.
template <class T>
std::vector<std::shared_ptr<const BsplineSpace1D<T>>>
make_directions(const std::vector<std::vector<T>>& knots,
                const std::vector<std::int64_t>& degrees) {
    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> out;
    for (std::size_t d = 0; d < knots.size(); ++d) {
        out.push_back(std::make_shared<const BsplineSpace1D<T>>(
            std::span<const T>(knots[d]), degrees[d], false, KnotSnapping::merge_near_duplicates));
    }
    return out;
}

/// Borrow the directions of a space, for `marsden_net`.
///
/// \tparam T The scalar type.
/// \param space The space.
/// \return One pointer per direction, in axis order.
template <class T>
std::vector<const BsplineSpace1D<T>*> borrow(const BsplineSpace<T>& space) {
    std::vector<const BsplineSpace1D<T>*> out;
    for (std::int64_t d = 0; d < space.dim(); ++d) {
        out.push_back(&space.space_ref(d));
    }
    return out;
}

/// The Marsden field over a space, at storage format `T`.
///
/// \tparam T The scalar type the field stores.
/// \param space The space, shared into the field.
/// \param is_rational Whether the last component is declared a homogeneous weight.
/// \return The field.
template <class T>
Bspline<T> marsden_field(const std::shared_ptr<const BsplineSpace<T>>& space, bool is_rational) {
    const Expectation expected = marsden_net<T>(borrow(*space));
    std::vector<T> stored(expected.values.size());
    for (std::size_t i = 0; i < stored.size(); ++i) {
        stored[i] = static_cast<T>(expected.values[i]);
    }
    return Bspline<T>(space,
                      typename Bspline<T>::net_type(std::span<const T>(stored),
                                                    std::span<const std::size_t>(expected.shape)),
                      is_rational);
}

/// The number of roundings the accuracy bound charges; see the file comment.
///
/// \param degree The largest degree among the field's directions.
/// \param num_refined How many directions received knots, the `D` of the derivation.
/// \return `K`.
std::int64_t bound_roundings(std::int64_t degree, std::int64_t num_refined) {
    return 2 * degree + 3 + num_refined * (7 * degree + 2);
}

/// The elementwise accuracy bound between a net and Marsden's closed form.
///
/// `gamma_K` times the magnitude, plus `K` underflow floors. The floor is the absolute
/// half of Higham's model and it is not decoration here: a component whose closed form
/// is an exact zero has a relative bound of zero, and a bound of zero asserts
/// bit-identity that nothing in this file has grounds for.
///
/// \tparam T The scalar type the field stores.
/// \param magnitude The per-entry magnitude from `marsden_net`.
/// \param roundings `K`, from `bound_roundings`.
/// \return One bound per entry, all strictly positive.
template <class T>
std::vector<double> accuracy_bound(const std::vector<double>& magnitude,
                                   std::int64_t roundings) {
    const double relative = gamma_of<T>(roundings);
    const double floor = static_cast<double>(std::numeric_limits<T>::denorm_min());
    std::vector<double> bound(magnitude.size());
    for (std::size_t i = 0; i < magnitude.size(); ++i) {
        bound[i] = relative * magnitude[i] + static_cast<double>(roundings) * floor;
    }
    return bound;
}

/// Assert that a refined field's coefficients are Marsden's closed form again.
///
/// Every direction must be clamped, which is what makes the identity describe the whole
/// net; `check_unclamped` is where a truncated band is handled, row by row and with its
/// own vacuity guard. This took a `clamped` flag until a review found that every call
/// site passed `true` and the `false` branch asserted nothing at all, which is the
/// shape of dead test code that reads as coverage.
///
/// \tparam T The scalar type the field stores.
/// \param refined The refined field.
/// \param label What is being checked, for the failure message.
/// \param num_refined How many directions received knots, the `D` of the bound.
template <class T>
void check_marsden_survives(const Bspline<T>& refined, const std::string& label,
                            std::int64_t num_refined) {
    const BsplineSpace<T>& space = refined.space_ref();
    const Expectation expected = marsden_net<T>(borrow(space));
    const std::span<const T> got = refined.net().values();

    PANTR_CHECK_MSG(got.size() == expected.values.size(),
                    label + ": the refined net has " + std::to_string(got.size())
                        + " coefficients and the closed form has "
                        + std::to_string(expected.values.size()));
    if (got.size() != expected.values.size()) {
        return;
    }

    std::int64_t worst_degree = 0;
    for (const std::int64_t degree : space.degrees()) {
        worst_degree = std::max(worst_degree, degree);
    }
    const std::vector<double> bound = accuracy_bound<T>(
        expected.magnitude, bound_roundings(worst_degree, num_refined));

    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double difference = std::abs(static_cast<double>(got[i]) - expected.values[i]);
        worst_ratio = std::max(worst_ratio, difference / bound[i]);
    }
    PANTR_CHECK_MSG(worst_ratio <= 1.0,
                    label + ": Marsden's identity does not survive the refinement; worst "
                            "difference is "
                        + std::to_string(worst_ratio) + " times its derived bound");
}

/// The discrete B-splines of a refinement are non-negative and their rows sum to one.
///
/// The premise the accuracy bound rests on, asserted rather than assumed; see the file
/// comment. The row sum is graded against `2 * gamma_{p+2}`, the same constant
/// `test_bspline_knot_insertion.cpp` derives for it.
///
/// **The non-negativity half is an exact comparison on purpose, and a review asked
/// why.** `most_negative` starts at `0.0` and only ever passes through `std::min`,
/// which returns one of its arguments unaltered -- no arithmetic is performed on it, so
/// `== 0.0` is a sign test over the weights and not an equality on a computed
/// magnitude. A tolerance here would be the wrong bar in the strict sense: it would
/// accept a slightly negative weight, which is exactly the defect this forbids, and
/// negativity is a discrete property that finite precision does not blur. A `-0.0`
/// weight compares equal and so passes, which is correct -- it is not negative.
void check_the_bands_are_a_convex_combination() {
    struct Case {
        std::int64_t degree;
        std::vector<double> knots;
        std::int64_t subdivisions;
    };
    const std::vector<Case> cases = {
        {1, {0.0, 0.0, 0.5, 1.0, 1.0}, 3},
        {2, {0.0, 0.0, 0.0, 1.0 / 3.0, 1.0, 1.0, 1.0}, 3},
        {3, {0.0, 0.0, 0.0, 0.0, 0.25, 3.0 / 7.0, 1.0, 1.0, 1.0, 1.0}, 2},
    };
    for (const Case& c : cases) {
        const BsplineSpace1D<double> coarse(std::span<const double>(c.knots), c.degree, false,
                                            KnotSnapping::merge_near_duplicates);
        const BsplineSpace1D<double> fine =
            subdivide<double>(coarse, c.subdivisions, std::nullopt);
        const auto bands = oslo_bands_1d<double>(c.degree, coarse.knots(), fine.knots());
        const std::int64_t num_cols = coarse.num_basis();

        double most_negative = 0.0;
        double worst_row = 0.0;
        for (std::int64_t row = 0; row < bands.num_rows; ++row) {
            double sum = 0.0;
            for (std::int64_t l = 0; l < bands.width; ++l) {
                const std::int64_t col = bands.first_col[static_cast<std::size_t>(row)] + l;
                if (col < 0 || col >= num_cols) {
                    continue;
                }
                const double value =
                    bands.alphas[static_cast<std::size_t>(row * bands.width + l)];
                most_negative = std::min(most_negative, value);
                sum += value;
            }
            worst_row = std::max(worst_row, std::abs(sum - 1.0));
        }
        PANTR_CHECK_MSG(most_negative == 0.0,
                        "a discrete B-spline of a degree-" + std::to_string(c.degree)
                            + " refinement is negative, which is the premise the accuracy "
                              "bound in this file rests on");
        PANTR_CHECK_MSG(worst_row <= 2.0 * gamma_of<double>(c.degree + 2),
                        "a row of the degree-" + std::to_string(c.degree)
                            + " two-scale matrix does not sum to one");
    }
}

/// Marsden's identity survives `subdivide` in one direction, over degrees and factors.
template <class T>
void check_subdivide_1d(const std::string& format) {
    // Non-dyadic knots and a factor of three appear so the bound is exercised rather
    // than trivially satisfied; see the file comment.
    const std::vector<std::vector<T>> vectors = {
        {T(0), T(0), T(0.5), T(1), T(1)},
        {T(0), T(0), T(0), T(0.25), T(0.75), T(1), T(1), T(1)},
        {T(0), T(0), T(0), T(1.0 / 3.0), T(1), T(1), T(1)},
        {T(0), T(0), T(0), T(0), T(0.25), T(3.0 / 7.0), T(1), T(1), T(1), T(1)},
    };
    const std::vector<std::int64_t> degrees = {1, 2, 2, 3};

    for (std::size_t k = 0; k < vectors.size(); ++k) {
        for (const std::int64_t factor : {2, 3, 4}) {
            const auto directions = make_directions<T>({vectors[k]}, {degrees[k]});
            const auto space = std::make_shared<const BsplineSpace<T>>(directions);
            const Bspline<T> field = marsden_field<T>(space, false);
            const std::vector<std::int64_t> counts = {factor};
            const Bspline<T> refined =
                subdivide<T>(field, std::span<const std::int64_t>(counts), std::nullopt);

            const std::string label = format + " subdivide degree "
                                      + std::to_string(degrees[k]) + " by "
                                      + std::to_string(factor);
            PANTR_CHECK_MSG(refined.space_ref().space_ref(0).num_intervals()
                                == space->space_ref(0).num_intervals() * factor,
                            label + ": the interval count did not multiply");
            PANTR_CHECK_MSG(refined.rank() == field.rank(), label + ": the rank moved");
            PANTR_CHECK_MSG(refined.is_rational() == field.is_rational(),
                            label + ": the rationality flag moved");
            check_marsden_survives<T>(refined, label, 1);
        }
    }
}

/// Marsden's identity survives an explicit `insert_knots`, repeats included.
template <class T>
void check_insert_knots_1d(const std::string& format) {
    const std::vector<T> knots = {T(0), T(0), T(0), T(0.25), T(0.75), T(1), T(1), T(1)};
    const auto directions = make_directions<T>({knots}, {2});
    const auto space = std::make_shared<const BsplineSpace<T>>(directions);
    const Bspline<T> field = marsden_field<T>(space, false);

    // A repeat raises the multiplicity to two, which is legal at degree 2 and drops the
    // continuity to C^0 there; an existing knot gets a second copy; and 1/3 is not
    // representable, so the bound is what decides the comparison.
    const std::vector<T> to_insert = {T(1.0 / 3.0), T(1.0 / 3.0), T(0.25), T(0.9)};
    const std::vector<std::span<const T>> per_direction = {std::span<const T>(to_insert)};
    const Bspline<T> refined =
        insert_knots<T>(field, std::span<const std::span<const T>>(per_direction));

    const std::string label = format + " insert_knots with a repeat";
    PANTR_CHECK_MSG(refined.space_ref().space_ref(0).knots().size() == knots.size() + 4,
                    label + ": the refined knot vector is the wrong length");
    check_marsden_survives<T>(refined, label, 1);
}

/// Marsden's identity survives refinement of a surface and of a volume.
///
/// The multi-direction cases are asymmetric in the degree, the basis count and the
/// domain at once, so no permutation of the net's shape is another admissible shape and
/// a transposed sweep cannot pass.
template <class T>
void check_multidimensional(const std::string& format) {
    const std::vector<std::vector<T>> surface = {
        {T(0), T(0), T(0), T(1.0 / 3.0), T(1), T(1), T(1)},
        {T(10), T(10), T(11), T(12), T(12)},
    };
    const std::vector<std::int64_t> surface_degrees = {2, 1};

    {
        const auto space =
            std::make_shared<const BsplineSpace<T>>(make_directions<T>(surface, surface_degrees));
        const Bspline<T> field = marsden_field<T>(space, false);
        // Only the second direction is refined: the first has to come back untouched,
        // handle and all.
        const std::vector<std::int64_t> counts = {1, 3};
        const Bspline<T> refined =
            subdivide<T>(field, std::span<const std::int64_t>(counts), std::nullopt);
        PANTR_CHECK_MSG(refined.space_ref().space(0) == space->space(0),
                        format + " surface: an unrefined direction did not keep its space");
        PANTR_CHECK_MSG(refined.space_ref().space(1) != space->space(1),
                        format + " surface: the refined direction kept its space");
        check_marsden_survives<T>(refined, format + " surface, one direction", 1);
    }
    {
        const auto space =
            std::make_shared<const BsplineSpace<T>>(make_directions<T>(surface, surface_degrees));
        const Bspline<T> field = marsden_field<T>(space, false);
        const std::vector<std::int64_t> counts = {2, 3};
        const Bspline<T> refined =
            subdivide<T>(field, std::span<const std::int64_t>(counts), std::nullopt);
        check_marsden_survives<T>(refined, format + " surface, both directions", 2);
    }
    {
        const std::vector<std::vector<T>> volume = {
            {T(0), T(0), T(0), T(1.0 / 3.0), T(1), T(1), T(1)},
            {T(10), T(10), T(11), T(12), T(12)},
            {T(0), T(0), T(0), T(0), T(0.25), T(3.0 / 7.0), T(1), T(1), T(1), T(1)},
        };
        const auto space =
            std::make_shared<const BsplineSpace<T>>(make_directions<T>(volume, {2, 1, 3}));
        // Declared rational, so the last component is a homogeneous weight. Refinement
        // is component-blind and linear, so the closed form describes the weight column
        // exactly as it describes the coordinates -- which is the whole reason a
        // weighted net can be refined at all.
        const Bspline<T> field = marsden_field<T>(space, true);
        const std::vector<std::int64_t> counts = {2, 1, 3};
        const Bspline<T> refined =
            subdivide<T>(field, std::span<const std::int64_t>(counts), std::nullopt);
        PANTR_CHECK_MSG(refined.is_rational(), format + " volume: the weight flag was dropped");
        PANTR_CHECK_MSG(refined.rank() == field.rank(), format + " volume: the rank moved");
        check_marsden_survives<T>(refined, format + " rational volume", 2);
    }
}

/// `regularity` drives the multiplicity, and the identity survives every legal value.
void check_regularity() {
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.0, 0.25, 3.0 / 7.0, 1.0, 1.0, 1.0, 1.0};
    const auto space =
        std::make_shared<const BsplineSpace<double>>(make_directions<double>({knots}, {3}));
    const Bspline<double> field = marsden_field<double>(space, false);
    const std::vector<std::int64_t> counts = {2};

    for (const std::int64_t regularity : {-1, 0, 1, 2}) {
        const Bspline<double> refined =
            subdivide<double>(field, std::span<const std::int64_t>(counts), regularity);
        const std::int64_t multiplicity = 3 - regularity;
        // Three in-domain spans, so three new interior breakpoints, each at
        // multiplicity `degree - regularity`.
        PANTR_CHECK_MSG(refined.space_ref().space_ref(0).knots().size()
                            == knots.size() + static_cast<std::size_t>(3 * multiplicity),
                        "regularity " + std::to_string(regularity)
                            + ": the wrong number of knots was inserted");
        check_marsden_survives<double>(
            refined, "regularity " + std::to_string(regularity), 1);
    }
}

/// An unclamped, non-periodic direction: legal, refined, and only partly a closed form.
///
/// See the file comment for why the leading rows are not Marsden's numbers and why the
/// count of verified rows is asserted.
void check_unclamped() {
    // Strictly increasing and unclamped at both ends, so the leading and trailing bands
    // address columns outside the coarse space.
    const std::vector<double> knots = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto space =
        std::make_shared<const BsplineSpace<double>>(make_directions<double>({knots}, {2}));
    const Bspline<double> field = marsden_field<double>(space, false);
    const std::vector<std::int64_t> counts = {3};
    const Bspline<double> refined =
        subdivide<double>(field, std::span<const std::int64_t>(counts), std::nullopt);

    const BsplineSpace1D<double>& coarse = space->space_ref(0);
    const BsplineSpace1D<double>& fine = refined.space_ref().space_ref(0);
    const auto bands = oslo_bands_1d<double>(2, coarse.knots(), fine.knots());
    const Expectation expected = marsden_net<double>(borrow(refined.space_ref()));
    const std::span<const double> got = refined.net().values();
    const std::size_t num_components = expected.shape.back();

    const std::vector<double> bound =
        accuracy_bound<double>(expected.magnitude, bound_roundings(2, 1));
    std::int64_t verified = 0;
    std::int64_t truncated = 0;
    for (std::int64_t row = 0; row < bands.num_rows; ++row) {
        const std::int64_t base = bands.first_col[static_cast<std::size_t>(row)];
        if (base < 0 || base + bands.width > coarse.num_basis()) {
            ++truncated;
            continue;
        }
        ++verified;
        for (std::size_t c = 0; c < num_components; ++c) {
            const std::size_t flat = static_cast<std::size_t>(row) * num_components + c;
            PANTR_CHECK_MSG(std::abs(got[flat] - expected.values[flat]) <= bound[flat],
                            "unclamped: an interior row does not reproduce Marsden's "
                            "coefficient at row "
                                + std::to_string(row));
        }
    }
    PANTR_CHECK_MSG(verified > 0, "the vacuity guard: no whole band was found on the unclamped "
                                  "vector, so the identity was never compared");
    PANTR_CHECK_MSG(truncated > 0, "the unclamped case has no truncated band, so it is not "
                                   "exercising what it was written for");
}

/// The strided sweep addresses the axis it was asked for.
///
/// The refinement matrix is `oslo_matrix_1d`'s, so what is under test here is the index
/// arithmetic and nothing else: the reference is written as an explicit gather of each
/// fibre of a three-dimensional array, one dense matrix-vector product per fibre. A
/// transposed stride, an outer and inner swapped, or a component axis counted as
/// parametric all move it.
void check_the_axis_sweep_addresses_the_right_fibres() {
    const std::vector<double> knots = {0.0, 0.0, 0.0, 1.0 / 3.0, 1.0, 1.0, 1.0};
    const BsplineSpace1D<double> coarse(std::span<const double>(knots), 2, false,
                                        KnotSnapping::merge_near_duplicates);
    const BsplineSpace1D<double> fine = subdivide<double>(coarse, 3, std::nullopt);
    const auto bands = oslo_bands_1d<double>(2, coarse.knots(), fine.knots());
    const std::vector<double> dense = oslo_matrix_1d<double>(2, coarse.knots(), fine.knots());
    const std::int64_t num_cols = coarse.num_basis();
    const std::int64_t num_rows = fine.num_basis();

    for (const std::int64_t outer : {1, 2, 5}) {
        for (const std::int64_t inner : {1, 3, 4}) {
            std::vector<double> values(
                static_cast<std::size_t>(outer * num_cols * inner));
            for (std::size_t i = 0; i < values.size(); ++i) {
                // Distinct per position and spread over decades, so a swapped stride
                // cannot coincide with the right answer.
                values[i] = 1.0 + static_cast<double>(i) * 0.125;
            }
            const std::vector<double> got = refine_along_axis<double>(
                bands, num_cols, std::span<const double>(values), outer, inner);
            PANTR_CHECK(static_cast<std::int64_t>(got.size()) == outer * num_rows * inner);

            for (std::int64_t o = 0; o < outer; ++o) {
                for (std::int64_t k = 0; k < inner; ++k) {
                    for (std::int64_t row = 0; row < num_rows; ++row) {
                        double reference = 0.0;
                        for (std::int64_t col = 0; col < num_cols; ++col) {
                            reference +=
                                dense[static_cast<std::size_t>(row * num_cols + col)]
                                * values[static_cast<std::size_t>((o * num_cols + col) * inner
                                                                  + k)];
                        }
                        const double mine =
                            got[static_cast<std::size_t>((o * num_rows + row) * inner + k)];
                        // Equality, not a bound, and that is the whole point: the two
                        // visit the same non-zero terms in the same ascending order, and
                        // the reference's extra terms are `dense[row][col] * value` with
                        // `dense` an exact zero outside the band, so `s + 0.0` is a
                        // no-op in IEEE-754 -- the values here are all positive, so no
                        // signed-zero tie arises. A tolerance would let a wrong stride
                        // that landed close to the right fibre pass, which is exactly
                        // what this check exists to refuse. It survives contraction too:
                        // the operands of the fused form are the same on both sides,
                        // and `fma(0, v, s)` is `s`.
                        PANTR_CHECK_MSG(mine == reference,
                                        "the axis sweep and the dense product disagree at "
                                        "outer "
                                            + std::to_string(outer) + ", inner "
                                            + std::to_string(inner));
                    }
                }
            }
        }
    }
}

/// Subdividing is inserting the knots `uniform_subdivision_knots` names.
///
/// Not a mirror of the recurrence: it pins that the two entry points agree on *which*
/// knots go in and on nothing else, which is the one thing a reader would otherwise
/// have to take on trust when reading `subdivide` beside `insert_knots`.
void check_subdivide_is_insert_knots_of_the_subdivision_knots() {
    const std::vector<double> knots = {0.0, 0.0, 0.0, 1.0 / 3.0, 1.0, 1.0, 1.0};
    const auto space =
        std::make_shared<const BsplineSpace<double>>(make_directions<double>({knots}, {2}));
    const Bspline<double> field = marsden_field<double>(space, false);

    const std::vector<std::int64_t> counts = {3};
    const Bspline<double> by_subdivide =
        subdivide<double>(field, std::span<const std::int64_t>(counts), 0);
    const std::vector<double> to_insert = pantr::bspline::uniform_subdivision_knots<double>(
        space->space_ref(0).knots(), 2, space->space_ref(0).tolerance(), 3, 0);
    const std::vector<std::span<const double>> per_direction = {
        std::span<const double>(to_insert)};
    const Bspline<double> by_insert = insert_knots<double>(
        field, std::span<const std::span<const double>>(per_direction));

    PANTR_CHECK(by_subdivide.net().values().size() == by_insert.net().values().size());
    bool identical = by_subdivide.space_ref().space_ref(0).knots().size()
                     == by_insert.space_ref().space_ref(0).knots().size();
    for (std::size_t i = 0; identical && i < by_insert.net().values().size(); ++i) {
        identical = by_subdivide.net().values()[i] == by_insert.net().values()[i];
    }
    PANTR_CHECK_MSG(identical, "subdivide and insert_knots disagree on the same knots");
}

/// Every refusal, by its message.
void check_refusals() {
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const auto space =
        std::make_shared<const BsplineSpace<double>>(make_directions<double>({knots}, {2}));
    const Bspline<double> field = marsden_field<double>(space, false);

    const auto refused = [](const std::string& expected, const auto& call) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            const std::string text = error.what();
            PANTR_CHECK_MSG(text == expected,
                            "expected \"" + expected + "\" and got \"" + text + "\"");
            return;
        }
        PANTR_CHECK_MSG(false, "no refusal where \"" + expected + "\" was expected");
    };

    const std::vector<double> one = {0.25};
    const std::vector<std::span<const double>> two_lists = {std::span<const double>(one),
                                                            std::span<const double>(one)};
    refused("new_knots sequence length (2) must match dim (1).", [&] {
        static_cast<void>(
            insert_knots<double>(field, std::span<const std::span<const double>>(two_lists)));
    });

    const std::vector<std::span<const double>> empty_list = {std::span<const double>()};
    refused("At least one direction must have a non-empty array of knots to insert.", [&] {
        static_cast<void>(
            insert_knots<double>(field, std::span<const std::span<const double>>(empty_list)));
    });

    const std::vector<double> outside = {1.5};
    const std::vector<std::span<const double>> outside_list = {std::span<const double>(outside)};
    refused("new_knots contains values outside the domain [0.0, 1.0]: [1.5]", [&] {
        static_cast<void>(
            insert_knots<double>(field, std::span<const std::span<const double>>(outside_list)));
    });

    // Degree 2, so a knot may reach multiplicity 3; a fourth copy is refused.
    const std::vector<double> crowded = {0.5, 0.5, 0.5};
    const std::vector<std::span<const double>> crowded_list = {std::span<const double>(crowded)};
    refused("Inserting these knots would exceed the maximum multiplicity of 3. Maximum "
            "multiplicity found: 4.",
            [&] {
                static_cast<void>(insert_knots<double>(
                    field, std::span<const std::span<const double>>(crowded_list)));
            });

    const std::vector<std::int64_t> two_counts = {2, 2};
    refused("n_subdivisions sequence length (2) must match dim (1).", [&] {
        static_cast<void>(
            subdivide<double>(field, std::span<const std::int64_t>(two_counts), std::nullopt));
    });

    const std::vector<std::int64_t> zero = {0};
    refused("n_subdivisions must be >= 1, got 0", [&] {
        static_cast<void>(
            subdivide<double>(field, std::span<const std::int64_t>(zero), std::nullopt));
    });

    const std::vector<std::int64_t> ones = {1};
    refused("At least one direction must have n_subdivisions >= 2.", [&] {
        static_cast<void>(
            subdivide<double>(field, std::span<const std::int64_t>(ones), std::nullopt));
    });

    const std::vector<std::int64_t> twos = {2};
    refused("regularity must be in [-1, degree - 1] = [-1, 1] for direction 0, got 2", [&] {
        static_cast<void>(subdivide<double>(field, std::span<const std::int64_t>(twos), 2));
    });
    refused("regularity must be in [-1, degree - 1] = [-1, 1] for direction 0, got -2", [&] {
        static_cast<void>(subdivide<double>(field, std::span<const std::int64_t>(twos), -2));
    });

    // The count check runs over every direction before any regularity is looked at, so a
    // field bad in both ways reports the count. Only the order decides which message a
    // caller reads, which is why it is pinned rather than left to the implementation.
    const std::vector<std::vector<double>> pair = {knots, knots};
    const auto surface =
        std::make_shared<const BsplineSpace<double>>(make_directions<double>(pair, {2, 2}));
    const Bspline<double> field_2d = marsden_field<double>(surface, false);
    const std::vector<std::int64_t> bad_both = {2, 0};
    refused("n_subdivisions must be >= 1, got 0", [&] {
        static_cast<void>(
            subdivide<double>(field_2d, std::span<const std::int64_t>(bad_both), 5));
    });

    // `regularity = -1` lands ON the ceiling rather than above it, so it succeeds. This
    // is the positive half of the reachability argument in `subdivide`'s own note, and
    // it is asserted because a comment in `cpp/tests/test_bspline_knot_insertion.cpp`
    // states the opposite -- that this very call is what makes the multiplicity refusal
    // reachable from a subdivision. It is not: the interior points of a span collide
    // with nothing, so `degree - regularity <= degree + 1` is never exceeded.
    const std::vector<double> linear = {0.0, 0.0, 0.5, 1.0, 1.0};
    const auto linear_space =
        std::make_shared<const BsplineSpace<double>>(make_directions<double>({linear}, {1}));
    const Bspline<double> linear_field = marsden_field<double>(linear_space, false);
    const Bspline<double> discontinuous =
        subdivide<double>(linear_field, std::span<const std::int64_t>(twos), -1);
    const BsplineSpace1D<double>& refined = discontinuous.space_ref().space_ref(0);
    // Two spans, one interior point each, each at multiplicity `degree + 1 = 2`.
    PANTR_CHECK_MSG(refined.knots().size() == linear.size() + 4,
                    "regularity -1 at degree 1 did not insert two double knots");
    std::int64_t worst = 0;
    for (const std::int64_t mult : refined.multiplicity()) {
        worst = std::max(worst, mult);
    }
    PANTR_CHECK_MSG(worst == 2, "regularity -1 at degree 1 should reach multiplicity "
                                "degree + 1 exactly, and reached "
                                    + std::to_string(worst));
    check_marsden_survives<double>(discontinuous, "regularity -1 at degree 1", 1);
}

/// A periodic direction is refused, and the message says it is a boundary.
void check_periodic_is_refused() {
    const std::vector<double> uniform = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<double> clamped = {0.0, 0.0, 0.5, 1.0, 1.0};
    std::vector<std::shared_ptr<const BsplineSpace1D<double>>> directions;
    directions.push_back(std::make_shared<const BsplineSpace1D<double>>(
        std::span<const double>(uniform), 2, true, KnotSnapping::merge_near_duplicates));
    directions.push_back(std::make_shared<const BsplineSpace1D<double>>(
        std::span<const double>(clamped), 1, false, KnotSnapping::merge_near_duplicates));
    const auto space = std::make_shared<const BsplineSpace<double>>(directions);
    const Bspline<double> field = marsden_field<double>(space, false);

    const std::vector<double> one = {0.25};
    const std::vector<std::span<const double>> refine_periodic = {
        std::span<const double>(one), std::span<const double>()};
    bool refused = false;
    try {
        static_cast<void>(insert_knots<double>(
            field, std::span<const std::span<const double>>(refine_periodic)));
    } catch (const std::invalid_argument& error) {
        refused = true;
        PANTR_CHECK_MSG(std::string(error.what())
                            == "refining a periodic direction needs the open-to-periodic "
                               "conversion, which is not part of pantr's C++ core; direction "
                               "0 is periodic.",
                        std::string("the periodic refusal reads \"") + error.what() + "\"");
    }
    PANTR_CHECK_MSG(refused, "a periodic direction was refined rather than refused");

    // The other direction still refines, and the periodic one is carried over whole:
    // untouched means untouched, not converted.
    const std::vector<std::span<const double>> refine_clamped = {std::span<const double>(),
                                                                 std::span<const double>(one)};
    const Bspline<double> refined =
        insert_knots<double>(field, std::span<const std::span<const double>>(refine_clamped));
    PANTR_CHECK_MSG(refined.space_ref().space(0) == space->space(0),
                    "the periodic direction did not keep its space");
    PANTR_CHECK_MSG(refined.space_ref().space_ref(0).periodic(),
                    "the periodic direction stopped being periodic");
}

/// `refine_along_axis` refuses a shape it cannot address.
void check_the_sweep_refuses_a_bad_shape() {
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const BsplineSpace1D<double> coarse(std::span<const double>(knots), 2, false,
                                        KnotSnapping::merge_near_duplicates);
    const BsplineSpace1D<double> fine = subdivide<double>(coarse, 2, std::nullopt);
    const auto bands = oslo_bands_1d<double>(2, coarse.knots(), fine.knots());
    const std::vector<double> values(7, 1.0);

    bool refused = false;
    try {
        static_cast<void>(refine_along_axis<double>(
            bands, coarse.num_basis(), std::span<const double>(values), 1, 1));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    PANTR_CHECK_MSG(refused, "a buffer that is not outer * num_cols * inner was accepted");

    refused = false;
    try {
        static_cast<void>(refine_along_axis<double>(
            bands, coarse.num_basis(), std::span<const double>(values), -1, 1));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    PANTR_CHECK_MSG(refused, "a negative extent was accepted");
}

}  // namespace

int main() {
    check_the_bands_are_a_convex_combination();
    check_subdivide_1d<double>("float64");
    check_subdivide_1d<float>("float32");
    check_insert_knots_1d<double>("float64");
    check_insert_knots_1d<float>("float32");
    check_multidimensional<double>("float64");
    check_multidimensional<float>("float32");
    check_regularity();
    check_unclamped();
    check_the_axis_sweep_addresses_the_right_fibres();
    check_subdivide_is_insert_knots_of_the_subdivision_knots();
    check_refusals();
    check_periodic_is_refused();
    check_the_sweep_refuses_a_bad_shape();
    return pantr::test::summary("test_bspline_refinement");
}
