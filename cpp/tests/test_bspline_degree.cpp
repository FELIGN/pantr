/// \file
/// Changing a B-spline field's degree: the hodograph, and degree elevation.
///
/// ## The independent oracle, and why it is not a rerun of the code
///
/// Both operations are checked against **Marsden's identity**, which gives the B-spline
/// coefficients of a monomial in closed form. For a knot vector `t` and degree `p`,
///
///     u^r = sum_i [ e_r(t_{i+1}, ..., t_{i+p}) / C(p, r) ] * N_{i,p}(u),
///
/// with `e_r` the elementary symmetric polynomial;
/// `cpp/tests/test_bspline_refinement.cpp` derives it there and this file forms the same
/// closed form independently, from elementary symmetric polynomials and a binomial
/// coefficient and nothing else.
///
/// It gives each operation its own corollary.
///
/// **Elevation.** A field whose coefficients are the closed form *is* the map
/// `u -> u^r`. Elevation must not move it, so the elevated coefficients must be the same
/// closed form evaluated on the **elevated** knot vector at the **elevated** degree.
/// Taking every `r` in `[0, p]` pins each row completely rather than up to an affine
/// map, which is the argument the refinement file spells out.
///
/// **The hodograph, and here the corollary is an identity rather than a statement about
/// the answer.** Differentiating `u^r` gives `r u^{r-1}`, whose coefficients on the
/// derived vector `t[1:-1]` at degree `p - 1` are
/// `r e_{r-1}(t_{i+2}, ..., t_{i+p}) / C(p-1, r-1)`. That the code's formula produces
/// exactly those numbers is provable in two lines, and the proof is worth having here
/// because it says the check is exact rather than approximate:
///
///     A_{i+1} - A_i = [e_r(t_{i+2..i+p+1}) - e_r(t_{i+1..i+p})] / C(p, r)
///                   = (t_{i+p+1} - t_{i+1}) e_{r-1}(t_{i+2..i+p}) / C(p, r),
///
/// using `e_r(S + {x}) = e_r(S) + x e_{r-1}(S)` on the shared window
/// `S = {t_{i+2..i+p}}`. So `p (A_{i+1} - A_i) / (t_{i+p+1} - t_{i+1})` is
/// `p e_{r-1}(S) / C(p, r)`, and `C(p, r) = (p/r) C(p-1, r-1)` makes that equal to
/// `r e_{r-1}(S) / C(p-1, r-1)`, which is the derived vector's own closed form. The
/// identity holds for **any** knot vector, clamped or not, wherever the denominator is
/// non-zero.
///
/// ## The bound for the hodograph
///
/// `gamma_K = K u / (1 - K u)` times the magnitude the value is built from, with `K`
/// counted rather than fitted:
///
///  - **The oracle's own formation on each side.** `e_r` by the recurrence
///    `E_j <- E_j + x E_{j-1}` commits two roundings per knot along the dominant chain,
///    then one division by a binomial: `2p + 1` for the input window of `p` knots and
///    `2(p-1) + 1` for the derived window of `p - 1`.
///  - **The cast into the field's storage**, which at `float32` is a real rounding on
///    the input the formula then propagates: `1`.
///  - **The formula itself**: the coefficient difference, the knot difference, the
///    scaling by the degree and the division: `4`.
///  - **The store into the result**, symmetric to the cast in: `1`.
///
/// `K = 4p + 5`. The magnitude is **not** the result's own: the difference
/// `A_{i+1} - A_i` can cancel to nothing while its operands do not, so the relative
/// budget has to be charged against what was subtracted. Rule 2 of
/// `design/backend_parity.md` is the same point in the other direction. So
///
///     amplification_i = p (|A_i| + |A_{i+1}|) / |t_{i+p+1} - t_{i+1}|,
///
/// which majorises both the computed value and every partial result that fed it, since
/// the only operations are one subtraction, one scaling and one division.
///
/// ## The bound for the elevation, and why a convex companion will not do
///
/// The finished elevation operator is non-negative with rows summing to one -- degree
/// elevation writes each `N_{i,p}` as a non-negative combination of the elevated basis --
/// so every *output* coefficient is a convex combination of input ones and `max |c|`
/// bounds it. **That is not the bound this needs**, and the distinction is
/// `design/backend_parity.md` Rule 10's, recorded there for another kernel: a rounding
/// budget has to be charged against the intermediates, and A5.9's are not convex
/// combinations of anything. Its knot-removal step blends with
/// `alf = (ub - ik[i]) / (ua - ik[i])`, whose numerator exceeds its denominator whenever
/// `ik[i] < ua < ub`, so `alf > 1` and `1 - alf < 0`. Partial results there can exceed
/// every input coefficient, and a companion built from the finished row would be
/// exceeded rather than merely loose.
///
/// What works is Rule 10's own answer: **the majorant of the recursion itself**. Run A5.9
/// with every coefficient replaced by its modulus and every blending weight by its, which
/// bounds each intermediate by induction on a linear recursion with signed coefficients;
/// with the signs gone every partial sum is monotone, so the final value majorises all of
/// them. `elevate_majorant` below is that run, and it is a bound rather than a second
/// answer: it never produces the value the test compares, which comes from Marsden.
///
/// `K` for the elevation is counted the same way:
///
///  - the two `e_r` recurrences, on the input vector at degree `p` and on the elevated
///    one at degree `p + t`: `2p + 1` and `2(p + t) + 1`;
///  - the cast in and the store out: `2`;
///  - **the kernel's own chain**, `min(p, t) + 1` stages of three accumulator roundings
///    each. That stage count is `design/backend_parity.md` Rule 10's for the Bézier
///    elevation and it is the right one here for the same reason: the accumulation into
///    `ebpts[i]` runs `j` from `max(0, i - t)` to `min(p, i)`. Charging `p + 1` instead
///    is the over-count that rule records.
///
/// The test asserts that the majorant **dominates** every observed deviation, which is
/// the premise, and separately that the bound is **approached** rather than merely
/// satisfied -- a bound nothing comes near asserts nothing, and a dyadic sweep would
/// round too little to ask it a question.
///
/// ## Where the oracles do not hold, and the vacuity guards that say so
///
/// A row of the hodograph whose knot difference vanishes -- an interior knot of
/// multiplicity `degree + 1`, where the field is `C^-1` -- is left at an exact zero by
/// both backends, while Marsden's closed form there is whatever it is. The basis function
/// that coefficient multiplies has empty support, so every value gives the same map and
/// the row is excluded; the count of excluded rows is asserted non-zero on the case built
/// to produce them, so the exclusion cannot silently become the whole check.
///
/// A **periodic** direction has no monomial to be the coefficients of: `u^r` is not
/// periodic, so Marsden's closed form is not a periodic field and the identity says
/// nothing about one. What is checked there instead is stated at the check itself, and it
/// is weaker on purpose: the exact zero hodograph of a constant field, the periodicity of
/// the expanded derivative net, and the structure of the result.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/control_net.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/degree.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"

namespace {

using pantr::bspline::Bspline;
using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::degree_elevate_1d;
using pantr::bspline::derivative;
using pantr::bspline::derivative_along_axis;
using pantr::bspline::elevate_degree;
using pantr::bspline::KnotSnapping;

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

/// The smallest positive normal of `T`, the absolute floor of Higham's model.
///
/// \tparam T The storage format.
/// \return The floor, one per rounding charged.
template <class T>
double underflow_floor() {
    return static_cast<double>(std::numeric_limits<T>::min());
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
/// `A_i = e_r(t_{i+1}, ..., t_{i+degree}) / C(degree, r)`; see the file comment. Formed
/// in `double` whatever the knots are stored in, by the elementary symmetric recurrence,
/// so nothing here re-runs the code under test.
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

/// A one-direction field over given knots, degree and coefficients.
///
/// \tparam T The scalar type the field stores.
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \param periodic Whether the direction wraps.
/// \param coefficients One scalar coefficient per basis function.
/// \return The field, non-rational, of rank 1.
template <class T>
Bspline<T> curve(std::span<const T> knots, std::int64_t degree, bool periodic,
                 const std::vector<double>& coefficients) {
    auto space = std::make_shared<const BsplineSpace1D<T>>(knots, degree, periodic,
                                                           KnotSnapping::merge_near_duplicates);
    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions{space};
    std::vector<T> values(coefficients.size());
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        values[i] = static_cast<T>(coefficients[i]);
    }
    const std::vector<std::size_t> shape{coefficients.size(), std::size_t{1}};
    return Bspline<T>(std::make_shared<const BsplineSpace<T>>(std::move(directions)),
                      typename Bspline<T>::net_type(std::span<const T>(values),
                                                    std::span<const std::size_t>(shape)),
                      false);
}

/// The majorant of A5.9, run on the moduli of the coefficients and of the weights.
///
/// A transliteration of `degree_elevate_1d`'s recursion with every sign removed, in
/// `double` throughout. It bounds every intermediate of the real run by induction, and
/// because every term is non-negative its partial sums are monotone, so the value it
/// returns for an output coefficient majorises everything that fed that coefficient. See
/// the file comment for why a companion built from the finished operator does not.
///
/// It computes no value the test compares against; it computes only a magnitude.
///
/// \param degree The original degree.
/// \param magnitudes One non-negative magnitude per coefficient.
/// \param knots The knot vector.
/// \param increment Degrees to add.
/// \return One majorant per elevated coefficient.
std::vector<double> elevate_majorant(std::int64_t degree, const std::vector<double>& magnitudes,
                                     const std::vector<double>& knots, std::int64_t increment) {
    const auto num_rows = static_cast<std::int64_t>(magnitudes.size());
    const auto knot_count = static_cast<std::int64_t>(knots.size());
    const std::int64_t d = degree;
    const std::int64_t t = increment;
    const std::int64_t ph = d + t;
    const std::int64_t ph2 = ph / 2;
    const std::int64_t m = (num_rows - 1) + d + 1;

    std::vector<double> bezalfs(static_cast<std::size_t>((d + 1) * (ph + 1)), 0.0);
    std::vector<double> alfs(static_cast<std::size_t>(std::max<std::int64_t>(d, 1)), 0.0);
    std::vector<double> bpts(static_cast<std::size_t>(d + 1), 0.0);
    std::vector<double> ebpts(static_cast<std::size_t>(ph + 1), 0.0);
    std::vector<double> next_bpts(static_cast<std::size_t>(d + 1), 0.0);

    const auto bez = [ph](std::int64_t j, std::int64_t i) {
        return static_cast<std::size_t>(j * (ph + 1) + i);
    };
    bezalfs[bez(0, 0)] = 1.0;
    bezalfs[bez(d, ph)] = 1.0;
    for (std::int64_t i = 1; i <= ph2; ++i) {
        const double inv = 1.0 / static_cast<double>(binomial(ph, i));
        for (std::int64_t j = std::max<std::int64_t>(0, i - t); j <= std::min(d, i); ++j) {
            bezalfs[bez(j, i)] = inv * static_cast<double>(binomial(d, j))
                                 * static_cast<double>(binomial(t, i - j));
        }
    }
    for (std::int64_t i = ph2 + 1; i < ph; ++i) {
        for (std::int64_t j = std::max<std::int64_t>(0, i - t); j <= std::min(d, i); ++j) {
            bezalfs[bez(j, i)] = bezalfs[bez(d - j, ph - i)];
        }
    }

    std::int64_t kind = ph + 1;
    std::int64_t r = -1;
    std::int64_t a = d;
    std::int64_t b = d + 1;
    std::int64_t cind = 1;
    double ua = knots[0];
    std::vector<double> ik(static_cast<std::size_t>(knot_count + t * knot_count), 0.0);
    std::vector<double> ic(static_cast<std::size_t>(num_rows + t * knot_count), 0.0);

    ic[0] = std::abs(magnitudes[0]);
    for (std::int64_t i = 0; i <= ph; ++i) {
        ik[static_cast<std::size_t>(i)] = ua;
    }
    for (std::int64_t i = 0; i <= d; ++i) {
        bpts[static_cast<std::size_t>(i)] = std::abs(magnitudes[static_cast<std::size_t>(i)]);
    }

    while (b <= m) {
        const std::int64_t run_start = b;
        while (b < m
               && knots[static_cast<std::size_t>(b)] == knots[static_cast<std::size_t>(b + 1)]) {
            ++b;
        }
        const std::int64_t mul = b - run_start + 1;
        const double ub = knots[static_cast<std::size_t>(b)];
        const std::int64_t oldr = r;
        r = d - mul;

        std::int64_t lbz = 1;
        if (oldr > 0) {
            lbz = (oldr + 2) / 2;
        } else if (oldr < 0 && a != d) {
            lbz = 0;
        }
        const std::int64_t rbz = (r > 0) ? ph - (r + 1) / 2 : ph;

        if (r > 0) {
            const double numer = ub - ua;
            for (std::int64_t q = d; q > mul; --q) {
                alfs[static_cast<std::size_t>(q - mul - 1)] =
                    numer / (knots[static_cast<std::size_t>(a + q)] - ua);
            }
            for (std::int64_t j = 1; j <= r; ++j) {
                const std::int64_t save = r - j;
                const std::int64_t s = mul + j;
                for (std::int64_t q = d; q >= s; --q) {
                    const double alf = alfs[static_cast<std::size_t>(q - s)];
                    bpts[static_cast<std::size_t>(q)] =
                        std::abs(alf) * bpts[static_cast<std::size_t>(q)]
                        + std::abs(1.0 - alf) * bpts[static_cast<std::size_t>(q - 1)];
                }
                next_bpts[static_cast<std::size_t>(save)] = bpts[static_cast<std::size_t>(d)];
            }
        }

        for (std::int64_t i = lbz; i <= ph; ++i) {
            ebpts[static_cast<std::size_t>(i)] = 0.0;
            for (std::int64_t j = std::max<std::int64_t>(0, i - t); j <= std::min(d, i); ++j) {
                ebpts[static_cast<std::size_t>(i)] +=
                    std::abs(bezalfs[bez(j, i)]) * bpts[static_cast<std::size_t>(j)];
            }
        }

        if (oldr > 1) {
            std::int64_t first = kind - 2;
            std::int64_t last = kind;
            const double den = ub - ua;
            const double bet = (ub - ik[static_cast<std::size_t>(kind - 1)]) / den;
            for (std::int64_t tr = 1; tr < oldr; ++tr) {
                std::int64_t i = first;
                std::int64_t j = last;
                std::int64_t kj = j - kind + 1;
                while (j - i > tr) {
                    if (i < cind) {
                        const double alf = (ub - ik[static_cast<std::size_t>(i)])
                                           / (ua - ik[static_cast<std::size_t>(i)]);
                        ic[static_cast<std::size_t>(i)] =
                            std::abs(alf) * ic[static_cast<std::size_t>(i)]
                            + std::abs(1.0 - alf) * ic[static_cast<std::size_t>(i - 1)];
                    }
                    if (j >= lbz) {
                        const double weight = (j - tr <= kind - ph + oldr)
                                                  ? (ub - ik[static_cast<std::size_t>(j - tr)]) / den
                                                  : bet;
                        ebpts[static_cast<std::size_t>(kj)] =
                            std::abs(weight) * ebpts[static_cast<std::size_t>(kj)]
                            + std::abs(1.0 - weight) * ebpts[static_cast<std::size_t>(kj + 1)];
                    }
                    ++i;
                    --j;
                    --kj;
                }
                --first;
                ++last;
            }
        }

        if (a != d) {
            for (std::int64_t written = 0; written < ph - oldr; ++written) {
                ik[static_cast<std::size_t>(kind)] = ua;
                ++kind;
            }
        }
        for (std::int64_t j = lbz; j <= rbz; ++j) {
            ic[static_cast<std::size_t>(cind)] = ebpts[static_cast<std::size_t>(j)];
            ++cind;
        }
        if (b < m) {
            for (std::int64_t j = 0; j < r; ++j) {
                bpts[static_cast<std::size_t>(j)] = next_bpts[static_cast<std::size_t>(j)];
            }
            for (std::int64_t j = std::max<std::int64_t>(0, r); j <= d; ++j) {
                bpts[static_cast<std::size_t>(j)] =
                    std::abs(magnitudes[static_cast<std::size_t>(b - d + j)]);
            }
            a = b;
            ++b;
            ua = ub;
        } else {
            for (std::int64_t i = 0; i <= ph; ++i) {
                ik[static_cast<std::size_t>(kind + i)] = ub;
            }
            break;
        }
    }
    return std::vector<double>(ic.begin(), ic.begin() + static_cast<std::ptrdiff_t>(cind));
}

/// How close a comparison came to its own bound, and whether it stayed inside it.
struct Margin {
    /// The largest `|deviation| / bound` seen, which must not exceed 1.
    double worst_ratio = 0.0;
    /// How many elements were compared, so that a filter matching nothing is visible.
    std::int64_t compared = 0;
};

/// Compare one elevated curve against Marsden's closed form on the elevated vector.
///
/// \tparam T The scalar type the field stores.
/// \param knots The original knot vector, stored in `T`.
/// \param degree The original degree.
/// \param increment Degrees to add.
/// \param label What to name the case in a failure message.
/// \return The margin over every power `r` in `[0, degree]`.
template <class T>
Margin check_elevation_case(const std::vector<T>& knots, std::int64_t degree,
                            std::int64_t increment, const std::string& label) {
    Margin margin;
    const std::vector<double> knots_wide(knots.begin(), knots.end());
    for (std::int64_t r = 0; r <= degree; ++r) {
        const std::vector<double> coefficients =
            marsden_coefficients<T>(std::span<const T>(knots), degree, r);
        const Bspline<T> field = curve<T>(std::span<const T>(knots), degree, false, coefficients);
        const std::vector<std::int64_t> increments{increment};
        const Bspline<T> raised =
            elevate_degree<T>(field, std::span<const std::int64_t>(increments));

        const BsplineSpace1D<T>& space = raised.space_ref().space_ref(0);
        PANTR_CHECK_MSG(space.degree() == degree + increment,
                        label + ": the elevated degree is " + std::to_string(space.degree()));
        const std::vector<double> expected =
            marsden_coefficients<T>(space.knots(), space.degree(), r);
        const std::span<const T> got = raised.net().values();
        if (static_cast<std::int64_t>(got.size()) != static_cast<std::int64_t>(expected.size())) {
            PANTR_CHECK_MSG(false, label + ": the elevation returned "
                                       + std::to_string(got.size()) + " coefficients and the "
                                       "elevated vector calls for "
                                       + std::to_string(expected.size()));
            continue;
        }

        std::vector<double> magnitudes(coefficients.size());
        for (std::size_t i = 0; i < coefficients.size(); ++i) {
            magnitudes[i] = std::abs(static_cast<double>(static_cast<T>(coefficients[i])));
        }
        const std::vector<double> amplification =
            elevate_majorant(degree, magnitudes, knots_wide, increment);
        PANTR_CHECK_MSG(amplification.size() == expected.size(),
                        label + ": the majorant and the elevation disagree on the coefficient "
                                "count, so one of them walked a different set of segments");

        const std::int64_t stages = std::min(degree, increment) + 1;
        const std::int64_t roundings =
            (2 * degree + 1) + (2 * (degree + increment) + 1) + 2 + 3 * stages;
        const double relative = gamma_of<T>(roundings);
        const double floor = static_cast<double>(roundings) * underflow_floor<T>();
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const double bound = relative * amplification[i] + floor;
            const double deviation = std::abs(static_cast<double>(got[i]) - expected[i]);
            ++margin.compared;
            margin.worst_ratio = std::max(margin.worst_ratio, deviation / bound);
            PANTR_CHECK_MSG(deviation <= bound,
                            label + ": elevated coefficient " + std::to_string(i) + " of u^"
                                + std::to_string(r) + " is off by "
                                + std::to_string(deviation) + " against a bound of "
                                + std::to_string(bound));
        }
    }
    return margin;
}

/// Degree elevation reproduces the polynomial, on every multiplicity structure.
///
/// \tparam T The scalar type the field stores.
/// \param label The storage format, for failure messages.
template <class T>
void check_the_elevation_reproduces_the_polynomial(const std::string& label) {
    struct Case {
        std::vector<T> knots;
        std::int64_t degree;
        std::int64_t increment;
        const char* name;
    };
    // Every case carries a non-dyadic knot so that the arithmetic rounds and the bound is
    // asked a question; a halving sweep alone would report agreement without one.
    const std::vector<Case> cases{
        {{T(0), T(0), T(0), T(1), T(1), T(1)}, 2, 1, "one quadratic Bezier span"},
        {{T(0), T(0), T(0), T(0), T(1), T(1), T(1), T(1)}, 3, 3, "one cubic span, t = 3"},
        {{T(0), T(0), T(0), T(0.3), T(0.7), T(1), T(1), T(1)}, 2, 2, "simple interior knots"},
        {{T(0), T(0), T(0), T(0), T(0.25), T(0.5), T(0.75), T(1), T(1), T(1), T(1)},
         3,
         1,
         "cubic, four simple interior knots"},
        {{T(0), T(0), T(0), T(0), T(0.5), T(0.5), T(0.5), T(1), T(1), T(1), T(1)},
         3,
         2,
         "cubic, a C^0 breakpoint"},
        {{T(0), T(0), T(0), T(0), T(0.5), T(0.5), T(0.5), T(0.5), T(1), T(1), T(1), T(1)},
         3,
         1,
         "cubic, a C^-1 breakpoint"},
        {{T(0), T(0), T(0), T(0), T(0), T(0), T(0.1), T(0.15), T(0.9), T(1), T(1), T(1), T(1),
          T(1), T(1)},
         5,
         2,
         "quintic, a narrow span"},
    };

    Margin total;
    for (const Case& one : cases) {
        const Margin margin = check_elevation_case<T>(one.knots, one.degree, one.increment,
                                                      label + " " + one.name);
        total.compared += margin.compared;
        total.worst_ratio = std::max(total.worst_ratio, margin.worst_ratio);
    }
    PANTR_CHECK_MSG(total.compared > 0, label + ": the elevation check compared nothing");
    // The bound has to be approached, or it asserts nothing. A tenth of it is the
    // acknowledged slack: the budget charges the two oracle recurrences at the storage
    // format although they run in `double`, which alone over-states the float64 case by
    // most of `K`.
    PANTR_CHECK_MSG(total.worst_ratio > 0.01,
                    label + ": the elevation bound was never approached, worst ratio "
                        + std::to_string(total.worst_ratio)
                        + "; either nothing rounded or the bound is vacuous");
}

/// The hodograph is the analytic derivative, on every knot vector shape.
///
/// \tparam T The scalar type the field stores.
/// \param label The storage format, for failure messages.
template <class T>
void check_the_hodograph_is_the_analytic_derivative(const std::string& label) {
    struct Case {
        std::vector<T> knots;
        std::int64_t degree;
        const char* name;
    };
    const std::vector<Case> cases{
        {{T(0), T(0), T(0), T(0.3), T(0.7), T(1), T(1), T(1)}, 2, "quadratic, simple knots"},
        {{T(0), T(0), T(0), T(0), T(0.25), T(0.5), T(0.75), T(1), T(1), T(1), T(1)}, 3,
         "cubic, four simple knots"},
        {{T(0), T(0), T(0), T(0), T(0.5), T(0.5), T(0.5), T(1), T(1), T(1), T(1)}, 3,
         "cubic, a C^0 breakpoint"},
        // Unclamped, and legal: `BsplineSpace1D` accepts it with `periodic = false`. The
        // hodograph identity is a statement about the coefficient formula and does not
        // need a clamp, unlike A5.9 -- which is exactly why elevation refuses this vector
        // and the derivative does not.
        {{T(-0.3), T(-0.2), T(-0.1), T(0), T(0.25), T(0.5), T(0.75), T(1), T(1.1), T(1.2),
          T(1.3)},
         3,
         "unclamped"},
        {{T(0), T(0), T(0), T(0), T(0), T(0), T(0.1), T(0.15), T(0.9), T(1), T(1), T(1), T(1),
          T(1), T(1)},
         5,
         "quintic, a narrow span"},
    };

    std::int64_t compared = 0;
    double worst_ratio = 0.0;
    for (const Case& one : cases) {
        const std::int64_t p = one.degree;
        const std::vector<T> derived(one.knots.begin() + 1, one.knots.end() - 1);
        for (std::int64_t r = 0; r <= p; ++r) {
            const std::vector<double> coefficients =
                marsden_coefficients<T>(std::span<const T>(one.knots), p, r);
            const Bspline<T> field =
                curve<T>(std::span<const T>(one.knots), p, false, coefficients);
            const Bspline<T> hodograph = derivative<T>(field, 0);

            const BsplineSpace1D<T>& space = hodograph.space_ref().space_ref(0);
            PANTR_CHECK_MSG(space.degree() == p - 1,
                            label + " " + one.name + ": the hodograph's degree is "
                                + std::to_string(space.degree()));

            // `r u^{r-1}`, and an exact zero for the constant.
            std::vector<double> expected(coefficients.size() - 1, 0.0);
            if (r >= 1) {
                const std::vector<double> lower =
                    marsden_coefficients<T>(std::span<const T>(derived), p - 1, r - 1);
                for (std::size_t i = 0; i < expected.size(); ++i) {
                    expected[i] = static_cast<double>(r) * lower[i];
                }
            }

            const std::int64_t roundings = (2 * p + 1) + (2 * (p - 1) + 1) + 2 + 4;
            const double relative = gamma_of<T>(roundings);
            const double floor = static_cast<double>(roundings) * underflow_floor<T>();
            const std::span<const T> got = hodograph.net().values();
            for (std::size_t i = 0; i < expected.size(); ++i) {
                const double denominator =
                    static_cast<double>(one.knots[i + static_cast<std::size_t>(p) + 1])
                    - static_cast<double>(one.knots[i + 1]);
                if (denominator == 0.0) {
                    // A `C^-1` breakpoint: the basis function this coefficient multiplies
                    // has empty support, so both backends leave it at an exact zero and
                    // Marsden's closed form says nothing about it.
                    PANTR_CHECK_MSG(got[i] == T(0), label + " " + one.name
                                                        + ": a zero-width span did not give an "
                                                          "exact zero");
                    continue;
                }
                const double amplification =
                    static_cast<double>(p)
                    * (std::abs(coefficients[i]) + std::abs(coefficients[i + 1]))
                    / std::abs(denominator);
                const double bound = relative * amplification + floor;
                const double deviation = std::abs(static_cast<double>(got[i]) - expected[i]);
                ++compared;
                worst_ratio = std::max(worst_ratio, deviation / bound);
                PANTR_CHECK_MSG(deviation <= bound,
                                label + " " + one.name + ": hodograph coefficient "
                                    + std::to_string(i) + " of u^" + std::to_string(r)
                                    + " is off by " + std::to_string(deviation)
                                    + " against a bound of " + std::to_string(bound));
            }
        }
    }
    PANTR_CHECK_MSG(compared > 0, label + ": the hodograph check compared nothing");
    PANTR_CHECK_MSG(worst_ratio > 0.0,
                    label + ": every hodograph coefficient was exact, so the bound was never "
                            "asked a question");
}

/// A `C^-1` breakpoint leaves an exact zero, and the case really contains one.
void check_a_zero_width_span_is_left_alone() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0};
    const std::vector<double> coefficients{1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0};
    const Bspline<double> field = curve<double>(std::span<const double>(knots), 3, false,
                                                coefficients);
    const Bspline<double> hodograph = derivative<double>(field, 0);
    const std::span<const double> got = hodograph.net().values();

    std::int64_t zeros = 0;
    for (std::size_t i = 0; i + 1 < coefficients.size(); ++i) {
        if (knots[i + 4] == knots[i + 1]) {
            ++zeros;
            PANTR_CHECK_MSG(got[i] == 0.0,
                            "a zero-width span gave " + std::to_string(got[i])
                                + " rather than an exact zero");
        }
    }
    // The guard that stops the check above from being about nothing: a `C^-1` breakpoint
    // at multiplicity `degree + 1` collapses exactly one span of the derived vector.
    PANTR_CHECK_MSG(zeros == 1, "the C^-1 case produced " + std::to_string(zeros)
                                    + " zero-width spans, and it was built for one");
}

/// The hodograph of a periodic direction: what can be checked without a monomial.
///
/// `u^r` is not periodic, so Marsden gives no periodic field and the identity says
/// nothing here. Three statements that do not need one are checked instead: a constant
/// field differentiates to an exact zero, the expanded derivative net is itself periodic,
/// and the result's structure is the derived vector at one degree less, still periodic.
void check_the_periodic_hodograph() {
    const std::vector<double> knots{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::int64_t degree = 2;
    const auto space = std::make_shared<const BsplineSpace1D<double>>(
        std::span<const double>(knots), degree, true, KnotSnapping::merge_near_duplicates);
    const std::int64_t count = space->num_basis();
    PANTR_CHECK_MSG(count > 0, "the periodic fixture has no basis functions");

    // A constant field: a periodic B-spline basis is a partition of unity, so equal
    // coefficients are the constant map and its derivative is identically zero. Nothing
    // about the derivative formula is assumed -- the differences are exact zeros and the
    // quotient of an exact zero is one.
    const std::vector<double> ones(static_cast<std::size_t>(count), 1.0);
    const Bspline<double> flat = curve<double>(std::span<const double>(knots), degree, true, ones);
    const Bspline<double> flat_derivative = derivative<double>(flat, 0);
    for (const double value : flat_derivative.net().values()) {
        PANTR_CHECK_MSG(value == 0.0, "the hodograph of a constant periodic field is "
                                          + std::to_string(value) + " rather than zero");
    }

    const BsplineSpace1D<double>& derived = flat_derivative.space_ref().space_ref(0);
    PANTR_CHECK_MSG(derived.periodic(), "the hodograph of a periodic direction came back open");
    PANTR_CHECK_MSG(derived.degree() == degree - 1,
                    "the periodic hodograph's degree is " + std::to_string(derived.degree()));
    PANTR_CHECK_MSG(derived.knots().size() == knots.size() - 2,
                    "the periodic hodograph kept " + std::to_string(derived.knots().size())
                        + " knots and the derived vector has " + std::to_string(knots.size() - 2));

    // The other half: on data that is not constant, the derivative net must still be the
    // periodic representation of a periodic function, so expanding it by its own modulo
    // wrap has to agree with differentiating the expanded original. That is a property of
    // the result rather than a rerun of the formula: it says the trim kept the right rows.
    std::vector<double> varied(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < varied.size(); ++i) {
        varied[i] = static_cast<double>(i) - 1.5;
    }
    const Bspline<double> wave = curve<double>(std::span<const double>(knots), degree, true,
                                               varied);
    const Bspline<double> wave_derivative = derivative<double>(wave, 0);
    const std::span<const double> got = wave_derivative.net().values();
    const std::int64_t derived_count = wave_derivative.space_ref().space_ref(0).num_basis();
    PANTR_CHECK_MSG(static_cast<std::int64_t>(got.size()) == derived_count,
                    "the periodic hodograph carries " + std::to_string(got.size())
                        + " coefficients on a space of " + std::to_string(derived_count));

    // Differentiating the expanded net directly, with the expansion the result implies.
    const std::int64_t full = static_cast<std::int64_t>(knots.size()) - degree - 1;
    std::vector<double> expanded(static_cast<std::size_t>(full));
    for (std::int64_t i = 0; i < full; ++i) {
        expanded[static_cast<std::size_t>(i)] = varied[static_cast<std::size_t>(i % count)];
    }
    const std::vector<double> whole = derivative_along_axis<double>(
        std::span<const double>(knots), degree, std::span<const double>(expanded), full, 1, 1);
    for (std::size_t i = 0; i < got.size(); ++i) {
        PANTR_CHECK_MSG(got[i] == whole[i],
                        "the periodic trim kept a different row at " + std::to_string(i));
    }
    // And the expanded derivative wraps with the derived space's own period, which is what
    // makes the trim lossless rather than merely short.
    std::int64_t wrapped = 0;
    for (std::size_t i = 0; i + static_cast<std::size_t>(derived_count) < whole.size(); ++i) {
        ++wrapped;
        PANTR_CHECK_MSG(whole[i] == whole[i + static_cast<std::size_t>(derived_count)],
                        "the expanded hodograph is not periodic at " + std::to_string(i));
    }
    PANTR_CHECK_MSG(wrapped > 0, "no wrapped row was checked, so the trim is untested");
}

/// Each operation touches its own axis and leaves every other one alone.
///
/// A rank-1 field over two directions whose coefficients are `A^0_{i,r0} A^1_{j,r1}` is
/// the map `u^r0 v^r1`. Operating on direction `d` must move that direction's factor to
/// its own new closed form and leave the other factor exactly where it was, which is what
/// catches a sweep applied to the wrong axis, a transposed stride, or a component axis
/// mistaken for a parametric one. The degrees differ between the directions so that a
/// transposition cannot pass by symmetry.
void check_the_axis_sweeps_touch_only_their_own_axis() {
    const std::vector<double> first{0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0};
    const std::vector<double> second{0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0};
    const std::int64_t degree_first = 2;
    const std::int64_t degree_second = 3;

    for (std::int64_t r0 = 0; r0 <= degree_first; ++r0) {
        for (std::int64_t r1 = 0; r1 <= degree_second; ++r1) {
            const std::vector<double> table_first =
                marsden_coefficients<double>(std::span<const double>(first), degree_first, r0);
            const std::vector<double> table_second =
                marsden_coefficients<double>(std::span<const double>(second), degree_second, r1);

            std::vector<std::shared_ptr<const BsplineSpace1D<double>>> directions;
            directions.push_back(std::make_shared<const BsplineSpace1D<double>>(
                std::span<const double>(first), degree_first, false,
                KnotSnapping::merge_near_duplicates));
            directions.push_back(std::make_shared<const BsplineSpace1D<double>>(
                std::span<const double>(second), degree_second, false,
                KnotSnapping::merge_near_duplicates));
            std::vector<double> values(table_first.size() * table_second.size());
            for (std::size_t i = 0; i < table_first.size(); ++i) {
                for (std::size_t j = 0; j < table_second.size(); ++j) {
                    values[i * table_second.size() + j] = table_first[i] * table_second[j];
                }
            }
            const std::vector<std::size_t> shape{table_first.size(), table_second.size(),
                                                 std::size_t{1}};
            const Bspline<double> field(
                std::make_shared<const BsplineSpace<double>>(std::move(directions)),
                Bspline<double>::net_type(std::span<const double>(values),
                                          std::span<const std::size_t>(shape)),
                false);

            // Differentiating direction 0.
            const Bspline<double> along_first = derivative<double>(field, 0);
            PANTR_CHECK_MSG(along_first.space_ref().space_ref(1).knots().size() == second.size(),
                            "differentiating direction 0 moved direction 1's knots");
            PANTR_CHECK_MSG(along_first.space_ref().space_ref(1).degree() == degree_second,
                            "differentiating direction 0 moved direction 1's degree");
            const std::vector<double> lowered_first =
                (r0 == 0) ? std::vector<double>(table_first.size() - 1, 0.0)
                          : marsden_coefficients<double>(
                                along_first.space_ref().space_ref(0).knots(), degree_first - 1,
                                r0 - 1);
            const std::span<const double> got_first = along_first.net().values();
            for (std::size_t i = 0; i + 1 < table_first.size(); ++i) {
                for (std::size_t j = 0; j < table_second.size(); ++j) {
                    const double want = static_cast<double>(r0) * lowered_first[i]
                                        * table_second[j];
                    const double bound =
                        gamma_of<double>(8 * degree_second + 16)
                            * (std::abs(static_cast<double>(degree_first)
                                        * (std::abs(table_first[i]) + std::abs(table_first[i + 1]))
                                        / std::abs(first[i + degree_first + 1] - first[i + 1]))
                               * std::abs(table_second[j]))
                        + 16.0 * underflow_floor<double>();
                    PANTR_CHECK_MSG(
                        std::abs(got_first[i * table_second.size() + j] - want) <= bound,
                        "differentiating direction 0 gave a wrong fibre at ("
                            + std::to_string(i) + ", " + std::to_string(j) + ")");
                }
            }

            // Elevating direction 1 alone.
            const std::vector<std::int64_t> increments{0, 2};
            const Bspline<double> raised =
                elevate_degree<double>(field, std::span<const std::int64_t>(increments));
            PANTR_CHECK_MSG(raised.space_ref().space_ref(0).degree() == degree_first,
                            "elevating direction 1 moved direction 0's degree");
            PANTR_CHECK_MSG(raised.space_ref().space_ref(0).knots().size() == first.size(),
                            "elevating direction 1 moved direction 0's knots");
            const std::vector<double> raised_second = marsden_coefficients<double>(
                raised.space_ref().space_ref(1).knots(), degree_second + 2, r1);
            const std::span<const double> got_raised = raised.net().values();
            std::vector<double> magnitudes(table_second.size());
            for (std::size_t j = 0; j < table_second.size(); ++j) {
                magnitudes[j] = std::abs(table_second[j]);
            }
            const std::vector<double> amplification =
                elevate_majorant(degree_second, magnitudes, second, 2);
            const std::int64_t roundings = (2 * degree_second + 1)
                                           + (2 * (degree_second + 2) + 1) + 2
                                           + 3 * (std::min<std::int64_t>(degree_second, 2) + 1);
            for (std::size_t i = 0; i < table_first.size(); ++i) {
                for (std::size_t j = 0; j < raised_second.size(); ++j) {
                    const double want = table_first[i] * raised_second[j];
                    const double bound =
                        gamma_of<double>(roundings) * std::abs(table_first[i]) * amplification[j]
                        + static_cast<double>(roundings) * underflow_floor<double>();
                    PANTR_CHECK_MSG(
                        std::abs(got_raised[i * raised_second.size() + j] - want) <= bound,
                        "elevating direction 1 gave a wrong fibre at (" + std::to_string(i)
                            + ", " + std::to_string(j) + ")");
                }
            }
        }
    }
}

/// Elevation raises every knot's multiplicity by the increment, and nothing else.
///
/// Exact integers and exact knot values, so no tolerance enters. It is the statement the
/// Marsden check leans on -- that file's closed form is evaluated on the vector the code
/// produced -- so it is asserted rather than assumed.
void check_the_elevated_knot_vector() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0};
    const std::int64_t degree = 3;
    for (std::int64_t increment = 1; increment <= 3; ++increment) {
        const std::vector<double> coefficients =
            marsden_coefficients<double>(std::span<const double>(knots), degree, 1);
        const Bspline<double> field =
            curve<double>(std::span<const double>(knots), degree, false, coefficients);
        const std::vector<std::int64_t> increments{increment};
        const Bspline<double> raised =
            elevate_degree<double>(field, std::span<const std::int64_t>(increments));
        const BsplineSpace1D<double>& space = raised.space_ref().space_ref(0);

        // Every distinct knot survives, with its multiplicity raised by the increment.
        std::vector<double> distinct;
        std::vector<std::int64_t> multiplicity;
        for (const double knot : knots) {
            if (distinct.empty() || knot != distinct.back()) {
                distinct.push_back(knot);
                multiplicity.push_back(1);
            } else {
                ++multiplicity.back();
            }
        }
        std::vector<double> want;
        for (std::size_t i = 0; i < distinct.size(); ++i) {
            for (std::int64_t k = 0; k < multiplicity[i] + increment; ++k) {
                want.push_back(distinct[i]);
            }
        }
        const std::span<const double> got = space.knots();
        PANTR_CHECK_MSG(got.size() == want.size(),
                        "elevating by " + std::to_string(increment) + " gave "
                            + std::to_string(got.size()) + " knots and the multiplicity rule "
                            "calls for " + std::to_string(want.size()));
        if (got.size() == want.size()) {
            for (std::size_t i = 0; i < want.size(); ++i) {
                PANTR_CHECK_MSG(got[i] == want[i], "elevated knot " + std::to_string(i)
                                                       + " is " + std::to_string(got[i])
                                                       + " and should be "
                                                       + std::to_string(want[i]));
            }
        }
        PANTR_CHECK_MSG(
            static_cast<std::int64_t>(raised.net().values().size()) == space.num_basis(),
            "the elevated net and the elevated space disagree on the coefficient count");
    }
}

/// Every refusal, by its message.
void check_the_refusals() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0};
    const std::vector<double> coefficients{1.0, 2.0, 3.0, 4.0};
    const Bspline<double> field =
        curve<double>(std::span<const double>(knots), 2, false, coefficients);

    const auto refuses = [](auto&& call, const std::string& fragment, const std::string& what) {
        bool refused = false;
        try {
            call();
        } catch (const std::invalid_argument& error) {
            refused = true;
            PANTR_CHECK_MSG(std::string(error.what()).find(fragment) != std::string::npos,
                            what + ": the message was \"" + error.what() + "\"");
        }
        PANTR_CHECK_MSG(refused, what + ": nothing was refused");
    };

    // The oracle's Layer 1 messages, character for character.
    refuses([&] { static_cast<void>(derivative<double>(field, 1)); },
            "direction must be in [0, 1), got 1.", "an out-of-range direction");
    refuses([&] { static_cast<void>(derivative<double>(field, -1)); },
            "direction must be in [0, 1), got -1.", "a negative direction");

    const std::vector<double> flat_knots{0.0, 0.4, 1.0};
    const std::vector<double> flat{1.0, 2.0};
    const Bspline<double> degree_zero =
        curve<double>(std::span<const double>(flat_knots), 0, false, flat);
    refuses([&] { static_cast<void>(derivative<double>(degree_zero, 0)); },
            "Derivative of a degree-0 B-spline is not defined.", "a degree-0 direction");

    const std::vector<std::int64_t> two{1, 1};
    refuses([&] {
                static_cast<void>(
                    elevate_degree<double>(field, std::span<const std::int64_t>(two)));
            },
            "Number of degree increments (2) must match dimension (1).",
            "the wrong number of increments");
    const std::vector<std::int64_t> negative{-1};
    refuses([&] {
                static_cast<void>(
                    elevate_degree<double>(field, std::span<const std::int64_t>(negative)));
            },
            "Degree increments must be non-negative.", "a negative increment");
    const std::vector<std::int64_t> zero{0};
    refuses([&] {
                static_cast<void>(
                    elevate_degree<double>(field, std::span<const std::int64_t>(zero)));
            },
            "At least one degree increment must be positive.", "an all-zero increment");
    const std::vector<std::int64_t> beyond{60};
    refuses([&] {
                static_cast<void>(
                    elevate_degree<double>(field, std::span<const std::int64_t>(beyond)));
            },
            "Degree elevation to degree 62 in direction 0 needs binomial coefficients up to "
            "C(62, k)",
            "an elevated degree past the binomial envelope");

    // The two declared boundaries.
    const std::vector<double> periodic_knots{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto periodic_space = std::make_shared<const BsplineSpace1D<double>>(
        std::span<const double>(periodic_knots), 2, true, KnotSnapping::merge_near_duplicates);
    const std::vector<double> periodic_coefficients(
        static_cast<std::size_t>(periodic_space->num_basis()), 1.0);
    const Bspline<double> periodic = curve<double>(std::span<const double>(periodic_knots), 2,
                                                   true, periodic_coefficients);
    const std::vector<std::int64_t> one{1};
    refuses([&] {
                static_cast<void>(
                    elevate_degree<double>(periodic, std::span<const std::int64_t>(one)));
            },
            "elevating a periodic direction needs the open-to-periodic conversion",
            "elevating a periodic direction");

    const std::vector<double> unclamped_knots{-0.3, -0.2, -0.1, 0.0,  0.25, 0.5,
                                              0.75, 1.0,  1.1,  1.2, 1.3};
    const std::vector<double> unclamped_coefficients{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const Bspline<double> unclamped = curve<double>(std::span<const double>(unclamped_knots), 3,
                                                    false, unclamped_coefficients);
    refuses([&] {
                static_cast<void>(
                    elevate_degree<double>(unclamped, std::span<const std::int64_t>(one)));
            },
            "elevating a direction whose knot vector is not clamped",
            "elevating an unclamped direction");
    // And the primitive refuses it too, so a caller reaching past the field entry point
    // cannot perform the read either.
    refuses([&] {
                static_cast<void>(degree_elevate_1d<double>(
                    3, std::span<const double>(unclamped_coefficients), 7, 1,
                    std::span<const double>(unclamped_knots), 1));
            },
            "the knot vector must close with 4 equal knots",
            "the elevation kernel on an unclamped vector");

    // A rational field. Unreachable from Python, which routes a rational derivative to the
    // oracle, and reachable by a C++ caller holding the flag.
    const auto rational_space = std::make_shared<const BsplineSpace1D<double>>(
        std::span<const double>(knots), 2, false, KnotSnapping::merge_near_duplicates);
    std::vector<std::shared_ptr<const BsplineSpace1D<double>>> rational_directions{
        rational_space};
    const std::vector<double> weighted{1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0};
    const std::vector<std::size_t> rational_shape{std::size_t{4}, std::size_t{2}};
    const Bspline<double> rational(
        std::make_shared<const BsplineSpace<double>>(std::move(rational_directions)),
        Bspline<double>::net_type(std::span<const double>(weighted),
                                  std::span<const std::size_t>(rational_shape)),
        true);
    refuses([&] { static_cast<void>(derivative<double>(rational, 0)); },
            "the derivative of a rational B-spline is not part of pantr's C++ core",
            "the derivative of a rational field");
}

/// The primitives refuse a shape they cannot serve.
void check_the_primitives_refuse_a_bad_shape() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0};
    const std::vector<double> values{1.0, 2.0, 3.0, 4.0};

    const auto refuses = [](auto&& call, const std::string& what) {
        bool refused = false;
        try {
            call();
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        PANTR_CHECK_MSG(refused, what + " was accepted");
    };

    refuses([&] {
                static_cast<void>(derivative_along_axis<double>(
                    std::span<const double>(knots), 0, std::span<const double>(values), 4, 1, 1));
            },
            "a degree-0 hodograph");
    refuses([&] {
                static_cast<void>(derivative_along_axis<double>(
                    std::span<const double>(knots), 2, std::span<const double>(values), 4, -1, 1));
            },
            "a negative extent");
    refuses([&] {
                static_cast<void>(derivative_along_axis<double>(
                    std::span<const double>(knots), 2, std::span<const double>(values), 3, 1, 1));
            },
            "a buffer that does not match its shape");
    refuses([&] {
                static_cast<void>(derivative_along_axis<double>(
                    std::span<const double>(knots), 2, std::span<const double>(values), 1, 1, 1));
            },
            "a direction with one coefficient");
    refuses([&] {
                static_cast<void>(degree_elevate_1d<double>(
                    2, std::span<const double>(values), 4, 1, std::span<const double>(knots), 0));
            },
            "a zero increment");
    refuses([&] {
                static_cast<void>(degree_elevate_1d<double>(
                    2, std::span<const double>(values), 4, 2, std::span<const double>(knots), 1));
            },
            "a rank the buffer cannot hold");
}

}  // namespace

int main() {
    check_the_hodograph_is_the_analytic_derivative<double>("float64");
    check_the_hodograph_is_the_analytic_derivative<float>("float32");
    check_a_zero_width_span_is_left_alone();
    check_the_periodic_hodograph();
    check_the_elevation_reproduces_the_polynomial<double>("float64");
    check_the_elevation_reproduces_the_polynomial<float>("float32");
    check_the_elevated_knot_vector();
    check_the_axis_sweeps_touch_only_their_own_axis();
    check_the_refusals();
    check_the_primitives_refuse_a_bad_shape();
    return pantr::test::summary("test_bspline_degree");
}
