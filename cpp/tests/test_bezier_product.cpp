/// \file
/// Mathematical properties of `pantr::bezier::multiply`, `pantr::bezier::compose`
/// and their table-sizing helpers, from `pantr/bezier/product.hpp`.
///
/// ## This is NOT a parity test
///
/// `tests/parity/test_bezier_product.py` compares both entry points against
/// their NumPy/Numba oracles. Nothing here repeats that comparison, and nothing
/// here compares the two backends at all. Every assertion below is against a
/// property the mathematics fixes exactly, against an independently-derived
/// bound on the floating-point rounding a computation carries, or against a
/// second computation of the same quantity by a different route (a division
/// where the code multiplies, or `evaluate` where the code did not evaluate at
/// all) -- the same discipline `test_bezier_shape.cpp` and
/// `test_bezier_degree.cpp` already use, and for the same reason: a comparison
/// against an oracle cannot catch a bug the port and the oracle share.
///
/// ## Where every tolerance comes from
///
/// Three related budgets recur, all built from Higham's standard bound for a
/// chain of `n` relative perturbations of size at most `eps` each, composed in
/// any order: `gamma_n = n*eps / (1 - n*eps)` (Higham, *Accuracy and Stability
/// of Numerical Algorithms*, 2nd ed., Lemma 3.1).
///
///  1. **`multiply`'s own construction budget.** Every intermediate in
///     `bernstein_product_1d` and `bernstein_product_nd` is `T` (product.hpp's
///     file comment), and a coefficient at index `gamma` carries at most
///     `S + 6` roundings, where `S = prod_d (min(p_d, q_d) + 1)` is the largest
///     number of terms landing in one output coefficient. Because the
///     Bernstein product's weights `C(p,i)C(q,j)/C(p+q,k)` are non-negative and
///     sum to one over the terms reaching a given `k` (Vandermonde's identity),
///     the *reachable* magnitude at `gamma` is exactly the same product run on
///     the absolute values of both operands -- call that array `A`. So
///     `|h_gamma - exact_gamma| <= gamma_n(S + 6, eps) * A_gamma`, and
///     contracting with the (non-negative, partition-of-unity) Bernstein basis
///     gives `|h_exact(t) - f_exact(t) g_exact(t)| <= gamma_n(S + 6, eps) *
///     evaluate(A_as_a_bezier, t)`.
///  2. **`evaluate`'s own contraction budget**, `gamma_n(stages, eps)` with
///     `stages = sum_d (degree_d + 1)`, exactly as `evaluate.hpp`'s file
///     comment and `test_bezier_evaluate.cpp` derive it.
///  3. **Products of independently-evaluated quantities** carry one further
///     `gamma_n` per multiplication or division committed combining them.
///
/// `check_product_evaluated_equals_product_of_evaluations` composes budgets 1
/// and 2 (once for `h`, once each for `f` and `g`, the latter multiplied
/// together per budget 3). `check_rational_product_projects_correctly`
/// propagates that same numerator budget through the rational quotient.
/// `check_composition_evaluated_matches_outer_at_inner_value` composes an
/// analogous *chain* budget for `compose`, admittedly coarser: see that
/// check's own comment for where the margin is charged.
///
/// Every companion "run on the absolute values" (`A`, `|f|`, `|g|`, `|h|`, ...)
/// is computed in `double` regardless of `T`, because these are *magnitudes
/// bounding the true budget*, not results the port claims to reproduce; running
/// them at `T` would let the companion itself go slack from `T`'s own rounding
/// and silently loosen the bound it is supposed to certify. The `gamma_n`
/// prefactor is still evaluated at `T`'s own epsilon, because that is what
/// actually governs the rounding of the real, `T`-typed computation being
/// bounded.
///
/// ## What each check would catch
///
///  - **P1** (`check_product_evaluated_equals_product_of_evaluations`): a wrong
///    binomial weight, a term routed to the wrong output coefficient, or an
///    accumulation order that silently drifts outside its own stated budget.
///  - **P2** (`check_product_shape_and_rationality`): a wrong output degree, a
///    dropped rank or dimension, or a rationality flag that does not follow
///    "rational whenever either operand is."
///  - **P3** (`check_product_is_commutative_within_budget`): an asymmetric bug
///    that only one operand order exercises -- a swapped `f`/`g` inside a
///    single branch, for instance -- while still asserting a genuine floating-
///    point (not bit-exact) claim, since the two operand orders are different
///    computations.
///  - **P4** (`check_rational_product_projects_correctly`): a wrong
///    numerator/denominator split, or the weight column leaking into the
///    linear part of the product.
///  - **P5** (`check_product_table_order`, `check_product_rejections`): a wrong
///    table-sizing formula, or a validation whose message has drifted from the
///    oracle's.
///  - **C1** (`check_composing_with_identity_reproduces_outer`): a wrong
///    `bernstein_bases_at` recurrence, caught at the one point where the
///    algebra collapses to "reproduce the input," in both the 1-D and the
///    n-D (`bernstein_product_nd`-routed) composition path.
///  - **C2** (`check_composition_evaluated_matches_outer_at_inner_value`): a
///    wrong composition formula in general (not merely at the identity), on
///    configurations exercising every routing combination `use_1d_kernel`
///    chooses between.
///  - **C3** (`check_composition_shape`, `check_composition_rejections`): a
///    wrong composed degree formula, a wrong `composition_table_order` branch,
///    or a validation message that has drifted from the oracle's.
///
/// ## Test-scaffolding binomial tables
///
/// `multiply` and `compose` take their binomial tables as arguments (the
/// library assembles them once, in exact arithmetic, on the Python side; see
/// `product.hpp`'s file comment). A C++-only caller -- this file -- has to
/// build its own, and `math.comb` has no C++ equivalent that reaches the
/// degrees this file exercises. `build_binomial_tables` below does it with a
/// Pascal-triangle recurrence in `long double`, cast down to `T` at the end.
/// **This is test scaffolding, not what the library does**: production tables
/// come from Python's arbitrary-precision `math.comb`, rounded once.
/// `check_binomial_table_matches_bincoeff` checks this scaffolding's low-degree
/// entries against `pantr::core::bincoeff` exactly, so a defective table here
/// cannot silently make every other check in this file pass for the wrong
/// reason.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/evaluate.hpp"
#include "pantr/bezier/product.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

using pantr::span2d;
using pantr::bezier::Bezier;
using pantr::bezier::ControlNet;
using pantr::bezier::evaluate;

/// Alias for the functions under test, matching `test_bezier_shape.cpp`'s
/// convention.
namespace ops = pantr::bezier;

/// Higham's standard bound for a chain of `n` relative perturbations of size at
/// most `eps` each, composed in any order: `gamma_n = n*eps / (1 - n*eps)`
/// (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma
/// 3.1). The same helper every other `test_bezier_*.cpp` file defines for the
/// same reason.
///
/// \param n The number of perturbations composed.
/// \param eps The scalar type's machine epsilon.
/// \return The relative bound `gamma_n`.
double gamma_n(std::size_t n, double eps) {
    const double n_eps = static_cast<double>(n) * eps;
    return n_eps / (1.0 - n_eps);
}

/// The total number of terms `evaluate`'s own contraction sums across every
/// parametric direction: `sum_d (degree_d + 1)`, `evaluate.hpp`'s own `stages`.
///
/// \param degrees One degree per direction.
/// \return The summed extent.
std::size_t stages_of(const std::vector<std::size_t>& degrees) {
    std::size_t total = 0;
    for (const std::size_t d : degrees) {
        total += d + 1;
    }
    return total;
}

/// Build a Bézier from a flat coefficient list and a shape, for brevity below.
///
/// \param values The coefficients, row-major.
/// \param shape The control-net shape, `(extent(0), ..., num_components())`.
/// \param is_rational Whether the last component is a homogeneous weight.
/// \return The Bézier.
template <class T>
Bezier<T> make_bezier(std::vector<T> values, std::vector<std::size_t> shape, bool is_rational) {
    return Bezier<T>(
        ControlNet<T>(std::span<const T>(values), std::span<const std::size_t>(shape)),
        is_rational);
}

/// Evaluate `bez` at a list of parameter points, for brevity below.
///
/// \param bez The Bézier.
/// \param samples One parameter tuple per sample, `bez.dim()` entries each.
/// \return The values, row-major flattened as `(samples.size(), bez.rank())`
///         but returned as a flat vector since every call site here reads a
///         single sample.
template <class T>
std::vector<T> evaluate_at(const Bezier<T>& bez, const std::vector<std::vector<T>>& samples) {
    const std::size_t dim = bez.dim();
    const std::size_t rank = bez.rank();
    std::vector<T> pts(samples.size() * dim);
    for (std::size_t k = 0; k < samples.size(); ++k) {
        for (std::size_t d = 0; d < dim; ++d) {
            pts[(k * dim) + d] = samples[k][d];
        }
    }
    const span2d<const T> pts_view(pts.data(), samples.size(), dim);
    std::vector<T> out(samples.size() * rank);
    const span2d<T> out_view(out.data(), samples.size(), rank);
    evaluate<T>(bez, pts_view, out_view);
    return out;
}

/// Evaluate `bez` at one parameter tuple.
///
/// \param bez The Bézier.
/// \param t The parameter tuple, `bez.dim()` entries.
/// \return The `bez.rank()` values at `t`.
template <class T>
std::vector<T> evaluate_one(const Bezier<T>& bez, const std::vector<T>& t) {
    return evaluate_at<T>(bez, {t});
}

/// The message of the `std::invalid_argument` that `fn` throws.
///
/// \param fn The call to attempt.
/// \return The exception's `what()`, or a marker saying what happened instead.
template <class F>
std::string message_of(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    } catch (...) {
        return "<threw something other than std::invalid_argument>";
    }
    return "<did not throw>";
}

// ---------------------------------------------------------------------------
// A tiny, deterministic PRNG (test scaffolding, not the library).
// ---------------------------------------------------------------------------

/// A 64-bit linear congruential generator, seeded explicitly by every caller.
///
/// Deliberately not `<random>`'s default engine: this file wants a fixed,
/// portable sequence from a fixed seed so a failure is reproducible across
/// machines, and a hand-rolled LCG is a few lines against `<random>`'s
/// unspecified-by-standard engine internals.
class Lcg64 {
  public:
    explicit Lcg64(std::uint64_t seed) : state_(seed | std::uint64_t{1}) {}

    /// The next value, uniform in `[0, 1)`.
    double uniform01() {
        state_ = (state_ * kMultiplier) + kIncrement;
        const std::uint64_t bits = state_ >> 11;
        return static_cast<double>(bits) * kScale;
    }

  private:
    static constexpr std::uint64_t kMultiplier = 6364136223846793005ULL;
    static constexpr std::uint64_t kIncrement = 1442695040888963407ULL;
    static constexpr double kScale = 1.0 / 9007199254740992.0;  // 2^-53

    std::uint64_t state_;
};

/// A pseudo-random value spanning about three decades, `[1e-1, 1e2]` in
/// magnitude, signed.
///
/// \param rng The generator, advanced by two draws.
/// \return The value.
double random_value_three_decades(Lcg64& rng) {
    const double magnitude = std::pow(10.0, -1.0 + (3.0 * rng.uniform01()));
    const double sign = (rng.uniform01() < 0.5) ? -1.0 : 1.0;
    return sign * magnitude;
}

/// A pseudo-random value in `[0, 1]`, for an inner map's coefficients.
///
/// \param rng The generator, advanced by one draw.
/// \return The value.
double random_value_unit(Lcg64& rng) { return rng.uniform01(); }

/// A non-rational Bézier with pseudo-random control points spanning about
/// three decades in magnitude, signed.
///
/// \param degrees The degree in every parametric direction.
/// \param rank The number of value components.
/// \param seed The generator seed; distinct seeds give distinct nets.
/// \return The Bézier.
template <class T>
Bezier<T> random_bezier(const std::vector<std::size_t>& degrees, std::size_t rank,
                        std::uint64_t seed) {
    Lcg64 rng(seed);
    const std::size_t dim = degrees.size();
    std::vector<std::size_t> shape(dim + 1);
    for (std::size_t d = 0; d < dim; ++d) {
        shape[d] = degrees[d] + 1;
    }
    shape.back() = rank;

    std::size_t total = 1;
    for (const std::size_t extent : shape) {
        total *= extent;
    }
    std::vector<T> values(total);
    for (std::size_t i = 0; i < total; ++i) {
        values[i] = static_cast<T>(random_value_three_decades(rng));
    }
    return make_bezier<T>(std::move(values), shape, false);
}

/// A non-rational Bézier with pseudo-random control points in `[0, 1]`, for
/// use as an inner map: the composition budget's absolute-value companion is
/// exact only while the inner map's own coefficients are non-negative.
///
/// \param degrees The degree in every parametric direction.
/// \param rank The number of value components.
/// \param seed The generator seed.
/// \return The Bézier.
template <class T>
Bezier<T> random_unit_bezier(const std::vector<std::size_t>& degrees, std::size_t rank,
                             std::uint64_t seed) {
    Lcg64 rng(seed);
    const std::size_t dim = degrees.size();
    std::vector<std::size_t> shape(dim + 1);
    for (std::size_t d = 0; d < dim; ++d) {
        shape[d] = degrees[d] + 1;
    }
    shape.back() = rank;

    std::size_t total = 1;
    for (const std::size_t extent : shape) {
        total *= extent;
    }
    std::vector<T> values(total);
    for (std::size_t i = 0; i < total; ++i) {
        values[i] = static_cast<T>(random_value_unit(rng));
    }
    return make_bezier<T>(std::move(values), shape, false);
}

/// A rational Bézier with pseudo-random numerator control points spanning
/// about three decades and weights bounded in `[0.5, 2]`, so that no division
/// in `check_rational_product_projects_correctly` amplifies wildly.
///
/// \param degrees The degree in every parametric direction.
/// \param rank The number of value components, weight column excluded.
/// \param seed The generator seed.
/// \return The rational Bézier.
template <class T>
Bezier<T> random_rational_bezier(const std::vector<std::size_t>& degrees, std::size_t rank,
                                 std::uint64_t seed) {
    Lcg64 rng(seed);
    const std::size_t dim = degrees.size();
    std::vector<std::size_t> shape(dim + 1);
    for (std::size_t d = 0; d < dim; ++d) {
        shape[d] = degrees[d] + 1;
    }
    shape.back() = rank + 1;

    std::size_t coefficients = 1;
    for (std::size_t d = 0; d < dim; ++d) {
        coefficients *= shape[d];
    }
    std::vector<T> values(coefficients * (rank + 1));
    for (std::size_t i = 0; i < coefficients; ++i) {
        for (std::size_t s = 0; s < rank; ++s) {
            values[(i * (rank + 1)) + s] = static_cast<T>(random_value_three_decades(rng));
        }
        const double weight = 0.5 + (1.5 * rng.uniform01());  // [0.5, 2]
        values[(i * (rank + 1)) + rank] = static_cast<T>(weight);
    }
    return make_bezier<T>(std::move(values), shape, true);
}

/// A copy of `bez` widened to `double`, with every stored value replaced by
/// its absolute value: the magnitude companion `A`, `|f|`, `|h|`, ... every
/// bounded check below is stated in terms of, and always computed in `double`
/// regardless of the source's storage type. See the file comment.
///
/// \param bez The Bézier to widen and take the absolute value of.
/// \return The companion, non-rational (the weight column, if any, is treated
///         as an ordinary stored value; every caller below only ever applies
///         this to a non-rational operand).
template <class T>
Bezier<double> abs_double_bezier(const Bezier<T>& bez) {
    const std::span<const T> src = bez.net().values();
    std::vector<double> values(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        values[i] = std::abs(static_cast<double>(src[i]));
    }
    const std::span<const std::size_t> shape = bez.net().shape();
    return make_bezier<double>(std::move(values),
                               std::vector<std::size_t>(shape.begin(), shape.end()), false);
}

/// A copy of `bez` widened to `double`, values unchanged.
///
/// \param bez The Bézier to widen.
/// \return The widened Bézier, same rationality flag.
template <class T>
Bezier<double> to_double_bezier(const Bezier<T>& bez) {
    const std::span<const T> src = bez.net().values();
    std::vector<double> values(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        values[i] = static_cast<double>(src[i]);
    }
    const std::span<const std::size_t> shape = bez.net().shape();
    return make_bezier<double>(std::move(values),
                               std::vector<std::size_t>(shape.begin(), shape.end()),
                               bez.is_rational());
}

/// One parameter tuple, widened to `double`.
///
/// \param t The tuple, at `T`.
/// \return The same tuple, at `double`.
template <class T>
std::vector<double> widen(const std::vector<T>& t) {
    return std::vector<double>(t.begin(), t.end());
}

/// A non-rational Bézier holding only the numerator of a rational one: the
/// first `rank` stored components of every coefficient.
///
/// \param rational The rational Bézier.
/// \return The numerator net, rank `rational.rank()`, non-rational.
template <class T>
Bezier<T> extract_numerator(const Bezier<T>& rational) {
    const ControlNet<T>& net = rational.net();
    const std::size_t components = net.num_components();
    const std::size_t rank = components - 1;
    const std::size_t coefficients = net.size() / components;
    std::vector<T> values(coefficients * rank);
    for (std::size_t i = 0; i < coefficients; ++i) {
        for (std::size_t s = 0; s < rank; ++s) {
            values[(i * rank) + s] = net.values()[(i * components) + s];
        }
    }
    std::vector<std::size_t> shape(net.shape().begin(), net.shape().end());
    shape.back() = rank;
    return make_bezier<T>(std::move(values), std::move(shape), false);
}

/// A rank-1, non-rational Bézier holding only the weight column of a rational
/// one.
///
/// \param rational The rational Bézier.
/// \return The weight net, rank 1, non-rational.
template <class T>
Bezier<T> extract_weight(const Bezier<T>& rational) {
    const ControlNet<T>& net = rational.net();
    const std::size_t components = net.num_components();
    const std::size_t coefficients = net.size() / components;
    std::vector<T> values(coefficients);
    for (std::size_t i = 0; i < coefficients; ++i) {
        values[i] = net.values()[(i * components) + components - 1];
    }
    std::vector<std::size_t> shape(net.shape().begin(), net.shape().end());
    shape.back() = 1;
    return make_bezier<T>(std::move(values), std::move(shape), false);
}

// ---------------------------------------------------------------------------
// Test-scaffolding binomial tables (see the file comment).
// ---------------------------------------------------------------------------

/// `C(n, k)` and `1 / C(n, k)` at the storage format, for every `n, k <=
/// order`, built by a Pascal-triangle recurrence in `long double` and cast
/// down at the end.
///
/// Test scaffolding only: the library never assembles these itself (see the
/// file comment and `product.hpp`'s own).
template <class T>
struct BinomialTables {
    std::vector<T> binom;      ///< `(order + 1, order + 1)` row-major, `C(n, k)`.
    std::vector<T> inv_binom;  ///< Likewise, `1 / C(n, k)`.
    std::size_t order = 0;     ///< The largest upper index the tables reach.

    /// A view suitable for `multiply`'s or `compose`'s `binomials` argument.
    [[nodiscard]] span2d<const T> binom_view() const {
        return span2d<const T>(binom.data(), order + 1, order + 1);
    }

    /// A view suitable for `multiply`'s or `compose`'s `inverse_binomials`
    /// argument.
    [[nodiscard]] span2d<const T> inv_view() const {
        return span2d<const T>(inv_binom.data(), order + 1, order + 1);
    }
};

/// Build `BinomialTables<T>` reaching upper index `order`.
///
/// \param order The largest `n` (and `k`) the tables must answer for.
/// \return The tables, `(order + 1, order + 1)` each.
template <class T>
BinomialTables<T> build_binomial_tables(std::size_t order) {
    const std::size_t width = order + 1;
    std::vector<long double> pascal(width * width, 0.0L);
    const auto at_ld = [&](std::size_t n, std::size_t k) -> long double& {
        return pascal[(n * width) + k];
    };
    for (std::size_t n = 0; n <= order; ++n) {
        at_ld(n, 0) = 1.0L;
        for (std::size_t k = 1; k <= n; ++k) {
            const long double left = at_ld(n - 1, k - 1);
            const long double right = (k <= n - 1) ? at_ld(n - 1, k) : 0.0L;
            at_ld(n, k) = left + right;
        }
    }

    BinomialTables<T> tables;
    tables.order = order;
    tables.binom.assign(width * width, T(0));
    tables.inv_binom.assign(width * width, T(0));
    for (std::size_t n = 0; n <= order; ++n) {
        for (std::size_t k = 0; k <= n; ++k) {
            const long double c = at_ld(n, k);
            tables.binom[(n * width) + k] = static_cast<T>(c);
            tables.inv_binom[(n * width) + k] = static_cast<T>(1.0L / c);
        }
    }
    return tables;
}

/// The scaffolding's `C(n, k)` agrees with `pantr::core::bincoeff` exactly for
/// every `n <= 20`: every such binomial is well under `2^53`, so both routes
/// are exact integers and there is no rounding to argue a tolerance over. Pins
/// the scaffolding itself, so a defective table cannot make every other check
/// below pass for the wrong reason.
void check_binomial_table_matches_bincoeff() {
    const BinomialTables<double> tables = build_binomial_tables<double>(20);
    for (int n = 0; n <= 20; ++n) {
        for (int k = 0; k <= n; ++k) {
            const double expected = pantr::core::bincoeff(n, k);
            const double got =
                tables.binom[(static_cast<std::size_t>(n) * 21) + static_cast<std::size_t>(k)];
            PANTR_CHECK_MSG(got == expected,
                            "the test's own Pascal-recurrence binomial table disagreed with "
                            "pantr::core::bincoeff for n <= 20, where both must be exact");
        }
    }
}

// ---------------------------------------------------------------------------
// P1: the product evaluated equals the product of the evaluations.
// ---------------------------------------------------------------------------

/// The per-sample error model for `multiply(f, g, ...)`, composing budgets 1
/// through 3 of the file comment. Reused unchanged for the rational
/// numerator's own budget in `check_rational_product_projects_correctly`,
/// since a rational product's numerator is exactly a non-rational product of
/// the two numerator nets.
template <class T>
struct ProductErrorModel {
    Bezier<T> h;             ///< `multiply<T>(f, g, ...)`.
    Bezier<double> a;        ///< Budget 1's companion, `multiply<double>(|f|, |g|, ...)`.
    Bezier<double> f_abs_d;  ///< Budget 3's companion, `|f|` at `double`.
    Bezier<double> g_abs_d;  ///< Budget 3's companion, `|g|` at `double`.
    Bezier<double> h_abs_d;  ///< Budget 2's companion, `|h|` at `double`.
    double gamma_construction = 0.0;  ///< `gamma_n(S + 6, eps_t)`.
    double gamma_evaluate_h = 0.0;    ///< `gamma_n(E_h, eps_t)`.
    double gamma_evaluate_fg = 0.0;   ///< `gamma_n(E_f + E_g + 1, eps_t)`.

    ProductErrorModel(const Bezier<T>& f, const Bezier<T>& g, span2d<const T> binom_t,
                      span2d<const T> inv_t, span2d<const double> binom_d,
                      span2d<const double> inv_d)
        : h(ops::multiply<T>(f, g, binom_t, inv_t)),
          a(ops::multiply<double>(abs_double_bezier<T>(f), abs_double_bezier<T>(g), binom_d,
                                  inv_d)),
          f_abs_d(abs_double_bezier<T>(f)),
          g_abs_d(abs_double_bezier<T>(g)),
          h_abs_d(abs_double_bezier<T>(h)) {
        const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
        std::size_t s_terms = 1;
        for (std::size_t d = 0; d < f.dim(); ++d) {
            const std::size_t min_pq = std::min(f.degree(d), g.degree(d));
            s_terms *= (min_pq + 1);
        }
        const std::size_t e_h = stages_of(h.degree());
        const std::size_t e_f = stages_of(f.degree());
        const std::size_t e_g = stages_of(g.degree());
        gamma_construction = gamma_n(s_terms + 6, eps_t);
        gamma_evaluate_h = gamma_n(e_h, eps_t);
        gamma_evaluate_fg = gamma_n(e_f + e_g + 1, eps_t);
    }

    /// The bound on `|evaluate(h, t)[s] - evaluate(f, t)[s] * evaluate(g, t)[s]|`
    /// at every component `s`.
    ///
    /// \param t The parameter tuple, at `T`.
    /// \return One bound per component of `h`.
    [[nodiscard]] std::vector<double> bound_at(const std::vector<T>& t) const {
        const std::vector<double> t_d = widen<T>(t);
        const std::vector<double> a_val = evaluate_one<double>(a, t_d);
        const std::vector<double> h_abs_val = evaluate_one<double>(h_abs_d, t_d);
        const std::vector<double> f_abs_val = evaluate_one<double>(f_abs_d, t_d);
        const std::vector<double> g_abs_val = evaluate_one<double>(g_abs_d, t_d);
        const std::size_t rank = h.rank();
        std::vector<double> bound(rank);
        for (std::size_t s = 0; s < rank; ++s) {
            bound[s] = (gamma_construction * a_val[s]) + (gamma_evaluate_h * h_abs_val[s])
                     + (gamma_evaluate_fg * f_abs_val[s] * g_abs_val[s]);
        }
        return bound;
    }
};

/// One parameter tuple per "corner-ish" value, plus a couple of mixed tuples
/// once `dim > 1` so no direction is left untested against the others: `0, 1,
/// 0.9, 0.1` (per the ticket) and two interior values.
///
/// \param dim The number of parametric directions.
/// \return The tuples, each of length `dim`.
std::vector<std::vector<double>> unit_sample_tuples(std::size_t dim) {
    static const std::vector<double> kBase{0.0, 1.0, 0.9, 0.1, 0.37, 0.63};
    std::vector<std::vector<double>> samples;
    for (const double v : kBase) {
        samples.emplace_back(dim, v);
    }
    if (dim > 1) {
        std::vector<double> mixed_a(dim);
        std::vector<double> mixed_b(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            mixed_a[d] = kBase[d % kBase.size()];
            mixed_b[d] = kBase[(d + 2) % kBase.size()];
        }
        samples.push_back(mixed_a);
        samples.push_back(mixed_b);
    }
    return samples;
}

/// One product configuration: the degrees of the two operands.
struct ProductCase {
    std::vector<std::size_t> degrees_f;
    std::vector<std::size_t> degrees_g;
};

/// The configurations `check_product_evaluated_equals_product_of_evaluations`
/// and `check_product_is_commutative_within_budget` share, at ranks 1 and 3:
/// two univariate pairs at unequal small degrees, one with a degree-0 operand
/// (`S = 1`, the minimal overlap), one surface and one volume.
const std::vector<ProductCase>& product_cases() {
    static const std::vector<ProductCase> kCases{
        {{3}, {2}}, {{1}, {5}}, {{0}, {4}}, {{2, 2}, {1, 3}}, {{1, 1, 1}, {2, 1, 2}}};
    return kCases;
}

/// P1: `evaluate(multiply(f, g, ...), t)` equals `evaluate(f, t) *
/// evaluate(g, t)` componentwise, within `ProductErrorModel`'s bound, at
/// every configuration of `product_cases()`, at ranks 1 and 3, at a spread of
/// parameter tuples including the domain's corners. Would catch a wrong
/// binomial weight, a misrouted term, or an accumulation order that drifts
/// past its own stated budget.
///
/// \param label The scalar type's name, for the worst-ratio report.
template <class T>
void check_product_evaluated_equals_product_of_evaluations(const char* label) {
    double worst_ratio = 0.0;
    std::size_t case_index = 0;
    for (const ProductCase& c : product_cases()) {
        for (const std::size_t rank : {std::size_t{1}, std::size_t{3}}) {
            const std::uint64_t seed_f = (1000ULL * case_index) + (10ULL * rank) + 1ULL;
            const std::uint64_t seed_g = (1000ULL * case_index) + (10ULL * rank) + 2ULL;
            const Bezier<T> f = random_bezier<T>(c.degrees_f, rank, seed_f);
            const Bezier<T> g = random_bezier<T>(c.degrees_g, rank, seed_g);

            const std::size_t order = ops::product_table_order<T>(f, g);
            const BinomialTables<T> tables_t = build_binomial_tables<T>(order);
            const BinomialTables<double> tables_d = build_binomial_tables<double>(order);
            const ProductErrorModel<T> model(f, g, tables_t.binom_view(), tables_t.inv_view(),
                                             tables_d.binom_view(), tables_d.inv_view());

            const std::size_t dim = c.degrees_f.size();
            for (const std::vector<double>& sample_d : unit_sample_tuples(dim)) {
                std::vector<T> t(dim);
                for (std::size_t d = 0; d < dim; ++d) {
                    t[d] = static_cast<T>(sample_d[d]);
                }
                const std::vector<T> h_val = evaluate_one<T>(model.h, t);
                const std::vector<T> f_val = evaluate_one<T>(f, t);
                const std::vector<T> g_val = evaluate_one<T>(g, t);
                const std::vector<double> bound = model.bound_at(t);

                for (std::size_t s = 0; s < rank; ++s) {
                    const double diff =
                        std::abs(static_cast<double>(h_val[s])
                                 - (static_cast<double>(f_val[s]) * static_cast<double>(g_val[s])));
                    PANTR_CHECK_MSG(diff <= bound[s],
                                    "the product evaluated did not equal the product of the "
                                    "evaluations within its construction-plus-evaluation budget");
                    if (bound[s] > 0.0) {
                        worst_ratio = std::max(worst_ratio, diff / bound[s]);
                    } else {
                        PANTR_CHECK_MSG(diff == 0.0,
                                        "a zero bound was not met by an exactly zero difference");
                    }
                }
            }
        }
        ++case_index;
    }
    std::printf("P1[%s] worst observed ratio |diff|/bound = %.6e\n", label, worst_ratio);
}

// ---------------------------------------------------------------------------
// P2: shape, degree, rank and rationality.
// ---------------------------------------------------------------------------

/// P2: `multiply` gives degree `p_d + q_d` per direction, `dim` and `rank`
/// unchanged, and is rational exactly when either operand is. Integer facts,
/// asserted exactly. Would catch a wrong output degree, a dropped rank or
/// dimension, or a rationality flag that does not follow "rational whenever
/// either operand is."
template <class T>
void check_product_shape_and_rationality() {
    for (const ProductCase& c : product_cases()) {
        const std::size_t rank = 2;
        const Bezier<T> f = random_bezier<T>(c.degrees_f, rank, 1);
        const Bezier<T> g = random_bezier<T>(c.degrees_g, rank, 2);
        const std::size_t order = ops::product_table_order<T>(f, g);
        const BinomialTables<T> tables = build_binomial_tables<T>(order);
        const Bezier<T> h = ops::multiply<T>(f, g, tables.binom_view(), tables.inv_view());

        std::vector<std::size_t> expected_degree(c.degrees_f.size());
        for (std::size_t d = 0; d < c.degrees_f.size(); ++d) {
            expected_degree[d] = c.degrees_f[d] + c.degrees_g[d];
        }
        PANTR_CHECK(h.degree() == expected_degree);
        PANTR_CHECK(h.dim() == f.dim());
        PANTR_CHECK(h.rank() == rank);
        PANTR_CHECK_MSG(!h.is_rational(), "two non-rational operands gave a rational product");
    }

    // One rational, one not, both orders: rational exactly when either is.
    const std::vector<std::size_t> degrees{2};
    const Bezier<T> plain = random_bezier<T>(degrees, 2, 3);
    const Bezier<T> rational = random_rational_bezier<T>(degrees, 2, 4);
    const std::size_t order = ops::product_table_order<T>(plain, rational);
    const BinomialTables<T> tables = build_binomial_tables<T>(order);

    const Bezier<T> mixed_a =
        ops::multiply<T>(plain, rational, tables.binom_view(), tables.inv_view());
    PANTR_CHECK_MSG(mixed_a.is_rational(),
                    "a non-rational-times-rational product was not reported rational");
    const Bezier<T> mixed_b =
        ops::multiply<T>(rational, plain, tables.binom_view(), tables.inv_view());
    PANTR_CHECK_MSG(mixed_b.is_rational(),
                    "a rational-times-non-rational product was not reported rational");

    const Bezier<T> rational_b = random_rational_bezier<T>(degrees, 2, 5);
    const Bezier<T> both =
        ops::multiply<T>(rational, rational_b, tables.binom_view(), tables.inv_view());
    PANTR_CHECK_MSG(both.is_rational(), "two rational operands did not give a rational product");
}

// ---------------------------------------------------------------------------
// P3: the product is commutative, within the construction budget.
// ---------------------------------------------------------------------------

/// P3: `multiply(f, g)` and `multiply(g, f)` are the same polynomial computed
/// two different ways -- the accumulation order differs -- so they agree
/// coefficient for coefficient within `2 * gamma_n(S + 6, eps) * A_gamma`
/// (twice budget 1, once per operand order), never bit for bit. Would catch an
/// asymmetric bug that only one operand order exercises.
///
/// \param label The scalar type's name, for the worst-ratio report.
template <class T>
void check_product_is_commutative_within_budget(const char* label) {
    double worst_ratio = 0.0;
    std::size_t case_index = 0;
    for (const ProductCase& c : product_cases()) {
        const std::size_t rank = 2;
        const std::uint64_t seed_f = (2000ULL * case_index) + 1ULL;
        const std::uint64_t seed_g = (2000ULL * case_index) + 2ULL;
        const Bezier<T> f = random_bezier<T>(c.degrees_f, rank, seed_f);
        const Bezier<T> g = random_bezier<T>(c.degrees_g, rank, seed_g);

        const std::size_t order = ops::product_table_order<T>(f, g);
        const BinomialTables<T> tables_t = build_binomial_tables<T>(order);
        const BinomialTables<double> tables_d = build_binomial_tables<double>(order);

        const Bezier<T> h1 = ops::multiply<T>(f, g, tables_t.binom_view(), tables_t.inv_view());
        const Bezier<T> h2 = ops::multiply<T>(g, f, tables_t.binom_view(), tables_t.inv_view());
        PANTR_CHECK(h1.degree() == h2.degree());

        const Bezier<double> a = ops::multiply<double>(
            abs_double_bezier<T>(f), abs_double_bezier<T>(g), tables_d.binom_view(),
            tables_d.inv_view());

        double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
        std::size_t s_terms = 1;
        for (std::size_t d = 0; d < f.dim(); ++d) {
            s_terms *= (std::min(f.degree(d), g.degree(d)) + 1);
        }
        const double per_coeff_gamma = 2.0 * gamma_n(s_terms + 6, eps_t);

        const std::span<const T> h1_vals = h1.net().values();
        const std::span<const T> h2_vals = h2.net().values();
        const std::span<const double> a_vals = a.net().values();
        PANTR_CHECK(h1_vals.size() == a_vals.size());
        for (std::size_t i = 0; i < h1_vals.size(); ++i) {
            const double diff =
                std::abs(static_cast<double>(h1_vals[i]) - static_cast<double>(h2_vals[i]));
            const double bound = per_coeff_gamma * a_vals[i];
            PANTR_CHECK_MSG(diff <= bound,
                            "multiply(f, g) and multiply(g, f) disagreed beyond twice the "
                            "construction budget");
            if (bound > 0.0) {
                worst_ratio = std::max(worst_ratio, diff / bound);
            } else {
                PANTR_CHECK_MSG(diff == 0.0, "a zero commutative bound was not met exactly");
            }
        }
        ++case_index;
    }
    std::printf("P3[%s] worst observed ratio |diff|/bound = %.6e\n", label, worst_ratio);
}

// ---------------------------------------------------------------------------
// P4: a rational product projects to the product of the projections.
// ---------------------------------------------------------------------------

/// One P4 configuration: the degrees of the two rational operands and the
/// (shared) rank.
struct RationalProductCase {
    std::vector<std::size_t> degrees_f;
    std::vector<std::size_t> degrees_g;
    std::size_t rank;
};

/// P4: for rational `f`, `g` with weights bounded in `[0.5, 2]`, `h = f * g`
/// satisfies `h_num(t)/h_w(t) == (f_num(t)/f_w(t)) * (g_num(t)/g_w(t))` within
/// `ProductErrorModel`'s numerator budget (applied to the two numerator nets,
/// since a rational product's numerator is exactly the non-rational product
/// of the numerators) propagated through the quotient by `|h_w(t)|`, plus
/// `gamma_n(3, eps)` for the three divisions (`f`'s, `g`'s and `h`'s own
/// rational projection inside `evaluate`) and the one multiplication
/// combining the two projected factors. Would catch a wrong numerator/
/// denominator split or the weight column leaking into the linear part.
///
/// \param label The scalar type's name, for the worst-ratio report.
template <class T>
void check_rational_product_projects_correctly(const char* label) {
    double worst_ratio = 0.0;
    const std::vector<RationalProductCase> cases{
        {{2}, {1}, 2}, {{1, 1}, {2, 1}, 1}, {{0}, {3}, 2}};
    std::size_t case_index = 0;
    for (const RationalProductCase& c : cases) {
        const std::uint64_t seed_f = (3000ULL * case_index) + 1ULL;
        const std::uint64_t seed_g = (3000ULL * case_index) + 2ULL;
        const Bezier<T> f = random_rational_bezier<T>(c.degrees_f, c.rank, seed_f);
        const Bezier<T> g = random_rational_bezier<T>(c.degrees_g, c.rank, seed_g);

        const std::size_t order = ops::product_table_order<T>(f, g);
        const BinomialTables<T> tables_t = build_binomial_tables<T>(order);
        const BinomialTables<double> tables_d = build_binomial_tables<double>(order);
        const Bezier<T> h = ops::multiply<T>(f, g, tables_t.binom_view(), tables_t.inv_view());
        PANTR_CHECK_MSG(h.is_rational(), "two rational operands did not give a rational product");

        const Bezier<T> f_num = extract_numerator<T>(f);
        const Bezier<T> g_num = extract_numerator<T>(g);
        const Bezier<T> h_w = extract_weight<T>(h);
        const ProductErrorModel<T> numerator_model(f_num, g_num, tables_t.binom_view(),
                                                   tables_t.inv_view(), tables_d.binom_view(),
                                                   tables_d.inv_view());

        const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
        const std::size_t dim = c.degrees_f.size();
        for (const std::vector<double>& sample_d : unit_sample_tuples(dim)) {
            std::vector<T> t(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                t[d] = static_cast<T>(sample_d[d]);
            }
            const std::vector<T> h_proj = evaluate_one<T>(h, t);
            const std::vector<T> f_proj = evaluate_one<T>(f, t);
            const std::vector<T> g_proj = evaluate_one<T>(g, t);
            const std::vector<T> h_w_val = evaluate_one<T>(h_w, t);
            const double h_w_double = std::abs(static_cast<double>(h_w_val[0]));
            const std::vector<double> numerator_bound = numerator_model.bound_at(t);

            for (std::size_t s = 0; s < c.rank; ++s) {
                const double rhs =
                    static_cast<double>(f_proj[s]) * static_cast<double>(g_proj[s]);
                const double diff = std::abs(static_cast<double>(h_proj[s]) - rhs);
                const double bound = (numerator_bound[s] / h_w_double)
                                    + (gamma_n(3, eps_t) * std::abs(rhs));
                PANTR_CHECK_MSG(diff <= bound,
                                "a rational product's projection did not equal the product of "
                                "the projections within budget");
                if (bound > 0.0) {
                    worst_ratio = std::max(worst_ratio, diff / bound);
                }
            }
        }
        ++case_index;
    }
    std::printf("P4[%s] worst observed ratio |diff|/bound = %.6e\n", label, worst_ratio);
}

// ---------------------------------------------------------------------------
// P5: product_table_order, and what a bad table does.
// ---------------------------------------------------------------------------

/// P5: `product_table_order` returns `max_d (p_d + q_d)`, exactly, for several
/// configurations; the two operand-compatibility rejections and the two
/// table-too-small rejections carry the oracle's messages verbatim (the first
/// two duplicate `require_compatible_operands`'s own doc comment, restated
/// here because a caller of `product_table_order` alone hits them without
/// ever reaching `multiply`).
template <class T>
void check_product_table_order() {
    for (const ProductCase& c : product_cases()) {
        const Bezier<T> f = random_bezier<T>(c.degrees_f, 1, 10);
        const Bezier<T> g = random_bezier<T>(c.degrees_g, 1, 11);
        std::size_t expected = 0;
        for (std::size_t d = 0; d < c.degrees_f.size(); ++d) {
            const std::size_t summed = c.degrees_f[d] + c.degrees_g[d];
            expected = std::max(expected, summed);
        }
        PANTR_CHECK(ops::product_table_order<T>(f, g) == expected);
    }

    {
        const Bezier<T> a2d = random_bezier<T>({2, 3}, 1, 12);
        const Bezier<T> a1d = random_bezier<T>({2}, 1, 13);
        PANTR_CHECK_MSG(
            message_of([&] { (void)ops::product_table_order<T>(a2d, a1d); })
                == "Operands must have the same dimension. Got 2 and 1.",
            "product_table_order did not reject mismatched dimensions with the expected "
            "message");
    }
    {
        const Bezier<T> rank1 = random_bezier<T>({2}, 1, 14);
        const Bezier<T> rank2 = random_bezier<T>({2}, 2, 15);
        PANTR_CHECK_MSG(
            message_of([&] { (void)ops::product_table_order<T>(rank1, rank2); })
                == "Operands must have the same rank. Got 1 and 2.",
            "product_table_order did not reject mismatched ranks with the expected message");
    }
}

/// P5's second half: `multiply` refuses a table one row short of
/// `product_table_order + 1`, naming which table.
template <class T>
void check_product_rejections() {
    const Bezier<T> f = random_bezier<T>({2}, 1, 20);
    const Bezier<T> g = random_bezier<T>({2}, 1, 21);
    const std::size_t order = ops::product_table_order<T>(f, g);
    const BinomialTables<T> full = build_binomial_tables<T>(order);
    const BinomialTables<T> short_table = build_binomial_tables<T>(order - 1);

    {
        // binomials one row short, inverse_binomials full size.
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::multiply<T>(f, g, short_table.binom_view(), full.inv_view());
            }) == "binomials must have shape at least (" + std::to_string(order + 1) + ", "
                   + std::to_string(order + 1) + "), got (" + std::to_string(order) + ", "
                   + std::to_string(order) + ").",
            "multiply did not reject an undersized binomials table with the expected message");
    }
    {
        // inverse_binomials one row short, binomials full size.
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::multiply<T>(f, g, full.binom_view(), short_table.inv_view());
            }) == "inverse_binomials must have shape at least (" + std::to_string(order + 1)
                   + ", " + std::to_string(order + 1) + "), got (" + std::to_string(order) + ", "
                   + std::to_string(order) + ").",
            "multiply did not reject an undersized inverse_binomials table with the expected "
            "message");
    }
}

// ---------------------------------------------------------------------------
// C1: composing with the identity reproduces the outer map.
// ---------------------------------------------------------------------------

/// The degree-1 identity curve, `{0, 1}`, rank 1: `t -> t`.
///
/// \return The Bézier.
template <class T>
Bezier<T> identity_1d() {
    return make_bezier<T>({T(0), T(1)}, {2, 1}, false);
}

/// The bilinear identity map `(u, v) -> (u, v)`, rank 2, degree `(1, 1)`.
///
/// \return The Bézier.
template <class T>
Bezier<T> identity_2d() {
    // Row-major (2, 2, 2): index (i, j, c) -> i if c == 0, j if c == 1.
    return make_bezier<T>({T(0), T(0), T(0), T(1), T(1), T(0), T(1), T(1)}, {2, 2, 2}, false);
}

/// C1, univariate: composing a degree-`m` outer map with the degree-1
/// identity inner map reproduces the outer map's own control points, within
/// `gamma_n(4, eps)` per the file comment on the recurrence
/// `fl(C(m, i) * fl(1 / C(m, i)))`. Would catch a wrong `bernstein_bases_at`
/// recurrence.
///
/// \param worst_ratio Updated with the worst observed ratio seen so far.
/// \param m The outer map's degree; see the caller for why several are swept.
template <class T>
void check_composing_with_identity_1d(double& worst_ratio, std::size_t m) {
    const std::size_t rank = 2;
    const Bezier<T> outer = random_bezier<T>({m}, rank, 30 + static_cast<std::uint64_t>(m));
    const Bezier<T> inner = identity_1d<T>();
    PANTR_CHECK(inner.rank() == outer.dim());

    const std::size_t order = ops::composition_table_order<T>(outer, inner);
    const BinomialTables<T> tables = build_binomial_tables<T>(order);
    const Bezier<T> composed =
        ops::compose<T>(outer, inner, tables.binom_view(), tables.inv_view());

    PANTR_CHECK(composed.degree() == std::vector<std::size_t>{m});
    PANTR_CHECK(composed.dim() == 1);
    PANTR_CHECK(composed.rank() == outer.rank());

    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::span<const T> outer_vals = outer.net().values();
    const std::span<const T> composed_vals = composed.net().values();
    for (std::size_t i = 0; i <= m; ++i) {
        for (std::size_t c = 0; c < rank; ++c) {
            const double outer_val = static_cast<double>(outer_vals[(i * rank) + c]);
            const double composed_val = static_cast<double>(composed_vals[(i * rank) + c]);
            const double diff = std::abs(composed_val - outer_val);
            const double bound = gamma_n(4, eps_t) * std::abs(outer_val);
            PANTR_CHECK_MSG(diff <= bound,
                            "composing a univariate outer map with the identity did not "
                            "reproduce its control points within budget");
            if (bound > 0.0) {
                worst_ratio = std::max(worst_ratio, diff / bound);
            }
        }
    }
}

/// C1, 2-D: composing a 2-D outer map -- degree `m` in direction 0, degree 0
/// (a single row) in direction 1 -- with the bilinear identity inner map
/// reproduces the outer map's control points, broadcast across the composed
/// shape's new rows in direction 1 (a degree-0 direction composed with the
/// identity commits no rounding of its own: `bernstein_bases_at`'s `m == 0`
/// branch returns the literal `1.0` with no arithmetic). Routes through
/// `bernstein_product_nd` because the inner map is 2-D. Same bound as the
/// univariate case, `gamma_n(4, eps)`, since the only rounding is direction
/// 0's own identity recurrence.
///
/// \param worst_ratio Updated with the worst observed ratio seen so far.
template <class T>
void check_composing_with_identity_2d(double& worst_ratio) {
    const std::size_t m = 4;
    const std::size_t rank = 2;
    const Bezier<T> outer = random_bezier<T>({m, 0}, rank, 31);
    const Bezier<T> inner = identity_2d<T>();
    PANTR_CHECK(inner.rank() == outer.dim());

    const std::size_t order = ops::composition_table_order<T>(outer, inner);
    const BinomialTables<T> tables = build_binomial_tables<T>(order);
    const Bezier<T> composed =
        ops::compose<T>(outer, inner, tables.binom_view(), tables.inv_view());

    PANTR_CHECK(composed.dim() == 2);
    PANTR_CHECK(composed.rank() == outer.rank());
    PANTR_CHECK(composed.degree() == std::vector<std::size_t>({m, m}));

    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::span<const T> outer_vals = outer.net().values();
    const std::span<const T> composed_vals = composed.net().values();
    for (std::size_t i0 = 0; i0 <= m; ++i0) {
        const double outer_val_base = static_cast<double>(outer_vals[i0 * rank]);
        for (std::size_t j = 0; j <= m; ++j) {
            for (std::size_t c = 0; c < rank; ++c) {
                const double outer_val = static_cast<double>(outer_vals[(i0 * rank) + c]);
                const std::size_t flat_h = ((i0 * (m + 1)) + j) * rank;
                const double composed_val = static_cast<double>(composed_vals[flat_h + c]);
                const double diff = std::abs(composed_val - outer_val);
                const double bound = gamma_n(4, eps_t) * std::abs(outer_val);
                PANTR_CHECK_MSG(diff <= bound,
                                "composing a 2-D outer map with the bilinear identity did not "
                                "reproduce its control points, broadcast across the new "
                                "direction, within budget");
                if (bound > 0.0) {
                    worst_ratio = std::max(worst_ratio, diff / bound);
                }
            }
        }
        (void)outer_val_base;
    }
}

/// C1: both the univariate and 2-D identity-composition checks, sharing one
/// worst-ratio report.
///
/// \param label The scalar type's name, for the worst-ratio report.
template <class T>
void check_composing_with_identity_reproduces_outer(const char* label) {
    double worst_ratio = 0.0;
    // Four degrees, and the two large ones are not decoration. `fl(C(m,i) * fl(1/C(m,i)))`
    // is exactly 1 for every `i` at every degree below 11 in `float` and below 17 in
    // `double`, so a case list of small degrees checks this bound only against an exact
    // case -- which is not checking it. Degree 11 is the first that makes it inexact at
    // `float`, 17 the first at `double`, and 18 has the most such `i`. Measured: at 5
    // alone the worst ratio was 0 at `double`.
    for (const std::size_t m : {std::size_t{5}, std::size_t{11}, std::size_t{17},
                                std::size_t{18}}) {
        check_composing_with_identity_1d<T>(worst_ratio, m);
    }
    check_composing_with_identity_2d<T>(worst_ratio);
    std::printf("C1[%s] worst observed ratio |diff|/bound = %.6e\n", label, worst_ratio);
}

// ---------------------------------------------------------------------------
// C2: the composition evaluated equals the outer map at the inner map's
// value.
// ---------------------------------------------------------------------------

/// One C2 configuration.
struct ComposeCase {
    std::vector<std::size_t> outer_degrees;
    std::size_t outer_rank;
    std::vector<std::size_t> inner_degrees;
    std::size_t inner_rank;  ///< Must equal `outer_degrees.size()`.
};

/// C2: `evaluate(compose(outer, inner, ...), t)` equals
/// `evaluate(outer, evaluate(inner, t))`, within a coarse, admittedly
/// over-charged bound. The chain from an input coefficient to an output one
/// is at most `L = max_d m_d + dim_outer` products (power products building
/// each direction's Bernstein basis, plus cross products combining
/// directions); each product commits at most `S_max + 6` roundings by P1's
/// own count, with `S_max` the composed result's own coefficient count (an
/// over-estimate of any single product's term count); the binomial scaling
/// adds one more rounding, and the final accumulation over the outer's own
/// control points adds at most `prod_d (m_d + 1)` terms. So
/// `N = L * (S_max + 6) + 1 + prod_d (m_d + 1)`, and
/// `|composed_gamma - exact_gamma| <= gamma_n(N, eps) * A_gamma`, with `A`
/// `compose`'s own construction run on the absolute value of the outer net
/// (the inner map's coefficients are kept in `[0, 1]`, so its Bernstein bases
/// are non-negative and the absolute value never has to touch it). Added to
/// that: `evaluate`'s own budget for the composed map, for the outer map (at
/// the inner map's -- possibly slightly wrong -- value) and, admittedly
/// coarsely, for the inner map itself. That last term has no natural home in
/// the same units as the other three (an error in the *input* to `outer`
/// is not, in general, an equal-sized error in `outer`'s *output* without a
/// sensitivity/Lipschitz factor this file does not derive), so it is charged
/// at its own scale, summed over every component of the inner map's value,
/// uniformly against every output component -- an admitted, coarse
/// over-estimate rather than a tight propagation, exactly the margin this
/// check's own docstring warns a reader to expect and to report rather than
/// to silently tighten.
///
/// \param label The scalar type's name, for the worst-ratio report.
template <class T>
void check_composition_evaluated_matches_outer_at_inner_value(const char* label) {
    double worst_ratio = 0.0;
    const std::vector<ComposeCase> cases{
        {{2}, 2, {2}, 1},
        {{3}, 1, {1, 1}, 1},
        {{1, 2}, 2, {2}, 2},
        {{1, 2}, 1, {1, 1}, 2},
    };
    std::size_t case_index = 0;
    for (const ComposeCase& c : cases) {
        const std::uint64_t outer_seed = (4000ULL * case_index) + 1ULL;
        const std::uint64_t inner_seed = (4000ULL * case_index) + 2ULL;
        const Bezier<T> outer = random_bezier<T>(c.outer_degrees, c.outer_rank, outer_seed);
        const Bezier<T> inner = random_unit_bezier<T>(c.inner_degrees, c.inner_rank, inner_seed);
        PANTR_CHECK(inner.rank() == outer.dim());

        const std::size_t order = ops::composition_table_order<T>(outer, inner);
        const BinomialTables<T> tables_t = build_binomial_tables<T>(order);
        const BinomialTables<double> tables_d = build_binomial_tables<double>(order);
        const Bezier<T> composed =
            ops::compose<T>(outer, inner, tables_t.binom_view(), tables_t.inv_view());

        const Bezier<double> outer_abs_d = abs_double_bezier<T>(outer);
        const Bezier<double> inner_double = to_double_bezier<T>(inner);
        const Bezier<double> a =
            ops::compose<double>(outer_abs_d, inner_double, tables_d.binom_view(),
                                 tables_d.inv_view());
        const Bezier<double> composed_abs_d = abs_double_bezier<T>(composed);
        const Bezier<double> outer_abs_only = abs_double_bezier<T>(outer);
        const Bezier<double> inner_abs_d = abs_double_bezier<T>(inner);

        std::size_t max_outer_degree = 0;
        std::size_t outer_coeff_count = 1;
        for (const std::size_t d : outer.degree()) {
            max_outer_degree = std::max(max_outer_degree, d);
            outer_coeff_count *= (d + 1);
        }
        std::size_t s_max = 1;
        for (const std::size_t d : composed.degree()) {
            s_max *= (d + 1);
        }
        const std::size_t chain_length = max_outer_degree + outer.dim();
        const std::size_t big_n = (chain_length * (s_max + 6)) + 1 + outer_coeff_count;

        const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
        const double gamma_construction = gamma_n(big_n, eps_t);
        const double gamma_composed = gamma_n(stages_of(composed.degree()), eps_t);
        const double gamma_outer = gamma_n(stages_of(outer.degree()), eps_t);
        const double gamma_inner = gamma_n(stages_of(inner.degree()), eps_t);

        const std::size_t dim_inner = c.inner_degrees.size();
        for (const std::vector<double>& sample_d : unit_sample_tuples(dim_inner)) {
            std::vector<T> t(dim_inner);
            for (std::size_t d = 0; d < dim_inner; ++d) {
                t[d] = static_cast<T>(sample_d[d]);
            }
            const std::vector<T> inner_val = evaluate_one<T>(inner, t);
            const std::vector<T> outer_at_inner = evaluate_one<T>(outer, inner_val);
            const std::vector<T> composed_val = evaluate_one<T>(composed, t);

            const std::vector<double> t_d = widen<T>(t);
            const std::vector<double> a_val = evaluate_one<double>(a, t_d);
            const std::vector<double> composed_abs_val = evaluate_one<double>(composed_abs_d, t_d);
            const std::vector<double> inner_val_d = evaluate_one<double>(inner_double, t_d);
            const std::vector<double> outer_abs_val =
                evaluate_one<double>(outer_abs_only, inner_val_d);
            const std::vector<double> inner_abs_val = evaluate_one<double>(inner_abs_d, t_d);
            double inner_abs_total = 0.0;
            for (const double v : inner_abs_val) {
                inner_abs_total += v;
            }
            const double inner_term = gamma_inner * inner_abs_total;

            for (std::size_t s = 0; s < c.outer_rank; ++s) {
                const double diff = std::abs(static_cast<double>(composed_val[s])
                                             - static_cast<double>(outer_at_inner[s]));
                const double bound = (gamma_construction * a_val[s])
                                    + (gamma_composed * composed_abs_val[s])
                                    + (gamma_outer * outer_abs_val[s]) + inner_term;
                PANTR_CHECK_MSG(diff <= bound,
                                "the composition evaluated did not equal the outer map at the "
                                "inner map's value within its (coarse) budget");
                if (bound > 0.0) {
                    worst_ratio = std::max(worst_ratio, diff / bound);
                }
            }
        }
        ++case_index;
    }
    std::printf("C2[%s] worst observed ratio |diff|/bound = %.6e\n", label, worst_ratio);
}

// ---------------------------------------------------------------------------
// C3: composition shape and refusals.
// ---------------------------------------------------------------------------

/// C3, shape: `dim` is `inner.dim()`, `rank` is `outer.rank()`, and degree in
/// direction `s` is `sum_d m_d * n_s`, exactly.
template <class T>
void check_composition_shape() {
    const Bezier<T> outer = random_bezier<T>({2, 3}, 2, 40);
    const Bezier<T> inner = random_unit_bezier<T>({1, 1}, 2, 41);
    const std::size_t order = ops::composition_table_order<T>(outer, inner);
    const BinomialTables<T> tables = build_binomial_tables<T>(order);
    const Bezier<T> composed =
        ops::compose<T>(outer, inner, tables.binom_view(), tables.inv_view());
    PANTR_CHECK(composed.dim() == inner.dim());
    PANTR_CHECK(composed.rank() == outer.rank());
    PANTR_CHECK(composed.degree() == std::vector<std::size_t>({5, 5}));

    const Bezier<T> outer_1d = random_bezier<T>({4}, 1, 42);
    const Bezier<T> inner_1d = random_unit_bezier<T>({2}, 1, 43);
    const std::size_t order_1d = ops::composition_table_order<T>(outer_1d, inner_1d);
    const BinomialTables<T> tables_1d = build_binomial_tables<T>(order_1d);
    const Bezier<T> composed_1d =
        ops::compose<T>(outer_1d, inner_1d, tables_1d.binom_view(), tables_1d.inv_view());
    PANTR_CHECK(composed_1d.dim() == 1);
    PANTR_CHECK(composed_1d.rank() == 1);
    PANTR_CHECK(composed_1d.degree() == std::vector<std::size_t>{8});
}

/// C3, `composition_table_order`: `max_d m_d` for a univariate inner map,
/// `max(max_d m_d, max_s (sum_d m_d) * n_s)` otherwise, exact for several
/// configurations including one where the two candidates in the max differ
/// in which one wins.
template <class T>
void check_composition_table_order() {
    {
        // Univariate inner: only max_d m_d.
        const Bezier<T> outer = random_bezier<T>({4, 2}, 1, 50);
        const Bezier<T> inner = random_unit_bezier<T>({3}, 1, 51);
        PANTR_CHECK(ops::composition_table_order<T>(outer, inner) == 4);
    }
    {
        // Multi-d inner, second candidate wins: total = 1 + 2 = 3, n_s = 3 both
        // directions, so 3 * 3 = 9 beats max_d m_d = 2.
        const Bezier<T> outer = random_bezier<T>({1, 2}, 1, 52);
        const Bezier<T> inner = random_unit_bezier<T>({3, 3}, 2, 53);
        PANTR_CHECK(ops::composition_table_order<T>(outer, inner) == 9);
    }
    {
        // Multi-d inner, first candidate wins: inner degree 0 in both
        // directions makes the second candidate 0, so max_d m_d = 5 wins.
        const Bezier<T> outer = random_bezier<T>({5, 1}, 1, 54);
        const Bezier<T> inner = random_unit_bezier<T>({0, 0}, 2, 55);
        PANTR_CHECK(ops::composition_table_order<T>(outer, inner) == 5);
    }
}

/// C3, refusals: every documented rejection of `compose`, message asserted
/// verbatim, plus the two exemptions honoured rather than refused.
template <class T>
void check_composition_rejections() {
    const BinomialTables<T> tiny = build_binomial_tables<T>(1);

    {
        const Bezier<T> outer = random_rational_bezier<T>({2}, 1, 60);
        const Bezier<T> inner = random_unit_bezier<T>({2}, 1, 61);
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::compose<T>(outer, inner, tiny.binom_view(), tiny.inv_view());
            }) == "Composition is not supported for rational Béziers (outer is rational).",
            "compose did not reject a rational outer map with the expected message");
    }
    {
        const Bezier<T> outer = random_bezier<T>({2}, 1, 62);
        const Bezier<T> inner = random_rational_bezier<T>({2}, 1, 63);
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::compose<T>(outer, inner, tiny.binom_view(), tiny.inv_view());
            }) == "Composition is not supported for rational Béziers (inner is rational).",
            "compose did not reject a rational inner map with the expected message");
    }
    {
        const Bezier<T> outer = random_bezier<T>({1, 1}, 1, 64);  // dim 2
        const Bezier<T> inner = random_unit_bezier<T>({2}, 1, 65);  // rank 1
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::compose<T>(outer, inner, tiny.binom_view(), tiny.inv_view());
            }) == "Inner Bézier rank (1) must equal outer Bézier parametric dimension (2).",
            "compose did not reject a rank/dimension mismatch with the expected message");
    }
    {
        // Univariate outer of degree 30, univariate inner of degree 3:
        // composed degree 90 exceeds kBincoeffMaxN (61).
        const Bezier<T> outer = random_bezier<T>({30}, 1, 66);
        const Bezier<T> inner = random_unit_bezier<T>({3}, 1, 67);
        const std::string message = message_of([&] {
            (void)ops::compose<T>(outer, inner, tiny.binom_view(), tiny.inv_view());
        });
        const std::string expected_prefix =
            "Composition to degree 90 with a 1D inner Bézier needs binomial coefficients up "
            "to C(90, k), beyond the largest upper index 61";
        PANTR_CHECK_MSG(message.rfind(expected_prefix, 0) == 0,
                        "compose did not reject a degree past the binomial envelope with the "
                        "expected message prefix");
    }
    {
        // Exemption: a degree-1 (or degree-0) univariate outer map forms no
        // product at all, so a very high degree univariate inner map must not
        // throw regardless of the composed degree.
        const Bezier<T> outer = random_bezier<T>({1}, 1, 68);
        const Bezier<T> inner = random_unit_bezier<T>({70}, 1, 69);
        const std::size_t order = ops::composition_table_order<T>(outer, inner);
        const BinomialTables<T> tables = build_binomial_tables<T>(order);
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::compose<T>(outer, inner, tables.binom_view(), tables.inv_view());
            }) == "<did not throw>",
            "compose refused a degree-1 outer map with a very high degree univariate inner "
            "map, which forms no product and must be exempt from the binomial envelope");
    }
    {
        // Exemption: an n-dimensional inner map is never checked against the
        // binomial envelope at all, at any degree (the check only fires for
        // use_1d_kernel). Supply a table sized to the genuine
        // composition_table_order, which for a 2-D inner map does depend on
        // its degree and here comfortably exceeds kBincoeffMaxN.
        const Bezier<T> outer = random_bezier<T>({2, 2}, 1, 70);
        const Bezier<T> inner = random_unit_bezier<T>({20, 20}, 2, 71);
        const std::size_t order = ops::composition_table_order<T>(outer, inner);
        PANTR_CHECK(order > static_cast<std::size_t>(pantr::core::kBincoeffMaxN));
        const BinomialTables<T> tables = build_binomial_tables<T>(order);
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::compose<T>(outer, inner, tables.binom_view(), tables.inv_view());
            }) == "<did not throw>",
            "compose refused an n-dimensional inner map at a degree past kBincoeffMaxN, which "
            "the envelope check does not (and must not) apply to");
    }
}

}  // namespace

int main() {
    check_binomial_table_matches_bincoeff();

    check_product_evaluated_equals_product_of_evaluations<double>("double");
    check_product_evaluated_equals_product_of_evaluations<float>("float");
    check_product_shape_and_rationality<double>();
    check_product_shape_and_rationality<float>();
    check_product_is_commutative_within_budget<double>("double");
    check_product_is_commutative_within_budget<float>("float");
    check_rational_product_projects_correctly<double>("double");
    check_rational_product_projects_correctly<float>("float");
    check_product_table_order<double>();
    check_product_table_order<float>();
    check_product_rejections<double>();
    check_product_rejections<float>();

    check_composing_with_identity_reproduces_outer<double>("double");
    check_composing_with_identity_reproduces_outer<float>("float");
    check_composition_evaluated_matches_outer_at_inner_value<double>("double");
    check_composition_evaluated_matches_outer_at_inner_value<float>("float");
    check_composition_shape<double>();
    check_composition_shape<float>();
    check_composition_table_order<double>();
    check_composition_table_order<float>();
    check_composition_rejections<double>();
    check_composition_rejections<float>();

    return pantr::test::summary("test_bezier_product");
}
