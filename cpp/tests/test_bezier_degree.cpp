/// \file
/// Mathematical properties of Bézier degree elevation, degree reduction, the
/// Bernstein-Gram `L2` norm, and the round-trip reduction error, from
/// `pantr/bezier/degree.hpp`.
///
/// ## These are NOT parity tests
///
/// `tests/parity/test_bezier_degree.py` compares every entry point against its
/// Numba/NumPy oracle. Nothing here repeats that comparison, and nothing here
/// compares the two backends at all. Every assertion below is against a property
/// the mathematics fixes exactly, or against an independently-derived bound on
/// the floating-point rounding a computation carries -- the same discipline
/// `test_bezier_evaluate.cpp` and `test_bezier_root_finding.cpp` already use, and
/// for the same reason: a comparison against an oracle cannot catch a bug the
/// port and the oracle share.
///
/// ## Where the reduction operator and the Gram matrices come from
///
/// `degree.hpp`'s file comment explains why `reduce_degree` and
/// `degree_reduction_error` take a reduction operator and a Bernstein Gram
/// matrix as arguments rather than assembling them: both are solved once in
/// exact rational arithmetic and rounded to `double`, and a faithful port of
/// that assembly needs arbitrary-precision arithmetic this tree does not carry.
/// So the literals below are not invented: each is the output of the Python
/// oracle that assembles it, pasted with the exact command that produced it
/// recorded beside it, in `reduction_operator_3_1()` and `bernstein_gram_2()` /
/// `bernstein_gram_3()`.
///
/// ## Where every tolerance comes from
///
/// `Acc = accumulator_t<T>` is `double` for every kernel `degree.hpp` composes
/// over (`degree_elevate_bezier_1d` and `core::apply_reduction_operator`), for
/// both `T = float` and `T = double` -- `pantr/core/scalar.hpp`'s
/// `accumulator_t` rule. Every bounded check below composes two kinds of budget,
/// in the same additive style `test_bezier_evaluate.cpp` already uses to combine
/// independent rounding sources, from Higham's `gamma_n = n*eps / (1 - n*eps)`
/// (*Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1 and its
/// extension to inner products, eq. (3.4)):
///
///  - **A construction budget**, `elevation_round_trip_bound` and
///    `reduction_round_trip_bound` below, for the rounding a kernel spends
///    building a new set of control points from an old one. Traced to the
///    kernel's own operations, not assumed: `degree_elevate_bezier_1d`
///    (`kernels_1d.hpp`) builds each interior `bezalfs(j, i)` entry from three
///    `core::bincoeff` calls (each a correctly-rounded `double`, one rounding
///    apiece), one division and two multiplications -- six roundings,
///    `gamma_n(6, eps_d)` -- then contracts up to `degree + 1` of them as a
///    `double` inner product, `gamma_n(degree + 1, eps_d)` -- and narrows to `T`
///    on **every** term of that contraction rather than once at the end, because
///    its accumulator is the output array itself. That last part is charged at
///    `eps_t` and dominates at `T = float`; see `elevation_round_trip_bound`. The two literal boundary entries (`bezalfs(0, 0)` and
///    `bezalfs(p, ph)`, both exactly `1.0`) carry none of this, which is also
///    why the endpoint checks below are exact rather than bounded.
///    `core::apply_reduction_operator` (`reduction_operator.hpp`) is the same
///    shape: each operator entry already carries one correctly-rounded
///    conversion from an exact rational (`_interpolating_reduction_operator`'s
///    `float(value)`), `gamma_n(1, eps_d)`, contracted as a `double` inner
///    product over the pre-reduction degree's width, and cast once to `T`.
///  - **`evaluate`'s own contraction budget**, `gamma_n(stages, eps_t)`, exactly
///    as `evaluate.hpp`'s file comment and `test_bezier_evaluate.cpp` derive it,
///    with `stages` the summed `degree + 1` across directions.
///
/// `check_reduction_inverts_elevation` and `check_degree_reduction_error_at_floor`
/// also use one fact read directly off the embedded operator literal rather than
/// derived: `reduction_operator_3_1()`'s interior row is `(-0.25, 0.75, 0.75,
/// -0.25)`, whose absolute values sum to `2.0`, so a perturbation of the input to
/// `reduce_degree` is amplified by at most that factor (`kOp31RowLinfNorm`)
/// through the operator; the elevation matrix has no such factor because every
/// one of its rows is non-negative and sums to exactly `1` (a partition-of-unity
/// fact that follows from Vandermonde's identity, and which this file does NOT
/// re-verify), so it never amplifies.
///
/// `squared_l2_norm`'s own budget composes the Gram matrix's construction
/// (`_bernstein_gram_matrix_1d` computes each entry from exact-integer
/// binomials via one multiplication and two divisions, `gamma_n(3, eps_d)`, per
/// matrix) with the function's own per-axis contraction and final dot product,
/// each a `double` inner product over its own length.
///
/// ## What each check would catch
///
///  - **Elevation is exact as a mapping** (`check_elevation_preserves_mapping`):
///    a wrong `bezalfs` entry, a wrong gather/scatter permutation across
///    directions, or an increment applied to the wrong axis.
///  - **Elevation's endpoints are bit-exact**
///    (`check_elevation_preserves_endpoints_exactly`): the same, at the one
///    place the mathematics gives an exact rather than a bounded answer, so no
///    tolerance could hide a mistake there.
///  - **Elevation raises the right extents** (`check_elevation_raises_extents`):
///    an off-by-one in the new shape, or a flag dropped across the call.
///  - **Reduction inverts elevation** (`check_reduction_inverts_elevation`) and
///    **is at the rounding floor through `degree_reduction_error`**
///    (`check_degree_reduction_error_at_floor`): a wrong operator orientation,
///    a wrong axis, or an extra approximation where the two operations should
///    cancel exactly in exact arithmetic.
///  - **Reduction interpolates the endpoints, bit for bit**
///    (`check_reduction_interpolates_endpoints_exactly`): the operator's own
///    endpoint rows are unit vectors, so this is exact for the same reason
///    elevation's endpoints are.
///  - **`squared_l2_norm` of the constant polynomial 1 is 1**
///    (`check_squared_l2_norm_of_constant`): a Gram matrix whose entries do not
///    sum to one, or a contraction that drops or double-counts an axis.
///    **Narrower than it looks, and deliberately so.** Every row of every
///    Bernstein Gram matrix sums to `1/(n + 1)` whatever the degree, so this
///    check pins one scalar functional of the matrix -- its total entry sum --
///    and cannot see an axis swapped for another of the same degree, nor a wrong
///    matrix that still sums correctly. What it is not carrying is checked
///    elsewhere: the Gram entries' actual values are verified against exact
///    rational integration in `tests/parity/test_bezier_degree.py`, and the
///    per-axis assignment is pinned by the digit-tagged nets the elevation and
///    reduction checks use.
///  - **Rejections** (`check_rejections`): every validation `degree.hpp` states,
///    with the message asserted verbatim, because it mirrors the Python
///    oracle's own text and a reworded message here is a parity failure a
///    type-only check would not notice.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/degree.hpp"
#include "pantr/bezier/evaluate.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

using pantr::span2d;
using pantr::bezier::Bezier;
using pantr::bezier::ControlNet;
using pantr::bezier::degree_reduction_error;
using pantr::bezier::elevate_degree;
using pantr::bezier::evaluate;
using pantr::bezier::reduce_degree;
using pantr::bezier::squared_l2_norm;

/// Higham's standard bound for `n` sequential roundings, each of relative size
/// at most `eps`: `gamma_n = n*eps / (1 - n*eps)` (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1). Covers both a
/// sequential summation of `n` terms and an `n`-term inner product (eq. (3.4)),
/// which is how every use below reads it -- the same helper
/// `test_bezier_evaluate.cpp` defines for the same reason.
///
/// \param n The number of roundings (or terms) composed.
/// \param eps The scalar type's machine epsilon.
/// \return The relative bound `gamma_n`.
double gamma_n(std::size_t n, double eps) {
    const double n_eps = static_cast<double>(n) * eps;
    return n_eps / (1.0 - n_eps);
}

/// The largest absolute value among a Bézier's stored control coefficients.
///
/// \param values The coefficients.
/// \return The maximum absolute value, widened to `double`, or `0.0` if empty.
template <class T>
double max_abs_values(std::span<const T> values) {
    double m = 0.0;
    for (const T v : values) {
        m = std::max(m, std::abs(static_cast<double>(v)));
    }
    return m;
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

/// A non-rational Bézier whose control point at tensor index `(i_0, ...,
/// i_{dim-1})`, component `c`, is tagged with its own index, so a wrong
/// gather/scatter permutation is caught rather than merely a wrong value. Same
/// tagging scheme `test_bezier_evaluate.cpp` uses, duplicated locally.
///
/// \param degrees The degree in every parametric direction.
/// \param rank The number of value components.
/// \return The tagged Bézier.
template <class T>
Bezier<T> make_tagged_bezier(const std::vector<std::size_t>& degrees, std::size_t rank) {
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
    std::vector<std::size_t> idx(shape.size(), 0);
    for (std::size_t flat = 0; flat < total; ++flat) {
        std::size_t rem = flat;
        for (std::size_t d = shape.size(); d-- > 0;) {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        double tag = 1.0;
        double weight = 1.0;
        for (std::size_t d = 0; d < dim; ++d) {
            tag += weight * static_cast<double>(idx[d]);
            weight *= 10.0;
        }
        tag += static_cast<double>(idx.back());
        values[flat] = static_cast<T>(tag);
    }
    return make_bezier<T>(std::move(values), shape, false);
}

/// Row-major flat index into an array of the given shape.
///
/// \param shape The extents, outermost first.
/// \param idx One index per extent.
/// \return The flat offset.
std::size_t flat_index(std::span<const std::size_t> shape, std::span<const std::size_t> idx) {
    std::size_t flat = 0;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        flat = (flat * shape[d]) + idx[d];
    }
    return flat;
}

/// Evaluate `bez` at a list of parameter points, for brevity below.
///
/// \param bez The Bézier.
/// \param samples One parameter tuple per sample, `bez.dim()` entries each.
/// \return The values, row-major `(samples.size(), bez.rank())`.
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

/// The `(3, 1)` endpoint-interpolating reduction operator, generated by:
///
/// ```
/// PYTHONPATH="$(pwd)/src" conda run -n pantr python -c
///   "from pantr.bezier._bezier_degree import _interpolating_reduction_operator as f;
///    print([[repr(float(v)) for v in row] for row in f(3, 1)])"
/// ```
///
/// \return The `(3, 4)` operator, row-major.
std::vector<double> reduction_operator_3_1() {
    return {1.0, 0.0, 0.0, 0.0, -0.25, 0.75, 0.75, -0.25, 0.0, 0.0, 0.0, 1.0};
}

/// The maximum absolute row sum of `reduction_operator_3_1()`, read directly off
/// its literal: the two boundary rows are unit vectors (sum `1.0`), the
/// interior row is `(-0.25, 0.75, 0.75, -0.25)` (absolute sum `2.0`).
constexpr double kOp31RowLinfNorm = 2.0;

/// The degree-2 Bernstein Gram matrix, generated by:
///
/// ```
/// PYTHONPATH="$(pwd)/src" conda run -n pantr python -c
///   "from pantr.bezier._bezier_degree import _bernstein_gram_matrix_1d as g;
///    print([[repr(float(v)) for v in row] for row in g(2)])"
/// ```
///
/// \return The `(3, 3)` matrix, row-major.
std::vector<double> bernstein_gram_2() {
    return {0.2, 0.1, 0.03333333333333333, 0.1, 0.13333333333333333, 0.1, 0.03333333333333333,
            0.1, 0.2};
}

/// The degree-3 Bernstein Gram matrix, generated the same way as
/// `bernstein_gram_2` with `g(3)`.
///
/// \return The `(4, 4)` matrix, row-major.
std::vector<double> bernstein_gram_3() {
    return {0.14285714285714285,   0.07142857142857142,  0.028571428571428574,
            0.0071428571428571435, 0.07142857142857142,  0.08571428571428572,
            0.0642857142857143,    0.028571428571428574, 0.028571428571428574,
            0.0642857142857143,    0.08571428571428572,  0.07142857142857142,
            0.0071428571428571435, 0.028571428571428574, 0.07142857142857142,
            0.14285714285714285};
}

/// The absolute error `degree_elevate_bezier_1d` (`kernels_1d.hpp`) commits
/// constructing one direction's elevated control points, bounded relative to
/// the input's own magnitude. See the file comment for the six-rounding count
/// on each interior `bezalfs` entry and the `degree + 1`-term inner product
/// that contracts them.
///
/// \param base_degree The degree before elevation, in that direction.
/// \param eps_t The storage type's machine epsilon.
/// \param max_abs_ctrl The largest absolute control-point magnitude entering
///        the elevation.
/// \return The absolute error bound.
double elevation_round_trip_bound(std::size_t base_degree, double eps_t, double max_abs_ctrl) {
    const double eps_d = std::numeric_limits<double>::epsilon();
    constexpr std::size_t kElevationEntryRoundings = 6;
    // The last term is at `eps_t`, once per accumulated term rather than once in
    // total, and that correction is the whole point of this line. An earlier version
    // charged `gamma_n(1, eps_t)`, on the reading that the kernel accumulates in a
    // `double` local and narrows once. It does not: `degree_elevate_bezier_1d` writes
    // `at(out, iz, r)`, which is a `T&`, so every term is read back at `T`, widened,
    // added and narrowed again -- the "each `+=` computes wide and rounds narrow"
    // pattern `kernels_1d.hpp`'s own file comment describes for this kernel. At
    // `T = float` the omitted roundings dominate the bound by about `eps_t / eps_d`.
    // `core::apply_reduction_operator` genuinely does accumulate into a local, which
    // is why `reduction_round_trip_bound` below keeps the single `gamma_n(1, eps_t)`.
    //
    // `base_degree + 1` is used rather than the true chain length `min(p, t) + 1`,
    // which is shorter; the longer count is a safe over-estimate and keeps this
    // function independent of the increment.
    return (gamma_n(kElevationEntryRoundings, eps_d) + gamma_n(base_degree + 1, eps_d)
            + gamma_n(base_degree + 1, eps_t))
           * max_abs_ctrl;
}

/// The absolute error `core::apply_reduction_operator` (`reduction_operator.hpp`)
/// commits constructing one direction's reduced control points. See the file
/// comment: one rounding per operator entry (its own conversion from an exact
/// rational) and an inner product over the pre-reduction degree's width.
///
/// \param degree_before_reduction The degree of the curve being reduced, in
///        that direction.
/// \param eps_t The storage type's machine epsilon.
/// \param max_abs_ctrl The largest absolute control-point magnitude entering
///        the reduction.
/// \return The absolute error bound.
double reduction_round_trip_bound(std::size_t degree_before_reduction, double eps_t,
                                  double max_abs_ctrl) {
    const double eps_d = std::numeric_limits<double>::epsilon();
    constexpr std::size_t kReductionEntryRoundings = 1;
    return (gamma_n(kReductionEntryRoundings, eps_d) + gamma_n(degree_before_reduction + 1, eps_d)
            + gamma_n(1, eps_t))
           * max_abs_ctrl;
}

/// Elevation does not move the curve: `evaluate` at the same parameters agrees
/// before and after, within the elevation's own construction budget (one term
/// per direction actually elevated) plus each side's own `gamma_n(stages,
/// eps_t)` evaluation budget. Exercised in 1-D, in 2-D with every direction
/// elevated, and in 2-D with one direction left at a zero increment.
template <class T>
void check_elevation_preserves_mapping() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());

    struct Case {
        std::vector<std::size_t> degrees;
        std::size_t rank;
        std::vector<std::size_t> increments;
        std::vector<std::vector<T>> samples;
    };

    const std::vector<Case> cases{
        {{4}, 2, {3}, {{T(0.1)}, {T(0.37)}, {T(0.5)}, {T(0.81)}, {T(0.95)}}},
        {{2, 3}, 2, {1, 2}, {{T(0.2), T(0.7)}, {T(0.5), T(0.5)}, {T(0.9), T(0.1)}}},
        {{2, 3}, 1, {2, 0}, {{T(0.3), T(0.6)}, {T(0.8), T(0.2)}}},
    };

    for (const Case& c : cases) {
        const Bezier<T> original = make_tagged_bezier<T>(c.degrees, c.rank);
        const Bezier<T> elevated =
            elevate_degree<T>(original, std::span<const std::size_t>(c.increments));

        const double max_abs_ctrl = max_abs_values<T>(original.net().values());
        double construction_budget = 0.0;
        for (std::size_t d = 0; d < c.degrees.size(); ++d) {
            if (c.increments[d] > 0) {
                construction_budget +=
                    elevation_round_trip_bound(c.degrees[d], eps_t, max_abs_ctrl);
            }
        }
        const std::size_t stages_orig = stages_of(original.degree());
        const std::size_t stages_elev = stages_of(elevated.degree());
        const double bound = (gamma_n(stages_orig, eps_t) * max_abs_ctrl)
                            + (gamma_n(stages_elev, eps_t) * max_abs_ctrl) + construction_budget;

        const std::vector<T> out_orig = evaluate_at<T>(original, c.samples);
        const std::vector<T> out_elev = evaluate_at<T>(elevated, c.samples);
        for (std::size_t i = 0; i < out_orig.size(); ++i) {
            const double diff =
                std::abs(static_cast<double>(out_elev[i]) - static_cast<double>(out_orig[i]));
            PANTR_CHECK_MSG(diff <= bound,
                            "elevation moved the curve beyond its construction-plus-evaluation "
                            "budget");
        }
    }
}

/// Elevation's endpoints are bit-exact: `degree_elevate_bezier_1d`'s boundary
/// weights are the literals `1.0` and `0.0`, so the first and last control
/// point along an elevated direction reproduce the original ones with zero
/// rounding, in 1-D and, since the gather/scatter that carries this to n-D
/// moves data without computing anything, at every fixed index along the other
/// directions of a 2-D case too.
template <class T>
void check_elevation_preserves_endpoints_exactly() {
    {
        const Bezier<T> original = make_tagged_bezier<T>({4}, 3);
        const std::vector<std::size_t> increments{5};
        const Bezier<T> elevated =
            elevate_degree<T>(original, std::span<const std::size_t>(increments));
        const std::size_t rank = original.rank();
        const std::span<const T> orig_vals = original.net().values();
        const std::span<const T> elev_vals = elevated.net().values();
        for (std::size_t c = 0; c < rank; ++c) {
            PANTR_CHECK_MSG(elev_vals[c] == orig_vals[c],
                            "elevation's first control point was not bit-exact");
            PANTR_CHECK_MSG(
                elev_vals[(elevated.degree(0) * rank) + c]
                    == orig_vals[(original.degree(0) * rank) + c],
                "elevation's last control point was not bit-exact");
        }
    }
    {
        const std::vector<std::size_t> degrees{3, 2};
        const std::size_t rank = 2;
        const Bezier<T> original = make_tagged_bezier<T>(degrees, rank);
        const std::vector<std::size_t> increments{2, 0};
        const Bezier<T> elevated =
            elevate_degree<T>(original, std::span<const std::size_t>(increments));

        const std::vector<std::size_t> orig_shape{degrees[0] + 1, degrees[1] + 1, rank};
        const std::vector<std::size_t> elev_shape{elevated.degree(0) + 1, degrees[1] + 1, rank};
        const std::span<const T> orig_vals = original.net().values();
        const std::span<const T> elev_vals = elevated.net().values();

        for (std::size_t i1 = 0; i1 <= degrees[1]; ++i1) {
            for (std::size_t c = 0; c < rank; ++c) {
                const std::vector<std::size_t> orig_first{0, i1, c};
                const std::vector<std::size_t> elev_first{0, i1, c};
                PANTR_CHECK_MSG(
                    elev_vals[flat_index(elev_shape, elev_first)]
                        == orig_vals[flat_index(orig_shape, orig_first)],
                    "elevation's first slice along an elevated axis was not bit-exact");

                const std::vector<std::size_t> orig_last{degrees[0], i1, c};
                const std::vector<std::size_t> elev_last{elevated.degree(0), i1, c};
                PANTR_CHECK_MSG(
                    elev_vals[flat_index(elev_shape, elev_last)]
                        == orig_vals[flat_index(orig_shape, orig_last)],
                    "elevation's last slice along an elevated axis was not bit-exact");
            }
        }
    }
}

/// Elevation raises exactly the requested extents: the elevated direction's
/// degree grows by its increment, every other direction's degree is unchanged,
/// and `dim`, `rank` and `is_rational` are preserved -- for both a
/// non-rational and a rational Bézier.
template <class T>
void check_elevation_raises_extents() {
    const std::vector<std::size_t> degrees{2, 3, 1};
    const Bezier<T> original = make_tagged_bezier<T>(degrees, 2);
    const std::vector<std::size_t> increments{3, 0, 2};
    const Bezier<T> elevated =
        elevate_degree<T>(original, std::span<const std::size_t>(increments));

    PANTR_CHECK(elevated.degree() == std::vector<std::size_t>({5, 3, 3}));
    PANTR_CHECK(elevated.degree(0) == 5);
    PANTR_CHECK(elevated.degree(1) == 3);
    PANTR_CHECK(elevated.degree(2) == 3);
    PANTR_CHECK(elevated.dim() == original.dim());
    PANTR_CHECK(elevated.rank() == original.rank());
    PANTR_CHECK(elevated.is_rational() == original.is_rational());

    const Bezier<T> rational_original =
        make_bezier<T>({T(1), T(2), T(3), T(4), T(5), T(6)}, {3, 2}, true);
    const std::vector<std::size_t> rat_increments{2};
    const Bezier<T> rational_elevated =
        elevate_degree<T>(rational_original, std::span<const std::size_t>(rat_increments));
    PANTR_CHECK(rational_elevated.is_rational());
    PANTR_CHECK(rational_elevated.rank() == rational_original.rank());
    PANTR_CHECK(rational_elevated.degree(0) == 4);
}

/// Reducing an elevated curve recovers the original: elevating degree 2 to
/// degree 3 is an exact embedding, and `reduction_operator_3_1()` is `R@M = I`
/// exactly in exact arithmetic (`degree.hpp`'s file comment; `Gq = M^T Gp M`
/// makes the interpolating reduction the unique zero-residual solution when
/// its input already lies in the lower-degree subspace). What survives is the
/// floating-point construction error of both steps, composed through `R`'s row
/// `L-infinity` norm as the earlier elevation error passes through it: the
/// per-coefficient bound is `kOp31RowLinfNorm * elevation_round_trip_bound(2,
/// ...) + reduction_round_trip_bound(3, ...)`.
template <class T>
void check_reduction_inverts_elevation() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<T> base_values{T(1), T(4), T(-2)};
    const Bezier<T> base = make_bezier<T>(base_values, {3, 1}, false);

    const std::vector<std::size_t> increments{1};
    const Bezier<T> elevated = elevate_degree<T>(base, std::span<const std::size_t>(increments));
    PANTR_CHECK(elevated.degree(0) == 3);

    const std::vector<double> op_data = reduction_operator_3_1();
    const span2d<const double> op(op_data.data(), 3, 4);
    const std::vector<std::size_t> decrements{1};
    const std::vector<span2d<const double>> operators{op};
    const Bezier<T> reduced = reduce_degree<T>(
        elevated, std::span<const std::size_t>(decrements),
        std::span<const span2d<const double>>(operators));
    PANTR_CHECK(reduced.degree(0) == 2);

    const double max_abs_ctrl = max_abs_values<T>(base.net().values());
    const double bound = (kOp31RowLinfNorm * elevation_round_trip_bound(2, eps_t, max_abs_ctrl))
                        + reduction_round_trip_bound(3, eps_t, max_abs_ctrl);

    const std::span<const T> base_vals = base.net().values();
    const std::span<const T> reduced_vals = reduced.net().values();
    for (std::size_t i = 0; i < base_vals.size(); ++i) {
        const double diff =
            std::abs(static_cast<double>(reduced_vals[i]) - static_cast<double>(base_vals[i]));
        PANTR_CHECK_MSG(diff <= bound,
                        "reducing an elevated curve did not recover the original within the "
                        "round-trip construction budget");
    }
}

/// Reduction interpolates the endpoints, bit for bit: `reduction_operator_3_1()`'s
/// first and last rows are unit vectors, so `core::apply_reduction_operator`'s
/// inner product for those two output rows reduces to a single term with
/// coefficient exactly `1.0` -- zero rounding, for any degree-3 input, not only
/// one that came from elevation.
template <class T>
void check_reduction_interpolates_endpoints_exactly() {
    const Bezier<T> original = make_tagged_bezier<T>({3}, 2);
    const std::vector<double> op_data = reduction_operator_3_1();
    const span2d<const double> op(op_data.data(), 3, 4);
    const std::vector<std::size_t> decrements{1};
    const std::vector<span2d<const double>> operators{op};
    const Bezier<T> reduced = reduce_degree<T>(
        original, std::span<const std::size_t>(decrements),
        std::span<const span2d<const double>>(operators));

    const std::span<const T> orig_vals = original.net().values();
    const std::span<const T> reduced_vals = reduced.net().values();
    const std::size_t rank = original.rank();
    for (std::size_t c = 0; c < rank; ++c) {
        PANTR_CHECK_MSG(reduced_vals[c] == orig_vals[c],
                        "reduction's first control point was not bit-exact even though the "
                        "operator's first row is a unit vector");
        PANTR_CHECK_MSG(
            reduced_vals[(reduced.degree(0) * rank) + c]
                == orig_vals[(original.degree(0) * rank) + c],
            "reduction's last control point was not bit-exact even though the operator's last "
            "row is a unit vector");
    }
}

/// `squared_l2_norm` of the constant polynomial 1 (every Bernstein coefficient
/// 1) is 1 exactly, as a formula: the basis is a partition of unity, so those
/// coefficients represent the constant function 1, and its squared `L2` norm
/// over `[0, 1]^d` is the volume of the domain, `1`. The computation's own
/// rounding budget composes the Gram matrix's construction (once per axis) with
/// the per-axis contraction and the final dot product, all `double` inner
/// products. Checked in 1-D and in 2-D (a genuine "unit cube", the 1-D case
/// being the degenerate one).
void check_squared_l2_norm_of_constant() {
    const double eps_d = std::numeric_limits<double>::epsilon();

    {
        const std::vector<double> gram3 = bernstein_gram_3();
        const span2d<const double> g3(gram3.data(), 4, 4);
        const std::vector<double> coeffs(4, 1.0);
        const std::vector<std::size_t> shape{4};
        const std::vector<span2d<const double>> grams{g3};
        const double result =
            squared_l2_norm(std::span<const double>(coeffs), std::span<const std::size_t>(shape),
                            std::span<const span2d<const double>>(grams));
        const double bound = gamma_n(3, eps_d) + (2.0 * gamma_n(4, eps_d));
        PANTR_CHECK_MSG(std::abs(result - 1.0) <= bound,
                        "the squared L2 norm of the constant polynomial 1 was not 1 within the "
                        "Gram contraction's own rounding budget");
    }
    {
        const std::vector<double> gram2 = bernstein_gram_2();
        const std::vector<double> gram3 = bernstein_gram_3();
        const span2d<const double> g2(gram2.data(), 3, 3);
        const span2d<const double> g3(gram3.data(), 4, 4);
        const std::vector<double> coeffs(12, 1.0);
        const std::vector<std::size_t> shape{3, 4};
        const std::vector<span2d<const double>> grams{g2, g3};
        const double result =
            squared_l2_norm(std::span<const double>(coeffs), std::span<const std::size_t>(shape),
                            std::span<const span2d<const double>>(grams));
        const double bound = (2.0 * gamma_n(3, eps_d))  // one Gram-construction budget per axis
                            + gamma_n(3, eps_d) + gamma_n(4, eps_d)  // the two axis contractions
                            + gamma_n(12, eps_d);  // the final length-12 dot product
        PANTR_CHECK_MSG(std::abs(result - 1.0) <= bound,
                        "the squared L2 norm of the constant polynomial 1 over the unit square "
                        "was not 1 within the Gram contraction's own rounding budget");
    }
}

/// `degree_reduction_error` on an exactly-reducible curve is at the rounding
/// floor, not merely small: reducing `bezier` (itself an elevated degree-2
/// curve) and elevating the result back must, in exact arithmetic, reproduce
/// `bezier`, by the same `R@M = I` argument `check_reduction_inverts_elevation`
/// uses. What is left is floating-point construction error from three stages
/// (the elevation that built `bezier`, the reduction inside the function under
/// test, and the re-elevation inside it), composed through `R`'s row
/// `L-infinity` norm exactly as before, giving a per-coefficient bound of
/// `(kOp31RowLinfNorm + 2.0) * elevation_round_trip_bound(2, ...) +
/// reduction_round_trip_bound(3, ...)` -- the `+ 2.0` for the one elevation
/// term that appears unamplified (the curve's own pre-existing error) and the
/// one that appears fresh (the function's internal re-elevation). The Bernstein
/// Gram matrix is symmetric with non-negative entries whose rows sum to
/// `1 / (degree + 1) <= 1` (Perron-Frobenius bounds its spectral radius by that
/// row sum), so a per-coefficient bound `delta` on a length-`n` difference
/// bounds the `L2` norm by `delta * sqrt(n)`.
template <class T>
void check_degree_reduction_error_at_floor() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<T> base_values{T(1), T(4), T(-2)};
    const Bezier<T> base = make_bezier<T>(base_values, {3, 1}, false);
    const std::vector<std::size_t> increments{1};
    const Bezier<T> bezier = elevate_degree<T>(base, std::span<const std::size_t>(increments));

    const std::vector<double> op_data = reduction_operator_3_1();
    const span2d<const double> op(op_data.data(), 3, 4);
    const std::vector<double> gram3 = bernstein_gram_3();
    const span2d<const double> g3(gram3.data(), 4, 4);

    const std::vector<std::size_t> decrements{1};
    const std::vector<span2d<const double>> operators{op};
    const std::vector<span2d<const double>> grams{g3};

    const double error = degree_reduction_error<T>(
        bezier, std::span<const std::size_t>(decrements),
        std::span<const span2d<const double>>(operators),
        std::span<const span2d<const double>>(grams));
    PANTR_CHECK(error >= 0.0);

    const double max_abs_ctrl = max_abs_values<T>(base.net().values());
    const double per_coeff_bound =
        ((kOp31RowLinfNorm + 2.0) * elevation_round_trip_bound(2, eps_t, max_abs_ctrl))
        + reduction_round_trip_bound(3, eps_t, max_abs_ctrl);
    const double error_bound = per_coeff_bound * 2.0;  // sqrt(4 coefficients)
    PANTR_CHECK_MSG(error <= error_bound,
                    "degree_reduction_error on an exactly-reducible curve exceeded the "
                    "round-trip rounding floor");
}

/// The validation messages, asserted verbatim: they mirror the Python oracle's
/// own text (see `degree.hpp`'s parity notes), and a reworded message here is a
/// parity failure a type-only check would not notice.
void check_rejections() {
    const Bezier<double> curve = make_bezier<double>({0.0, 1.0, 2.0, 3.0}, {4, 1}, false);

    {
        const std::vector<std::size_t> increments{0, 0};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)elevate_degree<double>(curve, std::span<const std::size_t>(increments));
            }) == "increments must have one entry per parametric direction (1), got 2.",
            "elevate_degree did not reject a wrong-length increments with the expected message");
    }
    {
        const std::vector<std::size_t> decrements{0, 0};
        const std::vector<span2d<const double>> operators{};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)reduce_degree<double>(curve, std::span<const std::size_t>(decrements),
                                            std::span<const span2d<const double>>(operators));
            }) == "decrements must have one entry per parametric direction (1), got 2.",
            "reduce_degree did not reject a wrong-length decrements with the expected message");
    }
    {
        const std::vector<std::size_t> decrements{0};
        const std::vector<span2d<const double>> operators{};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)reduce_degree<double>(curve, std::span<const std::size_t>(decrements),
                                            std::span<const span2d<const double>>(operators));
            }) == "operators must have one entry per parametric direction (1), got 0.",
            "reduce_degree did not reject a wrong-length operators with the expected message");
    }
    {
        const std::vector<std::size_t> decrements{5};
        std::vector<double> dummy_op(3 * 4, 0.0);
        const span2d<const double> dummy(dummy_op.data(), 3, 4);
        const std::vector<span2d<const double>> operators{dummy};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)reduce_degree<double>(curve, std::span<const std::size_t>(decrements),
                                            std::span<const span2d<const double>>(operators));
            }) == "Degree decrement 5 exceeds the degree 3 in direction 0.",
            "reduce_degree did not reject a decrement exceeding the degree with the expected "
            "message");
    }
    {
        const std::vector<std::size_t> decrements{1};
        std::vector<double> wrong_shape_op(2 * 4, 0.0);
        const span2d<const double> wrong_shape(wrong_shape_op.data(), 2, 4);
        const std::vector<span2d<const double>> operators{wrong_shape};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)reduce_degree<double>(curve, std::span<const std::size_t>(decrements),
                                            std::span<const span2d<const double>>(operators));
            }) == "The reduction operator for direction 0 must have shape (3, 4).",
            "reduce_degree did not reject a wrong operator shape with the expected message");
    }
    {
        const std::vector<std::size_t> decrements{0};
        std::vector<double> op_data(3 * 4, 0.0);
        const span2d<const double> op(op_data.data(), 3, 4);
        const std::vector<span2d<const double>> operators{op};
        const std::vector<span2d<const double>> grams{};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)degree_reduction_error<double>(
                    curve, std::span<const std::size_t>(decrements),
                    std::span<const span2d<const double>>(operators),
                    std::span<const span2d<const double>>(grams));
            }) == "grams must have one entry per parametric direction (1), got 0.",
            "degree_reduction_error did not reject a wrong-length grams with the expected "
            "message");
    }
    {
        const std::vector<std::size_t> decrements{0};
        std::vector<double> op_data(3 * 4, 0.0);
        const span2d<const double> op(op_data.data(), 3, 4);
        const std::vector<span2d<const double>> operators{op};
        std::vector<double> wrong_gram(3 * 3, 0.0);
        const span2d<const double> wrong_gram_view(wrong_gram.data(), 3, 3);
        const std::vector<span2d<const double>> grams{wrong_gram_view};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)degree_reduction_error<double>(
                    curve, std::span<const std::size_t>(decrements),
                    std::span<const span2d<const double>>(operators),
                    std::span<const span2d<const double>>(grams));
            }) == "The Gram matrix for direction 0 must have shape (4, 4).",
            "degree_reduction_error did not reject a wrong Gram shape with the expected message");
    }
    {
        std::vector<double> high_values(61);
        for (std::size_t i = 0; i < 61; ++i) {
            high_values[i] = static_cast<double>(i);
        }
        const Bezier<double> high = make_bezier<double>(high_values, {61, 1}, false);
        const std::vector<std::size_t> increments{2};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)elevate_degree<double>(high, std::span<const std::size_t>(increments));
            }) == "Degree elevation to degree 62 in direction 0 needs binomial coefficients up "
                   "to C(62, k), beyond the largest upper index 61 that pantr's exact-integer "
                   "binomial kernel can compute without an int64 overflow. Past that the "
                   "coefficients wrap silently and the result is corrupted rather than merely "
                   "inaccurate.",
            "elevate_degree did not reject an elevation past the binomial envelope with the "
            "expected message");
    }
}

}  // namespace

int main() {
    check_elevation_preserves_mapping<double>();
    check_elevation_preserves_mapping<float>();
    check_elevation_preserves_endpoints_exactly<double>();
    check_elevation_preserves_endpoints_exactly<float>();
    check_elevation_raises_extents<double>();
    check_elevation_raises_extents<float>();
    check_reduction_inverts_elevation<double>();
    check_reduction_inverts_elevation<float>();
    check_reduction_interpolates_endpoints_exactly<double>();
    check_reduction_interpolates_endpoints_exactly<float>();
    check_squared_l2_norm_of_constant();
    check_degree_reduction_error_at_floor<double>();
    check_degree_reduction_error_at_floor<float>();
    check_rejections();
    return pantr::test::summary("test_bezier_degree");
}
