/// \file
/// Mathematical properties of the nine shape-changing operations in
/// `pantr/bezier/shape.hpp`: reversal, permutation, affine transformation,
/// restriction, splitting, slicing, boundaries and collapse.
///
/// ## These are NOT parity tests
///
/// `tests/parity/test_bezier_shape.py` compares every entry point against its
/// Numba/NumPy oracle. Nothing here repeats that comparison, and nothing here
/// compares the two backends at all. Every assertion below is against a property
/// the mathematics fixes exactly, against a second, independently-rounded
/// computation of the same value, or against an independently-derived bound on
/// the floating-point rounding a computation carries -- the same discipline
/// `test_bezier_evaluate.cpp`, `test_bezier_degree.cpp` and `test_bezier.cpp`
/// already use, and for the same reason: a comparison against an oracle cannot
/// catch a bug the port and the oracle share.
///
/// ## Three families, three kinds of claim (shape.hpp's own split)
///
///  - **Exact rearrangements** -- `reverse` and `permute_directions` -- compute
///    nothing, so every check on them below is bit for bit.
///  - **Compositions over a 1-D kernel** -- `restrict`, `split`, `slice`
///    (`slice_point` is its `dim == 1` twin) and `boundary` -- inherit
///    `restrict_bezier_1d` / `split_bezier_1d` / `slice_bezier_1d`'s own de
///    Casteljau construction error, then are checked against `evaluate`, which
///    carries a second, independent rounding budget of its own.
///  - **A contraction** -- `collapse_along_axis` (and `transform`, algebraically
///    the same shape) -- rounds building the result and is checked the same way.
///
/// ## Where every tolerance comes from
///
/// Three budgets are composed additively below, exactly as
/// `test_bezier_degree.cpp`'s `check_elevation_preserves_mapping` already
/// composes a construction budget with two evaluation budgets:
///
///  1. **A de Casteljau construction budget**, `de_casteljau_pass_bound`.
///     `split_bezier_1d` and `slice_bezier_1d` run one triangular pass of
///     `degree` levels; `restrict_bezier_1d` runs two. `test_bezier.cpp`'s
///     `test_a_split_reconstructs_the_original_curve` and
///     `test_restrict_agrees_with_evaluation_on_the_subinterval` derive and
///     measure the per-pass constant for these same kernels -- `8 * degree * eps
///     * max|c|` per pass, an admitted margin over `4p` roundings that also
///     covers the `pow`-free recurrence and the `eps = 2u` unit-roundoff
///     substitution (measured there at 31-45x margin to degree 55 across 600
///     decades). Reused here rather than rederived, because it is the same
///     kernel at the same accumulator width (`accumulator_t<T>`); only the
///     number of directions and passes composing it changes.
///  2. **`evaluate`'s own contraction budget**, `gamma_n(stages, eps)`, exactly
///     as `evaluate.hpp`'s file comment and `test_bezier_evaluate.cpp` derive
///     it: a summation of `n` terms committed in any order carries a relative
///     perturbation bounded by Higham's `gamma_n = n*eps / (1 - n*eps)`
///     (*Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1),
///     instantiated with `stages = sum_d (degree_d + 1)`.
///  3. **A Bernstein-tabulation-plus-contraction budget**, for
///     `collapse_along_axis` and `transform`, which both round entirely at `T`
///     precision (`shape.hpp`'s `acc` there is `T`, not `accumulator_t<T>`,
///     unlike every kernel in (1)). `collapse_along_axis` calls
///     `tabulate_bernstein_1d`, whose own file comment states its recurrence
///     step is three chained multiplications and no addition; Higham's Lemma
///     3.1 bounds a chain of `k` such relative perturbations by `gamma_k`
///     exactly as it bounds a sum, so the `along`-th basis entry carries at most
///     `gamma_n(3 * along, eps)`. The non-negative, partition-of-unity basis
///     (`test_bernstein.cpp`'s `endpoints_are_exact`, `evaluate.hpp`'s file
///     comment) turns that relative bound on each weight into an absolute bound
///     of about that size on a contraction whose operands are bounded by
///     `max_abs_ctrl`; the `along`-term contraction itself is a further
///     `gamma_n(along, eps)`, Higham's Lemma 3.1 again, this time literally a
///     sum. `transform`'s per-component contraction is the same shape at `n =
///     2`: one rounding casting each matrix entry from `double` to `T`, an
///     `n`-term `T`-precision dot product, and one more rounding for the
///     offset's own cast and its multiply-add into the sum.
///
/// ## What each check would catch
///
///  - **`reverse` is an involution and actually flips the direction**
///    (`check_reverse_is_an_involution`): a wrong mirrored index, or a direction
///    argument routed to the wrong axis.
///  - **`permute_directions` composes, inverts and is a no-op at the identity**
///    (`check_permute_directions_composes_and_inverts`), on a **non-symmetric**
///    degree tuple with a permutation that is not its own inverse: shape.hpp's
///    own file history records a transposed-stride bug that a symmetric case
///    would not have caught, so this file does not repeat that mistake.
///  - **`transform` reproduces the identity and a power-of-two scaling exactly,
///    a general affine map within its contraction budget, and never touches the
///    weight column** (`check_transform_*`): a wrong cast order (`shape.hpp`
///    casts the matrix to `T` before multiplying, not after), a transposed
///    matrix index, or a weight column caught in the linear part.
///  - **`restrict` refuses the full domain** (folded into `check_rejections`,
///    since it is one more verbatim-message case) **and reproduces a genuine
///    sub-box, leaving a full-domain direction untouched**
///    (`check_restrict_reproduces_the_subinterval`): a wrong direction restricted,
///    or the short-circuit for a full-domain direction committing roundings the
///    oracle does not.
///  - **`split` reproduces both halves and the two halves share the split point
///    bit for bit** (`check_split_reproduces_the_curve`): a wrong reparametrisation
///    formula, or a split point copied from the wrong triangle vertex.
///  - **`slice` reproduces a fixed-direction curve, and `slice_point` agrees with
///    `evaluate`'s `dim == 1` delegation** (`check_slice_reproduces_a_fixed_direction`,
///    `check_slice_point_matches_evaluate`): the latter bit for bit at an endpoint,
///    where both kernels carry their own exact shortcut, and bounded elsewhere,
///    because away from an endpoint the two are genuinely different algorithms
///    (a de Casteljau triangle against a Bernstein ratio recurrence) with no
///    reason to agree bit for bit.
///  - **`boundary` forwards to `slice` bit for bit and reproduces a corner
///    exactly** (`check_boundary_matches_slice_and_corners`): a swapped `side`
///    convention, or an off-by-one in which face `slice` is asked for.
///  - **`collapse_along_axis` keeps the right direction and the right
///    `values`-index convention** (`check_collapse_along_axis_keeps_the_right_direction`):
///    an asymmetric degree tuple and asymmetric `values`, so that swapping which
///    entry feeds which direction changes the numeric answer rather than
///    silently passing.
///  - **Every documented rejection, message asserted verbatim**
///    (`check_rejections`): they mirror the Python oracle's own text (see
///    `shape.hpp`'s parity notes on `require_direction`), and a reworded message
///    here is a parity failure a type-only check would not notice.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/evaluate.hpp"
#include "pantr/bezier/shape.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

using pantr::span2d;
using pantr::bezier::Bezier;
using pantr::bezier::ControlNet;
using pantr::bezier::evaluate;

/// Alias for the functions under test, so a call site reads `ops::reverse<T>(...)`
/// rather than colliding, even harmlessly, with `<algorithm>` names like
/// `std::transform` that this file also needs through ADL on a `std::span`
/// argument.
namespace ops = pantr::bezier;

/// Higham's standard bound for a chain of `n` relative perturbations of size at
/// most `eps` each, composed in any order: `gamma_n = n*eps / (1 - n*eps)`
/// (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma
/// 3.1). Applies to an `n`-term sum or inner product and, by the same lemma, to
/// a chain of `n` multiplicative roundings -- the same helper
/// `test_bezier_evaluate.cpp` and `test_bezier_degree.cpp` define for the same
/// reason.
///
/// \param n The number of perturbations composed.
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

/// The de Casteljau construction budget of one `restrict_bezier_1d` /
/// `split_bezier_1d` / `slice_bezier_1d` pass along a direction of the given
/// degree. See the file comment: `8 * degree * eps * max_abs_ctrl`, reused
/// unchanged from `test_bezier.cpp`'s own derivation for these three kernels.
///
/// \param degree The degree of the direction the pass runs along.
/// \param eps_t The storage type's machine epsilon.
/// \param max_abs_ctrl The largest absolute control-point magnitude entering
///        the pass.
/// \return The absolute error bound of one pass.
double de_casteljau_pass_bound(std::size_t degree, double eps_t, double max_abs_ctrl) {
    return 8.0 * static_cast<double>(degree) * eps_t * max_abs_ctrl;
}

/// The construction budget of `transform`'s per-component contraction. See the
/// file comment's item 3: casting `n` matrix entries to `T` (`gamma_n(n, ...)`
/// on the control-point scale), the `n`-term `T`-precision dot product
/// (Higham's Lemma 3.1 again, the same `n`), and the offset's own cast plus its
/// final multiply-add (`gamma_n(2, ...)` on the offset scale).
///
/// \param n The geometric rank (the matrix size).
/// \param eps_t The storage type's machine epsilon.
/// \param max_abs_ctrl The largest absolute control-point magnitude entering
///        the transform.
/// \param max_abs_offset The largest absolute offset component.
/// \return The absolute error bound of one transformed component.
double transform_bound(std::size_t n, double eps_t, double max_abs_ctrl, double max_abs_offset) {
    return (gamma_n(n, eps_t) + gamma_n(n, eps_t)) * max_abs_ctrl + (gamma_n(2, eps_t) * max_abs_offset);
}

/// The construction budget of one `collapse_along_axis` contraction stage over
/// a direction of extent `along`. See the file comment's item 3: the basis
/// row's chained-multiplication rounding, `gamma_n(3 * along, ...)`, plus the
/// `along`-term `T`-precision contraction, `gamma_n(along, ...)`.
///
/// \param along The extent of the collapsed direction (`degree + 1`).
/// \param eps_t The storage type's machine epsilon.
/// \param max_abs_ctrl The largest absolute control-point magnitude entering
///        the stage.
/// \return The absolute error bound of one collapse stage.
double collapse_stage_bound(std::size_t along, double eps_t, double max_abs_ctrl) {
    return (gamma_n(3 * along, eps_t) + gamma_n(along, eps_t)) * max_abs_ctrl;
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
/// gather/scatter permutation or a transposed axis is caught rather than merely
/// a wrong value. Same tagging scheme `test_bezier_evaluate.cpp` and
/// `test_bezier_degree.cpp` use, duplicated locally.
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

/// `reverse` is an involution, exactly, and actually flips the direction.
template <class T>
void check_reverse_is_an_involution() {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<std::size_t> degrees{2, 3, 4};
    const std::size_t rank = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);
    const double max_abs_ctrl = max_abs_values<T>(bez.net().values());
    const std::size_t stages = stages_of(degrees);

    const std::vector<std::vector<T>> samples{
        {T(0.2), T(0.55), T(0.9)}, {T(0.05), T(0.5), T(0.95)}, {T(0.37), T(0.81), T(0.13)}};

    for (std::size_t direction = 0; direction < degrees.size(); ++direction) {
        const Bezier<T> once = ops::reverse<T>(bez, direction);
        const Bezier<T> twice = ops::reverse<T>(once, direction);

        // A pure rearrangement: no arithmetic is performed, so reversing twice
        // must reproduce the original net bit for bit.
        const std::span<const T> orig_vals = bez.net().values();
        const std::span<const T> twice_vals = twice.net().values();
        for (std::size_t i = 0; i < orig_vals.size(); ++i) {
            PANTR_CHECK_MSG(twice_vals[i] == orig_vals[i],
                            "reversing twice did not reproduce the original net bit for bit");
        }

        // reverse actually mirrors the direction: evaluating the original at `t`
        // and the reversed net at `t` with `direction` flipped to `1 - t` agrees,
        // within the sum of the two independent evaluate() rounding budgets --
        // reverse contributes no construction error of its own, so this is
        // exactly check_evaluate_and_lattice_agree's reasoning
        // (test_bezier_evaluate.cpp) for two independently-rounded evaluations
        // of the same mathematics.
        std::vector<std::vector<T>> flipped = samples;
        for (std::vector<T>& s : flipped) {
            s[direction] = T(1) - s[direction];
        }
        const std::vector<T> out_orig = evaluate_at<T>(bez, samples);
        const std::vector<T> out_rev = evaluate_at<T>(once, flipped);
        const double bound = 2.0 * gamma_n(stages, eps) * max_abs_ctrl;
        for (std::size_t i = 0; i < out_orig.size(); ++i) {
            const double diff =
                std::abs(static_cast<double>(out_orig[i]) - static_cast<double>(out_rev[i]));
            PANTR_CHECK_MSG(diff <= bound,
                            "reversing direction " + std::to_string(direction)
                                + " did not mirror the curve within the combined evaluate() "
                                  "budget");
        }
    }
}

/// `permute_directions` composes, inverts, and is a no-op at the identity, all
/// bit for bit -- on a non-symmetric degree tuple with a permutation that is
/// not its own inverse, per the file comment.
template <class T>
void check_permute_directions_composes_and_inverts() {
    const std::vector<std::size_t> degrees{2, 3, 4};
    const std::size_t rank = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);

    const std::vector<std::size_t> perm{1, 2, 0};
    std::vector<std::size_t> inv(perm.size());
    for (std::size_t k = 0; k < perm.size(); ++k) {
        inv[perm[k]] = k;
    }
    PANTR_CHECK_MSG(perm != inv,
                    "the chosen permutation is its own inverse and would not catch a "
                    "transposed implementation");

    const Bezier<T> permuted = ops::permute_directions<T>(bez, std::span<const std::size_t>(perm));
    PANTR_CHECK(permuted.degree()
                == std::vector<std::size_t>({degrees[perm[0]], degrees[perm[1]], degrees[perm[2]]}));

    const Bezier<T> back = ops::permute_directions<T>(permuted, std::span<const std::size_t>(inv));
    const std::span<const T> orig_vals = bez.net().values();
    const std::span<const T> back_vals = back.net().values();
    for (std::size_t i = 0; i < orig_vals.size(); ++i) {
        PANTR_CHECK_MSG(back_vals[i] == orig_vals[i],
                        "permuting by p then by its inverse did not reproduce the original net "
                        "bit for bit");
    }

    const std::vector<std::size_t> identity{0, 1, 2};
    const Bezier<T> same = ops::permute_directions<T>(bez, std::span<const std::size_t>(identity));
    const std::span<const T> same_vals = same.net().values();
    for (std::size_t i = 0; i < orig_vals.size(); ++i) {
        PANTR_CHECK_MSG(same_vals[i] == orig_vals[i],
                        "the identity permutation was not a bit-for-bit no-op");
    }
}

/// `transform` with the identity matrix and a zero offset, and with a
/// power-of-two scaling and a zero offset, is exact -- both bit for bit.
template <class T>
void check_transform_identity_and_scaling_are_exact() {
    const std::vector<std::size_t> degrees{2, 3};
    const std::size_t n = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, n);
    const std::span<const T> orig_vals = bez.net().values();

    {
        // Every product is `x * 1` or `x * 0`, every sum `x + 0`: both exact for
        // any finite operand, so the whole contraction is exact.
        const std::vector<double> identity{1.0, 0.0, 0.0, 1.0};
        const span2d<const double> mat(identity.data(), 2, 2);
        const std::vector<double> zero_offset{0.0, 0.0};
        const Bezier<T> out = ops::transform<T>(bez, mat, std::span<const double>(zero_offset));
        const std::span<const T> out_vals = out.net().values();
        for (std::size_t i = 0; i < orig_vals.size(); ++i) {
            PANTR_CHECK_MSG(out_vals[i] == orig_vals[i],
                            "the identity transform was not exact");
        }
    }
    {
        // A power of two scales the exponent alone, with no rounding barring
        // overflow or underflow, neither of which occurs at these tagged
        // magnitudes; the off-diagonal zero contributes exactly 0, as above.
        const std::vector<double> scale{4.0, 0.0, 0.0, 0.25};
        const span2d<const double> mat(scale.data(), 2, 2);
        const std::vector<double> zero_offset{0.0, 0.0};
        const Bezier<T> out = ops::transform<T>(bez, mat, std::span<const double>(zero_offset));
        const std::span<const T> out_vals = out.net().values();
        const std::size_t coefficients = orig_vals.size() / n;
        for (std::size_t k = 0; k < coefficients; ++k) {
            const std::size_t base = k * n;
            const T expected0 = static_cast<T>(static_cast<double>(orig_vals[base + 0]) * 4.0);
            const T expected1 = static_cast<T>(static_cast<double>(orig_vals[base + 1]) * 0.25);
            PANTR_CHECK_MSG(out_vals[base + 0] == expected0,
                            "a power-of-two scaling was not exact in component 0");
            PANTR_CHECK_MSG(out_vals[base + 1] == expected1,
                            "a power-of-two scaling was not exact in component 1");
        }
    }
}

/// `transform` with a general matrix and offset matches a hand-computed value
/// within its construction budget, and never touches a rational net's weight
/// column.
template <class T>
void check_transform_general_matrix_and_rational_weight() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<std::size_t> degrees{2, 3};
    const std::size_t n = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, n);
    const double max_abs_ctrl = max_abs_values<T>(bez.net().values());

    const std::vector<double> matrix{0.6, -0.2, 0.3, 0.1};
    const std::vector<double> offset{0.25, -1.5};
    const double max_abs_offset = std::max(std::abs(offset[0]), std::abs(offset[1]));
    const span2d<const double> mat(matrix.data(), 2, 2);

    const Bezier<T> out = ops::transform<T>(bez, mat, std::span<const double>(offset));
    const std::span<const T> orig_vals = bez.net().values();
    const std::span<const T> out_vals = out.net().values();
    const std::size_t coefficients = orig_vals.size() / n;
    const double bound = transform_bound(n, eps_t, max_abs_ctrl, max_abs_offset);

    for (std::size_t k = 0; k < coefficients; ++k) {
        const std::size_t base = k * n;
        const double v0 = static_cast<double>(orig_vals[base + 0]);
        const double v1 = static_cast<double>(orig_vals[base + 1]);
        const double expected0 = (v0 * matrix[0]) + (v1 * matrix[1]) + offset[0];
        const double expected1 = (v0 * matrix[2]) + (v1 * matrix[3]) + offset[1];
        const double got0 = static_cast<double>(out_vals[base + 0]);
        const double got1 = static_cast<double>(out_vals[base + 1]);
        PANTR_CHECK_MSG(std::abs(got0 - expected0) <= bound,
                        "a general transform left its construction budget in component 0");
        PANTR_CHECK_MSG(std::abs(got1 - expected1) <= bound,
                        "a general transform left its construction budget in component 1");
    }

    // The rational case: `w (Ax + b) = A(wx) + wb`, so the weight column is
    // copied, never combined into the linear part. Exact, since it is a plain
    // copy in the code.
    const std::vector<T> rat_values{T(2), T(-1), T(4), T(3), T(1), T(6), T(-2), T(0), T(2), T(5),
                                     T(2), T(1)};
    const Bezier<T> rat = make_bezier<T>(rat_values, {2, 2, 3}, true);
    const Bezier<T> rat_out = ops::transform<T>(rat, mat, std::span<const double>(offset));
    const std::span<const T> rat_orig_vals = rat.net().values();
    const std::span<const T> rat_out_vals = rat_out.net().values();
    const std::size_t rat_coefficients = rat_orig_vals.size() / 3;
    for (std::size_t k = 0; k < rat_coefficients; ++k) {
        PANTR_CHECK_MSG(rat_out_vals[(k * 3) + 2] == rat_orig_vals[(k * 3) + 2],
                        "transform touched the weight column of a rational net");
    }
}

/// `restrict` to a genuine sub-box, with one direction left at the full domain,
/// agrees with evaluating the original at the mapped parameter. `restrict`
/// refusing the full domain outright is a rejection, checked verbatim in
/// `check_rejections`.
template <class T>
void check_restrict_reproduces_the_subinterval() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<std::size_t> degrees{2, 3};
    const std::size_t rank = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);
    const double max_abs_ctrl = max_abs_values<T>(bez.net().values());
    const std::size_t stages_orig = stages_of(degrees);

    // Direction 0 stays the full domain (shape.hpp's own short-circuit skips it
    // entirely); direction 1 is genuinely restricted.
    const std::vector<double> lower{0.0, 0.2};
    const std::vector<double> upper{1.0, 0.7};
    const Bezier<T> sub =
        ops::restrict<T>(bez, std::span<const double>(lower), std::span<const double>(upper));
    PANTR_CHECK(sub.degree() == bez.degree());

    const std::size_t stages_sub = stages_of(sub.degree());
    const double bound = (2.0 * de_casteljau_pass_bound(degrees[1], eps_t, max_abs_ctrl))
                        + (gamma_n(stages_sub, eps_t) * max_abs_ctrl)
                        + (gamma_n(stages_orig, eps_t) * max_abs_ctrl);

    const std::vector<std::vector<T>> sub_samples{
        {T(0.3), T(0.25)}, {T(0.6), T(0.75)}, {T(0.9), T(0.5)}};
    for (const std::vector<T>& s : sub_samples) {
        std::vector<T> whole{s[0], static_cast<T>(lower[1] + (static_cast<double>(s[1])
                                                               * (upper[1] - lower[1])))};
        const std::vector<T> from_sub = evaluate_at<T>(sub, {s});
        const std::vector<T> from_whole = evaluate_at<T>(bez, {whole});
        for (std::size_t c = 0; c < rank; ++c) {
            const double diff = std::abs(static_cast<double>(from_sub[c])
                                         - static_cast<double>(from_whole[c]));
            PANTR_CHECK_MSG(diff <= bound,
                            "restrict did not reproduce the sub-box within its construction "
                            "plus evaluation budget");
        }
    }
}

/// `split` reproduces both halves as reparametrisations of the original, and
/// the two halves share the split point bit for bit.
template <class T>
void check_split_reproduces_the_curve() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<std::size_t> degrees{2, 3};
    const std::size_t rank = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);
    const double max_abs_ctrl = max_abs_values<T>(bez.net().values());
    const std::size_t stages = stages_of(degrees);
    const std::vector<T> other{T(0.5), T(0.5)};
    const std::vector<std::size_t> shape{degrees[0] + 1, degrees[1] + 1, rank};

    for (std::size_t direction = 0; direction < degrees.size(); ++direction) {
        for (const T value : {T(0.3), T(0.6)}) {
            std::pair<Bezier<T>, Bezier<T>> halves = ops::split<T>(bez, direction, static_cast<pantr::accumulator_t<T>>(value));
            const Bezier<T>& left = halves.first;
            const Bezier<T>& right = halves.second;
            PANTR_CHECK(left.degree() == bez.degree());
            PANTR_CHECK(right.degree() == bez.degree());

            // A single de Casteljau pass produces both halves, so one
            // construction budget covers either comparison below.
            const double bound = de_casteljau_pass_bound(degrees[direction], eps_t, max_abs_ctrl)
                                + (2.0 * gamma_n(stages, eps_t) * max_abs_ctrl);

            for (const T s : {T(0), T(0.4), T(1)}) {
                std::vector<T> left_pt(2);
                std::vector<T> right_pt(2);
                std::vector<T> whole_left(2);
                std::vector<T> whole_right(2);
                for (std::size_t d = 0; d < 2; ++d) {
                    if (d == direction) {
                        left_pt[d] = s;
                        right_pt[d] = s;
                        whole_left[d] = static_cast<T>(static_cast<double>(value)
                                                       * static_cast<double>(s));
                        whole_right[d] = static_cast<T>(
                            static_cast<double>(value)
                            + ((1.0 - static_cast<double>(value)) * static_cast<double>(s)));
                    } else {
                        left_pt[d] = other[d];
                        right_pt[d] = other[d];
                        whole_left[d] = other[d];
                        whole_right[d] = other[d];
                    }
                }
                const std::vector<T> from_left = evaluate_at<T>(left, {left_pt});
                const std::vector<T> from_whole_left = evaluate_at<T>(bez, {whole_left});
                const std::vector<T> from_right = evaluate_at<T>(right, {right_pt});
                const std::vector<T> from_whole_right = evaluate_at<T>(bez, {whole_right});
                for (std::size_t c = 0; c < rank; ++c) {
                    const double diff_left = std::abs(static_cast<double>(from_left[c])
                                                       - static_cast<double>(from_whole_left[c]));
                    const double diff_right = std::abs(static_cast<double>(from_right[c])
                                                        - static_cast<double>(from_whole_right[c]));
                    PANTR_CHECK_MSG(diff_left <= bound,
                                    "split's left half did not reproduce the original curve "
                                    "within budget");
                    PANTR_CHECK_MSG(diff_right <= bound,
                                    "split's right half did not reproduce the original curve "
                                    "within budget");
                }
            }

            // The shared split point: de Casteljau produces it once
            // (`split_bezier_1d` reads the same final `d[0]` into both the
            // left triangle's last row and the right half's first row), so
            // this is bit for bit.
            const std::span<const T> left_vals = left.net().values();
            const std::span<const T> right_vals = right.net().values();
            const std::size_t other_extent = (direction == 0) ? (degrees[1] + 1) : (degrees[0] + 1);
            for (std::size_t other_idx = 0; other_idx < other_extent; ++other_idx) {
                for (std::size_t c = 0; c < rank; ++c) {
                    std::vector<std::size_t> left_idx(3);
                    std::vector<std::size_t> right_idx(3);
                    if (direction == 0) {
                        left_idx = {degrees[0], other_idx, c};
                        right_idx = {0, other_idx, c};
                    } else {
                        left_idx = {other_idx, degrees[1], c};
                        right_idx = {other_idx, 0, c};
                    }
                    PANTR_CHECK_MSG(
                        left_vals[flat_index(shape, left_idx)]
                            == right_vals[flat_index(shape, right_idx)],
                        "the two halves did not share the split point bit for bit");
                }
            }
        }
    }
}

/// `slice` at a fixed direction agrees with evaluating the original at the
/// corresponding point, within the slice kernel's own construction budget plus
/// two evaluate() budgets.
template <class T>
void check_slice_reproduces_a_fixed_direction() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<std::size_t> degrees{2, 3};
    const std::size_t rank = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);
    const double max_abs_ctrl = max_abs_values<T>(bez.net().values());
    const std::size_t stages_orig = stages_of(degrees);

    for (std::size_t axis = 0; axis < degrees.size(); ++axis) {
        const std::size_t other_dir = 1 - axis;
        for (const T v : {T(0.0), T(0.35), T(1.0)}) {
            const Bezier<T> sliced = ops::slice<T>(bez, axis, static_cast<pantr::accumulator_t<T>>(v));
            PANTR_CHECK(sliced.dim() == 1);
            PANTR_CHECK(sliced.degree(0) == degrees[other_dir]);

            const std::size_t stages_sliced = stages_of(sliced.degree());
            const double bound = de_casteljau_pass_bound(degrees[axis], eps_t, max_abs_ctrl)
                                + (gamma_n(stages_sliced, eps_t) * max_abs_ctrl)
                                + (gamma_n(stages_orig, eps_t) * max_abs_ctrl);

            for (const T t : {T(0.0), T(0.5), T(1.0)}) {
                const std::vector<T> from_sliced = evaluate_at<T>(sliced, {{t}});
                std::vector<T> whole(2);
                whole[axis] = v;
                whole[other_dir] = t;
                const std::vector<T> from_whole = evaluate_at<T>(bez, {whole});
                for (std::size_t c = 0; c < rank; ++c) {
                    const double diff = std::abs(static_cast<double>(from_sliced[c])
                                                 - static_cast<double>(from_whole[c]));
                    PANTR_CHECK_MSG(diff <= bound,
                                    "slice did not reproduce the fixed-direction curve within "
                                    "budget");
                }
            }
        }
    }
}

/// `slice_point` on a 1-D Bézier agrees with `evaluate` there: bit for bit at
/// an endpoint, where both kernels carry their own exact shortcut, and bounded
/// away from one, where the two run genuinely different algorithms with no
/// reason to round the same way.
template <class T>
void check_slice_point_matches_evaluate() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<std::size_t> degrees{3};
    const std::size_t rank = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);
    const double max_abs_ctrl = max_abs_values<T>(bez.net().values());
    const std::size_t stages = stages_of(degrees);

    for (const T v : {T(0), T(1)}) {
        const std::vector<T> sp = ops::slice_point<T>(bez, static_cast<pantr::accumulator_t<T>>(v));
        const std::vector<T> ev = evaluate_at<T>(bez, {{v}});
        for (std::size_t c = 0; c < rank; ++c) {
            PANTR_CHECK_MSG(sp[c] == ev[c],
                            "slice_point and evaluate disagreed bit for bit at an endpoint, "
                            "where both carry an exact shortcut");
        }
    }

    // slice_point runs slice_bezier_1d's de Casteljau triangle, structurally the
    // same single pass as split's left half; evaluate's dim() == 1 delegation
    // runs evaluate_bezier_1d's Bernstein ratio recurrence instead.
    const double bound = de_casteljau_pass_bound(degrees[0], eps_t, max_abs_ctrl)
                        + (gamma_n(stages, eps_t) * max_abs_ctrl);
    // The parameter is `accumulator_t<T>` by design -- narrowing it was the defect this
    // file's own history records -- so the widening is written out rather than left
    // implicit, which clang reports under `-Wdouble-promotion`. It converts from `T` and
    // not from a `double` literal on purpose: at `T = float`, `double(float(0.3))` is
    // 0.30000001192092896 and `double(0.3)` is not, so the shorter spelling would move
    // the value this bound is tested against.
    for (const T v : {T(0.3), T(0.5), T(0.8)}) {
        const std::vector<T> sp = ops::slice_point<T>(bez, static_cast<pantr::accumulator_t<T>>(v));
        const std::vector<T> ev = evaluate_at<T>(bez, {{v}});
        for (std::size_t c = 0; c < rank; ++c) {
            const double diff = std::abs(static_cast<double>(sp[c]) - static_cast<double>(ev[c]));
            PANTR_CHECK_MSG(diff <= bound,
                            "slice_point and evaluate left their combined construction budget "
                            "away from an endpoint");
        }
    }
}

/// `boundary` forwards to `slice` at parameter 0 or 1 bit for bit, and a corner
/// reached by composing two boundary calls reproduces the corresponding control
/// point exactly.
template <class T>
void check_boundary_matches_slice_and_corners() {
    const std::vector<std::size_t> degrees{2, 3};
    const std::size_t rank = 2;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);

    for (std::size_t axis = 0; axis < degrees.size(); ++axis) {
        for (const int side : {0, 1}) {
            const Bezier<T> b = ops::boundary<T>(bez, axis, side);
            const Bezier<T> s = ops::slice<T>(bez, axis, static_cast<pantr::accumulator_t<T>>(side == 0 ? T(0) : T(1)));
            const std::span<const T> b_vals = b.net().values();
            const std::span<const T> s_vals = s.net().values();
            for (std::size_t i = 0; i < b_vals.size(); ++i) {
                PANTR_CHECK_MSG(b_vals[i] == s_vals[i],
                                "boundary did not forward to slice bit for bit");
            }
        }
    }

    // A corner: the Bernstein basis at 0 or 1 is exactly {1, 0, ..., 0} or
    // {0, ..., 0, 1} (test_bernstein.cpp's endpoints_are_exact), and both
    // slice_bezier_1d and slice_point's own endpoint shortcuts return the raw
    // control point with no rounding at all -- so composing two boundary calls
    // down to a point stays exact.
    const std::vector<std::size_t> shape{degrees[0] + 1, degrees[1] + 1, rank};
    for (const int side0 : {0, 1}) {
        for (const int side1 : {0, 1}) {
            const Bezier<T> face = ops::boundary<T>(bez, 0, side0);
            const std::vector<T> corner = ops::slice_point<T>(face, static_cast<pantr::accumulator_t<T>>(side1 == 0 ? T(0) : T(1)));
            for (std::size_t c = 0; c < rank; ++c) {
                const std::vector<std::size_t> idx{side0 == 0 ? std::size_t{0} : degrees[0],
                                                    side1 == 0 ? std::size_t{0} : degrees[1], c};
                PANTR_CHECK_MSG(
                    corner[c] == bez.net().values()[flat_index(shape, idx)],
                    "a corner reached via two boundary calls did not reproduce the control "
                    "point exactly");
            }
        }
    }
}

/// `collapse_along_axis` keeps the requested direction and honours the
/// `values`-index convention: entry `i` is direction `i` below `axis` and
/// direction `i + 1` above it. Checked on an asymmetric degree tuple with
/// asymmetric `values`, so that a reversed convention would move the answer.
template <class T>
void check_collapse_along_axis_keeps_the_right_direction() {
    const double eps_t = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<std::size_t> degrees{2, 3, 4};
    const std::size_t rank = 2;
    const std::size_t axis = 1;
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);
    const double max_abs_ctrl = max_abs_values<T>(bez.net().values());
    const std::size_t stages_orig = stages_of(degrees);

    // entry 0 -> direction 0 (below axis), entry 1 -> direction 2 (above axis).
    const std::vector<T> values{T(0.2), T(0.8)};
    const Bezier<T> collapsed = ops::collapse_along_axis<T>(bez, axis, std::span<const T>(values));
    PANTR_CHECK(collapsed.dim() == 1);
    PANTR_CHECK(collapsed.degree(0) == degrees[axis]);

    const std::size_t stages_collapsed = stages_of(collapsed.degree());
    // The oracle contracts directions 2 then 0 (high to low, skipping axis --
    // shape.hpp's file comment), each stage independent, so the two per-stage
    // budgets are summed rather than merged into one gamma_n.
    double construction_budget = 0.0;
    for (const std::size_t d : {std::size_t{0}, std::size_t{2}}) {
        construction_budget += collapse_stage_bound(degrees[d] + 1, eps_t, max_abs_ctrl);
    }
    const double bound = construction_budget
                        + (gamma_n(stages_collapsed, eps_t) * max_abs_ctrl)
                        + (gamma_n(stages_orig, eps_t) * max_abs_ctrl);

    for (const T t : {T(0), T(0.4), T(1)}) {
        const std::vector<T> from_collapsed = evaluate_at<T>(collapsed, {{t}});
        const std::vector<T> whole{values[0], t, values[1]};
        const std::vector<T> from_whole = evaluate_at<T>(bez, {whole});
        for (std::size_t c = 0; c < rank; ++c) {
            const double diff = std::abs(static_cast<double>(from_collapsed[c])
                                         - static_cast<double>(from_whole[c]));
            PANTR_CHECK_MSG(diff <= bound,
                            "collapse_along_axis did not keep the right direction, honour the "
                            "values convention, or left its construction budget");
        }
    }
}

/// Every documented rejection of `shape.hpp`, message asserted verbatim.
void check_rejections() {
    const Bezier<double> bez2d = make_tagged_bezier<double>({2, 3}, 2);
    const Bezier<double> bez1d = make_tagged_bezier<double>({4}, 2);

    PANTR_CHECK_MSG(
        message_of([&] { (void)ops::reverse<double>(bez2d, 5); })
            == "direction must be in [0, 2), got 5.",
        "reverse did not reject an out-of-range direction with the expected message");

    {
        const std::vector<std::size_t> non_perm{0, 0};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::permute_directions<double>(bez2d, std::span<const std::size_t>(non_perm));
            }) == "permutation must be a permutation of range(2), got [0, 0].",
            "permute_directions did not reject a non-permutation with the expected message");
    }
    {
        std::vector<double> wrong_matrix(3 * 3, 0.0);
        const span2d<const double> mat(wrong_matrix.data(), 3, 3);
        const std::vector<double> offset{0.0, 0.0};
        PANTR_CHECK_MSG(
            message_of([&] { (void)ops::transform<double>(bez2d, mat, std::span<const double>(offset)); })
                == "Transform dimension (3) does not match the geometric rank (2) of the "
                   "control points.",
            "transform did not reject a wrong matrix size with the expected message");
    }
    {
        std::vector<double> identity{1.0, 0.0, 0.0, 1.0};
        const span2d<const double> mat(identity.data(), 2, 2);
        const std::vector<double> wrong_offset{0.0};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::transform<double>(bez2d, mat, std::span<const double>(wrong_offset));
            }) == "The translation must have 2 entries.",
            "transform did not reject a wrong offset length with the expected message");
    }
    {
        const std::vector<double> lower{0.0};
        const std::vector<double> upper{1.0};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::restrict<double>(bez2d, std::span<const double>(lower),
                                            std::span<const double>(upper));
            }) == "lower and upper must each have one entry per parametric direction (2).",
            "restrict did not reject wrong-length bounds with the expected message");
    }
    {
        const std::vector<double> lower{0.6, 0.0};
        const std::vector<double> upper{0.3, 1.0};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::restrict<double>(bez2d, std::span<const double>(lower),
                                            std::span<const double>(upper));
            }) == "The bounds of direction 0 must satisfy 0 <= lower <= upper <= 1.",
            "restrict did not reject inverted bounds with the expected message");
    }
    {
        const std::vector<double> lower{0.0, -0.1};
        const std::vector<double> upper{1.0, 1.0};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::restrict<double>(bez2d, std::span<const double>(lower),
                                            std::span<const double>(upper));
            }) == "The bounds of direction 1 must satisfy 0 <= lower <= upper <= 1.",
            "restrict did not reject an out-of-range bound with the expected message");
    }
    {
        const std::vector<double> lower{0.0, 0.0};
        const std::vector<double> upper{1.0, 1.0};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::restrict<double>(bez2d, std::span<const double>(lower),
                                            std::span<const double>(upper));
            }) == "Bounds match the full domain; at least one direction must be restricted.",
            "restrict did not reject the full domain with the expected message");
    }
    PANTR_CHECK_MSG(
        message_of([&] { (void)ops::split<double>(bez2d, 0, 1.5); }) == "value must be in [0, 1].",
        "split did not reject a value outside [0, 1] with the expected message");
    PANTR_CHECK_MSG(
        message_of([&] { (void)ops::slice_point<double>(bez2d, 0.5); })
            == "slice_point needs a one-dimensional Bézier, got dimension 2.",
        "slice_point did not reject a two-dimensional Bézier with the expected message");
    PANTR_CHECK_MSG(
        message_of([&] { (void)ops::slice<double>(bez1d, 0, 0.5); })
            == "slice needs a Bézier of dimension at least two; a one-dimensional one slices to "
               "a point, which slice_point returns.",
        "slice did not reject a one-dimensional Bézier with the expected message");
    PANTR_CHECK_MSG(
        message_of([&] { (void)ops::slice<double>(bez2d, 9, 0.5); })
            == "axis must be in [0, 2), got 9.",
        "slice did not reject an out-of-range axis with the expected message");
    PANTR_CHECK_MSG(
        message_of([&] { (void)ops::boundary<double>(bez2d, 0, 2); })
            == "side must be 0 or 1, got 2.",
        "boundary did not reject an invalid side with the expected message");
    {
        const std::vector<double> empty_values{};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::collapse_along_axis<double>(bez1d, 0, std::span<const double>(empty_values));
            }) == "collapse_along_axis needs a Bézier of dimension at least two, got 1.",
            "collapse_along_axis did not reject a one-dimensional Bézier with the expected "
            "message");
    }
    {
        const std::vector<double> wrong_length{0.1, 0.2};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::collapse_along_axis<double>(bez2d, 0, std::span<const double>(wrong_length));
            }) == "values must have length dim - 1 = 1, got 2.",
            "collapse_along_axis did not reject wrong-length values with the expected message");
    }
    {
        const std::vector<double> out_of_range{1.5};
        PANTR_CHECK_MSG(
            message_of([&] {
                (void)ops::collapse_along_axis<double>(bez2d, 0, std::span<const double>(out_of_range));
            }) == "All values must be in [0, 1].",
            "collapse_along_axis did not reject an out-of-range value with the expected message");
    }
}

}  // namespace

int main() {
    check_reverse_is_an_involution<double>();
    check_reverse_is_an_involution<float>();
    check_permute_directions_composes_and_inverts<double>();
    check_permute_directions_composes_and_inverts<float>();
    check_transform_identity_and_scaling_are_exact<double>();
    check_transform_identity_and_scaling_are_exact<float>();
    check_transform_general_matrix_and_rational_weight<double>();
    check_transform_general_matrix_and_rational_weight<float>();
    check_restrict_reproduces_the_subinterval<double>();
    check_restrict_reproduces_the_subinterval<float>();
    check_split_reproduces_the_curve<double>();
    check_split_reproduces_the_curve<float>();
    check_slice_reproduces_a_fixed_direction<double>();
    check_slice_reproduces_a_fixed_direction<float>();
    check_slice_point_matches_evaluate<double>();
    check_slice_point_matches_evaluate<float>();
    check_boundary_matches_slice_and_corners<double>();
    check_boundary_matches_slice_and_corners<float>();
    check_collapse_along_axis_keeps_the_right_direction<double>();
    check_collapse_along_axis_keeps_the_right_direction<float>();
    check_rejections();
    return pantr::test::summary("test_bezier_shape");
}
