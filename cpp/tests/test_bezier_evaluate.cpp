/// \file
/// Mathematical properties of the two n-dimensional Bézier evaluation entry
/// points, `pantr::bezier::evaluate` and `pantr::bezier::evaluate_on_lattice`.
///
/// ## These are NOT parity tests
///
/// `tests/parity/test_bezier_evaluate.py` compares both entry points against
/// their Numba oracles, within the two `gamma_N`-derived bounds
/// `evaluate.hpp`'s file comment states. Nothing here repeats that comparison,
/// and nothing here compares the two C++ entry points against Python at all.
/// Every assertion below is against a property the mathematics fixes, or
/// against a second computation of the same value by a different route --
/// the same discipline `test_bezier_arithmetic.cpp` and
/// `test_bezier_root_finding.cpp` already use, and for the same reason: a
/// comparison against an oracle cannot catch a bug the port and the oracle
/// share, and `evaluate` and `evaluate_on_lattice` were both transliterated
/// from oracles deliberately.
///
/// ## Where each tolerance comes from
///
/// Every bounded check below composes the same standard result
/// `evaluate.hpp`'s file comment already cites for its own (different)
/// claims: a summation of `n` terms committed in any order carries a relative
/// perturbation bounded by Higham's `gamma_n = n*eps / (1 - n*eps)`
/// (*Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1), and
/// the Bernstein basis being non-negative makes the absolute-value companion
/// exact rather than an over-estimate. `gamma_n` is instantiated below with
/// `N = sum_d (degree_d + 1)`, the total number of terms summed across the
/// `dim` contraction stages a single evaluation performs -- the same count
/// `evaluate.hpp` calls `stages`. No bound here is a bare constant; each
/// comment states which contraction it covers and why the companion is safe.
///
/// ## What each check would catch
///
///  - **Endpoint interpolation, exact.** At a corner of the parameter cube the
///    Bernstein basis is exactly `{1, 0, ..., 0}` or `{0, ..., 0, 1}`
///    (`test_bernstein.cpp`'s `endpoints_are_exact` pins this), so every stage
///    of the contraction reduces to `0 + 1*x = x` and `0 + 0*x = 0`, both
///    exact for finite operands. No tolerance is hiding anything here.
///  - **Partition of unity and affine reproduction.** Two properties the
///    mathematics fixes exactly, checked to within the `gamma_N` rounding
///    budget the floating contraction actually spends.
///  - **`evaluate` and `evaluate_on_lattice` agree.** Different summation
///    schedules over the same mathematics, so the comparison is bounded by
///    the sum of their two independent `gamma_N` budgets rather than bitwise.
///  - **The rational projection.** Exact at a corner (the weight contraction
///    is a plain selection there, not a sum), bounded elsewhere by the same
///    `gamma_N` argument applied to the weight column's own contraction.
///  - **`dim() == 1` delegation, rejections, a degree-0 direction, and the
///    `out` contract.** Structural pins, most of them exact; see each
///    function.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/evaluate.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

using pantr::span2d;
using pantr::bezier::Bezier;
using pantr::bezier::ControlNet;
using pantr::bezier::evaluate;
using pantr::bezier::evaluate_bezier_1d;
using pantr::bezier::evaluate_on_lattice;

/// Higham's standard bound for a summation of `n` terms committed in any
/// order: each term carries at most `n` roundings, so the relative
/// perturbation of the sum is at most `n*eps / (1 - n*eps)` (Higham,
/// *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1).
/// `evaluate.hpp`'s file comment composes the same bound across the `dim`
/// contraction stages a single evaluation performs; every bounded check below
/// does the same, independently, for its own contraction.
///
/// \param n The number of terms summed.
/// \param eps The scalar type's machine epsilon.
/// \return The relative bound `gamma_n`.
double gamma_n(std::size_t n, double eps) {
    const double n_eps = static_cast<double>(n) * eps;
    return n_eps / (1.0 - n_eps);
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
/// i_{dim-1})`, component `c`, is tagged with its own index rather than being
/// arbitrary, so a wrong corner or a wrong stage order is caught rather than
/// merely a wrong value. Every tag is a small integer, exactly representable
/// at `float` as well as `double`.
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

/// Evaluate `bezier` at every corner of its parameter cube and check that
/// each reproduces its control point bit for bit.
///
/// \param degrees The degree in every parametric direction, `dim` of them.
/// \param rank The number of value components.
template <class T>
void check_corners_are_exact(const std::vector<std::size_t>& degrees, std::size_t rank) {
    const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);
    const std::size_t dim = degrees.size();
    const std::size_t n_corners = std::size_t{1} << dim;

    std::vector<T> pts(n_corners * dim);
    for (std::size_t k = 0; k < n_corners; ++k) {
        for (std::size_t d = 0; d < dim; ++d) {
            pts[(k * dim) + d] = (((k >> d) & 1U) != 0U) ? T(1) : T(0);
        }
    }
    const span2d<const T> pts_view(pts.data(), n_corners, dim);
    std::vector<T> out(n_corners * rank);
    const span2d<T> out_view(out.data(), n_corners, rank);
    evaluate<T>(bez, pts_view, out_view);

    std::vector<std::size_t> shape(dim + 1);
    for (std::size_t d = 0; d < dim; ++d) {
        shape[d] = degrees[d] + 1;
    }
    shape.back() = rank;
    const std::span<const T> values = bez.net().values();

    for (std::size_t k = 0; k < n_corners; ++k) {
        std::vector<std::size_t> idx(dim + 1, 0);
        for (std::size_t d = 0; d < dim; ++d) {
            idx[d] = (((k >> d) & 1U) != 0U) ? degrees[d] : 0;
        }
        for (std::size_t c = 0; c < rank; ++c) {
            idx[dim] = c;
            const std::size_t flat = flat_index(shape, idx);
            PANTR_CHECK_MSG(out[(k * rank) + c] == values[flat],
                            "a corner did not reproduce its control point exactly");
        }
    }
}

/// Endpoint interpolation, exact, in 2-D and 3-D. See the file comment.
template <class T>
void check_endpoint_interpolation_is_exact() {
    check_corners_are_exact<T>({2, 3}, 2);     // 2-D, rank 2
    check_corners_are_exact<T>({1, 2, 1}, 2);  // 3-D, rank 2
}

/// Partition of unity: with every control point equal to `v`, the Bézier is
/// identically `v`, to within the `gamma_N` budget the floating contraction
/// spends computing it. See the file comment for the derivation.
template <class T>
void check_partition_of_unity() {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<double> v{1.5, -3.25};
    const std::size_t rank = v.size();

    for (const std::vector<std::size_t>& degrees :
         {std::vector<std::size_t>{2, 3}, std::vector<std::size_t>{2, 3, 2}}) {
        const std::size_t dim = degrees.size();
        std::vector<std::size_t> shape(dim + 1);
        std::size_t n_coeff = 1;
        std::size_t stages = 0;
        for (std::size_t d = 0; d < dim; ++d) {
            shape[d] = degrees[d] + 1;
            n_coeff *= shape[d];
            stages += shape[d];
        }
        shape.back() = rank;

        std::vector<T> values(n_coeff * rank);
        for (std::size_t i = 0; i < n_coeff; ++i) {
            for (std::size_t c = 0; c < rank; ++c) {
                values[(i * rank) + c] = static_cast<T>(v[c]);
            }
        }
        const Bezier<T> bez = make_bezier<T>(std::move(values), shape, false);

        std::vector<std::vector<T>> samples;
        if (dim == 2) {
            samples = {{T(0), T(0)},
                       {T(1), T(1)},
                       {T(0.37), T(0.81)},
                       {T(0.999), T(0.001)},
                       {T(0.5), T(0.5)}};
        } else {
            samples = {{T(0), T(0), T(0)},
                       {T(1), T(1), T(1)},
                       {T(0.37), T(0.81), T(0.2)},
                       {T(0.999), T(0.001), T(0.6)},
                       {T(0.5), T(0.5), T(0.5)}};
        }
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

        const double bound_factor = gamma_n(stages, eps);
        for (std::size_t k = 0; k < samples.size(); ++k) {
            for (std::size_t c = 0; c < rank; ++c) {
                const double got = static_cast<double>(out[(k * rank) + c]);
                const double bound = bound_factor * std::abs(v[c]);
                PANTR_CHECK_MSG(std::abs(got - v[c]) <= bound,
                                "partition of unity left its gamma_N budget");
            }
        }
    }
}

/// Reproduction of an affine map: a degree-1-per-direction Bézier whose
/// control points are `a0 + sum_d a_d * i_d` at every corner `i` reproduces
/// `a0 + sum_d a_d * u_d` everywhere, because each direction's Bernstein pair
/// `(1 - u_d, u_d)` reproduces a function linear in `u_d` exactly and the map
/// has no cross term to lose. Checked to within the same `gamma_N` budget as
/// partition_of_unity, with the companion the largest control-point magnitude
/// actually reachable (the Bernstein basis is non-negative, so no output
/// exceeds a convex combination of the corners).
template <class T>
void check_reproduces_an_affine_map() {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const std::vector<double> a0{0.25, -1.5};
    const std::vector<std::vector<double>> a{{0.6, -0.2}, {-0.4, 0.9}, {0.3, 0.1}};
    const std::size_t rank = a0.size();

    for (const std::size_t dim : {std::size_t{2}, std::size_t{3}}) {
        std::vector<std::size_t> shape(dim + 1, 2);
        shape.back() = rank;
        const std::size_t n_coeff = std::size_t{1} << dim;
        const std::size_t stages = 2 * dim;

        std::vector<T> values(n_coeff * rank);
        double max_abs_ctrl = 0.0;
        for (std::size_t k = 0; k < n_coeff; ++k) {
            for (std::size_t c = 0; c < rank; ++c) {
                double val = a0[c];
                for (std::size_t d = 0; d < dim; ++d) {
                    // Row-major: direction 0 is the SLOWEST-varying index, so it
                    // is bit (dim - 1 - d) of the corner index `k`, not bit `d`
                    // -- the same convention `flat_index` uses elsewhere in this
                    // file.
                    const double idx_d = (((k >> (dim - 1 - d)) & 1U) != 0U) ? 1.0 : 0.0;
                    val += a[d][c] * idx_d;
                }
                values[(k * rank) + c] = static_cast<T>(val);
                max_abs_ctrl = std::max(max_abs_ctrl, std::abs(val));
            }
        }
        const Bezier<T> bez = make_bezier<T>(std::move(values), shape, false);

        std::vector<std::vector<T>> samples;
        if (dim == 2) {
            samples = {{T(0.3), T(0.7)}, {T(0.05), T(0.95)}, {T(0.5), T(0.5)}};
        } else {
            samples = {{T(0.3), T(0.7), T(0.2)}, {T(0.05), T(0.95), T(0.6)},
                       {T(0.5), T(0.5), T(0.5)}};
        }
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

        const double bound_factor = gamma_n(stages, eps);
        for (std::size_t k = 0; k < samples.size(); ++k) {
            for (std::size_t c = 0; c < rank; ++c) {
                double expected = a0[c];
                for (std::size_t d = 0; d < dim; ++d) {
                    expected += a[d][c] * static_cast<double>(samples[k][d]);
                }
                const double got = static_cast<double>(out[(k * rank) + c]);
                const double bound = bound_factor * max_abs_ctrl;
                PANTR_CHECK_MSG(std::abs(got - expected) <= bound,
                                "the multilinear interpolant left its gamma_N budget around "
                                "the affine map it should reproduce");
            }
        }
    }
}

/// `evaluate` and `evaluate_on_lattice` agree mathematically at the lattice's
/// own points, built explicitly as the cartesian product (direction 0
/// slowest, the row-major order `evaluate_on_lattice` writes). The two
/// schedules are not compared bitwise: each independently sits within its own
/// `gamma_N` budget, so their difference is bounded by the sum.
template <class T>
void check_evaluate_and_lattice_agree() {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());

    for (const std::vector<std::size_t>& degrees :
         {std::vector<std::size_t>{2, 3}, std::vector<std::size_t>{1, 2, 1}}) {
        const std::size_t dim = degrees.size();
        const std::size_t rank = 2;
        const Bezier<T> bez = make_tagged_bezier<T>(degrees, rank);

        std::vector<std::vector<T>> dirs_vec;
        if (dim == 2) {
            dirs_vec = {{T(0), T(0.2), T(0.55), T(1)}, {T(0.1), T(0.5), T(0.9)}};
        } else {
            dirs_vec = {{T(0), T(0.4), T(1)}, {T(0.15), T(0.65)}, {T(0.05), T(0.5), T(0.95)}};
        }
        std::vector<std::size_t> m(dim);
        std::size_t lattice_size = 1;
        for (std::size_t d = 0; d < dim; ++d) {
            m[d] = dirs_vec[d].size();
            lattice_size *= m[d];
        }

        std::vector<std::span<const T>> dirs(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            dirs[d] = std::span<const T>(dirs_vec[d]);
        }
        std::vector<T> lattice_out(lattice_size * rank);
        evaluate_on_lattice<T>(bez, std::span<const std::span<const T>>(dirs.data(), dirs.size()),
                               std::span<T>(lattice_out));

        std::vector<T> pts(lattice_size * dim);
        std::vector<std::size_t> idx(dim, 0);
        for (std::size_t flat = 0; flat < lattice_size; ++flat) {
            std::size_t rem = flat;
            for (std::size_t d = dim; d-- > 0;) {
                idx[d] = rem % m[d];
                rem /= m[d];
            }
            for (std::size_t d = 0; d < dim; ++d) {
                pts[(flat * dim) + d] = dirs_vec[d][idx[d]];
            }
        }
        const span2d<const T> pts_view(pts.data(), lattice_size, dim);
        std::vector<T> pts_out(lattice_size * rank);
        const span2d<T> pts_out_view(pts_out.data(), lattice_size, rank);
        evaluate<T>(bez, pts_view, pts_out_view);

        std::size_t stages = 0;
        for (const std::size_t degree : degrees) {
            stages += degree + 1;
        }
        double max_abs_ctrl = 0.0;
        for (const T val : bez.net().values()) {
            max_abs_ctrl = std::max(max_abs_ctrl, std::abs(static_cast<double>(val)));
        }
        const double bound = 2.0 * gamma_n(stages, eps) * max_abs_ctrl;

        for (std::size_t i = 0; i < lattice_size * rank; ++i) {
            const double diff =
                std::abs(static_cast<double>(lattice_out[i]) - static_cast<double>(pts_out[i]));
            PANTR_CHECK_MSG(diff <= bound,
                            "evaluate and evaluate_on_lattice disagree beyond their combined "
                            "gamma_N budget");
        }
    }
}

/// The rational projection: every weight equal to a power of two `w`.
///
/// At a corner the weight contraction is a plain selection (see
/// `check_corners_are_exact`), not a sum, so it equals `w` exactly rather than
/// merely closely; projecting then divides the SAME two operands -- the
/// selected control point and `w` -- that building the comparison Bézier
/// divided ahead of time, and division is a deterministic function of its
/// operands, so the two orders agree bit for bit.
///
/// Away from a corner the weight contraction carries the same `gamma_N`
/// rounding `check_partition_of_unity` bounds for a constant net (it is the
/// same computation, scaled throughout by the power-of-two `w`, which is
/// exact: multiplying or dividing a finite float by a power of two only
/// shifts its exponent, so it commutes exactly with every rounding a
/// multiply-add or a division performs, barring overflow or underflow -- none
/// of which occurs here). So the rational result divides by a quantity within
/// `gamma_N` of `w` rather than by `w` itself, differing from the
/// already-divided comparison by about `gamma_N` of its own magnitude, plus
/// one more `eps` for the projection's own division. The factor of four below
/// is an **admitted heuristic**, not a derived constant -- a margin in the
/// spirit of the factor of two `test_bezier_root_finding.cpp` documents for
/// its own residual bound. It is stated as one so it can be argued with; an
/// invented derivation could not be. How much of it is actually used is not
/// recorded here, because a number nothing re-measures rots while reading as
/// current: lower the factor and re-run this test, which is the reproduction.
template <class T>
void check_rational_projection() {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    constexpr std::size_t n0 = 3;
    constexpr std::size_t n1 = 4;
    constexpr std::size_t rank = 2;
    constexpr std::size_t cp_size = rank + 1;
    constexpr double w = 8.0;

    std::vector<T> rat_values(n0 * n1 * cp_size);
    std::vector<T> nr_values(n0 * n1 * rank);
    for (std::size_t i0 = 0; i0 < n0; ++i0) {
        for (std::size_t i1 = 0; i1 < n1; ++i1) {
            for (std::size_t c = 0; c < rank; ++c) {
                const double y = 1.3 + (0.7 * static_cast<double>(i0)) -
                                 (0.4 * static_cast<double>(i1)) + (0.2 * static_cast<double>(c));
                rat_values[(((i0 * n1) + i1) * cp_size) + c] = static_cast<T>(y);
                nr_values[(((i0 * n1) + i1) * rank) + c] = static_cast<T>(y / w);
            }
            rat_values[(((i0 * n1) + i1) * cp_size) + rank] = static_cast<T>(w);
        }
    }
    const Bezier<T> rat = make_bezier<T>(std::move(rat_values), {n0, n1, cp_size}, true);
    const Bezier<T> nr = make_bezier<T>(std::move(nr_values), {n0, n1, rank}, false);

    {
        const std::vector<T> corners{T(0), T(0), T(0), T(1), T(1), T(0), T(1), T(1)};
        const span2d<const T> pts_view(corners.data(), 4, 2);
        std::vector<T> out_rat(4 * rank);
        std::vector<T> out_nr(4 * rank);
        const span2d<T> out_rat_view(out_rat.data(), 4, rank);
        const span2d<T> out_nr_view(out_nr.data(), 4, rank);
        evaluate<T>(rat, pts_view, out_rat_view);
        evaluate<T>(nr, pts_view, out_nr_view);
        for (std::size_t i = 0; i < 4 * rank; ++i) {
            PANTR_CHECK_MSG(out_rat[i] == out_nr[i],
                            "the rational projection was not exact at a corner");
        }
    }

    {
        const std::size_t stages = n0 + n1;
        const double bound_factor = 4.0 * (gamma_n(stages, eps) + eps);
        const std::vector<std::array<T, 2>> samples{
            {T(0.37), T(0.81)}, {T(0.05), T(0.62)}, {T(0.9), T(0.12)}, {T(0.5), T(0.5)}};
        std::vector<T> pts;
        pts.reserve(samples.size() * 2);
        for (const std::array<T, 2>& s : samples) {
            pts.push_back(s[0]);
            pts.push_back(s[1]);
        }
        const span2d<const T> pts_view(pts.data(), samples.size(), 2);
        std::vector<T> out_rat(samples.size() * rank);
        std::vector<T> out_nr(samples.size() * rank);
        const span2d<T> out_rat_view(out_rat.data(), samples.size(), rank);
        const span2d<T> out_nr_view(out_nr.data(), samples.size(), rank);
        evaluate<T>(rat, pts_view, out_rat_view);
        evaluate<T>(nr, pts_view, out_nr_view);
        for (std::size_t i = 0; i < samples.size() * rank; ++i) {
            const double nr_val = static_cast<double>(out_nr[i]);
            const double bound = bound_factor * std::abs(nr_val);
            PANTR_CHECK_MSG(std::abs(static_cast<double>(out_rat[i]) - nr_val) <= bound,
                            "the rational projection left its gamma_N budget away from a corner");
        }
    }
}

/// `dim() == 1` delegation: `evaluate` must be bit-identical to calling
/// `evaluate_bezier_1d` directly, projecting by hand for the rational case.
/// This pins the branch `evaluate.hpp`'s file comment says exists rather than
/// restating the 1-D kernel's own bitwise parity claim.
template <class T>
void check_dim_one_delegates() {
    {
        const std::vector<T> ctrl{T(1), T(-2), T(3), T(0.5)};
        const span2d<const T> ctrl_view(ctrl.data(), 4, 1);
        const Bezier<T> bez = make_bezier<T>(ctrl, {4}, false);

        const std::vector<T> params{T(0), T(0.25), T(0.5), T(0.75), T(1)};
        const span2d<const T> pts_view(params.data(), params.size(), 1);
        std::vector<T> via_evaluate(params.size());
        const span2d<T> via_evaluate_view(via_evaluate.data(), params.size(), 1);
        evaluate<T>(bez, pts_view, via_evaluate_view);

        std::vector<T> via_kernel(params.size());
        const span2d<T> via_kernel_view(via_kernel.data(), params.size(), 1);
        evaluate_bezier_1d<T>(ctrl_view, std::span<const T>(params), via_kernel_view);

        for (std::size_t i = 0; i < params.size(); ++i) {
            PANTR_CHECK_MSG(via_evaluate[i] == via_kernel[i],
                            "evaluate() no longer delegates dim==1 to evaluate_bezier_1d");
        }
    }
    {
        const std::vector<T> ctrl{T(1), T(2), T(-2), T(4), T(3), T(1), T(0.5), T(0.5)};
        const span2d<const T> ctrl_view(ctrl.data(), 4, 2);
        const Bezier<T> bez = make_bezier<T>(ctrl, {4, 2}, true);

        const std::vector<T> params{T(0), T(0.25), T(0.5), T(0.75), T(1)};
        const span2d<const T> pts_view(params.data(), params.size(), 1);
        std::vector<T> via_evaluate(params.size());
        const span2d<T> via_evaluate_view(via_evaluate.data(), params.size(), 1);
        evaluate<T>(bez, pts_view, via_evaluate_view);

        std::vector<T> raw(params.size() * 2);
        const span2d<T> raw_view(raw.data(), params.size(), 2);
        evaluate_bezier_1d<T>(ctrl_view, std::span<const T>(params), raw_view);

        for (std::size_t i = 0; i < params.size(); ++i) {
            const T expected = static_cast<T>(raw[(i * 2) + 0] / raw[(i * 2) + 1]);
            PANTR_CHECK_MSG(
                via_evaluate[i] == expected,
                "evaluate()'s rational dim==1 path no longer matches evaluate_bezier_1d "
                "projected by hand");
        }
    }
}

/// The four validation messages, asserted verbatim: they mirror the Python
/// oracle's own text (see `bezier.hpp`'s parity notes), and a reworded
/// message here is a parity failure a type-only check would not notice.
void check_rejections() {
    const Bezier<double> bez = make_bezier<double>({0.0, 1.0, 2.0, 3.0}, {2, 2, 1}, false);

    {
        const std::vector<double> pts_data{0.5, 0.5, 0.5};
        const span2d<const double> pts_view(pts_data.data(), 1, 3);
        std::vector<double> out_data(bez.rank());
        const span2d<double> out_view(out_data.data(), 1, bez.rank());
        PANTR_CHECK_MSG(
            message_of([&] { evaluate<double>(bez, pts_view, out_view); }) ==
                "pts must be a 2D array with 2 columns.",
            "evaluate() did not reject a wrong column count with the oracle's message");
    }
    {
        const std::vector<double> pts_data{0.5, 0.5};
        const span2d<const double> pts_view(pts_data.data(), 1, 2);
        std::vector<double> out_data(2);
        const span2d<double> out_view(out_data.data(), 1, 2);
        PANTR_CHECK_MSG(
            message_of([&] { evaluate<double>(bez, pts_view, out_view); }) ==
                "out must have shape (1, 1).",
            "evaluate() did not reject a wrong out shape with the oracle's message");
    }
    {
        const std::vector<double> dir0{0.0, 0.5, 1.0};
        const std::vector<std::span<const double>> dirs{std::span<const double>(dir0)};
        std::vector<double> out_data(dir0.size() * bez.rank());
        PANTR_CHECK_MSG(
            message_of([&] {
                evaluate_on_lattice<double>(
                    bez, std::span<const std::span<const double>>(dirs.data(), dirs.size()),
                    std::span<double>(out_data));
            }) == "PointsLattice dim (1) must match Bézier dim (2).",
            "evaluate_on_lattice() did not reject a wrong direction count with the oracle's "
            "message");
    }
    {
        const std::vector<double> dir0{0.0, 0.5, 1.0};
        const std::vector<double> dir1{0.0, 1.0};
        const std::vector<std::span<const double>> dirs{std::span<const double>(dir0),
                                                         std::span<const double>(dir1)};
        std::vector<double> out_data(1);
        PANTR_CHECK_MSG(
            message_of([&] {
                evaluate_on_lattice<double>(
                    bez, std::span<const std::span<const double>>(dirs.data(), dirs.size()),
                    std::span<double>(out_data));
            }) == "out must hold 6 values.",
            "evaluate_on_lattice() did not reject a wrong out size with the oracle's message");
    }
}

/// A degree-0 direction: a Bézier with `degree(1) == 0` is constant in `u1`.
/// Exact rather than bounded, because a degree-0 Bernstein "basis" is the
/// constant `1` (`tabulate_bernstein_1d`'s own degree-0 branch), so that
/// stage's contraction is a bit-exact copy regardless of `u1`.
void check_degree_zero_direction() {
    const Bezier<double> bez = make_bezier<double>({0.0, 1.0, 10.0, 11.0}, {2, 1, 2}, false);
    PANTR_CHECK(bez.degree(1) == 0);

    const std::vector<double> pts{0.3, 0.0, 0.3, 0.5, 0.3, 1.0};
    const span2d<const double> pts_view(pts.data(), 3, 2);
    std::vector<double> out(3 * bez.rank());
    const span2d<double> out_view(out.data(), 3, bez.rank());
    evaluate<double>(bez, pts_view, out_view);
    for (std::size_t c = 0; c < bez.rank(); ++c) {
        PANTR_CHECK_MSG(
            out[c] == out[bez.rank() + c] && out[c] == out[(2 * bez.rank()) + c],
            "a degree-0 direction was not exactly constant across its own parameter");
    }
}

/// `evaluate` writes `out` in full: a poisoned buffer matches a zeroed one.
/// Mirrors the contract `tests/test_bezier_core_out_contract.py` states in
/// Python -- a kernel that accumulates into a destination it forgot to zero
/// reads as correct against a fresh array and wrong against a reused one.
template <class T>
void check_evaluate_out_is_written_in_full() {
    constexpr T poison = static_cast<T>(-12345.0);
    const Bezier<T> bez = make_tagged_bezier<T>({2, 3}, 2);

    const std::vector<T> pts{T(0.2), T(0.7), T(0.5), T(0.5), T(0.9), T(0.1)};
    const span2d<const T> pts_view(pts.data(), 3, 2);

    std::vector<T> clean(3 * bez.rank(), T(0));
    const span2d<T> clean_view(clean.data(), 3, bez.rank());
    evaluate<T>(bez, pts_view, clean_view);

    std::vector<T> poisoned(3 * bez.rank(), poison);
    const span2d<T> poisoned_view(poisoned.data(), 3, bez.rank());
    evaluate<T>(bez, pts_view, poisoned_view);

    for (std::size_t i = 0; i < clean.size(); ++i) {
        PANTR_CHECK_MSG(poisoned[i] == clean[i],
                        "evaluate() left poison in an element it should have overwritten");
    }
}

/// `evaluate_on_lattice`'s own version of the same `out` contract.
template <class T>
void check_lattice_out_is_written_in_full() {
    constexpr T poison = static_cast<T>(-12345.0);
    const Bezier<T> bez = make_tagged_bezier<T>({1, 2}, 2);

    const std::vector<T> dir0{T(0.1), T(0.6), T(1.0)};
    const std::vector<T> dir1{T(0.2), T(0.8)};
    const std::vector<std::span<const T>> dirs{std::span<const T>(dir0), std::span<const T>(dir1)};
    const std::size_t total = dir0.size() * dir1.size() * bez.rank();

    std::vector<T> clean(total, T(0));
    evaluate_on_lattice<T>(bez, std::span<const std::span<const T>>(dirs.data(), dirs.size()),
                           std::span<T>(clean));

    std::vector<T> poisoned(total, poison);
    evaluate_on_lattice<T>(bez, std::span<const std::span<const T>>(dirs.data(), dirs.size()),
                           std::span<T>(poisoned));

    for (std::size_t i = 0; i < total; ++i) {
        PANTR_CHECK_MSG(
            poisoned[i] == clean[i],
            "evaluate_on_lattice() left poison in an element it should have overwritten");
    }
}

}  // namespace

int main() {
    check_endpoint_interpolation_is_exact<double>();
    check_endpoint_interpolation_is_exact<float>();
    check_partition_of_unity<double>();
    check_partition_of_unity<float>();
    check_reproduces_an_affine_map<double>();
    check_reproduces_an_affine_map<float>();
    check_evaluate_and_lattice_agree<double>();
    check_evaluate_and_lattice_agree<float>();
    check_rational_projection<double>();
    check_rational_projection<float>();
    check_dim_one_delegates<double>();
    check_dim_one_delegates<float>();
    check_rejections();
    check_degree_zero_direction();
    check_evaluate_out_is_written_in_full<double>();
    check_evaluate_out_is_written_in_full<float>();
    check_lattice_out_is_written_in_full<double>();
    check_lattice_out_is_written_in_full<float>();
    return pantr::test::summary("test_bezier_evaluate");
}
