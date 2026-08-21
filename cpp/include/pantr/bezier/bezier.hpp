#pragma once

/// \file
/// The seven Bézier arithmetic kernels of `src/pantr/bezier/_bezier_core.py`,
/// which stays as the parity oracle.
///
/// ## Three accumulation widths, and none of them is a choice made here
///
/// The oracle does not use one width. Reproducing which is which is most of the
/// work in this file, and getting one wrong is invisible at `float64` and moves
/// values at `float32`. Measured on the de Casteljau kernel: accumulating in
/// `float` where the oracle accumulates in `double` moves 125 of 630 values.
///
///  1. **`evaluate`, `slice`, `split`, `restrict` accumulate in `double`.** The
///     oracle's scalars are Python floats, and numba promotes a `float64` scalar
///     against a `float32` array, so each step is computed at full width and
///     rounded once on the store into the `float32` workspace. `Acc` is that
///     width.
///  2. **`evaluate_deriv` runs at the point dtype throughout.** Alone among the
///     seven, that kernel opens with `dtype = pts.dtype` and allocates every
///     workspace in it, so at `float32` the `ndu` table, the `a_arr` ping-pong
///     and the derivative table are all `float32`. The one exception is the
///     factorial scaling, where a `float32` against a Python `int` promotes.
///  3. **`degree_elevate` and `scalar_bernstein_product` mix.** Their coefficient
///     tables are `float64` unconditionally -- `bezalfs` is declared so, and
///     `bincoeff` returns a `double` -- while the destination is the control
///     points' dtype, so each `+=` computes wide and rounds narrow.
///
/// ## Where the two backends can disagree, and where they cannot
///
/// Every inner step here is `a * b + c * d` or `a * b`, so unlike the Bernstein
/// tabulation there **are** sites for `-ffp-contract` to fuse. At the baseline
/// target that is inert, since base `x86-64` has no FMA instruction; with
/// `-march=native` it is not, and 125 of 630 `float64` de Casteljau values move.
/// This is the `__fp_contract__` runtime gate of design/build_findings.md, not a
/// property of the code, and the parity tests read it rather than assuming it.
///
/// The one transcendental is `pow`, seeding the evaluation recurrence.
/// `cpp/include/pantr/basis/bernstein.hpp` records the same open question and the
/// same answer: numba's `np.power` and the platform `pow` agree bitwise on every
/// argument tested, which makes it observed rather than derived.
///
/// ## The de Casteljau workspace is read back, not carried in a register
///
/// `d[i] = one_minus_u * d[i] + u * d[i + 1]` reads two `float32` entries and
/// writes one, so the running value never carries at full width across steps.
/// Hoisting `d[i]` into an `Acc` register would be the natural C++ and would
/// drift from the oracle by a growing multiple of one `float32` ulp. The
/// evaluation kernel is the opposite case: there the oracle's `b_curr` *is* a
/// register, so it stays one here.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pantr/core/binomial.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr {

/// Midpoint at which the evaluation recurrence switches to its mirrored branch.
///
/// Bounds the seed below by `0.5^p`, so it never underflows regardless of degree.
/// Mirrors `_bezier_core._MIRROR_THRESHOLD`.
inline constexpr double kBezierMirrorThreshold = 0.5;

/// Evaluate a 1D Bézier at every point of `points`, fusing basis and contraction.
///
/// Runs the Bernstein ratio recurrence and contracts each term with the control
/// points in one pass, so no basis row is ever materialised. Above the midpoint
/// the recurrence runs on `1 - u` and walks the control points backwards, which
/// is what keeps the seed from underflowing at high degree.
///
/// \param ctrl Control points, shape `(degree + 1, rank)`.
/// \param points Evaluation points.
/// \param out Output of shape `(points.size(), rank)`, written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel.
///       For general use call `pantr.bezier.Bezier.evaluate`.
template <Real T>
void evaluate_bezier_1d(span2d<const T> ctrl, std::span<const T> points, span2d<T> out) {
    using Acc = accumulator_t<T>;
    using std::pow;

    const auto degree = static_cast<std::size_t>(ctrl.extent(0)) - 1;
    const std::size_t rank = ctrl.extent(1);
    const std::size_t num_pts = points.size();
    const Acc degree_acc = Acc(static_cast<double>(degree));

    for (std::size_t pt = 0; pt < num_pts; ++pt) {
        const Acc u = static_cast<Acc>(points[pt]);

        if (degree == 0) {
            for (std::size_t r = 0; r < rank; ++r) {
                at(out, pt, r) = at(ctrl, std::size_t{0}, r);
            }
            continue;
        }
        if (u == Acc(1)) {
            for (std::size_t r = 0; r < rank; ++r) {
                at(out, pt, r) = at(ctrl, degree, r);
            }
            continue;
        }

        const Acc one_minus_u = Acc(1) - u;

        if (u > Acc(kBezierMirrorThreshold)) {
            // `b_curr = B_{i,p}(1-u) = B_{p-i,p}(u)`, pairing with `ctrl[p - i]`.
            //
            // The seed is formed at `T`'s width, not `Acc`'s, and that asymmetry
            // is the oracle's rather than a choice here. This branch raises `u`,
            // which is the point array's own dtype; the branch below raises
            // `1 - u`, which numba has already promoted to `float64` by the
            // literal `1.0`. Computing both in `Acc` is the natural port and is
            // wrong at `float32`: measured at degree 17 and `u = 0.75`, where
            // `0.75^17 = 3^17 / 2^34` needs 27 significand bits, the wide seed
            // survives and the narrow one rounds, and the difference reaches the
            // output as one ulp.
            Acc b_curr = static_cast<Acc>(pow(static_cast<T>(u), static_cast<T>(degree_acc)));
            for (std::size_t r = 0; r < rank; ++r) {
                at(out, pt, r) = static_cast<T>(b_curr * static_cast<Acc>(at(ctrl, degree, r)));
            }
            const Acc ratio = one_minus_u / u;
            for (std::size_t i = 1; i <= degree; ++i) {
                const Acc i_acc = Acc(static_cast<double>(i));
                const Acc const_factor = (degree_acc - i_acc + Acc(1)) / i_acc;
                b_curr = b_curr * const_factor * ratio;
                for (std::size_t r = 0; r < rank; ++r) {
                    at(out, pt, r) = static_cast<T>(
                        static_cast<Acc>(at(out, pt, r)) +
                        b_curr * static_cast<Acc>(at(ctrl, degree - i, r)));
                }
            }
            continue;
        }

        Acc b_prev = pow(one_minus_u, degree_acc);
        for (std::size_t r = 0; r < rank; ++r) {
            at(out, pt, r) = static_cast<T>(b_prev * static_cast<Acc>(at(ctrl, std::size_t{0}, r)));
        }
        const Acc ratio = u / one_minus_u;
        for (std::size_t i = 1; i <= degree; ++i) {
            const Acc i_acc = Acc(static_cast<double>(i));
            const Acc b_curr = b_prev * ((degree_acc - i_acc + Acc(1)) / i_acc) * ratio;
            for (std::size_t r = 0; r < rank; ++r) {
                at(out, pt, r) =
                    static_cast<T>(static_cast<Acc>(at(out, pt, r)) +
                                   b_curr * static_cast<Acc>(at(ctrl, i, r)));
            }
            b_prev = b_curr;
        }
    }
}

/// Evaluate a 1D Bézier and its derivatives up to order `n_deriv`.
///
/// Algorithm A2.3 of Piegl & Tiller specialised to Bernstein polynomials: build
/// the `ndu` table, run the triangular derivative recursion, then scale by the
/// falling factorial and contract with the control points.
///
/// \param ctrl Control points, shape `(degree + 1, rank)`.
/// \param points Evaluation points.
/// \param n_deriv Highest derivative order, `>= 0`.
/// \param out Output of shape `(points.size(), n_deriv + 1, rank)`, written in
///        full.
///
/// \note No input validation is performed. This is a Layer 3 kernel.
///       Unlike its six siblings this kernel computes at `T`'s own width, since
///       the oracle allocates every workspace at the point dtype. See the file
///       comment.
///       For general use call `pantr.bezier.Bezier.evaluate_derivatives`.
template <Real T>
void evaluate_bezier_deriv_1d(span2d<const T> ctrl, std::span<const T> points, int n_deriv,
                              span_nd<T, 3> out) {
    using Acc = accumulator_t<T>;

    const auto degree = static_cast<std::size_t>(ctrl.extent(0)) - 1;
    const std::size_t rank = ctrl.extent(1);
    const std::size_t num_pts = points.size();
    const std::size_t order = degree + 1;
    const auto num_derivs = static_cast<std::size_t>(n_deriv);

    // Hoisted out of the point loop and re-zeroed inside it. The oracle
    // reallocates all three per point, which zeroes them; `a_arr` genuinely
    // depends on that, since the `rr` loop below reuses it without re-zeroing
    // and reads entries a later `rr` never wrote.
    std::vector<T> ndu(order * order);
    std::vector<T> a_arr(2 * (num_derivs + 1));
    std::vector<T> basis_derivs((num_derivs + 1) * order);

    for (std::size_t pt = 0; pt < num_pts; ++pt) {
        const T s = points[pt];

        for (std::size_t k = 0; k <= num_derivs; ++k) {
            for (std::size_t r = 0; r < rank; ++r) {
                at(out, pt, k, r) = T(0);
            }
        }
        for (T& v : ndu) {
            v = T(0);
        }
        for (T& v : a_arr) {
            v = T(0);
        }
        for (T& v : basis_derivs) {
            v = T(0);
        }

        const auto ndu_at = [&ndu, order](std::size_t i, std::size_t j) -> T& {
            return ndu[i * order + j];
        };
        const auto a_at = [&a_arr, num_derivs](std::size_t i, std::size_t j) -> T& {
            return a_arr[i * (num_derivs + 1) + j];
        };
        const auto bd_at = [&basis_derivs, order](std::size_t i, std::size_t j) -> T& {
            return basis_derivs[i * order + j];
        };

        // The ndu table for uniform Bernstein knots: every knot difference is 1.
        ndu_at(0, 0) = T(1);
        for (std::size_t j = 1; j < order; ++j) {
            T saved = T(0);
            for (std::size_t rr = 0; rr < j; ++rr) {
                ndu_at(j, rr) = T(1);
                const T temp = ndu_at(rr, j - 1);
                ndu_at(rr, j) = static_cast<T>(saved + (T(1) - s) * temp);
                saved = static_cast<T>(s * temp);
            }
            ndu_at(j, j) = saved;
        }

        for (std::size_t j = 0; j < order; ++j) {
            const T basis_val = ndu_at(j, degree);
            for (std::size_t r = 0; r < rank; ++r) {
                at(out, pt, 0, r) =
                    static_cast<T>(at(out, pt, 0, r) + basis_val * at(ctrl, j, r));
            }
            bd_at(0, j) = basis_val;
        }

        // A2.3's triangular recursion, on signed indices: `rk` and `pk` go
        // negative and the loop bounds are derived from them.
        const auto p_signed = static_cast<int>(degree);
        for (int rr = 0; rr < static_cast<int>(order); ++rr) {
            int s1 = 0;
            int s2 = 1;
            a_at(0, 0) = T(1);

            for (int k = 1; k <= n_deriv; ++k) {
                T d = T(0);
                const int rk = rr - k;
                const int pk = p_signed - k;

                if (rr >= k) {
                    a_at(static_cast<std::size_t>(s2), 0) = a_at(static_cast<std::size_t>(s1), 0);
                    d = static_cast<T>(a_at(static_cast<std::size_t>(s2), 0) *
                                       ndu_at(static_cast<std::size_t>(rk),
                                              static_cast<std::size_t>(pk)));
                }

                const int j1 = (rk >= -1) ? 1 : -rk;
                const int j2 = ((rr - 1) <= pk) ? (k - 1) : (p_signed - rr);

                for (int jj = j1; jj <= j2; ++jj) {
                    const auto jz = static_cast<std::size_t>(jj);
                    a_at(static_cast<std::size_t>(s2), jz) =
                        static_cast<T>(a_at(static_cast<std::size_t>(s1), jz) -
                                       a_at(static_cast<std::size_t>(s1), jz - 1));
                    d = static_cast<T>(d + a_at(static_cast<std::size_t>(s2), jz) *
                                               ndu_at(static_cast<std::size_t>(rk + jj),
                                                      static_cast<std::size_t>(pk)));
                }

                if (rr <= pk) {
                    a_at(static_cast<std::size_t>(s2), static_cast<std::size_t>(k)) =
                        static_cast<T>(-a_at(static_cast<std::size_t>(s1),
                                             static_cast<std::size_t>(k - 1)));
                    d = static_cast<T>(d + a_at(static_cast<std::size_t>(s2),
                                                static_cast<std::size_t>(k)) *
                                               ndu_at(static_cast<std::size_t>(rr),
                                                      static_cast<std::size_t>(pk)));
                }

                bd_at(static_cast<std::size_t>(k), static_cast<std::size_t>(rr)) = d;

                const int tmp_s = s1;
                s1 = s2;
                s2 = tmp_s;
            }
        }

        // The falling factorial, and it is accumulated UNSIGNED on purpose.
        //
        // The oracle holds it in a numba `int64`, which wraps on overflow, and it
        // does overflow: at degree 30 from `n_deriv = 14`, at degree 61 from 11,
        // both reachable through `Bezier.evaluate_derivatives`. Signed overflow is
        // undefined in C++ where numba's is merely wrong, so the natural
        // translation is worse than its original -- confirmed under
        // `-fsanitize=undefined`, which the repository's own `gcc-debug` preset
        // enables.
        //
        // Unsigned arithmetic is modular by definition and the C++20 conversion
        // back to `std::int64_t` is the two's-complement reinterpretation, so this
        // reproduces the oracle's wrapped value bit for bit while being defined.
        // Widening to `double` would have removed the undefined behaviour and
        // broken parity instead, since past 2^53 a double rounds where an int64
        // wraps.
        std::uint64_t fac = static_cast<std::uint64_t>(degree);
        for (std::size_t k = 1; k <= num_derivs; ++k) {
            const Acc fac_acc = Acc(static_cast<double>(static_cast<std::int64_t>(fac)));
            for (std::size_t j = 0; j < order; ++j) {
                const Acc scaled = static_cast<Acc>(bd_at(k, j)) * fac_acc;
                for (std::size_t r = 0; r < rank; ++r) {
                    at(out, pt, k, r) =
                        static_cast<T>(static_cast<Acc>(at(out, pt, k, r)) +
                                       scaled * static_cast<Acc>(at(ctrl, j, r)));
                }
            }
            fac *= static_cast<std::uint64_t>(static_cast<std::int64_t>(degree) -
                                              static_cast<std::int64_t>(k));
        }
    }
}

/// Degree-elevate a single Bézier segment by `degree_increment`.
///
/// Builds the `bezalfs` table of elevation coefficients and applies it. The
/// table is `double` regardless of `T`, matching the oracle, and its upper half
/// is filled by the symmetry `bezalfs[j, i] = bezalfs[p - j, ph - i]` rather than
/// recomputed.
///
/// \param degree Original degree `p >= 0`.
/// \param ctrl Control points, shape `(p + 1, rank)`.
/// \param degree_increment Degrees to add, `t >= 1`.
/// \param out Output of shape `(p + t + 1, rank)`. Zeroed, then written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel. In
///       particular `p + t` must not exceed `core::kBincoeffMaxN`; the Python
///       Layer 2 checks that before the call.
///       For general use call `pantr.bezier.Bezier.elevate_degree`.
template <Real T>
void degree_elevate_bezier_1d(int degree, span2d<const T> ctrl, int degree_increment,
                              span2d<T> out) {
    using Acc = accumulator_t<T>;

    const int p = degree;
    const int t = degree_increment;
    const int ph = p + t;
    const std::size_t rank = ctrl.extent(1);
    const int ph2 = ph / 2;

    std::vector<double> bezalfs(static_cast<std::size_t>(p + 1) *
                                static_cast<std::size_t>(ph + 1));
    const auto bez = [&bezalfs, ph](int j, int i) -> double& {
        return bezalfs[static_cast<std::size_t>(j) * static_cast<std::size_t>(ph + 1) +
                       static_cast<std::size_t>(i)];
    };

    bez(0, 0) = 1.0;
    bez(p, ph) = 1.0;

    for (int i = 1; i <= ph2; ++i) {
        const double inv = 1.0 / core::bincoeff(ph, i);
        const int mpi = (p < i) ? p : i;
        for (int j = (i - t > 0 ? i - t : 0); j <= mpi; ++j) {
            bez(j, i) = inv * core::bincoeff(p, j) * core::bincoeff(t, i - j);
        }
    }
    for (int i = ph2 + 1; i < ph; ++i) {
        const int mpi = (p < i) ? p : i;
        for (int j = (i - t > 0 ? i - t : 0); j <= mpi; ++j) {
            bez(j, i) = bez(p - j, ph - i);
        }
    }

    for (int i = 0; i <= ph; ++i) {
        for (std::size_t r = 0; r < rank; ++r) {
            at(out, static_cast<std::size_t>(i), r) = T(0);
        }
    }
    for (int i = 0; i <= ph; ++i) {
        const int mpi = (p < i) ? p : i;
        for (int j = (i - t > 0 ? i - t : 0); j <= mpi; ++j) {
            const Acc coeff = Acc(bez(j, i));
            for (std::size_t r = 0; r < rank; ++r) {
                const auto iz = static_cast<std::size_t>(i);
                at(out, iz, r) = static_cast<T>(
                    static_cast<Acc>(at(out, iz, r)) +
                    coeff * static_cast<Acc>(at(ctrl, static_cast<std::size_t>(j), r)));
            }
        }
    }
}

/// Evaluate a 1D Bézier at a single parameter by de Casteljau, per column.
///
/// \param ctrl Control points, shape `(degree + 1, n_cols)`.
/// \param value Parameter in `[0, 1]`.
/// \param out Output of shape `(n_cols,)`.
///
/// \note No input validation is performed. This is a Layer 3 kernel.
///       For general use call `pantr.bezier.Bezier.slice`.
template <Real T>
void slice_bezier_1d(span2d<const T> ctrl, accumulator_t<T> value, std::span<T> out) {
    using Acc = accumulator_t<T>;

    const auto degree = static_cast<std::size_t>(ctrl.extent(0)) - 1;
    const std::size_t n_cols = ctrl.extent(1);
    const Acc u = value;
    const Acc one_minus_u = Acc(1) - u;

    if (u == Acc(0)) {
        for (std::size_t col = 0; col < n_cols; ++col) {
            out[col] = at(ctrl, std::size_t{0}, col);
        }
        return;
    }
    if (u == Acc(1)) {
        for (std::size_t col = 0; col < n_cols; ++col) {
            out[col] = at(ctrl, degree, col);
        }
        return;
    }

    std::vector<T> d(degree + 1);
    for (std::size_t col = 0; col < n_cols; ++col) {
        for (std::size_t i = 0; i <= degree; ++i) {
            d[i] = at(ctrl, i, col);
        }
        for (std::size_t r = 1; r <= degree; ++r) {
            for (std::size_t i = 0; i + r <= degree; ++i) {
                d[i] = static_cast<T>(one_minus_u * static_cast<Acc>(d[i]) +
                                      u * static_cast<Acc>(d[i + 1]));
            }
        }
        out[col] = d[0];
    }
}

/// Split a 1D Bézier at `value` into its two halves, by de Casteljau.
///
/// \param ctrl Control points, shape `(degree + 1, n_cols)`.
/// \param value Parameter in `[0, 1]`.
/// \param out_left Left half, shape `(degree + 1, n_cols)`.
/// \param out_right Right half, shape `(degree + 1, n_cols)`.
///
/// \note No input validation is performed. This is a Layer 3 kernel. Unlike
///       `slice_bezier_1d` there are no endpoint shortcuts, matching the oracle:
///       a split at 0 or 1 still runs the triangle.
///       For general use call `pantr.bezier.Bezier.split`.
template <Real T>
void split_bezier_1d(span2d<const T> ctrl, accumulator_t<T> value, span2d<T> out_left,
                     span2d<T> out_right) {
    using Acc = accumulator_t<T>;

    const auto degree = static_cast<std::size_t>(ctrl.extent(0)) - 1;
    const std::size_t n_cols = ctrl.extent(1);
    const Acc u = value;
    const Acc one_minus_u = Acc(1) - u;

    std::vector<T> d(degree + 1);
    for (std::size_t col = 0; col < n_cols; ++col) {
        for (std::size_t i = 0; i <= degree; ++i) {
            d[i] = at(ctrl, i, col);
        }
        at(out_left, std::size_t{0}, col) = d[0];

        for (std::size_t r = 1; r <= degree; ++r) {
            for (std::size_t i = 0; i + r <= degree; ++i) {
                d[i] = static_cast<T>(one_minus_u * static_cast<Acc>(d[i]) +
                                      u * static_cast<Acc>(d[i + 1]));
            }
            at(out_left, r, col) = d[0];
        }
        for (std::size_t i = 0; i <= degree; ++i) {
            at(out_right, i, col) = d[i];
        }
    }
}

/// Restrict a 1D Bézier to `[lower, upper]`, reparametrized to `[0, 1]`.
///
/// Two de Casteljau passes, ordered so that neither divides by a small number:
/// if `|upper| >= |lower - 1|` the left pass runs at `upper` and the right at
/// `lower / upper`, otherwise the right pass runs at `lower` and the left at
/// `(upper - lower) / (1 - lower)`.
///
/// \param ctrl Control points, shape `(degree + 1, n_cols)`.
/// \param lower Left bound in `[0, 1)`.
/// \param upper Right bound in `(0, 1]`.
/// \param out Restricted coefficients, shape `(degree + 1, n_cols)`.
///
/// \note No input validation is performed. This is a Layer 3 kernel.
///       For general use call `pantr.bezier.Bezier.restrict`.
template <Real T>
void restrict_bezier_1d(span2d<const T> ctrl, accumulator_t<T> lower, accumulator_t<T> upper,
                        span2d<T> out) {
    using Acc = accumulator_t<T>;
    using std::abs;

    const auto degree = static_cast<std::size_t>(ctrl.extent(0)) - 1;
    const std::size_t n_cols = ctrl.extent(1);
    const auto p = static_cast<std::ptrdiff_t>(degree);

    const bool left_first = abs(upper) >= abs(lower - Acc(1));
    const Acc tau = left_first ? upper : lower;
    const Acc tau2 = left_first ? (lower / upper) : ((upper - lower) / (Acc(1) - lower));

    std::vector<T> d(degree + 1);
    for (std::size_t col = 0; col < n_cols; ++col) {
        for (std::size_t i = 0; i <= degree; ++i) {
            d[i] = at(ctrl, i, col);
        }

        // `left(a)` then `right(b)`, or `right(a)` then `left(b)`; the two passes
        // are the same two loops in the opposite order, with the opposite tau.
        const auto pass_left = [&d, p](Acc t_val) {
            for (std::ptrdiff_t step = 1; step <= p; ++step) {
                for (std::ptrdiff_t j = p; j >= step; --j) {
                    const auto jz = static_cast<std::size_t>(j);
                    d[jz] = static_cast<T>(static_cast<Acc>(d[jz]) * t_val +
                                           static_cast<Acc>(d[jz - 1]) * (Acc(1) - t_val));
                }
            }
        };
        const auto pass_right = [&d, p](Acc t_val) {
            for (std::ptrdiff_t step = 1; step <= p; ++step) {
                for (std::ptrdiff_t j = 0; j <= p - step; ++j) {
                    const auto jz = static_cast<std::size_t>(j);
                    d[jz] = static_cast<T>(static_cast<Acc>(d[jz]) * (Acc(1) - t_val) +
                                           static_cast<Acc>(d[jz + 1]) * t_val);
                }
            }
        };

        if (left_first) {
            pass_left(tau);
            pass_right(tau2);
        } else {
            pass_right(tau);
            pass_left(tau2);
        }

        for (std::size_t i = 0; i <= degree; ++i) {
            at(out, i, col) = d[i];
        }
    }
}

/// Multiply two scalar 1D Béziers in the Bernstein basis.
///
/// `c_k = (1 / C(p+q, k)) * sum_i C(p, i) C(q, k-i) a_i b_{k-i}`, formed as the
/// oracle forms it: scale `a_i` by `C(p, i)` once, accumulate `a_i_scaled * b_j *
/// C(q, j)` left to right, then divide the whole row by `C(p+q, k)`.
///
/// \param a Control points of the first curve, shape `(p + 1,)`.
/// \param b Control points of the second curve, shape `(q + 1,)`.
/// \param out Product control points, shape `(p + q + 1,)`. Zeroed, then written
///        in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel. `p + q`
///       must not exceed `core::kBincoeffMaxN`; the Python Layer 2 checks it.
///       For general use call `pantr.bezier.Bezier.multiply`.
template <Real T>
void scalar_bernstein_product_1d(std::span<const T> a, std::span<const T> b, std::span<T> out) {
    using Acc = accumulator_t<T>;

    const auto p = static_cast<int>(a.size()) - 1;
    const auto q = static_cast<int>(b.size()) - 1;
    const int r = p + q;

    for (std::size_t k = 0; k < out.size(); ++k) {
        out[k] = T(0);
    }

    for (int i = 0; i <= p; ++i) {
        const Acc a_scaled = static_cast<Acc>(a[static_cast<std::size_t>(i)]) *
                             Acc(core::bincoeff(p, i));
        for (int j = 0; j <= q; ++j) {
            const auto kz = static_cast<std::size_t>(i + j);
            out[kz] = static_cast<T>(
                static_cast<Acc>(out[kz]) +
                a_scaled * static_cast<Acc>(b[static_cast<std::size_t>(j)]) *
                    Acc(core::bincoeff(q, j)));
        }
    }

    for (int k = 0; k <= r; ++k) {
        const auto kz = static_cast<std::size_t>(k);
        out[kz] = static_cast<T>(static_cast<Acc>(out[kz]) / Acc(core::bincoeff(r, k)));
    }
}

}  // namespace pantr
