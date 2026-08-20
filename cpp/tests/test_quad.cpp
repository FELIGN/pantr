/// \file
/// Properties of the four quadrature kernels in `pantr/quad`, pinned against the
/// C++ on its own -- not against the Python oracle, which `tests/parity/` checks
/// separately from Python.
///
/// ## Shared apparatus: the `gamma` bound
///
/// Every summation-derived tolerance below uses the standard backward-error
/// quantity from Higham, *Accuracy and Stability of Numerical Algorithms*
/// (2nd ed.), Sec. 3.1: for `m` roundings each individually bounded by the unit
/// roundoff `u`,
///
///     gamma(m) := m*u / (1 - m*u)                                          (G)
///
/// bounds the compounded relative error, i.e. `(1+u)^m <= 1 + gamma(m)`. `u` is
/// `eps/2` throughout, matching `_GAUSS_LEGENDRE_NEWTON_STEPS`'s own convention
/// and the brief this file was written against.
///
/// (G) is deliberately **not** implemented as `std::pow(1.0 + u, m) - 1.0`: at
/// `u = eps/2`, `1.0 + u` itself rounds to exactly `1.0` (round-to-even at the
/// exact half-ulp boundary), which makes that expression identically zero for
/// every `m` -- a self-defeating tolerance that always reads "exact". Computing
/// `m*u` and `1.0 - m*u` directly avoids the add-a-half-ulp-to-one trap and is
/// the textbook form besides.
///
/// A single acknowledged safety factor, `kSafety = 2`, multiplies every bound
/// that is built from `gamma()` (or, for the two tanh-sinh integration checks,
/// from `gamma()` plus a derived truncation term) in this file. It is not
/// needed by the derivation itself -- each bound below is checked against the
/// worst observed ratio before being written down, and the bare (G)-derived
/// bound already clears every case measured -- but the per-term operation
/// counts feeding `gamma` are honest estimates, not a line-by-line rounding
/// proof for each summation order, so one shared margin covers that gap
/// uniformly rather than each site inventing its own. This mirrors the single
/// acknowledged fudge factor in test_cardinal_bspline.cpp.
///
/// One bound in the file is the deliberate exception:
/// `chebyshev_endpoints_ascending_and_symmetric`'s symmetry check is not built
/// from `gamma()` at all (see that function's own comment for its derivation),
/// so `kSafety` does not multiply it. Its own margin is already an
/// order-of-magnitude one over the worst case measured, which is what a
/// `kSafety` multiplication is for elsewhere in this file; a second one on top
/// would not be wrong, just redundant, so it is left off and this paragraph
/// says so rather than the "multiplies every bound" claim quietly not applying.
///
/// ## Two bound shapes reused throughout
///
/// **Sum of `m` non-negative terms whose true sum is `S`:** summing `m` terms
/// left-to-right commits a relative backward error of at most `gamma(m-1)`;
/// using `gamma(m)` is generous by exactly one term and absorbs the terms'
/// own formation error too. Absolute tolerance: `kSafety * S * gamma(m)`.
///
/// **Exactness of a quadrature rule against a closed form:** if each term
/// `w_i * f(x_i)` is formed with `p` roundings and the `n` terms are then
/// summed, the combined relative perturbation is `gamma(p + n)` (the two
/// `(1+u)^k` factors compose exactly into `(1+u)^{p+n}`), and it multiplies
/// the sum of the *magnitudes* of the exact terms, not the (possibly small
/// or zero) exact answer -- an odd-degree monomial integrates to zero, and a
/// relative-error statement about zero is not meaningful. Absolute tolerance:
/// `kSafety * (bound on sum of |exact term|) * gamma(p + n)`.
///
/// ## Gauss-Legendre nodes/weights are treated as exact inputs
///
/// legendre.hpp's own docstring: four Newton steps clear the unit roundoff by
/// 13 orders of magnitude at the outermost (worst-conditioned) root. In double
/// arithmetic a converged Newton iterate cannot be more accurate than the
/// correctly-rounded double closest to the true root in any case, so the
/// discrepancy between the computed node/weight and the true mathematical one
/// is itself an `O(u)`-relative quantity -- already the same order as every
/// other rounding counted below. No separate term is carried for it; the `+n`
/// in each bound's operation count is generous enough to cover it without
/// being named explicitly at every use.
///
/// ## One property in the brief that does not hold as stated, and is rephrased
///
/// For odd `n`, the two write positions of the central root coincide (see
/// legendre.hpp), so the final value is whatever Newton converged to, not a
/// value derived from a negation. It is *not* guaranteed to be the double
/// `0.0`: measured here, it is exactly `0.0` for every odd `n` from 1 through
/// 77, first comes out nonzero at `n = 79`, and the first non-zero `n` moves to
/// 57 under a one-ulp nudge of `std::cos`'s result at the Tricomi start -- so
/// asserting the literal value `0.0`, even over a range chosen to be safe on
/// this machine, is a claim about this machine's libm, not about the kernel.
///
/// What *is* a structural, platform-independent guarantee -- true of the
/// kernel's index arithmetic alone, independent of what Newton converges to --
/// is the mechanism the brief's `0.0` claim was standing in for: for odd `n`,
/// `gauss_legendre_symmetric`'s loop writes `out_nodes[i] = -x` and then
/// `out_nodes[count - 1 - i] = x` for `i` running up to `half - 1`, and at
/// `i = half - 1` (the last iteration, which is the central root for odd `n`)
/// `count - 1 - i` reduces to `i` itself, so both writes land at the same
/// index and only the second -- `x`, the positive branch -- survives. That
/// index coincidence is arithmetic (`i == count - 1 - i` at the middle slot of
/// an odd-length array), not a floating-point outcome, so it holds on every
/// platform without qualification. It is what the test below now checks, in
/// place of the exact-zero claim; whether the surviving `x` is itself the
/// correct root is what `tests/parity/` checks against the Python oracle, not
/// this file (see the file's opening comment: this suite pins C++ against
/// itself).
///
/// ## The two tanh-sinh integration checks are both derived
///
/// The smooth-integrand check (`exp`) has two derived terms. `tanh` has its
/// poles at odd multiples of `i*pi/2`, so `x(t) = tanh((pi/2) sinh t)` is
/// analytic in the open strip `|Im t| < pi/2`, and since `exp` is entire the
/// transformed integrand is analytic throughout that same strip. The heuristic
/// asymptotic form for double-exponential quadrature (Takahasi & Mori 1974;
/// Mori & Sugihara, *J. Comput. Appl. Math.* 127 (2001), 287-296) gives
/// truncation error `O(exp(-2*pi*d/h))` for a strip of half-width `d`; at the
/// `n = 41` case swept below, `h ~ 0.154599` and `d = pi/2` put that at
/// `exp(-pi^2/h) ~ 1.9e-28` -- already far below the rounding floor at that
/// `n`, which is why the test held for `n = 41` even before a truncation term
/// was added. But the same asymptotic decay is not negligible at the small `n`
/// this file now also sweeps (measured ratios of error to the old,
/// rounding-only bound: 955 at `n = 21`, 2.1e8 at `n = 11`, 4.3e11 at `n = 5`),
/// so an explicit truncation term is now carried at every `n`, not assumed away
/// by the choice of a single large one; see the test's own comment for its
/// derivation.
///
/// The endpoint-singular check (`1 / sqrt(1 - x^2)`) is likewise now a derived
/// bound rather than the flat, measured `1e-6` this file carried before: that
/// constant was violated outright below `n = 30` and was within 11% of failing
/// at `n = 31`, passing only because the test exercised `n = 41` alone. Its
/// replacement -- truncation, node-quantization and summation terms -- was
/// derived and verified outside this file and is implemented here as given;
/// see the test's own comment.

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/quad/lambert_w.hpp"
#include "pantr/quad/legendre.hpp"
#include "pantr/quad/simple_rules.hpp"
#include "pantr/quad/tanh_sinh.hpp"

namespace {

constexpr double kSafety = 2.0;

/// The unit roundoff, `eps/2`. See the file header for why (G) is not
/// implemented as `std::pow(1.0 + u, m) - 1.0`.
constexpr double kU = std::numeric_limits<double>::epsilon() / 2.0;

/// `gamma(m)` of the file header, equation (G).
constexpr double gamma_bound(int m) {
    const double mu = static_cast<double>(m) * kU;
    return mu / (1.0 - mu);
}

/// `x^k` by repeated multiplication, not `std::pow`. Deliberate: `std::pow`'s
/// accuracy is implementation-defined (tanh_sinh.hpp's header measures it
/// disagreeing with `x * x` on 163 of 200000 arguments), while `k - 1`
/// multiplications each individually correctly rounded is exactly the
/// quantity the `gamma(p + n)` bound above is stated for.
double monomial(double x, int k) {
    if (k == 0) {
        return 1.0;
    }
    double v = x;
    for (int j = 1; j < k; ++j) {
        v *= x;
    }
    return v;
}

// ---------------------------------------------------------------------------
// Gauss-Legendre
// ---------------------------------------------------------------------------

/// The defining property: exact for polynomials of degree `2n - 1`. Tested via
/// monomials, which is equivalent by linearity and gives closed forms directly.
///
/// Bound: `p = k` multiplications form `x^k` and multiply by the weight (see
/// `monomial` and the file header); the `n`-term sum contributes the rest.
/// `sum_i w_i |x_i|^k <= sum_i w_i = 2` since every node lies in `(-1, 1)` and
/// the mathematical weights sum to 2.
void gl_integrates_polynomials_exactly() {
    for (int n = 1; n <= 20; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::gauss_legendre_symmetric(n, nodes, weights);

        for (int k = 0; k <= 2 * n - 1; ++k) {
            double approx = 0.0;
            for (int i = 0; i < n; ++i) {
                approx += weights[static_cast<std::size_t>(i)] *
                          monomial(nodes[static_cast<std::size_t>(i)], k);
            }
            const double exact = (k % 2 == 0) ? 2.0 / static_cast<double>(k + 1) : 0.0;
            const double bound = kSafety * 2.0 * gamma_bound(k + n);
            PANTR_CHECK_MSG(std::abs(approx - exact) <= bound,
                            "n=" + std::to_string(n) + " k=" + std::to_string(k) + ": |" +
                                std::to_string(approx) + " - " + std::to_string(exact) +
                                "| > " + std::to_string(bound));
        }
    }
}

/// Nodes strictly ascending and strictly interior to `(-1, 1)`.
void gl_nodes_ascending_and_interior() {
    for (int n = 1; n <= 300; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::gauss_legendre_symmetric(n, nodes, weights);

        for (int i = 0; i < n; ++i) {
            PANTR_CHECK(nodes[static_cast<std::size_t>(i)] > -1.0);
            PANTR_CHECK(nodes[static_cast<std::size_t>(i)] < 1.0);
            if (i > 0) {
                PANTR_CHECK_MSG(nodes[static_cast<std::size_t>(i)] >
                                     nodes[static_cast<std::size_t>(i - 1)],
                                "n=" + std::to_string(n) + " i=" + std::to_string(i));
            }
        }
    }
}

/// Bit-exact symmetry. Off-center pairs are written from one root by negation
/// (`out_nodes[i] = -x; out_nodes[n-1-i] = x;`), so `node[i] == -node[n-1-i]`
/// and `weight[i] == weight[n-1-i]` hold definitionally -- no tolerance, no
/// dependency on `n`. Looping `i` only up to `n/2 - 1` naturally excludes the
/// central slot for odd `n` (integer division), which is checked separately.
void gl_off_center_symmetry_is_bitexact() {
    for (int n = 1; n <= 300; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::gauss_legendre_symmetric(n, nodes, weights);

        for (int i = 0; i < n / 2; ++i) {
            const std::size_t lo = static_cast<std::size_t>(i);
            const std::size_t hi = static_cast<std::size_t>(n - 1 - i);
            PANTR_CHECK_MSG(nodes[lo] == -nodes[hi],
                            "n=" + std::to_string(n) + " i=" + std::to_string(i));
            PANTR_CHECK_MSG(weights[lo] == weights[hi],
                            "n=" + std::to_string(n) + " i=" + std::to_string(i));
        }
    }
}

/// Central node for odd `n`: checks the structural property behind it, not
/// its value. See the file header for why an exact-`0.0` claim is a claim
/// about this machine's libm rather than about the kernel, and for what this
/// replaces.
///
/// `gauss_legendre_symmetric` writes each root's mirror pair as
/// `out_nodes[i] = -x` then `out_nodes[n - 1 - i] = x`, for `i` up to
/// `half - 1`. At the middle slot of an odd-length array that last `i` and
/// `n - 1 - i` are the same index, so both writes land there and only the
/// second -- the positive root -- survives. That index identity is checked
/// directly and arithmetically here, rather than inferred from the value it
/// produces: it holds for every odd `n` on every platform, with no
/// measured-safe range to maintain.
void gl_central_write_index_coincides_for_odd_n() {
    for (int n = 1; n <= 301; n += 2) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::gauss_legendre_symmetric(n, nodes, weights);

        const int half = (n + 1) / 2;
        const int mirror_index = half - 1;
        PANTR_CHECK_MSG(mirror_index == n - 1 - mirror_index,
                        "n=" + std::to_string(n) + " mirror_index=" + std::to_string(mirror_index));
    }
}

/// Weights strictly positive; sum to 2 to the sum-of-positive-terms bound.
void gl_weights_positive_and_sum_to_two() {
    for (int n = 1; n <= 300; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::gauss_legendre_symmetric(n, nodes, weights);

        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            PANTR_CHECK_MSG(weights[static_cast<std::size_t>(i)] > 0.0,
                            "n=" + std::to_string(n) + " i=" + std::to_string(i));
            sum += weights[static_cast<std::size_t>(i)];
        }
        const double bound = kSafety * 2.0 * gamma_bound(n);
        PANTR_CHECK_MSG(std::abs(sum - 2.0) <= bound,
                        "n=" + std::to_string(n) + ": |sum-2|=" + std::to_string(std::abs(sum - 2.0)));
    }
}

/// `n == 1`: node 0, weight 2, exactly.
void gl_n_equals_one_is_exact() {
    std::vector<double> nodes(1);
    std::vector<double> weights(1);
    pantr::gauss_legendre_symmetric(1, nodes, weights);
    PANTR_CHECK(nodes[0] == 0.0);
    PANTR_CHECK(weights[0] == 2.0);
}

// ---------------------------------------------------------------------------
// Lambert W
// ---------------------------------------------------------------------------

/// Residual `w*e^w - x`, over the argument range the rule uses (smallest 1.885
/// at `n = 2`) and well beyond it.
///
/// Bound: the header states the returned `w` is accurate to "about one unit of
/// roundoff" of the true `w_true`, i.e. `|w - w_true| <= C*u*w_true` for a
/// modest constant; `C = 8` here is `kSafety` doubled again to turn "about
/// one" into an explicit multiple, in the same spirit as this file's other
/// margins. By the mean value theorem, `w*e^w - x = F(w) - F(w_true)` for
/// `F(z) = z e^z`, equal to `F'(xi)(w - w_true)` for some `xi` between them,
/// and `F'(z) = e^z(z+1)`. Since `w` and `w_true` agree to `O(u)` relatively,
/// `F'(xi) ~ F'(w_true) = e^{w_true}(w_true+1) = (x/w_true)(w_true+1)` (from
/// `x = w_true e^{w_true}`). Substituting the assumed bound on `|w - w_true|`:
///
///     |residual| <~ C*u*w_true * (x/w_true)*(w_true+1) = C*u*x*(w_true+1)
///
/// and using the computed `w` in place of the unknown `w_true` (justified
/// since they differ only at `O(u)` relatively, negligible at this order).
void lambert_w_residual() {
    const double u = std::numeric_limits<double>::epsilon() / 2.0;
    constexpr double kC = 8.0;
    const std::vector<double> xs = {1.885, 2.0,  3.0,  5.0,   10.0,  50.0, 100.0,
                                    1e3,   1e4,  1e6,  1e9,   1e15,  1e50, 1e100, 1e300};
    for (const double x : xs) {
        const double w = pantr::lambert_w_principal(x);
        const double residual = std::abs(w * std::exp(w) - x);
        const double tol = kC * u * x * (w + 1.0);
        PANTR_CHECK_MSG(residual <= tol, "x=" + std::to_string(x) + ": residual=" +
                                              std::to_string(residual) + " tol=" +
                                              std::to_string(tol));
    }
}

// ---------------------------------------------------------------------------
// tanh-sinh
// ---------------------------------------------------------------------------

/// Returned count is at most `n` and at least 1.
void ts_count_is_bounded() {
    const double min_gap = std::numeric_limits<double>::epsilon();
    for (int n = 1; n <= 300; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        const std::size_t m = pantr::generate_tanh_sinh(n, min_gap, nodes, weights);
        PANTR_CHECK_MSG(m >= 1 && m <= static_cast<std::size_t>(n), "n=" + std::to_string(n));
    }
}

/// Every returned node is strictly interior to `(-1, 1)`, and its gap to the
/// nearer endpoint is at least `min_gap` -- the function's advertised contract.
void ts_nodes_interior_with_min_gap() {
    const double min_gap = std::numeric_limits<double>::epsilon();
    for (int n = 1; n <= 300; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        const std::size_t m = pantr::generate_tanh_sinh(n, min_gap, nodes, weights);
        for (std::size_t j = 0; j < m; ++j) {
            const double x = nodes[j];
            PANTR_CHECK_MSG(x > -1.0 && x < 1.0, "n=" + std::to_string(n));
            const double gap = 1.0 - std::abs(x);
            PANTR_CHECK_MSG(gap >= min_gap, "n=" + std::to_string(n) + " x=" + std::to_string(x));
        }
    }
}

/// Nodes are exact negations in pairs, bit for bit. Unlike Gauss-Legendre, both
/// members of every pair -- including the central one for odd `n`, set to the
/// literal `0.0` -- are written directly from a shared quantity, so this holds
/// unconditionally rather than in a measured-safe range.
void ts_nodes_are_exact_negations() {
    const double min_gap = std::numeric_limits<double>::epsilon();
    for (int n = 1; n <= 300; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        const std::size_t m = pantr::generate_tanh_sinh(n, min_gap, nodes, weights);

        std::size_t start = 0;
        if (n % 2 != 0) {
            PANTR_CHECK_MSG(nodes[0] == 0.0, "n=" + std::to_string(n));
            start = 1;
        }
        for (std::size_t j = start; j + 1 < m; j += 2) {
            PANTR_CHECK_MSG(nodes[j] == -nodes[j + 1],
                            "n=" + std::to_string(n) + " j=" + std::to_string(j));
        }
    }
}

/// Weights strictly positive, summing to 2 up to the rescaling's own rounding.
///
/// Bound: this is purely a rescaling-consistency argument, independent of the
/// weight formula (per the header: "dividing by a computed sum cannot make a
/// floating-point sum exact"). Raw weights `r_j` sum to `total` with relative
/// error `gamma(m)`; `scale = fl(2/total)` adds one more division (`+1`); each
/// output `fl(r_j * scale)` adds one multiplication per term; resumming the
/// `m` outputs for this check adds another `gamma(m)`. Folding all of that into
/// one operation count gives `gamma(2m + 2)`, applied to the target magnitude 2.
void ts_weights_positive_and_sum_to_two() {
    const double min_gap = std::numeric_limits<double>::epsilon();
    for (int n = 1; n <= 300; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        const std::size_t m = pantr::generate_tanh_sinh(n, min_gap, nodes, weights);

        double sum = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            PANTR_CHECK_MSG(weights[j] > 0.0, "n=" + std::to_string(n) + " j=" + std::to_string(j));
            sum += weights[j];
        }
        const double bound = kSafety * 2.0 * gamma_bound(2 * static_cast<int>(m) + 2);
        PANTR_CHECK_MSG(std::abs(sum - 2.0) <= bound,
                        "n=" + std::to_string(n) + ": |sum-2|=" + std::to_string(std::abs(sum - 2.0)));
    }
}

/// The transform-space geometry of the first node `generate_tanh_sinh` did
/// *not* write: `omega_stop`, the transform coordinate `omega(t) = (pi/2)
/// sinh(t)` at that point, and `d_omega`, the local rate of change of `omega`
/// over one step (`h * (pi/2) * cosh(t_stop)`). Both feed the truncation terms
/// in `ts_integrates_endpoint_singular_function` and
/// `ts_integrates_smooth_function_accurately`.
///
/// Recovered from the rule's own output, not from re-deriving the kernel's own
/// `h` -- doing that here would just mirror `generate_tanh_sinh`'s formula
/// instead of testing it. `gap = 2 / (1 + exp(2*omega))` inverts to
/// `omega = 0.5 * log(2/gap - 1)`; the last two written node pairs give two
/// values of `t = asinh(omega / (pi/2))`, whose difference is the uniform step
/// `h`, and the first unwritten point sits one step past the last one.
///
/// Needs at least two full node pairs to have two `t` values to difference:
/// `m >= 5` for odd `n`, `m >= 4` for even `n`. Both integration checks below
/// assert `m >= 5` before calling this.
struct TanhSinhTailGeometry {
    double omega_stop;
    double d_omega;
};

TanhSinhTailGeometry tanh_sinh_tail_geometry(const std::vector<double>& nodes, std::size_t m) {
    const double half_pi = 0.5 * std::numbers::pi;
    const auto omega_from_gap = [](double gap) { return 0.5 * std::log(2.0 / gap - 1.0); };
    const double g_last = 1.0 - std::abs(nodes[m - 1]);
    const double g_prev = 1.0 - std::abs(nodes[m - 3]);
    const double t_last = std::asinh(omega_from_gap(g_last) / half_pi);
    const double t_prev = std::asinh(omega_from_gap(g_prev) / half_pi);
    const double h = t_last - t_prev;
    const double t_stop = t_last + h;
    return {half_pi * std::sinh(t_stop), h * half_pi * std::cosh(t_stop)};
}

/// Exact (not linearized) relative sensitivity of `f(x) = 1/sqrt(1-x^2)` to the
/// half-ulp rounding in the kernel's `x = fl(1 - gap)`, at node gap `g`. Used
/// by `ts_integrates_endpoint_singular_function`'s node-quantization term.
///
/// The linear approximation `u/(4*g)` is *not* a strict bound on this quantity
/// -- it was checked and found to exceed the truth by 24% at `g = 2u` and up
/// to 66% as `g -> u` -- so both perturbed values of `gap` are evaluated
/// directly here and the larger deviation kept, rather than linearizing.
double endpoint_gap_relative_sensitivity(double g) {
    double rel = 0.0;
    for (const double sign : {1.0, -1.0}) {
        const double g_plus = g + sign * kU;
        const double g_minus = 2.0 - g - sign * kU;
        const double ratio = std::sqrt(g * (2.0 - g) / (g_plus * g_minus));
        rel = std::max(rel, std::abs(ratio - 1.0));
    }
    return rel;
}

/// Smooth integrand (`exp`), rounding plus truncation. The rounding term is as
/// before (`p = 3` per term: the `exp` call, taken as up to 2 ulp per
/// tanh_sinh.hpp's own convention for related transcendentals, plus the weight
/// multiplication; `+m` for the summation; `sum_j w_j e^{x_j} <= e * sum_j w_j
/// ~= 2e`, rounded up to 6 as the magnitude bound). The truncation term is new
/// -- see the file header for the measured ratios that show it was needed
/// below about `n = 35`.
///
/// Derivation of the truncation term: `cosh(y) >= e^y/2` for `y >= 0` bounds
/// the node weight exactly, `w(t) <= 2*pi*cosh(t)*exp(-2*omega(t))`. Unlike
/// the endpoint-singular integrand, `f = exp` has no matching blowup to cancel
/// one power of that decay -- only `|f(x)| <= M` on `[-1, 1]` is available
/// (`M = e`, at `x = 1`). Using `exp(-2*omega) <= exp(-omega)` for
/// `omega >= 0` relaxes the tighter decay to the same `exp(-omega)` form the
/// endpoint-singular term telescopes on, at the cost of an extra factor of 2
/// in the per-node bound (`2*pi*M*cosh(t)*exp(-omega(t))`, vs. that term's
/// `pi*cosh(t)*exp(-omega(t))`); the same telescoping and
/// discretization-correction argument then gives `8*M*(1+dOmega)*exp(-omega_stop)`
/// where that term has `4*(...)`.
///
/// This is deliberately loose, with no cancellation to exploit -- and the
/// measured ratios confirm it: worst observed ratio of error to the bare
/// (pre-`kSafety`) bound over `n` in {5, 11, 21, 31, 401, 1001} is 0.0076, at
/// `n = 5`. Never violated over that range; never approached either, which is
/// what a truncation term added for validity rather than tightness looks like.
void ts_integrates_smooth_function_accurately() {
    const double min_gap = std::numeric_limits<double>::epsilon();
    const double max_abs_exp_on_unit_interval = std::numbers::e;
    for (const int n : {5, 11, 21, 31, 401, 1001}) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        const std::size_t m = pantr::generate_tanh_sinh(n, min_gap, nodes, weights);
        PANTR_CHECK_MSG(m >= 5u, "n=" + std::to_string(n) + ": too few nodes for tail geometry");

        double approx = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            approx += weights[j] * std::exp(nodes[j]);
        }
        const double exact = std::exp(1.0) - std::exp(-1.0);

        const TanhSinhTailGeometry geom = tanh_sinh_tail_geometry(nodes, m);
        const double round_term = 6.0 * gamma_bound(static_cast<int>(m) + 3);
        const double trunc_term = 8.0 * max_abs_exp_on_unit_interval * (1.0 + geom.d_omega) *
                                   std::exp(-geom.omega_stop);
        const double bound = kSafety * (round_term + trunc_term);
        PANTR_CHECK_MSG(std::abs(approx - exact) <= bound,
                        "n=" + std::to_string(n) + ": |approx-exact|=" +
                            std::to_string(std::abs(approx - exact)) +
                            " bound=" + std::to_string(bound));
    }
}

/// Endpoint-singular integrand, `1 / sqrt(1 - x^2)`, exact value `pi`. The
/// interesting case for a double-exponential rule: truncation, not rounding,
/// dominates. This bound -- `kSafety * (a + b + c)` below -- was derived and
/// verified outside this file (a tolerance audit found the previous flat
/// `1e-6` constant violated below `n = 30`, and within 11% of failing at
/// `n = 31`, passing only because the test exercised `n = 41` alone) and is
/// implemented here as given, not re-derived.
///
/// (a) Truncation of the (doubly-infinite) transform-space sum past the last
/// written node. `w(t) <= 2*pi*cosh(t)*exp(-2*omega(t))` as in the smooth
/// case's derivation, but here `f(x) = 1/sqrt(1-x^2)` blows up as `gap -> 0`
/// (`gap ~ 2*exp(-2*omega)`, so `f ~ exp(omega)/2`), and that cancels one
/// power of the weight's decay: the term `w(t)*f(x(t))` is asymptotically
/// `~ pi*cosh(t)*exp(-omega(t))`. Since
/// `d/dt exp(-omega(t)) = -(pi/2)*cosh(t)*exp(-omega(t))`, the one-sided tail
/// integral from `t_stop` telescopes to `2*exp(-omega_stop)`; doubling for the
/// two symmetric tails and folding in the standard sum-vs-integral correction
/// for step `h` (`dOmega`, from `tanh_sinh_tail_geometry`) gives
/// `4*(1+dOmega)*exp(-omega_stop)`.
///
/// (b) Node quantization: `sum_j T_j * rel_j`, `T_j = w_j*f(x_j)` the term as
/// actually formed, `rel_j` from `endpoint_gap_relative_sensitivity` -- the
/// exact relative change in `f` from perturbing the node's gap by half a ulp
/// either way, which is what recovering `gap = 1 - |x|` from the stored
/// `x = fl(1 - gap)` actually costs.
///
/// (c) Formation and summation of `m` terms: `pi*gamma(m+4)`, `pi` bounding
/// the sum of exact term magnitudes (loose here; this term is not the
/// dominant one).
///
/// Verified worst ratio of observed error to the bare (pre-`kSafety`) bound
/// over `n` in {5, 11, 21, 31, 401, 1001}: 0.79 (at `n = 21` and `n = 31`); an
/// exploratory sweep to `n = 20000` found 0.84. `kSafety = 2` more than halves
/// the worst case.
void ts_integrates_endpoint_singular_function() {
    const double min_gap = std::numeric_limits<double>::epsilon();
    for (const int n : {5, 11, 21, 31, 401, 1001}) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        const std::size_t m = pantr::generate_tanh_sinh(n, min_gap, nodes, weights);
        PANTR_CHECK_MSG(m >= 5u, "n=" + std::to_string(n) + ": too few nodes for tail geometry");

        double approx = 0.0;
        double term_b = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            const double x = nodes[j];
            const double f = 1.0 / std::sqrt(1.0 - x * x);
            const double term = weights[j] * f;
            approx += term;
            const double g = 1.0 - std::abs(x);
            term_b += term * endpoint_gap_relative_sensitivity(g);
        }
        const double exact = std::numbers::pi;

        const TanhSinhTailGeometry geom = tanh_sinh_tail_geometry(nodes, m);
        const double term_a = 4.0 * (1.0 + geom.d_omega) * std::exp(-geom.omega_stop);
        const double term_c = std::numbers::pi * gamma_bound(static_cast<int>(m) + 4);
        const double bound = kSafety * (term_a + term_b + term_c);
        PANTR_CHECK_MSG(std::abs(approx - exact) <= bound,
                        "n=" + std::to_string(n) + ": |approx-exact|=" +
                            std::to_string(std::abs(approx - exact)) +
                            " bound=" + std::to_string(bound));
    }
}

// ---------------------------------------------------------------------------
// trapezoidal
// ---------------------------------------------------------------------------

/// First node exactly 0.0, last exactly 1.0 -- the last is *assigned*, not
/// computed, per simple_rules.hpp, so this is a bit-equality check. `n == 1` is
/// the midpoint rule (node 0.5), not this case, and is checked separately.
void trap_endpoints_are_exact() {
    for (int n = 2; n <= 500; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::trapezoidal(n, nodes, weights);
        PANTR_CHECK_MSG(nodes.front() == 0.0, "n=" + std::to_string(n));
        PANTR_CHECK_MSG(nodes.back() == 1.0, "n=" + std::to_string(n));
    }
}

/// `n == 1` is the midpoint rule: node 0.5, weight 1.0, exactly.
void trap_n_equals_one_is_midpoint() {
    std::vector<double> nodes(1);
    std::vector<double> weights(1);
    pantr::trapezoidal(1, nodes, weights);
    PANTR_CHECK(nodes[0] == 0.5);
    PANTR_CHECK(weights[0] == 1.0);
}

/// Weights sum to exactly 1 for the composite rule, to a derived tolerance; the
/// two endpoint weights are exactly half the shared interior weight, `step`.
///
/// `step = 1.0 / (n - 1)` is recomputed independently here, from the same
/// inputs and the same single division `simple_rules.hpp` performs, so IEEE
/// division being deterministic makes it bit-identical to the kernel's own
/// `step` -- not a tolerance-bearing comparison. For `n == 2` there is no
/// distinct interior index (both entries are endpoints), so only the
/// endpoint-vs-`step` check applies; `n >= 3` additionally pins every interior
/// entry to `step` itself.
///
/// Bound on the sum: `n` terms of a positive sum (`gamma(n)`) plus the one
/// division that forms `step` (`+1`), against the target magnitude 1.
void trap_weights_halve_at_endpoints_and_sum_to_one() {
    for (int n = 2; n <= 500; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::trapezoidal(n, nodes, weights);

        const double step = 1.0 / static_cast<double>(n - 1);
        PANTR_CHECK_MSG(weights.front() == 0.5 * step, "n=" + std::to_string(n));
        PANTR_CHECK_MSG(weights.back() == 0.5 * step, "n=" + std::to_string(n));
        for (int i = 1; i < n - 1; ++i) {
            PANTR_CHECK_MSG(weights[static_cast<std::size_t>(i)] == step,
                            "n=" + std::to_string(n) + " i=" + std::to_string(i));
        }

        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += weights[static_cast<std::size_t>(i)];
        }
        const double bound = kSafety * 1.0 * gamma_bound(n + 1);
        PANTR_CHECK_MSG(std::abs(sum - 1.0) <= bound,
                        "n=" + std::to_string(n) + ": |sum-1|=" + std::to_string(std::abs(sum - 1.0)));
    }
}

/// Exact for a linear function, `f(t) = a + b*t`, regardless of node spacing --
/// the well-known exactness class of the trapezoidal rule.
///
/// Bound: per term, evaluating `f` (mult + add, `p=2`) and multiplying by the
/// weight (`+1`) gives `p=3`; `+n` for the summation. `|f(t)| <= |a| + |b|` on
/// `[0, 1]` bounds the bound's per-term magnitude, and the weights sum to
/// (approximately) 1, so `|a| + |b|` bounds the sum of exact-term magnitudes.
void trap_integrates_linear_function_exactly() {
    constexpr double a = 2.0;
    constexpr double b = -3.0;
    for (int n = 2; n <= 500; ++n) {
        std::vector<double> nodes(static_cast<std::size_t>(n));
        std::vector<double> weights(static_cast<std::size_t>(n));
        pantr::trapezoidal(n, nodes, weights);

        double approx = 0.0;
        for (int i = 0; i < n; ++i) {
            const double t = nodes[static_cast<std::size_t>(i)];
            approx += weights[static_cast<std::size_t>(i)] * (a + b * t);
        }
        const double exact = a + b * 0.5;
        const double bound = kSafety * (std::abs(a) + std::abs(b)) * gamma_bound(n + 3);
        PANTR_CHECK_MSG(std::abs(approx - exact) <= bound,
                        "n=" + std::to_string(n) + ": |approx-exact|=" +
                            std::to_string(std::abs(approx - exact)));
    }
}

// ---------------------------------------------------------------------------
// modified Chebyshev nodes
// ---------------------------------------------------------------------------

/// `n` values safely below where float loses the resolution to keep the nodes
/// strictly ascending (measured: the first tie is at `n = 7431` for `float`; every
/// value below is comfortably inside the safe range for both `T`).
const std::vector<int>& chebyshev_test_sizes() {
    static const std::vector<int> sizes = {2,  3,   4,   5,   6,   7,  9,
                                            10, 16,  17,  20,  32,  50,
                                            64, 100, 200, 257, 500, 1000};
    return sizes;
}

/// Endpoints, ascending order, and symmetry about 0.5, at one `T`.
///
/// The endpoints are bit-exact for a derived reason, not by construction like
/// trapezoidal's assigned endpoint: at `index = 0` the angle is exactly `0`,
/// and `cos(0) = 1` exactly, so `out[0] = half - half*1 = 0` exactly. At the
/// last index, `index` and `divisor` are the identical `T` value, so the angle
/// is `pi*x/x` for `x = n-1`; `cos` is stationary at `pi` (`cos'(pi) = 0`), so
/// any ULP-scale perturbation of the argument there still rounds `cos` back to
/// the same nearest representable value, `-1`, and `out[n-1] = half - half*(-1)
/// = 1` exactly. Confirmed for `n` from 2 to 1000 at both `double` and `float`.
///
/// Symmetry is not bit-exact (nothing in the kernel forces it: the two angles
/// are computed independently, not as one value and its complement). Bound:
/// per node, forming the angle (`pi * index / divisor`, 2 ops, plus `pi`'s own
/// representation rounding) and `cos`'s own $\le$ 1 ULP libm rounding give an
/// absolute error in `cos` of about `pi * gamma(3) + u`; since `|cos'| <= 1`
/// this also bounds the propagated error into the angle-to-cos step, and the
/// final `half - half*cos` (2 more ops) adds another `gamma(2)`-scale term.
/// Two independent nodes are combined for the symmetry sum. Measured worst
/// case across the sizes below and both types: ~1.5 `eps`; `16 * eps` is a
/// order-of-magnitude margin over that, not a fitted value -- it is the
/// rounded-up sum of the per-node budget above, doubled for two nodes.
template <class T>
void chebyshev_endpoints_ascending_and_symmetric() {
    const T eps = std::numeric_limits<T>::epsilon();
    const T bound = static_cast<T>(16) * eps;

    for (const int n : chebyshev_test_sizes()) {
        std::vector<T> out(static_cast<std::size_t>(n));
        pantr::modified_chebyshev_nodes<T>(n, out);

        PANTR_CHECK_MSG(out.front() == T(0), "n=" + std::to_string(n));
        PANTR_CHECK_MSG(out.back() == T(1), "n=" + std::to_string(n));

        for (int i = 1; i < n; ++i) {
            PANTR_CHECK_MSG(out[static_cast<std::size_t>(i)] >
                                 out[static_cast<std::size_t>(i - 1)],
                            "n=" + std::to_string(n) + " i=" + std::to_string(i));
        }

        for (int i = 0; i < n / 2; ++i) {
            const T sum = out[static_cast<std::size_t>(i)] +
                          out[static_cast<std::size_t>(n - 1 - i)];
            const T err = std::abs(sum - T(1));
            PANTR_CHECK_MSG(err <= bound, "n=" + std::to_string(n) + " i=" + std::to_string(i));
        }
    }
}

}  // namespace

int main() {
    gl_integrates_polynomials_exactly();
    gl_nodes_ascending_and_interior();
    gl_off_center_symmetry_is_bitexact();
    gl_central_write_index_coincides_for_odd_n();
    gl_weights_positive_and_sum_to_two();
    gl_n_equals_one_is_exact();

    lambert_w_residual();

    ts_count_is_bounded();
    ts_nodes_interior_with_min_gap();
    ts_nodes_are_exact_negations();
    ts_weights_positive_and_sum_to_two();
    ts_integrates_smooth_function_accurately();
    ts_integrates_endpoint_singular_function();

    trap_endpoints_are_exact();
    trap_n_equals_one_is_midpoint();
    trap_weights_halve_at_endpoints_and_sum_to_one();
    trap_integrates_linear_function_exactly();

    chebyshev_endpoints_ascending_and_symmetric<double>();
    chebyshev_endpoints_ascending_and_symmetric<float>();

    return pantr::test::summary("test_quad");
}
