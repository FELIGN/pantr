/// \file
/// The tensor-product extraction kernels, checked against a materialised
/// Kronecker product.
///
/// The kernels exist so that `M = kron(M_0, ..., M_{d-1})` is never formed. The
/// only oracle that does not share their reasoning is therefore the one that
/// forms it: build `M` explicitly, multiply, and compare. At the sizes here that
/// is a few hundred elements, and it is independent of every choice the kernels
/// make -- the mode order, the ping-pong, the identity short-circuit.
///
/// ## Two claims, and only one of them needs a tolerance
///
/// **Exact.** With small-integer operator entries and an integer operand, every
/// product and every partial sum is representable in `double`, so both the kernel
/// and the materialised reference are exact and must agree **bit for bit**. This
/// is the claim that catches a transposed index or a wrong mode order, which no
/// tolerance can see because a wrong index order is not a small perturbation. Any
/// failure here is structural.
///
/// **Bounded.** With arbitrary entries the two sum the same terms in different
/// orders, so they differ by rounding. The bound is the standard inner-product
/// one, `gamma_n = n u / (1 - n u)`, composed over the stages, applied to the
/// elementwise magnitude reachable at each output -- which is what running the
/// same kernel on the absolute values gives. `n` is the total contraction length
/// along the chain, not the operator size, and it is counted from the identity
/// flags because an identity direction contracts nothing at all.
///
/// ## Why the sweep counts its own disagreements
///
/// A bound compared only against zero has not been checked, and the exact half of
/// this file arranges exactly that on purpose. So the bounded half counts how many
/// comparisons it made, how many actually disagreed, and the worst deviation as a
/// fraction of its own bound, and refuses all three degenerate outcomes: no
/// comparison, no disagreement, and a worst ratio so small that the bound is not
/// describing the arithmetic it claims to.
///
/// The `1e-3` floor on that ratio is a **heuristic**, not a derivation: it is
/// loose enough that the composed `gamma_n` bound (which is a worst case over
/// sign patterns that random draws do not reach) passes comfortably, and tight
/// enough that a bound accidentally inflated by orders of magnitude would not.
/// A derived floor would need the expected cancellation of the draw, which is a
/// property of the test data rather than of the kernel.
///
/// ## What the identity flag is checked to mean
///
/// `poison` fills a flagged-identity operator with values that would be visible in
/// the answer if it were read. The test then asserts the answer is the one the
/// identity gives. A kernel that read the matrix behind the flag would fail by a
/// wide margin rather than by a rounding.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/extraction_kernels.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

using pantr::span2d;
using pantr::bspline::ModeOperator;

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// How many bounded comparisons ran, how many of them actually disagreed, and the
/// worst deviation as a fraction of its own bound.
///
/// The three together are what stop the bounded half of this file from passing for
/// the wrong reason: a bound nothing approaches is not evidence, and a bound
/// nothing even evaluates is not a test.
std::size_t bounded_comparisons = 0;
std::size_t nonzero_deviations = 0;
double worst_ratio = 0.0;

/// Record one bounded comparison.
///
/// \param error The observed absolute deviation.
/// \param bound Its bound.
void record(double error, double bound) {
    ++bounded_comparisons;
    if (error > 0.0) {
        ++nonzero_deviations;
    }
    if (bound > 0.0) {
        const double ratio = error / bound;
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
        }
    }
}

/// A dense matrix with its shape, so a test can hand one around by value.
struct Dense {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> values;

    [[nodiscard]] double at(std::size_t i, std::size_t j) const { return values[i * cols + j]; }

    [[nodiscard]] span2d<const double> view() const {
        return span2d<const double>(values.data(), rows, cols);
    }
};

/// A small deterministic generator, so a failure is reproducible from the source.
///
/// A 64-bit LCG rather than `<random>`: the sequence has to be the same on every
/// libstdc++ and libc++ this is built with, and the distribution engines are not
/// specified to produce identical streams across implementations.
class Lcg {
  public:
    explicit Lcg(std::uint64_t seed) : state_(seed) {}

    /// \return The next value, uniform in `[-1, 1)`.
    double next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const auto mantissa = static_cast<double>(state_ >> 11U);
        return 2.0 * (mantissa / 9007199254740992.0) - 1.0;
    }

    /// \param magnitude The largest absolute value to return.
    /// \return The next value, an integer in `[-magnitude, magnitude]`.
    double next_small_integer(int magnitude) {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const auto span = static_cast<std::uint64_t>(2 * magnitude + 1);
        return static_cast<double>(static_cast<std::int64_t>((state_ >> 33U) % span) - magnitude);
    }

  private:
    std::uint64_t state_;
};

/// \param rows Row count.
/// \param cols Column count.
/// \param rng The generator.
/// \param integral Whether to draw small integers rather than reals.
/// \return A freshly drawn matrix.
Dense draw(std::size_t rows, std::size_t cols, Lcg& rng, bool integral) {
    Dense m{rows, cols, std::vector<double>(rows * cols)};
    for (double& value : m.values) {
        value = integral ? rng.next_small_integer(3) : rng.next();
    }
    return m;
}

/// \param rows Row count and column count.
/// \return The identity of that size.
Dense identity(std::size_t rows) {
    Dense m{rows, rows, std::vector<double>(rows * rows, 0.0)};
    for (std::size_t i = 0; i < rows; ++i) {
        m.values[i * rows + i] = 1.0;
    }
    return m;
}

/// \param a Left factor.
/// \param b Right factor.
/// \return `kron(a, b)`, materialised.
Dense kron(const Dense& a, const Dense& b) {
    Dense m{a.rows * b.rows, a.cols * b.cols, std::vector<double>(a.rows * b.rows * a.cols * b.cols)};
    for (std::size_t i = 0; i < a.rows; ++i) {
        for (std::size_t j = 0; j < a.cols; ++j) {
            for (std::size_t k = 0; k < b.rows; ++k) {
                for (std::size_t l = 0; l < b.cols; ++l) {
                    m.values[(i * b.rows + k) * m.cols + (j * b.cols + l)] = a.at(i, j) * b.at(k, l);
                }
            }
        }
    }
    return m;
}

/// \param factors The per-direction matrices, outermost first.
/// \return Their Kronecker product, materialised.
Dense kron_all(const std::vector<Dense>& factors) {
    Dense m = factors.front();
    for (std::size_t k = 1; k < factors.size(); ++k) {
        m = kron(m, factors[k]);
    }
    return m;
}

/// \param m The matrix.
/// \param v The vector.
/// \return `m @ v`.
std::vector<double> matvec(const Dense& m, std::span<const double> v) {
    std::vector<double> result(m.rows, 0.0);
    for (std::size_t i = 0; i < m.rows; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < m.cols; ++j) {
            acc += m.at(i, j) * v[j];
        }
        result[i] = acc;
    }
    return result;
}

/// \param m The matrix.
/// \param v The vector.
/// \return `m^T @ v`.
std::vector<double> matvec_transpose(const Dense& m, std::span<const double> v) {
    std::vector<double> result(m.cols, 0.0);
    for (std::size_t j = 0; j < m.cols; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < m.rows; ++i) {
            acc += m.at(i, j) * v[i];
        }
        result[j] = acc;
    }
    return result;
}

/// \param a Left factor.
/// \param b Right factor.
/// \return `a @ b`.
Dense matmul(const Dense& a, const Dense& b) {
    Dense m{a.rows, b.cols, std::vector<double>(a.rows * b.cols, 0.0)};
    for (std::size_t i = 0; i < a.rows; ++i) {
        for (std::size_t j = 0; j < b.cols; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < a.cols; ++k) {
                acc += a.at(i, k) * b.at(k, j);
            }
            m.values[i * m.cols + j] = acc;
        }
    }
    return m;
}

/// \param m The matrix.
/// \return Its transpose.
Dense transpose(const Dense& m) {
    Dense t{m.cols, m.rows, std::vector<double>(m.values.size())};
    for (std::size_t i = 0; i < m.rows; ++i) {
        for (std::size_t j = 0; j < m.cols; ++j) {
            t.values[j * t.cols + i] = m.at(i, j);
        }
    }
    return t;
}

/// Replace an operator's entries with values that could not go unnoticed.
///
/// Used on a direction flagged identity: if the kernel read the matrix, the answer
/// would move by a large multiple of itself rather than by a rounding.
///
/// \param m The matrix to overwrite.
void poison(Dense& m) {
    for (double& value : m.values) {
        value = -1000.0;
    }
}

/// \param ops The per-direction operators as dense matrices.
/// \param flags Which directions are the identity.
/// \return The `ModeOperator` list the kernels take.
std::vector<ModeOperator<double>> as_mode_operators(const std::vector<Dense>& ops,
                                                    const std::vector<bool>& flags) {
    std::vector<ModeOperator<double>> mode_ops;
    mode_ops.reserve(ops.size());
    for (std::size_t k = 0; k < ops.size(); ++k) {
        mode_ops.push_back(ModeOperator<double>{ops[k].view(), flags[k]});
    }
    return mode_ops;
}

/// The elementwise bound of this file's header, for a chain of contractions.
///
/// \param lengths The contraction length of each stage that actually runs.
/// \param magnitude The elementwise magnitude reachable at the output, i.e. the
///        same computation run on absolute values.
/// \return The elementwise absolute bound.
std::vector<double> chain_bound(std::span<const std::size_t> lengths,
                                std::span<const double> magnitude) {
    double growth = 1.0;
    for (const std::size_t n : lengths) {
        const double gamma = (static_cast<double>(n) * kEps) / (1.0 - static_cast<double>(n) * kEps);
        growth *= (1.0 + gamma);
    }
    growth -= 1.0;
    // Two backends, each within its own bound of the exact result.
    growth *= 2.0;

    std::vector<double> bound(magnitude.size());
    for (std::size_t i = 0; i < magnitude.size(); ++i) {
        bound[i] = growth * magnitude[i];
    }
    return bound;
}

/// \param ops The per-direction operators.
/// \return Their elementwise absolute values.
std::vector<Dense> absolute(const std::vector<Dense>& ops) {
    std::vector<Dense> result = ops;
    for (Dense& m : result) {
        for (double& value : m.values) {
            value = value < 0.0 ? -value : value;
        }
    }
    return result;
}

/// \param v A vector.
/// \return Its elementwise absolute value.
std::vector<double> absolute(std::span<const double> v) {
    std::vector<double> result(v.begin(), v.end());
    for (double& value : result) {
        value = value < 0.0 ? -value : value;
    }
    return result;
}

}  // namespace

namespace {

/// Every identity pattern over `d` directions, as a bit mask.
///
/// \param d The number of directions.
/// \param mask The pattern.
/// \return The per-direction identity flags.
std::vector<bool> flags_from(std::size_t d, unsigned mask) {
    std::vector<bool> flags(d, false);
    for (std::size_t k = 0; k < d; ++k) {
        flags[k] = ((mask >> k) & 1U) != 0U;
    }
    return flags;
}

/// The contraction lengths of the stages a unilateral apply actually runs.
///
/// \param ops The operators.
/// \param flags The identity flags.
/// \param transposed Whether the transpose is applied.
/// \return One length per stage that runs.
std::vector<std::size_t> unilateral_lengths(const std::vector<Dense>& ops,
                                            const std::vector<bool>& flags, bool transposed) {
    std::vector<std::size_t> lengths;
    for (std::size_t k = 0; k < ops.size(); ++k) {
        if (!flags[k]) {
            lengths.push_back(transposed ? ops[k].rows : ops[k].cols);
        }
    }
    return lengths;
}

/// Check `apply_kron` and `apply_kron_transpose` against a materialised product.
///
/// \param d The number of directions.
/// \param integral Whether to draw small integers, which makes the claim exact.
/// \param seed The generator seed.
void check_unilateral(std::size_t d, bool integral, std::uint64_t seed) {
    Lcg rng(seed);
    const std::vector<std::size_t> out_sizes{3, 2, 4};
    const std::vector<std::size_t> in_sizes{4, 2, 3};

    for (unsigned mask = 0; mask < (1U << d); ++mask) {
        const std::vector<bool> flags = flags_from(d, mask);

        std::vector<Dense> ops;
        std::vector<Dense> reference_ops;
        for (std::size_t k = 0; k < d; ++k) {
            if (flags[k]) {
                // An identity direction is square, and its stored matrix is poisoned:
                // the reference uses the identity, the kernel must not read the values.
                Dense poisoned = identity(in_sizes[k]);
                reference_ops.push_back(identity(in_sizes[k]));
                poison(poisoned);
                ops.push_back(poisoned);
            } else {
                Dense drawn = draw(out_sizes[k], in_sizes[k], rng, integral);
                reference_ops.push_back(drawn);
                ops.push_back(drawn);
            }
        }

        const Dense full = kron_all(reference_ops);
        const std::vector<ModeOperator<double>> mode_ops = as_mode_operators(ops, flags);

        std::vector<double> v(full.cols);
        for (double& value : v) {
            value = integral ? rng.next_small_integer(3) : rng.next();
        }

        std::vector<double> out(full.rows, 0.0);
        std::vector<double> scratch(4 * full.rows * full.cols, 0.0);
        pantr::bspline::apply_kron<double>(mode_ops, v, out, scratch);
        const std::vector<double> expected = matvec(full, v);

        const std::vector<Dense> abs_ops = absolute(reference_ops);
        const Dense abs_full = kron_all(abs_ops);
        const std::vector<double> magnitude = matvec(abs_full, absolute(v));
        const std::vector<std::size_t> lengths = unilateral_lengths(ops, flags, false);
        const std::vector<double> bound = chain_bound(lengths, magnitude);

        for (std::size_t i = 0; i < out.size(); ++i) {
            const double deviation = out[i] - expected[i];
            const double error = deviation < 0.0 ? -deviation : deviation;
            if (integral) {
                PANTR_CHECK_MSG(out[i] == expected[i],
                                "apply, exact, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            } else {
                record(error, bound[i]);
                PANTR_CHECK_MSG(error <= bound[i],
                                "apply, bounded, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            }
        }

        // The transpose, on its own operand.
        std::vector<double> w(full.rows);
        for (double& value : w) {
            value = integral ? rng.next_small_integer(3) : rng.next();
        }
        std::vector<double> out_t(full.cols, 0.0);
        pantr::bspline::apply_kron_transpose<double>(mode_ops, w, out_t, scratch);
        const std::vector<double> expected_t = matvec_transpose(full, w);
        const std::vector<double> magnitude_t = matvec_transpose(abs_full, absolute(w));
        const std::vector<std::size_t> lengths_t = unilateral_lengths(ops, flags, true);
        const std::vector<double> bound_t = chain_bound(lengths_t, magnitude_t);

        for (std::size_t i = 0; i < out_t.size(); ++i) {
            const double deviation = out_t[i] - expected_t[i];
            const double error = deviation < 0.0 ? -deviation : deviation;
            if (integral) {
                PANTR_CHECK_MSG(out_t[i] == expected_t[i],
                                "apply_T, exact, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            } else {
                record(error, bound_t[i]);
                PANTR_CHECK_MSG(error <= bound_t[i],
                                "apply_T, bounded, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            }
        }
    }
}

/// Check the two bilateral kernels against a materialised triple product.
///
/// \param d The number of directions.
/// \param integral Whether to draw small integers, which makes the claim exact.
/// \param seed The generator seed.
void check_bilateral(std::size_t d, bool integral, std::uint64_t seed) {
    Lcg rng(seed);
    const std::vector<std::size_t> out_sizes{3, 2, 4};
    const std::vector<std::size_t> in_sizes{4, 2, 3};

    for (unsigned mask = 0; mask < (1U << d); ++mask) {
        const std::vector<bool> flags = flags_from(d, mask);

        std::vector<Dense> ops;
        std::vector<Dense> reference_ops;
        for (std::size_t k = 0; k < d; ++k) {
            if (flags[k]) {
                Dense poisoned = identity(in_sizes[k]);
                reference_ops.push_back(identity(in_sizes[k]));
                poison(poisoned);
                ops.push_back(poisoned);
            } else {
                Dense drawn = draw(out_sizes[k], in_sizes[k], rng, integral);
                reference_ops.push_back(drawn);
                ops.push_back(drawn);
            }
        }

        const Dense full = kron_all(reference_ops);
        const std::vector<ModeOperator<double>> mode_ops = as_mode_operators(ops, flags);
        const std::size_t big = full.rows;
        const std::size_t small = full.cols;

        // out = M^T K M, with K of side `big`.
        Dense k_big = draw(big, big, rng, integral);
        std::vector<double> out_small(small * small, 0.0);
        std::vector<double> scratch(8 * big * big, 0.0);
        pantr::bspline::apply_kron_mt_k_m<double>(mode_ops, k_big.values, out_small, scratch);
        const Dense expected_small = matmul(matmul(transpose(full), k_big), full);
        for (std::size_t i = 0; i < out_small.size(); ++i) {
            if (integral) {
                PANTR_CHECK_MSG(out_small[i] == expected_small.values[i],
                                "MT_K_M, exact, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            } else {
                const double deviation = out_small[i] - expected_small.values[i];
                const double error = deviation < 0.0 ? -deviation : deviation;
                const Dense abs_full = kron_all(absolute(reference_ops));
                Dense abs_k = k_big;
                for (double& value : abs_k.values) {
                    value = value < 0.0 ? -value : value;
                }
                const Dense magnitude = matmul(matmul(transpose(abs_full), abs_k), abs_full);
                std::vector<std::size_t> lengths;
                for (std::size_t k = 0; k < d; ++k) {
                    if (!flags[k]) {
                        lengths.push_back(ops[k].rows);
                        lengths.push_back(ops[k].rows);
                    }
                }
                const std::vector<double> bound = chain_bound(lengths, magnitude.values);
                record(error, bound[i]);
                PANTR_CHECK_MSG(error <= bound[i],
                                "MT_K_M, bounded, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            }
        }

        // out = M K M^T, with K of side `small`.
        Dense k_small = draw(small, small, rng, integral);
        std::vector<double> out_big(big * big, 0.0);
        pantr::bspline::apply_kron_m_k_mt<double>(mode_ops, k_small.values, out_big, scratch);
        const Dense expected_big = matmul(matmul(full, k_small), transpose(full));
        for (std::size_t i = 0; i < out_big.size(); ++i) {
            if (integral) {
                PANTR_CHECK_MSG(out_big[i] == expected_big.values[i],
                                "M_K_MT, exact, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            } else {
                const double deviation = out_big[i] - expected_big.values[i];
                const double error = deviation < 0.0 ? -deviation : deviation;
                const Dense abs_full = kron_all(absolute(reference_ops));
                Dense abs_k = k_small;
                for (double& value : abs_k.values) {
                    value = value < 0.0 ? -value : value;
                }
                const Dense magnitude = matmul(matmul(abs_full, abs_k), transpose(abs_full));
                std::vector<std::size_t> lengths;
                for (std::size_t k = 0; k < d; ++k) {
                    if (!flags[k]) {
                        lengths.push_back(ops[k].cols);
                        lengths.push_back(ops[k].cols);
                    }
                }
                const std::vector<double> bound = chain_bound(lengths, magnitude.values);
                record(error, bound[i]);
                PANTR_CHECK_MSG(error <= bound[i],
                                "M_K_MT, bounded, d=" + std::to_string(d) + " mask=" +
                                    std::to_string(mask) + " i=" + std::to_string(i));
            }
        }
    }
}

}  // namespace

int main() {
    for (std::size_t d = 1; d <= 3; ++d) {
        check_unilateral(d, true, 20260901 + d);
        check_unilateral(d, false, 20260911 + d);
        check_bilateral(d, true, 20260921 + d);
        check_bilateral(d, false, 20260931 + d);
    }

    // A bound compared only against zero has not been checked. The bounded runs
    // above pass trivially if the kernel and the materialised reference happen to
    // agree bit for bit everywhere -- which they would if every draw were exactly
    // representable, and which is exactly what the integral runs arrange on
    // purpose. So the sweep has to be shown to have exercised the bound at all.
    PANTR_CHECK_MSG(bounded_comparisons > 0, "no bounded comparison was made");
    PANTR_CHECK_MSG(nonzero_deviations > 0,
                    "every bounded comparison agreed exactly, so the bound was only ever "
                    "compared against zero and nothing here tested it");
    PANTR_CHECK_MSG(worst_ratio > 1e-3,
                    "the worst observed deviation was " + std::to_string(worst_ratio) +
                        " of its bound, so the bound is far looser than the arithmetic it "
                        "describes and would not notice a real regression");
    PANTR_CHECK_MSG(worst_ratio <= 1.0, "a deviation exceeded its bound");

    std::printf("  bounded comparisons: %zu, of which %zu disagreed; worst ratio %.3f\n",
                bounded_comparisons, nonzero_deviations, worst_ratio);
    return pantr::test::summary("extraction_kernels");
}
