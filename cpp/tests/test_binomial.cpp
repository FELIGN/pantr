/// \file
/// The exact-integer binomial recurrence, against an independently-computed oracle.
///
/// ## Why Pascal's triangle is the oracle
///
/// `bincoeff` runs a multiplicative recurrence, `C(m, i) = C(m - 1, i - 1) * m / i`,
/// and its whole claim is that every one of those divisions is exact. A test that
/// recomputed the same recurrence would agree with a wrong one, so the oracle here
/// builds the triangle by **addition** instead: `C(n, k) = C(n-1, k-1) + C(n-1, k)`.
/// Two different algorithms, and addition needs no exactness argument at all.
///
/// The oracle is itself exact over the whole envelope: the largest coefficient in
/// range is `C(61, 30) = 232714176627630544`, well inside `int64`. So this test
/// covers *every* `(n, k)` with `n <= 61` rather than sampling.
///
/// ## What the other checks would catch
///
/// Symmetry and the row sum are cheap and independent of the oracle: `C(n, k)` must
/// equal `C(n, n - k)`, and a row must sum to `2^n`. The `double` return is exact up
/// to `n = 56` and correctly rounded above it, so the row-sum identity is asserted
/// in integers and the `double` comparison stops where representability does.

#include <cstdint>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/core/binomial.hpp"

namespace {

using pantr::core::bincoeff;
using pantr::core::kBincoeffExactDoubleMaxN;
using pantr::core::kBincoeffMaxN;

/// Pascal's triangle in exact `int64`, rows 0 through `kBincoeffMaxN`.
std::vector<std::vector<std::int64_t>> pascal() {
    std::vector<std::vector<std::int64_t>> rows;
    rows.reserve(static_cast<std::size_t>(kBincoeffMaxN) + 1);
    rows.push_back({1});
    for (int n = 1; n <= kBincoeffMaxN; ++n) {
        const std::vector<std::int64_t>& prev = rows.back();
        std::vector<std::int64_t> row(static_cast<std::size_t>(n) + 1, 1);
        for (int k = 1; k < n; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            row[kk] = prev[kk - 1] + prev[kk];
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

void test_against_pascal() {
    const std::vector<std::vector<std::int64_t>> oracle = pascal();
    for (int n = 0; n <= kBincoeffMaxN; ++n) {
        for (int k = 0; k <= n; ++k) {
            const std::int64_t want = oracle[static_cast<std::size_t>(n)][static_cast<std::size_t>(k)];
            const double got = bincoeff(n, k);
            // Above the double envelope the return is correctly rounded, not exact,
            // so compare against the rounded oracle rather than the integer.
            const double want_as_double = static_cast<double>(want);
            PANTR_CHECK_MSG(got == want_as_double,
                            "C(" + std::to_string(n) + ", " + std::to_string(k) + ") = " +
                                std::to_string(got) + ", oracle " + std::to_string(want));
        }
    }
}

void test_exact_below_the_double_envelope() {
    const std::vector<std::vector<std::int64_t>> oracle = pascal();
    for (int n = 0; n <= kBincoeffExactDoubleMaxN; ++n) {
        for (int k = 0; k <= n; ++k) {
            const std::int64_t want = oracle[static_cast<std::size_t>(n)][static_cast<std::size_t>(k)];
            // Exact means the round trip through double loses nothing.
            PANTR_CHECK_MSG(static_cast<std::int64_t>(bincoeff(n, k)) == want,
                            "C(" + std::to_string(n) + ", " + std::to_string(k) + ") not exact");
        }
    }
}

void test_out_of_range_is_zero() {
    PANTR_CHECK(bincoeff(5, -1) == 0.0);
    PANTR_CHECK(bincoeff(5, 6) == 0.0);
    PANTR_CHECK(bincoeff(0, 1) == 0.0);
    PANTR_CHECK(bincoeff(0, 0) == 1.0);
    // A negative upper index has no coefficients at all: every k is out of [0, n].
    PANTR_CHECK(bincoeff(-1, 0) == 0.0);
}

void test_symmetry() {
    for (int n = 0; n <= kBincoeffMaxN; ++n) {
        for (int k = 0; k <= n; ++k) {
            PANTR_CHECK_MSG(bincoeff(n, k) == bincoeff(n, n - k),
                            "asymmetric at n = " + std::to_string(n) + ", k = " + std::to_string(k));
        }
    }
}

void test_row_sums_are_powers_of_two() {
    // Exact in int64 while 2^n fits, which covers the whole envelope; and exact in
    // double only while every summand is, so the check stops at 56 in that frame.
    for (int n = 0; n <= kBincoeffExactDoubleMaxN; ++n) {
        double total = 0.0;
        for (int k = 0; k <= n; ++k) {
            total += bincoeff(n, k);
        }
        PANTR_CHECK_MSG(total == static_cast<double>(std::int64_t{1} << n),
                        "row " + std::to_string(n) + " sums to " + std::to_string(total));
    }
}

void test_constexpr_usable() {
    // The elevation kernel builds its coefficient table at run time, but a
    // compile-time constant is the cheapest proof that nothing in the body escapes
    // constant evaluation -- no allocation, no libm, no undefined shift.
    static_assert(bincoeff(10, 5) == 252.0);
    static_assert(bincoeff(61, 30) == 232714176627630544.0);
}

}  // namespace

int main() {
    test_against_pascal();
    test_exact_below_the_double_envelope();
    test_out_of_range_is_zero();
    test_symmetry();
    test_row_sums_are_powers_of_two();
    test_constexpr_usable();
    return pantr::test::summary("test_binomial");
}
