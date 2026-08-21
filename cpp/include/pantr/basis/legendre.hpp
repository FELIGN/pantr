#pragma once

/// \file
/// Three-term recurrence for the orthonormal shifted Legendre basis on `[0, 1]`.
///
/// Port of `_tabulate_Legendre_basis_1D_core` in
/// `src/pantr/basis/_basis_core.py`, which stays as the parity oracle. The
/// recurrence and the precomputation of its coefficients are unchanged.
///
/// ## The recurrence
///
/// With `y = 2x - 1` mapping `[0, 1]` onto `[-1, 1]`,
///
///     p_0 = 1
///     p_1 = sqrt(3) * y
///     p_i = a_i * y * p_{i-1} - b_i * p_{i-2}
///
/// where `a_i = sqrt(2i-1) sqrt(2i+1) / i` and
/// `b_i = ((i-1)/i) sqrt((2i+1)/(2i-3))`. The `a_i` and `b_i` depend only on the
/// degree, so they are formed once and shared across every point.
///
/// The two square roots in `a_i` are kept apart rather than combined into
/// `sqrt((2i-1)(2i+1))`. That is not an oversight: the oracle writes them apart,
/// the two forms round differently, and a port that tidies the algebra is a port
/// that stops being comparable.
///
/// ## Coefficient width is a property of the oracle, not of the output
///
/// The oracle allocates its coefficient arrays with `np.empty(n + 1)`, whose
/// dtype is float64 whatever the evaluation points are, so at float32 the
/// coefficients are still computed and stored at double width and only the
/// per-point values round. `accumulator_t<float>` is `double`, so forming them in
/// `Acc` here reproduces that rather than merely resembling it.
///
/// ## Where the two backends can disagree
///
/// **One fusible site, and unlike the cardinal B-spline kernel it is a
/// subtraction.** `a_i * y * p_{i-1} - b_i * p_{i-2}` is `X - Y` with `Y` a
/// product, which a compiler may contract to a negated fused multiply-add. As
/// shipped there is no `-march`, so the baseline x86-64 target has no FMA
/// instruction and `-ffp-contract=on` has nothing to fuse; the day
/// design/simd.md's ISA ladder turns on, this line is where the two backends
/// start differing by one rounding per degree. numba emits it unfused.
///
/// **The recurrence rounds through `out`.** `p_{i-1}` and `p_{i-2}` are read back
/// out of the output array, so at float32 each step rounds to float32 rather than
/// carrying the running value at double width. Holding the last two values in
/// registers would be the natural C++ and would disagree with the oracle by a
/// growing multiple of one float32 ulp. See the same note in bernstein.hpp.
///
/// **Unlike Bernstein there is no `pow`.** `sqrt` is the only libm call, and it
/// is correctly rounded in IEEE 754, so it cannot be a source of disagreement.
/// That makes this kernel's bitwise parity derivable rather than merely observed;
/// tests/parity/test_basis_tabulations.py asserts it and says which is which.

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr {

/// Tabulate the orthonormal shifted Legendre basis of `degree` at every point.
///
/// Evaluates the `degree + 1` polynomials orthonormal on `[0, 1]` under the
/// unweighted inner product, by the three-term recurrence above.
///
/// \param degree Degree of the basis. Must be non-negative.
/// \param points Evaluation points. Values outside `[0, 1]` are evaluated by the
///        same recurrence rather than clamped, matching the Python kernel.
/// \param out Output view of shape `(points.size(), degree + 1)`, written in
///        full.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md. `degree >= 0` and an `out` of exactly
///       `(points.size(), degree + 1)` are load-bearing for memory safety, not
///       merely for the answer: the degree is widened to `std::size_t`, and
///       nothing here bounds-checks a write. Non-finite points propagate into
///       non-finite outputs rather than being rejected.
///       For general use call `pantr.basis.tabulate_legendre_1d`.
template <Real T>
void tabulate_legendre_1d(int degree, std::span<const T> points, span2d<T> out) {
    using Acc = accumulator_t<T>;
    using std::sqrt;

    const std::size_t num_pts = points.size();

    if (degree == 0) {
        for (std::size_t j = 0; j < num_pts; ++j) {
            at(out, j, 0) = T(1);
        }
        return;
    }

    const auto n = static_cast<std::size_t>(degree);

    // O(n) work shared by every point. Indices 0 and 1 are unused, and are
    // allocated anyway so that `a[i]` reads as the `a_i` of the recurrence.
    std::vector<Acc> a(n + 1, Acc(0));
    std::vector<Acc> b(n + 1, Acc(0));
    for (std::size_t i = 2; i <= n; ++i) {
        const Acc i_acc = Acc(static_cast<double>(i));
        const Acc sqrt_2i_minus_1 = sqrt(Acc(2) * i_acc - Acc(1));
        const Acc sqrt_2i_plus_1 = sqrt(Acc(2) * i_acc + Acc(1));
        const Acc sqrt_2i_minus_3 = sqrt(Acc(2) * i_acc - Acc(3));
        a[i] = (sqrt_2i_minus_1 * sqrt_2i_plus_1) / i_acc;
        b[i] = ((i_acc - Acc(1)) / i_acc) * (sqrt_2i_plus_1 / sqrt_2i_minus_3);
    }

    const Acc sqrt3 = sqrt(Acc(3));

    for (std::size_t j = 0; j < num_pts; ++j) {
        const Acc two_x_minus_1 = Acc(2) * static_cast<Acc>(points[j]) - Acc(1);

        at(out, j, 0) = T(1);
        at(out, j, 1) = static_cast<T>(sqrt3 * two_x_minus_1);

        for (std::size_t i = 2; i <= n; ++i) {
            // Read back through `out`, not from two carried registers: see the
            // file comment.
            const Acc p1 = static_cast<Acc>(at(out, j, i - 1));
            const Acc p2 = static_cast<Acc>(at(out, j, i - 2));
            at(out, j, i) = static_cast<T>(a[i] * two_x_minus_1 * p1 - b[i] * p2);
        }
    }
}

}  // namespace pantr
