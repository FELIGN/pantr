#pragma once

/// \file
/// The two rules whose nodes are given in closed form: trapezoidal, and the
/// modified Chebyshev interpolation nodes.
///
/// Ports of `get_trapezoidal_1d` and `get_modified_chebyshev_nodes_1d` in
/// `src/pantr/quad/_rules.py`, which stay as the parity oracles.
///
/// ## These two do not agree on what arithmetic to compute in, and that is the point
///
/// Everywhere else in `pantr/quad` the rule is: compute in `double`, hand the
/// caller `[-1, 1]` data, and let the shared Python helper map and narrow. That
/// rule is **wrong for the Chebyshev nodes**, and it was measured rather than
/// reasoned about.
///
/// `get_modified_chebyshev_nodes_1d` builds its index array with
/// `dtype=dtype_obj`, so under NEP 50 every operation after it -- the multiply by
/// `pi`, the divide, the cosine, and both halvings -- happens in the *storage*
/// format. In `float32` that is genuinely `float32` arithmetic: computing in
/// double and narrowing at the end was measured to differ from a float32 `cos` on
///  of 4096 arguments, 62%. (A narrower figure, 712 of 4096, appears in an
/// earlier note and is a different measurement: the cosine alone, handed an
/// argument both sides already agree on. The formula as a whole is what matters
/// here, and it separates four times more often.) So `modified_chebyshev_nodes` below is templated
/// and computes in `T`, and it is the only kernel in this directory that is.
///
/// `get_trapezoidal_1d` is the other way round. `np.linspace` computes in
/// `float64` and narrows once at the end, and the weights are a `float64` scalar
/// stored into an array of the storage format. So this kernel emits `double` and
/// the adapter narrows, which is the same sequence of roundings.
///
/// ## `np.linspace` is reproducible, and that was checked
///
/// numpy forms `arange(num) * step + start` with `step = (stop - start) / (num - 1)`
/// and then **assigns** `stop` to the last element rather than computing it.
/// Reproducing that assignment is what makes the right endpoint exactly 1 instead
/// of one ulp below. Measured bit-identical at num = 2, 3, 5, 17, 101, 1000.

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>

#include "pantr/core/scalar.hpp"

namespace pantr {

/// Tabulate the `n`-point trapezoidal rule on `[0, 1]`.
///
/// Unlike the other rules in this directory the output is already on `[0, 1]`, so
/// no mapping follows; the adapter only narrows. For `n == 1` the rule is the
/// midpoint one, node `0.5` and weight `1.0`, matching the Python.
///
/// \param n Number of points. Must be at least 1.
/// \param out_nodes Output of length `n`, written in full, ascending from 0 to 1.
/// \param out_weights Output of length `n`, written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md. Load-bearing preconditions, established by the
///       Layer 2 caller: `n >= 1`, and both spans having length exactly `n`, since
///       nothing here bounds-checks a write. **`out_nodes` and `out_weights` must not
///       overlap**: each is written independently, so an aliased pair leaves one
///       silently overwriting the other. `cpp/bindings/quad.cpp` refuses that at
///       the boundary and the Python adapters cannot produce it, but a C++ caller
///       establishes it itself. For general use call
///       `pantr.quad.get_trapezoidal_1d`.
inline void trapezoidal(int n, std::span<double> out_nodes, std::span<double> out_weights) {
    if (n == 1) {
        out_nodes[0] = 0.5;
        out_weights[0] = 1.0;
        return;
    }

    const auto count = static_cast<std::size_t>(n);
    const double step = 1.0 / static_cast<double>(n - 1);

    for (std::size_t i = 0; i < count; ++i) {
        // `+ 0.0` is numpy's `y += start`, kept because it is part of the
        // sequence even though adding zero is exact.
        out_nodes[i] = static_cast<double>(i) * step + 0.0;
        out_weights[i] = step;
    }
    // numpy assigns the endpoint rather than computing it, which is what makes
    // the last node exactly 1 rather than one ulp below.
    out_nodes[count - 1] = 1.0;

    out_weights[0] = 0.5 * step;
    out_weights[count - 1] = 0.5 * step;
}

/// Tabulate the `n` modified Chebyshev interpolation nodes on `[0, 1]`.
///
/// Chebyshev nodes of the second kind, the Chebyshev-Lobatto points, mapped onto
/// `[0, 1]`. Both endpoints are included. These are interpolation nodes, so there
/// are no weights.
///
/// Computed in `T` rather than in `double`, which is the exception this file's
/// header explains: the Python's arithmetic is in the storage format and a
/// double-then-narrow port disagrees with it on 62% of `float32` arguments.
///
/// \param n Number of nodes. Must be at least 2.
/// \param out Output of length `n`, written in full, from 0 to 1.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md. Load-bearing preconditions, established by the
///       Layer 2 caller: `n >= 2`, since `n - 1` is a divisor and `n == 1` would
///       divide by zero; and `out` having length exactly `n`, since nothing here
///       bounds-checks a write. For general use call
///       `pantr.quad.get_modified_chebyshev_nodes_1d`.
template <std::floating_point T>
void modified_chebyshev_nodes(int n, std::span<T> out) {
    using std::cos;

    const auto count = static_cast<std::size_t>(n);
    const T pi = static_cast<T>(std::numbers::pi);
    const T divisor = static_cast<T>(n - 1);
    const T half = static_cast<T>(0.5);

    for (std::size_t i = 0; i < count; ++i) {
        const T index = static_cast<T>(i);
        out[i] = half - half * cos(pi * index / divisor);
    }
}

}  // namespace pantr
