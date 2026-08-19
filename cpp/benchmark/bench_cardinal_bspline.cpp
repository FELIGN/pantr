/// \file
/// Single-threaded throughput of the Cox-de Boor tabulation.
///
/// The number this produces is the kernel alone: no binding, no numpy, no
/// allocation inside the timed region. The user-facing comparison against the
/// numba backend lives in scripts/bench_parity.py, which times both through the
/// same Python entry point and so includes the overhead a caller actually pays.
/// Both are reported, because either alone is misleading -- the C++ kernel looks
/// better than it is without the binding, and worse than it is with it.
///
/// **This kernel is single-threaded and the numba one is not.** The numba
/// tabulation runs a `prange` over points above `_PARALLEL_MIN_NUM_PTS` and a
/// serial twin below it. Comparing a serial C++ kernel against a 20-thread numba
/// kernel would measure the thread pool, not the port. scripts/bench_parity.py
/// therefore reports against both numba variants, and the serial one is the
/// like-for-like figure.
///
/// Threading the C++ side is deliberately not in this prototype: it is a design
/// decision (design/user_functions_across_the_boundary.md fixes the rule that a
/// callback is invoked from one thread only, which constrains the pool) and
/// spending it here would buy a nicer number in exchange for a commitment
/// nothing yet requires.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

/// A deterministic, reproducible spread of points over [0, 1]. Not random: a
/// benchmark whose input changes between runs cannot be compared with an earlier
/// run, and the kernel is branch-free in the point value anyway, so nothing is
/// gained by randomising it.
std::vector<double> make_points(std::size_t n) {
    std::vector<double> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return pts;
}

double run(int degree, std::size_t num_pts, int repeats) {
    const auto pts = make_points(num_pts);
    const auto stride = static_cast<std::size_t>(degree + 1);
    std::vector<double> out(num_pts * stride, 0.0);
    const pantr::span2d<double> view(out.data(), num_pts, stride);

    // Warm the caches and the branch predictor, so the first timed repeat is not
    // measuring a cold page fault on `out`.
    pantr::tabulate_cardinal_bspline_1d<double>(degree, std::span<const double>(pts), view);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        pantr::tabulate_cardinal_bspline_1d<double>(degree, std::span<const double>(pts), view);
    }
    const auto stop = std::chrono::steady_clock::now();

    // Consume the result so the optimiser cannot delete the loop.
    double checksum = 0.0;
    for (const double v : out) {
        checksum += v;
    }
    if (checksum < 0.0) {
        std::fprintf(stderr, "impossible checksum %g\n", checksum);
    }

    const auto elapsed = std::chrono::duration<double>(stop - start).count();
    return elapsed / repeats;
}

}  // namespace

int main() {
    std::printf("%8s %12s %14s %14s\n", "degree", "points", "seconds", "Mpoint/s");
    for (const int degree : {1, 2, 3, 5, 8}) {
        for (const std::size_t num_pts : {std::size_t{1024}, std::size_t{65536},
                                          std::size_t{1048576}}) {
            const int repeats = num_pts >= 1048576 ? 20 : 200;
            const double seconds = run(degree, num_pts, repeats);
            std::printf("%8d %12zu %14.6e %14.2f\n", degree, num_pts, seconds,
                        static_cast<double>(num_pts) / seconds / 1e6);
        }
    }
    return 0;
}
