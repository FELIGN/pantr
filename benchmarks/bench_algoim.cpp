// Benchmark algoim C++ implicit quadrature for comparison with pantr.
//
// Compile (from the benchmarks/ directory):
//   c++ -std=c++17 -O2 -DWITH_LAPACK=1 \
//       -I/Users/antolin/Dev/pantr/non_public_3rdparty/algoim/algoim \
//       -I/Users/antolin/miniconda3/envs/pantr/include \
//       bench_algoim.cpp \
//       -L/Users/antolin/miniconda3/envs/pantr/lib -llapacke -llapack -lblas \
//       -o bench_algoim
//
// Run:
//   DYLD_LIBRARY_PATH=/Users/antolin/miniconda3/envs/pantr/lib ./bench_algoim

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "quadrature_multipoly.hpp"

using namespace algoim;
using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Timing helper: returns median time in microseconds over n_repeat trials
// ---------------------------------------------------------------------------
template <typename F>
double median_us(F&& func, int n_warmup, int n_repeat) {
    for (int i = 0; i < n_warmup; ++i) func();
    std::vector<double> times(n_repeat);
    for (int i = 0; i < n_repeat; ++i) {
        auto t0 = Clock::now();
        func();
        auto t1 = Clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return times[n_repeat / 2];
}

static constexpr int N_WARMUP = 5;
static constexpr int N_REPEAT = 100;
static const int q_values[] = {1, 2, 4, 8, 16};

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

template <int N>
void bench_vol_surf(ImplicitPolyQuadrature<N>& ipq, const xarray<real, N>& phi,
                    const char* name) {
    std::cout << "\n  Volume quadrature\n";
    std::cout << "  " << std::setw(4) << "q" << std::setw(14) << "time (us)"
              << std::setw(10) << "n_pts" << "\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    for (int q : q_values) {
        int n_pts = 0;
        double t = median_us(
            [&]() {
                n_pts = 0;
                ipq.integrate(AutoMixed, q,
                              [&](const uvector<real, N>& x, real w) { ++n_pts; });
            },
            N_WARMUP, N_REPEAT);
        std::cout << "  " << std::setw(4) << q << std::setw(14) << std::fixed
                  << std::setprecision(1) << t << std::setw(10) << n_pts << "\n";
    }

    if constexpr (N > 1) {
        std::cout << "\n  Surface quadrature\n";
        std::cout << "  " << std::setw(4) << "q" << std::setw(14) << "time (us)"
                  << std::setw(10) << "n_pts" << "\n";
        std::cout << "  " << std::string(50, '-') << "\n";

        for (int q : q_values) {
            int n_pts = 0;
            double t = median_us(
                [&]() {
                    n_pts = 0;
                    ipq.integrate_surf(
                        AutoMixed, q,
                        [&](const uvector<real, N>& x, real w,
                            const uvector<real, N>& wn) { ++n_pts; });
                },
                N_WARMUP, N_REPEAT);
            std::cout << "  " << std::setw(4) << q << std::setw(14) << std::fixed
                      << std::setprecision(1) << t << std::setw(10) << n_pts << "\n";
        }
    }
}

void bench_ellipse_2d() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  ellipse (2D)  (dim=2, deg=2)\n";
    std::cout << std::string(80, '=') << "\n";

    uvector<int, 2> P(3);
    xarray<real, 2> phi(nullptr, P);
    algoim_spark_alloc(real, phi);
    bernstein::bernsteinInterpolate<2>(
        [](const uvector<real, 2>& x) {
            real xx = -1.1 + x(0) * 2.2, yy = -1.1 + x(1) * 2.2;
            return xx * xx + 4.0 * yy * yy - 1.0;
        },
        phi);

    double build_us = median_us(
        [&]() { ImplicitPolyQuadrature<2> ipq(phi); }, N_WARMUP, N_REPEAT);
    std::cout << "\n  Build phase: " << std::fixed << std::setprecision(1) << build_us
              << " us\n";

    ImplicitPolyQuadrature<2> ipq(phi);
    bench_vol_surf(ipq, phi, "ellipse (2D)");
}

void bench_circle_2d() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  circle (2D)  (dim=2, deg=2)\n";
    std::cout << std::string(80, '=') << "\n";

    uvector<int, 2> P(3);
    xarray<real, 2> phi(nullptr, P);
    algoim_spark_alloc(real, phi);
    bernstein::bernsteinInterpolate<2>(
        [](const uvector<real, 2>& x) {
            return (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5) - 0.1;
        },
        phi);

    double build_us = median_us(
        [&]() { ImplicitPolyQuadrature<2> ipq(phi); }, N_WARMUP, N_REPEAT);
    std::cout << "\n  Build phase: " << std::fixed << std::setprecision(1) << build_us
              << " us\n";

    ImplicitPolyQuadrature<2> ipq(phi);
    bench_vol_surf(ipq, phi, "circle (2D)");
}

void bench_bilinear_2d() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  bilinear eps=0.1 (2D)  (dim=2, deg=1)\n";
    std::cout << std::string(80, '=') << "\n";

    uvector<int, 2> P(2);
    xarray<real, 2> phi(nullptr, P);
    algoim_spark_alloc(real, phi);
    bernstein::bernsteinInterpolate<2>(
        [](const uvector<real, 2>& x) {
            return (x(0) - 0.5) * (x(1) - 0.5) - 0.01;
        },
        phi);

    double build_us = median_us(
        [&]() { ImplicitPolyQuadrature<2> ipq(phi); }, N_WARMUP, N_REPEAT);
    std::cout << "\n  Build phase: " << std::fixed << std::setprecision(1) << build_us
              << " us\n";

    ImplicitPolyQuadrature<2> ipq(phi);
    bench_vol_surf(ipq, phi, "bilinear (2D)");
}

void bench_ellipsoid_3d() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  ellipsoid (3D)  (dim=3, deg=2)\n";
    std::cout << std::string(80, '=') << "\n";

    uvector<int, 3> P(3);
    xarray<real, 3> phi(nullptr, P);
    algoim_spark_alloc(real, phi);
    bernstein::bernsteinInterpolate<3>(
        [](const uvector<real, 3>& x) {
            real xx = -1.1 + x(0) * 2.2, yy = -1.1 + x(1) * 2.2, zz = -1.1 + x(2) * 2.2;
            return xx * xx + 4.0 * yy * yy + 9.0 * zz * zz - 1.0;
        },
        phi);

    double build_us = median_us(
        [&]() { ImplicitPolyQuadrature<3> ipq(phi); }, N_WARMUP, N_REPEAT);
    std::cout << "\n  Build phase: " << std::fixed << std::setprecision(1) << build_us
              << " us\n";

    ImplicitPolyQuadrature<3> ipq(phi);
    bench_vol_surf(ipq, phi, "ellipsoid (3D)");
}

void bench_sphere_3d() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  sphere (3D)  (dim=3, deg=2)\n";
    std::cout << std::string(80, '=') << "\n";

    uvector<int, 3> P(3);
    xarray<real, 3> phi(nullptr, P);
    algoim_spark_alloc(real, phi);
    bernstein::bernsteinInterpolate<3>(
        [](const uvector<real, 3>& x) {
            return (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5) +
                   (x(2) - 0.5) * (x(2) - 0.5) - 0.09;
        },
        phi);

    double build_us = median_us(
        [&]() { ImplicitPolyQuadrature<3> ipq(phi); }, N_WARMUP, N_REPEAT);
    std::cout << "\n  Build phase: " << std::fixed << std::setprecision(1) << build_us
              << " us\n";

    ImplicitPolyQuadrature<3> ipq(phi);
    bench_vol_surf(ipq, phi, "sphere (3D)");
}

void bench_tunnel_3d() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  trilinear tunnel (3D)  (dim=3, deg=1)\n";
    std::cout << std::string(80, '=') << "\n";

    uvector<int, 3> P(2);
    xarray<real, 3> phi(nullptr, P);
    algoim_spark_alloc(real, phi);
    bernstein::bernsteinInterpolate<3>(
        [](const uvector<real, 3>& x) {
            return 0.5 - 1.2 * x(0) - 1.3 * x(1) - 1.4 * x(2) + 2.9 * x(0) * x(1) +
                   3.2 * x(0) * x(2) + 3.3 * x(1) * x(2) - 6.5 * x(0) * x(1) * x(2);
        },
        phi);

    double build_us = median_us(
        [&]() { ImplicitPolyQuadrature<3> ipq(phi); }, N_WARMUP, N_REPEAT);
    std::cout << "\n  Build phase: " << std::fixed << std::setprecision(1) << build_us
              << " us\n";

    ImplicitPolyQuadrature<3> ipq(phi);
    bench_vol_surf(ipq, phi, "tunnel (3D)");
}

int main() {
    std::cout << "Algoim C++ implicit quadrature benchmark\n";
    std::cout << "N_WARMUP=" << N_WARMUP << ", N_REPEAT=" << N_REPEAT << "\n";

    bench_circle_2d();
    bench_ellipse_2d();
    bench_bilinear_2d();
    bench_sphere_3d();
    bench_ellipsoid_3d();
    bench_tunnel_3d();

    return 0;
}
