/// \file
/// Compiles and runs against every fetched dependency, under pantr's own
/// warning flags.
///
/// This is the test that answers "do Kokkos mdspan and Eigen survive -Werror
/// with SYSTEM, and CMake 4?". A dependency that is declared in CMake but never
/// `#include`d answers none of that: the include path is never exercised, no
/// header is ever parsed, and the warning flags never meet upstream's code. So
/// this file includes both and uses both.
///
/// Eigen is otherwise unused by this prototype. design/large_data_fitting.md and
/// design/adaptive_thb_approximation.md both settle on Eigen::SparseMatrix with
/// SimplicialLDLT for the fitting solves, so it is fetched now to find out
/// whether it builds clean here rather than on the day it is needed.

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

void mdspan_view_is_row_major() {
    std::vector<double> storage(6);
    const pantr::span2d<double> view(storage.data(), 2, 3);

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            pantr::at(view, i, j) = static_cast<double>(3 * i + j);
        }
    }

    // layout_right means element (i, j) sits at flat offset 3*i + j. Checking it
    // through the raw storage rather than through the view is the point: the
    // kernels hand this buffer to numpy, which assumes exactly this layout.
    for (std::size_t k = 0; k < storage.size(); ++k) {
        PANTR_CHECK(storage[k] == static_cast<double>(k));
    }

    PANTR_CHECK(view.extent(0) == 2);
    PANTR_CHECK(view.extent(1) == 3);
}

void eigen_solves_a_banded_spd_system() {
    // A 1D second-difference operator: symmetric positive definite, half
    // bandwidth 1. The same shape of system design/large_data_fitting.md routes
    // through SimplicialLDLT, at a size small enough to check by hand.
    constexpr int n = 5;
    Eigen::SparseMatrix<double> a(n, n);
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 2.0);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    a.setFromTriplets(entries.begin(), entries.end());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(a);
    PANTR_CHECK(solver.info() == Eigen::Success);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(n);
    const Eigen::VectorXd x = solver.solve(rhs);
    PANTR_CHECK(solver.info() == Eigen::Success);

    // Residual rather than a hard-coded solution: the bound is the standard
    // backward-stability statement for a direct solve, ||A x - b|| <= c(n)
    // kappa(A) eps ||b||, with c(n) a low-degree polynomial in n. At n = 5 the
    // second-difference matrix has kappa_2 ~ 13, so 64 * eps * ||b|| clears it
    // by a wide and deliberate margin -- this test exists to prove Eigen links
    // and runs, not to measure its accuracy.
    const double residual = (a * x - rhs).norm();
    const double bound = 64.0 * std::numeric_limits<double>::epsilon() * rhs.norm();
    PANTR_CHECK_MSG(residual <= bound,
                    "residual " + std::to_string(residual) + " > bound " + std::to_string(bound));
}

}  // namespace

int main() {
    mdspan_view_is_row_major();
    eigen_solves_a_banded_spd_system();
    return pantr::test::summary("test_dependency_smoke");
}
