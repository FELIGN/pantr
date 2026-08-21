#pragma once

/// \file
/// Application of a dense degree-reduction operator, the C++ twin of
/// `pantr.bspline._bspline_degree_core._apply_reduction_operator`.
///
/// ## Why this is in `core/` when the Python is in `bspline`
///
/// The Python docstring says why it lives where it does: `_bezier_core` already
/// imports `_bincoeff` from `_bspline_degree_core`, so putting this kernel in
/// `pantr.bezier` would close an import cycle. That is a packaging constraint of
/// the Python side, not a statement about where the code belongs, and C++ headers
/// have no cycle to avoid. Mirroring the Python location would be importing an
/// accident, so this sits beside `binomial.hpp` for the same reason that one
/// does: two modules need it, and neither should have to depend on the other.
///
/// ## What the operator is, and what is not here
///
/// `R` is assembled in **exact rational arithmetic** by
/// `_bezier_degree._l2_reduction_operator` and its interpolating sibling, cached
/// per `(p, q)`, and only then converted to `float64`. None of that is ported:
/// it runs once per degree pair, it is not a kernel, and the Python comment
/// records the reason it is exact at all -- a `float64` normal-equation solve for
/// the same operator loses eleven digits. What crosses the boundary is the
/// finished `float64` matrix, which is an array, so the type rule of
/// design/cross_backend_types.md is satisfied without argument.
///
/// ## The accumulation width is a promise, not a detail
///
/// The oracle accumulates in `float64` regardless of the control points' dtype
/// and rounds once on the write. At `float32` that is the difference between one
/// rounding per row and `p + 1` of them, and the docstring states it, so it is a
/// contract this port has to keep exactly rather than approximately.
///
/// Rows of `R` that pin an endpoint are unit vectors. Their outputs therefore
/// reproduce their inputs bit for bit, in either backend and at either dtype,
/// which is what makes the reduced segments of a spline stitch exactly C0.

#include <cstddef>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::core {

/// Apply a dense reduction operator to one set of control points: `out = R * ctrl`.
///
/// \param op Reduction operator `R`, shape `(q + 1, p + 1)`, always `double`.
/// \param ctrl Control points, shape `(p + 1, rank)`.
/// \param out Output, shape `(q + 1, rank)`, written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md.
///       For general use call `pantr.bezier.Bezier.reduce_degree`.
template <Real T>
void apply_reduction_operator(span2d<const double> op, span2d<const T> ctrl, span2d<T> out) {
    using Acc = accumulator_t<T>;

    const std::size_t n_out = op.extent(0);
    const std::size_t n_in = op.extent(1);
    const std::size_t rank = ctrl.extent(1);

    for (std::size_t i = 0; i < n_out; ++i) {
        for (std::size_t r = 0; r < rank; ++r) {
            Acc acc = Acc(0);
            for (std::size_t j = 0; j < n_in; ++j) {
                acc = acc + Acc(at(op, i, j)) * static_cast<Acc>(at(ctrl, j, r));
            }
            at(out, i, r) = static_cast<T>(acc);
        }
    }
}

}  // namespace pantr::core
