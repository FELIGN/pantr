#pragma once

/// \file
/// Evaluation of an n-dimensional Bézier: the tensor-product contraction, the
/// rational projection, and the two entry points a caller reaches them through.
///
/// Free functions over `Bezier<T>`, as `bezier.hpp` says: the type owns the value
/// and no operations, and this is one of the four operation ports over it.
///
/// ## The accumulation width is `T`, not `accumulator_t<T>`, and that is measured
///
/// Every kernel in `kernels_1d.hpp` but one accumulates in `accumulator_t<T>`,
/// which is `double` even for `float` storage. **This file does not**, and the
/// difference is not a stylistic one.
///
/// `accumulator_t` is `double` for a parity reason, stated in
/// `pantr/core/scalar.hpp`: Numba promotes a `float64` scalar against a `float32`
/// array, so a kernel whose oracle is a Numba kernel accumulates wide. **Nothing
/// in the n-d path is a Numba kernel.** The oracle here is `np.einsum` or
/// `np.tensordot` over arrays that `src/pantr/bezier/_bezier_eval.py` *requires*
/// to share the Bézier's own dtype, so at `float32` the oracle contracts in
/// `float32`. Inheriting the house policy would compute in the wrong arithmetic
/// and the parity claim would be about a computation nobody performs.
///
/// This is `design/backend_parity.md` Rule 9 -- an oracle's accumulation width is
/// a per-kernel fact, not a module convention -- and it was settled behaviourally
/// rather than by reading numpy: `scripts/measure_bezier_nd_widths.py` runs a
/// narrow model and a `float64` model against the oracle and reports how often the
/// two disagree, so a match cannot come from a check that could not fail.
///
/// ## Two entry points, two schedules, and neither is bitwise
///
/// `evaluate` mirrors `_evaluate_bezier_nd_pts_array` and contracts one direction
/// at a time against a per-point basis, exactly as the oracle's chain of
/// `np.einsum` calls does. `evaluate_on_lattice` mirrors
/// `_evaluate_bezier_nd_lattice`, which contracts one *axis* of the running result
/// at a time with `np.tensordot` and therefore reaches BLAS.
///
/// **`evaluate_on_lattice` has no `dim() == 1` special case, and the asymmetry with
/// `evaluate` is deliberate rather than an oversight.** Its general path is correct
/// at one dimension -- the outer extent is 1 and the contraction is the ordinary
/// one -- but Python never reaches it there, because `_evaluate_bezier` routes
/// `dim == 1` to the fused kernel before either n-d schedule exists. So a C++ caller
/// evaluating a one-dimensional Bézier on a lattice gets a *valid* result by a
/// rounding path **no parity claim covers**, differing in its last bits from what
/// `evaluate` gives for the same parameters. Nothing there is wrong; there is simply
/// no oracle for it.
///
/// The two are the same mathematics and different arithmetic, so **one kernel
/// cannot be exact against both** and each carries its own parity claim; the
/// derivations are in `tests/parity/test_bezier_evaluate.py`. What the
/// measurement found, and what a reader should not have to rediscover:
///
///  - the ascending-index contraction below reproduces `np.einsum` bit for bit
///    wherever the contraction's trailing block holds two or more elements. Where
///    that block is a single element -- which is every scalar-valued non-rational
///    Bézier -- it reproduces it only while the contraction stays short: numpy
///    dispatches a vectorised reduction from length 4 at `float32` and length 3 at
///    `float64`, and that reduction's summation tree is a property of the host
///    rather than of the expression. That the short cases still agree is **not** a
///    claim this file makes, since the threshold is numpy's to move;
///  - `np.tensordot` matches no width model at any shape swept, because it is a
///    matrix product.
///
/// So both claims are **bounded**, and both are bounded by the same standard
/// result rather than by a fitted constant: a contraction of length `n` commits at
/// most `n` roundings per term in any summation order, so Higham's `gamma_n`
/// against the absolute-value companion covers it. The companion is exact rather
/// than merely valid here because a Bernstein basis is non-negative.
///
/// One consequence worth stating, because it makes these claims cheaper than the
/// module's others: **the bound does not depend on `-ffp-contract`.** A fused
/// multiply-add removes a rounding rather than adding one, so a fusing build stays
/// inside a budget written for a non-fusing one, and neither claim needs the
/// conditional arm that `tests/parity/test_bezier_arithmetic.py` carries.
///
/// ## What a one-dimensional Bézier does here
///
/// `evaluate` delegates `dim() == 1` to `evaluate_bezier_1d`, because the oracle
/// does: `_evaluate_bezier` branches on `bezier.dim == 1` before it reaches either
/// contraction. That branch is not an optimisation to be tidied away -- the fused
/// 1-D kernel runs different arithmetic from a contraction against a tabulated
/// basis, and its parity claim is the bitwise one
/// `tests/parity/test_bezier_arithmetic.py` already carries.
///
/// ## Validating rather than asserting
///
/// Like `bezier.hpp`, and unlike `kernels_1d.hpp`: these are operations on a
/// domain type rather than Layer 3 kernels, so they validate and throw
/// `std::invalid_argument` in a release build as much as a debug one. A caller
/// with no Python cannot be protected by `cpp/bindings/`.

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/basis/bernstein.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bezier {

namespace detail {

/// The number of values a contraction result holds.
///
/// \param shape The extents, innermost last.
/// \param from The first extent to count, so that a partially contracted block
///        can be sized without rebuilding its shape.
/// \return The product of `shape[from]` through `shape.back()`, and 1 when the
///         range is empty.
[[nodiscard]] inline std::size_t block_size(std::span<const std::size_t> shape,
                                            std::size_t from) noexcept {
    std::size_t size = 1;
    for (std::size_t d = from; d < shape.size(); ++d) {
        size *= shape[d];
    }
    return size;
}

/// Tabulate the Bernstein basis of one direction at one column of a point array.
///
/// The column is gathered into a contiguous buffer first: `points` is row-major
/// `(n_pts, dim)`, so direction `d` is strided, and `tabulate_bernstein_1d` takes
/// a contiguous span. The gather is exact -- it moves values, it does not compute
/// on them -- so it is outside every parity claim in this file.
///
/// \param degree The polynomial degree in this direction.
/// \param points The point array, `(n_pts, dim)`.
/// \param direction The column to read.
/// \param scratch A buffer of at least `points.extent(0)` values, overwritten.
/// \param out The tabulation, `(n_pts, degree + 1)`.
template <Real T>
void tabulate_direction(std::size_t degree, span2d<const T> points, std::size_t direction,
                        std::span<T> scratch, span2d<T> out) {
    const std::size_t num_pts = points.extent(0);
    for (std::size_t p = 0; p < num_pts; ++p) {
        scratch[p] = at(points, p, direction);
    }
    tabulate_bernstein_1d<T>(static_cast<int>(degree),
                             std::span<const T>(scratch.data(), num_pts), out);
}

/// Contract the leading axis of a block against one weight per term.
///
/// This is the single arithmetic statement both schedules are built from, and the
/// only place in this file where a rounding happens. It accumulates in `T` and it
/// adds terms in ascending index order, which is what the oracle does; see the
/// file comment for why both halves of that sentence are load-bearing.
///
/// \param weights One weight per term of the contracted axis, `n_terms` of them.
/// \param block The values, `(n_terms, stride)` row-major.
/// \param stride The size of one term's block.
/// \param out The contracted block, `stride` values.
template <Real T>
void contract_leading_axis(std::span<const T> weights, std::span<const T> block,
                           std::size_t stride, std::span<T> out) {
    for (std::size_t t = 0; t < stride; ++t) {
        out[t] = T(0);
    }
    for (std::size_t term = 0; term < weights.size(); ++term) {
        const T weight = weights[term];
        const std::size_t offset = term * stride;
        for (std::size_t t = 0; t < stride; ++t) {
            out[t] = static_cast<T>(out[t] + weight * block[offset + t]);
        }
    }
}

/// Divide the value components of a raw result by its weight column, in place.
///
/// Reproduces `_project_rational`: a rational Bézier's raw contraction carries the
/// homogeneous weight in its last component, and the projected value is the
/// quotient. One correctly rounded division per component, at `T`, so this adds a
/// single rounding to whatever the contraction already committed.
///
/// \param raw The raw values, `(count, cp_size)` row-major.
/// \param count The number of coefficients.
/// \param cp_size The number of stored components, weight column included.
/// \param out The projected values, `(count, cp_size - 1)` row-major.
template <Real T>
void project_rational(std::span<const T> raw, std::size_t count, std::size_t cp_size,
                      std::span<T> out) {
    const std::size_t rank = cp_size - 1;
    for (std::size_t i = 0; i < count; ++i) {
        const T weight = raw[(i * cp_size) + rank];
        for (std::size_t r = 0; r < rank; ++r) {
            out[(i * rank) + r] = static_cast<T>(raw[(i * cp_size) + r] / weight);
        }
    }
}

/// Copy a raw result through unchanged, for a non-rational Bézier.
///
/// The oracle's `_project_rational` returns its argument untouched in this case,
/// so the copy is the whole of the projection and commits no arithmetic.
///
/// \param raw The raw values.
/// \param out The destination, the same length.
template <Real T>
void copy_through(std::span<const T> raw, std::span<T> out) {
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = raw[i];
    }
}

/// Refuse a point array whose shape does not describe this Bézier's parameters.
///
/// \param bezier The Bézier being evaluated.
/// \param points The point array.
/// \throws std::invalid_argument If the column count is not `bezier.dim()`.
template <Real T>
void require_point_columns(const Bezier<T>& bezier, span2d<const T> points) {
    if (points.extent(1) != bezier.dim()) {
        throw std::invalid_argument("pts must be a 2D array with "
                                    + std::to_string(bezier.dim()) + " columns.");
    }
}

}  // namespace detail

/// Evaluate a Bézier at an explicit list of parametric points.
///
/// Mirrors `pantr.bezier.Bezier.evaluate` for an array of points, dispatch
/// included: a one-dimensional Bézier goes to `evaluate_bezier_1d`, and anything
/// above it is contracted one direction at a time.
///
/// \param bezier The Bézier to evaluate.
/// \param points The parametric points, `(n_pts, bezier.dim())`, each row one
///        point. For a one-dimensional Bézier the single column holds the
///        parameters.
/// \param out The values, `(n_pts, bezier.rank())`. The trailing axis is kept even
///        for a scalar field; squeezing it is the wrapper's decision, and the
///        Python oracle makes it above this layer.
/// \throws std::invalid_argument If either shape is wrong.
template <Real T>
void evaluate(const Bezier<T>& bezier, span2d<const T> points, span2d<T> out) {
    detail::require_point_columns(bezier, points);
    const std::size_t num_pts = points.extent(0);
    const std::size_t rank = bezier.rank();
    if (out.extent(0) != num_pts || out.extent(1) != rank) {
        throw std::invalid_argument("out must have shape (" + std::to_string(num_pts) + ", "
                                    + std::to_string(rank) + ").");
    }

    const ControlNet<T>& net = bezier.net();
    const std::size_t dim = bezier.dim();
    const std::size_t cp_size = net.num_components();
    const std::span<const T> values = net.values();

    std::vector<T> raw(num_pts * cp_size);

    if (dim == 1) {
        // The oracle's own first branch. The fused 1-D kernel is different
        // arithmetic from a contraction against a tabulated basis, and it carries
        // its own bitwise claim; reproducing the branch is what inherits it.
        std::vector<T> parameters(num_pts);
        for (std::size_t p = 0; p < num_pts; ++p) {
            parameters[p] = at(points, p, 0);
        }
        evaluate_bezier_1d<T>(span2d<const T>(values.data(), net.extent(0), cp_size),
                              std::span<const T>(parameters), span2d<T>(raw.data(), num_pts,
                                                                        cp_size));
    } else {
        const std::span<const std::size_t> shape = net.shape();

        // One Bernstein tabulation per direction, each `(n_pts, n_d)`.
        std::vector<std::vector<T>> bases(dim);
        std::vector<T> gather(num_pts);
        for (std::size_t d = 0; d < dim; ++d) {
            bases[d].resize(num_pts * shape[d]);
            detail::tabulate_direction<T>(bezier.degree(d), points, d, std::span<T>(gather),
                                          span2d<T>(bases[d].data(), num_pts, shape[d]));
        }

        // After direction 0 the block for one point has extents
        // `(n_1, ..., n_{dim-1}, cp_size)`, and each later direction removes its
        // leading axis. Two buffers of that size are enough for the whole chain.
        const std::size_t widest = detail::block_size(shape, 1);
        std::vector<T> front(widest);
        std::vector<T> back(widest);

        for (std::size_t p = 0; p < num_pts; ++p) {
            std::size_t stride = detail::block_size(shape, 1);
            detail::contract_leading_axis<T>(
                std::span<const T>(&bases[0][p * shape[0]], shape[0]), values, stride,
                std::span<T>(front));
            for (std::size_t d = 1; d < dim; ++d) {
                stride /= shape[d];
                detail::contract_leading_axis<T>(
                    std::span<const T>(&bases[d][p * shape[d]], shape[d]),
                    std::span<const T>(front.data(), shape[d] * stride), stride,
                    std::span<T>(back.data(), stride));
                front.swap(back);
            }
            for (std::size_t c = 0; c < cp_size; ++c) {
                raw[(p * cp_size) + c] = front[c];
            }
        }
    }

    const std::span<T> flat_out(out.data_handle(), num_pts * rank);
    if (bezier.is_rational()) {
        detail::project_rational<T>(std::span<const T>(raw), num_pts, cp_size, flat_out);
    } else {
        detail::copy_through<T>(std::span<const T>(raw), flat_out);
    }
}

/// Evaluate a Bézier on a tensor-product lattice of parametric points.
///
/// Mirrors `pantr.bezier.Bezier.evaluate` for a `PointsLattice`. The schedule is
/// the oracle's: contract axis `d` of the running result with direction `d`'s
/// basis, leaving the new axis in position `d`, for `d` ascending.
///
/// This is a **different arithmetic** from `evaluate` over the same points written
/// out, not merely a faster one, and the two carry separate parity claims. See the
/// file comment.
///
/// \param bezier The Bézier to evaluate.
/// \param points_per_dir One span of parameters per parametric direction,
///        `bezier.dim()` of them. Direction `d` may hold any number of points.
/// \param out The values, `(m_0, ..., m_{dim-1}, bezier.rank())` row-major and
///        flat, where `m_d` is `points_per_dir[d].size()`.
/// \throws std::invalid_argument If the direction count or the output size is
///         wrong.
template <Real T>
void evaluate_on_lattice(const Bezier<T>& bezier,
                         std::span<const std::span<const T>> points_per_dir,
                         std::span<T> out) {
    const std::size_t dim = bezier.dim();
    if (points_per_dir.size() != dim) {
        throw std::invalid_argument("PointsLattice dim (" + std::to_string(points_per_dir.size())
                                    + ") must match Bézier dim (" + std::to_string(dim) + ").");
    }

    const ControlNet<T>& net = bezier.net();
    const std::size_t cp_size = net.num_components();
    const std::span<const std::size_t> shape = net.shape();
    const std::size_t rank = bezier.rank();

    std::size_t num_lattice = 1;
    for (const std::span<const T>& column : points_per_dir) {
        num_lattice *= column.size();
    }
    if (out.size() != num_lattice * rank) {
        throw std::invalid_argument("out must hold " + std::to_string(num_lattice * rank)
                                    + " values.");
    }

    // The running result starts as the control net and, after direction `d`, has
    // extents `(m_0, ..., m_d, n_{d+1}, ..., n_{dim-1}, cp_size)`. Its largest
    // size over the whole chain is what both ping-pong buffers are sized to.
    std::vector<std::size_t> extents(shape.begin(), shape.end());
    std::size_t widest = detail::block_size(shape, 0);
    {
        std::vector<std::size_t> running(shape.begin(), shape.end());
        for (std::size_t d = 0; d < dim; ++d) {
            running[d] = points_per_dir[d].size();
            widest = std::max(widest, detail::block_size(std::span<const std::size_t>(running), 0));
        }
    }
    std::vector<T> front(widest);
    std::vector<T> back(widest);
    detail::copy_through<T>(net.values(), std::span<T>(front.data(), net.size()));

    for (std::size_t d = 0; d < dim; ++d) {
        const std::span<const T>& column = points_per_dir[d];
        const std::size_t n_terms = extents[d];
        const std::size_t m_pts = column.size();

        // `outer` runs over the axes already contracted, `inner` over the ones
        // still to come plus the component axis. Contracting axis `d` in place
        // between them is what leaves the new axis in position `d`, which is the
        // `np.moveaxis` the oracle writes after every `np.tensordot`.
        std::size_t outer = 1;
        for (std::size_t e = 0; e < d; ++e) {
            outer *= extents[e];
        }
        const std::size_t inner = detail::block_size(std::span<const std::size_t>(extents), d + 1);

        std::vector<T> basis(m_pts * (bezier.degree(d) + 1));
        tabulate_bernstein_1d<T>(static_cast<int>(bezier.degree(d)), column,
                                 span2d<T>(basis.data(), m_pts, n_terms));

        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t m = 0; m < m_pts; ++m) {
                detail::contract_leading_axis<T>(
                    std::span<const T>(&basis[m * n_terms], n_terms),
                    std::span<const T>(&front[o * n_terms * inner], n_terms * inner), inner,
                    std::span<T>(&back[((o * m_pts) + m) * inner], inner));
            }
        }
        front.swap(back);
        extents[d] = m_pts;
    }

    if (bezier.is_rational()) {
        detail::project_rational<T>(std::span<const T>(front.data(), num_lattice * cp_size),
                                    num_lattice, cp_size, out);
    } else {
        detail::copy_through<T>(std::span<const T>(front.data(), num_lattice * cp_size), out);
    }
}

}  // namespace pantr::bezier
