#pragma once

/// \file
/// Tensor-product change-of-basis application: the C++ twin of
/// `pantr.bspline._extraction_kernels`.
///
/// Applies `M = kron(M_0, ..., M_{d-1})` to a vector or to a matrix on both
/// sides, without ever materialising `M`. `design/extraction_port.md` carries the
/// design; what follows is what a reader of this header needs.
///
/// ## Four functions here against twenty-four in the oracle
///
/// The oracle specialises every kernel per dimension because a `nopython` Numba
/// function cannot loop over a variable-length tuple of arrays. C++ has no such
/// constraint, and `design/backend_parity.md` accepts a **bounded** claim rather
/// than bit-identity precisely so that the C++ can be written as C++.
///
/// What that licence does **not** extend to is the arithmetic. The set of modes
/// contracted, the order they are contracted in, the modes skipped, and the
/// length of each contraction are all reproduced exactly, because they are what
/// fixes the rounding count the parity bound is built from. Only the buffer
/// bookkeeping differs -- the oracle's `d = 2` kernel uses a single scratch slice
/// where this uses one half of the ping-pong, which is within the same budget.
///
/// ## The accumulator is `T`, and that is a promise rather than a detail
///
/// All twelve of the oracle's per-cell kernels open their contraction with
/// `zero = M_0.dtype.type(0.0)`, so at `float32` the entire chain accumulates in
/// `float32`. Accumulating in `double` here would be more accurate and would not
/// be the same function; the difference would surface as a `float32` parity
/// failure attributed to the wrong cause. This is `change_basis.hpp`'s "arithmetic
/// width is the output dtype, not the accumulator" applied to a different kernel
/// family, and it is the opposite choice from `core/reduction_operator.hpp`, whose
/// oracle really does widen. Neither can be read off the source of the other.
///
/// ## An identity mode performs no arithmetic
///
/// A mode flagged identity is passed through: no contraction is run, and the
/// operator's **values are never read**. Its **extents are**, which is not the
/// same thing and is what the oracle does too -- it reads all `2 d` shapes before
/// testing any flag. An identity mode must therefore be square, and that is a
/// memory-safety obligation rather than a correctness one, so it carries
/// `PANTR_PRECONDITION`: a non-square identity would make the extent bookkeeping
/// below inconsistent with the storage it indexes.
///
/// When every mode is identity the operation degenerates to a copy, which is the
/// one case where `out` may alias the input. The copy runs ascending, as the
/// oracle's does.
///
/// ## The inner loop is contractable, and the parity claim has to say so
///
/// `contract_axis`'s accumulation is `acc = acc + coefficient * src[...]`, which a
/// compiler targeting an ISA with a fused multiply-add may contract to one
/// instruction with one rounding instead of two. The oracle does not, so on such a
/// build the two backends differ by exactly the budget
/// `design/backend_parity.md` Rule 10 derives, and the parity test that follows
/// must gate on `contraction_may_fuse()` rather than assume bit-identity.
///
/// Stated here for the same reason `bezier/kernels_1d.hpp` states it of its own
/// loops: nothing in the source says which way a given build went, and the header
/// is where a reader looks before writing the claim.
///
/// ## Why `bezier/axis_layout.hpp` is not reused
///
/// That header's `gather_axis_to_front` / `scatter_axis_from_front` move an axis
/// so that a kernel can work on a leading dimension. These kernels do not need
/// an axis moved: a mode contraction over a row-major tensor is an
/// `outer x n x inner` loop nest at any axis position, and moving the axis would
/// add a full copy of the intermediate per stage for no arithmetic benefit.
///
/// \note **What "no validation" means here, and it is not what it means in the
/// oracle.** These are transliterations of Numba kernels whose docstrings carry
/// the same sentence, where a violated precondition yields a *defined wrong
/// answer*. On this side the same violation is undefined behaviour. So a
/// **correctness** obligation is documented and not asserted, while a
/// **memory-safety** obligation carries `PANTR_PRECONDITION`. Grep for the macro
/// to see every one.

#include <cstddef>
#include <span>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/precondition.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// One direction's 1D operator, with the flag that says it is the identity.
///
/// \tparam T Scalar type of the operator and of everything it is applied to.
template <Real T>
struct ModeOperator {
    /// The operator, shape `(n_out, n_in)`. Its **values** are unread when
    /// `is_identity`; its extents are read either way.
    span2d<const T> matrix;

    /// Whether this direction is the identity, so its contraction is skipped.
    ///
    /// The matrix must be square when this is set. That invariant is checked where
    /// the list is *consumed* (`PANTR_PRECONDITION`, so in debug builds only) rather
    /// than at construction, because this is a flat Layer 3 kernel-argument bundle
    /// and not a domain type: `CLAUDE.md` puts validation in Layer 2 and states that
    /// a kernel checks nothing. A constructor that threw here would be the first
    /// kernel argument in the tree to do so. Whether that exemption should extend to
    /// a struct this file exports is a design question flagged for the architect, not
    /// settled here.
    bool is_identity;
};

namespace detail {

/// Contract one axis of a row-major tensor against a dense operator.
///
/// The tensor is viewed as `(outer, n_contracted, inner)` and the result as
/// `(outer, n_produced, inner)`, which is what makes one loop nest serve every
/// axis position: `outer` is the product of the extents before the axis and
/// `inner` the product of those after it.
///
/// The accumulation runs over the contracted index ascending, in `T`, exactly as
/// the oracle's inner loop does. That is what the parity bound is stated against.
///
/// \tparam T Scalar type.
/// \tparam Transposed Whether to apply `op^T` rather than `op`. A template
///         parameter rather than a runtime flag so the two loop nests are
///         separate functions after instantiation, which is what the project's
///         rule about behaviour-selecting booleans asks for.
/// \param op The operator, shape `(n_out, n_in)`.
/// \param outer Product of the extents before the contracted axis.
/// \param inner Product of the extents after the contracted axis.
/// \param src The tensor to read, `outer * n_contracted * inner` elements.
/// \param dst The tensor to write, `outer * n_produced * inner` elements.
///
/// \note Inputs are assumed correct (no validation performed). `src` and `dst`
///       must not overlap.
template <Real T, bool Transposed>
void contract_axis(span2d<const T> op, std::size_t outer, std::size_t inner,
                   std::span<const T> src, std::span<T> dst) {
    const std::size_t n_produced = Transposed ? op.extent(1) : op.extent(0);
    const std::size_t n_contracted = Transposed ? op.extent(0) : op.extent(1);

    PANTR_PRECONDITION(src.size() >= outer * n_contracted * inner, "src too small");
    PANTR_PRECONDITION(dst.size() >= outer * n_produced * inner, "dst too small");

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < n_produced; ++i) {
            for (std::size_t n = 0; n < inner; ++n) {
                T acc(0.0);
                for (std::size_t m = 0; m < n_contracted; ++m) {
                    const T coefficient = Transposed ? at(op, m, i) : at(op, i, m);
                    acc = acc + coefficient * src[(o * n_contracted + m) * inner + n];
                }
                dst[(o * n_produced + i) * inner + n] = acc;
            }
        }
    }
}

/// Copy `count` elements ascending, which is aliasing-safe for `dst == src`.
///
/// \tparam T Scalar type.
/// \param src Source.
/// \param dst Destination.
/// \param count Number of elements.
template <Real T>
void copy_through(std::span<const T> src, std::span<T> dst, std::size_t count) {
    PANTR_PRECONDITION(src.size() >= count, "src too small");
    PANTR_PRECONDITION(dst.size() >= count, "dst too small");
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = src[i];
    }
}

/// The extent an axis has before its own stage runs.
///
/// For a transposed application the operator's roles swap, which is the only
/// difference between `apply_kron` and `apply_kron_transpose`.
///
/// \tparam T Scalar type.
/// \tparam Transposed Whether the operators are applied transposed.
/// \param op The direction's operator.
/// \return The axis extent on the input side.
template <Real T, bool Transposed>
[[nodiscard]] std::size_t extent_in(const ModeOperator<T>& op) noexcept {
    return Transposed ? op.matrix.extent(0) : op.matrix.extent(1);
}

/// The extent an axis has after its own stage runs.
///
/// \tparam T Scalar type.
/// \tparam Transposed Whether the operators are applied transposed.
/// \param op The direction's operator.
/// \return The axis extent on the output side.
template <Real T, bool Transposed>
[[nodiscard]] std::size_t extent_out(const ModeOperator<T>& op) noexcept {
    return Transposed ? op.matrix.extent(1) : op.matrix.extent(0);
}

/// Check that every identity-flagged mode is square.
///
/// \tparam T Scalar type.
/// \param ops The per-direction operators.
/// \return True when every identity mode has equal extents.
template <Real T>
[[nodiscard]] bool identity_modes_are_square(std::span<const ModeOperator<T>> ops) noexcept {
    for (const ModeOperator<T>& op : ops) {
        if (op.is_identity && op.matrix.extent(0) != op.matrix.extent(1)) {
            return false;
        }
    }
    return true;
}

/// Index of the last direction that is not the identity, or `ops.size()` if none is.
///
/// The stage that writes it is the one that writes `out`: every direction after it
/// passes through, and an identity direction is square, so the tensor already has
/// the output's extents by then.
///
/// Where this lands relative to the oracle differs by kernel, and the helper is
/// shared by all four, so it is worth being exact. The **unilateral** `d = 2`
/// kernels do the same thing -- a lone identity direction contracts straight into
/// `out` (`_extraction_kernels.py:133`, `:144`). Everywhere else -- unilateral
/// `d = 3`, and the bilateral kernels at both dimensions -- the oracle's last
/// stage runs unconditionally and degenerates to a copy when its direction is the
/// identity (`:274`, `:733`, `:854`), which this skips. Skipping it changes no
/// value and no rounding: a copy performs no arithmetic, so the stage counts a
/// parity bound is built from are the same either way.
///
/// \tparam T Scalar type.
/// \param ops The per-direction operators.
/// \return The index, or `ops.size()` when every direction is the identity.
template <Real T>
[[nodiscard]] std::size_t last_contracted(std::span<const ModeOperator<T>> ops) noexcept {
    std::size_t found = ops.size();
    for (std::size_t k = 0; k < ops.size(); ++k) {
        if (!ops[k].is_identity) {
            found = k;
        }
    }
    return found;
}

/// Apply `kron(ops...)`, or its transpose, to a vector.
///
/// \tparam T Scalar type.
/// \tparam Transposed Whether to apply the transpose.
/// \param ops Per-direction operators, one per tensor-product direction.
/// \param v Input vector, `prod(extent_in)` elements.
/// \param out Output vector, `prod(extent_out)` elements.
/// \param scratch Ping-pong work buffer, two halves; unused when at most one
///        direction contracts.
template <Real T, bool Transposed>
void apply_kron_impl(std::span<const ModeOperator<T>> ops, std::span<const T> v, std::span<T> out,
                     std::span<T> scratch) {
    const std::size_t d = ops.size();
    PANTR_PRECONDITION(d >= 1, "at least one direction is required");
    PANTR_PRECONDITION(identity_modes_are_square<T>(ops), "an identity mode must be square");

    const std::size_t last = last_contracted<T>(ops);

    if (last == d) {  // every direction is the identity: a copy, and `out` may alias `v`
        std::size_t total = 1;
        for (const ModeOperator<T>& op : ops) {
            total *= extent_out<T, Transposed>(op);
        }
        copy_through<T>(v, out, total);
        return;
    }

    const std::size_t half = scratch.size() / 2;
    const std::span<T> halves[2] = {scratch.first(half), scratch.subspan(half)};
    std::size_t toggle = 0;

    std::span<const T> cur = v;
    for (std::size_t k = 0; k < d; ++k) {
        if (ops[k].is_identity) {
            continue;
        }
        std::size_t outer = 1;
        for (std::size_t j = 0; j < k; ++j) {
            outer *= extent_out<T, Transposed>(ops[j]);
        }
        std::size_t inner = 1;
        for (std::size_t j = k + 1; j < d; ++j) {
            inner *= extent_in<T, Transposed>(ops[j]);
        }

        const std::span<T> dst = (k == last) ? out : halves[toggle];
        contract_axis<T, Transposed>(ops[k].matrix, outer, inner, cur, dst);
        if (k != last) {
            toggle ^= 1U;
        }
        cur = std::span<const T>(dst.data(), dst.size());
    }
}

/// The extent of one axis of the rank-`2d` bilateral tensor, entering stage `stage`.
///
/// The stage sequence is `(row 0, column 0, row 1, column 1, ...)`, so at the start
/// of stage `s = 2k + side` the row axes below `k` and the column axes below `k`
/// are already on the output side, and row axis `k` is too when `side` is 1.
///
/// \tparam T Scalar type.
/// \tparam Transposed Whether the operators are applied transposed.
/// \param ops Per-direction operators.
/// \param axis The axis, in `[0, 2 * ops.size())`; `axis < d` is a row axis.
/// \param stage The stage about to run.
/// \return The axis extent at that point.
template <Real T, bool Transposed>
[[nodiscard]] std::size_t bilateral_extent(std::span<const ModeOperator<T>> ops, std::size_t axis,
                                           std::size_t stage) noexcept {
    const std::size_t d = ops.size();
    const std::size_t k = stage / 2;
    const bool on_column_side = (stage % 2) == 1;

    const bool is_row_axis = axis < d;
    const std::size_t direction = is_row_axis ? axis : axis - d;
    const bool done = is_row_axis ? (direction < k || (direction == k && on_column_side))
                                  : (direction < k);
    return done ? extent_out<T, Transposed>(ops[direction])
                : extent_in<T, Transposed>(ops[direction]);
}

/// Apply `kron(ops...)` to both sides of a matrix: `M^T K M` or `M K M^T`.
///
/// The matrix is viewed as a rank-`2d` tensor whose first `d` axes index its rows
/// and whose last `d` index its columns, and the `2d` stages run
/// `(row 0, column 0, row 1, column 1, ...)`. That is the order
/// `_bilateral_scratch_size` in `pantr.bspline._extraction_helpers` sizes for, so
/// following it is what keeps the caller's buffer big enough.
///
/// `Transposed` selects between the two kinds and nothing else: `M^T K M` applies
/// the transposed operator on both sides, `M K M^T` the plain one on both.
///
/// \tparam T Scalar type.
/// \tparam Transposed True for `M^T K M`, false for `M K M^T`.
/// \param ops Per-direction operators.
/// \param k_matrix Input matrix, square of side `prod(extent_in)`.
/// \param out Output matrix, square of side `prod(extent_out)`.
/// \param scratch Ping-pong work buffer, two halves.
template <Real T, bool Transposed>
void apply_kron_bilateral_impl(std::span<const ModeOperator<T>> ops, std::span<const T> k_matrix,
                               std::span<T> out, std::span<T> scratch) {
    const std::size_t d = ops.size();
    PANTR_PRECONDITION(d >= 1, "at least one direction is required");
    PANTR_PRECONDITION(identity_modes_are_square<T>(ops), "an identity mode must be square");

    const std::size_t last_direction = last_contracted<T>(ops);

    if (last_direction == d) {  // every direction is the identity: a copy of the whole matrix
        std::size_t side = 1;
        for (const ModeOperator<T>& op : ops) {
            side *= extent_out<T, Transposed>(op);
        }
        copy_through<T>(k_matrix, out, side * side);
        return;
    }

    // The last stage that runs is the column stage of the last contracting
    // direction, since a direction contributes both its stages or neither.
    const std::size_t last_stage = 2 * last_direction + 1;

    const std::size_t half = scratch.size() / 2;
    const std::span<T> halves[2] = {scratch.first(half), scratch.subspan(half)};
    std::size_t toggle = 0;

    std::span<const T> cur = k_matrix;
    for (std::size_t stage = 0; stage <= last_stage; ++stage) {
        const std::size_t direction = stage / 2;
        if (ops[direction].is_identity) {
            continue;
        }
        const std::size_t axis = ((stage % 2) == 0) ? direction : d + direction;

        std::size_t outer = 1;
        for (std::size_t a = 0; a < axis; ++a) {
            outer *= bilateral_extent<T, Transposed>(ops, a, stage);
        }
        std::size_t inner = 1;
        for (std::size_t a = axis + 1; a < 2 * d; ++a) {
            inner *= bilateral_extent<T, Transposed>(ops, a, stage);
        }

        const std::span<T> dst = (stage == last_stage) ? out : halves[toggle];
        contract_axis<T, Transposed>(ops[direction].matrix, outer, inner, cur, dst);
        if (stage != last_stage) {
            toggle ^= 1U;
        }
        cur = std::span<const T>(dst.data(), dst.size());
    }
}

}  // namespace detail

/// Compute `out = kron(ops[0], ..., ops[d-1]) @ v`.
///
/// \tparam T Scalar type.
/// \param ops Per-direction operators, `ops[k]` of shape `(n_out_k, n_in_k)`.
/// \param v Input vector of `prod(n_in_k)` elements.
/// \param out Output vector of `prod(n_out_k)` elements. Must not alias `v` unless
///        every direction is the identity.
/// \param scratch Work buffer; the size `pantr.bspline._extraction_helpers`
///        computes is sufficient for any identity pattern.
///
/// \note Inputs are assumed to be correct (no validation performed).
///       For general use, call the Layer 2 dispatcher in
///       `pantr.bspline._extraction_helpers`.
template <Real T>
void apply_kron(std::span<const ModeOperator<T>> ops, std::span<const T> v, std::span<T> out,
                std::span<T> scratch) {
    detail::apply_kron_impl<T, false>(ops, v, out, scratch);
}

/// Compute `out = kron(ops[0], ..., ops[d-1])^T @ v`.
///
/// \tparam T Scalar type.
/// \param ops Per-direction operators, `ops[k]` of shape `(n_out_k, n_in_k)`.
/// \param v Input vector of `prod(n_out_k)` elements.
/// \param out Output vector of `prod(n_in_k)` elements. Must not alias `v` unless
///        every direction is the identity.
/// \param scratch Work buffer; the size `pantr.bspline._extraction_helpers`
///        computes is sufficient for any identity pattern.
///
/// \note Inputs are assumed to be correct (no validation performed).
template <Real T>
void apply_kron_transpose(std::span<const ModeOperator<T>> ops, std::span<const T> v,
                          std::span<T> out, std::span<T> scratch) {
    detail::apply_kron_impl<T, true>(ops, v, out, scratch);
}

/// Compute `out = M^T K M` with `M = kron(ops[0], ..., ops[d-1])`.
///
/// \tparam T Scalar type.
/// \param ops Per-direction operators.
/// \param k_matrix Input matrix, square of side `prod(n_out_k)`.
/// \param out Output matrix, square of side `prod(n_in_k)`. Must not alias
///        `k_matrix` unless every direction is the identity.
/// \param scratch Work buffer; the size `pantr.bspline._extraction_helpers`
///        computes is sufficient for any identity pattern.
///
/// \note Inputs are assumed to be correct (no validation performed).
template <Real T>
void apply_kron_mt_k_m(std::span<const ModeOperator<T>> ops, std::span<const T> k_matrix,
                       std::span<T> out, std::span<T> scratch) {
    detail::apply_kron_bilateral_impl<T, true>(ops, k_matrix, out, scratch);
}

/// Compute `out = M K M^T` with `M = kron(ops[0], ..., ops[d-1])`.
///
/// \tparam T Scalar type.
/// \param ops Per-direction operators.
/// \param k_matrix Input matrix, square of side `prod(n_in_k)`.
/// \param out Output matrix, square of side `prod(n_out_k)`. Must not alias
///        `k_matrix` unless every direction is the identity.
/// \param scratch Work buffer; the size `pantr.bspline._extraction_helpers`
///        computes is sufficient for any identity pattern.
///
/// \note Inputs are assumed to be correct (no validation performed).
template <Real T>
void apply_kron_m_k_mt(std::span<const ModeOperator<T>> ops, std::span<const T> k_matrix,
                       std::span<T> out, std::span<T> scratch) {
    detail::apply_kron_bilateral_impl<T, false>(ops, k_matrix, out, scratch);
}

}  // namespace pantr::bspline
