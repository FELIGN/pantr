#pragma once

/// \file
/// Cell addressing and point location on a hierarchical grid.
///
/// Ports `src/pantr/grid/_hier_core.py`, which stays as the parity oracle.
///
/// ## Where this file's parity claim differs from its siblings'
///
/// `pantr/grid/locate.hpp` and `pantr/grid/bvh.hpp` claim an equality and can
/// argue it structurally: one performs no floating-point arithmetic at all, and
/// the other's every discrete verdict rests on a quantity with a single rounding
/// and no site a compiler could fuse. **This file cannot make that argument**, and
/// saying so is the point of this section.
///
/// Three of its five kernels are pure integer arithmetic and are exact for the
/// same dull reason as the others. The other two run a descent:
///
/// ```
/// size_k = (hi[k] - lo[k]) / factor[k]
/// j      = int((x[k] - lo[k]) / size_k)     // truncation: a discrete verdict
/// lo[k]  = lo[k] + j * size_k               // a multiply feeding an add: fusable
/// hi[k]  = lo[k] + size_k
/// ```
///
/// `lo[k]` is rewritten each level, and the rewritten value feeds the next level's
/// `size_k`, its truncation and its own update. Written as one expression that
/// update is contractible, and `-ffp-contract=on` permits exactly that; the oracle
/// never fuses, since numba defaults to `fastmath=False`. The two would then take
/// different branches, and `design/backend_parity.md` Rule 11 is explicit that no
/// tolerance bounds a branch.
///
/// **So the product is named rather than inlined, at both sites, and that is what
/// makes the equality hold.** It is a property of how the statement is written, not
/// of the target: `-ffp-contract=on` was chosen over `fast` precisely because it
/// confines fusion to within a single expression, so splitting the statement puts
/// the multiplication's rounding back where the oracle has it. Verified by
/// disassembly at `-march=x86-64-v3` and at `-march=native`.
///
/// **Two things are worth knowing about how this paragraph used to read**, because
/// both are traps a later reader could fall into again.
///
/// It claimed the descent could not diverge, and justified that with a sweep of
/// 300000 descents and a mechanism: that fusing moves a sum only when its two terms
/// are comparable in magnitude, which in a descent stops holding after a level or
/// two. Both parts were wrong. The sweep ran on a build with **no** `-march`, where
/// baseline x86-64 is SSE2 and has no fused multiply-add at all, so it compared two
/// unfused implementations and carried no information about contraction. And the
/// mechanism is false in the direction that matters: comparable magnitude is
/// neither necessary nor sufficient, and the ratio that actually decides the
/// verdict, `|dlo| / size_k`, is *invariant* rather than decaying, because the
/// truncation divides by a `size_k` that shrinks at the same geometric rate. Once
/// two backends disagree on `j` their indices never coincide again and the
/// difference grows.
///
/// The counterexample, for anyone tempted to inline the product again: root cell
/// `[1.0, 2.0]`, factor 10, `x = 1.71`. At level 0 the unfused sum is
/// `1.7000000000000002` and the fused one `1.7`, one ulp apart; at level 1 the
/// quotients are `0.9999999999999778` and exactly `1.0`, so the truncation gives
/// child 0 against child 1. Note which side is right: the exact value of
/// `1.0 + 7*fl(0.1)` is `1.7`, so the fused result is the correctly rounded one and
/// the oracle is one ulp high. Splitting the statement deliberately forbids the
/// more accurate answer, because this port's contract is parity and not correction.
///
/// **What protects the split, and what does not.** On a build with no `-march`
/// nothing fuses either way, so no test on such a build can catch a re-inlining --
/// `design/build_findings.md` is what says two-sided evidence is needed here, and
/// the disassembly is the other side. `tests/parity/test_grid.py` carries the
/// counterexample above as a named case that runs the real kernel at that input
/// under both backends, so it fails if the product is inlined again on a target
/// that fuses. That test also asserts, independently of any backend, that the input
/// is still a *discriminator* -- that a fused evaluation diverges from an unfused
/// one there -- which is the half that says something on a build where nothing
/// fuses at all.
///
/// Rule 11's operational consequence still stands and is independent of all this: a
/// caller comparing the two backends must assert the returned **ids** on their own,
/// before and separately from any numeric comparison, because a changed id is a
/// changed verdict and not a displaced value.
///
/// ## Two deviations from the oracle's shape, both deliberate
///
/// **The oracle's `_encode_midx_core` is merged into `block_of_midx` here.** In the
/// oracle they are two functions and the second is a one-line call to the first:
/// one is a `nopython` helper reached from inside the descent, the other a serial
/// entry point for a scalar Python caller. That split serves a Numba constraint --
/// no dispatch can be inserted between two Numba kernels -- and has no analogue in
/// C++, where one function serves both callers. Nothing computed changes.
///
/// **Both parallel kernels are serial here.** The oracle's outer loops are
/// `prange`. Each iteration writes only its own output entry and there is no
/// reduction, so the answer cannot move with the thread count either way; the same
/// argument and the same precedent as the bezier batch kernels. Only the speed
/// differs, and nothing here has been profiled.
///
/// ## Integer division, which is where a transliteration would quietly go wrong
///
/// The oracle uses Python's `//` and `%`, which floor; C++ truncates toward zero.
/// They agree only for non-negative operands, so each site needs its own reason
/// and gets one:
///
///  - `(lo_i + hi_i) / 2` in the binary search: both bounds start non-negative and
///    only move inward.
///  - `offset % extent` and `offset /= extent` in the decode: `offset` starts as
///    `cid - block_base[b]`, which the upper-bound search leaves non-negative, and
///    `extent` is a block width, so positive.
///  - `ik / m_pow[k]` and `ik % m_pow[k]` in the collect: `ik` is a cell index and
///    `m_pow` a positive power of the subdivision factor.
///
/// `m_pow` is built by repeated multiplication rather than by `pow`, matching the
/// oracle, so that the two cannot disagree through a floating-point exponentiation
/// neither of them should be doing.


/// \note **What "no validation" means here, and it is not what it means in the oracle.**
/// These kernels are transliterations of Numba kernels whose docstrings use this same
/// sentence, where a violated precondition yields a *defined wrong answer*: numpy
/// indexes negatively and a Python integer does not overflow. On this side the same
/// violation is undefined behaviour. So the obligations are of two kinds, and the code
/// says which:
///
/// - a **correctness** obligation is documented and not asserted; violating it gives a
///   wrong answer in both backends, which is what the sentence above promises;
/// - a **memory-safety** obligation carries `PANTR_PRECONDITION`, from
///   `pantr/core/precondition.hpp`. Grep for it to see every one in this file.
///
/// The macro is `assert`, so it costs nothing in a release build. The bindings under
/// `cpp/bindings/` refuse all of these before a Python caller can express them; the
/// macro is for the C++ caller who includes this header directly.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/precondition.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::grid {

/// Flat cell id of `(level, midx)`, or -1 when that position is not active.
///
/// Mirrors `_hier_core._block_of_midx`, and also `_hier_core._encode_midx_core`,
/// which is a one-line call to it; see the file comment for why the two are one
/// here. No validation is performed.
///
/// \param level Hierarchy level of the queried position.
/// \param midx Per-axis index, shape `(ndim,)`, in level-`level` coordinates.
/// \param block_lo Packed block lower bounds, shape `(n_blocks, ndim)`.
/// \param block_hi Packed block upper bounds, same shape.
/// \param block_base Flat-id base per block, shape `(n_blocks,)`.
/// \param level_block_start Block index range per level, shape `(n_levels + 1,)`.
/// \return The flat cell id, or -1 when `(level, midx)` is not an active leaf.
[[nodiscard]] inline std::int64_t block_of_midx(std::int64_t level,
                                                std::span<const std::int64_t> midx,
                                                span2d<const std::int64_t> block_lo,
                                                span2d<const std::int64_t> block_hi,
                                                std::span<const std::int64_t> block_base,
                                                std::span<const std::int64_t> level_block_start) {
    // Memory safety, and this is the function FELIGN/pantr#359 named as its third
    // site. A negative `level` indexes `level_block_start` from a wrapped cast:
    // measured, a heap-buffer-overflow READ. The oracle reads backwards from the
    // end and returns a wrong block instead.
    PANTR_PRECONDITION(level >= 0, "level must be non-negative");
    const std::size_t ndim = midx.size();
    const std::int64_t first = level_block_start[static_cast<std::size_t>(level)];
    const std::int64_t last = level_block_start[static_cast<std::size_t>(level) + 1];

    for (std::int64_t bi = first; bi < last; ++bi) {
        const auto b = static_cast<std::size_t>(bi);
        bool inside = true;
        for (std::size_t k = 0; k < ndim; ++k) {
            if (midx[k] < block_lo(b, k) || midx[k] >= block_hi(b, k)) {
                inside = false;
                break;
            }
        }
        if (!inside) {
            continue;
        }
        std::int64_t offset = 0;
        for (std::size_t k = 0; k < ndim; ++k) {
            offset = offset * (block_hi(b, k) - block_lo(b, k)) + (midx[k] - block_lo(b, k));
        }
        return block_base[b] + offset;
    }
    return -1;
}

/// Recover a cell's level and level-coordinate index from its flat id.
///
/// Mirrors `_hier_core._decode_flat_id_core`. No validation is performed.
///
/// \param cid Flat cell id in `[0, num_cells)`.
/// \param block_lo Packed block lower bounds, shape `(n_blocks, ndim)`.
/// \param block_hi Packed block upper bounds, same shape.
/// \param block_base Flat-id base per block, globally ascending in flat-id order.
/// \param level_block_start Block index range per level, shape `(n_levels + 1,)`.
/// \param out_midx Output per-axis index, shape `(ndim,)`, in level coordinates.
/// \return The cell's level.
[[nodiscard]] inline std::int64_t decode_flat_id(std::int64_t cid,
                                                 span2d<const std::int64_t> block_lo,
                                                 span2d<const std::int64_t> block_hi,
                                                 std::span<const std::int64_t> block_base,
                                                 std::span<const std::int64_t> level_block_start,
                                                 std::span<std::int64_t> out_midx) {
    const auto ndim = static_cast<std::int64_t>(out_midx.size());
    const auto n_blocks = static_cast<std::int64_t>(block_base.size());
    const std::int64_t n_levels = static_cast<std::int64_t>(level_block_start.size()) - 1;

    // upper_bound: the first block whose base exceeds cid, minus one.
    std::int64_t lo_b = 0;
    std::int64_t hi_b = n_blocks;
    while (lo_b < hi_b) {
        const std::int64_t mid = (lo_b + hi_b) / 2;
        if (block_base[static_cast<std::size_t>(mid)] <= cid) {
            lo_b = mid + 1;
        } else {
            hi_b = mid;
        }
    }
    // Memory safety, not correctness. `cid` below the first block's base leaves
    // `lo_b == 0`, so this is `-1` cast to an unsigned index. The oracle indexes
    // backwards from the end and returns a wrong answer; here it is out of bounds.
    PANTR_PRECONDITION(lo_b > 0, "cid must be at or above the first block's base");
    const auto b = static_cast<std::size_t>(lo_b - 1);

    std::int64_t level = 0;
    for (std::int64_t lev = 0; lev < n_levels; ++lev) {
        const auto l = static_cast<std::size_t>(lev);
        if (level_block_start[l] <= static_cast<std::int64_t>(b)
            && static_cast<std::int64_t>(b) < level_block_start[l + 1]) {
            level = lev;
            break;
        }
    }

    // Expand the C-order offset inside the block, last axis fastest.
    std::int64_t offset = cid - block_base[b];
    for (std::int64_t k = ndim - 1; k >= 0; --k) {
        const auto kk = static_cast<std::size_t>(k);
        const std::int64_t extent = block_hi(b, kk) - block_lo(b, kk);
        // Also memory safety rather than correctness: `%` and `/` by zero trap in
        // hardware here. The oracle is `@nb_jit` compiled, where int64 division by
        // zero raises `ZeroDivisionError` -- checked, not assumed, because the
        // plain-numpy answer differs and is not the one this oracle takes.
        PANTR_PRECONDITION(extent > 0, "every block must have a positive extent per axis");
        out_midx[kk] = block_lo(b, kk) + offset % extent;
        offset /= extent;
    }
    return level;
}

/// Locate a batch of points on a hierarchical grid.
///
/// Mirrors `_hier_core._hier_locate_points_core`. No validation is performed.
/// This is one of the two kernels whose equality claim rests on measurement
/// rather than on structure; see the file comment.
///
/// \tparam T Coordinate type.
/// \param points Query points, shape `(npts, ndim)`.
/// \param knots_flat Root per-axis breakpoints concatenated end to end.
/// \param knot_starts Per-axis start offset into `knots_flat`.
/// \param root_cells_per_axis Per-axis root cell counts.
/// \param factor Per-axis subdivision factor.
/// \param block_lo Packed block lower bounds, shape `(n_blocks, ndim)`.
/// \param block_hi Packed block upper bounds, same shape.
/// \param block_base Flat-id base per block.
/// \param level_block_start Block index range per level.
/// \param out Output flat cell ids, shape `(npts,)`; -1 outside the root domain.
template <Real T>
void hier_locate_points(span2d<const T> points, std::span<const T> knots_flat,
                        std::span<const std::int64_t> knot_starts,
                        std::span<const std::int64_t> root_cells_per_axis,
                        std::span<const std::int64_t> factor,
                        span2d<const std::int64_t> block_lo, span2d<const std::int64_t> block_hi,
                        std::span<const std::int64_t> block_base,
                        std::span<const std::int64_t> level_block_start,
                        std::span<std::int64_t> out) {
    using pantr::value_of;

    const std::size_t npts = points.extent(0);
    const std::size_t ndim = points.extent(1);
    const std::int64_t n_levels = static_cast<std::int64_t>(level_block_start.size()) - 1;

    std::vector<std::int64_t> midx(ndim);
    std::vector<T> lo(ndim);
    std::vector<T> hi(ndim);

    for (std::size_t p = 0; p < npts; ++p) {
        // Root-level location: one independent lower_bound per axis.
        bool inside = true;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::int64_t start = knot_starts[d];
            const std::int64_t ncells = root_cells_per_axis[d];
            const auto x = value_of(points(p, d));

            if (x < value_of(knots_flat[static_cast<std::size_t>(start)])
                || x > value_of(knots_flat[static_cast<std::size_t>(start + ncells)])) {
                inside = false;
                break;
            }
            std::int64_t lo_i = 0;
            std::int64_t hi_i = ncells + 1;
            while (lo_i < hi_i) {
                const std::int64_t mid = (lo_i + hi_i) / 2;
                if (value_of(knots_flat[static_cast<std::size_t>(start + mid)]) < x) {
                    lo_i = mid + 1;
                } else {
                    hi_i = mid;
                }
            }
            std::int64_t cell_i = lo_i - 1;
            if (cell_i < 0) {
                cell_i = 0;
            } else if (cell_i > ncells - 1) {
                cell_i = ncells - 1;
            }
            midx[d] = cell_i;
            lo[d] = knots_flat[static_cast<std::size_t>(start + cell_i)];
            hi[d] = knots_flat[static_cast<std::size_t>(start + cell_i + 1)];
        }
        if (!inside) {
            out[p] = -1;
            continue;
        }

        // Top-down descent through the levels.
        std::int64_t result = -1;
        for (std::int64_t level = 0; level < n_levels; ++level) {
            const std::int64_t cid =
                block_of_midx(level, std::span<const std::int64_t>(midx), block_lo, block_hi,
                              block_base, level_block_start);
            if (cid >= 0) {
                result = cid;
                break;
            }
            if (level >= n_levels - 1) {
                break;  // unreachable in a consistent grid
            }
            for (std::size_t k = 0; k < ndim; ++k) {
                const std::int64_t fk = factor[k];
                const T size_k = static_cast<T>((hi[k] - lo[k]) / T(static_cast<double>(fk)));
                auto j = static_cast<std::int64_t>(
                    value_of(static_cast<T>((points(p, k) - lo[k]) / size_k)));
                if (j < 0) {
                    j = 0;
                } else if (j > fk - 1) {
                    j = fk - 1;
                }
                // The product is named rather than inlined, and that is the whole of
                // this port's exactness guarantee rather than a style choice.
                // `lo + j * size` is one expression, so a target with a fused
                // multiply-add may contract it, and `-ffp-contract=on` permits
                // exactly that within an expression. The oracle never fuses
                // (numba defaults to `fastmath=False`), so the two would then
                // disagree -- and because the next line truncates a quotient, they
                // would disagree on a CELL ID, which Rule 11 says no tolerance
                // bounds. Splitting the statement puts the multiplication's
                // rounding back where the oracle has it. Measured: `vmulsd` plus
                // `vaddsd` at `-march=x86-64-v3` and at `-march=native`, against
                // `vfmadd132sd` for the single-expression form.
                const T step = T(static_cast<double>(j)) * size_k;
                lo[k] = lo[k] + step;
                hi[k] = lo[k] + size_k;
                midx[k] = midx[k] * fk + j;
            }
        }
        out[p] = result;
    }
}

/// Materialize per-cell `(lo, hi)` bounds in flat-id order.
///
/// Mirrors `_hier_core._hier_collect_cell_bounds_core`. No validation is
/// performed. The second of the two kernels whose equality rests on measurement.
///
/// \tparam T Coordinate type.
/// \param knots_flat Root per-axis breakpoints concatenated end to end.
/// \param knot_starts Per-axis start offset into `knots_flat`.
/// \param factor Per-axis subdivision factor.
/// \param block_lo Packed block lower bounds, shape `(n_blocks, ndim)`.
/// \param block_hi Packed block upper bounds, same shape.
/// \param block_base Flat-id base per block.
/// \param level_block_start Block index range per level.
/// \param out_lo Output lower corners, shape `(num_cells, ndim)`.
/// \param out_hi Output upper corners, same shape.
template <Real T>
void hier_collect_cell_bounds(std::span<const T> knots_flat,
                              std::span<const std::int64_t> knot_starts,
                              std::span<const std::int64_t> factor,
                              span2d<const std::int64_t> block_lo,
                              span2d<const std::int64_t> block_hi,
                              std::span<const std::int64_t> block_base,
                              std::span<const std::int64_t> level_block_start, span2d<T> out_lo,
                              span2d<T> out_hi) {
    const std::size_t ndim = block_lo.extent(1);
    const std::int64_t n_levels = static_cast<std::int64_t>(level_block_start.size()) - 1;
    const auto n_blocks = static_cast<std::int64_t>(block_base.size());

    std::vector<std::int64_t> m_pow(ndim);
    std::vector<std::int64_t> midx(ndim);

    for (std::int64_t bi = 0; bi < n_blocks; ++bi) {
        const auto b = static_cast<std::size_t>(bi);

        // Recover the block's level; blocks are packed level by level.
        std::int64_t level = 0;
        for (std::int64_t lev = 0; lev < n_levels; ++lev) {
            const auto l = static_cast<std::size_t>(lev);
            if (level_block_start[l] <= bi && bi < level_block_start[l + 1]) {
                level = lev;
                break;
            }
        }

        // Per-axis subdivision, by repeated multiplication as the oracle does.
        for (std::size_t k = 0; k < ndim; ++k) {
            std::int64_t mp = 1;
            for (std::int64_t i = 0; i < level; ++i) {
                mp *= factor[k];
            }
            m_pow[k] = mp;
        }

        // Odometer over the block's cells in C-order.
        std::int64_t n_block = 1;
        for (std::size_t k = 0; k < ndim; ++k) {
            midx[k] = block_lo(b, k);
            n_block *= block_hi(b, k) - block_lo(b, k);
        }

        std::int64_t flat_id = block_base[b];
        for (std::int64_t c = 0; c < n_block; ++c) {
            const auto row = static_cast<std::size_t>(flat_id);
            for (std::size_t k = 0; k < ndim; ++k) {
                const std::int64_t ik = midx[k];
                const std::int64_t root_ik = ik / m_pow[k];
                const std::int64_t sub_ik = ik % m_pow[k];
                const T root_lo_k = knots_flat[static_cast<std::size_t>(knot_starts[k] + root_ik)];
                const T root_hi_k =
                    knots_flat[static_cast<std::size_t>(knot_starts[k] + root_ik + 1)];
                const T size_k = static_cast<T>((root_hi_k - root_lo_k)
                                                / T(static_cast<double>(m_pow[k])));
                // Split for the reason given at the descent site. This one is
                // NOT protected by a small subdivision factor: `sub_ik` runs over
                // `[0, factor**level)`, so it reaches 3 at level 2 even at factor
                // 2, and a divergence was measured at factor 2 on the root cell
                // [1.0, 1.1] from level 6.
                const T offset = T(static_cast<double>(sub_ik)) * size_k;
                out_lo(row, k) = root_lo_k + offset;
                out_hi(row, k) = out_lo(row, k) + size_k;
            }
            ++flat_id;
            // Increment the odometer, last axis fastest.
            for (std::int64_t k = static_cast<std::int64_t>(ndim) - 1; k >= 0; --k) {
                const auto kk = static_cast<std::size_t>(k);
                midx[kk] += 1;
                if (midx[kk] < block_hi(b, kk)) {
                    break;
                }
                midx[kk] = block_lo(b, kk);
            }
        }
    }
}

}  // namespace pantr::grid
