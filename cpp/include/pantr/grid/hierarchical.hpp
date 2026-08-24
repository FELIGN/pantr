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
/// `lo[k]` is rewritten each level from an expression a compiler may contract into
/// a fused multiply-add, and the rewritten value feeds the next level's `size_k`,
/// its truncation and its own update. So the two backends can in principle take
/// different branches, and `design/backend_parity.md` Rule 11 is explicit that no
/// tolerance bounds a branch.
///
/// **Measured, they do not.** 300000 descents of twelve levels, spans and
/// subdivision factors varied, 23.8% of the 3.6 million child decisions taken at a
/// `j` that is not a power of two so the product is genuinely inexact, and half the
/// query points placed one ulp off an exact child boundary: zero child-index
/// sequences differ. The mechanism is that the perturbation is never introduced
/// rather than that it fails to grow -- fusing changes the sum only when `|lo|` and
/// `|j * size|` are comparable, and in a descent the second shrinks geometrically
/// while the first does not, so within two or three levels the ratio is past the
/// point where any difference survives.
///
/// **That is evidence and not proof**, and it is deliberately held to the same
/// standard as Rule 11's own note about the Bézier root set: the primitive
/// demonstrably *can* differ, so a divergence is possible and simply did not occur.
/// The operational consequence is the one Rule 11 states: a caller comparing the
/// two backends must assert the returned **ids** on their own, before and
/// separately from any numeric comparison, because a changed id is a changed
/// verdict and not a displaced value.
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

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pantr/core/mdspan.hpp"
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
                lo[k] = lo[k] + T(static_cast<double>(j)) * size_k;
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
                out_lo(row, k) = root_lo_k + T(static_cast<double>(sub_ik)) * size_k;
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
