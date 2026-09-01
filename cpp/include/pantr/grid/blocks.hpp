#pragma once

/// \file
/// The rectangle algebra a hierarchical grid's active set is made of.
///
/// Ports the module-level block helpers of `src/pantr/grid/_hierarchical_grid.py`
/// (`_block_size`, `_rect_intersect`, `_peel`, `_try_merge`, `_normalize_blocks`), which
/// stay as the parity oracle. A `HierarchicalGrid` stores its active leaves as a list of
/// axis-aligned integer rectangles per level; everything here operates on those lists
/// and nothing here knows about levels, coordinates or a grid.
///
/// ## Every quantity here is an integer, and that fixes the parity bar
///
/// There is no floating point below -- no coordinate, no tolerance, no rounding. So the
/// acceptance criterion against the oracle is **exact equality**, which is the one case
/// where bit-identity is a property of the mathematics rather than of the build. That is
/// also why the file needs no counterpart to `hierarchical.hpp`'s long argument about
/// contraction: there is no product of coordinates for a compiler to fuse.
///
/// ## Why the merge order is a contract and not an implementation detail
///
/// `normalize_blocks` merges greedily, and **the greedy merge is order-dependent**: two
/// permutations of one decomposition can normalise to two different -- both valid, both
/// equal-area -- partitions. Measured on the oracle: of 3886 random decompositions, 1852
/// changed partition under a reshuffle, while all 3886 were idempotent. Both properties
/// are pinned by tests on the Python side, and the sweep is reproduced in
/// `cpp/tests/test_grid_blocks.cpp`.
///
/// That would be harmless if the partition were private. It is not: flat cell ids are
/// handed out block by block, so **the partition is observable through the cell ids**.
/// A port that merged in a different order would give the same cells different numbers,
/// and the two backends would disagree about every id with everything else identical.
///
/// So the loop below is a transliteration and not a rewrite, and three parts of it are
/// load-bearing rather than incidental:
///
///  - **The pass repeats to a fixed point**, and each pass re-scans the list *in the
///    order the previous pass left it*, which is the order surviving blocks were
///    appended in -- not their sorted order.
///  - **A block absorbs further blocks within one inner scan.** `merged` is updated in
///    place as soon as a merge succeeds, so the next `j` is tried against the enlarged
///    rectangle rather than the original one.
///  - **The sort happens once, at the very end.** It fixes the *output* order; it does
///    not decide which merges happened.
///
/// `std::stable_sort` rather than `std::sort`, matching Python's `sorted`. For a valid
/// active-leaf decomposition the two agree, because non-overlapping rectangles have
/// distinct lower corners and the key is therefore a strict total order -- but the
/// stable spelling costs nothing and removes the question, rather than leaving a reader
/// to reconstruct that argument.
///
/// ## Contracts the caller owes, and what happens if it does not
///
/// A block is non-empty (`lo[k] < hi[k]` on every axis) and a list is pairwise
/// non-overlapping. `peel` additionally needs its inner rectangle contained in its
/// outer one. These are **correctness** obligations in the sense of
/// `pantr/core/precondition.hpp`: violating one gives a wrong answer here and a wrong
/// answer in the oracle, not undefined behaviour, so they are documented and not
/// asserted.
///
/// `block_size` multiplies extents in `std::int64_t` where the oracle multiplies in
/// Python's arbitrary-precision integers, so a rectangle whose cell count exceeds
/// `int64` overflows here and does not there. It is left unchecked at this level for the
/// same reason `TensorProductGrid` checks its product one level up: the accumulation
/// that can actually reach the limit is the grid's own cell count, which is where the
/// guard belongs and where the divergence is worth an exception.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace pantr::grid {

/// A view of one axis-aligned integer rectangle `[lo, hi)`.
///
/// Bundled rather than passed as two spans so that a call site cannot transpose the
/// corners, and so that the two rectangles a binary operation takes are one argument
/// each instead of four in a row.
struct BlockView {
    std::span<const std::int64_t> lo;  ///< Per-axis lower bound, inclusive.
    std::span<const std::int64_t> hi;  ///< Per-axis upper bound, exclusive.
};

/// A list of axis-aligned integer rectangles over a fixed number of axes.
///
/// Stored as two flat `size() * ndim()` buffers rather than as a vector of small
/// vectors. That is the layout the kernels in `pantr/grid/hierarchical.hpp` already
/// take, so a grid holding one of these per level needs no second, packed copy of its
/// own active set -- the list *is* the packed descriptor, in the order flat cell ids are
/// assigned.
class BlockList {
  public:
    /// Construct an empty list over zero axes.
    BlockList() = default;

    /// Construct an empty list over `ndim` axes.
    ///
    /// \param ndim Number of axes; must be non-negative.
    explicit BlockList(std::int64_t ndim) : ndim_(ndim) {}

    /// Number of axes each rectangle spans.
    ///
    /// \return The axis count fixed at construction.
    [[nodiscard]] std::int64_t ndim() const noexcept { return ndim_; }

    /// Number of rectangles held.
    ///
    /// \return The rectangle count.
    [[nodiscard]] std::size_t size() const noexcept {
        return ndim_ == 0 ? 0 : lo_.size() / static_cast<std::size_t>(ndim_);
    }

    /// Report whether the list holds no rectangles.
    ///
    /// \return `true` when `size()` is zero.
    [[nodiscard]] bool empty() const noexcept { return lo_.empty(); }

    /// View one rectangle.
    ///
    /// \param i Index in `[0, size())`.
    /// \return A view of rectangle `i`, valid until the list is modified.
    [[nodiscard]] BlockView operator[](std::size_t i) const noexcept {
        const auto d = static_cast<std::size_t>(ndim_);
        return BlockView{std::span<const std::int64_t>(lo_.data() + i * d, d),
                         std::span<const std::int64_t>(hi_.data() + i * d, d)};
    }

    /// Append a rectangle.
    ///
    /// \param block The rectangle to append; its corners are copied. Both spans must
    ///        have `ndim()` entries.
    void push_back(BlockView block) {
        lo_.insert(lo_.end(), block.lo.begin(), block.lo.end());
        hi_.insert(hi_.end(), block.hi.begin(), block.hi.end());
    }

    /// Reserve room for `n` rectangles.
    ///
    /// \param n Expected rectangle count.
    void reserve(std::size_t n) {
        const auto d = static_cast<std::size_t>(ndim_);
        lo_.reserve(n * d);
        hi_.reserve(n * d);
    }

    /// The flat lower-corner buffer, `size() * ndim()` entries in rectangle order.
    ///
    /// \return A view of the buffer, valid until the list is modified.
    [[nodiscard]] std::span<const std::int64_t> lo_flat() const noexcept { return lo_; }

    /// The flat upper-corner buffer, `size() * ndim()` entries in rectangle order.
    ///
    /// \return A view of the buffer, valid until the list is modified.
    [[nodiscard]] std::span<const std::int64_t> hi_flat() const noexcept { return hi_; }

    /// Compare two lists rectangle by rectangle.
    ///
    /// \param other The list to compare against.
    /// \return `true` when both hold the same rectangles in the same order.
    [[nodiscard]] bool operator==(const BlockList& other) const noexcept {
        return ndim_ == other.ndim_ && lo_ == other.lo_ && hi_ == other.hi_;
    }

  private:
    std::int64_t ndim_ = 0;            ///< Axis count; every rectangle spans this many.
    std::vector<std::int64_t> lo_;     ///< Lower corners, `size() * ndim_` entries.
    std::vector<std::int64_t> hi_;     ///< Upper corners, same layout.
};

/// Number of cells in the integer rectangle `[lo, hi)`.
///
/// \param block The rectangle.
/// \return The product of `hi[k] - lo[k]` over every axis; `1` for zero axes.
/// \note No validation. The product is accumulated in `std::int64_t` and is not checked
///       for overflow; see the file header for why the guard belongs one level up.
[[nodiscard]] inline std::int64_t block_size(BlockView block) noexcept {
    std::int64_t size = 1;
    for (std::size_t k = 0; k < block.lo.size(); ++k) {
        size *= block.hi[k] - block.lo[k];
    }
    return size;
}

/// Intersect two rectangles.
///
/// \param a First rectangle.
/// \param b Second rectangle; the operation is symmetric in the two.
/// \param out_lo Receives the intersection's lower corner. Untouched when disjoint.
/// \param out_hi Receives the intersection's upper corner. Untouched when disjoint.
/// \return `true` when the intersection is non-empty.
/// \note No validation. Both rectangles and both outputs must span the same axes.
[[nodiscard]] inline bool rect_intersect(BlockView a, BlockView b,
                                         std::span<std::int64_t> out_lo,
                                         std::span<std::int64_t> out_hi) noexcept {
    for (std::size_t k = 0; k < a.lo.size(); ++k) {
        const std::int64_t lo_k = std::max(a.lo[k], b.lo[k]);
        const std::int64_t hi_k = std::min(a.hi[k], b.hi[k]);
        if (lo_k >= hi_k) {
            return false;
        }
        out_lo[k] = lo_k;
        out_hi[k] = hi_k;
    }
    return true;
}

/// Subtract `inner` from `outer`, appending the remainder to `out`.
///
/// Peels axis by axis: on each axis in turn, the slab below `inner` and then the slab
/// above it are cut from what is left, so the result is at most `2 * ndim`
/// non-overlapping rectangles covering `outer \ inner`. Empty slabs are dropped.
///
/// \param outer The rectangle to subtract from.
/// \param inner The rectangle to remove; must be contained in `outer`.
/// \param out Receives the remainder, appended in axis order. Not cleared first.
/// \note No validation. `inner` outside `outer` gives a wrong decomposition, here and in
///       the oracle alike.
inline void peel(BlockView outer, BlockView inner, BlockList& out) {
    const std::size_t nd = outer.lo.size();
    std::vector<std::int64_t> a(outer.lo.begin(), outer.lo.end());
    std::vector<std::int64_t> b(outer.hi.begin(), outer.hi.end());
    std::vector<std::int64_t> slab_lo(nd);
    std::vector<std::int64_t> slab_hi(nd);
    for (std::size_t k = 0; k < nd; ++k) {
        if (a[k] < inner.lo[k]) {
            std::copy(a.begin(), a.end(), slab_lo.begin());
            std::copy(b.begin(), b.end(), slab_hi.begin());
            slab_hi[k] = inner.lo[k];
            const BlockView slab{slab_lo, slab_hi};
            if (block_size(slab) > 0) {
                out.push_back(slab);
            }
            a[k] = inner.lo[k];
        }
        if (inner.hi[k] < b[k]) {
            std::copy(a.begin(), a.end(), slab_lo.begin());
            std::copy(b.begin(), b.end(), slab_hi.begin());
            slab_lo[k] = inner.hi[k];
            const BlockView slab{slab_lo, slab_hi};
            if (block_size(slab) > 0) {
                out.push_back(slab);
            }
            b[k] = inner.hi[k];
        }
    }
}

/// Merge two rectangles that agree on every axis but one and are face-adjacent there.
///
/// \param a First rectangle.
/// \param b Second rectangle; the operation is symmetric in the two.
/// \param out_lo Receives the merged lower corner. Untouched when the merge fails.
/// \param out_hi Receives the merged upper corner. Untouched when the merge fails.
/// \return `true` when the two merged.
/// \note No validation, and the outputs must not alias the inputs: every axis is read
///       before any is written.
[[nodiscard]] inline bool try_merge(BlockView a, BlockView b, std::span<std::int64_t> out_lo,
                                    std::span<std::int64_t> out_hi) noexcept {
    std::size_t merge_axis = 0;
    bool found = false;
    for (std::size_t k = 0; k < a.lo.size(); ++k) {
        if (a.lo[k] == b.lo[k] && a.hi[k] == b.hi[k]) {
            continue;
        }
        if (found) {
            return false;  // differs on two axes
        }
        if (a.hi[k] == b.lo[k] || b.hi[k] == a.lo[k]) {
            merge_axis = k;
            found = true;
        } else {
            return false;  // differs on this axis without touching along it
        }
    }
    if (!found) {
        return false;  // the same rectangle twice
    }
    std::copy(a.lo.begin(), a.lo.end(), out_lo.begin());
    std::copy(a.hi.begin(), a.hi.end(), out_hi.begin());
    out_lo[merge_axis] = std::min(a.lo[merge_axis], b.lo[merge_axis]);
    out_hi[merge_axis] = std::max(a.hi[merge_axis], b.hi[merge_axis]);
    return true;
}

namespace detail {

/// Return `blocks` sorted lexicographically by lower corner, then upper corner.
///
/// \param blocks The list to sort; not modified.
/// \return A fresh list holding the same rectangles in sorted order.
[[nodiscard]] inline BlockList sorted_blocks(const BlockList& blocks) {
    const auto d = static_cast<std::size_t>(blocks.ndim());
    std::vector<std::size_t> order(blocks.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    const std::span<const std::int64_t> lo = blocks.lo_flat();
    const std::span<const std::int64_t> hi = blocks.hi_flat();
    std::stable_sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
        for (std::size_t k = 0; k < d; ++k) {
            if (lo[i * d + k] != lo[j * d + k]) {
                return lo[i * d + k] < lo[j * d + k];
            }
        }
        for (std::size_t k = 0; k < d; ++k) {
            if (hi[i * d + k] != hi[j * d + k]) {
                return hi[i * d + k] < hi[j * d + k];
            }
        }
        return false;
    });
    BlockList result(blocks.ndim());
    result.reserve(order.size());
    for (const std::size_t i : order) {
        result.push_back(blocks[i]);
    }
    return result;
}

}  // namespace detail

/// Sort a block list and greedily merge adjacent aligned pairs, to a fixed point.
///
/// The output is sorted and pairwise non-mergeable, so re-running this is the identity;
/// **which** rectangles it produces, however, depends on the order they arrive in. The
/// file header says why that order is a contract rather than an implementation detail.
///
/// \param blocks A pairwise non-overlapping list; not modified.
/// \return A fresh, sorted, compacted list covering the same cells.
/// \note No validation. Overlapping input gives a wrong answer, here and in the oracle.
[[nodiscard]] inline BlockList normalize_blocks(const BlockList& blocks) {
    if (blocks.size() <= 1) {
        return detail::sorted_blocks(blocks);
    }
    const auto d = static_cast<std::size_t>(blocks.ndim());
    BlockList current = blocks;
    std::vector<std::int64_t> merged_lo(d);
    std::vector<std::int64_t> merged_hi(d);
    std::vector<std::int64_t> candidate_lo(d);
    std::vector<std::int64_t> candidate_hi(d);

    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<bool> used(current.size(), false);
        BlockList next(blocks.ndim());
        next.reserve(current.size());
        for (std::size_t i = 0; i < current.size(); ++i) {
            if (used[i]) {
                continue;
            }
            const BlockView start = current[i];
            std::copy(start.lo.begin(), start.lo.end(), merged_lo.begin());
            std::copy(start.hi.begin(), start.hi.end(), merged_hi.begin());
            for (std::size_t j = i + 1; j < current.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                // `merged` grows in place, so the next `j` is tried against the enlarged
                // rectangle. That chaining is what makes the result order-dependent, and
                // it is the oracle's behaviour rather than an optimisation.
                if (try_merge(BlockView{merged_lo, merged_hi}, current[j], candidate_lo,
                              candidate_hi)) {
                    merged_lo = candidate_lo;
                    merged_hi = candidate_hi;
                    used[j] = true;
                    changed = true;
                }
            }
            next.push_back(BlockView{merged_lo, merged_hi});
        }
        current = std::move(next);
    }
    return detail::sorted_blocks(current);
}

}  // namespace pantr::grid
