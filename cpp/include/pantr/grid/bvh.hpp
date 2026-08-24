#pragma once

/// \file
/// A bounding-volume hierarchy over grid-cell AABBs: build, and box-overlap query.
///
/// Ports `src/pantr/grid/_bvh_core.py`, which stays as the parity oracle.
///
/// ## Why this port keeps pantr's own BVH rather than adopting a library
///
/// `design/bvh.md` decides that, and the reason is not the obvious one: the five
/// node arrays are **public API**, not an implementation detail. A downstream
/// consumer does not call the query at all -- it walks the tree itself, probing at
/// each internal node with its own predicate. So the layout is a contract, and
/// swapping implementations would rewrite that consumer. The note also records
/// that the only query is box overlap, while every candidate library is tuned for
/// *ray* queries, so their advantages buy nothing here.
///
/// That makes the layout the thing this port must not disturb, which is why the
/// header reproduces it exactly rather than choosing a better one.
///
/// ## Why the parity claim is an equality, and the argument for each verdict
///
/// The build takes three **discrete** decisions, and `design/backend_parity.md`
/// Rule 11 warns that no tolerance bounds a discrete verdict. None is needed,
/// because each rests on a quantity that is bit-identical between the backends:
///
///  1. **The node AABB** is a running min/max over the cells' own stored corners.
///     No arithmetic at all -- the values are copied, not computed -- so the only
///     question is tie behaviour, and Python's two-argument `min` keeps its first
///     argument on a tie. The form below keeps the incumbent for the same reason,
///     which also settles the signed-zero trap `CLAUDE.md` records for
///     `np.minimum(-0.0, 0.0)`: that trap is about numpy's ufunc, the oracle uses
///     the builtin, and "keep the incumbent" is identical on either sign of zero.
///  2. **The longest axis** compares `hi - lo` per axis with a strict `>`, so a tie
///     keeps the lower axis, matching `np.argmax`. One subtraction of two
///     bit-identical values; there is no fused-multiply-add site in it.
///  3. **The median split** is a *stable* sort of the node's cells by centroid.
///     Stability is what makes this exactly reproducible rather than merely
///     similar: a stable sort's output permutation is **uniquely determined** --
///     indices ordered by value, ties in increasing index order -- so any two
///     correct stable sorts agree, and `std::stable_sort` therefore reproduces
///     `np.argsort(kind="mergesort")` by argument and not by coincidence.
///
/// The centroid itself is `0.5 * (lo + hi)`: the addition rounds once and the
/// multiplication by `0.5` is exact, and because the multiply follows the add
/// there is no product for a compiler to fuse into it. So `-ffp-contract=on`
/// cannot move it, which is what separates this kernel from
/// `pantr/bezier/root_finding.hpp`, where contraction is live and Rule 10 has to
/// budget for it.
///
/// The stable sort's comparator is a valid strict weak ordering only on finite
/// values, and a NaN would make it undefined behaviour rather than merely
/// divergent. It cannot arrive: `_bvh.py:322` rejects any non-finite corner before
/// the kernel is reached. Checked rather than assumed, because the consequence of
/// being wrong here is UB and not a wrong answer.
///
/// ## The query is exact for a duller reason
///
/// Both query kernels are comparisons on stored node bounds plus an integer
/// stack. The overlap test is inclusive on every face, so a query box sharing a
/// face with a cell is reported -- matching `pantr.geometry.AABB.overlaps` -- and
/// that too is a discrete verdict decided without arithmetic.
///
/// Count and emit must agree on the output size, so they share one traversal
/// order: the left child is pushed first and the right last, so the stack pops
/// right first. That is the opposite direction from the build's left-to-right
/// preorder, and the oracle's own docstring records that only count/emit
/// consistency matters. Reproduced as it stands rather than tidied, because
/// tidying it would change which cells land at which output index.
///
/// ## Allocation
///
/// The build allocates its scratch; the queries do not. The split is deliberate
/// and follows `pantr/change_basis/change_basis.hpp`, which allocates a triangle
/// for the same reason: a one-off construction is not an inner loop, and the
/// alternative is a work-span argument per scratch array that Layer 2 would have
/// to size. The queries keep the fixed stack and no allocation, which is where the
/// kernel discipline earns its keep.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::grid {

/// Maximum depth of the iterative-descent stack.
///
/// Mirrors `_bvh_core._BVH_STACK_DEPTH`. The oracle's copy is the source of truth
/// and stays where it is: a downstream consumer imports it from that path, and
/// Layer 2 validates a tree against it before either query kernel runs, so this
/// constant is never the one a caller is checked against.
inline constexpr std::int64_t kBvhStackDepth = 128;

/// Build the BVH arrays from per-cell AABBs, by median-of-longest-axis splits.
///
/// Mirrors `_bvh_core._bvh_build_core`. No validation is performed.
///
/// \tparam T Coordinate type.
/// \param cell_lo Per-cell lo corners, shape `(n_cells, ndim)`, `n_cells >= 1`.
/// \param cell_hi Per-cell hi corners, same shape; `hi >= lo` is assumed.
/// \param node_lo Output per-node lo corners, shape `(2 * n_cells - 1, ndim)`.
/// \param node_hi Output per-node hi corners, same shape.
/// \param node_left Output left-child indices, shape `(2 * n_cells - 1,)`;
///        must be pre-filled with -1.
/// \param node_right Output right-child indices, same shape and pre-fill.
/// \param node_cell Output per-leaf cell ids, same shape and pre-fill.
template <Real T>
void bvh_build(span2d<const T> cell_lo, span2d<const T> cell_hi, span2d<T> node_lo,
               span2d<T> node_hi, std::span<std::int64_t> node_left,
               std::span<std::int64_t> node_right, std::span<std::int64_t> node_cell) {
    using pantr::value_of;

    const auto n_cells = static_cast<std::int64_t>(cell_lo.extent(0));
    const std::size_t ndim = cell_lo.extent(1);

    std::vector<T> centroid(static_cast<std::size_t>(n_cells) * ndim);
    for (std::int64_t c = 0; c < n_cells; ++c) {
        for (std::size_t k = 0; k < ndim; ++k) {
            const auto i = static_cast<std::size_t>(c);
            centroid[i * ndim + k] = T(0.5) * (cell_lo(i, k) + cell_hi(i, k));
        }
    }

    std::vector<std::int64_t> perm(static_cast<std::size_t>(n_cells));
    std::iota(perm.begin(), perm.end(), std::int64_t{0});

    // Work stack of (node_idx, cell_start, cell_end) triples. Its peak size is
    // bounded by the tree height, which Layer 2 has validated against the depth.
    std::vector<std::int64_t> stack_idx(static_cast<std::size_t>(kBvhStackDepth));
    std::vector<std::int64_t> stack_start(static_cast<std::size_t>(kBvhStackDepth));
    std::vector<std::int64_t> stack_end(static_cast<std::size_t>(kBvhStackDepth));
    stack_idx[0] = 0;
    stack_start[0] = 0;
    stack_end[0] = n_cells;
    std::int64_t sp = 1;
    std::int64_t next_idx = 1;

    std::vector<T> vals;
    std::vector<std::int64_t> order;
    std::vector<std::int64_t> sub;

    while (sp > 0) {
        --sp;
        const std::int64_t idx = stack_idx[static_cast<std::size_t>(sp)];
        const std::int64_t start = stack_start[static_cast<std::size_t>(sp)];
        const std::int64_t end = stack_end[static_cast<std::size_t>(sp)];
        const auto node = static_cast<std::size_t>(idx);

        // Tight AABB of the node: a running min/max over its cells, keeping the
        // incumbent on a tie exactly as the oracle's builtin min/max do.
        for (std::size_t k = 0; k < ndim; ++k) {
            const auto first = static_cast<std::size_t>(perm[static_cast<std::size_t>(start)]);
            T lo_k = cell_lo(first, k);
            T hi_k = cell_hi(first, k);
            for (std::int64_t c = start + 1; c < end; ++c) {
                const auto i = static_cast<std::size_t>(perm[static_cast<std::size_t>(c)]);
                const T v_lo = cell_lo(i, k);
                const T v_hi = cell_hi(i, k);
                if (value_of(v_lo) < value_of(lo_k)) {
                    lo_k = v_lo;
                }
                if (value_of(v_hi) > value_of(hi_k)) {
                    hi_k = v_hi;
                }
            }
            node_lo(node, k) = lo_k;
            node_hi(node, k) = hi_k;
        }

        const std::int64_t count = end - start;
        if (count == 1) {
            node_cell[node] = perm[static_cast<std::size_t>(start)];
            continue;
        }

        // Longest axis of the node AABB, first maximum, matching np.argmax.
        std::size_t axis = 0;
        auto best = value_of(static_cast<T>(node_hi(node, 0) - node_lo(node, 0)));
        for (std::size_t k = 1; k < ndim; ++k) {
            const auto extent_k = value_of(static_cast<T>(node_hi(node, k) - node_lo(node, k)));
            if (extent_k > best) {
                best = extent_k;
                axis = k;
            }
        }

        // Stable sort of the node's cells by centroid along the split axis.
        const auto n = static_cast<std::size_t>(count);
        vals.resize(n);
        order.resize(n);
        sub.resize(n);
        for (std::size_t c = 0; c < n; ++c) {
            const auto i = static_cast<std::size_t>(perm[static_cast<std::size_t>(start) + c]);
            vals[c] = centroid[i * ndim + axis];
        }
        std::iota(order.begin(), order.end(), std::int64_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&vals](std::int64_t a, std::int64_t b) {
                             return value_of(vals[static_cast<std::size_t>(a)])
                                    < value_of(vals[static_cast<std::size_t>(b)]);
                         });
        for (std::size_t c = 0; c < n; ++c) {
            sub[c] = perm[static_cast<std::size_t>(start) + c];
        }
        for (std::size_t c = 0; c < n; ++c) {
            perm[static_cast<std::size_t>(start) + c] =
                sub[static_cast<std::size_t>(order[c])];
        }

        const std::int64_t mid = start + count / 2;
        const std::int64_t left_idx = next_idx;
        const std::int64_t right_idx = next_idx + 1;
        next_idx += 2;
        node_left[node] = left_idx;
        node_right[node] = right_idx;

        // Push right first so left pops first: preorder, left to right, matching
        // the construction order the oracle's module docstring declares.
        stack_idx[static_cast<std::size_t>(sp)] = right_idx;
        stack_start[static_cast<std::size_t>(sp)] = mid;
        stack_end[static_cast<std::size_t>(sp)] = end;
        stack_idx[static_cast<std::size_t>(sp + 1)] = left_idx;
        stack_start[static_cast<std::size_t>(sp + 1)] = start;
        stack_end[static_cast<std::size_t>(sp + 1)] = mid;
        sp += 2;
    }
}

/// Whether a node's AABB overlaps the query box, inclusive on every face.
///
/// Factored out so that count and emit cannot drift from each other on the tie
/// contract, which is the one thing about them that is not obviously the same.
template <Real T>
[[nodiscard]] bool node_overlaps(std::span<const T> qlo, std::span<const T> qhi,
                                 span2d<const T> node_lo, span2d<const T> node_hi,
                                 std::size_t node) {
    using pantr::value_of;
    for (std::size_t d = 0; d < qlo.size(); ++d) {
        if (value_of(qhi[d]) < value_of(node_lo(node, d))
            || value_of(qlo[d]) > value_of(node_hi(node, d))) {
            return false;
        }
    }
    return true;
}

/// Count the leaves whose AABB overlaps the query box.
///
/// Mirrors `_bvh_core._bvh_query_count_core`. No validation is performed. Called
/// before the emit pass so Layer 2 can allocate an exact-size output array.
///
/// \tparam T Coordinate type.
/// \param qlo Query box lo corner, shape `(ndim,)`.
/// \param qhi Query box hi corner, same shape.
/// \param node_lo Per-node AABB lower corners, shape `(n_nodes, ndim)`.
/// \param node_hi Per-node AABB upper corners, same shape.
/// \param node_left Left-child indices; -1 marks a leaf.
/// \param node_right Right-child indices.
/// \param node_cell Per-leaf cell ids; -1 marks an internal node.
/// \return The number of overlapping leaves.
template <Real T>
[[nodiscard]] std::int64_t bvh_query_count(std::span<const T> qlo, std::span<const T> qhi,
                                           span2d<const T> node_lo, span2d<const T> node_hi,
                                           std::span<const std::int64_t> node_left,
                                           std::span<const std::int64_t> node_right,
                                           std::span<const std::int64_t> node_cell) {
    std::array<std::int64_t, static_cast<std::size_t>(kBvhStackDepth)> stack{};
    stack[0] = 0;
    std::int64_t top = 1;
    std::int64_t count = 0;

    while (top > 0) {
        --top;
        const auto node = static_cast<std::size_t>(stack[top]);
        if (!node_overlaps<T>(qlo, qhi, node_lo, node_hi, node)) {
            continue;
        }
        if (node_cell[node] >= 0) {
            ++count;
        } else {
            stack[top] = node_left[node];
            ++top;
            stack[top] = node_right[node];
            ++top;
        }
    }
    return count;
}

/// Fill `out` with the cell ids of the leaves whose AABB overlaps the query box.
///
/// Mirrors `_bvh_core._bvh_query_emit_core`. No validation is performed. Visits
/// nodes in the same order as `bvh_query_count`, so the two agree on the size.
///
/// \tparam T Coordinate type.
/// \param qlo Query box lo corner, shape `(ndim,)`.
/// \param qhi Query box hi corner, same shape.
/// \param node_lo Per-node AABB lower corners, shape `(n_nodes, ndim)`.
/// \param node_hi Per-node AABB upper corners, same shape.
/// \param node_left Left-child indices; -1 marks a leaf.
/// \param node_right Right-child indices.
/// \param node_cell Per-leaf cell ids; -1 marks an internal node.
/// \param out Output cell ids, sized by a prior `bvh_query_count`.
/// \return The number of cell ids written.
template <Real T>
std::int64_t bvh_query_emit(std::span<const T> qlo, std::span<const T> qhi,
                            span2d<const T> node_lo, span2d<const T> node_hi,
                            std::span<const std::int64_t> node_left,
                            std::span<const std::int64_t> node_right,
                            std::span<const std::int64_t> node_cell,
                            std::span<std::int64_t> out) {
    std::array<std::int64_t, static_cast<std::size_t>(kBvhStackDepth)> stack{};
    stack[0] = 0;
    std::int64_t top = 1;
    std::int64_t count = 0;

    while (top > 0) {
        --top;
        const auto node = static_cast<std::size_t>(stack[top]);
        if (!node_overlaps<T>(qlo, qhi, node_lo, node_hi, node)) {
            continue;
        }
        const std::int64_t cell = node_cell[node];
        if (cell >= 0) {
            out[static_cast<std::size_t>(count)] = cell;
            ++count;
        } else {
            stack[top] = node_left[node];
            ++top;
            stack[top] = node_right[node];
            ++top;
        }
    }
    return count;
}

}  // namespace pantr::grid
