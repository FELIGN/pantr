/// \file
/// The BVH build and its box-overlap query, checked against structural
/// properties rather than against the oracle.
///
/// Nothing here compares the two backends. What it checks is that the tree is a
/// tree, that every node's box actually bounds its subtree's cells, and that a
/// query returns exactly the cells whose boxes meet the query box -- the last one
/// against a brute-force scan, which is an independent algorithm rather than a
/// second copy of this one.

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "check.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/bvh.hpp"

namespace {

using pantr::span2d;
using pantr::grid::bvh_build;
using pantr::grid::bvh_query_count;
using pantr::grid::bvh_query_emit;

/// A built tree plus the cells it was built from.
struct Tree {
    std::int64_t n_cells;
    std::size_t ndim;
    std::vector<double> lo, hi;              // cells, row-major (n_cells, ndim)
    std::vector<double> nlo, nhi;            // nodes, row-major (2n-1, ndim)
    std::vector<std::int64_t> left, right, cell;
};

Tree build(std::int64_t n_cells, std::size_t ndim, const std::vector<double>& lo,
           const std::vector<double>& hi) {
    Tree t{n_cells, ndim, lo, hi, {}, {}, {}, {}, {}};
    const auto n_nodes = static_cast<std::size_t>(2 * n_cells - 1);
    t.nlo.assign(n_nodes * ndim, 0.0);
    t.nhi.assign(n_nodes * ndim, 0.0);
    t.left.assign(n_nodes, -1);
    t.right.assign(n_nodes, -1);
    t.cell.assign(n_nodes, -1);
    bvh_build<double>(span2d<const double>(t.lo.data(), static_cast<std::size_t>(n_cells), ndim),
                      span2d<const double>(t.hi.data(), static_cast<std::size_t>(n_cells), ndim),
                      span2d<double>(t.nlo.data(), n_nodes, ndim),
                      span2d<double>(t.nhi.data(), n_nodes, ndim), std::span<std::int64_t>(t.left),
                      std::span<std::int64_t>(t.right), std::span<std::int64_t>(t.cell));
    return t;
}

/// A deterministic spread of boxes; no Math.random-style nondeterminism.
Tree make_boxes(std::int64_t n, std::size_t ndim) {
    std::vector<double> lo(static_cast<std::size_t>(n) * ndim);
    std::vector<double> hi(lo.size());
    std::uint64_t s = 0x2545F4914F6CDD1DULL;
    auto next = [&s]() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<double>(s & 0xFFFFFFULL) / 16777216.0;
    };
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double a = -3.0 + 6.0 * next();
        lo[i] = a;
        hi[i] = a + 0.01 + 0.5 * next();
    }
    return build(n, ndim, lo, hi);
}

/// Collect the leaf cell ids under a node, and check the node bounds them.
void check_subtree(const Tree& t, std::size_t node, std::vector<std::int64_t>& leaves) {
    if (t.cell[node] >= 0) {
        PANTR_CHECK_MSG(t.left[node] == -1 && t.right[node] == -1,
                        "a leaf must have no children");
        leaves.push_back(t.cell[node]);
        return;
    }
    PANTR_CHECK_MSG(t.left[node] >= 0 && t.right[node] >= 0,
                    "an internal node must have both children");
    std::vector<std::int64_t> l, r;
    check_subtree(t, static_cast<std::size_t>(t.left[node]), l);
    check_subtree(t, static_cast<std::size_t>(t.right[node]), r);
    PANTR_CHECK_MSG(!l.empty() && !r.empty(), "neither side of a split may be empty");
    leaves.insert(leaves.end(), l.begin(), l.end());
    leaves.insert(leaves.end(), r.begin(), r.end());

    // The node's box must contain every cell below it, exactly -- it is a running
    // min/max of stored corners, so equality is the right assertion, not closeness.
    for (std::size_t k = 0; k < t.ndim; ++k) {
        double lo_k = t.hi[static_cast<std::size_t>(leaves[0]) * t.ndim + k];
        double hi_k = t.lo[static_cast<std::size_t>(leaves[0]) * t.ndim + k];
        for (const std::int64_t c : leaves) {
            lo_k = std::min(lo_k, t.lo[static_cast<std::size_t>(c) * t.ndim + k]);
            hi_k = std::max(hi_k, t.hi[static_cast<std::size_t>(c) * t.ndim + k]);
        }
        PANTR_CHECK_MSG(t.nlo[node * t.ndim + k] == lo_k, "node lo must be the exact min");
        PANTR_CHECK_MSG(t.nhi[node * t.ndim + k] == hi_k, "node hi must be the exact max");
    }
}

/// Every cell appears exactly once as a leaf, and every node bounds its subtree.
void the_tree_is_a_partition_and_its_boxes_bound() {
    for (const std::int64_t n : {1, 2, 3, 5, 8, 17, 64}) {
        for (const std::size_t ndim : {1u, 2u, 3u}) {
            const Tree t = make_boxes(n, ndim);
            std::vector<std::int64_t> leaves;
            check_subtree(t, 0, leaves);
            std::sort(leaves.begin(), leaves.end());
            PANTR_CHECK_MSG(static_cast<std::int64_t>(leaves.size()) == n,
                            "every cell must appear exactly once as a leaf");
            for (std::int64_t i = 0; i < n; ++i) {
                PANTR_CHECK(leaves[static_cast<std::size_t>(i)] == i);
            }
        }
    }
}

/// The query returns exactly the overlapping cells, against a brute-force scan.
void the_query_agrees_with_a_brute_force_scan() {
    for (const std::int64_t n : {1, 2, 7, 33}) {
        const std::size_t ndim = 2;
        const Tree t = make_boxes(n, ndim);
        const auto n_nodes = static_cast<std::size_t>(2 * n - 1);

        for (int q = 0; q < 40; ++q) {
            const double c = -3.5 + 0.2 * static_cast<double>(q);
            const std::vector<double> qlo{c, c - 0.3};
            const std::vector<double> qhi{c + 0.4, c + 0.6};

            std::vector<std::int64_t> expected;
            for (std::int64_t i = 0; i < n; ++i) {
                bool ov = true;
                for (std::size_t k = 0; k < ndim; ++k) {
                    const auto j = static_cast<std::size_t>(i) * ndim + k;
                    if (qhi[k] < t.lo[j] || qlo[k] > t.hi[j]) {
                        ov = false;
                        break;
                    }
                }
                if (ov) {
                    expected.push_back(i);
                }
            }

            const auto count = bvh_query_count<double>(
                std::span<const double>(qlo), std::span<const double>(qhi),
                span2d<const double>(t.nlo.data(), n_nodes, ndim),
                span2d<const double>(t.nhi.data(), n_nodes, ndim),
                std::span<const std::int64_t>(t.left), std::span<const std::int64_t>(t.right),
                std::span<const std::int64_t>(t.cell));
            PANTR_CHECK_MSG(count == static_cast<std::int64_t>(expected.size()),
                            "count must equal the brute-force overlap count");

            std::vector<std::int64_t> got(static_cast<std::size_t>(count), -1);
            const auto emitted = bvh_query_emit<double>(
                std::span<const double>(qlo), std::span<const double>(qhi),
                span2d<const double>(t.nlo.data(), n_nodes, ndim),
                span2d<const double>(t.nhi.data(), n_nodes, ndim),
                std::span<const std::int64_t>(t.left), std::span<const std::int64_t>(t.right),
                std::span<const std::int64_t>(t.cell), std::span<std::int64_t>(got));
            PANTR_CHECK_MSG(emitted == count, "emit must write exactly what count promised");
            std::sort(got.begin(), got.end());
            PANTR_CHECK(got == expected);
        }
    }
}

/// A query box sharing only a face with a cell is reported as overlapping.
///
/// The tie contract, and the one discrete verdict in this file worth its own case.
void a_shared_face_counts_as_overlap() {
    const Tree t = build(2, 1, {0.0, 1.0}, {1.0, 2.0});
    const std::size_t n_nodes = 3;
    // A degenerate query box exactly at the shared breakpoint x = 1.
    const std::vector<double> qlo{1.0};
    const std::vector<double> qhi{1.0};
    const auto count = bvh_query_count<double>(
        std::span<const double>(qlo), std::span<const double>(qhi),
        span2d<const double>(t.nlo.data(), n_nodes, 1),
        span2d<const double>(t.nhi.data(), n_nodes, 1), std::span<const std::int64_t>(t.left),
        std::span<const std::int64_t>(t.right), std::span<const std::int64_t>(t.cell));
    PANTR_CHECK_MSG(count == 2, "a box on the shared face must meet BOTH cells");
}

/// Cells with identical centroids still split, which is what stability buys.
///
/// With every centroid equal the sort has nothing to order, so the split rests
/// entirely on the stable sort keeping the input order. A non-stable sort would
/// still produce a valid tree here, so this case does not prove stability -- it
/// pins that the degenerate input does not hang or produce an empty side.
void identical_centroids_still_split() {
    const std::int64_t n = 8;
    std::vector<double> lo(static_cast<std::size_t>(n), -1.0);
    std::vector<double> hi(static_cast<std::size_t>(n), 1.0);
    const Tree t = build(n, 1, lo, hi);
    std::vector<std::int64_t> leaves;
    check_subtree(t, 0, leaves);
    PANTR_CHECK(static_cast<std::int64_t>(leaves.size()) == n);
}

}  // namespace

int main() {
    the_tree_is_a_partition_and_its_boxes_bound();
    the_query_agrees_with_a_brute_force_scan();
    a_shared_face_counts_as_overlap();
    identical_centroids_still_split();
    return pantr::test::summary("test_grid_bvh");
}
