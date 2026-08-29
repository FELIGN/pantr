/// \file
/// Properties of the bounding-volume hierarchy as a type.
///
/// `test_grid_bvh.cpp` covers the kernels. This file covers what the type adds: the
/// validation, the two boundaries of the traversal-stack limit, and the empty tree.
///
/// ## Why these cases and not others
///
///  - **The depth limit, from both sides.** A depth-`d` descent occupies at most
///    `d` stack slots -- the root takes one, then each of the `d - 1` internal
///    expansions along the deepest path pops one and pushes two, a net `+1` -- so
///    `d == kBvhStackDepth` fits exactly and `d + 1` does not. Both are asserted,
///    because a guard that refuses everything passes the refusal half on its own.
///  - **`CapacityError` rather than `std::invalid_argument`.** The tree is well
///    formed and this build cannot walk it, which is what `pantr/core/error.hpp`
///    reserves that type for. Asserted by catching the specific type, since a
///    `catch (const std::invalid_argument&)` and a `catch (const std::exception&)`
///    would both be satisfied by the wrong one.
///  - **A node array that is not a tree.** A cyclic child pointer must be refused
///    rather than walked forever: a validator that hangs is worse than the
///    out-of-bounds write it replaces.
///  - **The structural checks.** The kernels decide leafness from `node_cell` alone
///    and then push both children unconditionally, so a `-1` child on a node that
///    does not look like a leaf indexes out of bounds. That arrangement constructed
///    cleanly and returned the wrong cells before the check existed.
///  - **The empty tree.** Zero cells is legal, `n_nodes` is 0, and the query must
///    short-circuit rather than read node 0.

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/core/error.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/bvh_tree.hpp"

namespace {

using pantr::grid::BVH;
using pantr::grid::kBvhStackDepth;

/// The five node arrays of a hand-built tree, kept together so a test can poke one.
struct Arrays {
    std::vector<double> node_lo;         ///< Per-node lo corners, row-major.
    std::vector<double> node_hi;         ///< Per-node hi corners, row-major.
    std::vector<std::int64_t> node_left;  ///< Left-child indices.
    std::vector<std::int64_t> node_right; ///< Right-child indices.
    std::vector<std::int64_t> node_cell;  ///< Leaf cell ids.
    std::size_t ndim = 1;                 ///< Spatial dimension.
};

/// Build a tree from an `Arrays`, for brevity below.
///
/// \param a The node arrays.
/// \param n_cells The indexed cell count.
/// \return The hierarchy.
BVH<double> build(const Arrays& a, std::int64_t n_cells) {
    const std::size_t rows = a.node_left.size();
    return BVH<double>(pantr::span2d<const double>(a.node_lo.data(), rows, a.ndim),
                       pantr::span2d<const double>(a.node_hi.data(), rows, a.ndim), a.node_left,
                       a.node_right, a.node_cell, n_cells);
}

/// Node arrays for a maximally unbalanced (linear-chain) 1-D tree.
///
/// Leaf `k` covers the unit cell `[k, k+1]`. Node `i` for `i < n_leaves - 1` is
/// internal, with left child leaf `i` and right child the next internal node. This
/// is the deepest tree over `n_leaves` leaves: the longest root-to-leaf path has
/// exactly `n_leaves` nodes. Mirrors `_chain_bvh_arrays` in `tests/test_grid_bvh.py`.
///
/// \param n_leaves Number of leaves, at least 2.
/// \return The node arrays.
Arrays chain(std::int64_t n_leaves) {
    const auto n_internal = static_cast<std::size_t>(n_leaves - 1);
    const auto n_nodes = static_cast<std::size_t>(2 * n_leaves - 1);
    Arrays a;
    a.node_lo.assign(n_nodes, 0.0);
    a.node_hi.assign(n_nodes, 0.0);
    a.node_left.assign(n_nodes, -1);
    a.node_right.assign(n_nodes, -1);
    a.node_cell.assign(n_nodes, -1);
    for (std::size_t i = 0; i < n_internal; ++i) {
        a.node_lo[i] = static_cast<double>(i);
        a.node_hi[i] = static_cast<double>(n_leaves);
        a.node_left[i] = static_cast<std::int64_t>(n_internal + i);
        a.node_right[i] = i < n_internal - 1 ? static_cast<std::int64_t>(i + 1)
                                             : static_cast<std::int64_t>(n_internal + i + 1);
    }
    for (std::size_t k = 0; k < static_cast<std::size_t>(n_leaves); ++k) {
        const std::size_t leaf = n_internal + k;
        a.node_lo[leaf] = static_cast<double>(k);
        a.node_hi[leaf] = static_cast<double>(k + 1);
        a.node_cell[leaf] = static_cast<std::int64_t>(k);
    }
    return a;
}

/// Per-cell bounds for `n` unit cells laid end to end on one axis.
///
/// \param n Number of cells.
/// \return `(lo, hi)` flattened, one column.
std::pair<std::vector<double>, std::vector<double>> unit_cells(std::size_t n) {
    std::vector<double> lo(n);
    std::vector<double> hi(n);
    for (std::size_t k = 0; k < n; ++k) {
        lo[k] = static_cast<double>(k);
        hi[k] = static_cast<double>(k) + 1.0;
    }
    return {lo, hi};
}

/// Whether calling `fn` throws `std::invalid_argument`.
///
/// \tparam Fn The callable's type.
/// \param fn The call to attempt.
/// \return `true` when it threw.
template <class Fn>
bool rejects(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

/// Whether calling `fn` throws `pantr::CapacityError` specifically.
///
/// The specificity is the assertion: `CapacityError` derives from
/// `std::runtime_error`, so catching the base would pass for either.
///
/// \tparam Fn The callable's type.
/// \param fn The call to attempt.
/// \return `true` when it threw a `CapacityError`.
template <class Fn>
bool exceeds_capacity(Fn&& fn) {
    try {
        fn();
    } catch (const pantr::CapacityError&) {
        return true;
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

/// `from_cell_bounds` produces the documented shape and answers a query.
void test_from_cell_bounds_shape() {
    const auto [lo, hi] = unit_cells(4);
    const BVH<double> tree = BVH<double>::from_cell_bounds(
        pantr::span2d<const double>(lo.data(), 4, 1),
        pantr::span2d<const double>(hi.data(), 4, 1));
    PANTR_CHECK(tree.n_cells() == 4);
    PANTR_CHECK(tree.n_nodes() == 7);
    PANTR_CHECK(tree.ndim() == 1);

    const std::vector<double> qlo{1.5};
    const std::vector<double> qhi{2.5};
    const auto got = tree.query_aabb(qlo, qhi);
    PANTR_CHECK(got.size() == 2);
}

/// `from_cell_bounds` refuses a non-finite corner, an inverted cell, a bad shape.
void test_from_cell_bounds_validates() {
    const std::vector<double> lo{0.0};
    const std::vector<double> inverted{-1.0};
    const std::vector<double> nan_hi{std::numeric_limits<double>::quiet_NaN()};
    const std::vector<double> inf_hi{std::numeric_limits<double>::infinity()};
    const auto one = [&](const std::vector<double>& hi) {
        return BVH<double>::from_cell_bounds(pantr::span2d<const double>(lo.data(), 1, 1),
                                             pantr::span2d<const double>(hi.data(), 1, 1));
    };
    PANTR_CHECK(rejects([&] { return one(nan_hi); }));
    PANTR_CHECK(rejects([&] { return one(inf_hi); }));
    PANTR_CHECK(rejects([&] { return one(inverted); }));
    PANTR_CHECK(rejects([&] {
        return BVH<double>::from_cell_bounds(pantr::span2d<const double>(lo.data(), 1, 1),
                                             pantr::span2d<const double>(lo.data(), 1, 0));
    }));
}

/// A zero-cell tree is legal, has no nodes, and its query short-circuits.
void test_the_empty_tree() {
    const std::vector<double> none{};
    const BVH<double> tree = BVH<double>::from_cell_bounds(
        pantr::span2d<const double>(none.data(), 0, 3),
        pantr::span2d<const double>(none.data(), 0, 3));
    PANTR_CHECK(tree.n_cells() == 0);
    PANTR_CHECK(tree.n_nodes() == 0);
    PANTR_CHECK(tree.ndim() == 3);

    const std::vector<double> qlo{0.0, 0.0, 0.0};
    const std::vector<double> qhi{1.0, 1.0, 1.0};
    PANTR_CHECK(tree.query_aabb(qlo, qhi).empty());
}

/// The node-count relation is enforced, and so is the dimension.
void test_node_count_and_dimension() {
    Arrays a = chain(3);
    PANTR_CHECK(rejects([&] { return build(a, 2); }));
    PANTR_CHECK(build(a, 3).n_nodes() == 5);

    Arrays zero_dim = chain(2);
    zero_dim.ndim = 0;
    zero_dim.node_lo.clear();
    zero_dim.node_hi.clear();
    PANTR_CHECK(rejects([&] { return build(zero_dim, 2); }));
}

/// The structural checks reject each of the three ways the encodings can disagree.
void test_structure_is_validated() {
    Arrays missing_child = chain(3);
    const std::size_t internal = 0;
    missing_child.node_left[internal] = -1;
    PANTR_CHECK(rejects([&] { return build(missing_child, 3); }));

    Arrays leaf_marked_internal = chain(3);
    leaf_marked_internal.node_cell[2] = -1;
    PANTR_CHECK(rejects([&] { return build(leaf_marked_internal, 3); }));

    Arrays bad_child = chain(3);
    bad_child.node_right[internal] = -5;
    PANTR_CHECK(rejects([&] { return build(bad_child, 3); }));

    Arrays bad_cell = chain(3);
    bad_cell.node_cell[2] = 42;
    PANTR_CHECK(rejects([&] { return build(bad_cell, 3); }));
}

/// The depth limit is exact: `kBvhStackDepth` fits and one more does not.
///
/// The refusal is `CapacityError` rather than `std::invalid_argument`, because the
/// arrays describe a perfectly well formed tree that this build cannot walk.
void test_the_depth_limit_is_exact_and_reports_capacity() {
    const Arrays at_limit = chain(kBvhStackDepth);
    const BVH<double> tree = build(at_limit, kBvhStackDepth);
    const std::vector<double> qlo{50.5};
    const std::vector<double> qhi{52.5};
    PANTR_CHECK(tree.query_aabb(qlo, qhi).size() == 3);

    // `exceeds_capacity` returns false for any other exception, so this asserts the
    // specific type and not merely that something was thrown.
    const Arrays past_limit = chain(kBvhStackDepth + 1);
    PANTR_CHECK(exceeds_capacity([&] { return build(past_limit, kBvhStackDepth + 1); }));
}

/// A cyclic child pointer terminates the depth walk rather than hanging it.
void test_a_cyclic_node_graph_is_refused_without_hanging() {
    Arrays cyclic = chain(4);
    cyclic.node_right[0] = 0;  // the root points back at itself
    PANTR_CHECK(exceeds_capacity([&] { return build(cyclic, 4); }));
}

/// The five arrays are readable, and the preorder layout is exactly the declared one.
///
/// **Four cells rather than three, and the count is the assertion.** At three cells
/// the two children of the root claim indices 1 and 2 before either is expanded, and
/// only one of them splits further, so *reversing the build's push order leaves every
/// index unchanged* -- a mutation that flips the convention passes a three-cell
/// check. At four both children split, so the order in which they are expanded
/// decides whether the left subtree's children are nodes 3 and 4 or nodes 5 and 6.
/// The preorder determinism the oracle's module docstring declares is that choice,
/// and this is where it is pinned on this side.
void test_the_node_arrays_are_readable() {
    const auto [lo, hi] = unit_cells(4);
    const BVH<double> tree = BVH<double>::from_cell_bounds(
        pantr::span2d<const double>(lo.data(), 4, 1),
        pantr::span2d<const double>(hi.data(), 4, 1));
    PANTR_CHECK(tree.n_nodes() == 7);
    PANTR_CHECK(tree.node_left().size() == tree.n_nodes());
    PANTR_CHECK(tree.node_right().size() == tree.n_nodes());
    PANTR_CHECK(tree.node_cell().size() == tree.n_nodes());
    PANTR_CHECK(tree.node_lo().extent(0) == tree.n_nodes());
    PANTR_CHECK(tree.node_lo().extent(1) == 1);

    PANTR_CHECK(tree.node_left()[0] == 1 && tree.node_right()[0] == 2);
    PANTR_CHECK(tree.node_cell()[0] == -1);
    // Left to right: the root's LEFT child is expanded first, so it takes the next
    // two indices and the right child takes the two after them.
    PANTR_CHECK_MSG(tree.node_left()[1] == 3 && tree.node_right()[1] == 4,
                    "the build must expand the left subtree first");
    PANTR_CHECK_MSG(tree.node_left()[2] == 5 && tree.node_right()[2] == 6,
                    "the build must expand the left subtree first");
    PANTR_CHECK(tree.node_cell()[3] == 0 && tree.node_cell()[4] == 1);
    PANTR_CHECK(tree.node_cell()[5] == 2 && tree.node_cell()[6] == 3);
}

/// A query whose corners have the wrong length is refused.
void test_query_validates_its_dimension() {
    const auto [lo, hi] = unit_cells(2);
    const BVH<double> tree = BVH<double>::from_cell_bounds(
        pantr::span2d<const double>(lo.data(), 2, 1),
        pantr::span2d<const double>(hi.data(), 2, 1));
    const std::vector<double> wide_lo{0.0, 0.0};
    const std::vector<double> wide_hi{1.0, 1.0};
    PANTR_CHECK(rejects([&] { return tree.query_aabb(wide_lo, wide_hi); }));
}

/// An empty query box is reported against a cell that contains its reversed interval.
///
/// This is the divergence from `pantr::geometry::AABB::overlaps` that `bvh.hpp`'s
/// file comment records. It is pinned here rather than left implicit, so that
/// reconciling the two predicates has to change a test that says why.
void test_a_reversed_query_interval_is_reported() {
    const std::vector<double> lo{0.0};
    const std::vector<double> hi{10.0};
    const BVH<double> tree = BVH<double>::from_cell_bounds(
        pantr::span2d<const double>(lo.data(), 1, 1),
        pantr::span2d<const double>(hi.data(), 1, 1));
    const std::vector<double> qlo{5.0};
    const std::vector<double> qhi{3.0};
    PANTR_CHECK_MSG(tree.query_aabb(qlo, qhi) == std::vector<std::int64_t>({0}),
                    "the separating-axis predicate has no emptiness branch");
}

}  // namespace

// The class is a template, so its member bodies are only instantiated where they
// are used; naming both instantiations compiles every one of them, which the calls
// above do not on their own.
template class pantr::grid::BVH<double>;
template class pantr::grid::BVH<float>;

int main() {
    test_from_cell_bounds_shape();
    test_from_cell_bounds_validates();
    test_the_empty_tree();
    test_node_count_and_dimension();
    test_structure_is_validated();
    test_the_depth_limit_is_exact_and_reports_capacity();
    test_a_cyclic_node_graph_is_refused_without_hanging();
    test_the_node_arrays_are_readable();
    test_query_validates_its_dimension();
    test_a_reversed_query_interval_is_reported();
    return pantr::test::summary("test_grid_bvh_type");
}
