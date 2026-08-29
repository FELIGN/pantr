#pragma once

/// \file
/// The bounding-volume hierarchy as a type, over the kernels in `bvh.hpp`.
///
/// Ports `src/pantr/grid/_bvh.py`, whose class stays as the Python backend under
/// `PANTR_BACKEND=python`. Ownership moves under `design/cross_backend_types.md`'s
/// 2026-08-27 amendment: there is one hierarchy, it is this one, and
/// `pantr.grid.BVH` wraps it. The kernels it calls are unchanged and stay where
/// they are; this header adds the validation, the invariants and the lifetime that
/// `_bvh.py` used to own.
///
/// ## The layout is a contract, and is ported unchanged
///
/// `design/bvh.md` establishes the five node arrays -- `node_lo`, `node_hi`,
/// `node_left`, `node_right`, `node_cell` -- as **public API rather than
/// implementation detail**, so this type reproduces them exactly: no leaf
/// clustering, no `int32` indices, no reordering. That note's improvement 4 is
/// marked "now or never" and stays closed until an API exists that frees a caller
/// from reading the arrays; a per-node `traverse(visitor)` callback is not that
/// API, because `design/user_functions_across_the_boundary.md` records a callback
/// across the boundary at about a microsecond per call, which is fatal in a
/// traversal loop.
///
/// ## The traversal stack is a limit of this build, not a defect in the argument
///
/// A tree deeper than `kBvhStackDepth` is perfectly well formed and this
/// implementation cannot walk it, so the constructor throws `pantr::CapacityError`
/// rather than `std::invalid_argument`; `pantr/core/error.hpp` argues why, and this
/// is the first and only site that throws it. The Python wrapper presents the
/// pre-port `ValueError` with the same text, because `PANTR_BACKEND` must not
/// decide which exception a caller catches -- see `src/pantr/grid/_bvh.py`.
///
/// The depth is established **once, at construction**, by an explicit walk with an
/// unbounded stack. It has to be unbounded: the depth is the unknown being
/// measured, so a fixed buffer would reproduce the very overflow the walk exists to
/// refuse. The walk stops as soon as it passes the limit, which is all the caller
/// needs and which also makes it terminate on a node array that is not a tree -- a
/// cyclic child pointer would otherwise grow the stack without bound, and a
/// validator that hangs is worse than the out-of-bounds write it replaces.
///
/// ## Zero-copy accessors, and what #359 becomes here
///
/// The five arrays are exposed as views into this object's own storage; the binding
/// hands them to numpy read-only, aliasing rather than copying, because the oracle
/// does the same and copying would make the backend switch a performance switch.
///
/// **This widens open issue #359 and the widening is recorded rather than closed.**
/// The exposure *shape* does not change: `_bvh.py` already returned its stored
/// array directly with `flags.writeable` cleared. What changes is the consequence
/// of defeating that flag, which `ctypes` does in two lines. Today a corrupted
/// child index makes the numba kernel return a defined wrong answer; here the
/// traversal indexes out of bounds, and `PANTR_PRECONDITION` is `assert`
/// (`pantr/core/precondition.hpp`), which `NDEBUG` removes. Per-node bounds
/// checking was rejected on a cost that `precondition.hpp` records as **measured**
/// against the traversal loop, not estimated. So a caller who clears the read-only
/// flag and writes into these arrays is outside the contract, and on this backend
/// that is undefined behaviour rather than a wrong answer.
///
/// Read-only here means read-only against accident, not against malice. That is
/// what the flag buys and it is worth saying plainly, because a docstring claiming
/// nothing can write into C++-owned memory would be false.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/core/error.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/bvh.hpp"

namespace pantr::grid {

/// A bounding-volume hierarchy indexing a fixed collection of cell AABBs.
///
/// Stored as the five parallel node arrays `bvh.hpp`'s kernels consume. The root
/// is node `0`; for `N >= 1` cells the tree has exactly `2N - 1` nodes, and for
/// `N == 0` it has none. Instances are immutable: nothing changes a tree once it is
/// built, and every accessor hands back a view of storage that cannot move.
template <Real T>
class BVH {
  public:
    /// Store the raw node arrays after validating them.
    ///
    /// Prefer `from_cell_bounds`; this constructor exists for a caller that has the
    /// arrays already -- a test poking a specific tree shape, or a round trip
    /// through serialization.
    ///
    /// \param node_lo Per-node AABB lo corners, shape `(n_nodes, ndim)`.
    /// \param node_hi Per-node AABB hi corners, same shape.
    /// \param node_left Left-child indices, `-1` on leaves, length `n_nodes`.
    /// \param node_right Right-child indices, same convention and length.
    /// \param node_cell Leaf cell ids, `-1` on internal nodes, same length.
    /// \param n_cells Number of indexed cells, which fixes `n_nodes`.
    /// \throws std::invalid_argument If the shapes disagree, if `ndim < 1`, if
    ///         `n_nodes != 2 * n_cells - 1`, if `node_cell` and the child pointers
    ///         disagree about which nodes are leaves, or if a child index or leaf
    ///         cell id is out of range.
    /// \throws pantr::CapacityError If the tree is deeper than the traversal stack
    ///         this build carries.
    BVH(span2d<const T> node_lo, span2d<const T> node_hi,
        std::span<const std::int64_t> node_left, std::span<const std::int64_t> node_right,
        std::span<const std::int64_t> node_cell, std::int64_t n_cells)
        : node_lo_(node_lo.data_handle(), node_lo.data_handle() + node_lo.size()),
          node_hi_(node_hi.data_handle(), node_hi.data_handle() + node_hi.size()),
          node_left_(node_left.begin(), node_left.end()),
          node_right_(node_right.begin(), node_right.end()),
          node_cell_(node_cell.begin(), node_cell.end()),
          n_nodes_(node_lo.extent(0)),
          ndim_(node_lo.extent(1)),
          n_cells_(n_cells) {
        if (node_hi.extent(0) != n_nodes_ || node_hi.extent(1) != ndim_) {
            throw std::invalid_argument("node_hi shape (" + std::to_string(node_hi.extent(0))
                                        + ", " + std::to_string(node_hi.extent(1))
                                        + ") must match node_lo shape ("
                                        + std::to_string(n_nodes_) + ", "
                                        + std::to_string(ndim_) + ").");
        }
        if (ndim_ < 1) {
            throw std::invalid_argument("BVH ndim must be >= 1; got " + std::to_string(ndim_)
                                        + ".");
        }
        for (const auto& [array, name] :
             {std::pair{&node_left_, "node_left"}, std::pair{&node_right_, "node_right"},
              std::pair{&node_cell_, "node_cell"}}) {
            if (array->size() != n_nodes_) {
                throw std::invalid_argument(std::string(name) + " must have shape ("
                                            + std::to_string(n_nodes_) + ",); got ("
                                            + std::to_string(array->size()) + ",).");
            }
        }
        const std::int64_t expected = n_cells > 0 ? 2 * n_cells - 1 : 0;
        if (static_cast<std::int64_t>(n_nodes_) != expected) {
            throw std::invalid_argument("BVH: n_cells=" + std::to_string(n_cells)
                                        + " implies n_nodes=" + std::to_string(expected)
                                        + "; got node arrays with " + std::to_string(n_nodes_)
                                        + " rows.");
        }
        // Runs before the depth walk below, which indexes the children itself.
        check_tree_structure();
        check_depth();
    }

    /// Build a hierarchy over `n_cells` cell AABBs, by median-of-longest-axis splits.
    ///
    /// Cells are sorted by centroid on the longest axis and split at the median, so
    /// the tree is balanced and each leaf indexes exactly one cell.
    ///
    /// \param cell_lo Per-cell lo corners, shape `(n_cells, ndim)` with `ndim >= 1`.
    /// \param cell_hi Per-cell hi corners, same shape, with `hi >= lo` everywhere.
    /// \return The hierarchy.
    /// \throws std::invalid_argument If the shapes disagree, if `ndim < 1`, if a
    ///         corner is not finite, or if some cell has `hi < lo`.
    /// \throws pantr::CapacityError If the cell count implies a tree deeper than the
    ///         traversal stack this build carries.
    [[nodiscard]] static BVH from_cell_bounds(span2d<const T> cell_lo, span2d<const T> cell_hi) {
        const std::size_t n_cells = cell_lo.extent(0);
        const std::size_t ndim = cell_lo.extent(1);
        if (cell_hi.extent(0) != n_cells || cell_hi.extent(1) != ndim) {
            throw std::invalid_argument("cell_hi shape (" + std::to_string(cell_hi.extent(0))
                                        + ", " + std::to_string(cell_hi.extent(1))
                                        + ") must match cell_lo shape ("
                                        + std::to_string(n_cells) + ", "
                                        + std::to_string(ndim) + ").");
        }
        if (ndim < 1) {
            throw std::invalid_argument("BVH ndim must be >= 1; got " + std::to_string(ndim)
                                        + ".");
        }
        for (std::size_t k = 0; k < n_cells * ndim; ++k) {
            if (!std::isfinite(value_of(cell_lo.data_handle()[k]))
                || !std::isfinite(value_of(cell_hi.data_handle()[k]))) {
                throw std::invalid_argument(
                    "BVH.from_cell_bounds: cell_lo and cell_hi must contain only finite "
                    "values; got NaN or Inf.");
            }
        }
        for (std::size_t k = 0; k < n_cells * ndim; ++k) {
            if (value_of(cell_hi.data_handle()[k]) < value_of(cell_lo.data_handle()[k])) {
                throw std::invalid_argument(
                    "Every cell must satisfy cell_hi >= cell_lo on every axis; at least one "
                    "cell violates this.");
            }
        }
        if (n_cells == 0) {
            return BVH(Unchecked{}, ndim);
        }
        // The splits are balanced, so the height follows from the cell count and
        // needs no tree walk. `bit_width(n - 1)` is `ceil(log2(n))` for n >= 2, in
        // exact integers; the `+ 1` is the root push. Going through a `double` is
        // the wrong arithmetic for a question about an integer, and past a
        // threshold returns a height one too SMALL, which is the unsafe direction
        // for a bound. scripts/measure_bvh_depth_arithmetic.py enumerates the
        // disagreements rather than this comment carrying figures nothing
        // re-derives.
        const std::int64_t max_depth =
            n_cells > 1
                ? static_cast<std::int64_t>(std::bit_width(static_cast<std::uint64_t>(n_cells - 1)))
                      + 1
                : 1;
        if (max_depth > kBvhStackDepth) {
            throw CapacityError("BVH.from_cell_bounds: " + std::to_string(n_cells)
                                + " cells would produce a tree of depth >= "
                                + std::to_string(max_depth)
                                + ", exceeding the internal stack depth "
                                + std::to_string(kBvhStackDepth)
                                + ". This is a library limit; please report this as an issue.");
        }

        const std::size_t n_nodes = 2 * n_cells - 1;
        BVH tree(Unchecked{}, ndim);
        tree.node_lo_.assign(n_nodes * ndim, T{0});
        tree.node_hi_.assign(n_nodes * ndim, T{0});
        tree.node_left_.assign(n_nodes, -1);
        tree.node_right_.assign(n_nodes, -1);
        tree.node_cell_.assign(n_nodes, -1);
        tree.n_nodes_ = n_nodes;
        tree.n_cells_ = static_cast<std::int64_t>(n_cells);
        bvh_build<T>(cell_lo, cell_hi, span2d<T>(tree.node_lo_.data(), n_nodes, ndim),
                     span2d<T>(tree.node_hi_.data(), n_nodes, ndim), tree.node_left_,
                     tree.node_right_, tree.node_cell_);
        return tree;
    }

    /// The spatial dimension.
    ///
    /// \return The number of axes, `>= 1`.
    [[nodiscard]] std::size_t ndim() const noexcept { return ndim_; }

    /// The number of indexed cells, equal to the number of leaves.
    ///
    /// \return The cell count.
    [[nodiscard]] std::int64_t n_cells() const noexcept { return n_cells_; }

    /// The total number of nodes.
    ///
    /// \return `2 * n_cells - 1`, or `0` when there are no cells.
    [[nodiscard]] std::size_t n_nodes() const noexcept { return n_nodes_; }

    /// The per-node AABB lower corners.
    ///
    /// \return A `(n_nodes, ndim)` view, valid while the tree lives.
    [[nodiscard]] span2d<const T> node_lo() const noexcept {
        return span2d<const T>(node_lo_.data(), n_nodes_, ndim_);
    }

    /// The per-node AABB upper corners.
    ///
    /// \return A `(n_nodes, ndim)` view, valid while the tree lives.
    [[nodiscard]] span2d<const T> node_hi() const noexcept {
        return span2d<const T>(node_hi_.data(), n_nodes_, ndim_);
    }

    /// The per-node left-child indices.
    ///
    /// \return A length-`n_nodes` view; `-1` on a leaf.
    [[nodiscard]] std::span<const std::int64_t> node_left() const noexcept { return node_left_; }

    /// The per-node right-child indices.
    ///
    /// \return A length-`n_nodes` view; `-1` on a leaf.
    [[nodiscard]] std::span<const std::int64_t> node_right() const noexcept {
        return node_right_;
    }

    /// The per-leaf cell ids.
    ///
    /// \return A length-`n_nodes` view; `-1` on an internal node.
    [[nodiscard]] std::span<const std::int64_t> node_cell() const noexcept { return node_cell_; }

    /// The ids of every leaf cell whose AABB is not separated from the query box.
    ///
    /// Named for the query it answers rather than for the geometry it suggests:
    /// `bvh.hpp`'s predicate is a separating-axis test with no emptiness branch, so
    /// a **reversed** query interval is reported against any cell whose own
    /// interval contains it, where `pantr::geometry::AABB::overlaps` would report
    /// nothing. That divergence is reproduced deliberately and is argued in
    /// `bvh.hpp`'s file comment.
    ///
    /// The ids come back in the traversal's own preorder, which is right to left;
    /// a caller wanting another order sorts.
    ///
    /// \param qlo Query box lo corner, length `ndim`.
    /// \param qhi Query box hi corner, same length.
    /// \return The matching cell ids.
    /// \throws std::invalid_argument If either corner has the wrong length.
    /// \throws std::runtime_error If the count and emit passes disagree, which is a
    ///         defect in the kernels rather than in the argument.
    [[nodiscard]] std::vector<std::int64_t> query_aabb(std::span<const T> qlo,
                                                       std::span<const T> qhi) const {
        if (qlo.size() != ndim_ || qhi.size() != ndim_) {
            throw std::invalid_argument("BVH.query_aabb: aabb.ndim ("
                                        + std::to_string(qlo.size())
                                        + ") must match self.ndim (" + std::to_string(ndim_)
                                        + ").");
        }
        if (n_cells_ == 0) {
            return {};
        }
        const std::int64_t count = bvh_query_count<T>(qlo, qhi, node_lo(), node_hi(), node_left_,
                                                      node_right_, node_cell_);
        std::vector<std::int64_t> out(static_cast<std::size_t>(count));
        if (count == 0) {
            return out;
        }
        const std::int64_t written = bvh_query_emit<T>(qlo, qhi, node_lo(), node_hi(), node_left_,
                                                       node_right_, node_cell_, out);
        if (written != count) {
            throw std::runtime_error(
                "BVH.query_aabb: internal count/emit mismatch (count pass returned "
                + std::to_string(count) + ", emit pass wrote " + std::to_string(written)
                + "). This is a bug in the BVH kernel; please report it.");
        }
        return out;
    }

  private:
    /// Tag for the constructor that builds an empty shell for a factory to fill.
    struct Unchecked {};

    /// Build a zero-node tree of the given dimension.
    ///
    /// \param ndim Spatial dimension.
    BVH(Unchecked, std::size_t ndim) : n_nodes_(0), ndim_(ndim), n_cells_(0) {}

    /// Reject node arrays the traversal kernels cannot walk.
    ///
    /// The kernels decide "leaf or internal" from `node_cell` alone and then push
    /// both children without testing either against `-1`, so the two encodings of
    /// leafness have to agree and an internal node's children have to be real
    /// indices. A child left at `-1`, or any other negative value, indexes out of
    /// bounds rather than raising.
    ///
    /// \throws std::invalid_argument If the encodings disagree or an index is out
    ///         of range.
    void check_tree_structure() const {
        for (std::size_t node = 0; node < n_nodes_; ++node) {
            const bool is_leaf = node_cell_[node] >= 0;
            const bool no_children = node_left_[node] == -1 && node_right_[node] == -1;
            if (is_leaf != no_children) {
                throw std::invalid_argument(
                    "BVH: node_cell and the child pointers disagree about which nodes are "
                    "leaves. A node is a leaf iff node_cell >= 0, and exactly then must "
                    "node_left and node_right both be -1.");
            }
        }
        check_child_range(node_left_, "node_left");
        check_child_range(node_right_, "node_right");

        std::int64_t max_cell = -1;
        for (std::size_t node = 0; node < n_nodes_; ++node) {
            if (node_cell_[node] >= 0) {
                max_cell = std::max(max_cell, node_cell_[node]);
            }
        }
        if (max_cell >= n_cells_) {
            throw std::invalid_argument("BVH: node_cell contains values outside [0, "
                                        + std::to_string(n_cells_)
                                        + ") on leaves: maximum is " + std::to_string(max_cell)
                                        + ".");
        }
    }

    /// Reject an internal node's child index outside `[0, n_nodes)`.
    ///
    /// Reports the range over internal nodes only, matching the oracle: a leaf's
    /// `-1` is not a violation.
    ///
    /// \param children The child array to check.
    /// \param name Its name, for the message.
    /// \throws std::invalid_argument If some internal node's child is out of range.
    void check_child_range(const std::vector<std::int64_t>& children, const char* name) const {
        bool any = false;
        std::int64_t lo = 0;
        std::int64_t hi = 0;
        for (std::size_t node = 0; node < n_nodes_; ++node) {
            if (node_cell_[node] >= 0) {
                continue;
            }
            const std::int64_t child = children[node];
            lo = any ? std::min(lo, child) : child;
            hi = any ? std::max(hi, child) : child;
            any = true;
        }
        if (any && (lo < 0 || hi >= static_cast<std::int64_t>(n_nodes_))) {
            throw std::invalid_argument(std::string(name) + " contains values outside [0, "
                                        + std::to_string(n_nodes_)
                                        + ") on internal nodes: range is [" + std::to_string(lo)
                                        + ", " + std::to_string(hi) + "].");
        }
    }

    /// Reject a tree deeper than the traversal stack this build carries.
    ///
    /// Walks from the root with an unbounded stack -- the depth is the unknown being
    /// measured, so a fixed buffer would reproduce the overflow this exists to
    /// refuse -- and stops as soon as the limit is passed, which also makes it
    /// terminate on a cyclic child pointer.
    ///
    /// \throws pantr::CapacityError If the deepest root-to-leaf path does not fit.
    void check_depth() const {
        if (n_nodes_ == 0) {
            return;
        }
        std::vector<std::pair<std::int64_t, std::int64_t>> stack{{0, 1}};
        std::int64_t max_depth = 0;
        while (!stack.empty()) {
            const auto [node, depth] = stack.back();
            stack.pop_back();
            max_depth = std::max(max_depth, depth);
            if (max_depth > kBvhStackDepth) {
                throw CapacityError(
                    "BVH: the given node arrays encode a tree of depth "
                    + std::to_string(max_depth) + " or more, exceeding the stack depth "
                    + std::to_string(kBvhStackDepth)
                    + " that the traversal kernels allow. Build the tree more evenly, or use "
                      "BVH.from_cell_bounds, whose median split keeps the depth logarithmic in "
                      "the cell count.");
            }
            const std::int64_t left = node_left_[static_cast<std::size_t>(node)];
            const std::int64_t right = node_right_[static_cast<std::size_t>(node)];
            if (left != -1) {
                stack.emplace_back(left, depth + 1);
            }
            if (right != -1) {
                stack.emplace_back(right, depth + 1);
            }
        }
    }

    std::vector<T> node_lo_;               ///< Per-node lo corners, row-major.
    std::vector<T> node_hi_;               ///< Per-node hi corners, row-major.
    std::vector<std::int64_t> node_left_;  ///< Left-child indices, `-1` on a leaf.
    std::vector<std::int64_t> node_right_; ///< Right-child indices, `-1` on a leaf.
    std::vector<std::int64_t> node_cell_;  ///< Leaf cell ids, `-1` on an internal node.
    std::size_t n_nodes_;                  ///< Node count.
    std::size_t ndim_;                     ///< Spatial dimension.
    std::int64_t n_cells_;                 ///< Indexed cell count.
};

}  // namespace pantr::grid
