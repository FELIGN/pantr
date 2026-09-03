#pragma once

/// \file
/// The hierarchical grid type: a root tensor-product grid plus a per-level set of
/// active-leaf rectangles.
///
/// Ports `src/pantr/grid/_hierarchical_grid.py`, which stays as the parity oracle:
/// the query half, then refinement, coarsening, restriction and the leaf-cell mesh
/// export. Only the bindings and the Python wrapper are still absent.
///
/// ## Every operation returns a new grid, and that is what removes the cache problem
///
/// `refine`, `coarsen` and their by-id forms build a fresh grid and leave the receiver
/// untouched, so nothing here ever invalidates a cache: the result's BVH memo and its
/// two tag registries start empty, and the receiver's stay valid because its active set
/// never moved. That is why this type needs no protected `invalidate_caches()` on the
/// mixin -- see the section below on what the ticket asked for and why it is obsolete.
///
/// The paths that change nothing -- `refine` over a region with no active cell,
/// `refine_cells` with no ids, `coarsen_cells` demoting no complete family -- go through
/// `rebuilt()` rather than the copy constructor, so a result never aliases its receiver
/// and never inherits its tags either.
///
/// ## One representation, not two
///
/// The oracle keeps its active set twice: `_blocks`, a list of lists of tuples, and four
/// packed `int64` arrays rebuilt from it on every construction, because its Numba
/// kernels cannot take the first shape. There is no such constraint here, so this type
/// stores **only** the packed form -- `block_lo_`, `block_hi_`, `block_base_`,
/// `level_start_` -- and `active_blocks()` is a view into it rather than a copy.
///
/// That is sound because the two are the same object in a different spelling: blocks are
/// packed level by level and, within a level, in the order the level's list holds them,
/// which is the order flat cell ids are assigned in. Dropping the unpacked copy removes
/// the only way the two could ever disagree.
///
/// The root's breakpoints are not copied either. `TensorProductGrid` already stores them
/// in one flat buffer with a per-axis offset table, which is exactly the `knots_flat` /
/// `knot_starts` pair the kernels in `pantr/grid/hierarchical.hpp` take, so this type
/// borrows them. The oracle repacks them on every rebuild and says in a comment that it
/// does so only to keep one construction path.
///
/// ## Flat cell ids, which are the observable this whole file has to get right
///
/// > Ids are assigned level by level from level 0, block by block within a level, and in
/// > C-order (last axis fastest) within a block.
///
/// Two consequences that are easy to lose. **The partition is observable**: which
/// rectangles a level is cut into decides which cell gets which number, so
/// `normalize_blocks`' order-dependent merge is part of the contract rather than an
/// implementation detail -- `pantr/grid/blocks.hpp` carries that argument. And **ids are
/// not stable across a refinement**: every construction assigns them from scratch, so a
/// caller tracking a cell keeps the `(level, midx)` pair, which is stable, and resolves
/// it with `cell_id()`.
///
/// ## The floating-point discipline, in one line each
///
/// A cell's bounds are `root_lo + sub_index * size`, where `size` is the root cell's
/// width divided by `factor ** level`. **The product is named rather than inlined**, at
/// every site, for the reason `pantr/grid/hierarchical.hpp` sets out at length: written
/// as one expression it is contractible into a fused multiply-add, `-ffp-contract=on`
/// permits exactly that, the oracle never fuses, and in `locate`'s descent the result
/// feeds a truncation -- so the two backends would disagree on a **cell id**, which
/// `design/backend_parity.md` Rule 11 says no tolerance bounds. `factor ** level` is
/// accumulated by repeated integer multiplication, matching the oracle, so the two
/// cannot diverge through a floating-point exponentiation neither should be doing.
///
/// ## Where this diverges from the oracle deliberately, and why
///
/// **Counts that do not fit are refused rather than wrapped.** The oracle counts cells in
/// Python's arbitrary-precision integers and cannot overflow; this type accumulates in
/// `std::int64_t` and can, so the total cell count, each block's own product and
/// `factor ** level` are all checked and throw. Same trade `TensorProductGrid` already
/// makes for its own cell-count product: a grid that large is unusable on either side,
/// and raising beats a positive but meaningless count.
///
/// **The `invalidate_caches()` the ticket asked for is obsolete, and is deliberately not
/// here.** Its premise was that `_rebuild` writes `self._bvh`, `self._cell_tags` and
/// `self._facet_tags`, which are the base's private slots. That stopped being true when
/// refinement was made to return a new grid: `grep` for all three names in
/// `src/pantr/grid/_hierarchical_grid.py` now finds none, and `_rebuild`'s own docstring
/// says a grid is frozen once it returns. `design/grid_hierarchy_port.md` predicted this
/// and advised against building the mutator. A protected mutator nothing calls is the
/// kind of seam that becomes load-bearing by accident.
///
/// **A wrong-length `midx` is an error here and a `None` there.** `cell_id` and
/// `is_active_leaf` throw `std::invalid_argument` where the oracle's `cell_id`
/// (`_hierarchical_grid.py:1211`) folds a wrong-length index into its "not an active
/// leaf" answer. That is the port's stated rule -- `pantr/core/error.hpp` puts value and
/// range checks in C++ and leaves shape coercion to the Python wrapper -- but it is a
/// real behavioural difference and not only a message one, so **the wrapper owes the
/// `None`**: it must catch the length case before forwarding, or a parity test feeding a
/// badly sized index sees the two backends disagree in kind.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/blocks.hpp"
#include "pantr/grid/grid.hpp"
#include "pantr/grid/hierarchical.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

namespace pantr::grid {

template <Real T>
class HierarchicalGrid;

}  // namespace pantr::grid

namespace pantr::grid {

/// Cells named per reason when `HierarchicalGrid::coarsen` refuses a region.
///
/// Enough to show the pattern of the offending set while keeping the message readable
/// when a large region is rejected; the remainder is reported as a count.
inline constexpr std::int64_t kMaxNamedCells = 6;

/// Region size above which `HierarchicalGrid::coarsen`'s refusal skips naming cells.
///
/// The diagnostic materialises three region-shaped masks, so the cap is a memory budget.
/// A region may legitimately be far larger than the grid is -- level `l` has
/// `factor ** l` cells per root cell whether or not they exist -- and enumerating cells
/// that mostly do not exist would help nobody, so past the cap the message reports the
/// region's extent instead.
inline constexpr std::int64_t kMaxDiagnosedCells = 1 << 20;

/// The dimension at which `2 ** ndim` corners per cell stops fitting in `std::int64_t`.
///
/// Reachable in principle rather than in practice: a 63-axis grid may hold a single cell,
/// and its corner count would still overflow. `export_cells` refuses at or above this
/// rather than shifting past the width of the type.
inline constexpr std::int64_t kMaxExportNdim = 63;

/// A leaf-cell mesh: deduplicated vertices and one connectivity row per active leaf.
///
/// An aggregate rather than a class for the same reason `GridRestriction` is one: it
/// carries three buffers and no invariant beyond the shapes its producer documents.
///
/// \tparam T The coordinate type.
template <Real T>
struct CellExport {
    std::vector<T> points;           ///< `(num_vertices, ndim)` row-major coordinates.
    std::vector<std::int64_t> conn;  ///< `(num_cells, 2 ** ndim)` row-major, into `points`.
    std::int64_t num_vertices;       ///< Row count of `points`.
};

}  // namespace pantr::grid

template <pantr::Real T>
struct pantr::grid::grid_traits<pantr::grid::HierarchicalGrid<T>> {
    /// The coordinate type.
    using scalar_type = T;

    /// The six defaults this grid replaces.
    ///
    /// `cell_level` because a hierarchy has real levels where the generic answer is
    /// always zero; the next four because the packed descriptor answers them without a
    /// per-cell or per-facet query; and `restrict` because restriction is an optional
    /// capability that a hierarchy has, so the mixin's throwing default no longer stands.
    static constexpr pantr::grid::Hook hooks =
        pantr::grid::Hook::cell_level | pantr::grid::Hook::boundary_facets
        | pantr::grid::Hook::hanging_neighbors | pantr::grid::Hook::locate_many
        | pantr::grid::Hook::collect_cell_bounds | pantr::grid::Hook::restrict;
};

namespace pantr::grid {

/// A hierarchy of nested tensor-product grids, with the active leaves stored as blocks.
///
/// Level 0 is `root`; a level-`l` cell subdivides into `factor[k]` children along axis
/// `k`. The active leaves at each level are held as sorted, non-overlapping,
/// pairwise-non-mergeable integer rectangles in that level's own coordinates, and they
/// partition the root's domain exactly.
///
/// Immutable after construction apart from the base's lazily built spatial index and its
/// two tag registries.
template <Real T>
class HierarchicalGrid : public GridBase<HierarchicalGrid<T>> {
  public:
    /// The mixin, named publicly so a differential test can reach the hidden defaults.
    using Base = GridBase<HierarchicalGrid<T>>;

    /// Build an unrefined hierarchy: one level, one block spanning the whole root.
    ///
    /// \param root The level-0 grid.
    /// \param factor Per-axis subdivision factor, length `root.ndim()`. Each entry must
    ///        be `>= 1`; `1` on an axis prevents subdivision along it.
    /// \throws std::invalid_argument If `factor` has the wrong length or any entry
    ///         is `< 1`.
    HierarchicalGrid(TensorProductGrid<T> root, std::span<const std::int64_t> factor)
        : HierarchicalGrid(build_unrefined(std::move(root), factor)) {}

    /// Build from per-level block lists: the constructor every non-default path uses.
    ///
    /// \param root The level-0 root grid of the hierarchy.
    /// \param factor Per-axis subdivision factor.
    /// \param blocks `blocks[l]` holds the active-leaf rectangles at level `l`, in
    ///        level-`l` coordinates. Trailing empty levels are dropped.
    /// \param unnormalized_levels The levels whose lists the caller may have altered and
    ///        which must therefore be normalized here. Every other level is taken
    ///        verbatim. An empty optional normalizes every level and is always safe.
    /// \return The grid.
    /// \throws std::invalid_argument If `factor` is malformed, a level's rectangle count
    ///         disagrees with `root.ndim()`, or the total cell count overflows.
    /// \note No check that `blocks` is a valid active-leaf decomposition. Callers owe
    ///       non-overlapping rectangles at each level whose per-level sets collectively
    ///       partition the root's cells consistently with `factor`. A violation is a
    ///       wrong answer here and in the oracle alike, not undefined behaviour.
    /// \warning `unnormalized_levels` is a **cost** switch and never a behavioural one,
    ///          and naming too few levels is silent. Normalization is a fixed point, so
    ///          a level holding another grid's already-normalized list normalizes to
    ///          itself and skipping that re-run returns the same blocks in the same
    ///          order. Declaring a level clean when it is not moves every flat cell id.
    [[nodiscard]] static HierarchicalGrid from_blocks(
        TensorProductGrid<T> root, std::span<const std::int64_t> factor,
        std::vector<BlockList> blocks,
        std::optional<std::span<const std::int64_t>> unnormalized_levels) {
        return HierarchicalGrid(build(std::move(root), factor, std::move(blocks),
                                      unnormalized_levels));
    }

    // ----------------------------------------------------------------
    // This grid's own surface
    // ----------------------------------------------------------------

    /// The level-0 grid.
    ///
    /// \return A reference to a member of this grid, valid for as long as the grid is.
    [[nodiscard]] const TensorProductGrid<T>& root() const noexcept { return root_; }

    /// The per-axis subdivision factor.
    ///
    /// \return A view of length `ndim()`; every entry is `>= 1`.
    [[nodiscard]] std::span<const std::int64_t> factor() const noexcept { return factor_; }

    /// The index of the deepest level, `0` before any refinement.
    ///
    /// \return `n_levels - 1`.
    [[nodiscard]] std::int64_t max_level() const noexcept {
        return static_cast<std::int64_t>(level_start_.size()) - 2;
    }

    /// The number of cells along one axis of the level-`level` grid.
    ///
    /// A pure formula, so `level` need not exist: a value above `max_level()` gives the
    /// count for the hypothetical finer grid further uniform subdivision would produce.
    /// That is deliberately unlike `active_blocks`, which requires an existing level.
    ///
    /// \param level Hierarchy level; must be `>= 0`.
    /// \param axis Axis index in `[0, ndim())`.
    /// \return `root.cells_per_axis()[axis] * factor()[axis] ** level`.
    /// \throws std::invalid_argument If `level < 0`.
    /// \throws std::out_of_range If `axis` is out of range.
    /// \throws std::overflow_error If the count exceeds `int64`. The oracle counts in
    ///         Python integers and cannot reach this; see the file header.
    [[nodiscard]] std::int64_t level_cells_per_axis(std::int64_t level,
                                                    std::int64_t axis) const {
        if (level < 0) {
            throw std::invalid_argument("level must be >= 0; got " + std::to_string(level)
                                        + ".");
        }
        if (axis < 0 || axis >= this->ndim()) {
            throw std::out_of_range("axis " + std::to_string(axis) + " is out of range [0, "
                                    + std::to_string(this->ndim()) + ").");
        }
        const auto k = static_cast<std::size_t>(axis);
        return checked_scale(root_.cells_per_axis()[k], factor_[k], level);
    }

    /// The active-leaf rectangles at `level`, as views into this grid's own storage.
    ///
    /// \param level Hierarchy level in `[0, max_level()]`.
    /// \return `(lo, hi)`, each `(n_blocks_at_level, ndim())` row-major, sorted
    ///         lexicographically by lower corner. Valid for as long as the grid is.
    /// \throws std::invalid_argument If `level` is outside `[0, max_level()]`.
    [[nodiscard]] std::pair<span2d<const std::int64_t>, span2d<const std::int64_t>>
    active_blocks(std::int64_t level) const {
        check_level(level);
        const auto l = static_cast<std::size_t>(level);
        const auto first = static_cast<std::size_t>(level_start_[l]);
        const auto count = static_cast<std::size_t>(level_start_[l + 1] - level_start_[l]);
        const auto d = static_cast<std::size_t>(this->ndim());
        return {span2d<const std::int64_t>(block_lo_.data() + first * d, count, d),
                span2d<const std::int64_t>(block_hi_.data() + first * d, count, d)};
    }

    /// The per-axis index of cell `cid` in its own level's coordinates.
    ///
    /// \param cid Cell identifier.
    /// \param out Receives the index; must have `ndim()` entries.
    /// \throws std::out_of_range If `cid` is out of range.
    /// \throws std::invalid_argument If `out` is the wrong length.
    void cell_multi_index(std::int64_t cid, std::span<std::int64_t> out) const {
        this->check_cid(cid);
        require_ndim_span(out, "out");
        (void)decode(cid, out);
    }

    /// The flat id of the active leaf at `(level, midx)`, if there is one.
    ///
    /// The inverse of `cell_level` and `cell_multi_index` taken together, and the way to
    /// follow a cell across a refinement: `(level, midx)` is stable where a flat id is
    /// not.
    ///
    /// \param level Hierarchy level.
    /// \param midx Per-axis index in level-`level` coordinates, `ndim()` entries.
    /// \return The flat id when `(level, midx)` is an active leaf; empty when it is out
    ///         of range, not yet created, or refined away.
    /// \throws std::invalid_argument If `midx` is the wrong length.
    [[nodiscard]] std::optional<std::int64_t> cell_id(
        std::int64_t level, std::span<const std::int64_t> midx) const {
        require_ndim_span(midx, "midx");
        if (level < 0 || level > max_level()) {
            return std::nullopt;
        }
        if (std::any_of(midx.begin(), midx.end(), [](std::int64_t i) { return i < 0; })) {
            return std::nullopt;
        }
        return encode(level, midx);
    }

    /// Whether `(level, midx)` is an active leaf.
    ///
    /// \param level Hierarchy level.
    /// \param midx Per-axis index in level-`level` coordinates, `ndim()` entries.
    /// \return `true` iff `cell_id(level, midx)` holds a value.
    /// \throws std::invalid_argument If `midx` is the wrong length.
    [[nodiscard]] bool is_active_leaf(std::int64_t level,
                                      std::span<const std::int64_t> midx) const {
        return cell_id(level, midx).has_value();
    }

    /// A mask of the active-leaf cells at `level`, over that level's whole cell grid.
    ///
    /// \param level Hierarchy level in `[0, max_level()]`.
    /// \return `1` where the level-`level` cell is an active leaf, `0` elsewhere, in
    ///         C-order over `level_cells_per_axis(level, .)`.
    /// \throws std::invalid_argument If `level` is outside `[0, max_level()]`.
    /// \throws std::overflow_error If the level's cell count exceeds `int64`.
    /// \note `std::uint8_t` rather than `std::vector<bool>`: the proxy specialisation has
    ///       no contiguous buffer, and a binding needs one to present the mask as an
    ///       array without copying it element by element.
    [[nodiscard]] std::vector<std::uint8_t> active_leaf_mask(std::int64_t level) const {
        check_level(level);
        const std::vector<std::int64_t> shape = level_shape(level);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(level_count(shape)), 0U);
        const auto [lo, hi] = active_blocks(level);
        for (std::size_t b = 0; b < lo.extent(0); ++b) {
            fill_box(mask, shape, block_row(lo, b), block_row(hi, b), 1U);
        }
        return mask;
    }

    /// A mask of the level-`level` refined subdomain.
    ///
    /// A level-`level` cell is in the subdomain iff it is **not** covered by an active
    /// leaf of a coarser level. Computed by projecting every coarser block up to
    /// `level`'s resolution and clearing what it covers, which is what makes the answer
    /// independent of how the coarser levels happen to be cut into rectangles.
    ///
    /// \param level Hierarchy level in `[0, max_level()]`.
    /// \return `1` inside the subdomain, `0` outside, in C-order over
    ///         `level_cells_per_axis(level, .)`.
    /// \throws std::invalid_argument If `level` is outside `[0, max_level()]`.
    /// \throws std::overflow_error If the level's cell count exceeds `int64`.
    [[nodiscard]] std::vector<std::uint8_t> subdomain_mask(std::int64_t level) const {
        check_level(level);
        const std::vector<std::int64_t> shape = level_shape(level);
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(level_count(shape)), 1U);
        const auto d = static_cast<std::size_t>(this->ndim());
        std::vector<std::int64_t> scaled_lo(d);
        std::vector<std::int64_t> scaled_hi(d);
        std::vector<std::int64_t> scale(d);
        for (std::int64_t coarser = 0; coarser < level; ++coarser) {
            // Per coarser level, not per block: the projection factor depends only on the
            // level gap, and computing it is O(gap) checked multiplications.
            for (std::size_t k = 0; k < d; ++k) {
                scale[k] = checked_scale(1, factor_[k], level - coarser);
            }
            const auto [lo, hi] = active_blocks(coarser);
            for (std::size_t b = 0; b < lo.extent(0); ++b) {
                for (std::size_t k = 0; k < d; ++k) {
                    scaled_lo[k] = lo(b, k) * scale[k];
                    scaled_hi[k] = hi(b, k) * scale[k];
                }
                fill_box(mask, shape, scaled_lo, scaled_hi, 0U);
            }
        }
        return mask;
    }

    /// A compact representation, character for character the oracle's `repr`.
    ///
    /// \return `"HierarchicalGrid(ndim=..., root_cells=(...), factor=(...), "`
    ///         `"num_cells=..., max_level=...)"`.
    [[nodiscard]] std::string to_string() const {
        return "HierarchicalGrid(ndim=" + std::to_string(this->ndim())
               + ", root_cells=" + tuple_repr(root_.cells_per_axis())
               + ", factor=" + tuple_repr(factor_)
               + ", num_cells=" + std::to_string(this->num_cells())
               + ", max_level=" + std::to_string(max_level()) + ")";
    }

    // ----------------------------------------------------------------
    // The three primitives
    // ----------------------------------------------------------------

    /// The axis-aligned corners of cell `cid`.
    ///
    /// \param cid Cell identifier.
    /// \param lo Receives the lower corner; must have `ndim()` entries.
    /// \param hi Receives the upper corner; must have `ndim()` entries.
    /// \throws std::out_of_range If `cid` is out of range.
    /// \throws std::invalid_argument If either span is the wrong length.
    void cell_bounds(std::int64_t cid, std::span<T> lo, std::span<T> hi) const {
        this->check_cid(cid);
        require_ndim_span(lo, "lo");
        require_ndim_span(hi, "hi");
        const auto d = static_cast<std::size_t>(this->ndim());
        std::vector<std::int64_t> midx(d);
        const std::int64_t level = decode(cid, midx);
        bounds_at(level, midx, lo, hi);
    }

    /// The active leaf containing `pt`, by descent from the root level.
    ///
    /// \param pt A point in parametric coordinates, `ndim()` entries.
    /// \return The cell id, or empty when `pt` is outside the grid's domain.
    /// \throws std::invalid_argument If `pt` is the wrong length.
    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const T> pt) const {
        require_ndim_span(pt, "pt");
        const std::optional<std::int64_t> root_cid = root_.locate(pt);
        if (!root_cid) {
            return std::nullopt;
        }
        const auto d = static_cast<std::size_t>(this->ndim());
        std::vector<std::int64_t> midx(d);
        root_.cell_multi_index(*root_cid, midx);

        std::vector<T> lo(d);
        std::vector<T> hi(d);
        for (std::size_t k = 0; k < d; ++k) {
            const std::span<const T> bp = root_.breakpoints(static_cast<std::int64_t>(k));
            lo[k] = bp[static_cast<std::size_t>(midx[k])];
            hi[k] = bp[static_cast<std::size_t>(midx[k] + 1)];
        }

        const std::int64_t n_levels = max_level() + 1;
        for (std::int64_t level = 0; level < n_levels; ++level) {
            const std::optional<std::int64_t> cid = encode(level, midx);
            if (cid) {
                return cid;
            }
            if (level >= n_levels - 1) {
                return std::nullopt;  // unreachable in a consistent grid
            }
            descend(pt, midx, lo, hi);
        }
        return std::nullopt;  // unreachable
    }

    /// The cell across local facet `lfid` of `cid`.
    ///
    /// Handles a hanging interface at **any** level difference, since nothing here
    /// enforces 2:1 balance. A coarser neighbour is the active leaf covering the
    /// position however many levels up; a finer one is the first active descendant
    /// touching the shared face, in C-order along it. `hanging_neighbors` returns them
    /// all.
    ///
    /// \param cid Cell identifier.
    /// \param lfid Local facet identifier in `[0, 2 * ndim())`.
    /// \return The neighbouring cell id, or empty on an outer-boundary facet.
    /// \throws std::out_of_range If `cid` or `lfid` is out of range.
    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t cid,
                                                                    std::int64_t lfid) const {
        const std::optional<FacetPosition> position = facet_neighbor_position(cid, lfid);
        if (!position) {
            return std::nullopt;
        }
        const std::optional<std::int64_t> conforming = encode(position->level, position->midx);
        if (conforming) {
            return conforming;
        }
        const std::optional<std::int64_t> coarser =
            nearest_active_ancestor(position->level, position->midx);
        if (coarser) {
            return coarser;
        }
        std::vector<std::int64_t> finer;
        active_face_descendants(position->level, position->midx, position->axis,
                                position->face_j, finer);
        if (finer.empty()) {
            return std::nullopt;
        }
        return finer.front();
    }

    // ----------------------------------------------------------------
    // The five hooks
    // ----------------------------------------------------------------

    /// The refinement level of cell `cid`.
    ///
    /// \param cid Cell identifier.
    /// \return The level, `0` for an unrefined root cell.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] std::int64_t cell_level(std::int64_t cid) const {
        this->check_cid(cid);
        std::vector<std::int64_t> midx(static_cast<std::size_t>(this->ndim()));
        return decode(cid, midx);
    }

    /// Every active cell sharing local facet `lfid` of `cid`.
    ///
    /// One neighbour for a conforming or coarser interface. For a hanging one, every
    /// active leaf touching the face, descending as many levels as the interface needs,
    /// depth-first along the face.
    ///
    /// \param cid Cell identifier.
    /// \param lfid Local facet identifier in `[0, 2 * ndim())`.
    /// \return The neighbouring cell ids; empty on an outer-boundary facet.
    /// \throws std::out_of_range If `cid` or `lfid` is out of range.
    [[nodiscard]] std::vector<std::int64_t> hanging_neighbors(std::int64_t cid,
                                                              std::int64_t lfid) const {
        const std::optional<FacetPosition> position = facet_neighbor_position(cid, lfid);
        if (!position) {
            return {};
        }
        const std::optional<std::int64_t> conforming = encode(position->level, position->midx);
        if (conforming) {
            return {*conforming};
        }
        const std::optional<std::int64_t> coarser =
            nearest_active_ancestor(position->level, position->midx);
        if (coarser) {
            return {*coarser};
        }
        std::vector<std::int64_t> finer;
        active_face_descendants(position->level, position->midx, position->axis,
                                position->face_j, finer);
        return finer;
    }

    /// Every outer-boundary facet, as `(cid, lfid)` rows.
    ///
    /// The generic default asks `2 * ndim` neighbour questions per cell. Here a
    /// level-`l` cell's facet `(axis, side)` is on the outer boundary exactly when its
    /// level-`l` index along `axis` is `0` or `level_cells_per_axis - 1`, which is the
    /// same out-of-range test the neighbour query makes -- and since a block is a
    /// rectangle of contiguous indices, that is decided **once per block face** rather
    /// than once per cell.
    ///
    /// \return `2 * n` values: `n` row-major `(cid, lfid)` pairs, sorted by `(cid,
    ///         lfid)`.
    [[nodiscard]] std::vector<std::int64_t> boundary_facets() const {
        const auto d = static_cast<std::size_t>(this->ndim());
        std::vector<std::int64_t> rows;
        std::vector<std::int64_t> face_lo(d);
        std::vector<std::int64_t> face_hi(d);

        for (std::int64_t level = 0; level <= max_level(); ++level) {
            // Hoisted: `factor ** level` costs O(level) checked multiplications, and the
            // block loop below would otherwise pay it once per block per axis for a
            // quantity that only depends on the level.
            const std::vector<std::int64_t> n_per_axis = level_shape(level);
            const auto [lo, hi] = active_blocks(level);
            for (std::size_t b = 0; b < lo.extent(0); ++b) {
                const std::int64_t base = block_base_[
                    static_cast<std::size_t>(level_start_[static_cast<std::size_t>(level)])
                    + b];
                for (std::size_t axis = 0; axis < d; ++axis) {
                    const std::int64_t n_axis = n_per_axis[axis];
                    for (std::int64_t side = 0; side < 2; ++side) {
                        const bool touches =
                            (side == 0) ? lo(b, axis) == 0 : hi(b, axis) == n_axis;
                        if (!touches) {
                            continue;
                        }
                        // The boundary layer of the block: the normal axis pinned to the
                        // single index on the face, every other axis spanning the block.
                        for (std::size_t k = 0; k < d; ++k) {
                            face_lo[k] = lo(b, k);
                            face_hi[k] = hi(b, k);
                        }
                        if (side == 0) {
                            face_hi[axis] = face_lo[axis] + 1;
                        } else {
                            face_lo[axis] = face_hi[axis] - 1;
                        }
                        emit_face(base, block_row(lo, b), block_row(hi, b), face_lo, face_hi,
                                  2 * static_cast<std::int64_t>(axis) + side, rows);
                    }
                }
            }
        }
        sort_facet_rows(rows);
        return rows;
    }

    /// Locate a batch of points through the shared descent kernel.
    ///
    /// \param points `(npts, ndim())` row-major view of query points.
    /// \return `npts` cell ids; `-1` for a point outside the domain, and for one with a
    ///         non-finite coordinate.
    /// \throws std::invalid_argument If `points.extent(1)` is not `ndim()`.
    /// \note The kernel cannot answer the non-finite case itself and this masks it
    ///       afterwards, exactly as the oracle does. Its root-containment test is
    ///       `x < lo || x > hi`, and **every comparison against a NaN is false**, so the
    ///       row passes the test, descends, and lands in a real cell -- a plausible wrong
    ///       id rather than a crash. The scalar `locate` needs no such pass, because
    ///       `TensorProductGrid::locate` writes the same test as a negated comparison
    ///       that rejects a NaN and either infinity; the two spellings are not
    ///       interchangeable and `tensor_product_grid.hpp` says so at its own site.
    [[nodiscard]] std::vector<std::int64_t> locate_many(span2d<const T> points) const {
        if (points.extent(1) != static_cast<std::size_t>(this->ndim())) {
            throw std::invalid_argument("points must have ndim() columns.");
        }
        std::vector<std::int64_t> out(points.extent(0));
        hier_locate_points<T>(points, root_.breakpoints_flat(), root_.axis_starts(),
                              root_.cells_per_axis(), factor_, packed_lo(), packed_hi(),
                              block_base_, level_start_, out);
        for (std::size_t p = 0; p < points.extent(0); ++p) {
            for (std::size_t k = 0; k < points.extent(1); ++k) {
                if (!std::isfinite(value_of(points(p, k)))) {
                    out[p] = -1;
                    break;
                }
            }
        }
        return out;
    }

    /// Materialise per-cell `(lo, hi)` in flat-id order.
    ///
    /// \param cell_lo Output lo corners, shape `(num_cells(), ndim())`.
    /// \param cell_hi Output hi corners, same shape.
    /// \throws std::invalid_argument If either view is the wrong shape.
    void collect_cell_bounds(span2d<T> cell_lo, span2d<T> cell_hi) const {
        const auto n = static_cast<std::size_t>(this->num_cells());
        const auto d = static_cast<std::size_t>(this->ndim());
        if (cell_lo.extent(0) != n || cell_lo.extent(1) != d || cell_hi.extent(0) != n
            || cell_hi.extent(1) != d) {
            throw std::invalid_argument("cell_lo and cell_hi must have shape (num_cells, ndim).");
        }
        hier_collect_cell_bounds<T>(root_.breakpoints_flat(), root_.axis_starts(), factor_,
                                    packed_lo(), packed_hi(), block_base_, level_start_,
                                    cell_lo, cell_hi);
    }

    /// The root-cell-aligned bounding-box sub-grid spanning `cell_ids`.
    ///
    /// The window is the multi-index bounding box **in root-cell coordinates** of the
    /// root cells holding the requested leaves: a leaf at `(level, midx)` lives in root
    /// cell `midx[k] / factor[k] ** level`. The sub-grid's root is the matching slice of
    /// this grid's breakpoints, never re-clamped, and it keeps the same `factor`; its
    /// active leaves are the per-level intersections of this grid's blocks with the
    /// window, translated into the window's own coordinates.
    ///
    /// Because the window is root-cell-aligned, restricting one deep leaf returns the
    /// whole leaf tiling of its root cell, with only the requested leaf flagged in
    /// `in_subset`.
    ///
    /// Replaces the mixin's throwing default: restriction is an optional grid capability
    /// and a hierarchy has it.
    ///
    /// \param cell_ids Flat cell identifiers to span; duplicates are ignored.
    /// \return The windowed sub-grid, its `local_to_global_cell` map and the `in_subset`
    ///         mask separating requested cells from bounding-box fill.
    /// \throws std::invalid_argument If `cell_ids` is empty.
    /// \throws std::out_of_range If any cell id is outside `[0, num_cells())`.
    /// \throws std::logic_error If a windowed leaf does not map back to an active leaf of
    ///         this grid. That is an internal invariant, so it should be unreachable; the
    ///         oracle spells the same check as a bare `assert`, which `-O` deletes.
    [[nodiscard]] GridRestriction<HierarchicalGrid<T>> restrict(
        std::span<const std::int64_t> cell_ids) const {
        if (cell_ids.empty()) {
            throw std::invalid_argument("restrict: cell_ids must be non-empty.");
        }
        const auto d = static_cast<std::size_t>(this->ndim());
        for (const std::int64_t cid : cell_ids) {
            if (cid < 0 || cid >= this->num_cells()) {
                throw std::out_of_range("restrict: cell id out of range [0, "
                                        + std::to_string(this->num_cells()) + "); got "
                                        + std::to_string(cid) + ".");
            }
        }

        // `factor ** level` for every level, once. The three loops below each want it, and
        // `int_pow` is O(level) checked multiplications, so computing it per cell and per
        // axis makes the bounding-box walk quadratic in the depth for no reason.
        // `boundary_facets` and `subdomain_mask` hoist the same quantity the same way.
        const std::int64_t n_levels = max_level() + 1;
        std::vector<std::int64_t> span_at(static_cast<std::size_t>(n_levels) * d);
        for (std::int64_t level = 0; level < n_levels; ++level) {
            for (std::size_t k = 0; k < d; ++k) {
                span_at[static_cast<std::size_t>(level) * d + k] = int_pow(factor_[k], level);
            }
        }

        // Root-cell bounding box over the requested leaves.
        std::vector<std::int64_t> window_lo(root_.cells_per_axis().begin(),
                                            root_.cells_per_axis().end());
        std::vector<std::int64_t> window_hi(d, 0);
        std::vector<std::int64_t> midx(d);
        for (const std::int64_t cid : cell_ids) {
            const std::int64_t level = decode(cid, midx);
            for (std::size_t k = 0; k < d; ++k) {
                const std::int64_t root_ik =
                    midx[k] / span_at[static_cast<std::size_t>(level) * d + k];
                window_lo[k] = std::min(window_lo[k], root_ik);
                window_hi[k] = std::max(window_hi[k], root_ik + 1);
            }
        }

        std::vector<std::vector<T>> sub_breakpoints(d);
        for (std::size_t k = 0; k < d; ++k) {
            const std::span<const T> bp = root_.breakpoints(static_cast<std::int64_t>(k));
            sub_breakpoints[k].assign(bp.data() + window_lo[k], bp.data() + window_hi[k] + 1);
        }
        TensorProductGrid<T> sub_root(sub_breakpoints);

        // Clipping a normalized partition to a window can make two blocks mergeable that
        // were not, so every level is handed to the merge rather than taken verbatim.
        std::vector<BlockList> sub_blocks;
        sub_blocks.reserve(static_cast<std::size_t>(n_levels));
        std::vector<std::int64_t> level_lo(d);
        std::vector<std::int64_t> level_hi(d);
        std::vector<std::int64_t> clip_lo(d);
        std::vector<std::int64_t> clip_hi(d);
        for (std::int64_t level = 0; level < n_levels; ++level) {
            for (std::size_t k = 0; k < d; ++k) {
                const std::int64_t span_k = span_at[static_cast<std::size_t>(level) * d + k];
                level_lo[k] = checked_mul(window_lo[k], span_k);
                level_hi[k] = checked_mul(window_hi[k], span_k);
            }
            const BlockView window{level_lo, level_hi};
            const BlockList at_level = level_blocks(level);
            BlockList clipped(this->ndim());
            for (std::size_t b = 0; b < at_level.size(); ++b) {
                if (!rect_intersect(at_level[b], window, clip_lo, clip_hi)) {
                    continue;
                }
                for (std::size_t k = 0; k < d; ++k) {
                    clip_lo[k] -= level_lo[k];
                    clip_hi[k] -= level_lo[k];
                }
                clipped.push_back(BlockView{clip_lo, clip_hi});
            }
            sub_blocks.push_back(std::move(clipped));
        }
        HierarchicalGrid sub = from_blocks(std::move(sub_root), factor_, std::move(sub_blocks),
                                           std::nullopt);

        const auto sub_cells = static_cast<std::size_t>(sub.num_cells());
        std::vector<std::int64_t> local_to_global(sub_cells);
        std::vector<std::int64_t> sub_midx(d);
        std::vector<std::int64_t> global_midx(d);
        for (std::size_t local = 0; local < sub_cells; ++local) {
            const auto local_cid = static_cast<std::int64_t>(local);
            const std::int64_t level = sub.decode(local_cid, sub_midx);
            for (std::size_t k = 0; k < d; ++k) {
                global_midx[k] =
                    sub_midx[k]
                    + checked_mul(window_lo[k],
                                  span_at[static_cast<std::size_t>(level) * d + k]);
            }
            const std::optional<std::int64_t> global = encode(level, global_midx);
            if (!global.has_value()) {
                throw std::logic_error(
                    "restrict: a windowed leaf is not an active leaf of the grid it came "
                    "from, so the two block sets disagree.");
            }
            local_to_global[local] = *global;
        }

        // Membership in the REQUESTED set, which is a strict subset of the window
        // whenever the request is non-convex or does not fill its root cells. Sorting a
        // copy keeps the test logarithmic instead of quadratic in the window, exactly as
        // `TensorProductGrid::restrict` does.
        std::vector<std::int64_t> requested(cell_ids.begin(), cell_ids.end());
        std::sort(requested.begin(), requested.end());
        std::vector<std::uint8_t> in_subset(sub_cells);
        for (std::size_t local = 0; local < sub_cells; ++local) {
            in_subset[local] = std::binary_search(requested.begin(), requested.end(),
                                                  local_to_global[local])
                                   ? 1U
                                   : 0U;
        }
        return make_restriction(std::move(sub), std::move(local_to_global),
                                std::move(in_subset));
    }

    // ----------------------------------------------------------------
    // Refinement and coarsening
    // ----------------------------------------------------------------

    /// A new grid with the active part of `[lo, hi)` at `level` promoted to `level + 1`.
    ///
    /// This grid is left untouched; write `grid = grid.refine(...)`.
    ///
    /// **Union semantics.** Only the currently active portion of `[lo, hi)` is refined,
    /// so cells already deeper are left alone and overlapping calls safely extend the
    /// refined region. When the intersection with the active blocks at `level` is empty
    /// the result is an unrefined copy.
    ///
    /// That convenience costs invertibility: promoting only part of a box sends several
    /// distinct grids to one result, so `refine` is **not injective** and `coarsen` does
    /// not undo it in general. `coarsen(level, lo, hi)` reverses this call only when
    /// every cell of `[lo, hi)` was an active leaf at `level` beforehand; otherwise it
    /// either refuses the box or demotes all of it, children this call never created
    /// included. Name cells rather than a box on the destroying side where that matters:
    /// `coarsen_cells` reactivates a parent only when all of its children are named.
    ///
    /// The result numbers its cells afresh, so an id read from this grid does not name
    /// the same cell there; its BVH and its two tag registries start empty, and this
    /// grid keeps its own.
    ///
    /// \param level Level the region lives at, in `[0, max_level()]`.
    /// \param lo Per-axis start index, inclusive, in level-`level` coordinates.
    /// \param hi Per-axis end index, exclusive, in level-`level` coordinates.
    /// \return The refined grid.
    /// \throws std::invalid_argument If `level` is out of range, `lo` or `hi` is not
    ///         `ndim()` long, some `lo[k] >= hi[k]`, or `[lo, hi)` leaves the level's
    ///         domain.
    /// \throws std::overflow_error If a child index or the new cell count exceeds
    ///         `int64`; see the file header.
    [[nodiscard]] HierarchicalGrid refine(std::int64_t level, std::span<const std::int64_t> lo,
                                          std::span<const std::int64_t> hi) const {
        check_level(level);
        check_region(level, lo, hi);
        const auto d = static_cast<std::size_t>(this->ndim());

        const BlockView region{lo, hi};
        BlockList kept(this->ndim());
        BlockList children(this->ndim());
        std::vector<std::int64_t> cut_lo(d);
        std::vector<std::int64_t> cut_hi(d);
        std::vector<std::int64_t> child_lo(d);
        std::vector<std::int64_t> child_hi(d);
        const BlockList at_level = level_blocks(level);
        for (std::size_t b = 0; b < at_level.size(); ++b) {
            const BlockView block = at_level[b];
            if (!rect_intersect(block, region, cut_lo, cut_hi)) {
                kept.push_back(block);
                continue;
            }
            const BlockView cut{cut_lo, cut_hi};
            peel(block, cut, kept);
            for (std::size_t k = 0; k < d; ++k) {
                child_lo[k] = checked_mul(cut_lo[k], factor_[k]);
                child_hi[k] = checked_mul(cut_hi[k], factor_[k]);
            }
            children.push_back(BlockView{child_lo, child_hi});
        }
        if (children.empty()) {
            return rebuilt();  // no active cell in the requested region
        }

        std::vector<BlockList> levels = all_level_blocks();
        levels[static_cast<std::size_t>(level)] = std::move(kept);
        while (static_cast<std::int64_t>(levels.size()) <= level + 1) {
            levels.emplace_back(this->ndim());
        }
        BlockList& finer = levels[static_cast<std::size_t>(level + 1)];
        for (std::size_t i = 0; i < children.size(); ++i) {
            finer.push_back(children[i]);
        }
        const std::array<std::int64_t, 2> dirty{level, level + 1};
        return from_blocks(root_, factor_, std::move(levels),
                           std::span<const std::int64_t>{dirty});
    }

    /// A new grid with the named cells refined, level by level, by per-level bounding box.
    ///
    /// Groups `cell_ids` by level, takes the smallest rectangle containing the cells at
    /// each level, and applies `refine` once per level from the coarsest. This grid is
    /// left untouched.
    ///
    /// Not the mirror of `coarsen_cells`: refining the bounding box can promote a cell
    /// the caller never named. That costs nothing, because refining destroys no cell.
    ///
    /// \param cell_ids Flat cell ids to refine; repeats are ignored, several levels are
    ///        handled in one call, and an empty range yields an unrefined copy.
    /// \return The refined grid.
    /// \throws std::out_of_range If any id is outside `[0, num_cells())`.
    /// \throws std::overflow_error If a child index or the new cell count exceeds `int64`.
    [[nodiscard]] HierarchicalGrid refine_cells(std::span<const std::int64_t> cell_ids) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        const auto n_levels = static_cast<std::size_t>(max_level() + 1);
        std::vector<std::uint8_t> marked(n_levels, 0U);
        std::vector<std::int64_t> box_lo(n_levels * d, 0);
        std::vector<std::int64_t> box_hi(n_levels * d, 0);
        std::vector<std::int64_t> midx(d);
        for (const std::int64_t cid : cell_ids) {
            this->check_cid(cid);
            const auto level = static_cast<std::size_t>(decode(cid, midx));
            for (std::size_t k = 0; k < d; ++k) {
                if (marked[level] == 0U) {
                    box_lo[level * d + k] = midx[k];
                    box_hi[level * d + k] = midx[k] + 1;
                } else {
                    box_lo[level * d + k] = std::min(box_lo[level * d + k], midx[k]);
                    box_hi[level * d + k] = std::max(box_hi[level * d + k], midx[k] + 1);
                }
            }
            marked[level] = 1U;
        }

        std::optional<HierarchicalGrid> refined;
        for (std::size_t level = 0; level < n_levels; ++level) {
            if (marked[level] == 0U) {
                continue;
            }
            const HierarchicalGrid& current = refined.has_value() ? *refined : *this;
            HierarchicalGrid next =
                current.refine(static_cast<std::int64_t>(level),
                               std::span<const std::int64_t>(box_lo.data() + level * d, d),
                               std::span<const std::int64_t>(box_hi.data() + level * d, d));
            refined = std::move(next);
        }
        return refined.has_value() ? std::move(*refined) : rebuilt();
    }

    /// A new grid with `[lo, hi)` demoted from `level + 1` back to `level`.
    ///
    /// This grid is left untouched. The level-`level` cells of `[lo, hi)` are reactivated
    /// and their level-`(level + 1)` children removed, which requires the region to be
    /// **fully refined to exactly `level + 1`**: every child cell of
    /// `[lo * factor, hi * factor)` must be an active leaf there. Otherwise the call
    /// throws and the message names the cells that break it.
    ///
    /// The **whole** box is demoted, while `refine` promotes only its active portion. So
    /// this inverts `refine` only when every cell of `[lo, hi)` was an active leaf at
    /// `level` before that call; when it was not, this either refuses the box or removes
    /// children an earlier `refine` created. The opposite order carries no hypothesis:
    /// coarsening leaves the whole box active at `level`, so `refine(level, lo, hi)`
    /// always undoes `coarsen(level, lo, hi)`.
    ///
    /// The result numbers its cells afresh and its caches start empty, as `refine`'s does.
    ///
    /// \param level Level whose cells are reactivated, in `[0, max_level())`.
    /// \param lo Per-axis start index, inclusive, in level-`level` coordinates.
    /// \param hi Per-axis end index, exclusive, in level-`level` coordinates.
    /// \return The coarsened grid.
    /// \throws std::invalid_argument If `level` is out of range, `lo` or `hi` is not
    ///         `ndim()` long, some `lo[k] >= hi[k]`, `[lo, hi)` leaves the level's domain,
    ///         or the region is not fully refined to exactly `level + 1`.
    /// \throws std::overflow_error If a child index or a cell count exceeds `int64`.
    [[nodiscard]] HierarchicalGrid coarsen(std::int64_t level, std::span<const std::int64_t> lo,
                                           std::span<const std::int64_t> hi) const {
        if (level < 0 || level >= max_level()) {
            throw std::invalid_argument("level must be in [0, " + std::to_string(max_level())
                                        + "); got " + std::to_string(level) + ".");
        }
        check_region(level, lo, hi);
        const auto d = static_cast<std::size_t>(this->ndim());

        std::vector<std::int64_t> child_lo(d);
        std::vector<std::int64_t> child_hi(d);
        for (std::size_t k = 0; k < d; ++k) {
            child_lo[k] = checked_mul(lo[k], factor_[k]);
            child_hi[k] = checked_mul(hi[k], factor_[k]);
        }
        const BlockView children{child_lo, child_hi};
        const std::int64_t wanted = checked_block_size(children);

        std::int64_t covered = 0;
        BlockList kept(this->ndim());
        std::vector<std::int64_t> cut_lo(d);
        std::vector<std::int64_t> cut_hi(d);
        const BlockList at_finer = level_blocks(level + 1);
        for (std::size_t b = 0; b < at_finer.size(); ++b) {
            const BlockView block = at_finer[b];
            if (!rect_intersect(block, children, cut_lo, cut_hi)) {
                kept.push_back(block);
                continue;
            }
            const BlockView cut{cut_lo, cut_hi};
            covered = checked_add(covered, checked_block_size(cut));
            peel(block, cut, kept);
        }
        if (covered != wanted) {
            throw std::invalid_argument("cannot coarsen [" + tuple_repr(lo) + ", "
                                        + tuple_repr(hi) + ") at level "
                                        + std::to_string(level)
                                        + ": the region is not fully refined to exactly level "
                                        + std::to_string(level + 1) + "."
                                        + coarsen_refusal_detail(level, lo, hi));
        }

        // The order the demoted box is appended in is observable through the greedy
        // merge, so it goes last, as the oracle appends it.
        std::vector<BlockList> levels = all_level_blocks();
        levels[static_cast<std::size_t>(level + 1)] = std::move(kept);
        levels[static_cast<std::size_t>(level)].push_back(BlockView{lo, hi});
        const std::array<std::int64_t, 2> dirty{level, level + 1};
        return from_blocks(root_, factor_, std::move(levels),
                           std::span<const std::int64_t>{dirty});
    }

    /// A new grid with every parent demoted whose children are all named.
    ///
    /// The route that destroys only what the caller named. `cell_ids` are grouped by
    /// parent, and a parent is reactivated -- its children removed -- only when **every
    /// one** of its `prod(factor)` children is both an active leaf and present in
    /// `cell_ids`. Three cases are therefore skipped silently rather than refused: a
    /// parent only some of whose children are named, a parent one of whose children is
    /// refined further, and an id at level 0, which has no parent. A call that demotes
    /// nothing is not an error, and no cell outside `cell_ids` is ever removed.
    ///
    /// Ids spanning several levels are handled in one call, deepest first. That order is
    /// observable -- reactivated parents are appended in it, the greedy merge turns the
    /// result into a rectangle partition, and flat ids are handed out block by block --
    /// so it is fixed rather than incidental. Coarsening does not cascade in any order: a
    /// cell reborn by this call was not an active leaf when the caller chose its ids, so
    /// it cannot be among them and its own parent is never complete.
    ///
    /// \param cell_ids Flat ids of the active leaves to coarsen away. Repeats and level-0
    ///        ids are ignored; an empty range, and one that demotes nothing, both yield an
    ///        uncoarsened copy.
    /// \return The coarsened grid.
    /// \throws std::out_of_range If any id is outside `[0, num_cells())`.
    /// \throws std::overflow_error If a cell count exceeds `int64`.
    [[nodiscard]] HierarchicalGrid coarsen_cells(std::span<const std::int64_t> cell_ids) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        std::vector<std::int64_t> marked;  // (level, midx...) rows, sorted and unique
        marked.reserve(cell_ids.size() * (d + 1));
        std::vector<std::int64_t> midx(d);
        for (const std::int64_t cid : cell_ids) {
            this->check_cid(cid);
            const std::int64_t level = decode(cid, midx);
            marked.push_back(level);
            marked.insert(marked.end(), midx.begin(), midx.end());
        }
        sort_unique_rows(marked, d + 1);

        // Parents, **deepest level first and then ascending by index**, so the outcome
        // does not depend on the order the ids arrived in. Each row carries the NEGATED
        // level, because that is what makes one ascending lexicographic sort produce
        // exactly that order: sorting on the level itself and walking the list backwards
        // reverses the index too, which is a different and equally valid order that
        // numbers the cells differently -- see the demotion-order test.
        std::vector<std::int64_t> parents;
        for (std::size_t row = 0; row * (d + 1) < marked.size(); ++row) {
            const std::int64_t level = marked[row * (d + 1)];
            if (level < 1) {
                continue;
            }
            parents.push_back(-(level - 1));
            for (std::size_t k = 0; k < d; ++k) {
                parents.push_back(marked[row * (d + 1) + 1 + k] / factor_[k]);
            }
        }
        sort_unique_rows(parents, d + 1);

        std::optional<HierarchicalGrid> coarsened;
        std::vector<std::int64_t> child(d);
        std::vector<std::int64_t> parent_hi(d);
        for (std::size_t row = 0; row < parents.size(); row += d + 1) {
            const std::int64_t parent_level = -parents[row];
            const std::span<const std::int64_t> pmidx(parents.data() + row + 1, d);
            const HierarchicalGrid& current = coarsened.has_value() ? *coarsened : *this;
            if (!family_is_complete(current, marked, parent_level, pmidx, child)) {
                continue;
            }
            for (std::size_t k = 0; k < d; ++k) {
                parent_hi[k] = pmidx[k] + 1;
            }
            HierarchicalGrid next = current.coarsen(parent_level, pmidx, parent_hi);
            coarsened = std::move(next);
        }
        return coarsened.has_value() ? std::move(*coarsened) : rebuilt();
    }

    // ----------------------------------------------------------------
    // Leaf-cell mesh export
    // ----------------------------------------------------------------

    /// Deduplicated leaf-cell vertices and their connectivity.
    ///
    /// The mesh a dolfinx-style consumer needs: one vertex per distinct corner of the
    /// active leaves, shared exactly between the cells meeting there, corners shared
    /// across levels included, so a hanging vertex appears once.
    ///
    /// **Deduplication is exact and tolerance-free.** Corners are identified on the
    /// finest level's integer lattice, where a level-`l` cell at `midx` contributes
    /// `(midx[k] + bit[k]) * factor[k] ** (max_level() - l)`. Equal corners are equal
    /// integers, and one formula then maps each distinct node to coordinates, so
    /// coincident corners come out bitwise identical whatever the arithmetic does.
    ///
    /// Corner order within a row is the tensor-product (basix/dolfinx) convention: corner
    /// `c` takes the `hi` bound on axis `k` iff bit `k` of `c` is set, axis 0 the least
    /// significant. In 2D that is `[(lo0, lo1), (hi0, lo1), (lo0, hi1), (hi0, hi1)]`.
    /// It is a **corner** convention and deliberately unlike this library's C-order flat
    /// cell and dof ids, where the last axis varies fastest.
    ///
    /// \return The vertices, `(num_vertices, ndim())` row-major and sorted
    ///         lexicographically by lattice coordinate, and the connectivity,
    ///         `(num_cells(), 2 ** ndim())` row-major in flat cell-id order.
    /// \throws std::overflow_error If a lattice coordinate or the corner count exceeds
    ///         `int64`; the oracle counts in Python integers and cannot reach this.
    ///
    /// \note **How far this agrees with `cell_bounds`, and why not further.** Both build
    ///       the corner `b + s * w / factor ** l`, where `b` and `w` are the containing
    ///       root cell's lower breakpoint and width, but from different expressions: this
    ///       one divides by `factor ** max_level()` and scales by a lattice offset, while
    ///       `cell_bounds` divides by `factor ** l` and scales by a sub-index. Counting
    ///       roundings: forming `w`, the division, the integer-scaled multiply and the
    ///       addition are four on each side, and `cell_bounds` spends a fifth on a `hi`
    ///       corner, which it writes as `lo + size`. So with `u = eps / 2` and the closed
    ///       form `gamma_m = m u / (1 - m u)`,
    ///       `|export - cell_bounds| <= 2 gamma_2 |x| + 2 gamma_4 |x - b|`, for either
    ///       corner.
    ///
    ///       **The second term is what the bound is about, and a relative form drops it.**
    ///       `|x - b|` is bounded by the root cell's width and not by `|x|`, so only on a
    ///       domain where every coordinate dominates its own offset -- `b >= 0` is the
    ///       usual reason -- does this collapse to a multiple of `eps |x|`. Elsewhere it
    ///       does not, and a corner small against the width of the root cell holding it is
    ///       a counterexample rather than a corner case;
    ///       `test_grid_hierarchical_refine.cpp` pins one.
    ///
    ///       `design/backend_parity.md` Rule 2 is the general statement, **including its
    ///       converse**: a flat absolute bound would be vacuous on a corner near the
    ///       origin, admitting a relative error of one there. This is neither form. It
    ///       carries one term per mechanism -- the final addition, which scales with the
    ///       coordinate, and the three that build the offset, which scale with the root
    ///       cell's width -- which is what Rule 2 asks for.
    ///
    ///       Bitwise agreement is unattainable in general rather than merely
    ///       unimplemented: for a corner shared between two levels the two sides evaluate
    ///       different expressions and can land an ulp apart, and no single deduplicated
    ///       vertex equals both. It is exact whenever every intermediate is representable,
    ///       which covers the usual dyadic factor on dyadic breakpoints.
    ///
    ///       **Stated hypothesis: coordinates are normal, not subnormal.** The bound is
    ///       a product of relative factors with the magnitudes, so below roughly
    ///       `2e-308` both of its terms underflow to exactly zero while the difference
    ///       they bound is still a nonzero subnormal -- the expression stops
    ///       implementing the inequality even though the inequality still holds in the
    ///       reals. Measured: breakpoints near `1e-320` produce a zero bound against a
    ///       nonzero gap. Nothing excludes that domain, so it is recorded as a
    ///       hypothesis rather than a guarantee.
    ///
    /// \warning Materialises `num_cells() * 2 ** ndim() * ndim()` integers before
    ///          deduplication, so a large 3D hierarchy costs several times `conn` in peak
    ///          memory.
    [[nodiscard]] CellExport<T> export_cells() const {
        const auto d = static_cast<std::size_t>(this->ndim());
        const auto n_cells = static_cast<std::size_t>(this->num_cells());
        if (this->ndim() >= kMaxExportNdim) {
            throw std::overflow_error(
                "export_cells: 2 ** ndim corners per cell does not fit in int64 at ndim "
                + std::to_string(this->ndim()) + ".");
        }
        const auto corners = static_cast<std::size_t>(std::int64_t{1} << this->ndim());
        const std::int64_t top = max_level();
        // Every lattice coordinate on axis k is at most `level_cells_per_axis(top, k)`,
        // since `(midx[k] + 1) * factor[k] ** (top - l) <= n_l(k) * factor[k] ** (top - l)`
        // and `n_l(k) = n_root(k) * factor[k] ** l`. Checking that one product per axis
        // therefore covers every entry below, and the inner loop needs no guard.
        for (std::size_t k = 0; k < d; ++k) {
            static_cast<void>(level_cells_per_axis(top, static_cast<std::int64_t>(k)));
        }
        const auto lattice_size = static_cast<std::size_t>(
            checked_mul(checked_mul(static_cast<std::int64_t>(n_cells),
                                    static_cast<std::int64_t>(corners)),
                        static_cast<std::int64_t>(d)));
        std::vector<std::int64_t> lattice(lattice_size);

        std::vector<std::int64_t> scale(d);
        std::vector<std::int64_t> midx(d);
        for (std::int64_t level = 0; level <= top; ++level) {
            for (std::size_t k = 0; k < d; ++k) {
                scale[k] = int_pow(factor_[k], top - level);
            }
            const auto first = static_cast<std::size_t>(level_start_[
                static_cast<std::size_t>(level)]);
            const auto last = static_cast<std::size_t>(level_start_[
                static_cast<std::size_t>(level) + 1]);
            const span2d<const std::int64_t> lo = packed_lo();
            const span2d<const std::int64_t> hi = packed_hi();
            for (std::size_t b = first; b < last; ++b) {
                const std::span<const std::int64_t> block_lo = block_row(lo, b);
                const std::span<const std::int64_t> block_hi = block_row(hi, b);
                std::copy(block_lo.begin(), block_lo.end(), midx.begin());
                auto cell = static_cast<std::size_t>(block_base_[b]);
                while (true) {
                    for (std::size_t c = 0; c < corners; ++c) {
                        const std::size_t row = (cell * corners + c) * d;
                        for (std::size_t k = 0; k < d; ++k) {
                            const std::int64_t bit = (static_cast<std::int64_t>(c) >> k) & 1;
                            lattice[row + k] = (midx[k] + bit) * scale[k];
                        }
                    }
                    ++cell;
                    // C-order within the block, last axis fastest, which is the order
                    // flat ids are assigned in.
                    std::size_t k = d;
                    bool carried = true;
                    while (k > 0 && carried) {
                        --k;
                        ++midx[k];
                        if (midx[k] < block_hi[k]) {
                            carried = false;
                        } else {
                            midx[k] = block_lo[k];
                        }
                    }
                    if (carried) {
                        break;
                    }
                }
            }
        }

        CellExport<T> mesh;
        mesh.conn.resize(n_cells * corners);
        const std::vector<std::int64_t> nodes = deduplicate_rows(lattice, d, mesh.conn);
        mesh.num_vertices = static_cast<std::int64_t>(nodes.size() / std::max<std::size_t>(d, 1));
        mesh.points = lattice_to_coords(nodes);
        return mesh;
    }

  private:
    /// Everything the constructor computes, so the mixin's sizes can be passed up.
    ///
    /// A CRTP base is initialised before any member, so `GridBase(ndim, num_cells)`
    /// cannot read a member to find the cell count. `TensorProductGrid` solves this the
    /// same way: one static function computes the whole state, and a private constructor
    /// forwards its two sizes up and moves the rest in.
    struct State {
        TensorProductGrid<T> root;             ///< The level-0 grid.
        std::vector<std::int64_t> factor;      ///< Per-axis subdivision factor.
        std::vector<std::int64_t> block_lo;    ///< `(n_blocks, ndim)` lower corners.
        std::vector<std::int64_t> block_hi;    ///< `(n_blocks, ndim)` upper corners.
        std::vector<std::int64_t> block_base;  ///< Flat id of each block's first cell.
        std::vector<std::int64_t> level_start; ///< Block index range per level, + sentinel.
        std::int64_t num_cells;                ///< Total active cell count.
    };

    /// The same-level position across a facet, and how to reach the fine side of it.
    struct FacetPosition {
        std::int64_t level;                 ///< Level of the queried cell.
        std::vector<std::int64_t> midx;     ///< The neighbour position at that level.
        std::size_t axis;                   ///< The axis normal to the facet.
        std::int64_t face_j;                ///< Child offset on `axis` touching the plane.
    };

    /// Adopt an already-computed state.
    ///
    /// \param state The state to move in.
    explicit HierarchicalGrid(State&& state)
        : Base(state.root.ndim(), state.num_cells),
          root_(std::move(state.root)),
          factor_(std::move(state.factor)),
          block_lo_(std::move(state.block_lo)),
          block_hi_(std::move(state.block_hi)),
          block_base_(std::move(state.block_base)),
          level_start_(std::move(state.level_start)) {}

    /// The state of an unrefined hierarchy: one level, one block over the whole root.
    ///
    /// A function rather than a nested call in the delegating constructor's argument list,
    /// and that is load-bearing rather than tidiness: arguments are evaluated in an
    /// unspecified order, so a call that both moved `root` and read it to size the block
    /// could read a moved-from grid. Statements inside a body are sequenced, so this
    /// cannot.
    ///
    /// \param root The level-0 grid.
    /// \param factor Per-axis subdivision factor.
    /// \return The packed state.
    [[nodiscard]] static State build_unrefined(TensorProductGrid<T> root,
                                               std::span<const std::int64_t> factor) {
        std::vector<BlockList> blocks = single_root_block(root);
        return build(std::move(root), factor, std::move(blocks),
                     std::span<const std::int64_t>{});
    }

    /// One level holding one block spanning the whole root.
    ///
    /// \param root The level-0 grid.
    /// \return A single-level block list.
    [[nodiscard]] static std::vector<BlockList> single_root_block(
        const TensorProductGrid<T>& root) {
        const auto d = static_cast<std::size_t>(root.ndim());
        std::vector<std::int64_t> lo(d, 0);
        const std::span<const std::int64_t> hi = root.cells_per_axis();
        BlockList level0(root.ndim());
        level0.push_back(BlockView{lo, hi});
        return {std::move(level0)};
    }

    /// Validate, normalize where asked, and pack into the flat descriptor.
    ///
    /// \param root The level-0 grid.
    /// \param factor Per-axis subdivision factor.
    /// \param blocks Per-level block lists.
    /// \param unnormalized_levels Levels to re-normalize; empty optional means all.
    /// \return The packed state.
    /// \throws std::invalid_argument If `factor` or a level's rectangle width is wrong.
    /// \throws std::overflow_error If the total cell count exceeds `int64`.
    [[nodiscard]] static State build(
        TensorProductGrid<T> root, std::span<const std::int64_t> factor,
        std::vector<BlockList> blocks,
        std::optional<std::span<const std::int64_t>> unnormalized_levels) {
        const std::int64_t ndim = root.ndim();
        if (static_cast<std::int64_t>(factor.size()) != ndim) {
            throw std::invalid_argument("factor must have " + std::to_string(ndim)
                                        + " entries; got " + std::to_string(factor.size())
                                        + ".");
        }
        for (const std::int64_t f : factor) {
            if (f < 1) {
                throw std::invalid_argument(
                    "every factor entry must be >= 1 (1 = no subdivision on that axis).");
            }
        }
        for (const BlockList& level : blocks) {
            if (level.ndim() != ndim && !level.empty()) {
                throw std::invalid_argument("every block list must span " + std::to_string(ndim)
                                            + " axes.");
            }
        }

        // Normalize the levels the caller did not vouch for. A level is clean exactly
        // when it holds another grid's list verbatim; anything the caller built or
        // clipped is not, and the default assumes nothing.
        for (std::size_t l = 0; l < blocks.size(); ++l) {
            const bool clean =
                unnormalized_levels.has_value()
                && std::find(unnormalized_levels->begin(), unnormalized_levels->end(),
                             static_cast<std::int64_t>(l))
                       == unnormalized_levels->end();
            if (!clean) {
                blocks[l] = normalize_blocks(blocks[l]);
            }
        }
        while (blocks.size() > 1 && blocks.back().empty()) {
            blocks.pop_back();
        }

        State state{std::move(root), {factor.begin(), factor.end()}, {}, {}, {},
                    std::vector<std::int64_t>(blocks.size() + 1, 0), 0};
        std::int64_t cell_base = 0;
        std::int64_t b = 0;
        for (std::size_t l = 0; l < blocks.size(); ++l) {
            state.level_start[l] = b;
            for (std::size_t i = 0; i < blocks[l].size(); ++i) {
                const BlockView block = blocks[l][i];
                state.block_lo.insert(state.block_lo.end(), block.lo.begin(), block.lo.end());
                state.block_hi.insert(state.block_hi.end(), block.hi.begin(), block.hi.end());
                state.block_base.push_back(cell_base);
                cell_base = checked_add(cell_base, checked_block_size(block));
                ++b;
            }
        }
        state.level_start[blocks.size()] = b;
        state.num_cells = cell_base;
        return state;
    }

    // ----------------------------------------------------------------
    // Addressing
    // ----------------------------------------------------------------

    /// The packed lower corners as a two-dimensional view.
    ///
    /// \return `(n_blocks, ndim)` row-major.
    [[nodiscard]] span2d<const std::int64_t> packed_lo() const noexcept {
        return span2d<const std::int64_t>(block_lo_.data(), block_base_.size(),
                                          static_cast<std::size_t>(this->ndim()));
    }

    /// The packed upper corners as a two-dimensional view.
    ///
    /// \return `(n_blocks, ndim)` row-major.
    [[nodiscard]] span2d<const std::int64_t> packed_hi() const noexcept {
        return span2d<const std::int64_t>(block_hi_.data(), block_base_.size(),
                                          static_cast<std::size_t>(this->ndim()));
    }

    /// Flat id of `(level, midx)`, if it is an active leaf.
    ///
    /// \param level Hierarchy level.
    /// \param midx Per-axis index, `ndim()` entries.
    /// \return The flat id, or empty when the position is not an active leaf.
    [[nodiscard]] std::optional<std::int64_t> encode(
        std::int64_t level, std::span<const std::int64_t> midx) const {
        if (level < 0 || level > max_level()) {
            return std::nullopt;
        }
        const std::int64_t cid = block_of_midx(level, midx, packed_lo(), packed_hi(),
                                               block_base_, level_start_);
        if (cid < 0) {
            return std::nullopt;
        }
        return cid;
    }

    /// Level and per-axis index of a flat id.
    ///
    /// \param cid Cell identifier; must be in range.
    /// \param out Receives the index; must have `ndim()` entries.
    /// \return The level.
    [[nodiscard]] std::int64_t decode(std::int64_t cid, std::span<std::int64_t> out) const {
        return decode_flat_id(cid, packed_lo(), packed_hi(), block_base_, level_start_, out);
    }

    /// The corners of the cell at `(level, midx)`.
    ///
    /// \param level Hierarchy level.
    /// \param midx Per-axis index, `ndim()` entries.
    /// \param lo Receives the lower corner.
    /// \param hi Receives the upper corner.
    void bounds_at(std::int64_t level, std::span<const std::int64_t> midx, std::span<T> lo,
                   std::span<T> hi) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        for (std::size_t k = 0; k < d; ++k) {
            const std::int64_t m_pow = int_pow(factor_[k], level);
            const std::int64_t root_ik = midx[k] / m_pow;
            const std::int64_t sub_ik = midx[k] % m_pow;
            const std::span<const T> bp = root_.breakpoints(static_cast<std::int64_t>(k));
            const T root_lo_k = bp[static_cast<std::size_t>(root_ik)];
            const T root_hi_k = bp[static_cast<std::size_t>(root_ik + 1)];
            const T size_k =
                static_cast<T>((root_hi_k - root_lo_k) / T(static_cast<double>(m_pow)));
            // Named rather than inlined; see the file header. A fused multiply-add here
            // would move a corner by an ulp against the oracle, and the same expression
            // in `descend` decides a cell id.
            const T offset = T(static_cast<double>(sub_ik)) * size_k;
            lo[k] = root_lo_k + offset;
            hi[k] = lo[k] + size_k;
        }
    }

    /// Step one level down towards `pt`, updating the index and the bracketing box.
    ///
    /// The scalar counterpart of the loop inside `hier_locate_points`, and written to
    /// match it statement for statement: the clamp, the named product, and the rewrite of
    /// `hi` from the *new* `lo` are each observable through the id the descent returns.
    ///
    /// \param pt The query point.
    /// \param midx Per-axis index, replaced by the child's.
    /// \param lo The current cell's lower corner, replaced by the child's.
    /// \param hi The current cell's upper corner, replaced by the child's.
    void descend(std::span<const T> pt, std::span<std::int64_t> midx, std::span<T> lo,
                 std::span<T> hi) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        for (std::size_t k = 0; k < d; ++k) {
            const std::int64_t fk = factor_[k];
            const T size_k = static_cast<T>((hi[k] - lo[k]) / T(static_cast<double>(fk)));
            auto j = static_cast<std::int64_t>(
                value_of(static_cast<T>((pt[k] - lo[k]) / size_k)));
            if (j < 0) {
                j = 0;
            } else if (j > fk - 1) {
                j = fk - 1;
            }
            const T step = T(static_cast<double>(j)) * size_k;
            lo[k] = lo[k] + step;
            hi[k] = lo[k] + size_k;
            midx[k] = midx[k] * fk + j;
        }
    }

    // ----------------------------------------------------------------
    // Neighbours
    // ----------------------------------------------------------------

    /// Resolve facet `lfid` of `cid` to the same-level position across it.
    ///
    /// \param cid Cell identifier.
    /// \param lfid Local facet identifier.
    /// \return The position, or empty when the facet is on the outer boundary.
    /// \throws std::out_of_range If `cid` or `lfid` is out of range.
    [[nodiscard]] std::optional<FacetPosition> facet_neighbor_position(
        std::int64_t cid, std::int64_t lfid) const {
        this->check_lfid(cid, lfid);
        const auto axis = static_cast<std::size_t>(lfid / 2);
        const std::int64_t side = lfid % 2;

        std::vector<std::int64_t> midx(static_cast<std::size_t>(this->ndim()));
        const std::int64_t level = decode(cid, midx);
        const std::int64_t moved = midx[axis] + (side == 0 ? -1 : 1);
        if (moved < 0
            || moved >= level_cells_per_axis(level, static_cast<std::int64_t>(axis))) {
            return std::nullopt;  // the grid's outer boundary
        }
        midx[axis] = moved;
        const std::int64_t face_j = (side == 0) ? factor_[axis] - 1 : 0;
        return FacetPosition{level, std::move(midx), axis, face_j};
    }

    /// The active leaf covering `(level, midx)` at a strictly coarser level.
    ///
    /// \param level Level of the queried position.
    /// \param midx Per-axis index at that level.
    /// \return The covering leaf's flat id, or empty when no ancestor is active.
    [[nodiscard]] std::optional<std::int64_t> nearest_active_ancestor(
        std::int64_t level, std::span<const std::int64_t> midx) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        std::vector<std::int64_t> ancestor(midx.begin(), midx.end());
        for (std::int64_t lvl = level - 1; lvl >= 0; --lvl) {
            for (std::size_t k = 0; k < d; ++k) {
                ancestor[k] /= factor_[k];
            }
            const std::optional<std::int64_t> cid = encode(lvl, ancestor);
            if (cid) {
                return cid;
            }
        }
        return std::nullopt;
    }

    /// The active leaves inside `(level, midx)` that touch one of its faces.
    ///
    /// Descends the subdivision tree, keeping only the children on the facet plane and
    /// enumerating the remaining axes in C-order, so the ids come out ordered along the
    /// face. Stops at `(level, midx)` itself when that is an active leaf.
    ///
    /// \param level Level of the starting position.
    /// \param midx Per-axis index at that level.
    /// \param axis The axis normal to the face.
    /// \param face_j Child offset on `axis` adjacent to the facet plane.
    /// \param out Receives the ids, appended. Not cleared first.
    void active_face_descendants(std::int64_t level, std::span<const std::int64_t> midx,
                                 std::size_t axis, std::int64_t face_j,
                                 std::vector<std::int64_t>& out) const {
        const std::optional<std::int64_t> cid = encode(level, midx);
        if (cid) {
            out.push_back(*cid);
            return;
        }
        if (level + 1 > max_level()) {
            return;
        }
        const auto d = static_cast<std::size_t>(this->ndim());
        // The odometer runs over every axis but `axis`, which is pinned to `face_j`.
        std::vector<std::int64_t> offsets(d, 0);
        offsets[axis] = face_j;
        std::vector<std::int64_t> child(d);
        while (true) {
            for (std::size_t k = 0; k < d; ++k) {
                child[k] = midx[k] * factor_[k] + offsets[k];
            }
            active_face_descendants(level + 1, child, axis, face_j, out);

            std::size_t k = d;
            bool carried = true;
            while (k > 0 && carried) {
                --k;
                if (k == axis) {
                    continue;  // pinned to the face
                }
                ++offsets[k];
                if (offsets[k] < factor_[k]) {
                    carried = false;
                } else {
                    offsets[k] = 0;
                }
            }
            if (carried) {
                return;
            }
        }
    }


    // ----------------------------------------------------------------
    // Rebuilding the active set
    // ----------------------------------------------------------------

    /// The active-leaf rectangles at `level`, as an owned list.
    ///
    /// \param level Hierarchy level in `[0, max_level()]`.
    /// \return A fresh list holding that level's rectangles in their stored order.
    [[nodiscard]] BlockList level_blocks(std::int64_t level) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        const auto l = static_cast<std::size_t>(level);
        const auto first = static_cast<std::size_t>(level_start_[l]);
        const auto last = static_cast<std::size_t>(level_start_[l + 1]);
        BlockList out(this->ndim());
        out.reserve(last - first);
        for (std::size_t b = first; b < last; ++b) {
            out.push_back(
                BlockView{std::span<const std::int64_t>(block_lo_.data() + b * d, d),
                          std::span<const std::int64_t>(block_hi_.data() + b * d, d)});
        }
        return out;
    }

    /// Every level's rectangles, in level order.
    ///
    /// \return `max_level() + 1` lists, each in its stored order.
    [[nodiscard]] std::vector<BlockList> all_level_blocks() const {
        const std::int64_t n_levels = max_level() + 1;
        std::vector<BlockList> levels;
        levels.reserve(static_cast<std::size_t>(n_levels));
        for (std::int64_t level = 0; level < n_levels; ++level) {
            levels.push_back(level_blocks(level));
        }
        return levels;
    }

    /// An independent grid over the same active set, with cold caches.
    ///
    /// Every operation that changes nothing returns this rather than a copy of `*this`,
    /// so no result ever aliases its receiver and every result carries the empty BVH and
    /// tag registries `refine` and `coarsen` promise. The copy constructor would not do:
    /// it carries this grid's two tag registries across, which is exactly the difference.
    ///
    /// The dirty-level list is **empty rather than absent**: every level holds this
    /// grid's own already-normalized blocks, so re-running the merge is the identity and
    /// skipping it is a pure cost saving.
    ///
    /// \return The rebuilt grid.
    [[nodiscard]] HierarchicalGrid rebuilt() const {
        return from_blocks(root_, factor_, all_level_blocks(),
                           std::span<const std::int64_t>{});
    }

    /// Reject a region that is empty, mis-shaped, or outside its level's domain.
    ///
    /// \param level The level the region is expressed in.
    /// \param lo Per-axis start index, inclusive.
    /// \param hi Per-axis end index, exclusive.
    /// \throws std::invalid_argument If either corner is not `ndim()` long, some
    ///         `lo[k] >= hi[k]`, or the region leaves `[0, level_cells_per_axis)`.
    /// \throws std::overflow_error If the level's cell count exceeds `int64`.
    void check_region(std::int64_t level, std::span<const std::int64_t> lo,
                      std::span<const std::int64_t> hi) const {
        require_ndim_span(lo, "lo");
        require_ndim_span(hi, "hi");
        const auto d = static_cast<std::size_t>(this->ndim());
        for (std::size_t k = 0; k < d; ++k) {
            if (lo[k] >= hi[k]) {
                throw std::invalid_argument(
                    "lo must be strictly less than hi in every dimension; got lo="
                    + tuple_repr(lo) + ", hi=" + tuple_repr(hi) + ".");
            }
        }
        for (std::size_t k = 0; k < d; ++k) {
            const std::int64_t n_k = level_cells_per_axis(level, static_cast<std::int64_t>(k));
            if (lo[k] < 0 || hi[k] > n_k) {
                throw std::invalid_argument(
                    "[lo, hi) out of bounds at level " + std::to_string(level) + ": axis "
                    + std::to_string(k) + " needs [0, " + std::to_string(n_k) + "), got ["
                    + std::to_string(lo[k]) + ", " + std::to_string(hi[k]) + ").");
            }
        }
    }

    // ----------------------------------------------------------------
    // Row sets, used by `coarsen_cells`
    // ----------------------------------------------------------------

    /// Sort fixed-width integer rows lexicographically and drop duplicates, in place.
    ///
    /// \param rows The rows, held flat; replaced by the sorted unique ones.
    /// \param width Entries per row; must divide `rows.size()` and be `>= 1`.
    static void sort_unique_rows(std::vector<std::int64_t>& rows, std::size_t width) {
        const std::size_t n = rows.size() / width;
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&rows, width](std::size_t i, std::size_t j) {
            for (std::size_t k = 0; k < width; ++k) {
                if (rows[i * width + k] != rows[j * width + k]) {
                    return rows[i * width + k] < rows[j * width + k];
                }
            }
            return false;
        });
        std::vector<std::int64_t> unique;
        unique.reserve(rows.size());
        for (const std::size_t i : order) {
            const auto first = rows.begin() + static_cast<std::ptrdiff_t>(i * width);
            const auto last = first + static_cast<std::ptrdiff_t>(width);
            if (unique.empty()
                || !std::equal(first, last,
                               unique.end() - static_cast<std::ptrdiff_t>(width))) {
                unique.insert(unique.end(), first, last);
            }
        }
        rows = std::move(unique);
    }

    /// Whether a sorted unique row set holds `(head, tail...)`.
    ///
    /// \param rows Sorted unique rows, held flat.
    /// \param width Entries per row.
    /// \param head The row's first entry.
    /// \param tail The row's remaining `width - 1` entries.
    /// \return `true` when the row is present.
    [[nodiscard]] static bool row_present(const std::vector<std::int64_t>& rows,
                                          std::size_t width, std::int64_t head,
                                          std::span<const std::int64_t> tail) {
        std::size_t low = 0;
        std::size_t high = rows.size() / width;
        while (low < high) {
            const std::size_t mid = low + (high - low) / 2;
            const std::size_t base = mid * width;
            std::size_t k = 0;
            while (k < width && rows[base + k] == (k == 0 ? head : tail[k - 1])) {
                ++k;
            }
            if (k == width) {
                return true;
            }
            if (rows[base + k] < (k == 0 ? head : tail[k - 1])) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return false;
    }

    /// Whether every child of a parent cell is both named and an active leaf.
    ///
    /// \param current The grid the demotions applied so far produced.
    /// \param marked Sorted unique `(level, midx...)` rows the caller named.
    /// \param parent_level The parent's level.
    /// \param pmidx The parent's index at that level, `ndim()` entries.
    /// \param child Scratch of `ndim()` entries; contents on return are unspecified.
    /// \return `true` when all `prod(factor)` children qualify.
    [[nodiscard]] bool family_is_complete(const HierarchicalGrid& current,
                                          const std::vector<std::int64_t>& marked,
                                          std::int64_t parent_level,
                                          std::span<const std::int64_t> pmidx,
                                          std::vector<std::int64_t>& child) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        for (std::size_t k = 0; k < d; ++k) {
            child[k] = checked_mul(pmidx[k], factor_[k]);
        }
        while (true) {
            // The active-leaf test cannot fail today once a child is named: every flat id
            // names an active leaf, and the only call that could destroy one is this
            // parent's own. It stays because that is an argument about the caller rather
            // than a property of this loop, and it is what the documented rule says.
            if (!row_present(marked, d + 1, parent_level + 1, child)
                || !current.is_active_leaf(parent_level + 1, child)) {
                return false;
            }
            std::size_t k = d;
            bool carried = true;
            while (k > 0 && carried) {
                --k;
                ++child[k];
                if (child[k] < checked_mul(pmidx[k] + 1, factor_[k])) {
                    carried = false;
                } else {
                    child[k] = checked_mul(pmidx[k], factor_[k]);
                }
            }
            if (carried) {
                return true;
            }
        }
    }

    // ----------------------------------------------------------------
    // Why a coarsening was refused
    // ----------------------------------------------------------------

    /// The clause `coarsen` appends to its refusal, naming the cells that caused it.
    ///
    /// \param level Level whose cells were to be reactivated.
    /// \param lo Region lower bound, inclusive, in level-`level` coordinates.
    /// \param hi Region upper bound, exclusive.
    /// \return A leading-space clause, or the empty string when there is nothing to name.
    [[nodiscard]] std::string coarsen_refusal_detail(std::int64_t level,
                                                     std::span<const std::int64_t> lo,
                                                     std::span<const std::int64_t> hi) const {
        const std::int64_t region_cells = checked_block_size(BlockView{lo, hi});
        if (region_cells > kMaxDiagnosedCells) {
            return " The region spans " + std::to_string(region_cells)
                   + " cells, too many to name.";
        }
        const std::string obstacles = coarsen_obstacles(level, lo, hi);
        if (obstacles.empty()) {
            return "";
        }
        return " Offending cells in level-" + std::to_string(level) + " indices: " + obstacles
               + ".";
    }

    /// Name the cells of `[lo, hi)` that stop it being demoted from `level + 1`.
    ///
    /// A cell blocks `coarsen` in exactly one of three ways: it is still an active leaf
    /// at `level`, so it has no children to remove; it is refined past `level + 1`, so
    /// its children are not leaves either; or it is absent at `level` because a coarser
    /// active leaf covers it. Each class is read off the block lists -- active leaves
    /// from `level`, hidden cells from the coarser levels scaled up, over-refined cells
    /// from the deeper levels scaled down.
    ///
    /// \param level Level whose cells were to be reactivated.
    /// \param lo Region lower bound, inclusive.
    /// \param hi Region upper bound, exclusive.
    /// \return Semicolon-separated clauses, each a reason and its cells; empty when every
    ///         cell of the region is refined to exactly `level + 1`, which is when
    ///         `coarsen` accepts it.
    /// \note Reached only from `coarsen`'s failing path, so it favours a precise message
    ///       over speed and allocates one region-shaped mask per reason. The caller keeps
    ///       the region within `kMaxDiagnosedCells`, so those stay small.
    [[nodiscard]] std::string coarsen_obstacles(std::int64_t level,
                                                std::span<const std::int64_t> lo,
                                                std::span<const std::int64_t> hi) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        std::vector<std::int64_t> shape(d);
        for (std::size_t k = 0; k < d; ++k) {
            shape[k] = hi[k] - lo[k];
        }
        const auto size = static_cast<std::size_t>(checked_block_size(BlockView{lo, hi}));
        std::vector<std::uint8_t> still_leaf(size, 0U);
        std::vector<std::uint8_t> over_refined(size, 0U);
        std::vector<std::uint8_t> hidden(size, 0U);
        std::vector<std::int64_t> scaled_lo(d);
        std::vector<std::int64_t> scaled_hi(d);

        const BlockList leaves = level_blocks(level);
        for (std::size_t b = 0; b < leaves.size(); ++b) {
            mark_region(still_leaf, shape, lo, hi, leaves[b]);
        }
        for (std::int64_t coarser = 0; coarser < level; ++coarser) {
            const BlockList blocks = level_blocks(coarser);
            for (std::size_t b = 0; b < blocks.size(); ++b) {
                const BlockView block = blocks[b];
                for (std::size_t k = 0; k < d; ++k) {
                    const std::int64_t up = int_pow(factor_[k], level - coarser);
                    scaled_lo[k] = checked_mul(block.lo[k], up);
                    scaled_hi[k] = checked_mul(block.hi[k], up);
                }
                mark_region(hidden, shape, lo, hi, BlockView{scaled_lo, scaled_hi});
            }
        }
        for (std::int64_t deeper = level + 2; deeper <= max_level(); ++deeper) {
            const BlockList blocks = level_blocks(deeper);
            for (std::size_t b = 0; b < blocks.size(); ++b) {
                const BlockView block = blocks[b];
                for (std::size_t k = 0; k < d; ++k) {
                    // Floor the start and ceil the end: a level-`level` cell holding any
                    // part of a deeper block has a descendant below `level + 1`.
                    const std::int64_t down = int_pow(factor_[k], deeper - level);
                    scaled_lo[k] = block.lo[k] / down;
                    scaled_hi[k] = checked_add(block.hi[k], down - 1) / down;
                }
                mark_region(over_refined, shape, lo, hi, BlockView{scaled_lo, scaled_hi});
            }
        }

        std::string detail;
        const auto add = [&detail, &shape, lo](const std::vector<std::uint8_t>& mask,
                                               const std::string& reason) {
            if (std::find(mask.begin(), mask.end(), std::uint8_t{1}) == mask.end()) {
                return;
            }
            if (!detail.empty()) {
                detail += "; ";
            }
            detail += reason + ": " + name_marked_cells(mask, shape, lo);
        };
        add(still_leaf, "still active leaves at level " + std::to_string(level)
                            + ", with no children to remove");
        add(over_refined, "refined beyond level " + std::to_string(level + 1));
        add(hidden, "covered by a coarser active leaf and absent at level "
                        + std::to_string(level));
        return detail;
    }

    /// Mark the cells of `block` that fall inside the region `[region_lo, region_hi)`.
    ///
    /// \param mask Region-shaped mask, written in place.
    /// \param shape Per-axis extents of the mask.
    /// \param region_lo Region lower bound, inclusive.
    /// \param region_hi Region upper bound, exclusive.
    /// \param block The rectangle to mark, in the region's own coordinates. One that
    ///        misses the region leaves the mask untouched.
    static void mark_region(std::vector<std::uint8_t>& mask,
                            std::span<const std::int64_t> shape,
                            std::span<const std::int64_t> region_lo,
                            std::span<const std::int64_t> region_hi, BlockView block) {
        const std::size_t d = shape.size();
        std::vector<std::int64_t> cut_lo(d);
        std::vector<std::int64_t> cut_hi(d);
        if (!rect_intersect(block, BlockView{region_lo, region_hi}, cut_lo, cut_hi)) {
            return;
        }
        for (std::size_t k = 0; k < d; ++k) {
            cut_lo[k] -= region_lo[k];
            cut_hi[k] -= region_lo[k];
        }
        fill_box(mask, shape, cut_lo, cut_hi, 1U);
    }

    /// Render a region-shaped mask's marked cells as absolute index tuples.
    ///
    /// At most `kMaxNamedCells` are spelled out and the rest is summarised as a count, so
    /// a large rejected region still yields a readable message.
    ///
    /// \param mask Region-shaped mask of cells to name.
    /// \param shape Per-axis extents of the mask.
    /// \param origin Index of the region's first cell, added to every mask index so the
    ///        names come out in the level's own coordinates.
    /// \return Comma-separated index tuples, followed by `and N more` when truncated.
    [[nodiscard]] static std::string name_marked_cells(const std::vector<std::uint8_t>& mask,
                                                       std::span<const std::int64_t> shape,
                                                       std::span<const std::int64_t> origin) {
        const std::size_t d = shape.size();
        std::vector<std::int64_t> absolute(d);
        std::string named;
        std::int64_t total = 0;
        for (std::size_t flat = 0; flat < mask.size(); ++flat) {
            if (mask[flat] == 0U) {
                continue;
            }
            if (total < kMaxNamedCells) {
                std::size_t rest = flat;
                for (std::size_t k = d; k > 0; --k) {
                    const auto extent = static_cast<std::size_t>(shape[k - 1]);
                    absolute[k - 1] = static_cast<std::int64_t>(rest % extent) + origin[k - 1];
                    rest /= extent;
                }
                if (!named.empty()) {
                    named += ", ";
                }
                named += tuple_repr(absolute);
            }
            ++total;
        }
        const std::int64_t rest = total - kMaxNamedCells;
        return rest > 0 ? named + " and " + std::to_string(rest) + " more" : named;
    }

    // ----------------------------------------------------------------
    // The mesh export
    // ----------------------------------------------------------------

    /// Number the distinct rows of a flat row-major integer table.
    ///
    /// Reproduces `numpy.unique(rows, axis=0, return_inverse=True)`: the distinct rows
    /// come back in ascending lexicographic order, and `inverse[i]` is the position of
    /// row `i` among them. Verified against numpy that the ordering is numeric and
    /// lexicographic rather than bytewise, which is what a `void` view would have given.
    ///
    /// \param rows The table, `n * width` entries.
    /// \param width Entries per row, `>= 1`.
    /// \param inverse Receives one index per row; must already hold `n` entries.
    /// \return The distinct rows, held flat in ascending lexicographic order.
    [[nodiscard]] static std::vector<std::int64_t> deduplicate_rows(
        const std::vector<std::int64_t>& rows, std::size_t width,
        std::vector<std::int64_t>& inverse) {
        const std::size_t n = rows.size() / width;
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&rows, width](std::size_t i, std::size_t j) {
            for (std::size_t k = 0; k < width; ++k) {
                if (rows[i * width + k] != rows[j * width + k]) {
                    return rows[i * width + k] < rows[j * width + k];
                }
            }
            return false;
        });
        std::vector<std::int64_t> unique;
        std::int64_t current = -1;
        for (const std::size_t i : order) {
            const auto first = rows.begin() + static_cast<std::ptrdiff_t>(i * width);
            const auto last = first + static_cast<std::ptrdiff_t>(width);
            if (current < 0
                || !std::equal(first, last,
                               unique.end() - static_cast<std::ptrdiff_t>(width))) {
                ++current;
                unique.insert(unique.end(), first, last);
            }
            inverse[i] = current;
        }
        return unique;
    }

    /// Map finest-level integer lattice coordinates to parametric ones.
    ///
    /// The lattice has `factor[k] ** max_level()` steps per root cell on axis `k`, so
    /// every active-leaf corner at every level lands on an integer node. **One formula
    /// serves all of them**, which is what makes coincident corners map to identical
    /// floats and lets `export_cells` deduplicate without a tolerance.
    ///
    /// \param nodes `(n, ndim())` lattice coordinates, each in
    ///        `[0, level_cells_per_axis(max_level(), k)]`.
    /// \return `(n, ndim())` parametric coordinates, row-major.
    /// \throws std::overflow_error If the finest level's cell count exceeds `int64`.
    [[nodiscard]] std::vector<T> lattice_to_coords(
        const std::vector<std::int64_t>& nodes) const {
        const auto d = static_cast<std::size_t>(this->ndim());
        const std::size_t n = nodes.size() / d;
        std::vector<T> points(nodes.size());
        for (std::size_t k = 0; k < d; ++k) {
            const std::int64_t steps = int_pow(factor_[k], max_level());
            const std::int64_t n_root = root_.cells_per_axis()[k];
            const std::int64_t far_edge = checked_mul(steps, n_root);
            const std::span<const T> bp = root_.breakpoints(static_cast<std::int64_t>(k));
            for (std::size_t i = 0; i < n; ++i) {
                const std::int64_t coord = nodes[i * d + k];
                // Clamp so the domain's far edge borrows the last root cell; its exact
                // breakpoint is written below rather than reconstructed by arithmetic.
                const std::int64_t root_cell = std::min(coord / steps, n_root - 1);
                // `root_cell * steps` needs no guard of its own: `root_cell <= n_root - 1`
                // by the clamp above, so the product is below `far_edge`, which was
                // checked. Same shape of exemption as the one `export_cells` spells out
                // for its inner loop, and named here for the same reason -- the file's
                // rule is that every count that could exceed `int64` is checked, so an
                // unchecked one has to say why it cannot.
                const std::int64_t offset = coord - root_cell * steps;
                const auto cell = static_cast<std::size_t>(root_cell);
                const T root_lo = bp[cell];
                const T width = static_cast<T>(bp[cell + 1] - root_lo);
                // Both products are named rather than inlined, for the reason the file
                // header gives: as one expression this is contractible into a fused
                // multiply-add, `-ffp-contract=on` permits exactly that, and the oracle
                // -- numpy, evaluating `offset * (widths / steps)` -- never fuses.
                const T step = static_cast<T>(width / T(static_cast<double>(steps)));
                const T shift = T(static_cast<double>(offset)) * step;
                points[i * d + k] = coord == far_edge
                                        ? bp[static_cast<std::size_t>(n_root)]
                                        : static_cast<T>(root_lo + shift);
            }
        }
        return points;
    }

    // ----------------------------------------------------------------
    // Small shared helpers
    // ----------------------------------------------------------------

    /// One row of a packed block view, as a span.
    ///
    /// \param view The packed view.
    /// \param b Row index.
    /// \return A view of the row's `ndim` entries.
    [[nodiscard]] static std::span<const std::int64_t> block_row(
        span2d<const std::int64_t> view, std::size_t b) noexcept {
        return std::span<const std::int64_t>(&view(b, 0), view.extent(1));
    }

    /// Reject a level outside the hierarchy.
    ///
    /// \param level The level to check.
    /// \throws std::invalid_argument If `level` is outside `[0, max_level()]`.
    void check_level(std::int64_t level) const {
        if (level < 0 || level > max_level()) {
            throw std::invalid_argument("level must be in [0, " + std::to_string(max_level())
                                        + "]; got " + std::to_string(level) + ".");
        }
    }

    /// Reject a span that is not `ndim()` long.
    ///
    /// \tparam U The span's element type.
    /// \param s The span to check.
    /// \param what Its name, for the message.
    /// \throws std::invalid_argument If the length is wrong.
    template <class U>
    void require_ndim_span(std::span<U> s, const char* what) const {
        if (s.size() != static_cast<std::size_t>(this->ndim())) {
            throw std::invalid_argument(std::string(what) + " must have "
                                        + std::to_string(this->ndim()) + " entries; got "
                                        + std::to_string(s.size()) + ".");
        }
    }

    /// The per-axis cell count of a level, as a shape.
    ///
    /// \param level Hierarchy level.
    /// \return `ndim()` counts.
    [[nodiscard]] std::vector<std::int64_t> level_shape(std::int64_t level) const {
        std::vector<std::int64_t> shape(static_cast<std::size_t>(this->ndim()));
        for (std::size_t k = 0; k < shape.size(); ++k) {
            shape[k] = level_cells_per_axis(level, static_cast<std::int64_t>(k));
        }
        return shape;
    }

    /// The number of cells a shape holds.
    ///
    /// \param shape Per-axis counts.
    /// \return Their product.
    /// \throws std::overflow_error If the product exceeds `int64`.
    [[nodiscard]] static std::int64_t level_count(std::span<const std::int64_t> shape) {
        std::int64_t total = 1;
        for (const std::int64_t n : shape) {
            total = checked_mul(total, n);
        }
        return total;
    }

    /// Write `value` into every cell of `[lo, hi)` of a C-order mask over `shape`.
    ///
    /// \param mask The mask to write into.
    /// \param shape Per-axis extents of the mask.
    /// \param lo Lower corner, inclusive.
    /// \param hi Upper corner, exclusive.
    /// \param value The value to write.
    static void fill_box(std::vector<std::uint8_t>& mask,
                         std::span<const std::int64_t> shape,
                         std::span<const std::int64_t> lo, std::span<const std::int64_t> hi,
                         std::uint8_t value) {
        const std::size_t d = shape.size();
        std::vector<std::int64_t> index(lo.begin(), lo.end());
        for (std::size_t k = 0; k < d; ++k) {
            if (lo[k] >= hi[k]) {
                return;
            }
        }
        while (true) {
            std::int64_t flat = 0;
            for (std::size_t k = 0; k < d; ++k) {
                flat = flat * shape[k] + index[k];
            }
            mask[static_cast<std::size_t>(flat)] = value;

            std::size_t k = d;
            bool carried = true;
            while (k > 0 && carried) {
                --k;
                ++index[k];
                if (index[k] < hi[k]) {
                    carried = false;
                } else {
                    index[k] = lo[k];
                }
            }
            if (carried) {
                return;
            }
        }
    }

    /// Append `(cid, lfid)` for every cell of a block's boundary layer, in C-order.
    ///
    /// \param base Flat id of the block's first cell.
    /// \param block_lo The block's lower corner.
    /// \param block_hi The block's upper corner.
    /// \param face_lo Lower corner of the layer, inside the block.
    /// \param face_hi Upper corner of the layer.
    /// \param lfid The local facet id to emit.
    /// \param out Receives the pairs, appended.
    static void emit_face(std::int64_t base, std::span<const std::int64_t> block_lo,
                          std::span<const std::int64_t> block_hi,
                          std::span<const std::int64_t> face_lo,
                          std::span<const std::int64_t> face_hi, std::int64_t lfid,
                          std::vector<std::int64_t>& out) {
        const std::size_t d = block_lo.size();
        std::vector<std::int64_t> index(face_lo.begin(), face_lo.end());
        while (true) {
            // C-order offset within the block, last axis fastest -- the order flat ids
            // are assigned in.
            std::int64_t offset = 0;
            for (std::size_t k = 0; k < d; ++k) {
                offset = offset * (block_hi[k] - block_lo[k]) + (index[k] - block_lo[k]);
            }
            out.push_back(base + offset);
            out.push_back(lfid);

            std::size_t k = d;
            bool carried = true;
            while (k > 0 && carried) {
                --k;
                ++index[k];
                if (index[k] < face_hi[k]) {
                    carried = false;
                } else {
                    index[k] = face_lo[k];
                }
            }
            if (carried) {
                return;
            }
        }
    }

    /// Sort `(cid, lfid)` pairs held flat, lexicographically.
    ///
    /// \param rows `2 * n` values, `n` pairs.
    static void sort_facet_rows(std::vector<std::int64_t>& rows) {
        const std::size_t n = rows.size() / 2;
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&rows](std::size_t i, std::size_t j) {
            if (rows[2 * i] != rows[2 * j]) {
                return rows[2 * i] < rows[2 * j];
            }
            return rows[2 * i + 1] < rows[2 * j + 1];
        });
        std::vector<std::int64_t> sorted(rows.size());
        for (std::size_t i = 0; i < n; ++i) {
            sorted[2 * i] = rows[2 * order[i]];
            sorted[2 * i + 1] = rows[2 * order[i] + 1];
        }
        rows = std::move(sorted);
    }

    /// `base ** exponent`, by repeated multiplication as the oracle does.
    ///
    /// \param base The base; non-negative.
    /// \param exponent The exponent; non-negative.
    /// \return The power.
    /// \throws std::overflow_error If it exceeds `int64`.
    [[nodiscard]] static std::int64_t int_pow(std::int64_t base, std::int64_t exponent) {
        std::int64_t result = 1;
        for (std::int64_t i = 0; i < exponent; ++i) {
            result = checked_mul(result, base);
        }
        return result;
    }

    /// `value * base ** exponent`, checked.
    ///
    /// \param value The multiplicand.
    /// \param base The base.
    /// \param exponent The exponent.
    /// \return The product.
    /// \throws std::overflow_error If it exceeds `int64`.
    [[nodiscard]] static std::int64_t checked_scale(std::int64_t value, std::int64_t base,
                                                    std::int64_t exponent) {
        return checked_mul(value, int_pow(base, exponent));
    }

    /// A block's cell count, refusing to overflow.
    ///
    /// `blocks.hpp`'s `block_size` accumulates in `std::int64_t` without checking, and
    /// says the guard belongs one level up. This is that level. Checking the product as
    /// well as the running sum matters because the file's contract promises that an
    /// invalid decomposition is a *wrong answer* and not undefined behaviour, and a
    /// single block whose own extents multiply past `int64` would otherwise be signed
    /// overflow -- a narrower promise than the one written down.
    ///
    /// \param block The rectangle.
    /// \return Its cell count.
    /// \throws std::overflow_error If the product exceeds `int64`.
    [[nodiscard]] static std::int64_t checked_block_size(BlockView block) {
        std::int64_t size = 1;
        for (std::size_t k = 0; k < block.lo.size(); ++k) {
            size = checked_mul(size, block.hi[k] - block.lo[k]);
        }
        return size;
    }

    /// Multiply, refusing to overflow.
    ///
    /// Checked because the alternative is undefined behaviour rather than a wrong
    /// number, and because the oracle multiplies in Python integers and never reaches it.
    ///
    /// \param a First factor; non-negative.
    /// \param b Second factor; non-negative.
    /// \return `a * b`.
    /// \throws std::overflow_error If the product exceeds `int64`.
    [[nodiscard]] static std::int64_t checked_mul(std::int64_t a, std::int64_t b) {
        if (a != 0 && b > std::numeric_limits<std::int64_t>::max() / a) {
            throw std::overflow_error(
                "HierarchicalGrid: a cell count exceeds what int64 can hold, so the grid "
                "has no representable size.");
        }
        return a * b;
    }

    /// Add, refusing to overflow.
    ///
    /// \param a First term; non-negative.
    /// \param b Second term; non-negative.
    /// \return `a + b`.
    /// \throws std::overflow_error If the sum exceeds `int64`.
    [[nodiscard]] static std::int64_t checked_add(std::int64_t a, std::int64_t b) {
        if (b > std::numeric_limits<std::int64_t>::max() - a) {
            throw std::overflow_error(
                "HierarchicalGrid: the active cells exceed what int64 can count.");
        }
        return a + b;
    }

    /// Format a span of integers the way Python prints a tuple.
    ///
    /// \param values The values.
    /// \return `"(a, b)"`, with the trailing comma Python gives a one-element tuple.
    [[nodiscard]] static std::string tuple_repr(std::span<const std::int64_t> values) {
        std::string out = "(";
        for (std::size_t i = 0; i < values.size(); ++i) {
            out += std::to_string(values[i]);
            if (i + 1 < values.size()) {
                out += ", ";
            }
        }
        if (values.size() == 1) {
            out += ",";
        }
        return out + ")";
    }

    TensorProductGrid<T> root_;              ///< The level-0 grid.
    std::vector<std::int64_t> factor_;       ///< Per-axis subdivision factor.
    std::vector<std::int64_t> block_lo_;     ///< `(n_blocks, ndim)` lower corners.
    std::vector<std::int64_t> block_hi_;     ///< `(n_blocks, ndim)` upper corners.
    std::vector<std::int64_t> block_base_;   ///< Flat id of each block's first cell.
    std::vector<std::int64_t> level_start_;  ///< Block index range per level, + sentinel.
};

}  // namespace pantr::grid
