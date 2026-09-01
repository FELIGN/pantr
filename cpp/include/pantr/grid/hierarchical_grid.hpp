#pragma once

/// \file
/// The hierarchical grid type: a root tensor-product grid plus a per-level set of
/// active-leaf rectangles.
///
/// Ports the query half of `src/pantr/grid/_hierarchical_grid.py`, which stays as the
/// parity oracle. Refinement, coarsening and restriction are not here yet.
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
/// The oracle counts cells in Python's arbitrary-precision integers and cannot overflow;
/// this type accumulates in `std::int64_t` and can. Both the total cell count and
/// `factor ** level` are therefore checked and throw, which is the same trade
/// `TensorProductGrid` already makes for its own cell-count product: a grid that large
/// is unusable on either side, and raising is the smaller divergence than a positive but
/// meaningless count.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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

template <pantr::Real T>
struct pantr::grid::grid_traits<pantr::grid::HierarchicalGrid<T>> {
    /// The coordinate type.
    using scalar_type = T;

    /// The five defaults this grid replaces.
    ///
    /// `cell_level` because a hierarchy has real levels where the generic answer is
    /// always zero; the other four because the packed descriptor answers them without a
    /// per-cell or per-facet query. `restrict` is deliberately absent for now, so the
    /// mixin's throwing default stands; it arrives with refinement.
    static constexpr pantr::grid::Hook hooks =
        pantr::grid::Hook::cell_level | pantr::grid::Hook::boundary_facets
        | pantr::grid::Hook::hanging_neighbors | pantr::grid::Hook::locate_many
        | pantr::grid::Hook::collect_cell_bounds;
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
        for (std::int64_t coarser = 0; coarser < level; ++coarser) {
            const auto [lo, hi] = active_blocks(coarser);
            for (std::size_t b = 0; b < lo.extent(0); ++b) {
                for (std::size_t k = 0; k < d; ++k) {
                    const std::int64_t scale = checked_scale(1, factor_[k], level - coarser);
                    scaled_lo[k] = lo(b, k) * scale;
                    scaled_hi[k] = hi(b, k) * scale;
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
            const auto [lo, hi] = active_blocks(level);
            for (std::size_t b = 0; b < lo.extent(0); ++b) {
                const std::int64_t base = block_base_[
                    static_cast<std::size_t>(level_start_[static_cast<std::size_t>(level)])
                    + b];
                for (std::size_t axis = 0; axis < d; ++axis) {
                    const std::int64_t n_axis = level_cells_per_axis(
                        level, static_cast<std::int64_t>(axis));
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
    /// \return `npts` cell ids; `-1` for a point outside the domain.
    /// \throws std::invalid_argument If `points.extent(1)` is not `ndim()`.
    [[nodiscard]] std::vector<std::int64_t> locate_many(span2d<const T> points) const {
        if (points.extent(1) != static_cast<std::size_t>(this->ndim())) {
            throw std::invalid_argument("points must have ndim() columns.");
        }
        std::vector<std::int64_t> out(points.extent(0));
        hier_locate_points<T>(points, root_.breakpoints_flat(), root_.axis_starts(),
                              root_.cells_per_axis(), factor_, packed_lo(), packed_hi(),
                              block_base_, level_start_, out);
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
                cell_base = checked_add(cell_base, block_size(block));
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
