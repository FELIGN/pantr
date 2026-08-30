#pragma once

/// \file
/// The generic grid layer: a CRTP mixin holding its own state, dispatching by name
/// hiding, and carrying no vtable at all.
///
/// The reasoning behind every choice below is `design/grid_hierarchy_port.md`; what
/// follows is the part a reader of this header needs in order to write a grid.
///
/// ## Writing a grid: three things, in this order
///
/// ```cpp
/// template <class T> class MyGrid;                      // 1. forward declaration
///
/// template <class T> struct pantr::grid::grid_traits<MyGrid<T>> {
///     using scalar_type = T;
///     static constexpr Hook hooks = Hook::locate_many;  // 2. the traits
/// };
///
/// template <class T>
/// class MyGrid : public pantr::grid::GridBase<MyGrid<T>> {  // 3. the class
///     ...
/// };
///
/// PANTR_GRID_CENSUS(MyGrid<double>)                     // once per instantiation
/// ```
///
/// `grid_traits` is a separate template rather than a nested typedef because
/// `using scalar_type = typename Derived::scalar_type;` inside a CRTP base is
/// ill-formed: `Derived` is incomplete where the base is instantiated. The primary
/// template is deliberately left undefined, so a grid that forgets its traits is
/// rejected at its own definition rather than at the first default that needs the
/// scalar.
///
/// ## The three primitives
///
/// A grid supplies exactly three members, and `GridLike` pins each to an EXACT
/// member-pointer type rather than to callability:
///
/// ```cpp
/// void cell_bounds(std::int64_t cid, std::span<T> lo, std::span<T> hi) const;
/// std::optional<std::int64_t> locate(std::span<const T> pt) const;
/// std::optional<std::int64_t> neighbor_across_facet(std::int64_t cid,
///                                                   std::int64_t lfid) const;
/// ```
///
/// Exactness is load-bearing and the reason is not obvious. `std::span<T>` converts
/// implicitly to `std::span<const T>`, so a concept written as a callability probe --
/// `requires(std::span<T> out) { g.cell_bounds(cid, out, out); }` -- is satisfied by a
/// `cell_bounds` whose output spans are `std::span<const T>` and which therefore
/// CANNOT WRITE ITS OUTPUT. Measured: such a grid was accepted by g++ 14.4, g++ 10.5,
/// clang++ 18.1 and clang++ 10.0 alike under `-Wall -Wextra -Werror`, compiled, and
/// returned whatever was already in the caller's buffer. Pinning the member-pointer
/// type rejects it on all four.
///
/// `ndim` and `num_cells` are NOT primitives. They are base state, passed up from the
/// derived constructor, which is what lets `cell_tags()` have ordinary const and
/// non-const overloads with no `const_cast` anywhere near them, keeps the BVH cache
/// slot private, and makes `num_cells()` a field read in the hot loops. It assumes a
/// grid's cell count is fixed at construction; FELIGN/pantr#378 makes refinement
/// return a new grid, which is what makes that true for the hierarchical grid too.
///
/// ## Specialisation: name hiding dispatches, the bitmask is checked
///
/// A default is an ordinary non-virtual member of the mixin, so a `Derived` that
/// declares the same name HIDES it, and every call -- from outside, and from inside
/// another default through `self()` -- reaches the specialisation. Nothing reads
/// `grid_traits<G>::hooks` to route a call.
///
/// The bitmask is a DECLARATION, and `PANTR_GRID_CENSUS` checks the class against it
/// in both directions. That is the point: under bitmask dispatch, a hook written but
/// not declared is silently ignored and the default runs, which is a wrong answer with
/// no diagnostic. Here it is a compile error.
///
/// Detection compares member-pointer TYPES, not a `requires` probe:
/// `&D::locate_many` names the mixin's member -- type `R (GridBase<D>::*)(...)` -- when
/// `D` does not redeclare it, and `R (D::*)(...)` when it does. The two are the same
/// type exactly when the hook is absent, and that stays true when the hook's return
/// type is wrong or a parameter is const-qualified, where a `requires` probe answers
/// `true`. Known limit, and it is a loud one: `&D::name` is ill-formed if `D` overloads
/// the name or makes it a template. Both are design errors here.
///
/// The differential oracle a specialisation owes its default is the qualified call:
/// `g.pantr::grid::GridBase<G>::boundary_facets()` reaches the hidden default. A
/// derived class must therefore NOT privately alias its base. This is the C++ analogue
/// of `Grid.boundary_facets(g)` in `tests/test_grid_hierarchical.py`.
///
/// ## What is deliberately not hookable
///
/// `num_local_facets` is fixed at `2 * ndim` because `facet_tags_` is sized `2 * ndim`
/// once, at construction; a grid that changed it would desynchronise the registry from
/// the geometry. The census asserts it is not redeclared. `cell_tags`, `facet_tags` and
/// `cell_bvh` are base-owned state and are called directly rather than through
/// `self()`, so hiding one cannot change what another default sees.
///
/// ## What CRTP costs
///
/// There is no runtime grid type: `std::vector<Grid*>`, a grid chosen from a config
/// file, and a non-template function taking any grid are all unavailable without a
/// hand-written variant. Nothing in pantr does any of these. Every generic algorithm
/// over grids is a template, so it lives in a header and its errors are template
/// errors. What it buys is inlining: measured on the generic `boundary_facets`
/// neighbour loop, CRTP against a virtual base with the dynamic type hidden behind a
/// factory in another translation unit was 15.1x on the neighbour queries and 6.8x on
/// the whole method. `design/grid_hierarchy_port.md` carries the numbers, the machine
/// and the command, and records that a first attempt measured 1.05x because GCC
/// devirtualized the call and compared CRTP against CRTP.
///
/// ## Thread safety
///
/// `cell_bvh()` fills a `mutable std::optional` with no lock, no atomic and no
/// `std::once_flag`. That is the contract the Python oracle documents and no more:
/// concurrent first calls may each build a valid tree and one write wins, costing
/// redundant construction; a caller sharing a grid across threads should call
/// `cell_bvh()` once first. Everything else here is const and shares nothing.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/precondition.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/geometry/aabb.hpp"
#include "pantr/grid/bvh_tree.hpp"
#include "pantr/grid/tags.hpp"
#include "pantr/transform/affine.hpp"

namespace pantr::grid {

// ---------------------------------------------------------------------------
// Traits and the specialisation declaration
// ---------------------------------------------------------------------------

/// The defaults a grid may replace, as a bitmask.
///
/// A grid names the ones it writes in `grid_traits<G>::hooks`. Nothing reads this to
/// dispatch -- name hiding does that. `PANTR_GRID_CENSUS` reads it, and rejects a class
/// that disagrees with its own declaration in either direction.
///
/// Absent by design: `num_local_facets` (fixed at `2 * ndim`, see the file header),
/// and `cell_tags` / `facet_tags` / `cell_bvh` (base-owned state).
enum class Hook : std::uint32_t {
    none = 0,                          ///< Every default is inherited.
    cell_aabb = 1U << 0U,              ///< `cell_aabb`.
    cell_level = 1U << 1U,             ///< `cell_level`.
    child_cells = 1U << 2U,            ///< `child_cells`.
    reference_map = 1U << 3U,          ///< `reference_map`.
    neighbors = 1U << 4U,              ///< `neighbors`.
    restrict = 1U << 5U,               ///< `restrict`.
    local_facet_axis_side = 1U << 6U,  ///< `local_facet_axis_side`.
    local_facet_bounds = 1U << 7U,     ///< `local_facet_bounds`.
    is_mesh_boundary_facet = 1U << 8U, ///< `is_mesh_boundary_facet`.
    boundary_facets = 1U << 9U,        ///< `boundary_facets`.
    hanging_neighbors = 1U << 10U,     ///< `hanging_neighbors`.
    locate_many = 1U << 11U,           ///< `locate_many`.
    query_aabb = 1U << 12U,            ///< `query_aabb`.
    collect_cell_bounds = 1U << 13U,   ///< `collect_cell_bounds`.
};

/// Combine two hook flags.
///
/// \param a Left operand.
/// \param b Right operand.
/// \return The union of the two masks.
[[nodiscard]] constexpr Hook operator|(Hook a, Hook b) noexcept {
    return static_cast<Hook>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

/// Test whether a mask names a hook.
///
/// \param mask The grid's declared hooks.
/// \param hook The hook to look for.
/// \return `true` iff `hook` is set in `mask`.
[[nodiscard]] constexpr bool declares(Hook mask, Hook hook) noexcept {
    return (static_cast<std::uint32_t>(mask) & static_cast<std::uint32_t>(hook)) != 0U;
}

/// What the generic layer needs to know about a grid before the grid is complete.
///
/// Every grid specialises this, supplying `scalar_type` and a `static constexpr Hook
/// hooks`. The primary template is deliberately left UNDEFINED: a grid deriving from
/// `GridBase` without a specialisation is then rejected at its own definition, with
/// `grid_traits` named in the diagnostic, rather than at whichever default first
/// needed the scalar.
///
/// \tparam Derived The grid type.
template <class Derived>
struct grid_traits;

// ---------------------------------------------------------------------------
// GridRestriction
// ---------------------------------------------------------------------------

/// A windowed sub-grid plus the index maps relating it to the grid it came from.
///
/// The C++ counterpart of `pantr.grid.GridRestriction`. Deliberately UNCONSTRAINED in
/// `G`: a constrained `GridRestriction<G>` named as a return type *inside* the grid
/// being defined gives "satisfaction of atomic constraint depends on itself" on both
/// compiler families, and reordering the declarations does not help. The constraint
/// lives on `make_restriction` instead, which also gives the size-agreement check a
/// home.
///
/// Two things are lost by that and are worth stating rather than discovering: the type
/// is nameable for a non-grid, and aggregate initialisation bypasses the factory.
///
/// `in_subset` is `std::uint8_t` rather than `bool` because `std::vector<bool>` is a
/// bit-packed proxy container with no `data()`, which nothing downstream can hand to
/// nanobind or to a span.
///
/// \tparam G The grid type, which is the same concrete type as the grid restricted.
template <class G>
struct GridRestriction {
    G grid;                                          ///< The windowed sub-grid.
    std::vector<std::int64_t> local_to_global_cell;  ///< Sub-grid cell id to this grid's.
    std::vector<std::uint8_t> in_subset;             ///< `1` if requested, `0` if fill.
};

// ---------------------------------------------------------------------------
// The mixin
// ---------------------------------------------------------------------------

/// The generic grid layer: state, the fourteen replaceable defaults, and four that are not.
///
/// Every grid is a `GridBase<itself>`; `GridLike` says exactly that in one line. See the
/// file header for how to write one and for why the dispatch is name hiding.
///
/// \tparam Derived The grid deriving from this mixin.
template <class Derived>
class GridBase {
  public:
    /// The floating-point type this grid's coordinates are stored in.
    using scalar_type = typename grid_traits<Derived>::scalar_type;

    static_assert(Real<scalar_type>, "a grid's scalar_type must satisfy pantr::Real");

    // ----------------------------------------------------------------
    // State
    // ----------------------------------------------------------------

    /// The spatial dimension of the grid.
    ///
    /// \return The number of axes, `>= 1`.
    [[nodiscard]] std::int64_t ndim() const noexcept { return ndim_; }

    /// The number of cells in this (local) grid.
    ///
    /// \return The non-negative cell count.
    [[nodiscard]] std::int64_t num_cells() const noexcept { return num_cells_; }

    // ----------------------------------------------------------------
    // Cell accessors
    // ----------------------------------------------------------------

    /// The axis-aligned bounding box of cell `cid`.
    ///
    /// \param cid Cell identifier.
    /// \return `AABB(lo, hi)` for the cell's corners.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] geometry::AABB<scalar_type> cell_aabb(std::int64_t cid) const {
        const auto n = static_cast<std::size_t>(ndim_);
        std::vector<scalar_type> lo(n);
        std::vector<scalar_type> hi(n);
        self().cell_bounds(cid, std::span<scalar_type>(lo), std::span<scalar_type>(hi));
        return geometry::AABB<scalar_type>(std::span<const scalar_type>(lo),
                                           std::span<const scalar_type>(hi));
    }

    /// The refinement level of cell `cid`.
    ///
    /// Flat grids are all level zero; a hierarchical grid replaces this.
    ///
    /// \param cid Cell identifier.
    /// \return `0`.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] std::int64_t cell_level(std::int64_t cid) const {
        check_cid(cid);
        return 0;
    }

    /// The immediate refinement children of cell `cid`.
    ///
    /// \param cid Cell identifier.
    /// \return An empty vector; a flat grid has no children.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] std::vector<std::int64_t> child_cells(std::int64_t cid) const {
        check_cid(cid);
        return {};
    }

    /// The affine map from the unit cube onto cell `cid`.
    ///
    /// For an axis-aligned cell this is `T(u) = diag(hi - lo) u + lo`.
    ///
    /// \param cid Cell identifier.
    /// \return The push-forward from `[0, 1]^ndim` to the cell.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] transform::AffineTransform<scalar_type> reference_map(std::int64_t cid) const {
        const auto n = static_cast<std::size_t>(ndim_);
        std::vector<scalar_type> lo(n);
        std::vector<scalar_type> hi(n);
        self().cell_bounds(cid, std::span<scalar_type>(lo), std::span<scalar_type>(hi));
        std::vector<scalar_type> matrix(n * n, scalar_type{0});
        for (std::size_t k = 0; k < n; ++k) {
            matrix[(k * n) + k] = hi[k] - lo[k];
        }
        return transform::AffineTransform<scalar_type>(
            span2d<const scalar_type>(matrix.data(), n, n), std::span<const scalar_type>(lo));
    }

    /// The facet-neighbour cell ids of `cid`.
    ///
    /// Collects `neighbor_across_facet` over every local facet and drops the boundary
    /// ones.
    ///
    /// \param cid Cell identifier.
    /// \return The neighbouring cell ids, in local facet order.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] std::vector<std::int64_t> neighbors(std::int64_t cid) const {
        const std::int64_t n_facets = num_local_facets(cid);
        std::vector<std::int64_t> out;
        out.reserve(static_cast<std::size_t>(n_facets));
        for (std::int64_t lfid = 0; lfid < n_facets; ++lfid) {
            if (const std::optional<std::int64_t> nbr = self().neighbor_across_facet(cid, lfid)) {
                out.push_back(*nbr);
            }
        }
        return out;
    }

    // ----------------------------------------------------------------
    // Restriction
    // ----------------------------------------------------------------

    /// The structured sub-grid spanning a subset of cells.
    ///
    /// Restriction is an OPTIONAL grid capability and this default throws, exactly as
    /// `pantr.grid.Grid.restrict` raises `NotImplementedError`. It is a default rather
    /// than a fourth primitive so that the C++ layer does not change the documented
    /// contract of a public method for a reason internal to itself. The usual
    /// objection -- a base advertising an operation not every subtype supports -- does
    /// not bind here, because nobody holds a `GridBase<D>&` polymorphically and so
    /// there is no substitutability to violate.
    ///
    /// A binding owes the Python side the translation to `NotImplementedError`;
    /// nanobind's default translator maps `std::logic_error` to `RuntimeError`.
    ///
    /// \param cell_ids Flat cell identifiers to span.
    /// \return Never returns.
    /// \throws std::logic_error Always.
    [[nodiscard]] GridRestriction<Derived> restrict(std::span<const std::int64_t> cell_ids) const {
        static_cast<void>(cell_ids);
        throw std::logic_error("this grid kind does not support restrict().");
    }

    // ----------------------------------------------------------------
    // Facet accessors
    // ----------------------------------------------------------------

    /// The number of local facets of cell `cid`.
    ///
    /// Always `2 * ndim`, and NOT specialisable: `facet_tags_` is sized `2 * ndim` once
    /// at construction, so a grid that changed this would desynchronise the registry
    /// from the geometry. `PANTR_GRID_CENSUS` asserts no grid redeclares it.
    ///
    /// \param cid Cell identifier.
    /// \return `2 * ndim`.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] std::int64_t num_local_facets(std::int64_t cid) const {
        check_cid(cid);
        return 2 * ndim_;
    }

    /// The `(axis, side)` of local facet `lfid` of cell `cid`.
    ///
    /// Uses the `lfid = 2 * axis + side` encoding, with `side == 0` the low face.
    ///
    /// \param cid Cell identifier.
    /// \param lfid Local facet identifier in `[0, 2 * ndim)`.
    /// \return `(axis, side)`.
    /// \throws std::out_of_range If `cid` or `lfid` is out of range.
    [[nodiscard]] std::pair<std::int64_t, std::int64_t> local_facet_axis_side(
        std::int64_t cid, std::int64_t lfid) const {
        check_lfid(cid, lfid);
        return {lfid / 2, lfid % 2};
    }

    /// The degenerate `(lo, hi)` box of local facet `lfid` of cell `cid`.
    ///
    /// Both corners coincide on the facet's normal axis; the others span the cell.
    ///
    /// \param cid Cell identifier.
    /// \param lfid Local facet identifier in `[0, 2 * ndim)`.
    /// \param lo Output lo corner, length `ndim`.
    /// \param hi Output hi corner, length `ndim`.
    /// \throws std::out_of_range If `cid` or `lfid` is out of range.
    /// \note `lo` and `hi` must both have length `ndim`; a shorter span is indexed out
    ///       of bounds. Checked by `PANTR_PRECONDITION`.
    void local_facet_bounds(std::int64_t cid, std::int64_t lfid, std::span<scalar_type> lo,
                            std::span<scalar_type> hi) const {
        require_corner_spans(lo, hi);
        const auto [axis, side] = self().local_facet_axis_side(cid, lfid);
        self().cell_bounds(cid, lo, hi);
        const auto k = static_cast<std::size_t>(axis);
        if (side == 0) {
            hi[k] = lo[k];
        } else {
            lo[k] = hi[k];
        }
    }

    /// Whether local facet `lfid` of cell `cid` lies on the grid's outer boundary.
    ///
    /// \param cid Cell identifier.
    /// \param lfid Local facet identifier.
    /// \return `true` iff no neighbouring cell shares the facet.
    /// \throws std::out_of_range If `cid` or `lfid` is out of range.
    [[nodiscard]] bool is_mesh_boundary_facet(std::int64_t cid, std::int64_t lfid) const {
        return !self().neighbor_across_facet(cid, lfid).has_value();
    }

    /// Every outer-boundary facet, as `(cid, lfid)` rows.
    ///
    /// Only the outer boundary: a facet shared with any neighbour is excluded, whether
    /// that neighbour is conforming, coarser or finer. Costs
    /// `O(num_cells * 2 * ndim)` neighbour queries, which is why a grid with
    /// exploitable structure replaces it. This is the loop the CRTP shape was measured
    /// on.
    ///
    /// \return `2 * n` values, row-major `(cid, lfid)` pairs, sorted by construction.
    [[nodiscard]] std::vector<std::int64_t> boundary_facets() const {
        std::vector<std::int64_t> rows;
        for (std::int64_t cid = 0; cid < num_cells_; ++cid) {
            const std::int64_t n_facets = num_local_facets(cid);
            for (std::int64_t lfid = 0; lfid < n_facets; ++lfid) {
                if (self().is_mesh_boundary_facet(cid, lfid)) {
                    rows.push_back(cid);
                    rows.push_back(lfid);
                }
            }
        }
        return rows;
    }

    /// Every active cell sharing local facet `lfid` of `cid`.
    ///
    /// For a conforming grid this is `neighbor_across_facet` in a vector. A grid with
    /// hanging nodes, where one coarse face abuts several fine cells, replaces it.
    ///
    /// \param cid Cell identifier.
    /// \param lfid Local facet identifier.
    /// \return The neighbouring cell ids; empty on an outer-boundary facet.
    /// \throws std::out_of_range If `cid` or `lfid` is out of range.
    [[nodiscard]] std::vector<std::int64_t> hanging_neighbors(std::int64_t cid,
                                                              std::int64_t lfid) const {
        const std::optional<std::int64_t> nbr = self().neighbor_across_facet(cid, lfid);
        if (!nbr) {
            return {};
        }
        return {*nbr};
    }

    // ----------------------------------------------------------------
    // Point location and spatial queries
    // ----------------------------------------------------------------

    /// Locate a batch of points, one cell id per point.
    ///
    /// \param points `(npts, ndim)` row-major view of query points.
    /// \return `npts` cell ids; `-1` for a point outside every cell.
    /// \note `points.extent(1)` must equal `ndim`. Checked by `PANTR_PRECONDITION`.
    [[nodiscard]] std::vector<std::int64_t> locate_many(span2d<const scalar_type> points) const {
        const auto n = static_cast<std::size_t>(ndim_);
        PANTR_PRECONDITION(points.extent(1) == n, "points must have ndim columns");
        const std::size_t npts = points.extent(0);
        std::vector<std::int64_t> out(npts);
        for (std::size_t i = 0; i < npts; ++i) {
            const std::span<const scalar_type> pt(&at(points, i, std::size_t{0}), n);
            const std::optional<std::int64_t> cid = self().locate(pt);
            out[i] = cid.value_or(-1);
        }
        return out;
    }

    /// The ids of every cell whose AABB overlaps `box`.
    ///
    /// Backed by `cell_bvh()`. The overlap test is inclusive on every axis, so cells
    /// touching `box` on a face, an edge or a corner are included.
    ///
    /// \param box Query box; must match `ndim`.
    /// \return The overlapping cell ids, unordered.
    /// \throws std::invalid_argument If `box.ndim()` does not match.
    [[nodiscard]] std::vector<std::int64_t> query_aabb(
        const geometry::AABB<scalar_type>& box) const {
        return cell_bvh().query_aabb(box.lo(), box.hi());
    }

    /// The cached BVH over the grid's cell AABBs, built on first use.
    ///
    /// Building materialises `O(num_cells)` node arrays, so it is deferred: a grid that
    /// is never queried never pays for it. Not specialisable, and the cache slot is
    /// private -- nothing can hand it out.
    ///
    /// \return The grid's spatial index.
    /// \warning Not fully thread-safe. Concurrent first calls may each build a valid
    ///          tree and one write wins, costing redundant construction. Call this once
    ///          on the main thread before sharing the grid across threads.
    [[nodiscard]] const BVH<scalar_type>& cell_bvh() const {
        if (!bvh_.has_value()) {
            const auto n = static_cast<std::size_t>(num_cells_);
            const auto d = static_cast<std::size_t>(ndim_);
            std::vector<scalar_type> cell_lo(n * d);
            std::vector<scalar_type> cell_hi(n * d);
            self().collect_cell_bounds(span2d<scalar_type>(cell_lo.data(), n, d),
                                       span2d<scalar_type>(cell_hi.data(), n, d));
            bvh_ = BVH<scalar_type>::from_cell_bounds(
                span2d<const scalar_type>(cell_lo.data(), n, d),
                span2d<const scalar_type>(cell_hi.data(), n, d));
        }
        return *bvh_;
    }

    /// Materialise per-cell `(lo, hi)` into `(num_cells, ndim)` views.
    ///
    /// Iterates `cell_bounds` over every cell in id order. A grid with structure
    /// replaces it with a vectorised construction.
    ///
    /// \param cell_lo Output lo corners, shape `(num_cells, ndim)`.
    /// \param cell_hi Output hi corners, same shape.
    /// \note Both views must have exactly that shape. Checked by `PANTR_PRECONDITION`.
    void collect_cell_bounds(span2d<scalar_type> cell_lo, span2d<scalar_type> cell_hi) const {
        const auto n = static_cast<std::size_t>(num_cells_);
        const auto d = static_cast<std::size_t>(ndim_);
        PANTR_PRECONDITION(cell_lo.extent(0) == n && cell_lo.extent(1) == d,
                           "cell_lo must have shape (num_cells, ndim)");
        PANTR_PRECONDITION(cell_hi.extent(0) == n && cell_hi.extent(1) == d,
                           "cell_hi must have shape (num_cells, ndim)");
        for (std::size_t row = 0; row < n; ++row) {
            self().cell_bounds(static_cast<std::int64_t>(row),
                               std::span<scalar_type>(&at(cell_lo, row, std::size_t{0}), d),
                               std::span<scalar_type>(&at(cell_hi, row, std::size_t{0}), d));
        }
    }

    // ----------------------------------------------------------------
    // Tagging
    // ----------------------------------------------------------------

    /// The grid's sparse cell-tag registry.
    ///
    /// Eager rather than lazy: an empty registry has no per-cell footprint, so laziness
    /// would buy one allocation and cost a `mutable` and a branch.
    ///
    /// \return A mutable reference to the registry, valid for the grid's lifetime.
    [[nodiscard]] CellTags& cell_tags() noexcept { return cell_tags_; }

    /// The grid's sparse cell-tag registry, read only.
    ///
    /// \return A const reference to the registry.
    [[nodiscard]] const CellTags& cell_tags() const noexcept { return cell_tags_; }

    /// The grid's sparse facet-tag registry, sized `2 * ndim` facets per cell.
    ///
    /// \return A mutable reference to the registry, valid for the grid's lifetime.
    [[nodiscard]] FacetTags& facet_tags() noexcept { return facet_tags_; }

    /// The grid's sparse facet-tag registry, read only.
    ///
    /// \return A const reference to the registry.
    [[nodiscard]] const FacetTags& facet_tags() const noexcept { return facet_tags_; }

  protected:
    /// Establish the size metadata and the two tag registries.
    ///
    /// Protected because a `GridBase` is never constructed on its own. `Derived`
    /// computes the two sizes from its own constructor arguments and passes them up;
    /// the mixin cannot read them back out of `self()`, because `Derived` is not yet
    /// constructed when the base's own constructor runs.
    ///
    /// \param ndim Number of axes, `>= 1`.
    /// \param num_cells Number of cells, `>= 0`.
    /// \throws std::invalid_argument If `ndim < 1` or `num_cells < 0`.
    GridBase(std::int64_t ndim, std::int64_t num_cells)
        : ndim_(require_ndim(ndim)),
          num_cells_(require_num_cells(num_cells)),
          cell_tags_(num_cells),
          facet_tags_(num_cells, 2 * ndim) {}

    /// This object as the derived grid.
    ///
    /// \return A const reference to `Derived`.
    [[nodiscard]] const Derived& self() const noexcept {
        return static_cast<const Derived&>(*this);
    }

    /// This object as the derived grid.
    ///
    /// \return A mutable reference to `Derived`.
    [[nodiscard]] Derived& self() noexcept { return static_cast<Derived&>(*this); }

    /// Reject a cell id outside `[0, num_cells)`.
    ///
    /// The message is the Python oracle's verbatim, and the exception type is chosen so
    /// that nanobind's default translator reaches `IndexError` with `what()` preserved.
    ///
    /// \param cid Candidate cell identifier.
    /// \throws std::out_of_range If `cid` is negative or `>= num_cells`.
    void check_cid(std::int64_t cid) const {
        if (cid < 0 || cid >= num_cells_) {
            throw std::out_of_range("cell id " + std::to_string(cid) + " is out of range [0, "
                                    + std::to_string(num_cells_) + ").");
        }
    }

    /// Reject a local facet id that is not a facet of `cid`.
    ///
    /// \param cid Cell identifier, validated first.
    /// \param lfid Candidate local facet identifier.
    /// \throws std::out_of_range If `cid` is out of range, or `lfid` is not in
    ///         `[0, num_local_facets(cid))`.
    void check_lfid(std::int64_t cid, std::int64_t lfid) const {
        const std::int64_t n_facets = num_local_facets(cid);
        if (lfid < 0 || lfid >= n_facets) {
            throw std::out_of_range("local facet id " + std::to_string(lfid)
                                    + " is out of range [0, " + std::to_string(n_facets) + ").");
        }
    }

  private:
    /// Check the spatial dimension on the way into the member initialiser list.
    ///
    /// A free-standing check in the constructor body would run AFTER `facet_tags_` had
    /// already thrown its own, less specific, message for the same bad value.
    ///
    /// \param ndim The candidate dimension.
    /// \return `ndim`.
    /// \throws std::invalid_argument If `ndim < 1`.
    [[nodiscard]] static std::int64_t require_ndim(std::int64_t ndim) {
        if (ndim < 1) {
            throw std::invalid_argument("ndim must be >= 1; got " + std::to_string(ndim) + ".");
        }
        return ndim;
    }

    /// Check the cell count on the way into the member initialiser list.
    ///
    /// \param num_cells The candidate cell count.
    /// \return `num_cells`.
    /// \throws std::invalid_argument If `num_cells < 0`.
    [[nodiscard]] static std::int64_t require_num_cells(std::int64_t num_cells) {
        if (num_cells < 0) {
            throw std::invalid_argument("num_cells must be >= 0; got "
                                        + std::to_string(num_cells) + ".");
        }
        return num_cells;
    }

    /// Assert that two corner spans are both length `ndim`.
    ///
    /// \param lo Candidate lo corner.
    /// \param hi Candidate hi corner.
    void require_corner_spans(std::span<const scalar_type> lo,
                              std::span<const scalar_type> hi) const {
        const auto n = static_cast<std::size_t>(ndim_);
        PANTR_PRECONDITION(lo.size() == n, "lo must have length ndim");
        PANTR_PRECONDITION(hi.size() == n, "hi must have length ndim");
        static_cast<void>(lo);
        static_cast<void>(hi);
        static_cast<void>(n);
    }

    std::int64_t ndim_;                            ///< Number of axes.
    std::int64_t num_cells_;                       ///< Number of cells.
    CellTags cell_tags_;                           ///< The cell-tag registry.
    FacetTags facet_tags_;                         ///< The facet-tag registry.
    mutable std::optional<BVH<scalar_type>> bvh_;  ///< The lazily built spatial index.
};

// ---------------------------------------------------------------------------
// The concept
// ---------------------------------------------------------------------------

namespace detail {

/// The scalar a grid's traits declare, as an alias.
///
/// \tparam D The grid type.
template <class D>
using scalar_t = typename grid_traits<D>::scalar_type;

/// Satisfied when `D` has a `grid_traits` specialisation naming a scalar.
///
/// Written as a type-requirement so it is a substitution failure rather than a hard
/// error, and placed FIRST in `GridLike`'s conjunction so the short-circuit keeps the
/// remaining atomic constraints from being evaluated for a type that has no traits.
template <class D>
concept HasGridTraits = requires { typename grid_traits<D>::scalar_type; };

/// Satisfied when `D::cell_bounds` has EXACTLY the primitive's signature.
///
/// The exactness is what rejects an output span of `std::span<const T>`; see the file
/// header.
template <class D>
concept HasCellBounds = requires {
    {
        &D::cell_bounds
    } -> std::same_as<void (D::*)(std::int64_t, std::span<scalar_t<D>>,
                                  std::span<scalar_t<D>>) const>;
};

/// Satisfied when `D::locate` has exactly the primitive's signature.
template <class D>
concept HasLocate = requires {
    {
        &D::locate
    } -> std::same_as<std::optional<std::int64_t> (D::*)(std::span<const scalar_t<D>>) const>;
};

/// Satisfied when `D::neighbor_across_facet` has exactly the primitive's signature.
template <class D>
concept HasNeighborAcrossFacet = requires {
    {
        &D::neighbor_across_facet
    } -> std::same_as<std::optional<std::int64_t> (D::*)(std::int64_t, std::int64_t) const>;
};

}  // namespace detail

/// A grid: derived from its own `GridBase` and supplying the three primitives.
///
/// "Closed hierarchy" is `std::derived_from<G, GridBase<G>>` and nothing more -- not a
/// variant, not a sealed virtual base. Each primitive is pinned to an exact
/// member-pointer type rather than to callability, which is the only form that rejects
/// a `cell_bounds` unable to write its output.
///
/// \tparam G The candidate grid type.
template <class G>
concept GridLike = detail::HasGridTraits<G> && std::derived_from<G, GridBase<G>>
                   && detail::HasCellBounds<G> && detail::HasLocate<G>
                   && detail::HasNeighborAcrossFacet<G>;

/// Build a `GridRestriction`, checking what the unconstrained aggregate cannot.
///
/// This is where `GridLike` is enforced and where the two index arrays are checked
/// against the sub-grid they describe. Aggregate initialisation of `GridRestriction`
/// still bypasses both, which is the price of the type being unconstrained; see its
/// documentation.
///
/// \tparam G The grid type.
/// \param grid The windowed sub-grid.
/// \param local_to_global_cell Sub-grid cell id to the originating grid's, in sub-grid
///        cell-id order.
/// \param in_subset `1` for a requested cell, `0` for bounding-box fill.
/// \return The assembled restriction.
/// \throws std::invalid_argument If either array's length differs from
///         `grid.num_cells()`.
template <GridLike G>
[[nodiscard]] GridRestriction<G> make_restriction(G grid,
                                                  std::vector<std::int64_t> local_to_global_cell,
                                                  std::vector<std::uint8_t> in_subset) {
    const auto n = static_cast<std::size_t>(grid.num_cells());
    if (local_to_global_cell.size() != n || in_subset.size() != n) {
        throw std::invalid_argument(
            "make_restriction: local_to_global_cell (" + std::to_string(local_to_global_cell.size())
            + ") and in_subset (" + std::to_string(in_subset.size())
            + ") must both have length grid.num_cells() (" + std::to_string(n) + ").");
    }
    return GridRestriction<G>{std::move(grid), std::move(local_to_global_cell),
                              std::move(in_subset)};
}

// ---------------------------------------------------------------------------
// The census
// ---------------------------------------------------------------------------

namespace detail {

/// Declare the detector and the signature check for one replaceable default.
///
/// `__VA_ARGS__` is the hook's member-pointer type written in terms of `D`; it is a
/// variadic parameter because that type contains commas.
#define PANTR_GRID_DECLARE_HOOK(NAME, ...)                                                 \
    /** The member-pointer type `NAME` must have when it is written. */                    \
    template <class D>                                                                     \
    using hook_signature_##NAME##_t = __VA_ARGS__;                                         \
    /** Whether `D` redeclares `NAME`, by member-pointer type rather than by probe. */     \
    template <class D>                                                                     \
    [[nodiscard]] constexpr bool redeclares_##NAME() noexcept {                            \
        return !std::is_same_v<decltype(&D::NAME), decltype(&GridBase<D>::NAME)>;           \
    }                                                                                      \
    /** Whether `D`'s `NAME` has the signature of the default it replaces. */              \
    template <class D>                                                                     \
    [[nodiscard]] constexpr bool hook_signature_matches_##NAME() noexcept {                \
        return std::is_same_v<decltype(&D::NAME), hook_signature_##NAME##_t<D>>;           \
    }

PANTR_GRID_DECLARE_HOOK(cell_aabb, geometry::AABB<scalar_t<D>> (D::*)(std::int64_t) const)
PANTR_GRID_DECLARE_HOOK(cell_level, std::int64_t (D::*)(std::int64_t) const)
PANTR_GRID_DECLARE_HOOK(child_cells, std::vector<std::int64_t> (D::*)(std::int64_t) const)
PANTR_GRID_DECLARE_HOOK(reference_map,
                        transform::AffineTransform<scalar_t<D>> (D::*)(std::int64_t) const)
PANTR_GRID_DECLARE_HOOK(neighbors, std::vector<std::int64_t> (D::*)(std::int64_t) const)
PANTR_GRID_DECLARE_HOOK(restrict,
                        GridRestriction<D> (D::*)(std::span<const std::int64_t>) const)
PANTR_GRID_DECLARE_HOOK(local_facet_axis_side,
                        std::pair<std::int64_t, std::int64_t> (D::*)(std::int64_t, std::int64_t)
                            const)
PANTR_GRID_DECLARE_HOOK(local_facet_bounds,
                        void (D::*)(std::int64_t, std::int64_t, std::span<scalar_t<D>>,
                                    std::span<scalar_t<D>>) const)
PANTR_GRID_DECLARE_HOOK(is_mesh_boundary_facet, bool (D::*)(std::int64_t, std::int64_t) const)
PANTR_GRID_DECLARE_HOOK(boundary_facets, std::vector<std::int64_t> (D::*)() const)
PANTR_GRID_DECLARE_HOOK(hanging_neighbors,
                        std::vector<std::int64_t> (D::*)(std::int64_t, std::int64_t) const)
PANTR_GRID_DECLARE_HOOK(locate_many,
                        std::vector<std::int64_t> (D::*)(span2d<const scalar_t<D>>) const)
PANTR_GRID_DECLARE_HOOK(query_aabb, std::vector<std::int64_t> (D::*)(
                                        const geometry::AABB<scalar_t<D>>&) const)
PANTR_GRID_DECLARE_HOOK(collect_cell_bounds,
                        void (D::*)(span2d<scalar_t<D>>, span2d<scalar_t<D>>) const)

// Not a hook, and the detector exists so the census can assert that. `2 * ndim` is an
// invariant of the base's own `facet_tags_`, not a policy a grid may restate.
PANTR_GRID_DECLARE_HOOK(num_local_facets, std::int64_t (D::*)(std::int64_t) const)

}  // namespace detail

#undef PANTR_GRID_DECLARE_HOOK

/// Assert one hook against the grid's declaration, in both directions.
///
/// \param GRID The grid type. Must not contain a comma at the top level.
/// \param NAME The hook's member name.
#define PANTR_GRID_CENSUS_HOOK(GRID, NAME)                                                    \
    static_assert(::pantr::grid::detail::redeclares_##NAME<GRID>()                            \
                      == ::pantr::grid::declares(::pantr::grid::grid_traits<GRID>::hooks,     \
                                                 ::pantr::grid::Hook::NAME),                  \
                  "pantr grid census (" #GRID ", " #NAME "): the class and its grid_traits "  \
                  "disagree -- either the hook is written and not declared, in which case "   \
                  "it runs and nothing says so, or it is declared and not written, in which " \
                  "case the default runs and nothing says so");                               \
    static_assert(!::pantr::grid::declares(::pantr::grid::grid_traits<GRID>::hooks,           \
                                           ::pantr::grid::Hook::NAME)                         \
                      || ::pantr::grid::detail::hook_signature_matches_##NAME<GRID>(),        \
                  "pantr grid census (" #GRID ", " #NAME "): the hook does not have the "     \
                  "signature of the default it replaces")

/// Assert a grid against its own `grid_traits`, and force every default body.
///
/// Expand this once per instantiation, at namespace scope, in the grid's own
/// translation unit. It does four things, and the fourth is why it exists at all:
///
///  1. asserts `GridLike<GRID>`, which is where the three primitives are pinned;
///  2. asserts, per replaceable default, that the class and the bitmask agree;
///  3. asserts, per declared hook, that its signature is the default's;
///  4. explicitly instantiates `GridBase<GRID>`, which forces every default BODY --
///     so a default that would not compile for this grid, at a scalar nothing calls,
///     is a compile error here rather than a surprise at the first call.
///
/// What it does NOT establish: that a hook computes the same function as the default
/// it replaces. That is a differential test's job, and the qualified call
/// `g.pantr::grid::GridBase<GRID>::name()` is how it reaches the hidden default.
///
/// \param GRID The grid type. Must not contain a comma at the top level; wrap such a
///        type in an alias first.
#define PANTR_GRID_CENSUS(GRID)                                                               \
    static_assert(::pantr::grid::GridLike<GRID>,                                              \
                  "pantr grid census (" #GRID "): not a grid -- it must derive from "         \
                  "GridBase<itself> and declare cell_bounds, locate and "                     \
                  "neighbor_across_facet with exactly the primitive signatures");             \
    static_assert(!::pantr::grid::detail::redeclares_num_local_facets<GRID>(),                \
                  "pantr grid census (" #GRID "): num_local_facets is not specialisable. "    \
                  "facet_tags_ is sized 2 * ndim once at construction, so a grid that "       \
                  "changed the facet count would desynchronise the registry from the "        \
                  "geometry");                                                                \
    PANTR_GRID_CENSUS_HOOK(GRID, cell_aabb);                                                  \
    PANTR_GRID_CENSUS_HOOK(GRID, cell_level);                                                 \
    PANTR_GRID_CENSUS_HOOK(GRID, child_cells);                                                \
    PANTR_GRID_CENSUS_HOOK(GRID, reference_map);                                              \
    PANTR_GRID_CENSUS_HOOK(GRID, neighbors);                                                  \
    PANTR_GRID_CENSUS_HOOK(GRID, restrict);                                                   \
    PANTR_GRID_CENSUS_HOOK(GRID, local_facet_axis_side);                                      \
    PANTR_GRID_CENSUS_HOOK(GRID, local_facet_bounds);                                         \
    PANTR_GRID_CENSUS_HOOK(GRID, is_mesh_boundary_facet);                                     \
    PANTR_GRID_CENSUS_HOOK(GRID, boundary_facets);                                            \
    PANTR_GRID_CENSUS_HOOK(GRID, hanging_neighbors);                                          \
    PANTR_GRID_CENSUS_HOOK(GRID, locate_many);                                                \
    PANTR_GRID_CENSUS_HOOK(GRID, query_aabb);                                                 \
    PANTR_GRID_CENSUS_HOOK(GRID, collect_cell_bounds);                                        \
    template class ::pantr::grid::GridBase<GRID>

}  // namespace pantr::grid
