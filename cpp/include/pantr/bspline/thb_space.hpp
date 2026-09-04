#pragma once

/// \file
/// The (truncated) hierarchical B-spline space: the Kraft selection over a
/// hierarchical grid, and the truncation coefficients that restore the partition of
/// unity.
///
/// Ports the construction-and-query half of `src/pantr/bspline/_thb_spline_space.py`,
/// which stays as the parity oracle. Basis tabulation, the windowed restriction and
/// the prolongation operators are **not** here: each rests on an operation the C++
/// side does not have yet (Cox-de Boor tabulation, `BsplineSpace::restrict`, a
/// least-squares solve), and each is a computation *over* a space rather than a
/// property *of* one, which is the line `pantr/bspline/space_nd.hpp` already draws.
/// Refinement and coarsening *are* here, because they need nothing new and because
/// they re-enter this type's own constructor.
///
/// ## What this type owns
///
/// The root space, the grid, the truncation flag and the per-direction regularity are
/// its value. Everything else is derived from them at construction: the per-level
/// tensor-product spaces, the per-level per-direction function-to-cell support, the
/// Kraft-active function index sets, their cumulative offsets, and the truncated
/// functions' coefficient boxes. One thing is lazy -- the per-cell contribution table
/// -- and `design/bspline_derived_caches.md` fixes its shape; see below.
///
/// ## Why the class is not decomposed
///
/// FELIGN/pantr#397 says not to split it, and the reason it gives -- "its internals
/// are mutually recursive through the truncation" -- is not what the code does. There
/// is exactly one cycle among the oracle's own methods, `_refine_recursive` calling
/// itself, and that is graded *refinement* (Carraturo et al. 2019, Alg. 4), not
/// truncation; truncation is a forward `while` loop over levels with no recursion in
/// it. What actually resists splitting is three other things, and they are stronger
/// than the stated one:
///
///  - **Construction is one pipeline with no seam.** Level spaces feed the support,
///    the support feeds the Kraft selection, and the truncation of a level-`l`
///    function reads the active sets of *every* finer level. There is no prefix of it
///    that is useful on its own.
///  - **The level walk is shared.** Growing a coefficient box through the two-scale
///    matrix is what the construction-time truncation does and what the
///    prolongation's `_finest_tp_coeffs` does, once each, with no cache between them.
///  - **Three methods re-enter the constructor.** `refine`, `coarsen` and `restrict`
///    all end by building a fresh space, so a split that put construction on one side
///    and refinement on the other would leave refinement calling a half-built type.
///
/// ## The arithmetic, quantity by quantity
///
/// **Everything that selects is exact integer work.** The Kraft test is "is this box
/// entirely inside one mask and not entirely inside another", answered by a summed-area
/// table over `std::int64_t`; the flat indices, the offsets, the level assignment and
/// the contribution table are index arithmetic. Two backends must agree on them
/// **exactly**: no rounding takes place on a count, so bit-identity is the only bar
/// that says anything, and a bounded one could not see two answers of different
/// length at all.
///
/// **The truncation coefficients have digits, and they get a bound.** They are built
/// from the two-scale matrices, which `pantr/bspline/knot_insertion.hpp` computes in
/// `T` -- and then this file widens to `double`, because the oracle does: its
/// `_build_oslo_matrices` wraps the kernel's output in `np.asarray(..., float64)` and
/// every coefficient array it touches afterwards is `float64` whatever the root
/// space's storage is. Getting that backwards -- computing the whole truncation in `T`
/// -- would be a silent divergence at `float` storage that no shape or count would
/// reveal.
///
/// **The truncation never cancels, and that is what makes its bound tight.** Every
/// two-scale coefficient is non-negative, truncation only *zeroes* entries, and the
/// contraction sums non-negative products. So the computed value of an entry
/// accumulated over `m` additions satisfies `|fl(s) - s| <= gamma_m * s` with
/// `gamma_m = m u / (1 - m u)` and no cancellation term -- a relative bound rather
/// than the `sum |terms|` one a signed sum would owe. `m` is the number of terms
/// summed over all directions and all levels the function passes through, which is
/// bounded by the coefficient box's own size.
///
/// **The oracle's contraction goes through BLAS and this one does not.**
/// `_refine_box` reaches `numpy.tensordot`, which reshapes and calls `dgemm`, whose
/// summation order is the BLAS implementation's and differs between OpenBLAS and
/// Accelerate -- `CLAUDE.md` records exactly that trap. So the truncation coefficients
/// are a **bounded** parity quantity and not a bitwise one, and a parity test claiming
/// bit-identity for them would be claiming a property of one machine's BLAS. The
/// summation here is the naive ordered one.
///
/// ## The contribution table
///
/// `design/bspline_derived_caches.md` rules this one rather than leaving it to be
/// chosen here: the oracle's per-cell `dict[int, list[tuple]]` becomes **one flat CSR
/// table -- offsets plus entries -- filled for all cells behind a single
/// double-checked flag**, `max_active_per_cell` becomes a field of that table computed
/// by the same sweep, and the accessor returns a `span` rather than a list, which
/// retires the oracle's unenforced *"callers must not mutate it"* convention.
///
/// What that trades is incrementality: a caller touching one cell pays for the whole
/// grid. The note asks #397 for a footprint measurement before committing, and names
/// per-*level* granularity as the fallback if the table is too large. The measurement
/// is in the pull request rather than here, because a number in a comment rots.
///
/// ## Sharing, and the one exception to `const`
///
/// The root space is `shared_ptr<const BsplineSpace<T>>`, class **H** of
/// `design/bspline_ownership_lifetime.md`: the value is shared, the owner's death is
/// irrelevant, and `level_space(0)` hands back the very handle the space was built
/// from, so the oracle's `thb.level_space(0) is thb.root_space` survives.
///
/// The grid is `shared_ptr<HierarchicalGrid<T>>`, **non-const**, which is the single
/// exception the ownership note carves out and it is narrow: a grid holds `CellTags`
/// and `FacetTags`, which `pantr/grid/tags.hpp` already reasons as the
/// accumulating-container exception to construct-then-freeze, and a `const` handle
/// would reach only the `const` overload of `cell_tags()` and silently remove the
/// ability to tag cells through a THB space's grid. It is safe because refinement
/// returns a *new* grid, so the cell decomposition a space depends on cannot move
/// underneath it.
///
/// `refine` and `coarsen` never hand the result the receiver's grid, not even when
/// they change nothing: a shared grid would make a tag set through one space visible
/// through the other. The no-op path goes through `refine_cells({})`, which
/// `pantr/grid/hierarchical_grid.hpp` routes to `rebuilt()` -- a fresh grid with empty
/// caches and no inherited tags, which is what the oracle's `grid._copy()` is for.
///
/// ## Thread safety
///
/// Every accessor is safe to call concurrently on the same object with no external
/// locking. Everything but the contribution table is frozen at construction; the table
/// is a `pantr::LazySlot`, whose file comment carries the measurement behind that
/// choice and the reason a bare `mutable std::optional` is a data race no value
/// assertion will find.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/hierarchical_grid.hpp"

namespace pantr::bspline {

/// One truncated function's coefficients, as views into the space's own storage.
///
/// Valid for as long as the space is. A function that was not truncated has no entry
/// at all -- it is a plain tensor-product B-spline, and storing a one-element box for
/// it is what the oracle deliberately avoids.
struct TruncatedView {
    /// The level whose tensor-product basis `coeffs` is expressed in.
    std::int64_t rep_level = 0;

    /// Per-direction lower function index of the coefficient box.
    std::span<const std::int64_t> box_lo;

    /// Per-direction width of the coefficient box.
    std::span<const std::int64_t> shape;

    /// The coefficients, row-major over `shape`, always `double`; see the file
    /// comment on why they are not the space's storage type.
    std::span<const double> coeffs;
};

/// The functions active on one cell, as views into the space's contribution table.
///
/// The three views have a common length, `size()`, and index the same entries.
struct CellContributions {
    /// The global hierarchical dof of each entry, strictly increasing.
    std::span<const std::int64_t> dofs;

    /// The level owning each entry.
    std::span<const std::int64_t> levels;

    /// The per-axis function index of each entry in its level space, `size() * dim`
    /// values row-major.
    std::span<const std::int64_t> multi_indices;

    /// The number of active functions on the cell.
    ///
    /// \return The common length of `dofs` and `levels`.
    [[nodiscard]] std::int64_t size() const noexcept {
        return static_cast<std::int64_t>(dofs.size());
    }
};

/// A hierarchical B-spline space on a `pantr::grid::HierarchicalGrid`.
///
/// Built from a root tensor-product space (level 0) and a grid carrying the
/// active-cell hierarchy. The per-level spaces are the root uniformly subdivided by
/// the grid's per-direction factor, which keeps them nested, and the active basis is
/// the Kraft selection: a level-`l` function is active iff its support lies in the
/// level-`l` subdomain but not entirely in the further-refined region.
///
/// With `truncate` set, each active function that straddles a finer refinement
/// boundary has its components on active finer functions removed
/// (Giannelli-Juttler-Speleers), restoring the partition of unity. Only truncated
/// functions carry a coefficient vector; the rest remain plain tensor-product
/// B-splines.
///
/// \tparam T The scalar type the root space's knots are stored in.
template <Real T>
class THBSplineSpace {
  public:
    /// The scalar type the root space's knots are stored in.
    using scalar_type = T;

    /// The hierarchical grid type this space is built on.
    ///
    /// `double` whatever `T` is, and that is the oracle's shape rather than an
    /// omission: `pantr.grid` is `float64`-only by its own port's ruling
    /// (`src/pantr/grid/_grid_backend.py`, and `cpp/bindings/grid_hierarchical.cpp`
    /// registers no `float`), while a root B-spline space stores whichever float dtype
    /// it was handed. `tests/test_thb_spline_space.py`'s
    /// `test_a_float32_root_space_is_still_graded_in_float64` pins exactly that pair,
    /// so tying the grid's scalar to `T` would make the shipped case unrepresentable.
    using grid_type = pantr::grid::HierarchicalGrid<double>;

    /// Build a hierarchical space over a root space and a grid.
    ///
    /// \param root_space The level-0 tensor-product space. Shared, not copied.
    /// \param grid The hierarchy. Shared, not copied, and non-`const`; see the file
    ///        comment on the one exception to `const`.
    /// \param truncate Whether to build the truncated (THB) basis rather than the
    ///        plain hierarchical (HB) one.
    /// \param regularity Per-direction continuity at the knots inserted when
    ///        subdividing to finer levels, one entry per direction. An empty
    ///        `optional` means maximal smoothness, `degree - 1`. Must be `ndim()`
    ///        long; each present value must satisfy `-1 <= r < degree[k]`.
    /// \throws std::invalid_argument If either handle is null, the grid and the root
    ///         space disagree on dimension, on the root knot-span grid or on the
    ///         domain, `regularity` has the wrong length, or a regularity value is out
    ///         of range.
    THBSplineSpace(std::shared_ptr<const BsplineSpace<T>> root_space,
                   std::shared_ptr<grid_type> grid, bool truncate,
                   std::vector<std::optional<std::int64_t>> regularity)
        : root_space_(std::move(root_space)),
          grid_(std::move(grid)),
          truncate_(truncate),
          regularity_(std::move(regularity)) {
        check_arguments();
        build_level_spaces();
        build_support();
        select_active_functions();
        if (truncate_) {
            compute_truncated_coefficients();
        }
    }

    /// The level-0 tensor-product space.
    ///
    /// Class **H**: the returned handle keeps its value alive independently of this
    /// space, so a caller may outlive the owner.
    ///
    /// \return A handle on the root space, never null.
    [[nodiscard]] std::shared_ptr<const BsplineSpace<T>> root_space() const noexcept {
        return root_space_;
    }

    /// Borrow the level-0 space. Valid while `*this` is, and NOT bound; see the `_ref`
    /// rule in `design/bspline_ownership_lifetime.md`.
    ///
    /// \return A reference to the root space.
    [[nodiscard]] const BsplineSpace<T>& root_space_ref() const noexcept { return *root_space_; }

    /// The hierarchy this space is built on.
    ///
    /// Class **H**, and the one accessor in `pantr.bspline` whose handle is not
    /// `const`; the file comment gives the reason and the condition under which it
    /// would have to be reversed.
    ///
    /// \return A handle on the grid, never null.
    [[nodiscard]] std::shared_ptr<grid_type> grid() const noexcept { return grid_; }

    /// Borrow the hierarchy. Valid while `*this` is, and NOT bound.
    ///
    /// \return A reference to the grid.
    [[nodiscard]] grid_type& grid_ref() const noexcept { return *grid_; }

    /// The number of parametric directions.
    ///
    /// \return The root space's dimension.
    [[nodiscard]] std::int64_t dim() const noexcept { return root_space_->dim(); }

    /// The per-direction polynomial degrees.
    ///
    /// \return A view of `dim()` degrees, in axis order, valid while this space is.
    [[nodiscard]] std::span<const std::int64_t> degrees() const noexcept {
        return root_space_->degrees();
    }

    /// The number of hierarchy levels.
    ///
    /// \return `grid().max_level() + 1`.
    [[nodiscard]] std::int64_t num_levels() const noexcept {
        return static_cast<std::int64_t>(level_spaces_.size());
    }

    /// Whether the truncated basis is used.
    ///
    /// \return `true` for THB, `false` for the plain hierarchical basis.
    [[nodiscard]] bool truncate() const noexcept { return truncate_; }

    /// The per-direction regularity the finer levels were built with.
    ///
    /// \return A view of `dim()` entries, an empty `optional` meaning maximal
    ///         smoothness.
    [[nodiscard]] std::span<const std::optional<std::int64_t>> regularity() const noexcept {
        return std::span<const std::optional<std::int64_t>>(regularity_);
    }

    /// The total number of active hierarchical functions.
    ///
    /// \return The sum of the per-level active counts.
    [[nodiscard]] std::int64_t num_total_basis() const noexcept { return num_active_; }

    /// The number of active functions at each level.
    ///
    /// \return A view of `num_levels()` counts, valid while this space is.
    [[nodiscard]] std::span<const std::int64_t> num_basis_per_level() const noexcept {
        return std::span<const std::int64_t>(num_per_level_);
    }

    /// The global dof at which each level's functions begin.
    ///
    /// \return A view of `num_levels() + 1` cumulative counts; the last entry is
    ///         `num_total_basis()`.
    [[nodiscard]] std::span<const std::int64_t> level_offsets() const noexcept {
        return std::span<const std::int64_t>(func_offset_);
    }

    /// The parametric domain, per direction.
    ///
    /// \return The root space's domain, `2 * dim()` values as `[lo_0, hi_0, ...]`.
    [[nodiscard]] std::span<const T> domain() const noexcept { return root_space_->domain_flat(); }

    /// The parametric tolerance.
    ///
    /// \return The root space's tolerance.
    [[nodiscard]] double tolerance() const { return root_space_->tolerance(); }

    /// The tensor-product space at `level`.
    ///
    /// Class **H**. Level 0 is the very handle this space was built from, not a copy
    /// of it, which is what makes the oracle's `thb.level_space(0) is thb.root_space`
    /// hold through the binding.
    ///
    /// \param level Hierarchy level in `[0, num_levels())`.
    /// \return A handle on the level space.
    /// \throws std::invalid_argument If `level` is out of range.
    [[nodiscard]] std::shared_ptr<const BsplineSpace<T>> level_space(std::int64_t level) const {
        check_level(level);
        return level_spaces_[static_cast<std::size_t>(level)];
    }

    /// Borrow the tensor-product space at `level`. Valid while `*this` is, and NOT
    /// bound.
    ///
    /// \param level Hierarchy level in `[0, num_levels())`.
    /// \return A reference to the level space.
    /// \throws std::invalid_argument If `level` is out of range.
    [[nodiscard]] const BsplineSpace<T>& level_space_ref(std::int64_t level) const {
        check_level(level);
        return *level_spaces_[static_cast<std::size_t>(level)];
    }

    /// The flat indices of the functions the Kraft rule selects at `level`.
    ///
    /// Class **A**: a view of this space's own storage rather than a copy. The oracle
    /// returns a fresh array; the port returns a read-only view and lets the binding
    /// present it as a non-writable array owned by the Python object, which is the
    /// same rule `design/bspline_derived_caches.md` states for
    /// `SpanwiseElementExtraction.ops_1d`.
    ///
    /// \param level Hierarchy level in `[0, num_levels())`.
    /// \return Sorted flat (C-order) indices into the level space's tensor-product
    ///         basis, valid while this space is.
    /// \throws std::invalid_argument If `level` is out of range.
    [[nodiscard]] std::span<const std::int64_t> active_function_indices(
        std::int64_t level) const {
        check_level(level);
        return std::span<const std::int64_t>(active_funcs_[static_cast<std::size_t>(level)]);
    }

    /// The global dofs of the functions whose support intersects cell `cid`.
    ///
    /// Selects on tensor-product support, so under truncation a few of the functions
    /// listed may evaluate to exactly zero on the cell. That is the oracle's contract
    /// and it is what a fixed-width dofmap wants.
    ///
    /// \param cid Active cell flat id in `[0, grid().num_cells())`.
    /// \return Sorted global dofs, valid while this space is.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] std::span<const std::int64_t> active_basis(std::int64_t cid) const {
        return contributions(cid).dofs;
    }

    /// Everything the contribution table records about cell `cid`.
    ///
    /// The first call fills the table for **every** cell; see the file comment on what
    /// that trades. Concurrent first calls build it once.
    ///
    /// \param cid Active cell flat id in `[0, grid().num_cells())`.
    /// \return Views into the table, valid while this space is.
    /// \throws std::out_of_range If `cid` is out of range.
    [[nodiscard]] CellContributions contributions(std::int64_t cid) const {
        // Range first, table second. The oracle's per-cell cache refuses a bad id
        // without computing anything; building the whole grid's table and *then*
        // throwing would make a probe for a valid id cost an O(cells) sweep.
        if (cid < 0 || cid >= grid_->num_cells()) {
            throw std::out_of_range("cell id " + std::to_string(cid) + " is out of range [0, "
                                    + std::to_string(grid_->num_cells()) + ").");
        }
        const ContributionTable& table = contributions_table();
        const auto i = static_cast<std::size_t>(cid);
        const auto begin = static_cast<std::size_t>(table.offset[i]);
        const auto count = static_cast<std::size_t>(table.offset[i + 1] - table.offset[i]);
        const auto d = static_cast<std::size_t>(dim());
        return CellContributions{
            std::span<const std::int64_t>(table.dof.data() + begin, count),
            std::span<const std::int64_t>(table.level.data() + begin, count),
            std::span<const std::int64_t>(table.multi.data() + begin * d, count * d)};
    }

    /// The largest number of active functions on any single cell.
    ///
    /// The width a fixed-size dofmap needs. Truncation can annihilate a function on a
    /// cell it supports but never add one, so this is the same for the THB and HB
    /// bases. Fills the contribution table if it is not filled.
    ///
    /// \return The maximum over every active cell; zero for a grid with no cells.
    [[nodiscard]] std::int64_t max_active_per_cell() const {
        return contributions_table().max_per_cell;
    }

    /// The number of functions that carry truncation coefficients.
    ///
    /// \return Zero when `truncate()` is false, and otherwise the count of active
    ///         functions the truncation actually changed.
    [[nodiscard]] std::int64_t num_truncated() const noexcept {
        return static_cast<std::int64_t>(truncated_.size());
    }

    /// The truncation coefficients of global dof `dof`, if it has any.
    ///
    /// \param dof Global active-function index in `[0, num_total_basis())`.
    /// \return The views, or an empty `optional` when the function was not truncated
    ///         and is a plain tensor-product B-spline.
    /// \throws std::out_of_range If `dof` is out of range.
    [[nodiscard]] std::optional<TruncatedView> truncated(std::int64_t dof) const {
        check_dof(dof);
        const auto it = std::lower_bound(
            truncated_.begin(), truncated_.end(), dof,
            [](const TruncatedEntry& entry, std::int64_t key) { return entry.dof < key; });
        if (it == truncated_.end() || it->dof != dof) {
            return std::nullopt;
        }
        return TruncatedView{it->rep_level, std::span<const std::int64_t>(it->box_lo),
                             std::span<const std::int64_t>(it->shape),
                             std::span<const double>(it->coeffs)};
    }

    /// The level that owns global dof `dof`.
    ///
    /// \param dof Global active-function index in `[0, num_total_basis())`.
    /// \return The level whose dof range contains `dof`.
    /// \throws std::out_of_range If `dof` is out of range.
    [[nodiscard]] std::int64_t dof_level(std::int64_t dof) const {
        check_dof(dof);
        const auto it = std::upper_bound(func_offset_.begin(), func_offset_.end(), dof);
        return static_cast<std::int64_t>(it - func_offset_.begin()) - 1;
    }

    /// A new space with the marked cells refined.
    ///
    /// This space and its grid are untouched: the grid is refined by rebinding and a
    /// new space is built on the result. The result never holds this space's grid
    /// object, even when nothing was refined; see the file comment.
    ///
    /// With `admissible_class = m` the refinement is graded so the resulting mesh is
    /// admissible of class `m` -- the truncated functions acting on any cell span at
    /// most `m` successive levels -- by the recursive refinement-neighbourhood
    /// algorithm of Carraturo et al. (2019, Alg. 4). That assumes the current mesh is
    /// already admissible of class `m`, which holds for the root and for any mesh
    /// built by graded refinement. An empty `optional` refines exactly the marked
    /// cells.
    ///
    /// \param cell_ids Flat ids of active cells to refine; duplicates are ignored.
    /// \param admissible_class The class to maintain, at least 2, or empty for no
    ///        grading.
    /// \return A new space on the refined grid, with this space's root space,
    ///         truncation flag and regularity.
    /// \throws std::out_of_range If any id is outside `[0, grid().num_cells())`.
    /// \throws std::invalid_argument If `admissible_class` is present and below 2.
    [[nodiscard]] THBSplineSpace refine(std::span<const std::int64_t> cell_ids,
                                        std::optional<std::int64_t> admissible_class) const {
        check_admissible_class(admissible_class);
        const std::vector<std::int64_t> ids = unique_valid_cell_ids(cell_ids);

        // Resolved against the original grid before anything is refined: every
        // `refine` reassigns flat ids, so a `(level, midx)` pair is the only handle
        // that survives the loop.
        const auto d = static_cast<std::size_t>(dim());
        std::vector<Marked> marked;
        marked.reserve(ids.size());
        for (const std::int64_t cid : ids) {
            Marked cell{grid_->cell_level(cid), std::vector<std::int64_t>(d)};
            grid_->cell_multi_index(cid, std::span<std::int64_t>(cell.midx));
            marked.push_back(std::move(cell));
        }
        return refine_marked(marked, admissible_class);
    }

    /// A new space with the active cells of a rectangular region refined.
    ///
    /// The region is the cell-index box `[lo, hi)` in level-`level` coordinates, which
    /// is `HierarchicalGrid::refine`'s convention. Only the currently active leaves
    /// inside it are refined; a box holding none is a no-op that still returns a new
    /// space over a grid of its own.
    ///
    /// \param level Level the box lives at, in `[0, grid().max_level()]`.
    /// \param lo Per-direction start index, inclusive.
    /// \param hi Per-direction end index, exclusive.
    /// \param admissible_class As `refine`.
    /// \return A new space on the refined grid.
    /// \throws std::invalid_argument If `admissible_class` is present and below 2,
    ///         `level` is out of range, `lo` or `hi` is not `dim()` long, some
    ///         `lo[k] >= hi[k]`, or the box leaves the level's domain.
    [[nodiscard]] THBSplineSpace refine_region(
        std::int64_t level, std::span<const std::int64_t> lo, std::span<const std::int64_t> hi,
        std::optional<std::int64_t> admissible_class) const {
        check_admissible_class(admissible_class);
        validate_region(level, lo, hi);

        const auto d = static_cast<std::size_t>(dim());
        std::vector<Marked> marked;
        std::vector<std::int64_t> midx(lo.begin(), lo.end());
        for (;;) {
            if (grid_->is_active_leaf(level, std::span<const std::int64_t>(midx))) {
                marked.push_back(Marked{level, midx});
            }
            std::size_t axis = d;
            while (axis > 0) {
                --axis;
                ++midx[axis];
                if (midx[axis] < hi[axis]) {
                    break;
                }
                midx[axis] = lo[axis];
                if (axis == 0) {
                    return refine_marked(marked, admissible_class);
                }
            }
            if (d == 0) {
                return refine_marked(marked, admissible_class);
            }
        }
    }

    /// A new space with the marked cells coarsened away.
    ///
    /// A parent is reactivated only when **all** of its children are marked active
    /// leaves, which is `HierarchicalGrid::coarsen_cells`'s own rule and Alg. 5 of
    /// Carraturo et al. (2019); this drives it one parent at a time so the
    /// admissibility guard can veto a parent without affecting the rest. Parents are
    /// visited deepest first, so a veto is decided against a mesh whose finer
    /// coarsenings have already happened.
    ///
    /// With `admissible_class = m` a parent is reactivated only if its coarsening
    /// neighbourhood (Def. 3.5) is empty. With an empty `optional` that guard is
    /// skipped, and coarsening is then the exact inverse of `refine`.
    ///
    /// **Within a level the order is lexicographic here and the oracle's set-iteration
    /// order there, and that is not a divergence.** Two parents at one level have
    /// disjoint child sets, so demoting one cannot make the other's family incomplete;
    /// and the admissibility guard for a parent at level `L` reads only the active cells
    /// at level `L + m >= L + 2`, which a same-level peer's demotion -- confined to
    /// level `L + 1` -- cannot touch, since a complete family has no grandchildren under
    /// it. So the result does not depend on the within-level order. Fixing one anyway is
    /// what makes *this* side reproducible run to run, which the oracle's set is not.
    ///
    /// \param cell_ids Flat ids of active leaf cells to coarsen away; an empty range
    ///        is valid and coarsens nothing.
    /// \param admissible_class The class to maintain, at least 2, or empty for no
    ///        guard.
    /// \return A new space on the coarsened grid.
    /// \throws std::out_of_range If any id is outside `[0, grid().num_cells())`.
    /// \throws std::invalid_argument If `admissible_class` is present and below 2.
    [[nodiscard]] THBSplineSpace coarsen(std::span<const std::int64_t> cell_ids,
                                         std::optional<std::int64_t> admissible_class) const {
        check_admissible_class(admissible_class);
        const std::vector<std::int64_t> ids = unique_valid_cell_ids(cell_ids);

        const auto d = static_cast<std::size_t>(dim());
        const std::span<const std::int64_t> factor = grid_->factor();
        std::vector<Marked> marked;
        marked.reserve(ids.size());
        for (const std::int64_t cid : ids) {
            Marked cell{grid_->cell_level(cid), std::vector<std::int64_t>(d)};
            grid_->cell_multi_index(cid, std::span<std::int64_t>(cell.midx));
            marked.push_back(std::move(cell));
        }

        // Sorted once so the per-child membership test below is a binary search. The
        // order is the same one the parents are sorted by, and neither is observable:
        // see the note on within-level order above.
        std::sort(marked.begin(), marked.end(), marked_order);

        std::vector<Marked> parents;
        for (const Marked& cell : marked) {
            if (cell.level < 1) {
                continue;
            }
            Marked parent{cell.level - 1, std::vector<std::int64_t>(d)};
            for (std::size_t k = 0; k < d; ++k) {
                parent.midx[k] = cell.midx[k] / factor[k];
            }
            parents.push_back(std::move(parent));
        }
        std::sort(parents.begin(), parents.end(), [](const Marked& a, const Marked& b) {
            // Deepest first, then lexicographic, so the order does not depend on the
            // order the ids arrived in.
            if (a.level != b.level) {
                return a.level > b.level;
            }
            return a.midx < b.midx;
        });
        parents.erase(std::unique(parents.begin(), parents.end(),
                                  [](const Marked& a, const Marked& b) {
                                      return a.level == b.level && a.midx == b.midx;
                                  }),
                      parents.end());

        std::int64_t num_children = 1;
        for (std::size_t k = 0; k < d; ++k) {
            num_children *= factor[k];
        }

        std::shared_ptr<grid_type> current = grid_;
        bool changed = false;
        std::vector<std::int64_t> child(d);
        for (const Marked& parent : parents) {
            // Named afresh on the current grid: every coarsening reassigns flat ids.
            std::vector<std::int64_t> child_ids;
            for (std::size_t k = 0; k < d; ++k) {
                child[k] = parent.midx[k] * factor[k];
            }
            for (;;) {
                // A binary search rather than a scan: `marked` can hold every cell of
                // the mesh, and a linear test per child would make this quadratic in
                // the marked set. The oracle uses a Python `set` for the same reason.
                const bool is_marked = std::binary_search(
                    marked.begin(), marked.end(), Marked{parent.level + 1, child},
                    marked_order);
                if (is_marked) {
                    const std::optional<std::int64_t> cid = current->cell_id(
                        parent.level + 1, std::span<const std::int64_t>(child));
                    if (cid.has_value()) {
                        child_ids.push_back(*cid);
                    }
                }
                std::size_t axis = d;
                bool done = d == 0;
                while (axis > 0) {
                    --axis;
                    ++child[axis];
                    if (child[axis] < (parent.midx[axis] + 1) * factor[axis]) {
                        break;
                    }
                    child[axis] = parent.midx[axis] * factor[axis];
                    if (axis == 0) {
                        done = true;
                    }
                }
                if (done) {
                    break;
                }
            }
            // An incomplete family is one `coarsen_cells` would skip anyway, so
            // leaving here skips the only expensive test in the loop.
            if (static_cast<std::int64_t>(child_ids.size()) < num_children) {
                continue;
            }
            if (admissible_class.has_value()
                && !coarsening_neighborhood_empty(parent.level, parent.midx, *admissible_class,
                                                  *current)) {
                continue;
            }
            current = std::make_shared<grid_type>(
                current->coarsen_cells(std::span<const std::int64_t>(child_ids)));
            changed = true;
        }
        return rebound(changed ? current : detached_grid());
    }

    /// A compact string form.
    ///
    /// \return `"THBSplineSpace(dim=..., degrees=(...), num_levels=..., "`
    ///         `"num_total_basis=..., truncate=...)"`, the oracle's `repr` character
    ///         for character.
    [[nodiscard]] std::string to_string() const {
        return "THBSplineSpace(dim=" + std::to_string(dim()) + ", degrees="
               + tuple_repr(degrees()) + ", num_levels=" + std::to_string(num_levels())
               + ", num_total_basis=" + std::to_string(num_active_)
               + ", truncate=" + (truncate_ ? "True" : "False") + ")";
    }


  private:
    /// A cell named by its level and its per-axis index at that level.
    ///
    /// Flat ids are reassigned by every refinement, so this is the only handle that
    /// survives a loop that refines.
    struct Marked {
        std::int64_t level = 0;          ///< The level the cell lives at.
        std::vector<std::int64_t> midx;  ///< Its per-axis index at that level.
    };

    /// The cell support of every function of one 1D space.
    struct Support1D {
        std::vector<std::int64_t> first_basis;  ///< Per interval, its first function.
        std::vector<std::int64_t> first_cell;   ///< Per function, its first interval.
        std::vector<std::int64_t> last_cell;    ///< Per function, its last interval.
    };

    /// One direction's two-scale matrix, with the shape needed to index it.
    struct TwoScale {
        std::vector<double> alpha;  ///< `rows * cols` values, row-major.
        std::int64_t rows = 0;      ///< The refined level's basis count.
        std::int64_t cols = 0;      ///< The coarse level's basis count.
    };

    /// One truncated function's stored coefficients.
    struct TruncatedEntry {
        std::int64_t dof = 0;              ///< The global active-function index.
        std::int64_t rep_level = 0;        ///< The level the coefficients live in.
        std::vector<std::int64_t> box_lo;  ///< Per-direction lower function index.
        std::vector<std::int64_t> shape;   ///< Per-direction box width.
        std::vector<double> coeffs;        ///< Row-major over `shape`.
    };

    /// The per-cell active functions, in one flat CSR table.
    ///
    /// `design/bspline_derived_caches.md` fixes this shape; see the file comment.
    struct ContributionTable {
        std::vector<std::int64_t> offset;  ///< `num_cells + 1` entry offsets.
        std::vector<std::int64_t> dof;     ///< Per entry, the global dof.
        std::vector<std::int64_t> level;   ///< Per entry, the owning level.
        std::vector<std::int64_t> multi;   ///< Per entry, `dim` function indices.
        std::int64_t max_per_cell = 0;     ///< The widest cell, from the same sweep.
    };

    // ---------------------------------------------------------------- validation

    /// Refuse the constructor's arguments before anything is derived.
    ///
    /// \throws std::invalid_argument If a handle is null, the grid and the root space
    ///         disagree, or the regularity is mis-shaped or out of range.
    void check_arguments() const {
        if (root_space_ == nullptr) {
            throw std::invalid_argument("root_space must not be null.");
        }
        if (grid_ == nullptr) {
            throw std::invalid_argument("grid must not be null.");
        }
        const std::int64_t d = root_space_->dim();
        if (grid_->ndim() != d) {
            throw std::invalid_argument("grid.ndim (" + std::to_string(grid_->ndim())
                                        + ") must equal root_space.dim (" + std::to_string(d)
                                        + ").");
        }
        const std::span<const std::int64_t> cells = grid_->root().cells_per_axis();
        const std::span<const std::int64_t> intervals = root_space_->num_intervals();
        bool same = cells.size() == intervals.size();
        for (std::size_t k = 0; same && k < cells.size(); ++k) {
            same = cells[k] == intervals[k];
        }
        if (!same) {
            throw std::invalid_argument("grid root cells_per_axis " + tuple_repr(cells)
                                        + " must match root_space.num_intervals "
                                        + tuple_repr(intervals) + ".");
        }

        // Absolute, against the space's own knot tolerance. A relative comparison
        // accepts a fixed relative mismatch at every scale, and a mixed one passes a
        // factor-of-two wrong domain on a tiny domain; the oracle's comment carries
        // the case that argues it.
        const std::span<const T> domain_flat = root_space_->domain_flat();
        const double tol = root_space_->tolerance();
        bool matches = true;
        for (std::int64_t k = 0; matches && k < d; ++k) {
            // The grid has no `bounds` accessor of its own: the oracle's property is
            // the first and last breakpoint of each axis, and `grid_types.cpp` builds
            // it the same way.
            const std::span<const double> breakpoints = grid_->root().breakpoints(k);
            const auto i = static_cast<std::size_t>(k) * 2;
            matches = std::abs(static_cast<double>(breakpoints.front())
                               - static_cast<double>(domain_flat[i]))
                          <= tol
                      && std::abs(static_cast<double>(breakpoints.back())
                                  - static_cast<double>(domain_flat[i + 1]))
                             <= tol;
        }
        if (!matches) {
            throw std::invalid_argument("grid root bounds must match root_space domain.");
        }

        if (static_cast<std::int64_t>(regularity_.size()) != d) {
            throw std::invalid_argument("regularity must be a scalar or length-"
                                        + std::to_string(d) + " sequence; got length "
                                        + std::to_string(regularity_.size()) + ".");
        }
        const std::span<const std::int64_t> deg = root_space_->degrees();
        for (std::size_t k = 0; k < regularity_.size(); ++k) {
            if (!regularity_[k].has_value()) {
                continue;
            }
            const std::int64_t r = *regularity_[k];
            if (r < -1 || r >= deg[k]) {
                throw std::invalid_argument(
                    "regularity[" + std::to_string(k) + "]=" + std::to_string(r)
                    + " must be in [-1, degree[" + std::to_string(k)
                    + "]-1=" + std::to_string(deg[k] - 1) + "]; got " + std::to_string(r) + ".");
            }
        }
    }

    /// Order two named cells by level then by multi-index, for a sorted lookup.
    ///
    /// \param a The first cell.
    /// \param b The second.
    /// \return `true` if `a` orders before `b`.
    [[nodiscard]] static bool marked_order(const Marked& a, const Marked& b) {
        if (a.level != b.level) {
            return a.level < b.level;
        }
        return a.midx < b.midx;
    }

    /// Sort and deduplicate cell ids, refusing every out-of-range one at once.
    ///
    /// The oracle collects **all** the offending ids before it raises, and names them
    /// in the message; stopping at the first is a real loss for a caller debugging a
    /// list of them.
    ///
    /// \param cell_ids The ids as given.
    /// \return The ids, sorted and deduplicated, as `numpy.unique` returns them.
    /// \throws std::out_of_range If any id is outside `[0, grid().num_cells())`.
    [[nodiscard]] std::vector<std::int64_t> unique_valid_cell_ids(
        std::span<const std::int64_t> cell_ids) const {
        std::vector<std::int64_t> ids(cell_ids.begin(), cell_ids.end());
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        std::vector<std::int64_t> bad;
        for (const std::int64_t cid : ids) {
            if (cid < 0 || cid >= grid_->num_cells()) {
                bad.push_back(cid);
            }
        }
        if (!bad.empty()) {
            throw std::out_of_range("cell_ids must lie in [0, "
                                    + std::to_string(grid_->num_cells())
                                    + "); got out-of-range id(s): " + list_repr(bad) + ".");
        }
        return ids;
    }

    /// Refuse a level outside the hierarchy.
    ///
    /// \param level The level to check.
    /// \throws std::invalid_argument If it is outside `[0, num_levels())`.
    void check_level(std::int64_t level) const {
        if (level < 0 || level >= num_levels()) {
            throw std::invalid_argument("level must be in [0, " + std::to_string(num_levels() - 1)
                                        + "]; got " + std::to_string(level) + ".");
        }
    }

    /// Refuse a global dof outside the active basis.
    ///
    /// \param dof The dof to check.
    /// \throws std::out_of_range If it is outside `[0, num_total_basis())`.
    void check_dof(std::int64_t dof) const {
        if (dof < 0 || dof >= num_active_) {
            throw std::out_of_range("dof " + std::to_string(dof) + " is out of range [0, "
                                    + std::to_string(num_active_) + ").");
        }
    }

    /// Refuse an admissibility class below the definition's minimum.
    ///
    /// \param admissible_class The class, or an empty `optional` for no grading.
    /// \throws std::invalid_argument If it is present and below 2.
    static void check_admissible_class(std::optional<std::int64_t> admissible_class) {
        if (admissible_class.has_value() && *admissible_class < 2) {
            throw std::invalid_argument("admissible_class must be an integer >= 2 or None; got "
                                        + std::to_string(*admissible_class) + ".");
        }
    }

    /// Validate a `[lo, hi)` cell-index box at `level`.
    ///
    /// The same four checks `HierarchicalGrid::refine` applies: the level's range, the
    /// corners' lengths, `lo < hi` per axis, and the box inside the level's domain.
    ///
    /// \param level The level the box lives at.
    /// \param lo Per-direction start index, inclusive.
    /// \param hi Per-direction end index, exclusive.
    /// \throws std::invalid_argument If any of the four fails.
    void validate_region(std::int64_t level, std::span<const std::int64_t> lo,
                         std::span<const std::int64_t> hi) const {
        const std::int64_t max_level = grid_->max_level();
        if (level < 0 || level > max_level) {
            throw std::invalid_argument("level must be in [0, " + std::to_string(max_level)
                                        + "]; got " + std::to_string(level) + ".");
        }
        const auto d = static_cast<std::size_t>(dim());
        if (lo.size() != d || hi.size() != d) {
            throw std::invalid_argument("lo and hi must have length " + std::to_string(d)
                                        + "; got " + std::to_string(lo.size()) + " and "
                                        + std::to_string(hi.size()) + ".");
        }
        for (std::size_t k = 0; k < d; ++k) {
            if (lo[k] >= hi[k]) {
                throw std::invalid_argument(
                    "lo must be strictly less than hi in every dimension; got lo="
                    + tuple_repr(lo) + ", hi=" + tuple_repr(hi) + ".");
            }
        }
        for (std::size_t k = 0; k < d; ++k) {
            const std::int64_t n =
                grid_->level_cells_per_axis(level, static_cast<std::int64_t>(k));
            if (lo[k] < 0 || hi[k] > n) {
                throw std::invalid_argument(
                    "[lo, hi) out of bounds at level " + std::to_string(level) + ": axis "
                    + std::to_string(k) + " needs [0, " + std::to_string(n) + "), got ["
                    + std::to_string(lo[k]) + ", " + std::to_string(hi[k]) + ").");
            }
        }
    }

    // ------------------------------------------------------------- index helpers

    /// The flat C-order index of a multi-index.
    ///
    /// \param multi Per-axis indices.
    /// \param shape Per-axis extents.
    /// \return The flat index, last axis fastest.
    [[nodiscard]] static std::int64_t ravel(std::span<const std::int64_t> multi,
                                            std::span<const std::int64_t> shape) noexcept {
        std::int64_t flat = 0;
        for (std::size_t k = 0; k < shape.size(); ++k) {
            flat = flat * shape[k] + multi[k];
        }
        return flat;
    }

    /// The multi-index of a flat C-order index.
    ///
    /// \param flat The flat index.
    /// \param shape Per-axis extents.
    /// \param out Receives `shape.size()` per-axis indices.
    static void unravel(std::int64_t flat, std::span<const std::int64_t> shape,
                        std::span<std::int64_t> out) noexcept {
        for (std::size_t k = shape.size(); k > 0; --k) {
            out[k - 1] = flat % shape[k - 1];
            flat /= shape[k - 1];
        }
    }

    /// Python's `repr` of a tuple of integers.
    ///
    /// \param values The entries.
    /// \return `"(a, b, c)"`, and `"(a,)"` for one entry, as Python writes them.
    [[nodiscard]] static std::string tuple_repr(std::span<const std::int64_t> values) {
        std::string text = "(";
        for (std::size_t i = 0; i < values.size(); ++i) {
            text += std::to_string(values[i]);
            if (i + 1 < values.size() || values.size() == 1) {
                text += ",";
            }
            if (i + 1 < values.size()) {
                text += " ";
            }
        }
        return text + ")";
    }

    /// Python's `repr` of a list of integers.
    ///
    /// \param values The entries.
    /// \return `"[a, b, c]"`, and `"[]"` for none, as Python writes them.
    [[nodiscard]] static std::string list_repr(std::span<const std::int64_t> values) {
        std::string text = "[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                text += ", ";
            }
            text += std::to_string(values[i]);
        }
        return text + "]";
    }

    /// The number of cells a per-axis shape describes.
    ///
    /// \param shape Per-axis extents.
    /// \return Their product, 1 for an empty shape.
    [[nodiscard]] static std::int64_t count_of(std::span<const std::int64_t> shape) noexcept {
        std::int64_t total = 1;
        for (const std::int64_t n : shape) {
            total *= n;
        }
        return total;
    }

    // ------------------------------------------------------------- mask helpers

    /// The bounding box of a mask's set cells.
    ///
    /// \param mask A `count_of(shape)` mask, C-order.
    /// \param shape Per-axis extents.
    /// \param lo Receives the per-axis minimum, inclusive.
    /// \param hi Receives the per-axis maximum plus one, exclusive.
    /// \return `false` when the mask is empty, in which case `lo` and `hi` are
    ///         untouched.
    [[nodiscard]] static bool bounding_box(const std::vector<std::uint8_t>& mask,
                                           std::span<const std::int64_t> shape,
                                           std::vector<std::int64_t>& lo,
                                           std::vector<std::int64_t>& hi) {
        const auto d = shape.size();
        bool any = false;
        std::vector<std::int64_t> multi(d);
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 0U) {
                continue;
            }
            unravel(static_cast<std::int64_t>(i), shape, std::span<std::int64_t>(multi));
            if (!any) {
                for (std::size_t k = 0; k < d; ++k) {
                    lo[k] = multi[k];
                    hi[k] = multi[k] + 1;
                }
                any = true;
            } else {
                for (std::size_t k = 0; k < d; ++k) {
                    lo[k] = std::min(lo[k], multi[k]);
                    hi[k] = std::max(hi[k], multi[k] + 1);
                }
            }
        }
        return any;
    }

    /// The summed-area table of a mask's **complement**.
    ///
    /// Entry `(i_0, ..., i_{d-1})` of the returned array, whose extents are
    /// `shape + 1` per axis, is the number of cleared cells in the box
    /// `[0, i_0) x ... x [0, i_{d-1})`. A box is therefore all-set iff its
    /// inclusion-exclusion sum over `2 ** d` corners is zero, which is what
    /// `box_all_true` computes.
    ///
    /// Exact integer arithmetic throughout, so this is the oracle's answer rather than
    /// an equivalent one.
    ///
    /// \param mask A `count_of(shape)` mask, C-order.
    /// \param shape Per-axis extents.
    /// \return The table, C-order over `shape + 1`.
    [[nodiscard]] static std::vector<std::int64_t> summed_area(
        const std::vector<std::uint8_t>& mask, std::span<const std::int64_t> shape) {
        const auto d = shape.size();
        std::vector<std::int64_t> ext(d);
        for (std::size_t k = 0; k < d; ++k) {
            ext[k] = shape[k] + 1;
        }
        std::vector<std::int64_t> table(static_cast<std::size_t>(count_of(ext)), 0);

        std::vector<std::int64_t> multi(d);
        std::vector<std::int64_t> shifted(d);
        for (std::size_t i = 0; i < mask.size(); ++i) {
            unravel(static_cast<std::int64_t>(i), shape, std::span<std::int64_t>(multi));
            for (std::size_t k = 0; k < d; ++k) {
                shifted[k] = multi[k] + 1;
            }
            table[static_cast<std::size_t>(ravel(shifted, ext))] =
                static_cast<std::int64_t>(mask[i] == 0U);
        }

        // One prefix sum per axis, in place. Strides are recomputed per axis rather
        // than cached: `d` is at most 3 everywhere in this tree.
        for (std::size_t axis = 0; axis < d; ++axis) {
            std::int64_t stride = 1;
            for (std::size_t k = axis + 1; k < d; ++k) {
                stride *= ext[k];
            }
            const std::int64_t total = count_of(ext);
            for (std::int64_t flat = 0; flat < total; ++flat) {
                const std::int64_t index_on_axis = (flat / stride) % ext[axis];
                if (index_on_axis == 0) {
                    continue;
                }
                table[static_cast<std::size_t>(flat)] +=
                    table[static_cast<std::size_t>(flat - stride)];
            }
        }
        return table;
    }

    /// Whether a mask is set at every cell of a box, from its summed-area table.
    ///
    /// \param table The table `summed_area` returned.
    /// \param shape The mask's per-axis extents.
    /// \param lo Per-axis lower corner, inclusive.
    /// \param hi Per-axis upper corner, exclusive.
    /// \return `true` iff the box holds no cleared cell.
    [[nodiscard]] static bool box_all_true(const std::vector<std::int64_t>& table,
                                           std::span<const std::int64_t> shape,
                                           std::span<const std::int64_t> lo,
                                           std::span<const std::int64_t> hi) {
        const auto d = shape.size();
        std::vector<std::int64_t> ext(d);
        for (std::size_t k = 0; k < d; ++k) {
            ext[k] = shape[k] + 1;
        }
        std::vector<std::int64_t> corner(d);
        std::int64_t total = 0;
        const std::int64_t corners = std::int64_t{1} << d;
        for (std::int64_t bits = 0; bits < corners; ++bits) {
            std::int64_t set = 0;
            for (std::size_t k = 0; k < d; ++k) {
                const bool upper = ((bits >> k) & 1) != 0;
                corner[k] = upper ? hi[k] : lo[k];
                set += static_cast<std::int64_t>(upper);
            }
            const std::int64_t sign = ((static_cast<std::int64_t>(d) - set) % 2 == 0) ? 1 : -1;
            total += sign * table[static_cast<std::size_t>(ravel(corner, ext))];
        }
        return total == 0;
    }

    /// Whether a mask is set anywhere in a box.
    ///
    /// Scanned directly rather than through a table: the caller is the truncation's
    /// per-function loop, whose boxes are the size of one function's support, and a
    /// table would be built once per level for a query that touches `(p + 1) ** dim`
    /// cells.
    ///
    /// \param mask A `count_of(shape)` mask, C-order.
    /// \param shape Per-axis extents.
    /// \param lo Per-axis lower corner, inclusive.
    /// \param hi Per-axis upper corner, exclusive.
    /// \return `true` iff some cell of the box is set.
    [[nodiscard]] static bool box_any_true(const std::vector<std::uint8_t>& mask,
                                           std::span<const std::int64_t> shape,
                                           std::span<const std::int64_t> lo,
                                           std::span<const std::int64_t> hi) {
        const auto d = shape.size();
        if (d == 0) {
            return !mask.empty() && mask[0] != 0U;
        }
        for (std::size_t k = 0; k < d; ++k) {
            if (lo[k] >= hi[k]) {
                return false;
            }
        }
        std::vector<std::int64_t> cursor(lo.begin(), lo.end());
        for (;;) {
            if (mask[static_cast<std::size_t>(ravel(cursor, shape))] != 0U) {
                return true;
            }
            std::size_t axis = d;
            bool done = false;
            while (axis > 0) {
                --axis;
                ++cursor[axis];
                if (cursor[axis] < hi[axis]) {
                    break;
                }
                cursor[axis] = lo[axis];
                if (axis == 0) {
                    done = true;
                }
            }
            if (done) {
                return false;
            }
        }
    }

    // -------------------------------------------------------------- construction

    /// Build the nested per-level tensor-product spaces.
    ///
    /// Level `l + 1` is level `l` with every direction subdivided by the grid's
    /// factor, skipping the directions whose factor is 1. Level 0 is the root handle
    /// itself, not a copy; see `level_space`.
    void build_level_spaces() {
        const auto d = static_cast<std::size_t>(dim());
        const std::span<const std::int64_t> factor = grid_->factor();
        level_spaces_.push_back(root_space_);

        std::vector<std::shared_ptr<const BsplineSpace1D<T>>> current(d);
        for (std::size_t k = 0; k < d; ++k) {
            current[k] = root_space_->space(static_cast<std::int64_t>(k));
        }
        for (std::int64_t level = 1; level <= grid_->max_level(); ++level) {
            for (std::size_t k = 0; k < d; ++k) {
                if (factor[k] == 1) {
                    continue;
                }
                current[k] = std::make_shared<const BsplineSpace1D<T>>(
                    subdivide<T>(*current[k], factor[k], regularity_[k]));
            }
            level_spaces_.push_back(std::make_shared<const BsplineSpace<T>>(current));
        }
    }

    /// Build the per-level, per-direction function-to-cell support.
    ///
    /// `first_basis_per_interval` gives the first function of each interval; inverting
    /// it gives each function its inclusive interval range. Pure integer work.
    ///
    /// \throws std::invalid_argument If some function has empty support, which means
    ///         the space is invalid.
    void build_support() {
        const auto d = static_cast<std::size_t>(dim());
        support_.resize(level_spaces_.size());
        for (std::size_t level = 0; level < level_spaces_.size(); ++level) {
            support_[level].resize(d);
            for (std::size_t k = 0; k < d; ++k) {
                const BsplineSpace1D<T>& space =
                    level_spaces_[level]->space_ref(static_cast<std::int64_t>(k));
                const std::span<const std::int64_t> first = space.first_basis_per_interval();
                const auto n_basis = static_cast<std::size_t>(space.num_basis());
                Support1D& sup = support_[level][k];
                sup.first_basis.assign(first.begin(), first.end());
                sup.first_cell.assign(n_basis, -1);
                sup.last_cell.assign(n_basis, -1);
                for (std::size_t interval = 0; interval < first.size(); ++interval) {
                    const std::int64_t lo = first[interval];
                    for (std::int64_t i = lo; i <= lo + space.degree(); ++i) {
                        const auto f = static_cast<std::size_t>(i);
                        if (sup.first_cell[f] < 0) {
                            sup.first_cell[f] = static_cast<std::int64_t>(interval);
                        }
                        sup.last_cell[f] = static_cast<std::int64_t>(interval);
                    }
                }
                // The oracle's type and its message: a `RuntimeError` naming EVERY
                // index with empty support, not the first. nanobind maps
                // `std::runtime_error` to `RuntimeError`, so the kind survives too.
                std::vector<std::int64_t> empty_support;
                for (std::size_t f = 0; f < n_basis; ++f) {
                    if (sup.first_cell[f] < 0) {
                        empty_support.push_back(static_cast<std::int64_t>(f));
                    }
                }
                if (!empty_support.empty()) {
                    throw std::runtime_error(
                        "B-spline function(s) with empty support detected at indices "
                        + list_repr(empty_support)
                        + ". This indicates an invalid B-spline space.");
                }
            }
        }
    }

    /// The per-direction basis counts of one level.
    ///
    /// \param level The level.
    /// \return `dim()` counts, in axis order.
    [[nodiscard]] std::vector<std::int64_t> level_num_basis(std::size_t level) const {
        const auto d = static_cast<std::size_t>(dim());
        std::vector<std::int64_t> counts(d);
        for (std::size_t k = 0; k < d; ++k) {
            counts[k] = level_spaces_[level]->space_ref(static_cast<std::int64_t>(k)).num_basis();
        }
        return counts;
    }

    /// `base ** exponent` for a per-axis refinement factor, in exact integers.
    ///
    /// The oracle forms these with Python's `**`, which cannot overflow. This cannot
    /// either, and the reason is the grid rather than this function: every exponent it
    /// is called with is bounded by the grid's own `max_level()`, and
    /// `HierarchicalGrid::level_cells_per_axis` already refused that level unless
    /// `root_cells * factor ** level` fits in `std::int64_t`. So `factor ** level` is
    /// at most a count the grid accepted, and a grid that would overflow here threw
    /// before this space could be built. Written out because the three call sites read
    /// like an unchecked power and a reader is right to ask.
    ///
    /// \param base The per-axis factor, at least 1.
    /// \param exponent The level gap, non-negative.
    /// \return `base ** exponent`.
    [[nodiscard]] static std::int64_t level_power(std::int64_t base,
                                                  std::int64_t exponent) noexcept {
        std::int64_t value = 1;
        for (std::int64_t step = 0; step < exponent; ++step) {
            value *= base;
        }
        return value;
    }

    /// The per-axis cell counts of one level of the grid.
    ///
    /// \param level The level.
    /// \return `dim()` counts, in axis order.
    [[nodiscard]] std::vector<std::int64_t> level_shape(std::int64_t level) const {
        const auto d = static_cast<std::size_t>(dim());
        std::vector<std::int64_t> shape(d);
        for (std::size_t k = 0; k < d; ++k) {
            shape[k] = grid_->level_cells_per_axis(level, static_cast<std::int64_t>(k));
        }
        return shape;
    }

    /// The level-`level` refined region: in the subdomain but not an active leaf.
    ///
    /// \param level The level.
    /// \return `1` where the level's subdomain is refined further, `0` elsewhere.
    [[nodiscard]] std::vector<std::uint8_t> refined_mask(std::int64_t level) const {
        const std::vector<std::uint8_t> subdomain = grid_->subdomain_mask(level);
        const std::vector<std::uint8_t> leaves = grid_->active_leaf_mask(level);
        std::vector<std::uint8_t> refined(subdomain.size());
        for (std::size_t i = 0; i < subdomain.size(); ++i) {
            refined[i] = static_cast<std::uint8_t>(subdomain[i] != 0U && leaves[i] == 0U);
        }
        return refined;
    }

    /// Compute the Kraft selection of active functions, level by level.
    ///
    /// A level-`l` function is selected iff its support box lies entirely in the
    /// level-`l` subdomain and not entirely in the further-refined region. Both tests
    /// are "is this box all ones in this mask", answered by a summed-area table over
    /// the mask's complement: `O(cells * dim)` to build and `O(2 ** dim)` per box,
    /// against the `O(box volume)` of a direct scan. Exact integer arithmetic, so it
    /// is the oracle's answer and not merely an equivalent one.
    void select_active_functions() {
        const auto d = static_cast<std::size_t>(dim());
        const auto n_levels = level_spaces_.size();
        active_funcs_.resize(n_levels);
        num_per_level_.assign(n_levels, 0);

        for (std::size_t level = 0; level < n_levels; ++level) {
            const auto l = static_cast<std::int64_t>(level);
            const std::vector<std::int64_t> shape = level_shape(l);
            const std::vector<std::uint8_t> subdomain = grid_->subdomain_mask(l);
            const std::vector<std::uint8_t> refined = refined_mask(l);

            // The subdomain's own bounding box bounds every selectable function's
            // support, and so bounds the candidate set per direction.
            std::vector<std::int64_t> bbox_lo(d, 0);
            std::vector<std::int64_t> bbox_hi(d, 0);
            if (!bounding_box(subdomain, shape, bbox_lo, bbox_hi)) {
                continue;
            }

            const std::vector<std::int64_t> sat_sub = summed_area(subdomain, shape);
            const std::vector<std::int64_t> sat_ref = summed_area(refined, shape);
            const std::vector<std::int64_t> num_basis = level_num_basis(level);

            std::vector<std::vector<std::int64_t>> candidates(d);
            bool empty_direction = false;
            for (std::size_t k = 0; k < d; ++k) {
                const Support1D& sup = support_[level][k];
                for (std::int64_t f = 0; f < num_basis[k]; ++f) {
                    const auto i = static_cast<std::size_t>(f);
                    if (sup.last_cell[i] >= bbox_lo[k] && sup.first_cell[i] < bbox_hi[k]) {
                        candidates[k].push_back(f);
                    }
                }
                empty_direction = empty_direction || candidates[k].empty();
            }
            if (empty_direction) {
                continue;
            }

            std::vector<std::int64_t>& selected = active_funcs_[level];
            std::vector<std::size_t> cursor(d, 0);
            std::vector<std::int64_t> box_lo(d);
            std::vector<std::int64_t> box_hi(d);
            std::vector<std::int64_t> multi(d);
            for (;;) {
                for (std::size_t k = 0; k < d; ++k) {
                    multi[k] = candidates[k][cursor[k]];
                    const auto i = static_cast<std::size_t>(multi[k]);
                    box_lo[k] = support_[level][k].first_cell[i];
                    box_hi[k] = support_[level][k].last_cell[i] + 1;
                }
                const bool inside = box_all_true(sat_sub, shape, box_lo, box_hi);
                const bool buried = box_all_true(sat_ref, shape, box_lo, box_hi);
                if (inside && !buried) {
                    selected.push_back(ravel(multi, num_basis));
                }
                std::size_t axis = d;
                bool done = d == 0;
                while (axis > 0) {
                    --axis;
                    ++cursor[axis];
                    if (cursor[axis] < candidates[axis].size()) {
                        break;
                    }
                    cursor[axis] = 0;
                    if (axis == 0) {
                        done = true;
                    }
                }
                if (done) {
                    break;
                }
            }
            // The enumeration is C-order over per-direction candidates that are
            // themselves increasing, so the flat indices already come out increasing.
            // The sort is the oracle's own step, costs nothing on a sorted range, and
            // is kept so the "sorted" contract does not rest on the enumeration order.
            std::sort(selected.begin(), selected.end());
        }

        func_offset_.assign(n_levels + 1, 0);
        for (std::size_t level = 0; level < n_levels; ++level) {
            num_per_level_[level] = static_cast<std::int64_t>(active_funcs_[level].size());
            func_offset_[level + 1] = func_offset_[level] + num_per_level_[level];
        }
        num_active_ = func_offset_.back();
    }

    /// Build the per-level, per-direction two-scale matrices.
    ///
    /// Entry `[m][k]` maps the level-`m` basis of direction `k` into the level-`m + 1`
    /// one: `B_i = sum_j alpha(j, i) B_j`. When the direction's factor is 1 the two
    /// spaces are the same and the matrix is the identity, which the recurrence
    /// produces without a special case.
    ///
    /// **Computed in `T` and widened to `double`**, in that order, because the oracle
    /// does: its kernel returns the knots' own dtype and `_build_oslo_matrices` wraps
    /// the result in `np.asarray(..., dtype=np.float64)`. Computing in `double`
    /// throughout would be a silent divergence at `float` storage that no shape or
    /// count would reveal.
    ///
    /// \return The matrices, indexed `[m][k]`.
    [[nodiscard]] std::vector<std::vector<TwoScale>> build_oslo_matrices() const {
        const auto d = static_cast<std::size_t>(dim());
        std::vector<std::vector<TwoScale>> matrices;
        for (std::size_t m = 0; m + 1 < level_spaces_.size(); ++m) {
            std::vector<TwoScale> per_direction(d);
            for (std::size_t k = 0; k < d; ++k) {
                const BsplineSpace1D<T>& old_space =
                    level_spaces_[m]->space_ref(static_cast<std::int64_t>(k));
                const BsplineSpace1D<T>& new_space =
                    level_spaces_[m + 1]->space_ref(static_cast<std::int64_t>(k));
                const std::vector<T> narrow =
                    oslo_matrix_1d<T>(old_space.degree(), old_space.knots(), new_space.knots());
                TwoScale& scale = per_direction[k];
                scale.rows = new_space.num_basis();
                scale.cols = old_space.num_basis();
                scale.alpha.reserve(narrow.size());
                for (const T value : narrow) {
                    scale.alpha.push_back(static_cast<double>(value));
                }
            }
            matrices.push_back(std::move(per_direction));
        }
        return matrices;
    }

    /// Refine a dense coefficient box from one level to the next.
    ///
    /// Applies, per direction, the two-scale matrix restricted to the current function
    /// box, growing the box to the band of non-zero finer functions. The band is the
    /// span from the **first** to the **last** non-zero row, exactly as the oracle
    /// takes it, so an interior all-zero row stays inside the box rather than
    /// splitting it.
    ///
    /// The contraction sums in index order. The oracle reaches `numpy.tensordot`,
    /// hence BLAS, whose order is the implementation's, so the two backends' outputs
    /// are a bounded comparison rather than a bitwise one; the file comment carries
    /// the bound and why it has no cancellation term.
    ///
    /// \param coeffs The coefficients over the box; replaced by the refined ones.
    /// \param box_lo Per-direction lower function index; updated in place.
    /// \param box_hi Per-direction upper index, exclusive; updated in place.
    /// \param oslo_m This level transition's matrices, per direction.
    /// \param scratch A reusable buffer, so the per-function loop does not reallocate.
    /// \throws std::invalid_argument If a direction's matrix slice is entirely zero,
    ///         which means a degenerate box or an invalid knot refinement.
    void refine_box(std::vector<double>& coeffs, std::vector<std::int64_t>& box_lo,
                    std::vector<std::int64_t>& box_hi, const std::vector<TwoScale>& oslo_m,
                    std::vector<double>& scratch) const {
        const auto d = static_cast<std::size_t>(dim());
        for (std::size_t k = 0; k < d; ++k) {
            const TwoScale& scale = oslo_m[k];
            const std::int64_t width = box_hi[k] - box_lo[k];

            std::int64_t new_lo = -1;
            std::int64_t new_hi = -1;
            for (std::int64_t row = 0; row < scale.rows; ++row) {
                bool non_zero = false;
                for (std::int64_t col = box_lo[k]; col < box_hi[k] && !non_zero; ++col) {
                    non_zero = scale.alpha[static_cast<std::size_t>(row * scale.cols + col)]
                               != 0.0;
                }
                if (non_zero) {
                    if (new_lo < 0) {
                        new_lo = row;
                    }
                    new_hi = row + 1;
                }
            }
            if (new_lo < 0) {
                // The oracle's text, character for character, em dash included, so a
                // caller matching on it keeps working when the backend changes.
                throw std::invalid_argument(
                    "_refine_box: Oslo matrix slice for direction " + std::to_string(k)
                    + " (columns [" + std::to_string(box_lo[k]) + ":"
                    + std::to_string(box_hi[k])
                    + "]) is entirely zero \u2014 degenerate or invalid knot refinement.");
            }
            const std::int64_t new_width = new_hi - new_lo;

            // The box is row-major, so contracting axis `k` sees the array as
            // `outer x width x inner` and produces `outer x new_width x inner`.
            std::int64_t outer = 1;
            for (std::size_t a = 0; a < k; ++a) {
                outer *= box_hi[a] - box_lo[a];
            }
            std::int64_t inner = 1;
            for (std::size_t a = k + 1; a < d; ++a) {
                inner *= box_hi[a] - box_lo[a];
            }

            scratch.assign(static_cast<std::size_t>(outer * new_width * inner), 0.0);
            for (std::int64_t o = 0; o < outer; ++o) {
                for (std::int64_t i = 0; i < new_width; ++i) {
                    const std::int64_t row = new_lo + i;
                    for (std::int64_t j = 0; j < width; ++j) {
                        const double a = scale.alpha[static_cast<std::size_t>(
                            row * scale.cols + box_lo[k] + j)];
                        if (a == 0.0) {
                            continue;
                        }
                        const auto source = static_cast<std::size_t>((o * width + j) * inner);
                        const auto target =
                            static_cast<std::size_t>((o * new_width + i) * inner);
                        for (std::int64_t n = 0; n < inner; ++n) {
                            scratch[target + static_cast<std::size_t>(n)] +=
                                a * coeffs[source + static_cast<std::size_t>(n)];
                        }
                    }
                }
            }
            coeffs.swap(scratch);
            box_lo[k] = new_lo;
            box_hi[k] = new_hi;
        }
    }

    /// Zero the coefficients sitting on active functions of the refined level.
    ///
    /// This is the truncation itself: a hierarchical function loses its components on
    /// the finer functions that are themselves active.
    ///
    /// \param coeffs The coefficients over the box; modified in place.
    /// \param box_lo Per-direction lower function index.
    /// \param box_hi Per-direction upper index, exclusive.
    /// \param active_at_level Sorted flat indices of the refined level's active
    ///        functions.
    /// \param num_basis The refined level's per-direction basis counts.
    /// \return `true` iff at least one coefficient was zeroed.
    [[nodiscard]] bool truncate_box(std::vector<double>& coeffs,
                                    std::span<const std::int64_t> box_lo,
                                    std::span<const std::int64_t> box_hi,
                                    const std::vector<std::int64_t>& active_at_level,
                                    std::span<const std::int64_t> num_basis) const {
        const auto d = static_cast<std::size_t>(dim());
        if (d == 0) {
            return false;
        }
        // The same guard `box_any_true` carries. `refine_box` throws rather than
        // returning a zero-width band, so an empty box does not arise from the one
        // caller today -- but the oracle's numpy version degrades to "nothing zeroed"
        // for one, and holding an invariant across two functions is not a contract.
        for (std::size_t k = 0; k < d; ++k) {
            if (box_lo[k] >= box_hi[k]) {
                return false;
            }
        }
        std::vector<std::int64_t> cursor(box_lo.begin(), box_lo.end());
        std::size_t offset = 0;
        bool zeroed = false;
        for (;;) {
            const std::int64_t flat = ravel(cursor, num_basis);
            if (std::binary_search(active_at_level.begin(), active_at_level.end(), flat)) {
                coeffs[offset] = 0.0;
                zeroed = true;
            }
            ++offset;
            std::size_t axis = d;
            bool done = false;
            while (axis > 0) {
                --axis;
                ++cursor[axis];
                if (cursor[axis] < box_hi[axis]) {
                    break;
                }
                cursor[axis] = box_lo[axis];
                if (axis == 0) {
                    done = true;
                }
            }
            if (done) {
                return zeroed;
            }
        }
    }

    /// Build the truncated-coefficient map.
    ///
    /// For each active function that straddles a finer refinement boundary, the
    /// function is expressed in successively finer bases by the two-scale relation and
    /// its components on active finer functions are zeroed, until its support no longer
    /// reaches deeper refinement. Truncation is applied at **every** level in the
    /// chain, not only the first. A function whose support enters a refined region but
    /// whose coefficient box never overlaps an active finer function needs no
    /// truncation and is stored nowhere, which is what keeps this map small.
    ///
    /// The entries are appended in increasing `dof`, so `truncated()` can binary
    /// search them; a hash map would give the same answers in an order that depends on
    /// the container.
    void compute_truncated_coefficients() {
        const auto n_levels = level_spaces_.size();
        if (n_levels < 2) {
            return;
        }
        const auto d = static_cast<std::size_t>(dim());
        const std::vector<std::vector<TwoScale>> oslo = build_oslo_matrices();

        std::vector<std::vector<std::uint8_t>> refined(n_levels);
        std::vector<std::vector<std::int64_t>> shape(n_levels);
        for (std::size_t level = 0; level < n_levels; ++level) {
            const auto l = static_cast<std::int64_t>(level);
            shape[level] = level_shape(l);
            refined[level] = refined_mask(l);
        }

        std::vector<std::int64_t> box_lo(d);
        std::vector<std::int64_t> box_hi(d);
        std::vector<std::int64_t> cell_lo(d);
        std::vector<std::int64_t> cell_hi(d);
        std::vector<double> coeffs;
        std::vector<double> scratch;

        for (std::size_t level = 0; level + 1 < n_levels; ++level) {
            const std::vector<std::int64_t> num_basis = level_num_basis(level);
            const std::int64_t offset = func_offset_[level];
            const std::vector<std::int64_t>& active = active_funcs_[level];

            for (std::size_t pos = 0; pos < active.size(); ++pos) {
                unravel(active[pos], num_basis, std::span<std::int64_t>(box_lo));
                for (std::size_t k = 0; k < d; ++k) {
                    box_hi[k] = box_lo[k] + 1;
                }
                coeffs.assign(1, 1.0);
                std::size_t rep = level;
                bool any_zeroed = false;

                for (std::size_t m = level; m + 1 < n_levels; ++m) {
                    for (std::size_t k = 0; k < d; ++k) {
                        const Support1D& sup = support_[m][k];
                        cell_lo[k] = sup.first_cell[static_cast<std::size_t>(box_lo[k])];
                        cell_hi[k] = sup.last_cell[static_cast<std::size_t>(box_hi[k] - 1)] + 1;
                    }
                    if (!box_any_true(refined[m], shape[m], cell_lo, cell_hi)) {
                        break;
                    }
                    refine_box(coeffs, box_lo, box_hi, oslo[m], scratch);
                    rep = m + 1;
                    const bool zeroed = truncate_box(coeffs, box_lo, box_hi, active_funcs_[rep],
                                                     level_num_basis(rep));
                    any_zeroed = any_zeroed || zeroed;
                }

                if (!any_zeroed) {
                    continue;
                }
                TruncatedEntry entry;
                entry.dof = offset + static_cast<std::int64_t>(pos);
                entry.rep_level = static_cast<std::int64_t>(rep);
                entry.box_lo.assign(box_lo.begin(), box_lo.end());
                entry.shape.resize(d);
                for (std::size_t k = 0; k < d; ++k) {
                    entry.shape[k] = box_hi[k] - box_lo[k];
                }
                entry.coeffs = coeffs;
                truncated_.push_back(std::move(entry));
            }
        }
    }

    // -------------------------------------------------------- contribution table

    /// The contribution table, built on first use.
    ///
    /// \return The table, valid for as long as this space is.
    [[nodiscard]] const ContributionTable& contributions_table() const {
        return contributions_.get([this] { return build_contribution_table(); });
    }

    /// Sweep every cell and record the active functions supported on it.
    ///
    /// A cell at level `L` is covered by the functions of every level `l <= L` whose
    /// support contains the cell's ancestor at `l`; those are `degree + 1` consecutive
    /// functions per direction, named by `first_basis`. Each candidate is looked up in
    /// the level's sorted active set, and the ones present become entries.
    ///
    /// \return The filled table, including `max_per_cell`.
    [[nodiscard]] ContributionTable build_contribution_table() const {
        const auto d = static_cast<std::size_t>(dim());
        const auto num_cells = static_cast<std::size_t>(grid_->num_cells());
        const std::span<const std::int64_t> factor = grid_->factor();
        const std::span<const std::int64_t> deg = degrees();

        ContributionTable table;
        table.offset.assign(num_cells + 1, 0);

        std::vector<std::int64_t> cell_midx(d);
        std::vector<std::int64_t> at_level(d);
        std::vector<std::int64_t> first(d);
        std::vector<std::int64_t> multi(d);
        std::vector<std::pair<std::int64_t, std::size_t>> ordering;

        for (std::size_t cid = 0; cid < num_cells; ++cid) {
            const std::int64_t cell_level = grid_->cell_level(static_cast<std::int64_t>(cid));
            grid_->cell_multi_index(static_cast<std::int64_t>(cid),
                                    std::span<std::int64_t>(cell_midx));
            const std::size_t begin = table.dof.size();

            for (std::int64_t level = 0; level <= cell_level; ++level) {
                const auto l = static_cast<std::size_t>(level);
                for (std::size_t k = 0; k < d; ++k) {
                    const std::int64_t divisor = level_power(factor[k], cell_level - level);
                    at_level[k] = cell_midx[k] / divisor;
                    first[k] = support_[l][k].first_basis[static_cast<std::size_t>(at_level[k])];
                }
                const std::vector<std::int64_t> num_basis = level_num_basis(l);
                const std::vector<std::int64_t>& active = active_funcs_[l];
                const std::int64_t offset = func_offset_[l];

                std::vector<std::int64_t> cursor(first.begin(), first.end());
                for (;;) {
                    const std::int64_t flat = ravel(cursor, num_basis);
                    const auto it =
                        std::lower_bound(active.begin(), active.end(), flat);
                    if (it != active.end() && *it == flat) {
                        table.dof.push_back(offset
                                            + static_cast<std::int64_t>(it - active.begin()));
                        table.level.push_back(level);
                        table.multi.insert(table.multi.end(), cursor.begin(), cursor.end());
                    }
                    std::size_t axis = d;
                    bool done = d == 0;
                    while (axis > 0) {
                        --axis;
                        ++cursor[axis];
                        if (cursor[axis] <= first[axis] + deg[axis]) {
                            break;
                        }
                        cursor[axis] = first[axis];
                        if (axis == 0) {
                            done = true;
                        }
                    }
                    if (done) {
                        break;
                    }
                }
            }

            // Sorted by global dof. The sweep already produces them in that order --
            // levels ascend and each level's candidates are enumerated in increasing
            // flat index -- so this is the oracle's own step kept for its contract
            // rather than for its effect.
            const std::size_t count = table.dof.size() - begin;
            ordering.clear();
            ordering.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                ordering.emplace_back(table.dof[begin + i], i);
            }
            std::sort(ordering.begin(), ordering.end());
            std::vector<std::int64_t> sorted_dof(count);
            std::vector<std::int64_t> sorted_level(count);
            std::vector<std::int64_t> sorted_multi(count * d);
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t from = ordering[i].second;
                sorted_dof[i] = table.dof[begin + from];
                sorted_level[i] = table.level[begin + from];
                for (std::size_t k = 0; k < d; ++k) {
                    sorted_multi[i * d + k] = table.multi[(begin + from) * d + k];
                }
            }
            for (std::size_t i = 0; i < count; ++i) {
                table.dof[begin + i] = sorted_dof[i];
                table.level[begin + i] = sorted_level[i];
                for (std::size_t k = 0; k < d; ++k) {
                    table.multi[(begin + i) * d + k] = sorted_multi[i * d + k];
                }
            }

            table.offset[cid + 1] = static_cast<std::int64_t>(table.dof.size());
            table.max_per_cell =
                std::max(table.max_per_cell, static_cast<std::int64_t>(count));
        }
        return table;
    }

    // ------------------------------------------------------- refinement helpers

    /// A grid equal to this space's but sharing nothing with it.
    ///
    /// `refine_cells` with no ids goes through `HierarchicalGrid::rebuilt`, which
    /// gives a fresh grid with the same cell decomposition, empty caches and no
    /// inherited tags. That is what the oracle's `grid._copy()` is for: two spaces
    /// must never share a grid, or a tag set through one becomes visible through the
    /// other.
    ///
    /// \return The detached grid.
    [[nodiscard]] std::shared_ptr<grid_type> detached_grid() const {
        return std::make_shared<grid_type>(
            grid_->refine_cells(std::span<const std::int64_t>{}));
    }

    /// A new space over a given grid, carrying this one's other state.
    ///
    /// \param grid The grid the new space is built on.
    /// \return The new space.
    [[nodiscard]] THBSplineSpace rebound(std::shared_ptr<grid_type> grid) const {
        return THBSplineSpace(root_space_, std::move(grid), truncate_, regularity_);
    }

    /// Refine the marked cells and rebuild.
    ///
    /// \param marked `(level, midx)` pairs captured against this space's own grid.
    /// \param admissible_class The class to maintain, or empty for no grading.
    /// \return The new space.
    [[nodiscard]] THBSplineSpace refine_marked(
        const std::vector<Marked>& marked,
        std::optional<std::int64_t> admissible_class) const {
        std::shared_ptr<grid_type> current = grid_;
        bool changed = false;
        for (const Marked& cell : marked) {
            if (admissible_class.has_value()) {
                refine_recursive(current, cell.level, cell.midx, *admissible_class, changed);
            } else if (current->is_active_leaf(cell.level,
                                               std::span<const std::int64_t>(cell.midx))) {
                current = refined_once(*current, cell.level, cell.midx);
                changed = true;
            }
        }
        return rebound(changed ? current : detached_grid());
    }

    /// One cell refined, as a fresh grid.
    ///
    /// \param grid The grid to refine.
    /// \param level The cell's level.
    /// \param midx The cell's per-axis index at that level.
    /// \return The refined grid.
    [[nodiscard]] static std::shared_ptr<grid_type> refined_once(
        const grid_type& grid, std::int64_t level, const std::vector<std::int64_t>& midx) {
        std::vector<std::int64_t> hi(midx.size());
        for (std::size_t k = 0; k < midx.size(); ++k) {
            hi[k] = midx[k] + 1;
        }
        return std::make_shared<grid_type>(grid.refine(level,
                                                       std::span<const std::int64_t>(midx),
                                                       std::span<const std::int64_t>(hi)));
    }

    /// Refine one cell, grading its refinement neighbourhood first.
    ///
    /// Carraturo et al. (2019, Alg. 4): every cell of the neighbourhood is refined --
    /// recursively, at the coarser level `level - m + 1` -- before the cell itself.
    /// Each step queries the grid the previous one produced.
    ///
    /// The recursion depth is bounded by `level`, which is bounded by the grid's own
    /// maximum level, so it terminates for the same reason the oracle's does.
    ///
    /// \param grid The grid, rebound in place as refinements happen.
    /// \param level The cell's level.
    /// \param midx The cell's per-axis index at that level.
    /// \param m The admissibility class, at least 2.
    /// \param changed Set when any refinement actually happened.
    void refine_recursive(std::shared_ptr<grid_type>& grid, std::int64_t level,
                          const std::vector<std::int64_t>& midx, std::int64_t m,
                          bool& changed) const {
        for (const Marked& neighbour : refinement_neighborhood(level, midx, m, *grid)) {
            refine_recursive(grid, neighbour.level, neighbour.midx, m, changed);
        }
        if (grid->is_active_leaf(level, std::span<const std::int64_t>(midx))) {
            grid = refined_once(*grid, level, midx);
            changed = true;
        }
    }

    /// The refinement neighbourhood of a cell, for class `m`.
    ///
    /// Carraturo et al. (2019, Def. 3.4): the cells at level `level - m + 1` that are
    /// parents of a level-`level - m + 2` cell touched by any B-spline whose support
    /// covers the containing cell of `(level, midx)` at that level. Only the ones that
    /// are currently active leaves are returned.
    ///
    /// \param level The cell's level.
    /// \param midx The cell's per-axis index at that level.
    /// \param m The admissibility class, at least 2.
    /// \param grid The grid whose active set is queried.
    /// \return The neighbourhood, possibly empty.
    [[nodiscard]] std::vector<Marked> refinement_neighborhood(
        std::int64_t level, const std::vector<std::int64_t>& midx, std::int64_t m,
        const grid_type& grid) const {
        std::vector<Marked> out;
        const std::int64_t k_nbr = level - m + 1;
        if (k_nbr < 0) {
            return out;
        }
        const auto d = static_cast<std::size_t>(dim());
        // `k_ext = k_nbr + 1 <= level`, and `level` never exceeds the maximum level of
        // the grid this space was built on, so the support of that level exists.
        const std::int64_t k_ext = level - m + 2;
        const std::vector<Support1D>& support_ext =
            support_[static_cast<std::size_t>(k_ext)];
        const std::span<const std::int64_t> factor = grid_->factor();
        const std::span<const std::int64_t> deg = degrees();

        std::vector<std::int64_t> lo(d);
        std::vector<std::int64_t> hi(d);
        for (std::size_t k = 0; k < d; ++k) {
            const std::int64_t divisor = level_power(factor[k], level - k_ext);
            const std::int64_t q = midx[k] / divisor;
            const std::int64_t first = support_ext[k].first_basis[static_cast<std::size_t>(q)];
            const std::int64_t s_lo =
                support_ext[k].first_cell[static_cast<std::size_t>(first)];
            const std::int64_t s_hi =
                support_ext[k].last_cell[static_cast<std::size_t>(first + deg[k])] + 1;
            lo[k] = s_lo / factor[k];
            hi[k] = (s_hi - 1) / factor[k] + 1;
        }

        std::vector<std::int64_t> cursor(lo.begin(), lo.end());
        for (;;) {
            if (grid.is_active_leaf(k_nbr, std::span<const std::int64_t>(cursor))) {
                out.push_back(Marked{k_nbr, cursor});
            }
            std::size_t axis = d;
            // A dimensionless space has exactly one combination, the empty
            // multi-index, which is what `itertools.product()` gives the oracle and
            // what every other odometer in this file does. An early return for
            // `d == 0` would make this one function disagree with all of them.
            bool done = d == 0;
            while (axis > 0) {
                --axis;
                ++cursor[axis];
                if (cursor[axis] < hi[axis]) {
                    break;
                }
                cursor[axis] = lo[axis];
                if (axis == 0) {
                    done = true;
                }
            }
            if (done) {
                return out;
            }
        }
    }

    /// Whether reactivating a parent keeps the mesh admissible of class `m`.
    ///
    /// Carraturo et al. (2019, Def. 3.5): the coarsening neighbourhood is the set of
    /// active cells at level `parent_level + m` inside the multilevel support
    /// extension, at level `parent_level + 1`, of the parent's children. Empty means
    /// the coarsening is admissible.
    ///
    /// \param parent_level The level of the parent being considered.
    /// \param pmidx Its per-axis index at that level.
    /// \param m The admissibility class, at least 2.
    /// \param grid The grid whose active set is queried.
    /// \return `true` iff the neighbourhood is empty.
    ///
    /// \note Assumes `parent_level + 1 < num_levels()`, which the caller guarantees:
    ///       a parent exists only because a marked cell sat one level below it on this
    ///       space's own grid.
    [[nodiscard]] bool coarsening_neighborhood_empty(std::int64_t parent_level,
                                                     const std::vector<std::int64_t>& pmidx,
                                                     std::int64_t m,
                                                     const grid_type& grid) const {
        const auto d = static_cast<std::size_t>(dim());
        if (d == 0) {
            return true;
        }
        const std::span<const std::int64_t> factor = grid_->factor();
        const std::span<const std::int64_t> deg = degrees();
        const std::vector<Support1D>& support =
            support_[static_cast<std::size_t>(parent_level) + 1];

        std::vector<std::int64_t> ext_lo(d);
        std::vector<std::int64_t> ext_hi(d);
        for (std::size_t k = 0; k < d; ++k) {
            const std::int64_t c_lo = pmidx[k] * factor[k];
            const std::int64_t c_hi = (pmidx[k] + 1) * factor[k];
            const std::int64_t f_min =
                support[k].first_basis[static_cast<std::size_t>(c_lo)];
            const std::int64_t f_max =
                support[k].first_basis[static_cast<std::size_t>(c_hi - 1)] + deg[k];
            ext_lo[k] = support[k].first_cell[static_cast<std::size_t>(f_min)];
            ext_hi[k] = support[k].last_cell[static_cast<std::size_t>(f_max)] + 1;
        }

        const std::int64_t target = parent_level + m;
        if (target > grid.max_level()) {
            return true;
        }
        std::vector<std::int64_t> box_lo(d);
        std::vector<std::int64_t> box_hi(d);
        for (std::size_t k = 0; k < d; ++k) {
            // `target <= max_level` above bounds `m - 1`, which is what makes this
            // power safe; `level_power` carries the argument.
            const std::int64_t scale = level_power(factor[k], m - 1);
            box_lo[k] = ext_lo[k] * scale;
            box_hi[k] = ext_hi[k] * scale;
        }

        const auto [lo, hi] = grid.active_blocks(target);
        for (std::size_t b = 0; b < lo.extent(0); ++b) {
            bool overlaps = true;
            for (std::size_t k = 0; k < d && overlaps; ++k) {
                overlaps = std::max(box_lo[k], lo(b, k)) < std::min(box_hi[k], hi(b, k));
            }
            if (overlaps) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<const BsplineSpace<T>> root_space_;    ///< The level-0 space.
    std::shared_ptr<grid_type> grid_;                      ///< The hierarchy.
    bool truncate_ = true;                                 ///< Whether the basis is truncated.
    std::vector<std::optional<std::int64_t>> regularity_;  ///< Per-direction continuity.

    std::vector<std::shared_ptr<const BsplineSpace<T>>> level_spaces_;  ///< Per level.
    std::vector<std::vector<Support1D>> support_;          ///< Per level, per direction.
    std::vector<std::vector<std::int64_t>> active_funcs_;  ///< Per level, sorted flats.
    std::vector<std::int64_t> num_per_level_;              ///< Per level, the active count.
    std::vector<std::int64_t> func_offset_;                ///< Per level, the global base.
    std::int64_t num_active_ = 0;                          ///< The total active count.
    std::vector<TruncatedEntry> truncated_;                ///< Sorted by `dof`.

    LazySlot<ContributionTable> contributions_;  ///< Filled for all cells at once.
};

}  // namespace pantr::bspline
