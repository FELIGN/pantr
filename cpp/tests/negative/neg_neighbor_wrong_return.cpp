/// \file
/// A `neighbor_across_facet` returning a sentinel instead of an optional.
///
/// The third primitive, and the most dangerous of the three to get wrong: every generic
/// default that walks the connectivity -- `neighbors`, `is_mesh_boundary_facet`,
/// `hanging_neighbors`, `boundary_facets` -- decides "is this a boundary?" by asking
/// whether the optional is engaged. A grid returning `-1` for "no neighbour" would make
/// `std::optional<std::int64_t>(-1)` engaged, so every boundary facet would read as
/// interior and `boundary_facets()` would come back empty, with nothing to notice.
///
/// `GridLike` rejects it because the return type is part of the member-pointer type.

#include <cstdint>
#include <optional>
#include <span>

#include "pantr/grid/grid.hpp"

class SentinelNeighborGrid;

template <>
struct pantr::grid::grid_traits<SentinelNeighborGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::none;
};

class SentinelNeighborGrid : public pantr::grid::GridBase<SentinelNeighborGrid> {
  public:
    SentinelNeighborGrid() : pantr::grid::GridBase<SentinelNeighborGrid>(1, 1) {}

    void cell_bounds(std::int64_t, std::span<double> lo, std::span<double> hi) const {
        lo[0] = 0.0;
        hi[0] = 1.0;
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double>) const {
        return std::nullopt;
    }

    // The defect: a sentinel where the contract is an optional.
    [[nodiscard]] std::int64_t neighbor_across_facet(std::int64_t, std::int64_t) const {
        return -1;
    }
};

PANTR_GRID_CENSUS(SentinelNeighborGrid);
