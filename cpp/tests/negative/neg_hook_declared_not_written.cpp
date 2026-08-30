/// \file
/// A hook declared in the grid's `grid_traits` but never written.
///
/// The other direction of the same disagreement. Left unchecked it is a claim the
/// bitmask makes and the class does not honour, so the default runs where a reader of
/// the traits expects a specialisation -- and the census is what makes the bitmask
/// trustworthy enough to read at all. Also not in the ticket's list.

#include <cstdint>
#include <optional>
#include <span>

#include "pantr/grid/grid.hpp"

class UnwrittenHookGrid;

template <>
struct pantr::grid::grid_traits<UnwrittenHookGrid> {
    using scalar_type = double;
    // The defect: the class below writes no `boundary_facets`.
    static constexpr Hook hooks = Hook::boundary_facets;
};

class UnwrittenHookGrid : public pantr::grid::GridBase<UnwrittenHookGrid> {
  public:
    UnwrittenHookGrid() : pantr::grid::GridBase<UnwrittenHookGrid>(1, 1) {}

    void cell_bounds(std::int64_t, std::span<double> lo, std::span<double> hi) const {
        lo[0] = 0.0;
        hi[0] = 1.0;
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double>) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t,
                                                                    std::int64_t) const {
        return std::nullopt;
    }
};

PANTR_GRID_CENSUS(UnwrittenHookGrid);
