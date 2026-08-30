/// \file
/// A hook whose return type differs from the default it replaces.
///
/// A `requires`-based probe answers `true` here, which is why detection compares
/// member-pointer types instead. The census then asserts the declared hook's exact
/// signature, so the width of the returned integer is not something a caller has to
/// discover at the call site.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pantr/grid/grid.hpp"

class NarrowReturnGrid;

template <>
struct pantr::grid::grid_traits<NarrowReturnGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::locate_many;
};

class NarrowReturnGrid : public pantr::grid::GridBase<NarrowReturnGrid> {
  public:
    NarrowReturnGrid() : pantr::grid::GridBase<NarrowReturnGrid>(1, 1) {}

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

    // The defect: int32 where the default returns int64.
    [[nodiscard]] std::vector<std::int32_t> locate_many(pantr::span2d<const double>) const {
        return {};
    }
};

PANTR_GRID_CENSUS(NarrowReturnGrid);
