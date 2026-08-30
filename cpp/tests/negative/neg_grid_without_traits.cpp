/// \file
/// A grid that forgets its `grid_traits` specialisation is rejected at its own definition.
///
/// The primary template is deliberately left undefined so this is the diagnostic, rather
/// than a failure inside whichever default first needed the scalar. `grid_traits` is
/// named in the message, which is what makes the mistake self-explaining.

#include <cstdint>
#include <optional>
#include <span>

#include "pantr/grid/grid.hpp"

class TraitlessGrid : public pantr::grid::GridBase<TraitlessGrid> {
  public:
    TraitlessGrid() : pantr::grid::GridBase<TraitlessGrid>(1, 1) {}

    void cell_bounds(std::int64_t, std::span<double>, std::span<double>) const {}

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double>) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t,
                                                                    std::int64_t) const {
        return std::nullopt;
    }
};
