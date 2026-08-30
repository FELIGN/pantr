/// \file
/// `num_local_facets` is not specialisable.
///
/// `facet_tags_` is sized `2 * ndim` once, at construction, so a grid that reported a
/// different facet count would desynchronise the registry from the geometry -- and it
/// would do so silently, since nothing reads the two together. The Python version
/// leaves the coupling implicit; holding the sizes in the mixin is what makes it
/// assertable, so it is asserted rather than left to be discovered.

#include <cstdint>
#include <optional>
#include <span>

#include "pantr/grid/grid.hpp"

class ExtraFacetsGrid;

template <>
struct pantr::grid::grid_traits<ExtraFacetsGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::none;
};

class ExtraFacetsGrid : public pantr::grid::GridBase<ExtraFacetsGrid> {
  public:
    ExtraFacetsGrid() : pantr::grid::GridBase<ExtraFacetsGrid>(2, 1) {}

    void cell_bounds(std::int64_t, std::span<double> lo, std::span<double> hi) const {
        lo[0] = 0.0;
        lo[1] = 0.0;
        hi[0] = 1.0;
        hi[1] = 1.0;
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double>) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t,
                                                                    std::int64_t) const {
        return std::nullopt;
    }

    // The defect: a triangle's worth of facets on a box grid whose tag registry has four.
    [[nodiscard]] std::int64_t num_local_facets(std::int64_t) const { return 3; }
};

PANTR_GRID_CENSUS(ExtraFacetsGrid);
