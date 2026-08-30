/// \file
/// A hook written but not declared in the grid's `grid_traits`.
///
/// This is the failure mode that decided the dispatch mechanism. Under name hiding the
/// hook RUNS, so the bitmask and the class disagree and nothing at the call site says
/// so; under the alternative -- dispatch reading the bitmask -- the hook is silently
/// ignored and the default runs, which is a wrong answer with no diagnostic at all.
/// Either way the disagreement has to be a compile error, and this is the direction the
/// ticket's acceptance criteria did not list.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pantr/grid/grid.hpp"

class UndeclaredHookGrid;

template <>
struct pantr::grid::grid_traits<UndeclaredHookGrid> {
    using scalar_type = double;
    // The defect: the class below writes `boundary_facets` and this says it does not.
    static constexpr Hook hooks = Hook::none;
};

class UndeclaredHookGrid : public pantr::grid::GridBase<UndeclaredHookGrid> {
  public:
    UndeclaredHookGrid() : pantr::grid::GridBase<UndeclaredHookGrid>(1, 1) {}

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

    [[nodiscard]] std::vector<std::int64_t> boundary_facets() const { return {}; }
};

PANTR_GRID_CENSUS(UndeclaredHookGrid);
