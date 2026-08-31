/// \file
/// A `locate` whose point argument is a reference to a span, not a span.
///
/// The same shape as the `cell_bounds` trap on the neighbouring file, aimed at the
/// second primitive: the body compiles, the call `self().locate(pt)` binds, and only
/// the exact member-pointer form of `GridLike` notices that the signature is not the
/// one every generic default was written against. The ticket named the trap on the
/// hooks; it lands on the primitives, and it lands on all three of them.

#include <cstdint>
#include <optional>
#include <span>

#include "pantr/grid/grid.hpp"

class RefPointGrid;

template <>
struct pantr::grid::grid_traits<RefPointGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::none;
};

class RefPointGrid : public pantr::grid::GridBase<RefPointGrid> {
  public:
    RefPointGrid() : pantr::grid::GridBase<RefPointGrid>(1, 1) {}

    void cell_bounds(std::int64_t, std::span<double> lo, std::span<double> hi) const {
        lo[0] = 0.0;
        hi[0] = 1.0;
    }

    // The defect: `const std::span<const double>&` rather than `std::span<const double>`.
    [[nodiscard]] std::optional<std::int64_t> locate(const std::span<const double>&) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t,
                                                                    std::int64_t) const {
        return std::nullopt;
    }
};

PANTR_GRID_CENSUS(RefPointGrid);
