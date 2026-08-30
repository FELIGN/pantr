/// \file
/// A `cell_bounds` that cannot write its output is not a grid.
///
/// This is the trap the concept is written against, and it is on the PRIMITIVES rather
/// than among the hooks, where the ticket placed it. `std::span<T>` converts implicitly
/// to `std::span<const T>`, so a callability-based concept -- `requires(std::span<T>
/// out) { g.cell_bounds(cid, out, out); }` -- is satisfied by the grid below, which
/// compiles and hands back whatever was already in the caller's buffer. Measured
/// accepted by g++ 14.4, g++ 10.5, clang++ 18.1 and clang++ 10.0 alike under
/// `-Wall -Wextra -Werror`. Pinning the primitive to an exact member-pointer type is
/// what rejects it.

#include <cstdint>
#include <optional>
#include <span>

#include "pantr/grid/grid.hpp"

class ConstSpanGrid;

template <>
struct pantr::grid::grid_traits<ConstSpanGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::none;
};

class ConstSpanGrid : public pantr::grid::GridBase<ConstSpanGrid> {
  public:
    ConstSpanGrid() : pantr::grid::GridBase<ConstSpanGrid>(1, 1) {}

    // The defect: the output spans are const, so nothing can be written through them.
    void cell_bounds(std::int64_t, std::span<const double>, std::span<const double>) const {}

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double>) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t,
                                                                    std::int64_t) const {
        return std::nullopt;
    }
};

PANTR_GRID_CENSUS(ConstSpanGrid);
