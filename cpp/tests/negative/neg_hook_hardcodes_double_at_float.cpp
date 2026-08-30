/// \file
/// A hook that hard-codes `double`, censused at a `float` grid.
///
/// It is well formed at `double` and wrong at `float`, so nothing but an instantiation
/// at the second scalar can see it. That is what the `float` census device is for: it
/// forces every default body and every declared hook's signature at a scalar no binding
/// registers, and it opens no parity claim because it is never bound and never compared.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pantr/grid/grid.hpp"

template <class T>
class HardCodedScalarGrid;

template <class T>
struct pantr::grid::grid_traits<HardCodedScalarGrid<T>> {
    using scalar_type = T;
    static constexpr Hook hooks = Hook::locate_many;
};

template <class T>
class HardCodedScalarGrid : public pantr::grid::GridBase<HardCodedScalarGrid<T>> {
  public:
    HardCodedScalarGrid() : pantr::grid::GridBase<HardCodedScalarGrid<T>>(1, 1) {}

    void cell_bounds(std::int64_t, std::span<T> lo, std::span<T> hi) const {
        lo[0] = T{0};
        hi[0] = T{1};
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const T>) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t,
                                                                    std::int64_t) const {
        return std::nullopt;
    }

    // The defect: `double` rather than `T`. Correct at one instantiation, wrong at the other.
    [[nodiscard]] std::vector<std::int64_t> locate_many(pantr::span2d<const double>) const {
        return {};
    }
};

using FloatGrid = HardCodedScalarGrid<float>;

PANTR_GRID_CENSUS(FloatGrid);
