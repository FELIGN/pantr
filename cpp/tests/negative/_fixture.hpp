#pragma once

/// \file
/// The well-formed grid every negative translation unit starts from.
///
/// Each of them introduces exactly ONE defect on top of this, so that the diagnostic
/// they are matched against can only have come from that defect. Keeping the fixture
/// shared is also what keeps the files honest: if this header stopped compiling, every
/// negative test would still fail to build and none of them would still match its
/// message.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pantr/grid/grid.hpp"

class GoodGrid;

template <>
struct pantr::grid::grid_traits<GoodGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::none;
};

/// A one-dimensional grid of `n` unit cells, declaring no specialisations.
class GoodGrid : public pantr::grid::GridBase<GoodGrid> {
  public:
    using Base = pantr::grid::GridBase<GoodGrid>;

    explicit GoodGrid(std::int64_t n) : Base(1, n) {}

    void cell_bounds(std::int64_t cid, std::span<double> lo, std::span<double> hi) const {
        this->check_cid(cid);
        lo[0] = static_cast<double>(cid);
        hi[0] = static_cast<double>(cid + 1);
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double> pt) const {
        const auto k = static_cast<std::int64_t>(pt[0]);
        if (k < 0 || k >= this->num_cells()) {
            return std::nullopt;
        }
        return k;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t cid,
                                                                    std::int64_t lfid) const {
        this->check_lfid(cid, lfid);
        const std::int64_t nbr = (lfid == 0) ? cid - 1 : cid + 1;
        if (nbr < 0 || nbr >= this->num_cells()) {
            return std::nullopt;
        }
        return nbr;
    }
};
