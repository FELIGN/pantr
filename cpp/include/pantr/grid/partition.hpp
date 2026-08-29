#pragma once

/// \file
/// Cell-ownership partition for distributing a structured grid.
///
/// Ports `src/pantr/grid/_partition.py`, whose class stays as the Python backend
/// under `PANTR_BACKEND=python`. Ownership moves under
/// `design/cross_backend_types.md`'s 2026-08-27 amendment: there is one partition,
/// it is this one, and `pantr.grid.Partition` wraps it.
///
/// A partition records, for every cell of a grid, which rank owns it -- or `-1`
/// for an inactive cell excluded from the partition (an exterior or trimmed cell).
/// It is deliberately space-agnostic: an integer owner per cell and nothing else,
/// which is what lets the same descriptor serve a grid and the knot-span grid of a
/// B-spline space.
///
/// ## Why `int32` owners
///
/// The oracle coerces to `int32` on construction and this reproduces that, cast
/// included: a rank index needs nothing wider, and the array is one entry per cell
/// on a mesh that may be large. The **narrowing** is part of the contract rather
/// than an implementation detail -- a caller handing in an owner above `2^31 - 1`
/// gets a wrapped value, which the range check below then rejects, in both
/// backends and in that order.
///
/// ## What is deliberately not here
///
/// The oracle's `active_mask` is not a method on this type. It is
/// `cell_owner[i] >= 0`, one line for a C++ caller and a whole `numpy` array for a
/// Python one, so the wrapper computes it from `cell_owner` where it is a single
/// vectorised comparison. That keeps this type to the two questions it is the only
/// one able to answer: what the owner of a cell is, and which cells a rank owns.
///
/// ## Immutability, and what the binding may hand out
///
/// A partition has no mutator, which is what makes it safe for the binding to
/// expose `cell_owner` as a zero-copy read-only view rather than a copy. The
/// oracle exposes its stored array the same way, with `writeable` cleared.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pantr::grid {

/// A per-cell owner assignment over a grid's cells.
///
/// Records, for every cell, the rank that owns it, or `-1` for an inactive cell.
/// Instances are immutable once constructed.
class Partition {
  public:
    /// Build a partition from a per-cell owner array.
    ///
    /// \param cell_owner Per-cell owner ranks, `-1` for an inactive cell. Copied.
    /// \param n_parts Number of parts (ranks), `>= 1`.
    /// \throws std::invalid_argument If `n_parts < 1`, or if some owner falls
    ///         outside `[-1, n_parts)`.
    Partition(std::span<const std::int32_t> cell_owner, std::int64_t n_parts)
        : cell_owner_(cell_owner.begin(), cell_owner.end()), n_parts_(n_parts) {
        if (n_parts < 1) {
            throw std::invalid_argument("n_parts must be >= 1; got " + std::to_string(n_parts)
                                        + ".");
        }
        if (cell_owner_.empty()) {
            return;
        }
        const auto [min_it, max_it] = std::minmax_element(cell_owner_.begin(), cell_owner_.end());
        if (*min_it < -1 || static_cast<std::int64_t>(*max_it) >= n_parts) {
            throw std::invalid_argument("cell_owner values must lie in [-1, "
                                        + std::to_string(n_parts) + "); got range ["
                                        + std::to_string(*min_it) + ", "
                                        + std::to_string(*max_it) + "].");
        }
    }

    /// The per-cell owner array.
    ///
    /// \return A view of the stored owners, valid while the partition lives.
    [[nodiscard]] std::span<const std::int32_t> cell_owner() const noexcept {
        return cell_owner_;
    }

    /// The number of parts (ranks).
    ///
    /// \return The part count, `>= 1`.
    [[nodiscard]] std::int64_t n_parts() const noexcept { return n_parts_; }

    /// The total number of cells, active and inactive.
    ///
    /// \return The length of `cell_owner`.
    [[nodiscard]] std::size_t n_cells() const noexcept { return cell_owner_.size(); }

    /// The flat ids of the cells owned by `rank`, ascending.
    ///
    /// \param rank Owner rank in `[0, n_parts)`.
    /// \return The ids, in increasing order.
    /// \throws std::invalid_argument If `rank` is outside `[0, n_parts)`.
    [[nodiscard]] std::vector<std::int64_t> owned_cells(std::int64_t rank) const {
        if (rank < 0 || rank >= n_parts_) {
            throw std::invalid_argument("rank must be in [0, " + std::to_string(n_parts_)
                                        + "); got " + std::to_string(rank) + ".");
        }
        std::vector<std::int64_t> owned;
        for (std::size_t cell = 0; cell < cell_owner_.size(); ++cell) {
            if (static_cast<std::int64_t>(cell_owner_[cell]) == rank) {
                owned.push_back(static_cast<std::int64_t>(cell));
            }
        }
        return owned;
    }

    /// A compact representation naming the two counts.
    ///
    /// The owners themselves are left out: there is one per cell, and a partition
    /// over a real mesh would print a page of them.
    ///
    /// \return `"Partition(n_cells=..., n_parts=...)"`.
    [[nodiscard]] std::string to_string() const {
        return "Partition(n_cells=" + std::to_string(cell_owner_.size())
               + ", n_parts=" + std::to_string(n_parts_) + ")";
    }

  private:
    std::vector<std::int32_t> cell_owner_;  ///< Owner rank per cell, `-1` if inactive.
    std::int64_t n_parts_;                  ///< Number of parts.
};

}  // namespace pantr::grid
