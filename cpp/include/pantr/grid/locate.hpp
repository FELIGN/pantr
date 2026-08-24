#pragma once

/// \file
/// Batch point location on an axis-aligned tensor-product grid.
///
/// Ports `src/pantr/grid/_locate_core.py`, which stays as the parity oracle.
///
/// ## Why this one's parity claim is an equality, structurally
///
/// This kernel performs **no floating-point arithmetic at all**. Enumerated by
/// walking the oracle's syntax tree rather than by reading it: every binary
/// operation in `_locate_core.py` is on an integer index, and every use of a
/// coordinate is a *comparison*. So there is no rounding to bound, no
/// accumulation order to preserve and no fused-multiply-add site, and the two
/// backends agree bit for bit for reasons that do not depend on the compiler,
/// the vectorization width or the libm. That is a stronger claim than the
/// contraction bounds `design/backend_parity.md` Rules 9 and 10 state for the
/// bezier kernels, and it is stronger for a simpler reason: there is nothing
/// there to differ.
///
/// The oracle is `float64`-only -- `_locate_core.py` mentions `float32` nowhere --
/// so `T` is instantiated at `double` and nothing else. It is templated anyway,
/// which is the core's convention (`pantr/core/scalar.hpp`) and costs nothing:
/// the discipline that admits a `Dual` is the discipline that admits `float`, and
/// a kernel whose only use of the scalar is a comparison is the easiest place to
/// keep it.
///
/// ## The tie contract, and why it survives the port for free
///
/// A point exactly on an interior breakpoint belongs to the **lower**-indexed
/// cell sharing that face; a point on the outer boundary belongs to the adjacent
/// boundary cell; a point outside the domain maps to `-1`. That is a discrete
/// verdict, which `design/backend_parity.md` Rule 11 warns no tolerance can
/// bound. Here it needs none: the verdict is decided by `<` and `>` on
/// coordinates that were never arithmetic operands, so both backends read the
/// same bits and take the same branch.
///
/// ## One difference that is not parity
///
/// The oracle's outer loop is `prange`, so it runs one point per thread; this is
/// serial. Each point writes only its own entry of `out` and there is no
/// reduction, so the answer cannot move with the thread count either way. The
/// same argument and the same precedent as the bezier batch kernels
/// (`pantr/bezier/root_finding.hpp`); only the speed differs, and nothing here
/// has been profiled.

#include <cstddef>
#include <cstdint>
#include <span>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::grid {

/// Locate a batch of points on an axis-aligned tensor-product grid.
///
/// Mirrors `_locate_core._locate_points_core`. No validation is performed.
///
/// \tparam T Coordinate type.
/// \param points Query points, shape `(npts, ndim)`.
/// \param knots_flat All per-axis knot vectors concatenated end to end; axis `d`
///        occupies `[knot_starts[d], knot_starts[d] + cells_per_axis[d] + 1)` and
///        must be strictly increasing.
/// \param knot_starts Per-axis start offset into `knots_flat`, shape `(ndim,)`.
/// \param cells_per_axis Per-axis cell counts, shape `(ndim,)`.
/// \param strides Per-axis C-order flat strides, shape `(ndim,)`.
/// \param out Output flat cell ids, shape `(npts,)`; `-1` for a point outside the
///        grid domain.
template <Real T>
void locate_points(span2d<const T> points, std::span<const T> knots_flat,
                   std::span<const std::int64_t> knot_starts,
                   std::span<const std::int64_t> cells_per_axis,
                   std::span<const std::int64_t> strides, std::span<std::int64_t> out) {
    using pantr::value_of;

    const std::size_t npts = points.extent(0);
    const std::size_t ndim = points.extent(1);

    for (std::size_t p = 0; p < npts; ++p) {
        std::int64_t cid = 0;
        bool inside = true;

        for (std::size_t d = 0; d < ndim; ++d) {
            const std::int64_t start = knot_starts[d];
            const std::int64_t ncells = cells_per_axis[d];
            const auto x = value_of(points(p, d));

            const auto first = value_of(knots_flat[static_cast<std::size_t>(start)]);
            const auto last = value_of(knots_flat[static_cast<std::size_t>(start + ncells)]);
            if (x < first || x > last) {
                inside = false;
                break;
            }

            // lower_bound: the first index idx with knots[idx] >= x. Written out
            // rather than delegated to std::lower_bound so that the comparison is
            // the oracle's own and passes through value_of, which the standard
            // algorithm's default comparator would not.
            std::int64_t lo_i = 0;
            std::int64_t hi_i = ncells + 1;
            while (lo_i < hi_i) {
                const std::int64_t mid = (lo_i + hi_i) / 2;
                if (value_of(knots_flat[static_cast<std::size_t>(start + mid)]) < x) {
                    lo_i = mid + 1;
                } else {
                    hi_i = mid;
                }
            }

            std::int64_t cell_i = lo_i - 1;
            if (cell_i < 0) {
                cell_i = 0;
            } else if (cell_i > ncells - 1) {
                cell_i = ncells - 1;
            }
            cid += cell_i * strides[d];
        }

        out[p] = inside ? cid : -1;
    }
}

}  // namespace pantr::grid
