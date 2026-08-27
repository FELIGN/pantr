/// \file
/// nanobind bindings for the `pantr.grid` kernels.
///
/// Eight entry points over nine oracle kernels. The missing one is
/// `_hier_core._block_of_midx`, and it is missing structurally rather than by
/// oversight: on the Numba side it is called from inside `_hier_locate_points_core`
/// at a `nopython` call site, and no dispatch can be inserted between two Numba
/// kernels, so the boundary is forced up to the eight places Layer 2 reaches for a
/// kernel. Its C++ counterpart is bound anyway, under the name of the oracle's
/// one-line wrapper `_encode_midx_core`, because that wrapper *is* a Layer 2 seam
/// and the two functions are one here.
///
/// ## `double` only, and that is not laziness
///
/// Every sibling binding registers `double` and `float` overloads. These register
/// `double` alone, because the oracle is `float64`-only -- `_locate_core.py`,
/// `_bvh_core.py` and `_hier_core.py` mention `float32` nowhere. Registering
/// `float` would create a surface with no oracle behind it, and
/// `design/backend_parity.md` Rule 8 is explicit that a parity claim is only
/// defined where the comparison can say something. The headers are templated on
/// `Real` regardless, which is the core's convention and what keeps the Tier B
/// discipline live; only the instantiation is narrow.
///
/// ## What the checks are for, and the one the oracle does not make
///
/// nanobind settles dtype, rank and contiguity. What it cannot express is that a
/// pile of flat descriptor arrays agree with each other, and these kernels take
/// several: an inconsistent `ndim` between `points` and `knot_starts`, or a
/// `knot_starts[d] + cells_per_axis[d]` past the end of `knots_flat`, is an
/// out-of-bounds read rather than a wrong answer. Those are checked.
///
/// **The stack-depth contract is checked where the oracle checks it and nowhere
/// else.** `bvh_build` validates it in closed form, because median-of-longest-axis
/// splits are balanced by construction and the height follows from the cell count
/// alone. The two query bindings do **not** validate it: `_bvh.py` establishes it
/// once, at construction, with an O(n) walk it then does not repeat per query, and
/// a binding that walked the tree on every query would make a box query linear in
/// the tree it exists to avoid traversing. So a caller assembling node arrays by
/// hand owns that contract, exactly as it does on the Numba side.
///
/// **Nearly every argument is keyword-only, which is deliberate and is the one
/// place this file is stricter than the oracle.** These signatures are full of
/// adjacent, same-typed, mutually unordered `int64` descriptor arrays --
/// `block_lo`/`block_hi`, `node_left`/`node_right`,
/// `knot_starts`/`cells_per_axis`/`strides`. Transposed, every one of them
/// type-checks, runs, and returns a plausible wrong answer; `bezier_root_finding.cpp`
/// records two such traps found by an audit rather than by a test, which is the
/// evidence that a comment would not have been enough. Where an ordering constraint
/// exists it is checked instead of merely named: `cell_hi >= cell_lo` and
/// `block_hi > block_lo` are real invariants and are enforced.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "pantr/core/mdspan.hpp"
#include "pantr/grid/bvh.hpp"
#include "pantr/grid/hierarchical.hpp"
#include "pantr/grid/locate.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

template <class T>
using const_vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using const_mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_vec = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_mat = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

using i64_vec = const_vec<std::int64_t>;
using i64_mat = const_mat<std::int64_t>;

/// Raise unless two extents agree.
void require_same(std::size_t got, std::size_t want, const char* what, const char* against) {
    if (got != want) {
        throw nb::value_error((std::string(what) + " has " + std::to_string(got) +
                               " entries but " + against + " implies " + std::to_string(want))
                                  .c_str());
    }
}

/// Raise unless `out` can hold `needed` entries along `axis`.
///
/// A lower bound, not an equality: refusing a buffer that is merely generous would
/// be wrong. The axis is explicit because getting it wrong is silent -- the trap
/// `bezier_root_finding.cpp` records, where a 2-D output's row width went unchecked
/// because the first axis was checked twice.
template <class Arr>
void require_capacity(const Arr& out, std::size_t axis, std::size_t needed, const char* what) {
    if (out.shape(axis) < needed) {
        throw nb::value_error((std::string(what) + " holds " +
                               std::to_string(out.shape(axis)) + " entries along axis " +
                               std::to_string(axis) + ", but this call needs " +
                               std::to_string(needed))
                                  .c_str());
    }
}

/// Raise unless every axis declares at least one cell and stays inside `knots_flat`.
///
/// Both failures are out-of-bounds reads rather than wrong answers: the kernel
/// indexes `knots_flat[knot_starts[d] + cells_per_axis[d]]` before any loop bound
/// could stop it.
void require_knot_ranges(std::size_t n_knots, i64_vec knot_starts, i64_vec cells_per_axis) {
    for (std::size_t d = 0; d < knot_starts.shape(0); ++d) {
        const std::int64_t start = knot_starts(d);
        const std::int64_t ncells = cells_per_axis(d);
        if (ncells < 1) {
            throw nb::value_error((std::string("axis ") + std::to_string(d) + " declares " +
                                   std::to_string(ncells) + " cells; at least one is needed")
                                      .c_str());
        }
        if (start < 0 || static_cast<std::size_t>(start + ncells) >= n_knots) {
            throw nb::value_error(
                (std::string("axis ") + std::to_string(d) + " spans knots [" +
                 std::to_string(start) + ", " + std::to_string(start + ncells) +
                 "] but knots_flat holds " + std::to_string(n_knots))
                    .c_str());
        }
    }
}

/// Raise unless the packed block descriptor is self-consistent.
///
/// `level_block_start` must be non-decreasing and end at the block count, since the
/// kernels use consecutive entries as a half-open range, and every block must have
/// a positive extent, since the collect kernel divides by it.
void require_block_descriptor(i64_mat block_lo, i64_mat block_hi, i64_vec block_base,
                             i64_vec level_block_start) {
    const std::size_t n_blocks = block_base.shape(0);
    require_same(block_lo.shape(0), n_blocks, "block_lo rows", "block_base");
    require_same(block_hi.shape(0), n_blocks, "block_hi rows", "block_base");
    require_same(block_hi.shape(1), block_lo.shape(1), "block_hi columns", "block_lo");

    if (level_block_start.shape(0) < 2) {
        throw nb::value_error("level_block_start needs at least two entries, for one level");
    }
    if (level_block_start(0) != 0) {
        throw nb::value_error("level_block_start must begin at 0");
    }
    for (std::size_t i = 1; i < level_block_start.shape(0); ++i) {
        if (level_block_start(i) < level_block_start(i - 1)) {
            throw nb::value_error("level_block_start must be non-decreasing");
        }
    }
    if (static_cast<std::size_t>(level_block_start(level_block_start.shape(0) - 1)) != n_blocks) {
        throw nb::value_error("level_block_start must end at the block count");
    }
    for (std::size_t b = 0; b < n_blocks; ++b) {
        for (std::size_t k = 0; k < block_lo.shape(1); ++k) {
            if (block_hi(b, k) <= block_lo(b, k)) {
                throw nb::value_error(
                    (std::string("block ") + std::to_string(b) + " axis " + std::to_string(k) +
                     " has extent " + std::to_string(block_hi(b, k) - block_lo(b, k)) +
                     "; every block extent must be positive")
                        .c_str());
            }
        }
    }
}

void bind_locate_points(const_mat<double> points, const_vec<double> knots_flat,
                        i64_vec knot_starts, i64_vec cells_per_axis, i64_vec strides,
                        out_vec<std::int64_t> out) {
    const std::size_t ndim = points.shape(1);
    require_same(knot_starts.shape(0), ndim, "knot_starts", "the point array's second axis");
    require_same(cells_per_axis.shape(0), ndim, "cells_per_axis", "the point array's second axis");
    require_same(strides.shape(0), ndim, "strides", "the point array's second axis");
    require_knot_ranges(knots_flat.shape(0), knot_starts, cells_per_axis);
    require_capacity(out, 0, points.shape(0), "out");

    pantr::grid::locate_points<double>(
        pantr::span2d<const double>(points.data(), points.shape(0), ndim),
        std::span<const double>(knots_flat.data(), knots_flat.shape(0)),
        std::span<const std::int64_t>(knot_starts.data(), ndim),
        std::span<const std::int64_t>(cells_per_axis.data(), ndim),
        std::span<const std::int64_t>(strides.data(), ndim),
        std::span<std::int64_t>(out.data(), out.shape(0)));
}

void bind_bvh_build(const_mat<double> cell_lo, const_mat<double> cell_hi,
                    out_mat<double> node_lo, out_mat<double> node_hi,
                    out_vec<std::int64_t> node_left, out_vec<std::int64_t> node_right,
                    out_vec<std::int64_t> node_cell) {
    const std::size_t n_cells = cell_lo.shape(0);
    const std::size_t ndim = cell_lo.shape(1);
    if (n_cells == 0) {
        throw nb::value_error("cell_lo is empty; the build needs at least one cell");
    }
    require_same(cell_hi.shape(0), n_cells, "cell_hi rows", "cell_lo");
    require_same(cell_hi.shape(1), ndim, "cell_hi columns", "cell_lo");

    // hi >= lo is an ordering constraint, so it is checked rather than left to the
    // caller getting the two keywords the right way round.
    // Finiteness first, and `hi >= lo` is NOT enough to give it. That test rejects
    // a NaN corner, since every NaN comparison is false, but it ADMITS infinities:
    // `+inf >= -inf` holds. The build then computes the centroid as
    // `0.5 * (lo + hi)`, so a cell spanning `[-inf, +inf]` yields NaN from two
    // corners that passed -- and a NaN key makes the stable sort's comparator an
    // invalid strict weak ordering, which is undefined behaviour under
    // [alg.sorting]/4 rather than merely a divergence from the oracle. The guard
    // the sort needs is on the centroid, and no `hi >= lo` check can supply it.
    // `pantr.grid.BVH.from_cell_bounds` rejects non-finite corners for the same
    // reason; this is the same contract on the path that bypasses it.
    for (std::size_t i = 0; i < n_cells; ++i) {
        for (std::size_t k = 0; k < ndim; ++k) {
            if (!std::isfinite(cell_lo(i, k)) || !std::isfinite(cell_hi(i, k))) {
                throw nb::value_error((std::string("cell ") + std::to_string(i) + " axis " +
                                       std::to_string(k) +
                                       " has a non-finite corner; the centroid a "
                                       "non-finite corner produces is not orderable")
                                          .c_str());
            }
            if (!(cell_hi(i, k) >= cell_lo(i, k))) {
                throw nb::value_error((std::string("cell ") + std::to_string(i) + " axis " +
                                       std::to_string(k) + " has hi < lo")
                                          .c_str());
            }
        }
    }

    // The build's own splits are balanced, so the height follows from the cell count
    // and needs no tree walk. Mirrors BVH.from_cell_bounds, including the `+ 1` for
    // the root push.
    //
    // `bit_width(n - 1)` is `ceil(log2(n))` for n >= 2, in exact integers. The
    // previous form went through a `double`, which is the wrong arithmetic for a
    // question about an integer and not merely inelegant: past a threshold a
    // `double` stops separating n from the power of two below it, and the height
    // comes back one too SMALL -- the unsafe direction for a bound.
    //
    // Where the two disagree is unreachable, and by a wide margin: the smallest
    // such cell count needs petabytes for a single coordinate axis. So this changes
    // no attainable result. scripts/measure_bvh_depth_arithmetic.py enumerates the
    // disagreements, reports the threshold and asserts the direction, rather than
    // this comment carrying figures nothing re-derives.
    //
    // It is also what removes the last qualified `std::` math call from the
    // bindings, which cpp/include/pantr/core/scalar.hpp asks for.
    const std::int64_t max_depth =
        n_cells > 1
            ? static_cast<std::int64_t>(std::bit_width(static_cast<std::uint64_t>(n_cells - 1))) + 1
            : 1;
    if (max_depth > pantr::grid::kBvhStackDepth) {
        throw nb::value_error((std::to_string(n_cells) + " cells would produce a tree of depth >= " +
                               std::to_string(max_depth) + ", exceeding the traversal stack depth " +
                               std::to_string(pantr::grid::kBvhStackDepth))
                                  .c_str());
    }

    const std::size_t n_nodes = 2 * n_cells - 1;
    require_capacity(node_lo, 0, n_nodes, "node_lo");
    require_capacity(node_hi, 0, n_nodes, "node_hi");
    require_same(node_lo.shape(1), ndim, "node_lo columns", "cell_lo");
    require_same(node_hi.shape(1), ndim, "node_hi columns", "cell_lo");
    require_capacity(node_left, 0, n_nodes, "node_left");
    require_capacity(node_right, 0, n_nodes, "node_right");
    require_capacity(node_cell, 0, n_nodes, "node_cell");

    pantr::grid::bvh_build<double>(
        pantr::span2d<const double>(cell_lo.data(), n_cells, ndim),
        pantr::span2d<const double>(cell_hi.data(), n_cells, ndim),
        pantr::span2d<double>(node_lo.data(), node_lo.shape(0), ndim),
        pantr::span2d<double>(node_hi.data(), node_hi.shape(0), ndim),
        std::span<std::int64_t>(node_left.data(), node_left.shape(0)),
        std::span<std::int64_t>(node_right.data(), node_right.shape(0)),
        std::span<std::int64_t>(node_cell.data(), node_cell.shape(0)));
}

/// Shared shape checks for the two query kernels, so they cannot drift apart.
std::size_t check_query(const_vec<double> qlo, const_vec<double> qhi, const_mat<double> node_lo,
                        const_mat<double> node_hi, i64_vec node_left, i64_vec node_right,
                        i64_vec node_cell) {
    const std::size_t ndim = qlo.shape(0);
    const std::size_t n_nodes = node_cell.shape(0);
    require_same(qhi.shape(0), ndim, "qhi", "qlo");
    require_same(node_lo.shape(1), ndim, "node_lo columns", "qlo");
    require_same(node_hi.shape(1), ndim, "node_hi columns", "qlo");
    require_same(node_lo.shape(0), n_nodes, "node_lo rows", "node_cell");
    require_same(node_hi.shape(0), n_nodes, "node_hi rows", "node_cell");
    require_same(node_left.shape(0), n_nodes, "node_left", "node_cell");
    require_same(node_right.shape(0), n_nodes, "node_right", "node_cell");
    if (n_nodes == 0) {
        throw nb::value_error("the node arrays are empty; a query needs at least a root");
    }
    return n_nodes;
}

std::int64_t bind_bvh_query_count(const_vec<double> qlo, const_vec<double> qhi,
                                  const_mat<double> node_lo, const_mat<double> node_hi,
                                  i64_vec node_left, i64_vec node_right, i64_vec node_cell) {
    const std::size_t n_nodes =
        check_query(qlo, qhi, node_lo, node_hi, node_left, node_right, node_cell);
    const std::size_t ndim = qlo.shape(0);
    return pantr::grid::bvh_query_count<double>(
        std::span<const double>(qlo.data(), ndim), std::span<const double>(qhi.data(), ndim),
        pantr::span2d<const double>(node_lo.data(), n_nodes, ndim),
        pantr::span2d<const double>(node_hi.data(), n_nodes, ndim),
        std::span<const std::int64_t>(node_left.data(), n_nodes),
        std::span<const std::int64_t>(node_right.data(), n_nodes),
        std::span<const std::int64_t>(node_cell.data(), n_nodes));
}

std::int64_t bind_bvh_query_emit(const_vec<double> qlo, const_vec<double> qhi,
                                 const_mat<double> node_lo, const_mat<double> node_hi,
                                 i64_vec node_left, i64_vec node_right, i64_vec node_cell,
                                 out_vec<std::int64_t> out) {
    const std::size_t n_nodes =
        check_query(qlo, qhi, node_lo, node_hi, node_left, node_right, node_cell);
    const std::size_t ndim = qlo.shape(0);
    const std::int64_t count = pantr::grid::bvh_query_emit<double>(
        std::span<const double>(qlo.data(), ndim), std::span<const double>(qhi.data(), ndim),
        pantr::span2d<const double>(node_lo.data(), n_nodes, ndim),
        pantr::span2d<const double>(node_hi.data(), n_nodes, ndim),
        std::span<const std::int64_t>(node_left.data(), n_nodes),
        std::span<const std::int64_t>(node_right.data(), n_nodes),
        std::span<const std::int64_t>(node_cell.data(), n_nodes),
        std::span<std::int64_t>(out.data(), out.shape(0)));

    // The kernel counts unconditionally and writes only within capacity, so this
    // catches an undersized `out` after the fact instead of corrupting it. It is
    // also the count/emit agreement the oracle's module docstring names as the one
    // invariant the two opposite traversal orders must preserve.
    if (static_cast<std::size_t>(count) > out.shape(0)) {
        throw nb::value_error((std::string("out holds ") + std::to_string(out.shape(0)) +
                               " entries but the query matched " + std::to_string(count) +
                               " cells; size it from bvh_query_count")
                                  .c_str());
    }
    return count;
}

std::int64_t bind_encode_midx(std::int64_t level, i64_vec midx, i64_mat block_lo,
                              i64_mat block_hi, i64_vec block_base, i64_vec level_block_start) {
    require_block_descriptor(block_lo, block_hi, block_base, level_block_start);
    require_same(midx.shape(0), block_lo.shape(1), "midx", "block_lo columns");
    const std::int64_t n_levels = static_cast<std::int64_t>(level_block_start.shape(0)) - 1;
    if (level < 0 || level >= n_levels) {
        throw nb::value_error((std::string("level ") + std::to_string(level) +
                               " is outside [0, " + std::to_string(n_levels) + ")")
                                  .c_str());
    }
    return pantr::grid::block_of_midx(
        level, std::span<const std::int64_t>(midx.data(), midx.shape(0)),
        pantr::span2d<const std::int64_t>(block_lo.data(), block_lo.shape(0), block_lo.shape(1)),
        pantr::span2d<const std::int64_t>(block_hi.data(), block_hi.shape(0), block_hi.shape(1)),
        std::span<const std::int64_t>(block_base.data(), block_base.shape(0)),
        std::span<const std::int64_t>(level_block_start.data(), level_block_start.shape(0)));
}

std::int64_t bind_decode_flat_id(std::int64_t cid, i64_mat block_lo, i64_mat block_hi,
                                 i64_vec block_base, i64_vec level_block_start,
                                 out_vec<std::int64_t> out_midx) {
    require_block_descriptor(block_lo, block_hi, block_base, level_block_start);
    require_capacity(out_midx, 0, block_lo.shape(1), "out_midx");

    // The upper-bound search takes `lo_b - 1`, so a cid below the first block's base
    // would index -1. Layer 2 never produces one; a direct call could.
    if (cid < 0 || cid < block_base(0)) {
        throw nb::value_error((std::string("cid ") + std::to_string(cid) +
                               " is below the first block's base " +
                               std::to_string(block_base(0)))
                                  .c_str());
    }
    return pantr::grid::decode_flat_id(
        cid,
        pantr::span2d<const std::int64_t>(block_lo.data(), block_lo.shape(0), block_lo.shape(1)),
        pantr::span2d<const std::int64_t>(block_hi.data(), block_hi.shape(0), block_hi.shape(1)),
        std::span<const std::int64_t>(block_base.data(), block_base.shape(0)),
        std::span<const std::int64_t>(level_block_start.data(), level_block_start.shape(0)),
        std::span<std::int64_t>(out_midx.data(), out_midx.shape(0)));
}

void bind_hier_locate_points(const_mat<double> points, const_vec<double> knots_flat,
                             i64_vec knot_starts, i64_vec root_cells_per_axis, i64_vec factor,
                             i64_mat block_lo, i64_mat block_hi, i64_vec block_base,
                             i64_vec level_block_start, out_vec<std::int64_t> out) {
    const std::size_t ndim = points.shape(1);
    require_block_descriptor(block_lo, block_hi, block_base, level_block_start);
    require_same(block_lo.shape(1), ndim, "block_lo columns", "the point array's second axis");
    require_same(knot_starts.shape(0), ndim, "knot_starts", "the point array's second axis");
    require_same(root_cells_per_axis.shape(0), ndim, "root_cells_per_axis",
                 "the point array's second axis");
    require_same(factor.shape(0), ndim, "factor", "the point array's second axis");
    require_knot_ranges(knots_flat.shape(0), knot_starts, root_cells_per_axis);
    require_capacity(out, 0, points.shape(0), "out");
    for (std::size_t k = 0; k < ndim; ++k) {
        // `< 1`, not `< 2`. The oracle documents a factor of 1 as legal -- it
        // "prevents subdivision in that direction", which is what an anisotropic
        // grid needs -- validates `f < 1`, and has a committed test asserting it.
        // Rejecting 1 here made the two backends disagree on the DOMAIN rather
        // than on a value, and it rejected the one factor at which the descent's
        // contraction site provably cannot diverge: the clamp forces `j = 0`, so
        // the product is exactly zero.
        if (factor(k) < 1) {
            throw nb::value_error((std::string("factor[") + std::to_string(k) + "] is " +
                                   std::to_string(factor(k)) +
                                   "; a subdivision factor must be at least one")
                                      .c_str());
        }
    }

    pantr::grid::hier_locate_points<double>(
        pantr::span2d<const double>(points.data(), points.shape(0), ndim),
        std::span<const double>(knots_flat.data(), knots_flat.shape(0)),
        std::span<const std::int64_t>(knot_starts.data(), ndim),
        std::span<const std::int64_t>(root_cells_per_axis.data(), ndim),
        std::span<const std::int64_t>(factor.data(), ndim),
        pantr::span2d<const std::int64_t>(block_lo.data(), block_lo.shape(0), ndim),
        pantr::span2d<const std::int64_t>(block_hi.data(), block_hi.shape(0), ndim),
        std::span<const std::int64_t>(block_base.data(), block_base.shape(0)),
        std::span<const std::int64_t>(level_block_start.data(), level_block_start.shape(0)),
        std::span<std::int64_t>(out.data(), out.shape(0)));
}

void bind_hier_collect_cell_bounds(const_vec<double> knots_flat, i64_vec knot_starts,
                                   i64_vec factor, i64_mat block_lo, i64_mat block_hi,
                                   i64_vec block_base, i64_vec level_block_start,
                                   out_mat<double> out_lo, out_mat<double> out_hi) {
    require_block_descriptor(block_lo, block_hi, block_base, level_block_start);
    const std::size_t ndim = block_lo.shape(1);
    require_same(knot_starts.shape(0), ndim, "knot_starts", "block_lo columns");
    require_same(factor.shape(0), ndim, "factor", "block_lo columns");
    require_same(out_hi.shape(0), out_lo.shape(0), "out_hi rows", "out_lo");
    require_same(out_lo.shape(1), ndim, "out_lo columns", "block_lo");
    require_same(out_hi.shape(1), ndim, "out_hi columns", "block_lo");

    for (std::size_t k = 0; k < ndim; ++k) {
        // `< 1`, not `< 2`. The oracle documents a factor of 1 as legal -- it
        // "prevents subdivision in that direction", which is what an anisotropic
        // grid needs -- validates `f < 1`, and has a committed test asserting it.
        // Rejecting 1 here made the two backends disagree on the DOMAIN rather
        // than on a value, and it rejected the one factor at which the descent's
        // contraction site provably cannot diverge: the clamp forces `j = 0`, so
        // the product is exactly zero.
        if (factor(k) < 1) {
            throw nb::value_error((std::string("factor[") + std::to_string(k) + "] is " +
                                   std::to_string(factor(k)) +
                                   "; a subdivision factor must be at least one")
                                      .c_str());
        }
    }

    // Two bounds, both computed per block rather than taken on trust, because both
    // failures are out-of-bounds accesses rather than wrong answers.
    //
    // The rows: the odometer writes `block_base[b] + n_block` of them for each
    // block, so the output must cover the largest flat id any block reaches.
    //
    // The knots: the kernel reads `knots_flat[knot_starts[k] + root_ik + 1]` where
    // `root_ik = ik / factor[k]**level` over `ik < block_hi(b, k)`. So the bound is
    // the largest root index any block reaches at its own level, which is NOT
    // `factor[k]` and is not `block_hi` either -- an earlier version of this check
    // passed `factor` where the helper wanted a root cell count, which both
    // rejected valid grids and admitted out-of-range ones.
    std::size_t needed = 0;
    const auto n_levels = static_cast<std::int64_t>(level_block_start.shape(0)) - 1;
    for (std::size_t b = 0; b < block_base.shape(0); ++b) {
        std::int64_t level = 0;
        for (std::int64_t lev = 0; lev < n_levels; ++lev) {
            const auto l = static_cast<std::size_t>(lev);
            if (level_block_start(l) <= static_cast<std::int64_t>(b)
                && static_cast<std::int64_t>(b) < level_block_start(l + 1)) {
                level = lev;
                break;
            }
        }
        std::int64_t n_block = 1;
        for (std::size_t k = 0; k < ndim; ++k) {
            n_block *= block_hi(b, k) - block_lo(b, k);

            std::int64_t m_pow = 1;
            for (std::int64_t i = 0; i < level; ++i) {
                m_pow *= factor(k);
            }
            const std::int64_t max_root = (block_hi(b, k) - 1) / m_pow;
            const std::int64_t start = knot_starts(k);
            if (start < 0 || static_cast<std::size_t>(start + max_root + 1) >= knots_flat.shape(0)) {
                throw nb::value_error(
                    (std::string("block ") + std::to_string(b) + " axis " + std::to_string(k) +
                     " reaches root cell " + std::to_string(max_root) + " at knot index " +
                     std::to_string(start + max_root + 1) + ", but knots_flat holds " +
                     std::to_string(knots_flat.shape(0)))
                        .c_str());
            }
        }
        needed = std::max(needed, static_cast<std::size_t>(block_base(b) + n_block));
    }
    require_capacity(out_lo, 0, needed, "out_lo");
    require_capacity(out_hi, 0, needed, "out_hi");

    pantr::grid::hier_collect_cell_bounds<double>(
        std::span<const double>(knots_flat.data(), knots_flat.shape(0)),
        std::span<const std::int64_t>(knot_starts.data(), ndim),
        std::span<const std::int64_t>(factor.data(), ndim),
        pantr::span2d<const std::int64_t>(block_lo.data(), block_lo.shape(0), ndim),
        pantr::span2d<const std::int64_t>(block_hi.data(), block_hi.shape(0), ndim),
        std::span<const std::int64_t>(block_base.data(), block_base.shape(0)),
        std::span<const std::int64_t>(level_block_start.data(), level_block_start.shape(0)),
        pantr::span2d<double>(out_lo.data(), out_lo.shape(0), ndim),
        pantr::span2d<double>(out_hi.data(), out_hi.shape(0), ndim));
}

}  // namespace

void register_grid(nb::module_& m) {
    m.def("locate_points", &bind_locate_points, nb::arg("points").noconvert(), nb::kw_only(),
          nb::arg("knots_flat").noconvert(), nb::arg("knot_starts").noconvert(),
          nb::arg("cells_per_axis").noconvert(), nb::arg("strides").noconvert(),
          nb::arg("out").noconvert());

    m.def("bvh_build", &bind_bvh_build, nb::kw_only(), nb::arg("cell_lo").noconvert(),
          nb::arg("cell_hi").noconvert(), nb::arg("node_lo").noconvert(),
          nb::arg("node_hi").noconvert(), nb::arg("node_left").noconvert(),
          nb::arg("node_right").noconvert(), nb::arg("node_cell").noconvert());

    m.def("bvh_query_count", &bind_bvh_query_count, nb::kw_only(), nb::arg("qlo").noconvert(),
          nb::arg("qhi").noconvert(), nb::arg("node_lo").noconvert(),
          nb::arg("node_hi").noconvert(), nb::arg("node_left").noconvert(),
          nb::arg("node_right").noconvert(), nb::arg("node_cell").noconvert());

    m.def("bvh_query_emit", &bind_bvh_query_emit, nb::kw_only(), nb::arg("qlo").noconvert(),
          nb::arg("qhi").noconvert(), nb::arg("node_lo").noconvert(),
          nb::arg("node_hi").noconvert(), nb::arg("node_left").noconvert(),
          nb::arg("node_right").noconvert(), nb::arg("node_cell").noconvert(),
          nb::arg("out").noconvert());

    m.def("encode_midx", &bind_encode_midx, nb::arg("level"), nb::kw_only(),
          nb::arg("midx").noconvert(), nb::arg("block_lo").noconvert(),
          nb::arg("block_hi").noconvert(), nb::arg("block_base").noconvert(),
          nb::arg("level_block_start").noconvert());

    m.def("decode_flat_id", &bind_decode_flat_id, nb::arg("cid"), nb::kw_only(),
          nb::arg("block_lo").noconvert(), nb::arg("block_hi").noconvert(),
          nb::arg("block_base").noconvert(), nb::arg("level_block_start").noconvert(),
          nb::arg("out_midx").noconvert());

    m.def("hier_locate_points", &bind_hier_locate_points, nb::arg("points").noconvert(),
          nb::kw_only(), nb::arg("knots_flat").noconvert(), nb::arg("knot_starts").noconvert(),
          nb::arg("root_cells_per_axis").noconvert(), nb::arg("factor").noconvert(),
          nb::arg("block_lo").noconvert(), nb::arg("block_hi").noconvert(),
          nb::arg("block_base").noconvert(), nb::arg("level_block_start").noconvert(),
          nb::arg("out").noconvert());

    m.def("hier_collect_cell_bounds", &bind_hier_collect_cell_bounds, nb::kw_only(),
          nb::arg("knots_flat").noconvert(), nb::arg("knot_starts").noconvert(),
          nb::arg("factor").noconvert(), nb::arg("block_lo").noconvert(),
          nb::arg("block_hi").noconvert(), nb::arg("block_base").noconvert(),
          nb::arg("level_block_start").noconvert(), nb::arg("out_lo").noconvert(),
          nb::arg("out_hi").noconvert());
}
