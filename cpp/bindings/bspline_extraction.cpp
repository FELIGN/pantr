/// \file
/// Bindings for the tensor-product extraction kernels of
/// `pantr/bspline/extraction_kernels.hpp`.
///
/// ## Twenty-four entry points over four functions
///
/// The header is generic in the number of directions; these are not. The names,
/// the argument order and the arities mirror `pantr.bspline._extraction_kernels`
/// exactly, so the catalogue in `pantr.bspline._extraction_backend` selects
/// between two functions with one signature rather than between two conventions
/// -- the same reason `change_basis.cpp`'s builders take their nodes as an
/// argument. A single dimension-generic entry point taking a list of arrays would
/// also not be on `design/cross_backend_types.md`'s table of what may cross.
///
/// ## What nanobind checks, and what is checked here
///
/// The aliases below pin dtype, rank, C-contiguity and device, so those are
/// validated before any body runs. What a type cannot express is the relation
/// between arguments, and for these kernels that is nearly everything: whether
/// `v` is as long as the operators' input extents multiply out to, whether
/// `scratch` is big enough for the stage sequence, whether `out` overlaps `v`.
///
/// Those are checked here rather than in the kernel, because
/// `CLAUDE.md` puts validation in Layer 2 and this file is the C++ half of Layer 2.
/// The kernel's own `PANTR_PRECONDITION`s are a debug-build backstop for the
/// memory-safety subset and are compiled out in a release build, so they are not
/// the check.
///
/// ## `.noconvert()` everywhere, and here it decides the arithmetic
///
/// `basis.cpp` argues the general case. This file has a sharper one: the kernels
/// accumulate in the **operator's** scalar type, so a silent `float32` to
/// `float64` cast on `M_0` would not merely copy, it would change the accumulation
/// width and make the result disagree with the oracle for a reason no caller could
/// see. The overload set is resolved on the arrays' own dtypes or not at all.
///
/// ## The batch kernels are serial
///
/// The oracle's eight `*_many_*` kernels are `parallel=True` with `prange` over
/// cells; these loop. Each cell writes only its own rows of `out` and there is no
/// reduction, so the answer cannot move with the thread count either way -- the
/// same argument, and the same precedent, as `pantr/grid/locate.hpp:40-45` and
/// `pantr/bezier/root_finding.hpp:1308-1311`. Only the speed differs, and nothing
/// here has been profiled.
///
/// One detail of the oracle's batch loop that is reproduced deliberately: it reads
/// `ops_k[idx_map_k[i_k]]` for **every** direction, including one flagged identity,
/// whose `idx_map` entry is then 0 and whose row is the sentinel. Only the pointer
/// is taken; the values are still unread. Compact storage always has at least one
/// row, which is what makes that indexing safe.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "pantr/bspline/extraction_kernels.hpp"
#include "pantr/core/mdspan.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

using pantr::span2d;
using pantr::bspline::ModeOperator;

template <class T>
using const_matrix = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using const_vector = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using const_stack = nb::ndarray<const T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_vector = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_matrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A per-direction compact index map, `np.intp` on the Python side.
///
/// `std::int64_t` rather than a platform-width type: `CLAUDE.md`'s rule that
/// `size_t` must not cross the seam is about a JIT being unable to infer one, and
/// the same argument applies to a binding whose stub has to name a fixed type.
/// `np.intp` is `int64` on every platform this is built for, so `.noconvert()`
/// accepts the oracle's own arrays unchanged.
using index_map = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A per-direction identity mask.
using flag_mask = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A `(n_cells, d)` block of per-direction cell indices.
using cell_block = nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// Batch operands and outputs are one row per cell.
template <class T>
using batch_2d = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using const_batch_2d = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using batch_3d = nb::ndarray<T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

template <class T>
using const_batch_3d = nb::ndarray<const T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

/// \param value The extent product to name in a message.
/// \param name What it is the product of.
/// \return A short phrase for an error message.
std::string describe(std::size_t value, const char* name) {
    return std::string(name) + " " + std::to_string(value);
}

/// Refuse two buffers that share storage, unless the operation is a pure copy.
///
/// The kernels write `out` while reading the operand, so an overlap is a wrong
/// answer rather than a crash. The all-identity case is the documented exception:
/// it is a forward elementwise copy, the oracle states it is aliasing-safe, and
/// `tests/test_extraction_kernels.py::test_all_identity_apply_aliasing` passes `v`
/// as `out` and expects it to work. An earlier version of this file refused that
/// too, on the grounds that the stricter contract is the safer one; it is not this
/// binding's contract to tighten, and the suite said so.
///
/// **The exemption is a property of the call, and the batch form computes it
/// differently.** For a single cell it follows from the operator list; for a batch it
/// follows from every cell the block references, because one contracting cell makes
/// the shared buffers overlap in a way no later cell's copy undoes. A first fix
/// reached the single-cell functions only and left the batch ones refusing
/// unconditionally -- the same defect, half-applied, and invisible because the two
/// batch aliasing tests use a *mixed* batch and so correctly expect rejection.
///
/// \param first Start of the first buffer.
/// \param first_count Its element count.
/// \param second Start of the second buffer.
/// \param second_count Its element count.
/// \param what A phrase naming the pair, for the message.
/// \param exempt Whether the operation is a pure copy, which may alias.
template <class T, class U>
void refuse_overlap(const T* first, std::size_t first_count, const U* second,
                    std::size_t second_count, const char* what, bool exempt = false) {
    if (exempt) {
        return;
    }
    const auto* lo = reinterpret_cast<const std::byte*>(first);
    const auto* hi = lo + first_count * sizeof(T);
    const auto* other_lo = reinterpret_cast<const std::byte*>(second);
    const auto* other_hi = other_lo + second_count * sizeof(U);
    if (lo < other_hi && other_lo < hi) {
        throw nb::value_error((std::string(what) + " must not share memory").c_str());
    }
}

/// Build the kernel's operator list from the flat argument pack.
///
/// \param matrices The per-direction operators.
/// \param flags Their identity flags.
/// \return The list the kernel takes.
template <class T, std::size_t D>
std::array<ModeOperator<T>, D> pack(const std::array<const_matrix<T>, D>& matrices,
                                    const std::array<bool, D>& flags) {
    std::array<ModeOperator<T>, D> ops{};
    for (std::size_t k = 0; k < D; ++k) {
        ops[k] = ModeOperator<T>{
            span2d<const T>(matrices[k].data(), matrices[k].shape(0), matrices[k].shape(1)), flags[k]};
        if (flags[k] && matrices[k].shape(0) != matrices[k].shape(1)) {
            throw nb::value_error(
                ("direction " + std::to_string(k) +
                 " is flagged identity but its operator is not square")
                    .c_str());
        }
    }
    return ops;
}

/// The product of the operators' input or output extents.
///
/// \param ops The operator list.
/// \param transposed Whether the transpose is being applied, which swaps the roles.
/// \param want_output Whether to take the output extents.
/// \return The product.
template <class T, std::size_t D>
std::size_t extent_product(const std::array<ModeOperator<T>, D>& ops, bool transposed,
                           bool want_output) {
    std::size_t total = 1;
    for (const ModeOperator<T>& op : ops) {
        const bool take_rows = transposed ? !want_output : want_output;
        total *= take_rows ? op.matrix.extent(0) : op.matrix.extent(1);
    }
    return total;
}

/// Whether every direction is the identity, so the driver is a copy and may alias.
///
/// \param ops The operator list.
/// \return True when no direction contracts.
template <class T, std::size_t D>
bool is_pure_copy(const std::array<ModeOperator<T>, D>& ops) {
    for (const ModeOperator<T>& op : ops) {
        if (!op.is_identity) {
            return false;
        }
    }
    return true;
}

/// Whether every cell in a batch is all-identity, so the batch is a copy and may alias.
///
/// The batch counterpart of `is_pure_copy`, computed per **batch** rather than per
/// cell: the exemption applies to the whole call, because one contracting cell makes
/// the shared operand and output buffers overlap in a way that a later cell's copy
/// cannot undo.
///
/// This mirrors `_prepare_apply_many_call`'s own test exactly -- it reads the masks
/// at the **referenced** elements, `is_identity_masks[k][cell_indices[:, k]]`, not
/// the whole mask, so a batch that never visits a contracting element is exempt even
/// where such an element exists. An empty batch is exempt for the reason it is there:
/// nothing is read or written.
///
/// \param masks The per-direction identity masks.
/// \param cells The `(n_cells, D)` index block.
/// \return True when no cell in the batch contracts anything.
template <std::size_t D>
bool batch_is_pure_copy(const std::array<flag_mask, D>& masks, cell_block cells) {
    const std::size_t n_cells = cells.shape(0);
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t k = 0; k < D; ++k) {
            if (!masks[k](static_cast<std::size_t>(cells(cell, k)))) {
                return false;
            }
        }
    }
    return true;
}

/// Check a buffer is at least as long as the kernel will address.
///
/// \param have The buffer's element count.
/// \param need The count the kernel addresses.
/// \param what The buffer's name, for the message.
void check_length(std::size_t have, std::size_t need, const char* what) {
    if (have < need) {
        throw nb::value_error(
            (std::string(what) + " has " + std::to_string(have) + " elements, needs " +
             std::to_string(need))
                .c_str());
    }
}

/// The scratch the unilateral driver addresses, mirroring `_apply_scratch_size`.
///
/// Recomputed here rather than trusted: the caller's `scratch` is validated
/// against it, so an undersized buffer is a `ValueError` instead of a write past
/// the allocation. The formula is `pantr.bspline._extraction_helpers`'s, and
/// `design/extraction_port.md` records why it covers every identity pattern.
///
/// \param ops The operator list.
/// \param transposed Whether the transpose is being applied.
/// \return The element count the driver may address.
template <class T, std::size_t D>
std::size_t unilateral_scratch(const std::array<ModeOperator<T>, D>& ops, bool transposed) {
    if constexpr (D <= 1) {
        return 0;
    } else {
        auto extent = [&](std::size_t k, bool output) {
            const bool take_rows = transposed ? !output : output;
            return take_rows ? ops[k].matrix.extent(0) : ops[k].matrix.extent(1);
        };
        std::size_t largest = 0;
        for (std::size_t k = 0; k + 1 < D; ++k) {
            std::size_t size = 1;
            for (std::size_t j = 0; j <= k; ++j) {
                size *= extent(j, true);
            }
            for (std::size_t j = k + 1; j < D; ++j) {
                size *= extent(j, false);
            }
            largest = std::max(largest, size);
        }
        return 2 * largest;
    }
}

/// The scratch the bilateral driver addresses, mirroring `_bilateral_scratch_size`.
///
/// \param ops The operator list.
/// \param transposed True for `M^T K M`, false for `M K M^T`.
/// \return The element count the driver may address.
template <class T, std::size_t D>
std::size_t bilateral_scratch(const std::array<ModeOperator<T>, D>& ops, bool transposed) {
    auto extent = [&](std::size_t k, bool output) {
        const bool take_rows = transposed ? !output : output;
        return take_rows ? ops[k].matrix.extent(0) : ops[k].matrix.extent(1);
    };
    std::array<std::size_t, 2 * D> shape{};
    for (std::size_t k = 0; k < D; ++k) {
        shape[k] = extent(k, false);
        shape[D + k] = extent(k, false);
    }
    std::size_t largest = 0;
    for (std::size_t stage = 0; stage + 1 < 2 * D; ++stage) {
        const std::size_t k = stage / 2;
        const std::size_t axis = (stage % 2 == 0) ? k : D + k;
        shape[axis] = extent(k, true);
        std::size_t size = 1;
        for (const std::size_t value : shape) {
            size *= value;
        }
        largest = std::max(largest, size);
    }
    return 2 * largest;
}

}  // namespace

namespace {

/// One cell's `M @ v` or `M^T @ v`, validated and dispatched.
///
/// \param matrices The per-direction operators.
/// \param flags Their identity flags.
/// \param v The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T, std::size_t D, bool Transposed>
void unilateral(const std::array<const_matrix<T>, D>& matrices, const std::array<bool, D>& flags,
                const_vector<T> v, out_vector<T> out, out_vector<T> scratch) {
    const auto ops = pack<T, D>(matrices, flags);
    check_length(v.size(), extent_product(ops, Transposed, false), "v");
    check_length(out.size(), extent_product(ops, Transposed, true), "out");
    check_length(scratch.size(), unilateral_scratch(ops, Transposed), "scratch");
    const bool copies = is_pure_copy<T, D>(ops);
    refuse_overlap(v.data(), v.size(), out.data(), out.size(), "v and out", copies);
    refuse_overlap(scratch.data(), scratch.size(), out.data(), out.size(), "scratch and out");

    const std::span<const T> operand(v.data(), v.size());
    const std::span<T> result(out.data(), out.size());
    const std::span<T> work(scratch.data(), scratch.size());

    const nb::gil_scoped_release release;
    if constexpr (Transposed) {
        pantr::bspline::apply_kron_transpose<T>(ops, operand, result, work);
    } else {
        pantr::bspline::apply_kron<T>(ops, operand, result, work);
    }
}

/// One cell's `M^T K M` or `M K M^T`, validated and dispatched.
///
/// \param matrices The per-direction operators.
/// \param flags Their identity flags.
/// \param k_matrix The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T, std::size_t D, bool Transposed>
void bilateral(const std::array<const_matrix<T>, D>& matrices, const std::array<bool, D>& flags,
               const_matrix<T> k_matrix, out_matrix<T> out, out_vector<T> scratch) {
    const auto ops = pack<T, D>(matrices, flags);
    const std::size_t side_in = extent_product(ops, Transposed, false);
    const std::size_t side_out = extent_product(ops, Transposed, true);
    if (k_matrix.shape(0) != side_in || k_matrix.shape(1) != side_in) {
        throw nb::value_error(("K must be square of side " + describe(side_in, "")).c_str());
    }
    if (out.shape(0) != side_out || out.shape(1) != side_out) {
        throw nb::value_error(("out must be square of side " + describe(side_out, "")).c_str());
    }
    const std::size_t k_count = k_matrix.shape(0) * k_matrix.shape(1);
    const std::size_t out_count = out.shape(0) * out.shape(1);
    check_length(scratch.size(), bilateral_scratch(ops, Transposed), "scratch");
    const bool copies = is_pure_copy<T, D>(ops);
    refuse_overlap(k_matrix.data(), k_count, out.data(), out_count, "K and out", copies);
    refuse_overlap(scratch.data(), scratch.size(), out.data(), out_count, "scratch and out");

    const std::span<const T> operand(k_matrix.data(), k_count);
    const std::span<T> result(out.data(), out_count);
    const std::span<T> work(scratch.data(), scratch.size());

    const nb::gil_scoped_release release;
    if constexpr (Transposed) {
        pantr::bspline::apply_kron_mt_k_m<T>(ops, operand, result, work);
    } else {
        pantr::bspline::apply_kron_m_k_mt<T>(ops, operand, result, work);
    }
}

/// Validate the parts of a batch call that do not depend on the operation kind.
///
/// \param maps The per-direction compact index maps.
/// \param masks The per-direction identity masks.
/// \param cells The `(n_cells, D)` index block.
/// \param rows The row counts of the operand, output and scratch, in that order.
template <std::size_t D>
void check_batch(const std::array<index_map, D>& maps, const std::array<flag_mask, D>& masks,
                 cell_block cells, std::array<std::size_t, 3> rows) {
    if (cells.shape(1) != D) {
        throw nb::value_error(
            ("cell_indices must have " + std::to_string(D) + " columns").c_str());
    }
    const std::size_t n_cells = cells.shape(0);
    const char* names[3] = {"the operand", "out", "scratch"};
    for (std::size_t i = 0; i < 3; ++i) {
        if (rows[i] != n_cells) {
            throw nb::value_error((std::string(names[i]) + " has " + std::to_string(rows[i]) +
                                   " rows, expected " + std::to_string(n_cells))
                                      .c_str());
        }
    }
    for (std::size_t k = 0; k < D; ++k) {
        if (maps[k].size() != masks[k].size()) {
            throw nb::value_error(
                ("idx_map and is_id disagree in direction " + std::to_string(k)).c_str());
        }
    }
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t k = 0; k < D; ++k) {
            const std::int64_t index = cells(cell, k);
            if (index < 0 || static_cast<std::size_t>(index) >= masks[k].size()) {
                throw nb::value_error(("cell_indices[" + std::to_string(cell) + ", " +
                                       std::to_string(k) + "] is out of range")
                                          .c_str());
            }
        }
    }
}

/// Assemble one cell's operator list out of the compact per-direction storage.
///
/// Reads `ops_stack[k]` at `maps[k][i_k]` for every direction, including one
/// flagged identity -- whose map entry is 0 and whose row is the sentinel. Only
/// the pointer is taken. This mirrors the oracle's own batch loop.
///
/// \param ops_stack The per-direction compact operator stacks.
/// \param maps The per-direction compact index maps.
/// \param masks The per-direction identity masks.
/// \param cells The index block.
/// \param cell Which cell.
/// \return The operator list for that cell.
template <class T, std::size_t D>
std::array<ModeOperator<T>, D> operators_for(const std::array<const_stack<T>, D>& ops_stack,
                                             const std::array<index_map, D>& maps,
                                             const std::array<flag_mask, D>& masks,
                                             cell_block cells, std::size_t cell) {
    std::array<ModeOperator<T>, D> ops{};
    for (std::size_t k = 0; k < D; ++k) {
        const auto element = static_cast<std::size_t>(cells(cell, k));
        const auto row = static_cast<std::size_t>(maps[k](element));
        const std::size_t rows = ops_stack[k].shape(1);
        const std::size_t cols = ops_stack[k].shape(2);
        if (row >= ops_stack[k].shape(0)) {
            throw nb::value_error(
                ("idx_map points past the compact storage in direction " + std::to_string(k))
                    .c_str());
        }
        ops[k] = ModeOperator<T>{
            span2d<const T>(ops_stack[k].data() + row * rows * cols, rows, cols),
            masks[k](element)};
    }
    return ops;
}

}  // namespace

namespace {

/// A batch of `M @ v` or `M^T @ v`, one cell per row.
///
/// \param ops_stack The per-direction compact operator stacks.
/// \param maps The per-direction compact index maps.
/// \param masks The per-direction identity masks.
/// \param cells The `(n_cells, D)` index block.
/// \param v The operands, one row per cell.
/// \param out The results, one row per cell.
/// \param scratch The work buffers, one row per cell.
template <class T, std::size_t D, bool Transposed>
void unilateral_many(const std::array<const_stack<T>, D>& ops_stack,
                     const std::array<index_map, D>& maps,
                     const std::array<flag_mask, D>& masks, cell_block cells,
                     const_batch_2d<T> v, batch_2d<T> out, batch_2d<T> scratch) {
    check_batch<D>(maps, masks, cells, {v.shape(0), out.shape(0), scratch.shape(0)});
    const bool copies = batch_is_pure_copy<D>(masks, cells);
    refuse_overlap(v.data(), v.size(), out.data(), out.size(), "the operand and out", copies);

    const std::size_t n_cells = cells.shape(0);
    const std::size_t v_stride = v.shape(1);
    const std::size_t out_stride = out.shape(1);
    const std::size_t work_stride = scratch.shape(1);

    // Every cell is validated before any is computed, so the loop below cannot
    // half-fill `out` and then raise.
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        const auto ops = operators_for<T, D>(ops_stack, maps, masks, cells, cell);
        check_length(v_stride, extent_product(ops, Transposed, false), "each operand row");
        check_length(out_stride, extent_product(ops, Transposed, true), "each out row");
        check_length(work_stride, unilateral_scratch(ops, Transposed), "each scratch row");
    }

    const nb::gil_scoped_release release;
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        const auto ops = operators_for<T, D>(ops_stack, maps, masks, cells, cell);
        const std::span<const T> operand(v.data() + cell * v_stride, v_stride);
        const std::span<T> result(out.data() + cell * out_stride, out_stride);
        const std::span<T> work(scratch.data() + cell * work_stride, work_stride);
        if constexpr (Transposed) {
            pantr::bspline::apply_kron_transpose<T>(ops, operand, result, work);
        } else {
            pantr::bspline::apply_kron<T>(ops, operand, result, work);
        }
    }
}

/// A batch of `M^T K M` or `M K M^T`, one cell per slab.
///
/// \param ops_stack The per-direction compact operator stacks.
/// \param maps The per-direction compact index maps.
/// \param masks The per-direction identity masks.
/// \param cells The `(n_cells, D)` index block.
/// \param k_matrix The operands, one square slab per cell.
/// \param out The results, one square slab per cell.
/// \param scratch The work buffers, one row per cell.
template <class T, std::size_t D, bool Transposed>
void bilateral_many(const std::array<const_stack<T>, D>& ops_stack,
                    const std::array<index_map, D>& maps, const std::array<flag_mask, D>& masks,
                    cell_block cells, const_batch_3d<T> k_matrix, batch_3d<T> out,
                    batch_2d<T> scratch) {
    check_batch<D>(maps, masks, cells, {k_matrix.shape(0), out.shape(0), scratch.shape(0)});
    const bool copies = batch_is_pure_copy<D>(masks, cells);
    refuse_overlap(k_matrix.data(), k_matrix.size(), out.data(), out.size(), "K and out", copies);

    const std::size_t n_cells = cells.shape(0);
    const std::size_t k_side = k_matrix.shape(1);
    const std::size_t out_side = out.shape(1);
    if (k_matrix.shape(2) != k_side) {
        throw nb::value_error("each K slab must be square");
    }
    if (out.shape(2) != out_side) {
        throw nb::value_error("each out slab must be square");
    }
    const std::size_t work_stride = scratch.shape(1);

    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        const auto ops = operators_for<T, D>(ops_stack, maps, masks, cells, cell);
        check_length(k_side, extent_product(ops, Transposed, false), "each K slab's side");
        check_length(out_side, extent_product(ops, Transposed, true), "each out slab's side");
        check_length(work_stride, bilateral_scratch(ops, Transposed), "each scratch row");
    }

    const nb::gil_scoped_release release;
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        const auto ops = operators_for<T, D>(ops_stack, maps, masks, cells, cell);
        const std::span<const T> operand(k_matrix.data() + cell * k_side * k_side, k_side * k_side);
        const std::span<T> result(out.data() + cell * out_side * out_side, out_side * out_side);
        const std::span<T> work(scratch.data() + cell * work_stride, work_stride);
        if constexpr (Transposed) {
            pantr::bspline::apply_kron_mt_k_m<T>(ops, operand, result, work);
        } else {
            pantr::bspline::apply_kron_m_k_mt<T>(ops, operand, result, work);
        }
    }
}

}  // namespace

namespace {
/// Flat entry point for `cell_1`.
///
/// \param m0 Direction 0's operator.
/// \param f0 Direction 0's identity flag.
/// \param v The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_1(const_matrix<T> m0, bool f0, const_vector<T> v, out_vector<T> out, out_vector<T> scratch) {
    unilateral<T, 1, false>(std::array<const_matrix<T>, 1>{m0},
                           std::array<bool, 1>{f0}, v, out, scratch);
}
/// Flat entry point for `cell_2`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param v The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_2(const_matrix<T> m0, const_matrix<T> m1, bool f0, bool f1, const_vector<T> v, out_vector<T> out, out_vector<T> scratch) {
    unilateral<T, 2, false>(std::array<const_matrix<T>, 2>{m0, m1},
                           std::array<bool, 2>{f0, f1}, v, out, scratch);
}
/// Flat entry point for `cell_3`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param m2 Direction 2's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param f2 Direction 2's identity flag.
/// \param v The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_3(const_matrix<T> m0, const_matrix<T> m1, const_matrix<T> m2, bool f0, bool f1, bool f2, const_vector<T> v, out_vector<T> out, out_vector<T> scratch) {
    unilateral<T, 3, false>(std::array<const_matrix<T>, 3>{m0, m1, m2},
                           std::array<bool, 3>{f0, f1, f2}, v, out, scratch);
}
/// Flat entry point for `cell_t_1`.
///
/// \param m0 Direction 0's operator.
/// \param f0 Direction 0's identity flag.
/// \param v The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_t_1(const_matrix<T> m0, bool f0, const_vector<T> v, out_vector<T> out, out_vector<T> scratch) {
    unilateral<T, 1, true>(std::array<const_matrix<T>, 1>{m0},
                           std::array<bool, 1>{f0}, v, out, scratch);
}
/// Flat entry point for `cell_t_2`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param v The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_t_2(const_matrix<T> m0, const_matrix<T> m1, bool f0, bool f1, const_vector<T> v, out_vector<T> out, out_vector<T> scratch) {
    unilateral<T, 2, true>(std::array<const_matrix<T>, 2>{m0, m1},
                           std::array<bool, 2>{f0, f1}, v, out, scratch);
}
/// Flat entry point for `cell_t_3`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param m2 Direction 2's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param f2 Direction 2's identity flag.
/// \param v The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_t_3(const_matrix<T> m0, const_matrix<T> m1, const_matrix<T> m2, bool f0, bool f1, bool f2, const_vector<T> v, out_vector<T> out, out_vector<T> scratch) {
    unilateral<T, 3, true>(std::array<const_matrix<T>, 3>{m0, m1, m2},
                           std::array<bool, 3>{f0, f1, f2}, v, out, scratch);
}
/// Flat entry point for `cell_mt_k_m_1`.
///
/// \param m0 Direction 0's operator.
/// \param f0 Direction 0's identity flag.
/// \param k_matrix The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_mt_k_m_1(const_matrix<T> m0, bool f0, const_matrix<T> k_matrix, out_matrix<T> out, out_vector<T> scratch) {
    bilateral<T, 1, true>(std::array<const_matrix<T>, 1>{m0},
                          std::array<bool, 1>{f0}, k_matrix, out, scratch);
}
/// Flat entry point for `cell_mt_k_m_2`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param k_matrix The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_mt_k_m_2(const_matrix<T> m0, const_matrix<T> m1, bool f0, bool f1, const_matrix<T> k_matrix, out_matrix<T> out, out_vector<T> scratch) {
    bilateral<T, 2, true>(std::array<const_matrix<T>, 2>{m0, m1},
                          std::array<bool, 2>{f0, f1}, k_matrix, out, scratch);
}
/// Flat entry point for `cell_mt_k_m_3`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param m2 Direction 2's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param f2 Direction 2's identity flag.
/// \param k_matrix The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_mt_k_m_3(const_matrix<T> m0, const_matrix<T> m1, const_matrix<T> m2, bool f0, bool f1, bool f2, const_matrix<T> k_matrix, out_matrix<T> out, out_vector<T> scratch) {
    bilateral<T, 3, true>(std::array<const_matrix<T>, 3>{m0, m1, m2},
                          std::array<bool, 3>{f0, f1, f2}, k_matrix, out, scratch);
}
/// Flat entry point for `cell_m_k_mt_1`.
///
/// \param m0 Direction 0's operator.
/// \param f0 Direction 0's identity flag.
/// \param k_matrix The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_m_k_mt_1(const_matrix<T> m0, bool f0, const_matrix<T> k_matrix, out_matrix<T> out, out_vector<T> scratch) {
    bilateral<T, 1, false>(std::array<const_matrix<T>, 1>{m0},
                          std::array<bool, 1>{f0}, k_matrix, out, scratch);
}
/// Flat entry point for `cell_m_k_mt_2`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param k_matrix The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_m_k_mt_2(const_matrix<T> m0, const_matrix<T> m1, bool f0, bool f1, const_matrix<T> k_matrix, out_matrix<T> out, out_vector<T> scratch) {
    bilateral<T, 2, false>(std::array<const_matrix<T>, 2>{m0, m1},
                          std::array<bool, 2>{f0, f1}, k_matrix, out, scratch);
}
/// Flat entry point for `cell_m_k_mt_3`.
///
/// \param m0 Direction 0's operator.
/// \param m1 Direction 1's operator.
/// \param m2 Direction 2's operator.
/// \param f0 Direction 0's identity flag.
/// \param f1 Direction 1's identity flag.
/// \param f2 Direction 2's identity flag.
/// \param k_matrix The operand.
/// \param out The result.
/// \param scratch The work buffer.
template <class T>
void cell_m_k_mt_3(const_matrix<T> m0, const_matrix<T> m1, const_matrix<T> m2, bool f0, bool f1, bool f2, const_matrix<T> k_matrix, out_matrix<T> out, out_vector<T> scratch) {
    bilateral<T, 3, false>(std::array<const_matrix<T>, 3>{m0, m1, m2},
                          std::array<bool, 3>{f0, f1, f2}, k_matrix, out, scratch);
}
/// Flat batch entry point for `batch_1`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param cells The `(n_cells, 1)` index block.
/// \param v The operands, one row per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_1(const_stack<T> s0, index_map p0, flag_mask q0, cell_block cells, const_batch_2d<T> v, batch_2d<T> out, batch_2d<T> scratch) {
    unilateral_many<T, 1, false>(std::array<const_stack<T>, 1>{s0},
                                std::array<index_map, 1>{p0},
                                std::array<flag_mask, 1>{q0}, cells,
                                v, out, scratch);
}
/// Flat batch entry point for `batch_2`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param cells The `(n_cells, 2)` index block.
/// \param v The operands, one row per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_2(const_stack<T> s0, const_stack<T> s1, index_map p0, index_map p1, flag_mask q0, flag_mask q1, cell_block cells, const_batch_2d<T> v, batch_2d<T> out, batch_2d<T> scratch) {
    unilateral_many<T, 2, false>(std::array<const_stack<T>, 2>{s0, s1},
                                std::array<index_map, 2>{p0, p1},
                                std::array<flag_mask, 2>{q0, q1}, cells,
                                v, out, scratch);
}
/// Flat batch entry point for `batch_3`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param s2 Direction 2's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param p2 Direction 2's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param q2 Direction 2's identity mask.
/// \param cells The `(n_cells, 3)` index block.
/// \param v The operands, one row per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_3(const_stack<T> s0, const_stack<T> s1, const_stack<T> s2, index_map p0, index_map p1, index_map p2, flag_mask q0, flag_mask q1, flag_mask q2, cell_block cells, const_batch_2d<T> v, batch_2d<T> out, batch_2d<T> scratch) {
    unilateral_many<T, 3, false>(std::array<const_stack<T>, 3>{s0, s1, s2},
                                std::array<index_map, 3>{p0, p1, p2},
                                std::array<flag_mask, 3>{q0, q1, q2}, cells,
                                v, out, scratch);
}
/// Flat batch entry point for `batch_t_1`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param cells The `(n_cells, 1)` index block.
/// \param v The operands, one row per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_t_1(const_stack<T> s0, index_map p0, flag_mask q0, cell_block cells, const_batch_2d<T> v, batch_2d<T> out, batch_2d<T> scratch) {
    unilateral_many<T, 1, true>(std::array<const_stack<T>, 1>{s0},
                                std::array<index_map, 1>{p0},
                                std::array<flag_mask, 1>{q0}, cells,
                                v, out, scratch);
}
/// Flat batch entry point for `batch_t_2`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param cells The `(n_cells, 2)` index block.
/// \param v The operands, one row per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_t_2(const_stack<T> s0, const_stack<T> s1, index_map p0, index_map p1, flag_mask q0, flag_mask q1, cell_block cells, const_batch_2d<T> v, batch_2d<T> out, batch_2d<T> scratch) {
    unilateral_many<T, 2, true>(std::array<const_stack<T>, 2>{s0, s1},
                                std::array<index_map, 2>{p0, p1},
                                std::array<flag_mask, 2>{q0, q1}, cells,
                                v, out, scratch);
}
/// Flat batch entry point for `batch_t_3`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param s2 Direction 2's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param p2 Direction 2's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param q2 Direction 2's identity mask.
/// \param cells The `(n_cells, 3)` index block.
/// \param v The operands, one row per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_t_3(const_stack<T> s0, const_stack<T> s1, const_stack<T> s2, index_map p0, index_map p1, index_map p2, flag_mask q0, flag_mask q1, flag_mask q2, cell_block cells, const_batch_2d<T> v, batch_2d<T> out, batch_2d<T> scratch) {
    unilateral_many<T, 3, true>(std::array<const_stack<T>, 3>{s0, s1, s2},
                                std::array<index_map, 3>{p0, p1, p2},
                                std::array<flag_mask, 3>{q0, q1, q2}, cells,
                                v, out, scratch);
}
/// Flat batch entry point for `batch_mt_k_m_1`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param cells The `(n_cells, 1)` index block.
/// \param k_matrix The operands, one square slab per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_mt_k_m_1(const_stack<T> s0, index_map p0, flag_mask q0, cell_block cells, const_batch_3d<T> k_matrix, batch_3d<T> out, batch_2d<T> scratch) {
    bilateral_many<T, 1, true>(std::array<const_stack<T>, 1>{s0},
                               std::array<index_map, 1>{p0},
                               std::array<flag_mask, 1>{q0}, cells,
                               k_matrix, out, scratch);
}
/// Flat batch entry point for `batch_mt_k_m_2`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param cells The `(n_cells, 2)` index block.
/// \param k_matrix The operands, one square slab per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_mt_k_m_2(const_stack<T> s0, const_stack<T> s1, index_map p0, index_map p1, flag_mask q0, flag_mask q1, cell_block cells, const_batch_3d<T> k_matrix, batch_3d<T> out, batch_2d<T> scratch) {
    bilateral_many<T, 2, true>(std::array<const_stack<T>, 2>{s0, s1},
                               std::array<index_map, 2>{p0, p1},
                               std::array<flag_mask, 2>{q0, q1}, cells,
                               k_matrix, out, scratch);
}
/// Flat batch entry point for `batch_mt_k_m_3`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param s2 Direction 2's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param p2 Direction 2's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param q2 Direction 2's identity mask.
/// \param cells The `(n_cells, 3)` index block.
/// \param k_matrix The operands, one square slab per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_mt_k_m_3(const_stack<T> s0, const_stack<T> s1, const_stack<T> s2, index_map p0, index_map p1, index_map p2, flag_mask q0, flag_mask q1, flag_mask q2, cell_block cells, const_batch_3d<T> k_matrix, batch_3d<T> out, batch_2d<T> scratch) {
    bilateral_many<T, 3, true>(std::array<const_stack<T>, 3>{s0, s1, s2},
                               std::array<index_map, 3>{p0, p1, p2},
                               std::array<flag_mask, 3>{q0, q1, q2}, cells,
                               k_matrix, out, scratch);
}
/// Flat batch entry point for `batch_m_k_mt_1`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param cells The `(n_cells, 1)` index block.
/// \param k_matrix The operands, one square slab per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_m_k_mt_1(const_stack<T> s0, index_map p0, flag_mask q0, cell_block cells, const_batch_3d<T> k_matrix, batch_3d<T> out, batch_2d<T> scratch) {
    bilateral_many<T, 1, false>(std::array<const_stack<T>, 1>{s0},
                               std::array<index_map, 1>{p0},
                               std::array<flag_mask, 1>{q0}, cells,
                               k_matrix, out, scratch);
}
/// Flat batch entry point for `batch_m_k_mt_2`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param cells The `(n_cells, 2)` index block.
/// \param k_matrix The operands, one square slab per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_m_k_mt_2(const_stack<T> s0, const_stack<T> s1, index_map p0, index_map p1, flag_mask q0, flag_mask q1, cell_block cells, const_batch_3d<T> k_matrix, batch_3d<T> out, batch_2d<T> scratch) {
    bilateral_many<T, 2, false>(std::array<const_stack<T>, 2>{s0, s1},
                               std::array<index_map, 2>{p0, p1},
                               std::array<flag_mask, 2>{q0, q1}, cells,
                               k_matrix, out, scratch);
}
/// Flat batch entry point for `batch_m_k_mt_3`.
///
/// \param s0 Direction 0's compact operator stack.
/// \param s1 Direction 1's compact operator stack.
/// \param s2 Direction 2's compact operator stack.
/// \param p0 Direction 0's compact index map.
/// \param p1 Direction 1's compact index map.
/// \param p2 Direction 2's compact index map.
/// \param q0 Direction 0's identity mask.
/// \param q1 Direction 1's identity mask.
/// \param q2 Direction 2's identity mask.
/// \param cells The `(n_cells, 3)` index block.
/// \param k_matrix The operands, one square slab per cell.
/// \param out The results.
/// \param scratch The work buffers.
template <class T>
void batch_m_k_mt_3(const_stack<T> s0, const_stack<T> s1, const_stack<T> s2, index_map p0, index_map p1, index_map p2, flag_mask q0, flag_mask q1, flag_mask q2, cell_block cells, const_batch_3d<T> k_matrix, batch_3d<T> out, batch_2d<T> scratch) {
    bilateral_many<T, 3, false>(std::array<const_stack<T>, 3>{s0, s1, s2},
                               std::array<index_map, 3>{p0, p1, p2},
                               std::array<flag_mask, 3>{q0, q1, q2}, cells,
                               k_matrix, out, scratch);
}
}  // namespace

void register_bspline_extraction(nb::module_& m) {
    // `.noconvert()` on every array: these kernels accumulate in the operator's
    // own scalar type, so a silent cast would change the arithmetic width rather
    // than merely copy. Argument names mirror `pantr.bspline._extraction_kernels`
    // and there is no `nb::kw_only()`, because the Layer 2 dispatcher calls
    // positionally.
    m.def("apply_kron_1d", &cell_1<double>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_1d", &cell_1<float>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_2d", &cell_2<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_2d", &cell_2<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_3d", &cell_3<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_3d", &cell_3<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_T_1d", &cell_t_1<double>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_T_1d", &cell_t_1<float>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_T_2d", &cell_t_2<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_T_2d", &cell_t_2<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_T_3d", &cell_t_3<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_T_3d", &cell_t_3<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_1d", &cell_mt_k_m_1<double>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_1d", &cell_mt_k_m_1<float>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_2d", &cell_mt_k_m_2<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_2d", &cell_mt_k_m_2<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_3d", &cell_mt_k_m_3<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_3d", &cell_mt_k_m_3<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_1d", &cell_m_k_mt_1<double>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_1d", &cell_m_k_mt_1<float>, nb::arg("M_0").noconvert(), nb::arg("is_id_0"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_2d", &cell_m_k_mt_2<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_2d", &cell_m_k_mt_2<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_3d", &cell_m_k_mt_3<double>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_3d", &cell_m_k_mt_3<float>, nb::arg("M_0").noconvert(), nb::arg("M_1").noconvert(), nb::arg("M_2").noconvert(), nb::arg("is_id_0"), nb::arg("is_id_1"), nb::arg("is_id_2"), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());

    m.def("apply_kron_apply_many_1d", &batch_1<double>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_many_1d", &batch_1<float>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_many_2d", &batch_2<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_many_2d", &batch_2<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_many_3d", &batch_3<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_many_3d", &batch_3<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_T_many_1d", &batch_t_1<double>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_T_many_1d", &batch_t_1<float>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_T_many_2d", &batch_t_2<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_T_many_2d", &batch_t_2<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_T_many_3d", &batch_t_3<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_apply_T_many_3d", &batch_t_3<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("v").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_many_1d", &batch_mt_k_m_1<double>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_many_1d", &batch_mt_k_m_1<float>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_many_2d", &batch_mt_k_m_2<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_many_2d", &batch_mt_k_m_2<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_many_3d", &batch_mt_k_m_3<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_MT_K_M_many_3d", &batch_mt_k_m_3<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_many_1d", &batch_m_k_mt_1<double>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_many_1d", &batch_m_k_mt_1<float>, nb::arg("ops_0").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_many_2d", &batch_m_k_mt_2<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_many_2d", &batch_m_k_mt_2<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_many_3d", &batch_m_k_mt_3<double>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
    m.def("apply_kron_M_K_MT_many_3d", &batch_m_k_mt_3<float>, nb::arg("ops_0").noconvert(), nb::arg("ops_1").noconvert(), nb::arg("ops_2").noconvert(), nb::arg("idx_map_0").noconvert(), nb::arg("idx_map_1").noconvert(), nb::arg("idx_map_2").noconvert(), nb::arg("is_id_0").noconvert(), nb::arg("is_id_1").noconvert(), nb::arg("is_id_2").noconvert(), nb::arg("cell_indices").noconvert(), nb::arg("K").noconvert(), nb::arg("out").noconvert(), nb::arg("scratch").noconvert());
}
