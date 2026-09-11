#pragma once

/// \file
/// Changing a B-spline field's degree: the hodograph, and degree elevation.
///
/// Ports `pantr.bspline.Bspline.derivative` on its non-rational, degree-lowering path
/// -- `_bspline_derivative.py`'s `_derivative_nonrational_nd` over `_derivative_ctrl_1d`
/// -- and `pantr.bspline.Bspline.elevate_degree` on an open direction, whose substance
/// is `_bspline_degree.py`'s `_degree_elevate_bspline` over `_bspline_degree_core.py`'s
/// `_degree_elevate_1d_core` (Piegl and Tiller, *The NURBS Book*, A5.9). The Python side
/// stays as the parity oracle.
///
/// ## What this file adds, and what it only calls
///
/// The two new numerics are `derivative_along_axis` and `degree_elevate_1d`. Everything
/// structural around them is already here and is called rather than rewritten:
/// `pantr/bspline/structural.hpp`'s `select_rows_along_axis` performs both of the
/// derivative's periodic row moves -- the modulo expansion on the way in and the trim on
/// the way out -- and its `extents_before`, `extents_after`, `require_axis` and
/// `assemble` carry the axis bookkeeping. `pantr/bspline/knots.hpp`'s `num_basis` decides
/// how many rows the trim keeps, and `pantr/core/binomial.hpp`'s `bincoeff` and
/// `require_bincoeff_envelope` supply A5.9's coefficient table and the refusal that keeps
/// it inside the exact-integer envelope.
///
/// So the only recurrences below are the two the oracle spells out itself.
///
/// ## Free functions, not methods
///
/// `pantr/bspline/bspline.hpp` lists these among the computations *over* a field rather
/// than properties *of* one, and states that each is "a separate port over free functions
/// taking a `const Bspline&`". `pantr/bspline/refinement.hpp` and
/// `pantr/bspline/structural.hpp` draw the same line. So `derivative` and
/// `elevate_degree` take a `const Bspline<T>&` and return a new field; the type has no
/// mutator and this file adds none.
///
/// ## The floating-point discipline, quantity by quantity
///
/// **The derivative's coefficients are computed entirely in `T`, and nothing in them can
/// fuse.** `_derivative_ctrl_1d` is plain numpy rather than a numba kernel, so NEP 50
/// decides its widths: `knots[p + 1 : n + p] - knots[1 : n]` and `ctrl[1:] - ctrl[:-1]`
/// are array expressions in the storage format, the Python `int` degree is *weak* and is
/// converted to that format rather than promoting it, and `np.divide`'s output array is
/// allocated at `diff.dtype`. Each output coefficient is therefore
/// `fl(fl(p * fl(c[i+1] - c[i])) / fl(k[i+p+1] - k[i+1]))` with every rounding in `T`,
/// which is what the sweep below computes operand for operand.
///
/// That expression contains no sum of a product, so **no site in it can contract**, and
/// the bitwise claim for the hodograph is a property of the code rather than of the host.
/// That is worth stating because the two sibling ports say the opposite of their own
/// kernels: `pantr/bspline/refinement.hpp` and the elevation below both carry an
/// `a * b + c * d` that a target with a fused multiply-add would contract, and both
/// condition their claims on `pantr._pantr_cpp.__fp_contract__` per
/// `design/backend_parity.md` Rule 7.
///
/// **A zero denominator produces an exact zero rather than a division.** The oracle
/// divides under `where=denom != 0.0` into a pre-zeroed array, so an interior knot of
/// multiplicity `degree + 1` -- a `C^-1` breakpoint, where the two sides share no
/// coefficient -- leaves its derivative coefficient at zero without evaluating the
/// quotient. The sweep skips the same rows. Reproducing the division instead would raise
/// a division-by-zero the oracle deliberately never performs.
///
/// **Degree elevation mixes three widths inside one kernel, and every one of them had to
/// be measured.** `design/backend_parity.md` Rule 9 says an accumulation width is a
/// per-kernel fact; `_degree_elevate_1d_core` is sharper than that, because the width
/// varies *within* the kernel and follows numba's SSA type inference rather than anything
/// a reader can see. Its `tmp1` is assigned in four places, and the four do not agree:
/// `alfs[q - s] * bpts[q, ii]` reads a `float64` table and is `float64`, while
/// `alf * ic[i, ii]`, `gam * ebpts[kj, ii]` and `bet * ebpts[kj, ii]` read scalars that
/// are themselves `T` and are `float32` on a `float32` net. `tmp2` is `float64` at all
/// five of its assignments, because each carries the literal `1.0` or a `float64` table
/// entry. `scripts/measure_bspline_degree_widths.py` prints numba's own inferred type for
/// every one of these, so the table is reproducible rather than asserted:
///
/// | quantity | oracle's width at a `float32` net | what this file does |
/// |---|---|---|
/// | `bezalfs[j, i] = inv * C(d, j) * C(t, i - j)` | `float64` throughout | `double` throughout |
/// | `alfs[q - mul - 1] = numer / (knots[a + q] - ua)` | divides in `T`, widens on the store | divides in `T`, stores into a `double` |
/// | `bpts[q, ii] = tmp1 + tmp2` (Boehm insertion) | both terms `float64`, narrows on the store | both terms `double`, narrows on the store |
/// | `ebpts[i, ii] += bezalfs[j, i] * bpts[j, ii]` | term `float64`, **narrows every term** | term `double`, narrows every term |
/// | `ic[i, ii] = tmp1 + tmp2` and the two `ebpts` blends | `tmp1` in `T`, `tmp2` in `float64` | product in `T`, companion in `double` |
///
/// The last row is the one no reading of the source gives: `alf`, `gam` and `bet` are
/// quotients of `T` knots and stay `T`, so `alf * ic[i, ii]` rounds to `float32` before it
/// is added to a `float64` companion, and a C++ side that formed both products in `double`
/// would be exact at `float64` and quietly wrong at `float32` -- which Rule 9 names as the
/// half of the matrix nobody reads first.
///
/// **The companion halves of those last three blends are the one place in the table where
/// the width cannot matter, and saying so is what stops the parity suite being asked for
/// a case that does not exist.** All three weights are at least one: `ik` is built
/// non-decreasingly out of knots at or below the segment's own left endpoint `ua`, so
/// `alf = (ub - ik[i]) / (ua - ik[i])` and `bet`, `gam` = `(ub - ik[.]) / (ub - ua)` each
/// have a numerator at least their denominator. For `1 <= w < 2^24` in `float`,
/// `float(1) - w` is exact -- the exact difference is a multiple of `ulp(w)`, and `1` is
/// itself such a multiple exactly while `ulp(w) <= 1`, which is the `2^24` -- so it equals
/// `1.0 - double(w)` bit for bit. The upper hypothesis is real rather than a formality:
/// `float(1) - 16777218` is off by one ulp. It holds here because these weights are ratios
/// of knot differences within one vector, and a vector whose spans differ by a factor of
/// `2^24` has other problems; a sweep of three thousand vectors to degree 8 reached 623.
/// Written in the storage format or in `double`, that subtraction is the same number, and
/// a mutation of it passes every parity test there could be. Measured over a sweep of
/// three thousand knot vectors to degree 8: the smallest `alf` was 1.0007, the smallest
/// `bet` 1.0001 and the smallest `gam` 1.0073, and `float(1) - w` was inexact on none of
/// 200000 sampled `w >= 1`.
///
/// **The elevated knot vector is written, not computed.** Every entry of `ik` is a copy of
/// `ua` or `ub`, which are elements of the input vector, so the output knots carry no
/// arithmetic at all and a difference there could only be a lost value or a miscount.
///
/// **The elevation sweeps outer blocks rather than transposing.** The oracle calls
/// `_flatten_along_axis`, which is `np.moveaxis` plus `np.ascontiguousarray`: a
/// permutation of values, not an arithmetic step. Under the `(outer, num_rows, inner)`
/// reading this package uses, each outer block is already contiguous and is handed to the
/// curve kernel as it stands, with `inner` in the role of A5.9's `rank`. Nothing in A5.9
/// mixes two components -- every array it touches is indexed `[., ii]` and every scalar it
/// forms is a function of the knots alone -- so the blocks are independent destinations
/// and the permutation changes no operation and no operand.
///
/// The cost of not transposing is that the scalar bookkeeping runs once per outer block
/// instead of once: `bezalfs`, `alfs`, and the `alf`/`bet`/`gam` blends are recomputed
/// identically each time. That is also why the first block's knot vector is the one
/// returned -- the bookkeeping reads the knots and the degree and nothing else, so every
/// block writes the same vector.
///
/// ## What is not ported, and it is a declared boundary
///
/// **`reduce_degree` is not here, and neither is degree elevation on a periodic
/// direction.** Both round-trip through conversions `pantr/bspline/structural.hpp` has
/// already declared as its own boundary: `_degree_elevate_bspline` converts a periodic
/// direction to its open form, elevates, and calls `_to_periodic_bspline_1d_impl` to
/// convert it back, and `_degree_reduce_bspline` additionally calls
/// `_remove_knot_bspline_1d_impl` for every interior breakpoint it has to coarsen. That
/// file states why neither conversion is here -- each decides *whether an operation is
/// admissible* by a tolerance on a geometric quantity, and the periodic one reaches
/// `numpy.linalg.qr` for a least-squares residual whose factorization is LAPACK's on one
/// side and Eigen's on the other. Porting either under cover of this file would be
/// porting that operation, not this one.
///
/// So `elevate_degree` refuses a periodic direction that is to be elevated, exactly as
/// `pantr/bspline/refinement.hpp` refuses one that is to receive knots; a direction with a
/// zero increment is untouched and its space handle is carried over whatever it is.
///
/// **`elevate_degree` also refuses a direction whose knot vector is not clamped, and that
/// one is a defect in the oracle rather than a missing port.** A5.9 walks one segment per
/// knot run and stops when the run it is on reaches the last index of the vector, which
/// only happens where the vector closes with `degree + 1` equal knots. Given an unclamped
/// vector it walks past the end and reads `ctrl[b - degree + j]` for a `b` the coefficient
/// array does not have. On a degree-3 curve over
/// `[-0.3, -0.2, -0.1, 0, 0.25, 0.5, 0.75, 1, 1.1, 1.2, 1.3]` with seven coefficients the
/// walk asks for `ctrl[7]`. Interpreted, under `NUMBA_DISABLE_JIT=1`, that call raises
/// `IndexError: index 7 is out of bounds for axis 0 with size 7`; compiled, numba does not
/// bounds check in `nopython` mode, so it returns whatever the read found and nothing
/// reports it in the configuration the library normally runs in. What those last
/// coefficients hold is not quoted here, because it is not a measurement of anything.
/// `scripts/measure_bspline_degree_widths.py` prints both halves.
///
/// That is not a boundary this port may quietly adopt a different answer for, so the
/// refusal is paired with a route: `pantr.bspline._degree_backend` sends an unclamped
/// direction to the oracle, exactly as it sends a periodic one, and what the library
/// accepts does not change with `PANTR_BACKEND`. What the refusal does buy is that the
/// C++ core never performs the read, which in C++ would be undefined behaviour rather
/// than a wrong number.
///
/// **`derivative` has no `keep_degree` parameter and no rational path, and the reason is
/// not that they are hard.** Both are reachable in the oracle and both currently have an
/// open defect, so there is nothing stable for a C++ side to be at parity with:
///
/// - `derivative(keep_degree=True)` on an **unclamped, non-periodic** direction returns a
///   wrong function, and the cause is very likely the one the next section documents:
///   that path re-elevates through A5.9, and the vector it hands it is `knots[1:-1]` of
///   an unclamped vector, which is unclamped too -- so it meets the same out-of-bounds
///   walk. On a curve where the counts happen to line up the walk returns rather than
///   raising and the result is simply wrong; on one where they do not, the field's own
///   constructor refuses it on the coefficient count. No figure is quoted for how wrong,
///   because none taken here would be reproducible by anything in this tree.
/// - the **rational** derivative raises a multiplicity error whenever the differentiated
///   direction is periodic, so the call does not complete at all.
///
/// Both are with the repository's owner. A port pinned to a result that is about to change
/// would have to be re-derived when it does, and a parity test over it would be asserting
/// that two backends reproduce the same wrong answer -- which
/// `design/backend_parity.md` opens by warning is invisible to every parity test that will
/// ever be written. `keep_degree` is therefore absent from the signature rather than
/// refused at run time: the strongest way to declare a boundary is not to offer the door.
/// A rational field *is* refused, because `Bspline<T>` carries the flag and a C++ caller
/// could hand one over.
///
/// ## Validating rather than asserting
///
/// This is the C++ counterpart of Layer 2, so it validates and throws in a release build
/// as much as in a debug one, with the oracle's messages character for character.
/// `pantr/core/error.hpp` sets the split: value and range checks here, type-kind checks in
/// the Python wrapper.
///
/// Five of the refusals below are the oracle's **Layer 1** checks -- the direction range,
/// the degree-0 derivative, and degree elevation's three argument checks. They are
/// near-vacuous on the Python path, where `pantr.bspline.Bspline` has already made them,
/// and load-bearing for a C++ caller with no wrapper in front of it. That is the same
/// reason `pantr/bspline/refinement.hpp` restates three of `insert_knots`' Layer 1 checks.
///
/// One refusal of the oracle's cannot live here and stays the wrapper's:
/// `Bspline.elevate_degree` accepts a bare `int` and broadcasts it over the directions,
/// and a `std::span` cannot tell a scalar from a sequence of one.
///
/// ## Thread safety
///
/// Both entry points read their argument and allocate their result, and every type they
/// touch is immutable. They are safe to call concurrently on the same field with no
/// external locking, which is the contract `design/bspline_derived_caches.md` states for
/// this package.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/bspline/structural.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// Differentiate one axis of a row-major array by the B-spline derivative formula.
///
/// The array is read as `(outer, num_rows, inner)` and written as
/// `(outer, num_rows - 1, inner)`, with
///
///     out(o, i, k) = degree * (values(o, i + 1, k) - values(o, i, k))
///                    / (knots[i + degree + 1] - knots[i + 1]),
///
/// every operation in `T`, and an exact `T(0)` wherever that denominator vanishes. For a
/// control net of shape `(e_0, ..., e_{d-1}, num_components)` differentiated along
/// direction `d`, `outer` is the product of the extents before `d` and `inner` the
/// product of those after it, the component axis included --
/// `pantr/bspline/structural.hpp`'s own convention.
///
/// A vanishing denominator is an interior knot of multiplicity `degree + 1`, where the
/// field is `C^-1` and the two sides share no coefficient. The oracle's
/// `np.divide(..., where=denom != 0.0)` leaves the pre-zeroed output alone there, and so
/// does this; the quotient is never formed.
///
/// \param knots The direction's knot vector, at least `num_rows + degree + 1` entries.
/// \param degree The direction's polynomial degree, at least 1.
/// \param values The coefficients, row-major under `(outer, num_rows, inner)`.
/// \param num_rows The differentiated direction's extent in `values`.
/// \param outer The product of the extents ahead of that axis; 1 when it is the first.
/// \param inner The product of the extents behind it, component axis included.
/// \return `outer * (num_rows - 1) * inner` coefficients, row-major.
/// \throws std::invalid_argument If `degree` is below 1, if an extent is negative, if
///         `num_rows` is below 2, if the knot vector is too short, or if `values.size()`
///         is not `outer * num_rows * inner`.
///
/// \note Every rounding is in `T` and no site can contract; see the file comment for why
///       that makes the claim independent of the host, unlike this file's elevation.
template <Real T>
[[nodiscard]] std::vector<T> derivative_along_axis(std::span<const T> knots, std::int64_t degree,
                                                   std::span<const T> values,
                                                   std::int64_t num_rows, std::int64_t outer,
                                                   std::int64_t inner) {
    if (degree < 1) {
        throw std::invalid_argument("derivative_along_axis: the degree must be at least 1, got "
                                    + std::to_string(degree) + ".");
    }
    if (outer < 0 || inner < 0) {
        throw std::invalid_argument("derivative_along_axis: the extents are counts, so none of "
                                    "them can be negative.");
    }
    if (num_rows < 2) {
        throw std::invalid_argument("derivative_along_axis: a direction needs at least two "
                                    "coefficients to differentiate, got "
                                    + std::to_string(num_rows) + ".");
    }
    const auto knot_count = static_cast<std::int64_t>(knots.size());
    if (knot_count < num_rows + degree + 1) {
        throw std::invalid_argument(
            "derivative_along_axis: " + std::to_string(num_rows) + " coefficients at degree "
            + std::to_string(degree) + " need at least " + std::to_string(num_rows + degree + 1)
            + " knots, got " + std::to_string(knot_count) + ".");
    }
    const std::int64_t expected = outer * num_rows * inner;
    if (static_cast<std::int64_t>(values.size()) != expected) {
        throw std::invalid_argument(
            "derivative_along_axis: the buffer holds " + std::to_string(values.size())
            + " coefficients and the shape (" + std::to_string(outer) + ", "
            + std::to_string(num_rows) + ", " + std::to_string(inner) + ") needs "
            + std::to_string(expected) + ".");
    }

    const std::int64_t out_rows = num_rows - 1;
    std::vector<T> out(static_cast<std::size_t>(outer * out_rows * inner), T(0));

    // The oracle forms the whole `denom` array before dividing, so it is built here too
    // rather than recomputed per outer block: same subtraction, same operands, once.
    std::vector<T> denominators(static_cast<std::size_t>(out_rows));
    for (std::int64_t i = 0; i < out_rows; ++i) {
        denominators[static_cast<std::size_t>(i)] =
            knots[static_cast<std::size_t>(i + degree + 1)] - knots[static_cast<std::size_t>(i + 1)];
    }

    // Exact while `degree` is below 2^24, which is where `float` stops holding every
    // integer. Nothing on this path enforces that -- `require_bincoeff_envelope` guards
    // the elevation below and not the hodograph, and `BsplineSpace1D` caps no degree at
    // construction -- so it is a property of the degrees anyone builds rather than of a
    // check, and it is said that way rather than attributed to one.
    const T scale = static_cast<T>(degree);

    for (std::int64_t o = 0; o < outer; ++o) {
        const T* const in_block = values.data() + o * num_rows * inner;
        T* const out_block = out.data() + o * out_rows * inner;
        for (std::int64_t i = 0; i < out_rows; ++i) {
            const T denominator = denominators[static_cast<std::size_t>(i)];
            if (denominator == T(0)) {
                // The oracle's `where=denom != 0.0`: the row keeps the zero it was
                // allocated with, and the quotient is never formed.
                continue;
            }
            const T* const lower = in_block + i * inner;
            const T* const upper = in_block + (i + 1) * inner;
            T* const row = out_block + i * inner;
            for (std::int64_t k = 0; k < inner; ++k) {
                // The oracle's `np.divide(degree * diff, denom, ...)`, operand for
                // operand: the difference, then the scaling, then the single division,
                // all three in `T`.
                row[k] = scale * (upper[k] - lower[k]) / denominator;
            }
        }
    }
    return out;
}

/// One direction's coefficients and knots after degree elevation.
///
/// \tparam T The scalar type the coefficients and knots are stored in.
template <Real T>
struct ElevatedAxis {
    /// The elevated coefficients, row-major under `(outer, num_rows, inner)`.
    std::vector<T> values;
    /// The elevated knot vector.
    std::vector<T> knots;
    /// The elevated direction's extent in `values`.
    std::int64_t num_rows;
};

/// Degree-elevate one B-spline curve, Piegl and Tiller A5.9.
///
/// Elevation preserves smoothness: a breakpoint where the curve is `C^s` stays `C^s`, so
/// its multiplicity goes from `m` to `m + increment`, which is what the knot-writing step
/// emits. That includes `m = degree + 1`, a `C^-1` breakpoint -- the two adjacent Bézier
/// segments share no coefficient there, so the segment after it starts contributing at its
/// own index 0 rather than at 1. This is the oracle's amended A5.9 rather than the
/// published one, in the two places `_degree_elevate_1d_core` says so: that `lbz = 0` case,
/// and a `b <= m` loop bound in place of `b < m` so that a degree-0 curve keeps its last
/// segment.
///
/// The output is over-allocated and trimmed, as the oracle does: neither side derives a
/// tight size from the multiplicity structure, and deriving one here would be a second
/// answer to a question the oracle answers by counting as it writes.
///
/// \param degree The curve's polynomial degree.
/// \param coefficients The coefficients, row-major `(num_rows, rank)`.
/// \param num_rows The number of coefficients, which is the space's basis count.
/// \param rank The number of components per coefficient.
/// \param knots The knot vector, at least `num_rows + degree + 1` entries, **clamped**:
///        its last `degree + 1` entries must be equal, or the segment walk reads past the
///        coefficients. `elevate_degree` refuses an unclamped direction for that reason;
///        see the file comment.
/// \param increment Degrees to add, at least 1. `degree + increment` must be inside the
///        exact-integer binomial envelope; the caller establishes that.
/// \return The elevated coefficients, knots and coefficient count.
/// \throws std::invalid_argument If `degree` is negative, if `increment` is below 1, if an
///         extent is below 1, if the knot vector is too short, or if
///         `coefficients.size()` is not `num_rows * rank`.
///
/// \note The widths are the oracle's, site by site, and they are not uniform; the file
///       comment tabulates them and names the script that measures them.
template <Real T>
[[nodiscard]] ElevatedAxis<T> degree_elevate_1d(std::int64_t degree,
                                                std::span<const T> coefficients,
                                                std::int64_t num_rows, std::int64_t rank,
                                                std::span<const T> knots,
                                                std::int64_t increment) {
    if (degree < 0) {
        throw std::invalid_argument("degree_elevate_1d: the degree is a count, so it cannot be "
                                    "negative, got " + std::to_string(degree) + ".");
    }
    if (increment < 1) {
        throw std::invalid_argument("degree_elevate_1d: the increment must be at least 1, got "
                                    + std::to_string(increment) + ".");
    }
    if (num_rows < 1 || rank < 1) {
        throw std::invalid_argument("degree_elevate_1d: a curve needs at least one coefficient "
                                    "of at least one component.");
    }
    const auto knot_count = static_cast<std::int64_t>(knots.size());
    if (knot_count < num_rows + degree + 1) {
        throw std::invalid_argument(
            "degree_elevate_1d: " + std::to_string(num_rows) + " coefficients at degree "
            + std::to_string(degree) + " need at least " + std::to_string(num_rows + degree + 1)
            + " knots, got " + std::to_string(knot_count) + ".");
    }
    if (static_cast<std::int64_t>(coefficients.size()) != num_rows * rank) {
        throw std::invalid_argument(
            "degree_elevate_1d: the buffer holds " + std::to_string(coefficients.size())
            + " coefficients and the shape (" + std::to_string(num_rows) + ", "
            + std::to_string(rank) + ") needs " + std::to_string(num_rows * rank) + ".");
    }

    const std::int64_t d = degree;
    const std::int64_t t = increment;
    const std::int64_t n = num_rows - 1;
    const std::int64_t ph = d + t;
    const std::int64_t ph2 = ph / 2;
    const std::int64_t m = n + d + 1;

    // The segment walk ends when a run of equal knots reaches index `m`, and nothing else
    // stops it: without that run it steps past the coefficients. Checked here rather than
    // assumed, because the oracle does not check it and reads out of bounds instead; the
    // file comment records the measurement.
    for (std::int64_t i = m - d; i < m; ++i) {
        if (knots[static_cast<std::size_t>(i)] != knots[static_cast<std::size_t>(m)]) {
            throw std::invalid_argument(
                "degree_elevate_1d: the knot vector must close with " + std::to_string(d + 1)
                + " equal knots, and knots[" + std::to_string(i) + "] differs from knots["
                + std::to_string(m) + "].");
        }
    }

    // `bezalfs` and `alfs` are `float64` in the oracle whatever the net stores, and
    // `bpts`, `ebpts`, `next_bpts` and `ic` follow the net; see the file comment's table.
    std::vector<double> bezalfs(static_cast<std::size_t>((d + 1) * (ph + 1)), 0.0);
    std::vector<double> alfs(static_cast<std::size_t>(d), 0.0);
    std::vector<T> bpts(static_cast<std::size_t>((d + 1) * rank), T(0));
    std::vector<T> ebpts(static_cast<std::size_t>((ph + 1) * rank), T(0));
    std::vector<T> next_bpts(static_cast<std::size_t>((d + 1) * rank), T(0));

    const auto bez = [ph](std::int64_t j, std::int64_t i) {
        return static_cast<std::size_t>(j * (ph + 1) + i);
    };

    bezalfs[bez(0, 0)] = 1.0;
    bezalfs[bez(d, ph)] = 1.0;
    for (std::int64_t i = 1; i <= ph2; ++i) {
        const double inv = 1.0 / core::bincoeff(static_cast<int>(ph), static_cast<int>(i));
        const std::int64_t mpi = std::min(d, i);
        for (std::int64_t j = std::max<std::int64_t>(0, i - t); j <= mpi; ++j) {
            bezalfs[bez(j, i)] = inv * core::bincoeff(static_cast<int>(d), static_cast<int>(j))
                                 * core::bincoeff(static_cast<int>(t), static_cast<int>(i - j));
        }
    }
    for (std::int64_t i = ph2 + 1; i < ph; ++i) {
        const std::int64_t mpi = std::min(d, i);
        for (std::int64_t j = std::max<std::int64_t>(0, i - t); j <= mpi; ++j) {
            bezalfs[bez(j, i)] = bezalfs[bez(d - j, ph - i)];
        }
    }

    std::int64_t kind = ph + 1;
    std::int64_t r = -1;
    std::int64_t a = d;
    std::int64_t b = d + 1;
    std::int64_t cind = 1;
    T ua = knots[0];

    // The oracle's own over-allocation, reproduced rather than tightened.
    std::vector<T> ik(static_cast<std::size_t>(knot_count + t * knot_count), T(0));
    std::vector<T> ic(static_cast<std::size_t>((num_rows + t * knot_count) * rank), T(0));

    for (std::int64_t ii = 0; ii < rank; ++ii) {
        ic[static_cast<std::size_t>(ii)] = coefficients[static_cast<std::size_t>(ii)];
    }
    for (std::int64_t i = 0; i <= ph; ++i) {
        ik[static_cast<std::size_t>(i)] = ua;
    }
    for (std::int64_t i = 0; i <= d; ++i) {
        for (std::int64_t ii = 0; ii < rank; ++ii) {
            bpts[static_cast<std::size_t>(i * rank + ii)] =
                coefficients[static_cast<std::size_t>(i * rank + ii)];
        }
    }

    while (b <= m) {
        const std::int64_t run_start = b;
        while (b < m
               && knots[static_cast<std::size_t>(b)] == knots[static_cast<std::size_t>(b + 1)]) {
            ++b;
        }
        const std::int64_t mul = b - run_start + 1;
        const T ub = knots[static_cast<std::size_t>(b)];
        const std::int64_t oldr = r;
        r = d - mul;

        std::int64_t lbz = 1;
        if (oldr > 0) {
            lbz = (oldr + 2) / 2;
        } else if (oldr < 0 && a != d) {
            lbz = 0;
        }
        const std::int64_t rbz = (r > 0) ? ph - (r + 1) / 2 : ph;

        if (r > 0) {
            // `numer` and the knot difference are both `T`, so the quotient is formed in
            // `T` and widens only on the store into a `float64` table.
            const T numer = ub - ua;
            for (std::int64_t q = d; q > mul; --q) {
                alfs[static_cast<std::size_t>(q - mul - 1)] = static_cast<double>(
                    numer / (knots[static_cast<std::size_t>(a + q)] - ua));
            }
            for (std::int64_t j = 1; j <= r; ++j) {
                const std::int64_t save = r - j;
                const std::int64_t s = mul + j;
                for (std::int64_t q = d; q >= s; --q) {
                    const double alf = alfs[static_cast<std::size_t>(q - s)];
                    for (std::int64_t ii = 0; ii < rank; ++ii) {
                        // Both terms are `float64` in the oracle -- a `float64` table entry
                        // against a `T` array element -- and the sum narrows on the store.
                        const double kept =
                            alf * static_cast<double>(bpts[static_cast<std::size_t>(q * rank + ii)]);
                        const double carried =
                            (1.0 - alf)
                            * static_cast<double>(bpts[static_cast<std::size_t>((q - 1) * rank + ii)]);
                        bpts[static_cast<std::size_t>(q * rank + ii)] =
                            static_cast<T>(kept + carried);
                    }
                }
                for (std::int64_t ii = 0; ii < rank; ++ii) {
                    next_bpts[static_cast<std::size_t>(save * rank + ii)] =
                        bpts[static_cast<std::size_t>(d * rank + ii)];
                }
            }
        }

        for (std::int64_t i = lbz; i <= ph; ++i) {
            for (std::int64_t ii = 0; ii < rank; ++ii) {
                ebpts[static_cast<std::size_t>(i * rank + ii)] = T(0);
            }
            const std::int64_t mpi = std::min(d, i);
            for (std::int64_t j = std::max<std::int64_t>(0, i - t); j <= mpi; ++j) {
                for (std::int64_t ii = 0; ii < rank; ++ii) {
                    // The oracle's `ebpts[i, ii] += bezalfs[j, i] * bpts[j, ii]`: a
                    // `float64` term added to a `T` accumulator, so **every** term rounds
                    // to `T` rather than the sum rounding once.
                    const double term =
                        bezalfs[bez(j, i)]
                        * static_cast<double>(bpts[static_cast<std::size_t>(j * rank + ii)]);
                    ebpts[static_cast<std::size_t>(i * rank + ii)] = static_cast<T>(
                        static_cast<double>(ebpts[static_cast<std::size_t>(i * rank + ii)]) + term);
                }
            }
        }

        if (oldr > 1) {
            std::int64_t first = kind - 2;
            std::int64_t last = kind;
            const T den = ub - ua;
            const T bet = (ub - ik[static_cast<std::size_t>(kind - 1)]) / den;

            for (std::int64_t tr = 1; tr < oldr; ++tr) {
                std::int64_t i = first;
                std::int64_t j = last;
                std::int64_t kj = j - kind + 1;
                while (j - i > tr) {
                    if (i < cind) {
                        const T alf = (ub - ik[static_cast<std::size_t>(i)])
                                      / (ua - ik[static_cast<std::size_t>(i)]);
                        for (std::int64_t ii = 0; ii < rank; ++ii) {
                            // `alf` is `T`, so its product rounds to `T` before the sum;
                            // the companion carries the literal `1.0` and is `double`.
                            const T kept = alf * ic[static_cast<std::size_t>(i * rank + ii)];
                            const double carried =
                                (1.0 - static_cast<double>(alf))
                                * static_cast<double>(ic[static_cast<std::size_t>((i - 1) * rank + ii)]);
                            ic[static_cast<std::size_t>(i * rank + ii)] =
                                static_cast<T>(static_cast<double>(kept) + carried);
                        }
                    }
                    if (j >= lbz) {
                        const T weight = (j - tr <= kind - ph + oldr)
                                             ? (ub - ik[static_cast<std::size_t>(j - tr)]) / den
                                             : bet;
                        for (std::int64_t ii = 0; ii < rank; ++ii) {
                            const T kept = weight * ebpts[static_cast<std::size_t>(kj * rank + ii)];
                            const double carried =
                                (1.0 - static_cast<double>(weight))
                                * static_cast<double>(ebpts[static_cast<std::size_t>((kj + 1) * rank + ii)]);
                            ebpts[static_cast<std::size_t>(kj * rank + ii)] =
                                static_cast<T>(static_cast<double>(kept) + carried);
                        }
                    }
                    ++i;
                    --j;
                    --kj;
                }
                --first;
                ++last;
            }
        }

        if (a != d) {
            for (std::int64_t written = 0; written < ph - oldr; ++written) {
                ik[static_cast<std::size_t>(kind)] = ua;
                ++kind;
            }
        }

        for (std::int64_t j = lbz; j <= rbz; ++j) {
            for (std::int64_t ii = 0; ii < rank; ++ii) {
                ic[static_cast<std::size_t>(cind * rank + ii)] =
                    ebpts[static_cast<std::size_t>(j * rank + ii)];
            }
            ++cind;
        }

        if (b < m) {
            for (std::int64_t j = 0; j < r; ++j) {
                for (std::int64_t ii = 0; ii < rank; ++ii) {
                    bpts[static_cast<std::size_t>(j * rank + ii)] =
                        next_bpts[static_cast<std::size_t>(j * rank + ii)];
                }
            }
            // The oracle's `for j in range(r, d + 1)`, whose `r` is negative at a `C^-1`
            // breakpoint -- `-1` there, and `r = d - mul` cannot go below that for a
            // multiplicity within `degree + 1`. Python indexes `bpts[j]` for a negative
            // `j` as row `d + 1 + j` and reads `ctrl[b - d + j]` as a real row, so each
            // such pass writes a row from a coefficient before the segment; and the
            // matching non-negative `j = d + 1 + j` later in the *same* range overwrites
            // that row, since the range runs to `d` and `j` increases monotonically
            // through it. Every wrapped write is therefore dead whatever `r` is, and
            // skipping it is the same computation rather than an approximation of it.
            // Reproducing the wrap would be reproducing an accident; performing the
            // negative index is undefined behaviour here.
            for (std::int64_t j = std::max<std::int64_t>(0, r); j <= d; ++j) {
                for (std::int64_t ii = 0; ii < rank; ++ii) {
                    bpts[static_cast<std::size_t>(j * rank + ii)] =
                        coefficients[static_cast<std::size_t>((b - d + j) * rank + ii)];
                }
            }
            a = b;
            ++b;
            ua = ub;
        } else {
            for (std::int64_t i = 0; i <= ph; ++i) {
                ik[static_cast<std::size_t>(kind + i)] = ub;
            }
            break;
        }
    }

    return ElevatedAxis<T>{
        std::vector<T>(ic.begin(), ic.begin() + static_cast<std::ptrdiff_t>(cind * rank)),
        std::vector<T>(ik.begin(), ik.begin() + static_cast<std::ptrdiff_t>(kind + ph + 1)), cind};
}

/// Degree-elevate one axis of a row-major array, block by block.
///
/// The array is read as `(outer, num_rows, inner)` and written as
/// `(outer, elevated_rows, inner)`. Each outer block is contiguous and is handed to
/// `degree_elevate_1d` as a curve of `inner` components; see the file comment for why the
/// blocks are independent and why the first one's knot vector is the result's.
///
/// \param degree The direction's polynomial degree.
/// \param knots The direction's knot vector.
/// \param increment Degrees to add, at least 1.
/// \param values The coefficients, row-major under `(outer, num_rows, inner)`.
/// \param num_rows The elevated direction's extent in `values`.
/// \param outer The product of the extents ahead of that axis; 1 when it is the first.
/// \param inner The product of the extents behind it, component axis included.
/// \return The elevated coefficients, knot vector and row count.
/// \throws std::invalid_argument If an extent is negative, if `values.size()` is not
///         `outer * num_rows * inner`, or if `degree_elevate_1d` refuses the block.
template <Real T>
[[nodiscard]] ElevatedAxis<T> elevate_along_axis(std::int64_t degree, std::span<const T> knots,
                                                 std::int64_t increment, std::span<const T> values,
                                                 std::int64_t num_rows, std::int64_t outer,
                                                 std::int64_t inner) {
    if (outer < 1 || inner < 1) {
        throw std::invalid_argument("elevate_along_axis: the extents are counts, so neither of "
                                    "them can be below one.");
    }
    const std::int64_t expected = outer * num_rows * inner;
    if (static_cast<std::int64_t>(values.size()) != expected) {
        throw std::invalid_argument(
            "elevate_along_axis: the buffer holds " + std::to_string(values.size())
            + " coefficients and the shape (" + std::to_string(outer) + ", "
            + std::to_string(num_rows) + ", " + std::to_string(inner) + ") needs "
            + std::to_string(expected) + ".");
    }

    ElevatedAxis<T> result{};
    for (std::int64_t o = 0; o < outer; ++o) {
        const std::span<const T> block =
            values.subspan(static_cast<std::size_t>(o * num_rows * inner),
                           static_cast<std::size_t>(num_rows * inner));
        ElevatedAxis<T> elevated =
            degree_elevate_1d<T>(degree, block, num_rows, inner, knots, increment);
        if (o == 0) {
            result.knots = std::move(elevated.knots);
            result.num_rows = elevated.num_rows;
            result.values.resize(static_cast<std::size_t>(outer * elevated.num_rows * inner));
        }
        std::copy(elevated.values.begin(), elevated.values.end(),
                  result.values.begin()
                      + static_cast<std::ptrdiff_t>(o * result.num_rows * inner));
    }
    return result;
}

/// The field's first partial derivative in one direction, as a B-spline field.
///
/// The hodograph: a field of degree `p - 1` in `direction` and unchanged degree elsewhere,
/// whose value at every parametric point is this field's partial derivative there. The
/// differentiated direction's knot vector loses its first and last entry; every other
/// direction's space handle is carried into the result rather than rebuilt.
///
/// A periodic direction stays periodic. Its coefficients are expanded by their own modulo
/// wrap to the count the knot vector implies, differentiated, and trimmed back to the
/// periodic count the derivative's own knot vector implies -- which is
/// `_derivative_nonrational_nd`'s route and reaches no boundary or periodic conversion.
///
/// \param field The field to differentiate. Must not be rational; see the file comment.
/// \param direction The direction to differentiate, in `[0, field.dim())`.
/// \return The hodograph.
/// \throws std::invalid_argument If `direction` is out of range or its degree is 0, both
///         with the oracle's Layer 1 messages, or if the field is rational.
template <Real T>
[[nodiscard]] Bspline<T> derivative(const Bspline<T>& field, std::int64_t direction) {
    const BsplineSpace<T>& space = field.space_ref();
    const std::int64_t dim = space.dim();
    detail::require_axis("direction", direction, dim);

    const BsplineSpace1D<T>& along = space.space_ref(direction);
    const std::int64_t p = along.degree();
    if (p < 1) {
        throw std::invalid_argument("Derivative of a degree-0 B-spline is not defined.");
    }
    if (field.is_rational()) {
        throw std::invalid_argument(
            "the derivative of a rational B-spline is not part of pantr's C++ core; it "
            "applies the quotient rule, and direction " + std::to_string(direction)
            + " belongs to a rational field.");
    }

    const std::span<const T> knots = along.knots();
    std::vector<std::size_t> shape(field.net().shape().begin(), field.net().shape().end());
    const std::int64_t outer =
        detail::extents_before(std::span<const std::size_t>(shape), direction);
    const std::int64_t inner =
        detail::extents_after(std::span<const std::size_t>(shape), direction);

    std::vector<T> values(field.net().values().begin(), field.net().values().end());
    auto num_rows = static_cast<std::int64_t>(shape[static_cast<std::size_t>(direction)]);

    if (along.periodic()) {
        // The oracle's `ctrl[np.arange(n_full) % n_periodic]`, spelled as the row list
        // `select_rows_along_axis` already exists for.
        const std::int64_t full = static_cast<std::int64_t>(knots.size()) - p - 1;
        std::vector<std::int64_t> rows(static_cast<std::size_t>(full));
        for (std::int64_t i = 0; i < full; ++i) {
            rows[static_cast<std::size_t>(i)] = i % num_rows;
        }
        values = select_rows_along_axis<T>(std::span<const T>(values), num_rows,
                                           std::span<const std::int64_t>(rows), outer, inner);
        num_rows = full;
    }

    values = derivative_along_axis<T>(knots, p, std::span<const T>(values), num_rows, outer, inner);
    std::int64_t out_rows = num_rows - 1;

    // The oracle's `knots[1:-1]`, and the vector the trim's count is read off: the space
    // built from it may snap a near-duplicate, and the count has to come from the same
    // vector the oracle counts on.
    const std::vector<T> derived_knots(knots.begin() + 1, knots.end() - 1);

    if (along.periodic()) {
        const std::int64_t kept = num_basis<T>(std::span<const T>(derived_knots), p - 1, true,
                                               along.tolerance());
        std::vector<std::int64_t> rows(static_cast<std::size_t>(kept));
        for (std::int64_t i = 0; i < kept; ++i) {
            rows[static_cast<std::size_t>(i)] = i;
        }
        values = select_rows_along_axis<T>(std::span<const T>(values), out_rows,
                                           std::span<const std::int64_t>(rows), outer, inner);
        out_rows = kept;
    }

    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions(space.spaces().begin(),
                                                                     space.spaces().end());
    directions[static_cast<std::size_t>(direction)] = std::make_shared<const BsplineSpace1D<T>>(
        std::span<const T>(derived_knots), p - 1, along.periodic(),
        KnotSnapping::merge_near_duplicates);
    shape[static_cast<std::size_t>(direction)] = static_cast<std::size_t>(out_rows);

    return detail::assemble<T>(std::move(directions), values, shape, false);
}

/// The field written at a higher degree, over the same geometry.
///
/// Exact: the elevated field is the same map, written on a richer space. Each direction
/// with a positive increment has its degree raised by it and its knot vector re-emitted
/// with every multiplicity raised by the same amount, which keeps the field's continuity
/// at every breakpoint. A direction whose increment is zero is left alone and its space
/// handle is carried into the result.
///
/// \param field The field to elevate.
/// \param increments Degrees to add per direction, in axis order, `field.dim()` of them.
///        At least one must be positive and none may be negative.
/// \return The elevated field, rational exactly when `field` is.
/// \throws std::invalid_argument If `increments` has the wrong length, if one is negative,
///         if all are zero, if an elevated degree would leave the exact-integer binomial
///         envelope, or if a direction to be elevated is periodic or unclamped.
///
/// \note The first four refusals are the oracle's, in the oracle's order: the argument
///       checks first, then every direction's envelope check before any direction is
///       elevated. The periodic and unclamped refusals are this file's own declared
///       boundaries and are reached only after all of them.
template <Real T>
[[nodiscard]] Bspline<T> elevate_degree(const Bspline<T>& field,
                                        std::span<const std::int64_t> increments) {
    const BsplineSpace<T>& space = field.space_ref();
    const std::int64_t dim = space.dim();
    if (static_cast<std::int64_t>(increments.size()) != dim) {
        throw std::invalid_argument("Number of degree increments ("
                                    + std::to_string(increments.size())
                                    + ") must match dimension (" + std::to_string(dim) + ").");
    }
    for (const std::int64_t increment : increments) {
        if (increment < 0) {
            throw std::invalid_argument("Degree increments must be non-negative.");
        }
    }
    bool any = false;
    for (const std::int64_t increment : increments) {
        any = any || increment > 0;
    }
    if (!any) {
        throw std::invalid_argument("At least one degree increment must be positive.");
    }

    // Every envelope check runs before any elevation, which is the oracle's order: two
    // directions out of range must produce the first one's message on both sides.
    for (std::int64_t d = 0; d < dim; ++d) {
        const std::int64_t increment = increments[static_cast<std::size_t>(d)];
        if (increment > 0) {
            const std::int64_t elevated = space.space_ref(d).degree() + increment;
            core::require_bincoeff_envelope(elevated, "Degree elevation to degree "
                                                          + std::to_string(elevated)
                                                          + " in direction " + std::to_string(d));
        }
    }

    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions(space.spaces().begin(),
                                                                     space.spaces().end());
    std::vector<std::size_t> shape(field.net().shape().begin(), field.net().shape().end());
    std::vector<T> values(field.net().values().begin(), field.net().values().end());

    for (std::int64_t d = 0; d < dim; ++d) {
        const std::int64_t increment = increments[static_cast<std::size_t>(d)];
        if (increment == 0) {
            continue;
        }
        const BsplineSpace1D<T>& along = *directions[static_cast<std::size_t>(d)];
        if (along.periodic()) {
            throw std::invalid_argument(
                "elevating a periodic direction needs the open-to-periodic conversion, which "
                "is not part of pantr's C++ core; direction "
                + std::to_string(d) + " is periodic.");
        }
        if (!along.has_open_knots()) {
            throw std::invalid_argument(
                "elevating a direction whose knot vector is not clamped is not part of pantr's "
                "C++ core: A5.9 walks segments until it reaches the closing run of `degree + 1` "
                "equal knots, and direction "
                + std::to_string(d) + " has none.");
        }
        const std::int64_t degree = along.degree();
        ElevatedAxis<T> elevated = elevate_along_axis<T>(
            degree, along.knots(), increment, std::span<const T>(values),
            static_cast<std::int64_t>(shape[static_cast<std::size_t>(d)]),
            detail::extents_before(std::span<const std::size_t>(shape), d),
            detail::extents_after(std::span<const std::size_t>(shape), d));

        values = std::move(elevated.values);
        shape[static_cast<std::size_t>(d)] = static_cast<std::size_t>(elevated.num_rows);
        directions[static_cast<std::size_t>(d)] = std::make_shared<const BsplineSpace1D<T>>(
            std::span<const T>(elevated.knots), degree + increment, false,
            KnotSnapping::merge_near_duplicates);
    }

    return detail::assemble<T>(std::move(directions), values, shape, field.is_rational());
}

}  // namespace pantr::bspline
