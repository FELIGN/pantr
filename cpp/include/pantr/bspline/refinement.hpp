#pragma once

/// \file
/// Refining a B-spline field: knot insertion and uniform subdivision.
///
/// Ports `pantr.bspline.Bspline.insert_knots` and `.subdivide`, whose substance is
/// `_bspline_knot_insertion.py`'s `_insert_knots_bspline`. The Python side stays as
/// the parity oracle.
///
/// ## What this file adds, and what it only calls
///
/// `pantr/bspline/knot_insertion.hpp` already carries the whole *space* half --
/// `oslo_bands_1d`, `oslo_matrix_1d`, `inserted_knot_vector` and
/// `uniform_subdivision_knots` -- and says of itself that "the control-point half of
/// knot insertion is not here: a space carries no control points". This is that half,
/// and it is deliberately thin: **the only new numerics below is
/// `refine_along_axis`**, which applies bands that file computes. There is no second
/// Oslo recurrence here, no second merge and no second subdivision-point formula. A
/// reader looking for the recurrence should read that file; a reader looking for what
/// a *field* does with it should read this one.
///
/// ## Free functions, not methods
///
/// `pantr/bspline/bspline.hpp` lists knot insertion among the computations *over* a
/// field rather than properties *of* one, and states that each is "a separate port
/// over free functions taking a `const Bspline&`". `pantr/bspline/space_1d.hpp` and
/// `pantr/bspline/space_nd.hpp` draw the same line for a space. So `insert_knots`
/// and `subdivide` take a `const Bspline<T>&` and return a new field; the type has
/// no mutator and this file adds none, which is the `in_place=` decision
/// `bspline.hpp` records in full.
///
/// `subdivide` overloads the one in `knot_insertion.hpp`, which takes a
/// `const BsplineSpace1D<T>&`. The two are unambiguous on their first argument and
/// they *are* the same operation at two levels, so one name is right: a caller
/// refining a space and a caller refining a field over it should not have to
/// remember two spellings.
///
/// ## The floating-point discipline, quantity by quantity
///
/// **The refined knot vectors are `inserted_knot_vector`'s and
/// `uniform_subdivision_knots`', unchanged.** Their own file comment derives why they
/// are bit-identical to the oracle's -- numpy's `linspace` expression reproduced
/// rather than simplified, the arithmetic in `double` whatever `T` is, a stable
/// merge -- and nothing here re-derives or re-rounds them.
///
/// **The control-point sweep accumulates in `T`, not in `double`.** That is
/// `design/backend_parity.md` Rule 9 applied to this kernel rather than inferred from
/// its neighbours: `_insert_knots_1d_core` allocates `result` in `ctrl.dtype`, and its
/// `result[i, r] += weight * ctrl[col, r]` reads three array elements and writes one,
/// so no assignment ever unifies a variable to `float64` and the whole accumulation is
/// `float32` on a `float32` net. Accumulating in `double` here would be exact at
/// `float64` and quietly wrong at `float32`, which is the half of the matrix Rule 9
/// says nobody reads first.
///
/// The band recurrence *does* widen one variable, and it does not matter. Numba
/// unifies `saved`, `to_left` and `to_right` to `float64` because each is seeded with
/// the literal `0.0`; but each then holds the value of a `float32` expression, and the
/// exact sum of two `float32` numbers is representable in `float64`, so
/// `fl32(saved + to_left)` and `fl64(saved + to_left)` narrowed on the store are the
/// same number. There is no double-rounding hazard to inherit, which is why
/// `oslo_bands_1d` computing entirely in `T` matches.
///
/// **The summation order per output coefficient is the oracle's**: ascending `l` over
/// the band, terms whose column falls outside `[0, num_cols)` skipped rather than
/// added as zero, starting from an exact `T(0)`. Nothing else about the loop nest is
/// load-bearing, and that is a statement worth making precisely: output coefficient
/// `(i, r)` accumulates over `l` *alone*, so every other loop -- over the outer block,
/// over the component, over the direction's own index -- is a loop over independent
/// destinations. Permuting them, or vectorising the innermost one, changes no
/// operation and no operand.
///
/// That is also why this file sweeps the net **strided in place of the oracle's
/// transpose**. `_insert_knots_bspline` calls `_flatten_along_axis`, which is
/// `np.moveaxis` plus `np.ascontiguousarray`: a permutation of values, not an
/// arithmetic step. Reproducing it would cost two full copies of the geometry per
/// direction and could not change a bit, so `refine_along_axis` takes the axis's
/// `outer` and `inner` extents instead.
///
/// Under this build -- `-ffp-contract=on`, no `-march`, so baseline x86-64 with no FMA
/// to fuse into, `cpp/README.md` -- the two backends therefore execute the same
/// IEEE-754 operations in the same order and every coefficient is bit-identical. That
/// is a property of the *host* rather than of the code, which is
/// `design/backend_parity.md` Rule 7: a target with an FMA fuses
/// `row[k] + weight * source[k]` and the claim becomes Rule 10's budget.
/// `pantr._pantr_cpp.__fp_contract__` is the gate that tells them apart.
///
/// ## The one operation of the oracle's this file does not reproduce
///
/// **A periodic direction is refused, and that is a declared boundary rather than a
/// defect.** `_insert_knots_bspline` refines a periodic direction by a round trip
/// through the open representation: `_to_open_bspline_1d_impl`, insert, then
/// `_to_periodic_bspline_1d_impl`. Those two are the *boundary and periodic
/// conversions*, which `pantr/bspline/bspline.hpp` lists as their own port and which
/// no ticket in this milestone covers -- exactly as `space_1d.hpp` treats
/// `get_cardinal_intervals`. Porting them here would be porting two more operations
/// under cover of this one.
///
/// So `refine_along_axis` is dimension-agnostic and periodicity-agnostic, and the two
/// entry points refuse a periodic direction that would receive knots. A periodic
/// direction that receives *none* is untouched: its space handle is carried over, the
/// same way the oracle carries over its wrapper.
/// `pantr.bspline._knot_insertion_backend` routes such a field to the oracle rather
/// than meeting this refusal.
///
/// ## Validating rather than asserting
///
/// This is the C++ counterpart of Layer 2, so it validates and throws in a release
/// build as much as in a debug one, with the oracle's messages character for
/// character. `pantr/core/error.hpp` sets the split: value and range checks here,
/// type-kind checks in the Python wrapper.
///
/// Three of the refusals below are the oracle's **Layer 1** checks -- the
/// per-direction argument count, "at least one direction", and the per-direction
/// regularity range. They are near-vacuous on the Python path, where
/// `pantr.bspline.Bspline` has already made them, and load-bearing for a C++ caller
/// with no wrapper in front of it. That is the same reason `bspline.hpp` re-checks the
/// coefficient count its own wrapper checked.
///
/// One refusal of the oracle's cannot live here and stays the wrapper's: the rank of
/// the insertion array. `f"new_knots must be a 1D array-like, got shape ..."` quotes a
/// numpy shape, and a `std::span` has no rank to check.
///
/// ## Thread safety
///
/// Both entry points read their argument and allocate their result, and every type
/// they touch is immutable. They are safe to call concurrently on the same field with
/// no external locking, which is the contract `design/bspline_derived_caches.md`
/// states for this package.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/bezier/control_net.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// Apply a two-scale refinement to one axis of a row-major array.
///
/// The array is read as `(outer, num_cols, inner)` and written as
/// `(outer, bands.num_rows, inner)`: for a control net of shape
/// `(e_0, ..., e_{d-1}, num_components)` refined along direction `d`, `outer` is the
/// product of the extents before `d` and `inner` the product of those after it, the
/// component axis included.
///
/// Output coefficient `(o, i, k)` is `sum_l alphas(i, l) * values(o, first_col(i) + l,
/// k)` over the columns of the band that fall inside `[0, num_cols)`, accumulated in
/// `T` in ascending `l` from an exact zero. Off-band columns are skipped rather than
/// added as zeros; a dense sweep computes exact zeros there, so the two agree entry
/// for entry, and skipping is what makes one output coefficient cost
/// `O(bands.width)` instead of `O(num_cols)`.
///
/// A row whose `first_col` is negative -- which happens on a non-clamped vector -- has
/// leading band entries that address no column, and they are the ones skipped.
///
/// \param bands The refinement matrix in banded form, from `oslo_bands_1d`.
/// \param num_cols The refined direction's extent in `values`, which is the coarse
///        space's basis count.
/// \param values The coefficients, row-major under `(outer, num_cols, inner)`.
/// \param outer The product of the extents ahead of the refined axis; 1 when it is
///        the first.
/// \param inner The product of the extents behind it, component axis included.
/// \return `outer * bands.num_rows * inner` coefficients, row-major.
/// \throws std::invalid_argument If any extent is negative, or if `values.size()` is
///         not `outer * num_cols * inner`.
///
/// \note The accumulation is in `T` and in the oracle's order; see the file comment
///       for why that is a per-kernel fact rather than a module convention.
template <Real T>
[[nodiscard]] std::vector<T> refine_along_axis(const OsloBands<T>& bands, std::int64_t num_cols,
                                               std::span<const T> values, std::int64_t outer,
                                               std::int64_t inner) {
    if (num_cols < 0 || outer < 0 || inner < 0) {
        throw std::invalid_argument("refine_along_axis: the extents are counts, so none of "
                                    "them can be negative.");
    }
    const std::int64_t expected = outer * num_cols * inner;
    if (static_cast<std::int64_t>(values.size()) != expected) {
        throw std::invalid_argument(
            "refine_along_axis: the buffer holds " + std::to_string(values.size())
            + " coefficients and the shape (" + std::to_string(outer) + ", "
            + std::to_string(num_cols) + ", " + std::to_string(inner) + ") needs "
            + std::to_string(expected) + ".");
    }

    const std::int64_t num_rows = bands.num_rows;
    std::vector<T> out(static_cast<std::size_t>(outer * num_rows * inner), T(0));

    for (std::int64_t o = 0; o < outer; ++o) {
        const T* const in_block = values.data() + o * num_cols * inner;
        T* const out_block = out.data() + o * num_rows * inner;
        for (std::int64_t i = 0; i < num_rows; ++i) {
            T* const row = out_block + i * inner;
            const std::int64_t base = bands.first_col[static_cast<std::size_t>(i)];
            for (std::int64_t l = 0; l < bands.width; ++l) {
                const std::int64_t col = base + l;
                if (col < 0 || col >= num_cols) {
                    continue;
                }
                const T weight = bands.alphas[static_cast<std::size_t>(i * bands.width + l)];
                const T* const source = in_block + col * inner;
                for (std::int64_t k = 0; k < inner; ++k) {
                    // The oracle's `result[i, r] += weight * ctrl[col, r]`, operand for
                    // operand. `row[k]` starts at an exact zero, so the first term is
                    // added to zero exactly as it is there.
                    row[k] = row[k] + weight * source[k];
                }
            }
        }
    }
    return out;
}

namespace detail {

/// Refine every direction that was given knots, one after another.
///
/// The shared body of `insert_knots` and `subdivide`: each of those validates its own
/// arguments and reduces them to a per-direction list of values to insert, and this
/// walks the directions in axis order, refining the space and the net together.
///
/// A direction whose list is empty is skipped, and its space handle is carried into
/// the result rather than rebuilt -- which is what the oracle does with its wrapper,
/// and what lets the Python side keep `refined.space.spaces[d] is field.space.spaces[d]`
/// for an untouched direction. When every list is empty the field itself is returned.
///
/// \tparam T The scalar type the field stores.
/// \param field The field to refine.
/// \param per_direction One list of values to insert per direction, in axis order.
///        Repeats raise a knot's multiplicity by that many.
/// \return The refined field.
/// \throws std::invalid_argument If `per_direction` does not have one entry per
///         direction, if a direction with a non-empty list is periodic, or if
///         `inserted_knot_vector` refuses a list.
template <Real T>
[[nodiscard]] Bspline<T> refine_field(const Bspline<T>& field,
                                      std::span<const std::span<const T>> per_direction) {
    const BsplineSpace<T>& space = field.space_ref();
    const std::int64_t dim = space.dim();
    if (static_cast<std::int64_t>(per_direction.size()) != dim) {
        throw std::invalid_argument("one list of knots per direction is needed; got "
                                    + std::to_string(per_direction.size()) + " for a field of "
                                    + std::to_string(dim) + " direction(s).");
    }

    bool refined_anything = false;
    for (const std::span<const T> to_insert : per_direction) {
        refined_anything = refined_anything || !to_insert.empty();
    }
    if (!refined_anything) {
        return field;
    }

    // The handles start as the field's own and are replaced only where knots go in,
    // so an untouched direction is shared rather than copied.
    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions(space.spaces().begin(),
                                                                     space.spaces().end());
    std::vector<std::size_t> shape(field.net().shape().begin(), field.net().shape().end());
    std::vector<T> values(field.net().values().begin(), field.net().values().end());

    for (std::int64_t d = 0; d < dim; ++d) {
        const std::span<const T> to_insert = per_direction[static_cast<std::size_t>(d)];
        if (to_insert.empty()) {
            continue;
        }
        const BsplineSpace1D<T>& coarse = *directions[static_cast<std::size_t>(d)];
        if (coarse.periodic()) {
            throw std::invalid_argument(
                "refining a periodic direction needs the open-to-periodic conversion, which "
                "is not part of pantr's C++ core; direction "
                + std::to_string(d) + " is periodic.");
        }
        const std::int64_t degree = coarse.degree();

        // The merged vector, not the refined space's stored knots. Snapping may move a
        // knot on the way into `BsplineSpace1D`, and the oracle applies the recurrence
        // to the merged vector and *then* builds the space from it; applying it to the
        // snapped vector instead would be a different matrix.
        const std::vector<T> merged =
            inserted_knot_vector<T>(coarse.knots(), degree, to_insert, coarse.tolerance());

        const std::int64_t num_cols = static_cast<std::int64_t>(coarse.num_basis());
        const OsloBands<T> bands =
            oslo_bands_1d<T>(degree, coarse.knots(), std::span<const T>(merged));

        std::int64_t outer = 1;
        for (std::int64_t before = 0; before < d; ++before) {
            outer *= static_cast<std::int64_t>(shape[static_cast<std::size_t>(before)]);
        }
        std::int64_t inner = 1;
        for (std::size_t after = static_cast<std::size_t>(d) + 1; after < shape.size(); ++after) {
            inner *= static_cast<std::int64_t>(shape[after]);
        }

        values = refine_along_axis<T>(bands, num_cols, std::span<const T>(values), outer, inner);
        shape[static_cast<std::size_t>(d)] = static_cast<std::size_t>(bands.num_rows);
        directions[static_cast<std::size_t>(d)] = std::make_shared<const BsplineSpace1D<T>>(
            std::span<const T>(merged), degree, false, KnotSnapping::merge_near_duplicates);
    }

    return Bspline<T>(std::make_shared<const BsplineSpace<T>>(std::move(directions)),
                      typename Bspline<T>::net_type(std::span<const T>(values),
                                                    std::span<const std::size_t>(shape)),
                      field.is_rational());
}

}  // namespace detail

/// The field with knots inserted, over the same geometry.
///
/// Each direction's knots are merged into its knot vector and the control net is
/// pushed through the two-scale matrix, so the result is the same map over a refined
/// space: exactly, up to the rounding the file comment bounds. A direction given no
/// knots is left alone.
///
/// \param field The field to refine.
/// \param new_knots_per_direction One list of knot values per direction, in axis
///        order. Repeats raise that knot's multiplicity by as many. An empty list
///        skips its direction; at least one must be non-empty.
/// \return The refined field.
/// \throws std::invalid_argument If `new_knots_per_direction` does not have one entry
///         per direction, if every entry is empty, if a direction receiving knots is
///         periodic, if a value lies outside its direction's domain by more than the
///         space's tolerance, or if a merge would push a knot's multiplicity above
///         `degree + 1`.
///
/// \note The domain and multiplicity refusals are `inserted_knot_vector`'s, so their
///       messages are the oracle's; see that function.
template <Real T>
[[nodiscard]] Bspline<T> insert_knots(const Bspline<T>& field,
                                      std::span<const std::span<const T>> new_knots_per_direction) {
    const std::int64_t dim = field.dim();
    if (static_cast<std::int64_t>(new_knots_per_direction.size()) != dim) {
        // The oracle's Layer 1 message, which the wrapper has already raised on the
        // Python path; see the file comment on why it is restated.
        throw std::invalid_argument("new_knots sequence length ("
                                    + std::to_string(new_knots_per_direction.size())
                                    + ") must match dim (" + std::to_string(dim) + ").");
    }
    bool any = false;
    for (const std::span<const T> to_insert : new_knots_per_direction) {
        any = any || !to_insert.empty();
    }
    if (!any) {
        throw std::invalid_argument(
            "At least one direction must have a non-empty array of knots to insert.");
    }
    return detail::refine_field<T>(field, new_knots_per_direction);
}

/// The field with every knot span of the named directions split into equal sub-spans.
///
/// For each in-domain span of direction `d`, `n_subdivisions[d] - 1` equally spaced
/// interior values, each repeated `degree - regularity` times so that the refined
/// field is `C^regularity` at every inserted knot. The geometry does not move: the
/// control net is pushed through the two-scale matrix exactly as `insert_knots` does.
///
/// \param field The field to refine.
/// \param n_subdivisions Equal sub-spans per existing span, one per direction. A
///        direction whose count is 1 is skipped rather than subdivided by one, which
///        is the oracle's reading of `None`; at least one count must be at least 2.
/// \param regularity The continuity at each inserted knot, applied to every
///        subdivided direction. Empty for `degree - 1` per direction, the maximal
///        smoothness each degree admits.
/// \return The refined field.
/// \throws std::invalid_argument If `n_subdivisions` does not have one entry per
///         direction, if any count is below 1, if no count reaches 2, if `regularity`
///         is outside `[-1, degree - 1]` for a subdivided direction, if such a
///         direction is periodic, or if the merge would exceed the maximum
///         multiplicity.
///
/// \note **The multiplicity refusal is all but unreachable from here**, which is worth
///       saying because a comment in `cpp/tests/test_bspline_knot_insertion.cpp` says
///       the opposite. `regularity = -1` asks for multiplicity `degree + 1` at each
///       inserted knot, which is exactly the ceiling and not above it, and the points
///       `uniform_subdivision_knots` produces are strictly interior to their own span,
///       so they collide with no existing knot and with no other span's. The one way
///       left is a span so narrow that an interior point lands within the space's
///       tolerance of an endpoint. `cpp/tests/test_bspline_refinement.cpp` pins the
///       positive half of that: `regularity = -1` succeeds and lands on `degree + 1`
///       exactly.
template <Real T>
[[nodiscard]] Bspline<T> subdivide(const Bspline<T>& field,
                                   std::span<const std::int64_t> n_subdivisions,
                                   std::optional<std::int64_t> regularity) {
    const BsplineSpace<T>& space = field.space_ref();
    const std::int64_t dim = space.dim();
    if (static_cast<std::int64_t>(n_subdivisions.size()) != dim) {
        throw std::invalid_argument("n_subdivisions sequence length ("
                                    + std::to_string(n_subdivisions.size())
                                    + ") must match dim (" + std::to_string(dim) + ").");
    }
    // The oracle checks every count before looking at any regularity, so a field that
    // is bad in both ways reports the count. Only the order decides which message a
    // caller reads.
    for (const std::int64_t count : n_subdivisions) {
        if (count < 1) {
            throw std::invalid_argument("n_subdivisions must be >= 1, got "
                                        + std::to_string(count));
        }
    }
    bool any = false;
    for (const std::int64_t count : n_subdivisions) {
        any = any || count >= 2;
    }
    if (!any) {
        throw std::invalid_argument("At least one direction must have n_subdivisions >= 2.");
    }

    std::vector<std::vector<T>> owned(static_cast<std::size_t>(dim));
    std::vector<std::span<const T>> per_direction(static_cast<std::size_t>(dim));
    for (std::int64_t d = 0; d < dim; ++d) {
        const std::int64_t count = n_subdivisions[static_cast<std::size_t>(d)];
        if (count == 1) {
            continue;
        }
        const BsplineSpace1D<T>& direction = space.space_ref(d);
        const std::int64_t degree = direction.degree();
        const std::int64_t effective = regularity.value_or(degree - 1);
        if (effective < -1 || effective > degree - 1) {
            // The oracle's message names the direction, which the space-level
            // `subdivide` in knot_insertion.hpp does not: there is only one direction
            // there to blame.
            throw std::invalid_argument("regularity must be in [-1, degree - 1] = [-1, "
                                        + std::to_string(degree - 1) + "] for direction "
                                        + std::to_string(d) + ", got "
                                        + std::to_string(effective));
        }
        owned[static_cast<std::size_t>(d)] = uniform_subdivision_knots<T>(
            direction.knots(), degree, direction.tolerance(), count, effective);
        per_direction[static_cast<std::size_t>(d)] =
            std::span<const T>(owned[static_cast<std::size_t>(d)]);
    }

    return detail::refine_field<T>(field, std::span<const std::span<const T>>(per_direction));
}

}  // namespace pantr::bspline
