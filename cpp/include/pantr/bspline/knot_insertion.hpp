#pragma once

/// \file
/// Knot insertion and the two-scale (Oslo) refinement matrix.
///
/// Ports `src/pantr/bspline/_bspline_knot_insertion_core.py` and the two Layer 2
/// helpers a *space* needs from `_bspline_knot_insertion.py`, which stay as the
/// parity oracle. The control-point half of knot insertion is not here: a space
/// carries no control points, and `Bspline` is FELIGN/pantr#398.
///
/// ## Why this file exists ahead of its own ticket
///
/// `THBSplineSpace`'s *state* is defined by two operations this tree did not have.
/// Its per-level spaces are the root space subdivided (`_thb_spline_space.py`'s
/// `_build_level_spaces`), and its truncated coefficients are built by pushing a
/// function through the two-scale matrix level by level (`_compute_truncated_coeffs`).
/// FELIGN/pantr#396 moved only *state and what it determines* and left every
/// operation in Python, which was right for a tensor-product space; a THB space is
/// the first type whose state is the output of an operation, so the operation has to
/// come with it. `design/cross_backend_types.md` forbids the alternative of computing
/// the level spaces in Python and handing them to a C++ constructor -- that is the
/// conversion function the ownership rule exists to prevent.
///
/// ## Free functions, not methods
///
/// `pantr/bspline/space_1d.hpp` keeps operations off the type on purpose, and
/// `pantr/bspline/space_nd.hpp` restates the line: a space owns its value and the
/// quantities that value fixes, and nothing else. Subdivision *constructs a new
/// space* from an old one, so it is a computation over a space rather than a
/// property of one, and it lives here as a free function taking a
/// `const BsplineSpace1D<T>&`.
///
/// ## The floating-point discipline, quantity by quantity
///
/// **The Oslo recurrence is transcribed operation for operation**, in the oracle's
/// own order: the quotient is formed first and multiplied by the carried band value
/// second (`(t[j+k] - x) / denom * band[l]`), the guard is `denom > 0` and not a
/// tolerance, and the two terms sharing a denominator are computed from one test.
/// Writing it as `band[l] * (t[j+k] - x) / denom` is the same real number and a
/// different floating-point one, and `design/backend_parity.md` Rule 10 is what makes
/// that the difference between a bitwise claim and a bound. Under this build --
/// `-ffp-contract=on`, no `-march`, so baseline x86-64 with no FMA to fuse into,
/// `cpp/README.md` -- the two backends execute the same IEEE-754 operations in the
/// same order and the result is bit-identical. That is a property of the *host*, not
/// of the code, which is Rule 7; a target with FMA fuses `saved + to_left` and the
/// claim becomes a bound. `pantr._pantr_cpp.__fp_contract__` is the gate that tells
/// them apart.
///
/// **The subdivision points are computed in `double` whatever the knots are stored
/// in**, and cast once at the end. That is not a widening for accuracy: the oracle
/// forms them with `float(unique[k])` and `numpy.linspace(..., dtype=float64)` and
/// casts the result to the knot dtype, so `double` *is* the arithmetic on both sides
/// and computing a `float` space's new knots in `float` would be the divergence.
/// `design/cross_backend_types.md` states the same rule for `pantr.quad`: compute in
/// `double`, template only the store type.
///
/// **`linspace`'s own expression is reproduced rather than simplified.** numpy forms
/// `j * step + start` with `step = (hi - lo) / n`, so that is what this computes.
/// `lo + j * (hi - lo) / n` and `lo + (hi - lo) * j / n` are both better-looking and
/// both differ in the last bit.
///
/// **The merge is a stable sort.** `numpy.sort`'s default is introsort, which is not
/// stable, but the input is a concatenation of two exact arrays and equal `double`s
/// are indistinguishable -- except for a `-0.0` next to a `+0.0`, where the two
/// differ in a bit pattern that compares equal. `std::stable_sort` keeps the
/// concatenation order for that pair, which is the order `numpy.sort` also happens to
/// produce here, and `CLAUDE.md` records that the sign of such a tie is exactly what
/// must never be asserted. Stability costs nothing at these sizes and removes the
/// question.
///
/// ## Validation
///
/// This is the C++ counterpart of Layer 2, so it validates and throws in a release
/// build as much as in a debug one, with the oracle's messages. `pantr/core/error.hpp`
/// sets the split: value and range checks here, type-kind checks in the Python
/// wrapper.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// The two-scale refinement matrix, in the banded form the recurrence produces.
///
/// Row `i` of the refinement matrix is a discrete B-spline supported on the columns
/// `[first_col[i], first_col[i] + degree]`, so `num_rows * (degree + 1)` values
/// describe a matrix with `num_rows * (n + 1)` entries. `oslo_matrix_1d` scatters
/// this into the dense form; nothing else needs the dense one.
///
/// \tparam T The scalar type the knots are stored in.
template <Real T>
struct OsloBands {
    /// `num_rows * (degree + 1)` coefficients, row-major. Entry `(i, l)` multiplies
    /// old control point `first_col[i] + l`.
    std::vector<T> alphas;

    /// `num_rows` column offsets, one per row.
    ///
    /// Negative when the span sits inside the first `degree` knots, which happens on
    /// a non-clamped vector. The leading entries of such a row are meaningless and a
    /// consumer must skip columns outside `[0, num_cols)`; `oslo_matrix_1d` does.
    std::vector<std::int64_t> first_col;

    /// The number of rows, which is the refined space's basis count.
    std::int64_t num_rows = 0;

    /// The band width, `degree + 1`.
    std::int64_t width = 0;
};

/// Build the two-scale refinement matrix one banded row at a time.
///
/// Row `i` holds the discrete B-splines `alpha_{j,p+1}(i)`, which vanish outside
/// `j` in `[mu(i) - degree, mu(i)]` where `mu(i)` is the old-knot span containing
/// `new_knots[i]`. Only that window is computed, by the recurrence of Cohen, Lyche
/// and Riesenfeld (1980) run for `k = 1 .. degree` from the seed `alpha_mu = 1`; see
/// the oracle's docstring for the recurrence in full. A term whose denominator
/// vanishes is dropped, and the two terms consuming `alpha_j^{(k)}` share that
/// denominator, so one test settles both -- which is what lets a level run in place
/// with a single carried value.
///
/// Costs `O(num_rows * degree^2)` against the `O(num_rows * num_cols)` of a dense
/// sweep, and reproduces it entry for entry: outside the band the dense sweep
/// computes exact zeros.
///
/// \param degree The polynomial degree, non-negative.
/// \param old_knots The original knot vector, `n + degree + 2` entries.
/// \param new_knots The refined knot vector, `m + degree + 2` entries. Must be a
///        superset of `old_knots`; this is not checked.
/// \return The bands, with `num_rows = m + 1`.
/// \throws std::invalid_argument If the degree is negative or either vector is too
///         short to describe a space of that degree.
///
/// \note No check that `new_knots` refines `old_knots` is performed; a caller that
///       has not established it gets a matrix that does not reproduce the geometry.
///       `inserted_knot_vector` is how the guarantee is obtained.
template <Real T>
[[nodiscard]] OsloBands<T> oslo_bands_1d(std::int64_t degree, std::span<const T> old_knots,
                                         std::span<const T> new_knots) {
    if (degree < 0) {
        throw std::invalid_argument("degree must be non-negative; got "
                                    + std::to_string(degree) + ".");
    }
    const std::int64_t need = degree + 2;
    if (static_cast<std::int64_t>(old_knots.size()) < need
        || static_cast<std::int64_t>(new_knots.size()) < need) {
        throw std::invalid_argument("a knot vector of degree " + std::to_string(degree)
                                    + " needs at least " + std::to_string(need)
                                    + " entries to describe one basis function.");
    }

    const std::int64_t p = degree;
    // The oracle's `n` and `m`: the LAST index of the old and new control points,
    // not their counts. Kept in that spelling so the recurrence below reads against
    // the source it was transcribed from.
    const std::int64_t n = static_cast<std::int64_t>(old_knots.size()) - p - 2;
    const std::int64_t m = static_cast<std::int64_t>(new_knots.size()) - p - 2;

    OsloBands<T> out;
    out.num_rows = m + 1;
    out.width = p + 1;
    out.alphas.assign(static_cast<std::size_t>(out.num_rows * out.width), T(0));
    out.first_col.resize(static_cast<std::size_t>(out.num_rows));

    // The two vectors are read through pointers rather than through `operator[]`.
    // The recurrence indexes them with signed offsets throughout -- `j + k`, `mu - p`
    // -- and `std::span::operator[]` takes a `size_type`, so every one of those
    // becomes a `static_cast<std::size_t>` under `-Wsign-conversion` and the
    // transcription stops being readable against the source it came from.
    const T* const t = old_knots.data();
    const T* const tau = new_knots.data();

    for (std::int64_t i = 0; i <= m; ++i) {
        // The span of the old knot vector containing `new_knots[i]`. The right
        // endpoint lands past the last span and is clamped back onto it.
        const auto upper = std::upper_bound(old_knots.begin(), old_knots.end(), tau[i]);
        std::int64_t mu = static_cast<std::int64_t>(upper - old_knots.begin()) - 1;
        mu = std::min(mu, n);
        mu = std::max(mu, std::int64_t{0});
        out.first_col[static_cast<std::size_t>(i)] = mu - p;

        T* band = out.alphas.data() + i * out.width;
        band[p] = T(1);

        for (std::int64_t k = 1; k <= p; ++k) {
            const T x = tau[i + k];
            T saved = T(0);
            // The band grows one entry to the left per level; entries left of column
            // 0 only ever feed columns further left, so they are skipped.
            const std::int64_t l_start = std::max(p - k + 1, p - mu);
            for (std::int64_t l = l_start; l <= p; ++l) {
                const std::int64_t j = mu - p + l;
                const T denom = t[j + k] - t[j];
                T to_left = T(0);
                T to_right = T(0);
                if (denom > T(0)) {
                    // Quotient first, then the carried value: the oracle's order, and
                    // the file comment says why reordering it costs the bitwise claim.
                    to_left = (t[j + k] - x) / denom * band[l];
                    to_right = (x - t[j]) / denom * band[l];
                }
                band[l - 1] = saved + to_left;
                saved = to_right;
            }
            band[p] = saved;
        }
    }
    return out;
}

/// Build the dense two-scale refinement matrix.
///
/// The matrix `alpha` of shape `(m + 1, n + 1)`, row-major, such that new control
/// points `Q = alpha @ P` reproduce the original geometry exactly, and such that an
/// old basis function `B_i` equals `sum_j alpha[j, i] B_j` in the refined basis.
/// Assembled by scattering `oslo_bands_1d`'s rows; every entry outside a band is an
/// exact zero.
///
/// \param degree The polynomial degree, non-negative.
/// \param old_knots The original knot vector, `n + degree + 2` entries.
/// \param new_knots The refined knot vector, `m + degree + 2` entries.
/// \return `(m + 1) * (n + 1)` values in row-major order.
/// \throws std::invalid_argument If `oslo_bands_1d` refuses its arguments.
template <Real T>
[[nodiscard]] std::vector<T> oslo_matrix_1d(std::int64_t degree, std::span<const T> old_knots,
                                            std::span<const T> new_knots) {
    const OsloBands<T> bands = oslo_bands_1d<T>(degree, old_knots, new_knots);
    const std::int64_t num_cols = static_cast<std::int64_t>(old_knots.size()) - degree - 1;

    std::vector<T> dense(static_cast<std::size_t>(bands.num_rows)
                             * static_cast<std::size_t>(num_cols),
                         T(0));
    for (std::int64_t i = 0; i < bands.num_rows; ++i) {
        const std::int64_t base = bands.first_col[static_cast<std::size_t>(i)];
        for (std::int64_t l = 0; l < bands.width; ++l) {
            const std::int64_t col = base + l;
            if (col >= 0 && col < num_cols) {
                dense[static_cast<std::size_t>(i * num_cols + col)] =
                    bands.alphas[static_cast<std::size_t>(i * bands.width + l)];
            }
        }
    }
    return dense;
}

/// The knots that subdivide every non-empty span into `n_subdivisions` equal parts.
///
/// For each in-domain span `[u_k, u_{k+1})`, `n_subdivisions - 1` equally spaced
/// interior values, each repeated `degree - regularity` times so the refined space is
/// `C^regularity` at every inserted knot.
///
/// The lower bound on `n_subdivisions` is the oracle's, 2. A direction whose refinement
/// factor is 1 is not subdivided at all and its caller skips it, which is what
/// `THBSplineSpace`'s `build_level_spaces` does; an earlier version of this file
/// accepted 1 and returned nothing, nothing exercised it, and a widened contract with no
/// exerciser is how an unexamined precedent starts.
///
/// \param knots The knot vector to subdivide.
/// \param degree The polynomial degree.
/// \param tol The tolerance that decides which knots are the same knot; the space's
///        own `tolerance()`.
/// \param n_subdivisions Equal sub-spans per existing span, at least 2.
/// \param regularity The continuity at each inserted knot, in `[-1, degree - 1]`.
/// \return The values to insert, non-decreasing, in `T`.
/// \throws std::invalid_argument If `n_subdivisions < 2` or `regularity` is out of
///         range.
///
/// \note The arithmetic is `double` whatever `T` is, and the result is cast once at
///       the end; see the file comment.
template <Real T>
[[nodiscard]] std::vector<T> uniform_subdivision_knots(std::span<const T> knots,
                                                       std::int64_t degree, double tol,
                                                       std::int64_t n_subdivisions,
                                                       std::int64_t regularity) {
    if (n_subdivisions < 2) {
        throw std::invalid_argument("n_subdivisions must be >= 2, got "
                                    + std::to_string(n_subdivisions));
    }
    if (regularity < -1 || regularity > degree - 1) {
        throw std::invalid_argument("regularity must be in [-1, degree - 1] = [-1, "
                                    + std::to_string(degree - 1) + "], got "
                                    + std::to_string(regularity));
    }

    std::vector<T> out;
    const KnotClasses<T> classes = unique_knots_and_multiplicity<T>(knots, degree, tol);
    const std::size_t begin = classes.domain_begin;
    const std::size_t end = classes.domain_end;
    if (end <= begin + 1) {
        return out;
    }

    const std::int64_t repeat = degree - regularity;
    const std::int64_t per_span = n_subdivisions - 1;
    out.reserve(static_cast<std::size_t>(static_cast<std::int64_t>(end - begin - 1) * per_span
                                         * repeat));

    for (std::size_t k = begin; k + 1 < end; ++k) {
        const double lo = static_cast<double>(classes.unique[k]);
        const double hi = static_cast<double>(classes.unique[k + 1]);
        // numpy's own expression, not a tidier equivalent; see the file comment.
        const double step = (hi - lo) / static_cast<double>(n_subdivisions);
        for (std::int64_t j = 1; j <= per_span; ++j) {
            const double value = static_cast<double>(j) * step + lo;
            for (std::int64_t r = 0; r < repeat; ++r) {
                out.push_back(static_cast<T>(value));
            }
        }
    }
    return out;
}

/// Merge knots into a knot vector, refusing the insertions a space cannot take.
///
/// \param knots The original knot vector.
/// \param degree The polynomial degree.
/// \param to_insert The values to insert; repeats raise the multiplicity by that
///        many. Must not be empty.
/// \param tol The tolerance that decides which knots are the same knot.
/// \return The merged vector, non-decreasing.
/// \throws std::invalid_argument If `to_insert` is empty, any value lies outside the
///         domain `[knots[degree], knots[size - degree - 1]]` by more than `tol`, or
///         the merge would give some knot a multiplicity above `degree + 1`.
template <Real T>
[[nodiscard]] std::vector<T> inserted_knot_vector(std::span<const T> knots, std::int64_t degree,
                                                  std::span<const T> to_insert, double tol) {
    if (to_insert.empty()) {
        throw std::invalid_argument("new_knots_to_insert must not be empty.");
    }

    // The domain check forms the difference and compares it against `tol`, rather
    // than shifting the bound: `value <= hi + tol` reads more directly and rounds the
    // sum, so once `tol` drops below `ulp(hi)` the effective tolerance becomes
    // `ulp(hi)`. That is the oracle's own argument, transcribed with its predicate --
    // strict inequality first, then the absolute difference -- rather than with the
    // one-sided equivalent, so that a reader can put the two side by side.
    const T lo = knots[static_cast<std::size_t>(degree)];
    const T hi = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
    // Unqualified, with a using-declaration, as `pantr/core/scalar.hpp` requires:
    // `std::abs(x)` names the overload directly and suppresses ADL, so a Tier B
    // scalar could never supply its own. `T` here is the template parameter.
    using std::abs;
    std::vector<T> outside;
    for (const T value : to_insert) {
        const bool above_lo = lo < value || static_cast<double>(abs(lo - value)) <= tol;
        const bool below_hi = value < hi || static_cast<double>(abs(value - hi)) <= tol;
        if (!(above_lo && below_hi)) {
            outside.push_back(value);
        }
    }
    if (!outside.empty()) {
        // The list is rendered space-separated in brackets, which is `numpy`'s shape
        // but not necessarily its spelling: numpy chooses a shared precision across
        // the elements and pads them, and reproducing that is a formatting port
        // nobody needs. A parity test on this message compares the prefix through
        // the closing `]:` and the element *values*, not the list's rendering.
        std::string listed = "[";
        for (std::size_t i = 0; i < outside.size(); ++i) {
            if (i != 0) {
                listed += ' ';
            }
            listed += pantr::detail::format_scalar(outside[i]);
        }
        listed += ']';
        throw std::invalid_argument("new_knots contains values outside the domain ["
                                    + pantr::detail::format_scalar(lo) + ", "
                                    + pantr::detail::format_scalar(hi) + "]: " + listed);
    }

    std::vector<T> merged;
    merged.reserve(knots.size() + to_insert.size());
    merged.insert(merged.end(), knots.begin(), knots.end());
    merged.insert(merged.end(), to_insert.begin(), to_insert.end());
    // Stable, so a `-0.0` and a `+0.0` keep the concatenation's order rather than an
    // order nobody chose; see the file comment.
    std::stable_sort(merged.begin(), merged.end());

    const std::int64_t max_allowed = degree + 1;
    const KnotClasses<T> classes = unique_knots_and_multiplicity<T>(merged, degree, tol);
    std::int64_t worst = 0;
    for (const std::int64_t mult : classes.multiplicity) {
        worst = std::max(worst, mult);
    }
    if (worst > max_allowed) {
        throw std::invalid_argument(
            "Inserting these knots would exceed the maximum multiplicity of "
            + std::to_string(max_allowed)
            + ". Maximum multiplicity found: " + std::to_string(worst) + ".");
    }
    return merged;
}

/// The space with every knot span split into `n_subdivisions` equal sub-spans.
///
/// The refined space is nested in `space`: every function of `space` is a linear
/// combination of the result's, with `oslo_matrix_1d(space.degree(), space.knots(),
/// result.knots())` the coefficients.
///
/// \param space The space to subdivide.
/// \param n_subdivisions Equal sub-spans per existing span, at least 2 -- the oracle's
///        own bound. A direction whose refinement factor is 1 is skipped by its caller
///        rather than subdivided by one; `THBSplineSpace`'s `build_level_spaces` does
///        exactly that.
/// \param regularity The continuity at each inserted knot, in `[-1, degree - 1]`.
///        Empty for `degree - 1`, the maximal smoothness the degree admits.
/// \return The refined space.
/// \throws std::invalid_argument If `n_subdivisions < 2`, `regularity` is out of
///         range, or the merged vector exceeds the maximum multiplicity.
///
/// \note The result is non-periodic whatever `space` is, which is the oracle's
///       behaviour: `BsplineSpace1D.subdivide` builds `BsplineSpace1D(merged,
///       degree)` and lets `periodic` take its `False` default. Nothing in the THB
///       port reaches it with a periodic space -- `first_basis_per_interval` refuses
///       one first -- so this reproduces the oracle rather than repairing it.
template <Real T>
[[nodiscard]] BsplineSpace1D<T> subdivide(const BsplineSpace1D<T>& space,
                                          std::int64_t n_subdivisions,
                                          std::optional<std::int64_t> regularity) {
    if (n_subdivisions < 2) {
        throw std::invalid_argument("n_subdivisions must be >= 2, got "
                                    + std::to_string(n_subdivisions));
    }
    const std::int64_t degree = space.degree();
    const std::int64_t effective = regularity.value_or(degree - 1);
    if (effective < -1 || effective > degree - 1) {
        throw std::invalid_argument("regularity must be in [-1, degree - 1] = [-1, "
                                    + std::to_string(degree - 1) + "], got "
                                    + std::to_string(effective));
    }
    const std::vector<T> to_insert = uniform_subdivision_knots<T>(
        space.knots(), degree, space.tolerance(), n_subdivisions, effective);
    if (to_insert.empty()) {
        return space;
    }
    const std::vector<T> merged =
        inserted_knot_vector<T>(space.knots(), degree, to_insert, space.tolerance());
    return BsplineSpace1D<T>(std::span<const T>(merged), degree, false,
                             KnotSnapping::merge_near_duplicates);
}

}  // namespace pantr::bspline
