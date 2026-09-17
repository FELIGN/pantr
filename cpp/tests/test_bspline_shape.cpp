/// \file
/// Tests for `pantr/bspline/shape.hpp`: reversal, direction permutation and the
/// affine transform.
///
/// ## What is checked against what
///
/// None of the three introduces a numerical kernel, so nothing here needs the closed
/// forms `test_bspline_structural.cpp` had to build. What they *do* introduce is
/// bookkeeping over strides, knot indices and space handles, and each check below is
/// chosen for a bookkeeping error it would catch that a value comparison would not.
///
/// **`reverse` is checked geometrically, which is an oracle and not a mirror.**
/// Reversing a curve reparametrizes it so that `S_rev(a + b - u) = S(u)` for every `u`
/// in the domain. Sampling both sides with `slice_point` -- whose corner cut
/// `test_bspline_structural.cpp` certifies against partition of unity and linear
/// precision -- tests the flip, the knot reflection and the periodic roll *together*
/// against a statement none of the three appears in. A wrong roll, a vector reflected
/// about the wrong midpoint, or a flip on the wrong axis all displace the map, and
/// none is visible in the control net alone.
///
/// It is also checked as an **involution on a dyadic knot vector**, where the check is
/// exact rather than bounded: `a + b` and every `(a + b) - k` are then representable,
/// so the second reflection returns the first's operands unrounded and
/// `reverse(reverse(f)) == f` holds bit for bit. On a non-dyadic vector it need not,
/// and asserting that it did would be asserting an identity floating point does not
/// have -- so a non-dyadic vector gets the bounded geometric check and nothing
/// stronger.
///
/// **`permute_directions` computes nothing at all**, so every check on it is exact.
/// The load-bearing one is an asymmetric permutation of a field whose three extents,
/// three degrees and component count all differ: a stride computed over the wrong axis
/// set, or one left a factor of `num_components` too large, then reads a different
/// coefficient while the shape still agrees. `pantr/bezier/shape.hpp` records that
/// exact defect as having shipped once.
///
/// **`transform` is checked against closed forms it cannot round.** The identity map,
/// and a power-of-two scaling with an integer translation, are exactly representable
/// in both storage formats, so the expected net is known in closed form and the
/// comparison is bitwise rather than bounded -- which is what makes it an independent
/// check in `design/backend_parity.md`'s sense rather than a second spelling of
/// `A x + b`. A rational field is checked separately, because its weight column is
/// copied rather than computed and a transform that touched it would still pass every
/// non-rational check.
///
/// **The space handles are checked by pointer**, not by value. `reverse` must carry
/// every untouched direction's handle through, `permute_directions` every handle, and
/// `transform` the whole space handle -- that is what
/// `design/bspline_ownership_lifetime.md`'s F6 needs from this backend, and a rebuild
/// with an equal value would satisfy every value comparison in this file.
///
/// ## The bound, derived
///
/// Only the geometric `reverse` check needs one. It compares two corner cuts of the
/// same curve over two knot vectors, so it charges `gamma_K` with `K` the sum of one
/// cut on each side plus the reflection that separates them:
///
///  - **`5p` per corner cut**, the count `test_bspline_structural.cpp` derives: at
///    most `p` passes at 5 roundings each, and the weights are in `[0, 1]` for a
///    parameter inside its span, so no pass amplifies the magnitude.
///  - **2 for the reflection**, being `fl(a + b)` and `fl((a + b) - k)`, charged
///    against the **domain magnitude** `|a| + |b|` rather than against the reflected
///    knot: `(a + b) - k` cancels near either domain end, and a rounding error is
///    bounded relative to the operands that produced it, not relative to what the
///    expression cancels down to. A parametric displacement reaches the sampled point
///    through the curve's own slope, which is bounded here by the net's coordinate
///    range over the domain width.
///
/// The magnitude is the net's largest coordinate, since a corner cut is a convex
/// combination of coefficients and cannot leave their range, plus `K` denormal floors
/// for the absolute half of Higham's model.
///
/// ## Both storage formats
///
/// Every templated check runs at `double` and at `float`. The refusal checks are
/// `double`-only: none of the three messages depends on the storage format.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/shape.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/bspline/structural.hpp"

using pantr::bspline::Bspline;
using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;
using pantr::bspline::permute_directions;
using pantr::bspline::reverse;
using pantr::bspline::slice_point;
using pantr::bspline::transform;

namespace {

/// `gamma_m = m u / (1 - m u)`, the standard accumulation constant.
///
/// Defined here rather than shared, which is the convention the sibling test files
/// follow; collecting them is a cleanup this file does not take.
///
/// \tparam T The format the roundings are charged in.
/// \param m The number of roundings.
/// \return The constant, in units of the value being bounded.
template <class T>
double gamma_of(std::int64_t m) {
    const double unit = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    const double count = static_cast<double>(m);
    return (count * unit) / (1.0 - (count * unit));
}

/// A space over a knot vector, shared.
///
/// \tparam T The storage format.
/// \param knots The knot vector.
/// \param degree The degree.
/// \param periodic Whether the space is periodic.
/// \return The space, in a handle a field can take.
template <class T>
std::shared_ptr<const BsplineSpace1D<T>> direction(const std::vector<T>& knots,
                                                   std::int64_t degree, bool periodic) {
    return std::make_shared<const BsplineSpace1D<T>>(
        std::span<const T>(knots), degree, periodic, KnotSnapping::merge_near_duplicates);
}

/// A field over the given directions, with the given coefficients.
///
/// \tparam T The storage format.
/// \param directions One space handle per direction.
/// \param values The coefficients, row-major under `(*num_basis, num_components)`.
/// \param num_components How many components each coefficient has.
/// \param is_rational Whether the last component is a homogeneous weight.
/// \return The field.
template <class T>
Bspline<T> field_of(std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions,
                    const std::vector<T>& values, std::size_t num_components,
                    bool is_rational) {
    auto space = std::make_shared<const BsplineSpace<T>>(std::move(directions));
    std::vector<std::size_t> shape;
    for (const std::int64_t count : space->num_basis()) {
        shape.push_back(static_cast<std::size_t>(count));
    }
    shape.push_back(num_components);
    return Bspline<T>(space,
                      typename Bspline<T>::net_type(std::span<const T>(values),
                                                    std::span<const std::size_t>(shape)),
                      is_rational);
}

/// A ramp of distinct coefficients.
///
/// Distinct values are what makes a misplaced coefficient visible: a net of equal
/// values passes every rearrangement check ever written. The ramp starts at one so
/// that it can serve a rational field, whose weight column must stay positive.
///
/// \tparam T The storage format.
/// \param count How many values.
/// \return `1, 2, ..., count`.
template <class T>
std::vector<T> ramp(std::size_t count) {
    std::vector<T> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<T>(i + 1);
    }
    return values;
}

/// Whether two nets are bit-for-bit equal.
///
/// \tparam T The storage format.
/// \param left One net's values.
/// \param right The other's.
/// \return `true` when the two agree exactly, lengths included.
template <class T>
bool same_values(std::span<const T> left, std::span<const T> right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

/// A bound on `|S'(u)|` over the whole domain, from the de Boor derivative formula.
///
/// `S'(u) = p * sum_i (P_{i+1} - P_i) / (t_{i+p+1} - t_{i+1}) * N_{i,p-1}(u)`, and the
/// degree-`(p-1)` basis is a partition of unity, so
/// `|S'| <= p * max_i |P_{i+1} - P_i| / min_i (t_{i+p+1} - t_{i+1})`. The denominator
/// is replaced by the **smallest positive single knot span**, which is no larger than
/// any `p`-span, so the quotient is an over-estimate and the bound stays an upper one.
///
/// This replaces an earlier `magnitude / (hi - lo)`, which read as derived and was a
/// heuristic: a curve's local sensitivity is set by its local span, and a vector whose
/// narrowest span is well below the domain width has a slope the domain-average
/// quotient understates. The knot vectors in this file are already such vectors.
///
/// \tparam T The storage format.
/// \param knots The knot vector.
/// \param degree The degree `p`.
/// \param values The control net, components interleaved.
/// \param components How many components each coefficient has.
/// \return The Lipschitz constant, or 0 at degree 0, where the curve is piecewise
///         constant and a parametric displacement moves the value by at most the
///         coefficient range rather than through a slope.
template <class T>
double lipschitz_bound(const std::vector<T>& knots, std::int64_t degree,
                       const std::vector<T>& values, std::size_t components) {
    if (degree == 0) {
        return 0.0;
    }
    double widest_step = 0.0;
    const std::size_t coefficients = values.size() / components;
    for (std::size_t i = 0; i + 1 < coefficients; ++i) {
        for (std::size_t c = 0; c < components; ++c) {
            widest_step = std::max(widest_step,
                                   std::abs(static_cast<double>(values[((i + 1) * components) + c])
                                            - static_cast<double>(values[(i * components) + c])));
        }
    }
    double narrowest_span = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < knots.size(); ++i) {
        const double span =
            static_cast<double>(knots[i + 1]) - static_cast<double>(knots[i]);
        if (span > 0.0) {
            narrowest_span = std::min(narrowest_span, span);
        }
    }
    return static_cast<double>(degree) * widest_step / narrowest_span;
}

// ---------------------------------------------------------------------------
// reverse
// ---------------------------------------------------------------------------

/// `reverse` reparametrizes a curve without moving it.
///
/// `S_rev(a + b - u) = S(u)` at every sampled parameter, within the bound the file
/// comment derives. Run on a non-dyadic, non-uniform knot vector, which is where the
/// reflection actually rounds.
///
/// The vector is also deliberately **not symmetric about its own domain midpoint**.
/// A symmetric vector is its own reflection, so this identity holds against a
/// `reverse` that flipped the net and left the knot vector entirely alone -- which is
/// a mutation this check is here to catch, and did not while the interior knots sat
/// at 0.3 and 0.7.
///
/// \tparam T The storage format.
template <class T>
void check_reverse_does_not_move_the_curve() {
    const std::vector<T> knots{T(0.1), T(0.1), T(0.1), T(0.3), T(0.4), T(0.9), T(0.9), T(0.9)};
    const std::int64_t degree = 2;
    const std::vector<T> values = ramp<T>(5 * 2);
    const Bspline<T> curve = field_of<T>({direction<T>(knots, degree, false)}, values, 2, false);
    const Bspline<T> flipped = reverse<T>(curve, 0);

    const double lo = static_cast<double>(knots[static_cast<std::size_t>(degree)]);
    const double hi = static_cast<double>(knots[knots.size() - static_cast<std::size_t>(degree)
                                                - 1]);
    const double magnitude =
        static_cast<double>(*std::max_element(values.begin(), values.end()));
    // A parametric displacement of the reflected knot reaches the sampled point
    // through the curve's own slope, which `lipschitz_bound` derives from the de Boor
    // derivative formula rather than from the domain width; the two corner cuts
    // contribute directly.
    const double slope = lipschitz_bound<T>(knots, degree, values, 2);
    const std::int64_t cut_roundings = 5 * degree;
    const double bound =
        (gamma_of<T>(2 * cut_roundings) * magnitude)
        + (gamma_of<T>(2) * (std::abs(lo) + std::abs(hi)) * slope)
        + (static_cast<double>(2 * cut_roundings + 2)
           * static_cast<double>(std::numeric_limits<T>::denorm_min()));

    for (const double u : {0.1, 0.17, 0.3, 0.35, 0.4, 0.62, 0.83, 0.9}) {
        const std::vector<T> here = slice_point<T>(curve, u);
        const std::vector<T> there = slice_point<T>(flipped, (lo + hi) - u);
        for (std::size_t c = 0; c < here.size(); ++c) {
            const double deviation =
                std::abs(static_cast<double>(here[c]) - static_cast<double>(there[c]));
            PANTR_CHECK_MSG(deviation <= bound,
                            "reverse moved the curve at u = " + std::to_string(u)
                                + ", component " + std::to_string(c) + ": deviation "
                                + std::to_string(deviation) + " against bound "
                                + std::to_string(bound));
        }
    }
}

/// `reverse` is an involution on a dyadic knot vector, exactly.
///
/// See the file comment: every value the two reflections produce is representable
/// here, so this is a bitwise check and not a bounded one. It is also where the
/// untouched directions' handles are checked to have been carried through.
///
/// **Both directions are exercised, and the extents differ** (5 against 3). Reversing
/// only direction 0 would leave the `outer`/`along`/`inner` decomposition untested for
/// any axis but the first, so a `reverse` that ignored its `direction` argument and
/// always took axis 0 would pass. Both vectors are asymmetric about their own domain
/// midpoints, which is what lets the knot vacuity guards below fire at all.
///
/// \tparam T The storage format.
template <class T>
void check_reverse_is_an_exact_involution_on_a_dyadic_vector() {
    const std::vector<T> along{T(0), T(0), T(0), T(0.25), T(0.5), T(1), T(1), T(1)};
    const std::vector<T> across{T(0), T(0), T(0.25), T(1), T(1)};
    const Bspline<T> surface =
        field_of<T>({direction<T>(along, 2, false), direction<T>(across, 1, false)},
                    ramp<T>(5 * 3 * 2), 2, false);

    const Bspline<T> once = reverse<T>(surface, 0);
    const Bspline<T> twice = reverse<T>(once, 0);

    PANTR_CHECK_MSG(same_values<T>(twice.net().values(), surface.net().values()),
                    "reverse twice over a dyadic vector did not return the control net");
    PANTR_CHECK_MSG(same_values<T>(twice.space_ref().space_ref(0).knots(),
                                   surface.space_ref().space_ref(0).knots()),
                    "reverse twice over a dyadic vector did not return the knot vector");
    PANTR_CHECK_MSG(!same_values<T>(once.net().values(), surface.net().values()),
                    "reverse left the control net alone, so the involution is vacuous");
    // The knot half of the same vacuity guard, and it is not redundant with the net
    // half: a `reverse` that flips the net and never reflects the vector passes the
    // round trip above, because a no-op is its own inverse. `along` is asymmetric
    // about its domain midpoint precisely so this can fire.
    PANTR_CHECK_MSG(!same_values<T>(once.space_ref().space_ref(0).knots(),
                                    surface.space_ref().space_ref(0).knots()),
                    "reverse left the knot vector alone, so the involution is vacuous");

    PANTR_CHECK_MSG(once.space()->spaces()[1] == surface.space()->spaces()[1],
                    "reverse rebuilt the untouched direction instead of carrying its handle");
    PANTR_CHECK_MSG(once.space()->spaces()[0] != surface.space()->spaces()[0],
                    "reverse carried the reversed direction's handle through unchanged");

    // The same statement about direction 1. This is the only place any `reverse` here
    // is asked for an axis other than the first, so it is what stands between the
    // suite and a `reverse` that ignores its `direction` argument.
    const Bspline<T> once_across = reverse<T>(surface, 1);
    const Bspline<T> twice_across = reverse<T>(once_across, 1);

    PANTR_CHECK_MSG(same_values<T>(twice_across.net().values(), surface.net().values()),
                    "reverse twice over direction 1 did not return the control net");
    PANTR_CHECK_MSG(same_values<T>(twice_across.space_ref().space_ref(1).knots(),
                                   surface.space_ref().space_ref(1).knots()),
                    "reverse twice over direction 1 did not return the knot vector");
    PANTR_CHECK_MSG(!same_values<T>(once_across.net().values(), surface.net().values()),
                    "reverse over direction 1 left the control net alone, so it is vacuous");
    PANTR_CHECK_MSG(!same_values<T>(once_across.space_ref().space_ref(1).knots(),
                                    surface.space_ref().space_ref(1).knots()),
                    "reverse over direction 1 left the knot vector alone, so it is vacuous");
    // Direction 0 must come back untouched, which is the half that fails loudly if
    // `reverse` reached for axis 0 regardless of what it was asked for.
    PANTR_CHECK_MSG(same_values<T>(once_across.space_ref().space_ref(0).knots(),
                                   surface.space_ref().space_ref(0).knots()),
                    "reverse over direction 1 altered direction 0's knot vector");
    PANTR_CHECK_MSG(once_across.space()->spaces()[0] == surface.space()->spaces()[0],
                    "reverse over direction 1 rebuilt direction 0 instead of carrying it");
}

/// `reverse` flips a periodic direction with its cyclic shift.
///
/// The geometric statement is the same as the non-periodic one; what this adds is the
/// roll, which is the only place a periodic field differs. A missing roll leaves the
/// net a plain flip and displaces the map by whole control points, far outside the
/// bound.
///
/// **The vector is deliberately not uniform.** A uniform periodic vector over `[0, 1]`
/// is symmetric about its own domain midpoint, so `(a + b) - knots[::-1]` reproduces it
/// exactly and this check would hold against a `reverse` that never reflected the knots
/// at all. The interior spans here are `(0.125, 0.375, 0.25, 0.25)`, which keeps the
/// cyclic ghost structure a periodic space needs while moving four of the seven knots.
/// All entries stay dyadic, so the reflection is exact in both storage formats.
///
/// \tparam T The storage format.
template <class T>
void check_reverse_handles_a_periodic_direction() {
    const std::vector<T> knots{T(-0.25), T(0), T(0.125), T(0.5), T(0.75), T(1), T(1.125)};
    const std::int64_t degree = 1;
    const auto space = direction<T>(knots, degree, true);
    const auto stored = static_cast<std::size_t>(space->num_basis());
    const std::vector<T> values = ramp<T>(stored * 2);
    const Bspline<T> curve = field_of<T>({space}, values, 2, false);
    const Bspline<T> flipped = reverse<T>(curve, 0);

    const double lo = static_cast<double>(knots[static_cast<std::size_t>(degree)]);
    const double hi = static_cast<double>(knots[knots.size() - static_cast<std::size_t>(degree)
                                                - 1]);
    const double magnitude =
        static_cast<double>(*std::max_element(values.begin(), values.end()));
    const std::int64_t cut_roundings = 5 * degree;
    const double slope = lipschitz_bound<T>(knots, degree, values, 2);
    const double bound = (gamma_of<T>(2 * cut_roundings) * magnitude)
                         + (gamma_of<T>(2) * (std::abs(lo) + std::abs(hi)) * slope)
                         + (static_cast<double>(2 * cut_roundings + 2)
                            * static_cast<double>(std::numeric_limits<T>::denorm_min()));

    for (const double u : {0.0, 0.125, 0.25, 0.6, 0.75, 1.0}) {
        const std::vector<T> here = slice_point<T>(curve, u);
        const std::vector<T> there = slice_point<T>(flipped, (lo + hi) - u);
        for (std::size_t c = 0; c < here.size(); ++c) {
            const double deviation =
                std::abs(static_cast<double>(here[c]) - static_cast<double>(there[c]));
            PANTR_CHECK_MSG(deviation <= bound,
                            "reverse moved a periodic curve at u = " + std::to_string(u)
                                + ", component " + std::to_string(c) + ": deviation "
                                + std::to_string(deviation) + " against bound "
                                + std::to_string(bound));
        }
    }
}

// ---------------------------------------------------------------------------
// permute_directions
// ---------------------------------------------------------------------------

/// An asymmetric permutation reads the coefficient the definition names.
///
/// The field's three extents, three degrees and component count all differ, so a
/// stride taken over the wrong axis set reads a different coefficient while the shape
/// still agrees. The expected value is computed by direct multi-index arithmetic on
/// the source, which is a different expression from the strided walk under test.
///
/// \tparam T The storage format.
template <class T>
void check_permute_directions_reads_the_named_coefficient() {
    // Distinct extents are what makes a transposed stride visible, so the middle
    // direction carries five basis functions against the first's four and the last's
    // two, and the three degrees differ too.
    const std::vector<T> a{T(0), T(0), T(0), T(0.5), T(1), T(1), T(1)};        // 4, p=2
    const std::vector<T> b_wide{T(0), T(0), T(0.25), T(0.5), T(0.75), T(1), T(1)};  // 5, p=1
    const std::vector<T> c{T(0), T(0), T(1), T(1)};                           // 2, p=1
    const std::size_t components = 3;
    const std::size_t na = 4;
    const std::size_t nb_wide = 5;
    const std::size_t nc = 2;

    const std::vector<T> values = ramp<T>(na * nb_wide * nc * components);
    const Bspline<T> volume =
        field_of<T>({direction<T>(a, 2, false), direction<T>(b_wide, 1, false),
                     direction<T>(c, 1, false)},
                    values, components, false);

    const std::vector<std::int64_t> permutation{1, 2, 0};
    const Bspline<T> permuted =
        permute_directions<T>(volume, std::span<const std::int64_t>(permutation));

    PANTR_CHECK_MSG(permuted.net().shape()[0] == nb_wide && permuted.net().shape()[1] == nc
                        && permuted.net().shape()[2] == na
                        && permuted.net().shape()[3] == components,
                    "permute_directions produced the wrong net shape");

    const std::span<const T> out = permuted.net().values();
    for (std::size_t i = 0; i < nb_wide; ++i) {
        for (std::size_t j = 0; j < nc; ++j) {
            for (std::size_t k = 0; k < na; ++k) {
                for (std::size_t comp = 0; comp < components; ++comp) {
                    // New index (i, j, k) is old index (k, i, j), because new
                    // direction 0 is old direction 1, new 1 is old 2 and new 2 is
                    // old 0.
                    const std::size_t source =
                        ((((k * nb_wide) + i) * nc) + j) * components + comp;
                    const std::size_t destination =
                        ((((i * nc) + j) * na) + k) * components + comp;
                    PANTR_CHECK_MSG(out[destination] == values[source],
                                    "permute_directions read the wrong coefficient at new index ("
                                        + std::to_string(i) + ", " + std::to_string(j) + ", "
                                        + std::to_string(k) + ")");
                }
            }
        }
    }

    PANTR_CHECK_MSG(permuted.space()->spaces()[0] == volume.space()->spaces()[1]
                        && permuted.space()->spaces()[1] == volume.space()->spaces()[2]
                        && permuted.space()->spaces()[2] == volume.space()->spaces()[0],
                    "permute_directions rebuilt a direction instead of reordering the handles");
}

/// The identity permutation is a no-op, and a permutation composed with its inverse
/// returns the field.
///
/// Both are exact: nothing is computed. The inverse check is what catches a
/// permutation applied in the opposite sense, which the identity check cannot see.
///
/// \tparam T The storage format.
template <class T>
void check_permute_directions_composes_and_inverts() {
    const std::vector<T> a{T(0), T(0), T(0), T(0.5), T(1), T(1), T(1)};
    const std::vector<T> b{T(0), T(0), T(0.25), T(0.5), T(0.75), T(1), T(1)};
    const std::vector<T> c{T(0), T(0), T(1), T(1)};
    const Bspline<T> volume =
        field_of<T>({direction<T>(a, 2, false), direction<T>(b, 1, false),
                     direction<T>(c, 1, false)},
                    ramp<T>(4 * 5 * 2 * 3), 3, false);

    const std::vector<std::int64_t> identity{0, 1, 2};
    const Bspline<T> same =
        permute_directions<T>(volume, std::span<const std::int64_t>(identity));
    PANTR_CHECK_MSG(same_values<T>(same.net().values(), volume.net().values()),
                    "the identity permutation moved a coefficient");

    const std::vector<std::int64_t> forward{1, 2, 0};
    const std::vector<std::int64_t> backward{2, 0, 1};
    const Bspline<T> there =
        permute_directions<T>(volume, std::span<const std::int64_t>(forward));
    const Bspline<T> back =
        permute_directions<T>(there, std::span<const std::int64_t>(backward));
    PANTR_CHECK_MSG(same_values<T>(back.net().values(), volume.net().values()),
                    "a permutation composed with its inverse did not return the control net");
    PANTR_CHECK_MSG(!same_values<T>(there.net().values(), volume.net().values()),
                    "the forward permutation moved nothing, so the inverse check is vacuous");
}

// ---------------------------------------------------------------------------
// transform
// ---------------------------------------------------------------------------

/// A two-by-two matrix as a `span2d`.
///
/// \param storage The four entries, row-major; kept alive by the caller.
/// \return A view of it.
pantr::span2d<const double> as_2x2(const std::vector<double>& storage) {
    return pantr::span2d<const double>(storage.data(), 2, 2);
}

/// The identity map and a power-of-two scaling are exact, and the space is shared.
///
/// Both matrices are exactly representable in either storage format and every product
/// is a multiplication by a power of two followed by an addition of an integer, so the
/// expected net is a closed form the comparison can be bitwise against.
///
/// \tparam T The storage format.
template <class T>
void check_transform_is_exact_on_a_representable_map() {
    const std::vector<T> knots{T(0), T(0), T(0), T(0.5), T(1), T(1), T(1)};
    const std::vector<T> values = ramp<T>(4 * 2);
    const Bspline<T> curve = field_of<T>({direction<T>(knots, 2, false)}, values, 2, false);

    const std::vector<double> identity{1.0, 0.0, 0.0, 1.0};
    const std::vector<double> nothing{0.0, 0.0};
    const Bspline<T> unchanged =
        transform<T>(curve, as_2x2(identity), std::span<const double>(nothing));
    PANTR_CHECK_MSG(same_values<T>(unchanged.net().values(), curve.net().values()),
                    "the identity affine map moved a control point");
    PANTR_CHECK_MSG(unchanged.space() == curve.space(),
                    "transform rebuilt the space instead of sharing the field's handle");

    // Deliberately **not symmetric**. A diagonal matrix is its own transpose, so a
    // transform reading `A` where it should read `A.T` agrees with its closed form on
    // every diagonal map -- a mutation that survived this check while the only
    // matrices here were the identity and a scaling. Every entry is a power of two or
    // zero and every coefficient is a small integer, so the products and sums below
    // are exact in both storage formats and the comparison stays bitwise.
    const std::vector<double> shear{2.0, 0.5, 0.0, 4.0};
    const std::vector<double> shift{-3.0, 8.0};
    const Bspline<T> mapped =
        transform<T>(curve, as_2x2(shear), std::span<const double>(shift));
    for (std::size_t k = 0; k * 2 < values.size(); ++k) {
        const T u = values[k * 2];
        const T v = values[(k * 2) + 1];
        const T x = static_cast<T>((u * T(2)) + (v * T(0.5)) + T(-3));
        const T y = static_cast<T>((u * T(0)) + (v * T(4)) + T(8));
        PANTR_CHECK_MSG(mapped.net().values()[k * 2] == x
                            && mapped.net().values()[(k * 2) + 1] == y,
                        "transform disagreed with its closed form at coefficient "
                            + std::to_string(k));
    }
}

/// A rational field's weight column is copied and its coordinates carry the weight.
///
/// `w (A x + b) = A (w x) + w b`, so a translation scales by the stored weight. A
/// transform that touched the weight, or that forgot to scale the translation by it,
/// passes every non-rational check in this file.
///
/// \tparam T The storage format.
template <class T>
void check_transform_carries_a_rational_weight() {
    const std::vector<T> knots{T(0), T(0), T(1), T(1)};
    // Two coefficients of (w x, w y, w), with dyadic weights so the check stays exact.
    const std::vector<T> values{T(4), T(8), T(2), T(3), T(6), T(0.5)};
    const Bspline<T> curve = field_of<T>({direction<T>(knots, 1, false)}, values, 3, true);

    const std::vector<double> identity{1.0, 0.0, 0.0, 1.0};
    const std::vector<double> shift{2.0, -4.0};
    const Bspline<T> mapped =
        transform<T>(curve, as_2x2(identity), std::span<const double>(shift));

    const std::span<const T> out = mapped.net().values();
    PANTR_CHECK_MSG(out[2] == values[2] && out[5] == values[5],
                    "transform changed a rational field's weight column");
    PANTR_CHECK_MSG(out[0] == static_cast<T>(values[0] + (values[2] * T(2)))
                        && out[1] == static_cast<T>(values[1] + (values[2] * T(-4))),
                    "transform did not scale the translation by the first weight");
    PANTR_CHECK_MSG(out[3] == static_cast<T>(values[3] + (values[5] * T(2)))
                        && out[4] == static_cast<T>(values[4] + (values[5] * T(-4))),
                    "transform did not scale the translation by the second weight");
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// Run a call and return the `std::invalid_argument` text it threw, or a marker.
///
/// \param call The call to run.
/// \return The message, or `"<no throw>"`.
template <class F>
std::string refusal_of(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return "<no throw>";
}

/// Each of the three refuses its bad argument with the oracle's own message.
///
/// The texts are the Python wrapper's, character for character, because a caller with
/// no Python reads these and `tests/parity/test_bspline_shape.py` compares them.
void check_the_refusals_carry_the_oracle_text() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const Bspline<double> curve =
        field_of<double>({direction<double>(knots, 2, false)}, ramp<double>(4 * 2), 2, false);

    const std::string bad_direction =
        refusal_of([&] { static_cast<void>(reverse<double>(curve, 1)); });
    PANTR_CHECK_MSG(bad_direction == "direction must be in [0, 1), got 1.",
                    "reverse's refusal reads \"" + bad_direction + "\"");

    const std::vector<std::int64_t> not_a_permutation{0, 0};
    const std::string bad_permutation = refusal_of([&] {
        static_cast<void>(permute_directions<double>(
            curve, std::span<const std::int64_t>(not_a_permutation)));
    });
    PANTR_CHECK_MSG(bad_permutation == "permutation must be a permutation of range(1), got [0, 0].",
                    "permute_directions' refusal reads \"" + bad_permutation + "\"");

    const std::vector<double> three_by_three(9, 0.0);
    const std::vector<double> three_offsets(3, 0.0);
    const std::string bad_rank = refusal_of([&] {
        static_cast<void>(transform<double>(curve,
                                            pantr::span2d<const double>(three_by_three.data(), 3,
                                                                        3),
                                            std::span<const double>(three_offsets)));
    });
    PANTR_CHECK_MSG(bad_rank
                        == "Transform dimension (3) does not match the geometric rank (2) of the "
                           "control points.",
                    "transform's refusal reads \"" + bad_rank + "\"");

    const std::vector<double> two_by_two{1.0, 0.0, 0.0, 1.0};
    const std::vector<double> one_offset(1, 0.0);
    const std::string bad_offset = refusal_of([&] {
        static_cast<void>(transform<double>(curve, as_2x2(two_by_two),
                                            std::span<const double>(one_offset)));
    });
    PANTR_CHECK_MSG(bad_offset == "The translation must have 2 entries.",
                    "transform's offset refusal reads \"" + bad_offset + "\"");
}

/// Run every templated check at one storage format.
///
/// \tparam T The storage format.
template <class T>
void check_every_format() {
    check_reverse_does_not_move_the_curve<T>();
    check_reverse_is_an_exact_involution_on_a_dyadic_vector<T>();
    check_reverse_handles_a_periodic_direction<T>();
    check_permute_directions_reads_the_named_coefficient<T>();
    check_permute_directions_composes_and_inverts<T>();
    check_transform_is_exact_on_a_representable_map<T>();
    check_transform_carries_a_rational_weight<T>();
}

}  // namespace

int main() {
    check_every_format<double>();
    check_every_format<float>();
    check_the_refusals_carry_the_oracle_text();
    return pantr::test::summary("bspline_shape");
}
