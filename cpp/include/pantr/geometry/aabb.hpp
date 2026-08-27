#pragma once

/// \file
/// Axis-aligned bounding box, and the first owned type in the C++ core.
///
/// ## Why this header contains a class at all
///
/// Every other header in this tree is free functions over spans, because
/// `design/cross_backend_types.md` used to say that types are owned by Python.
/// That rule was superseded on 2026-08-27: a C++ program links pantr with no
/// interpreter present, so a domain type must exist on this side or every such
/// program reimplements validation and invariants for itself. `AABB` is the
/// first type moved under that amendment, chosen because the superseded note
/// named exactly this shape as the hard case it had ducked -- a box with
/// equality and hashing semantics that both sides must agree on.
///
/// ## Validating rather than asserting, and why that is not a contradiction
///
/// `pantr/core/precondition.hpp` says a Layer 3 kernel asserts, and that the
/// bindings are where a *user* is protected. Both stay true: this is not a
/// kernel. It is the C++ counterpart of Layer 2, so it validates its arguments
/// and throws `std::invalid_argument`, in a release build as much as a debug
/// one. A caller with no Python cannot be protected by `cpp/bindings/`.
///
/// ## Parity notes for the Python oracle
///
/// Two places where reproducing `pantr.geometry.AABB` exactly takes care:
///
///  - **`transform` sums in a loop, and numpy does not always.** `np.sum` is
///    pairwise above a block size of 8, so a naive accumulation matches the
///    oracle bit for bit only while `ndim <= 8`. Above it the two summation
///    orders differ and the claim becomes a bound rather than an equality.
///    Nothing in pantr builds a box beyond a handful of axes today, but the
///    condition is a property of the input, not of the codebase, so it is
///    stated here rather than assumed.
///  - **Zero times an infinite bound.** The oracle masks `A[i, j] == 0` before
///    multiplying, because `0 * inf` is NaN and would poison an output axis the
///    transform projects out. The same mask is applied here, and for the same
///    reason.
///
/// ## What this type does NOT resolve
///
/// `pantr/grid/bvh.hpp`'s `node_overlaps` implements the same overlap predicate
/// over flattened per-node arrays, and its file comment claims to match
/// `pantr.geometry.AABB.overlaps`. **That claim is false for an empty query
/// box**: `AABB::overlaps` reports that an empty box overlaps nothing, while
/// the BVH predicate tests separating axes only and reports an overlap. Both
/// backends of the BVH behave that way, so the two agree with each other and
/// disagree with the box. This header reproduces the box's semantics, which is
/// what a port owes its oracle; reconciling the two is a separate decision and
/// is deliberately not taken here.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::geometry {

/// An axis-aligned bounding box in any spatial dimension `ndim >= 1`.
///
/// Stores two corner vectors of equal length. Entries may be finite or infinite;
/// NaN is rejected at construction. A box with `lo[i] > hi[i]` on some axis is
/// *empty* and contains no point, which `is_empty` reports.
///
/// Instances are immutable: there is no operation that changes an existing box,
/// and every derived box is returned by value.
template <Real T>
class AABB {
  public:
    /// Build a box from its two corners, validating them.
    ///
    /// \param lo Lower corner, length `ndim >= 1`.
    /// \param hi Upper corner, same length as `lo`.
    /// \throws std::invalid_argument If the lengths differ, `ndim` is zero, or
    ///         either corner contains NaN.
    AABB(std::span<const T> lo, std::span<const T> hi)
        : lo_(lo.begin(), lo.end()), hi_(hi.begin(), hi.end()) {
        if (lo_.size() != hi_.size()) {
            throw std::invalid_argument("AABB.lo and AABB.hi must share shape; got ("
                                        + std::to_string(lo_.size()) + ",) vs ("
                                        + std::to_string(hi_.size()) + ",).");
        }
        if (lo_.empty()) {
            throw std::invalid_argument("AABB: ndim must be >= 1; got 0.");
        }
        for (std::size_t d = 0; d < lo_.size(); ++d) {
            if (is_nan(lo_[d]) || is_nan(hi_[d])) {
                throw std::invalid_argument("AABB: bounds must not contain NaN.");
            }
        }
    }

    /// The everywhere-true box `(-inf, +inf)^ndim`.
    ///
    /// \param ndim Spatial dimension, `>= 1`.
    /// \return The unbounded box.
    /// \throws std::invalid_argument If `ndim < 1`.
    [[nodiscard]] static AABB unbounded(std::size_t ndim) {
        require_ndim(ndim, "AABB::unbounded");
        const T inf = std::numeric_limits<T>::infinity();
        return AABB(Unchecked{}, std::vector<T>(ndim, -inf), std::vector<T>(ndim, inf));
    }

    /// An empty box, `lo = +inf` and `hi = -inf`, the neutral element of `merge`.
    ///
    /// \param ndim Spatial dimension, `>= 1`.
    /// \return An empty box.
    /// \throws std::invalid_argument If `ndim < 1`.
    [[nodiscard]] static AABB empty(std::size_t ndim) {
        require_ndim(ndim, "AABB::empty");
        const T inf = std::numeric_limits<T>::infinity();
        return AABB(Unchecked{}, std::vector<T>(ndim, inf), std::vector<T>(ndim, -inf));
    }

    /// Build a box from an `(ndim, 2)` table of `[lo, hi]` rows.
    ///
    /// The dual of `as_bounds`, and the shape a histogram-style bounds argument
    /// already has.
    ///
    /// \param bounds The table, `(ndim, 2)` with `ndim >= 1`.
    /// \return The box.
    /// \throws std::invalid_argument If the second extent is not 2, `ndim` is
    ///         zero, or an entry is NaN.
    [[nodiscard]] static AABB from_bounds(span2d<const T> bounds) {
        if (bounds.extent(1) != 2) {
            throw std::invalid_argument("AABB::from_bounds: bounds must be (ndim, 2).");
        }
        const std::size_t n = bounds.extent(0);
        require_ndim(n, "AABB::from_bounds");
        std::vector<T> lo(n);
        std::vector<T> hi(n);
        for (std::size_t d = 0; d < n; ++d) {
            lo[d] = bounds(d, 0);
            hi[d] = bounds(d, 1);
        }
        return AABB(std::span<const T>(lo), std::span<const T>(hi));
    }

    /// Write the corners into an `(ndim, 2)` table of `[lo, hi]` rows.
    ///
    /// Takes the destination rather than returning one, because the C++ caller
    /// owns its storage and the Python wrapper wants a numpy array it allocated.
    ///
    /// \param out The destination, `(ndim, 2)`.
    /// \throws std::invalid_argument If `out` does not have shape `(ndim, 2)`.
    void as_bounds(span2d<T> out) const {
        if (out.extent(0) != ndim() || out.extent(1) != 2) {
            throw std::invalid_argument("AABB::as_bounds: out must be (ndim, 2).");
        }
        for (std::size_t d = 0; d < ndim(); ++d) {
            out(d, 0) = lo_[d];
            out(d, 1) = hi_[d];
        }
    }

    /// A compact representation, matching the oracle's `__repr__`.
    ///
    /// \return `"AABB(lo=[...], hi=[...])"`.
    [[nodiscard]] std::string to_string() const {
        const auto join = [](std::span<const T> v) {
            std::string s;
            for (std::size_t d = 0; d < v.size(); ++d) {
                if (d != 0) {
                    s += ", ";
                }
                s += std::to_string(value_of(v[d]));
            }
            return s;
        };
        return "AABB(lo=[" + join(lo_) + "], hi=[" + join(hi_) + "])";
    }

    /// The spatial dimension.
    ///
    /// \return The number of axes, `>= 1`.
    [[nodiscard]] std::size_t ndim() const noexcept { return lo_.size(); }

    /// The lower corner.
    ///
    /// \return A view of the stored lower corner, valid while the box lives.
    [[nodiscard]] std::span<const T> lo() const noexcept { return lo_; }

    /// The upper corner.
    ///
    /// \return A view of the stored upper corner, valid while the box lives.
    [[nodiscard]] std::span<const T> hi() const noexcept { return hi_; }

    /// Whether the box contains no point.
    ///
    /// \return `true` when `lo[i] > hi[i]` on at least one axis.
    [[nodiscard]] bool is_empty() const noexcept {
        for (std::size_t d = 0; d < ndim(); ++d) {
            if (value_of(lo_[d]) > value_of(hi_[d])) {
                return true;
            }
        }
        return false;
    }

    /// Whether `x` lies inside the box or on its boundary.
    ///
    /// An empty box contains no point, which falls out of the comparison rather
    /// than needing its own branch.
    ///
    /// \param x The point, length `ndim`.
    /// \return `true` when `lo[i] <= x[i] <= hi[i]` on every axis.
    /// \throws std::invalid_argument If `x` has the wrong length or contains NaN.
    [[nodiscard]] bool contains_point(std::span<const T> x) const {
        require_len(x.size(), "AABB::contains_point: x");
        for (std::size_t d = 0; d < ndim(); ++d) {
            if (is_nan(x[d])) {
                throw std::invalid_argument("AABB::contains_point: x must not contain NaN.");
            }
        }
        for (std::size_t d = 0; d < ndim(); ++d) {
            if (value_of(x[d]) < value_of(lo_[d]) || value_of(x[d]) > value_of(hi_[d])) {
                return false;
            }
        }
        return true;
    }

    /// Whether the two boxes share at least one point.
    ///
    /// An empty box on either side overlaps nothing. That short-circuit is the
    /// one place this type deliberately differs from `pantr/grid/bvh.hpp`'s
    /// predicate; see the file comment.
    ///
    /// \param other The box to test against, same `ndim`.
    /// \return `true` when the two intersect.
    /// \throws std::invalid_argument If the dimensions differ.
    [[nodiscard]] bool overlaps(const AABB& other) const {
        require_same_ndim(other, "overlaps");
        if (is_empty() || other.is_empty()) {
            return false;
        }
        for (std::size_t d = 0; d < ndim(); ++d) {
            const auto lo = std::max(value_of(lo_[d]), value_of(other.lo_[d]));
            const auto hi = std::min(value_of(hi_[d]), value_of(other.hi_[d]));
            if (lo > hi) {
                return false;
            }
        }
        return true;
    }

    /// The smallest box containing both operands.
    ///
    /// Named `merge` rather than `union`, which is a keyword. Empty boxes are
    /// neutral, and an empty operand is returned *as it is* rather than
    /// normalized -- the oracle does the same, and its equality is by value, so
    /// two empties with different corners are two different boxes.
    ///
    /// \param other The box to merge with, same `ndim`.
    /// \return The bounding box of the two.
    /// \throws std::invalid_argument If the dimensions differ.
    [[nodiscard]] AABB merge(const AABB& other) const {
        require_same_ndim(other, "union");
        if (is_empty()) {
            return other;
        }
        if (other.is_empty()) {
            return *this;
        }
        std::vector<T> lo(ndim());
        std::vector<T> hi(ndim());
        for (std::size_t d = 0; d < ndim(); ++d) {
            lo[d] = value_of(lo_[d]) < value_of(other.lo_[d]) ? lo_[d] : other.lo_[d];
            hi[d] = value_of(hi_[d]) > value_of(other.hi_[d]) ? hi_[d] : other.hi_[d];
        }
        return AABB(Unchecked{}, std::move(lo), std::move(hi));
    }

    /// The axis-aligned intersection, if the two boxes meet.
    ///
    /// \param other The box to intersect with, same `ndim`.
    /// \return The intersection, or no value when the two are disjoint, which
    ///         includes either operand being empty.
    /// \throws std::invalid_argument If the dimensions differ.
    [[nodiscard]] std::optional<AABB> intersect(const AABB& other) const {
        require_same_ndim(other, "intersect");
        if (is_empty() || other.is_empty()) {
            return std::nullopt;
        }
        std::vector<T> lo(ndim());
        std::vector<T> hi(ndim());
        for (std::size_t d = 0; d < ndim(); ++d) {
            lo[d] = value_of(lo_[d]) > value_of(other.lo_[d]) ? lo_[d] : other.lo_[d];
            hi[d] = value_of(hi_[d]) < value_of(other.hi_[d]) ? hi_[d] : other.hi_[d];
            if (value_of(lo[d]) > value_of(hi[d])) {
                return std::nullopt;
            }
        }
        return AABB(Unchecked{}, std::move(lo), std::move(hi));
    }

    /// Inflate every axis symmetrically by its own radius.
    ///
    /// A negative radius shrinks and may produce an empty box, which is allowed
    /// and reported by `is_empty`. An infinite bound stays infinite, since
    /// `inf + finite == inf`.
    ///
    /// \param r Per-axis radius, length `ndim`; every entry must be finite.
    /// \return The padded box.
    /// \throws std::invalid_argument If `r` has the wrong length or a non-finite entry.
    [[nodiscard]] AABB pad(std::span<const T> r) const {
        if (r.size() != ndim()) {
            throw std::invalid_argument("pad(r) requires r scalar or shape ("
                                        + std::to_string(ndim()) + ",); got ("
                                        + std::to_string(r.size()) + ",).");
        }
        for (std::size_t d = 0; d < ndim(); ++d) {
            if (!std::isfinite(value_of(r[d]))) {
                throw std::invalid_argument("pad(r) entries must be finite.");
            }
        }
        std::vector<T> lo(ndim());
        std::vector<T> hi(ndim());
        for (std::size_t d = 0; d < ndim(); ++d) {
            lo[d] = lo_[d] - r[d];
            hi[d] = hi_[d] + r[d];
        }
        return AABB(Unchecked{}, std::move(lo), std::move(hi));
    }

    /// Inflate every axis symmetrically by one radius.
    ///
    /// \param r The radius, applied to every axis; must be finite.
    /// \return The padded box.
    /// \throws std::invalid_argument If `r` is not finite.
    [[nodiscard]] AABB pad(T r) const {
        const std::vector<T> vec(ndim(), r);
        return pad(std::span<const T>(vec));
    }

    /// The tight axis-aligned wrap of this box's image under `x -> A x + b`.
    ///
    /// Per output axis `i` and input axis `j` the contribution is the pair
    /// `(A[i, j] * lo[j], A[i, j] * hi[j])`; the smaller goes into the new lower
    /// corner and the larger into the new upper one. A zero entry of `A`
    /// contributes nothing even against an infinite bound, which is what keeps
    /// `0 * inf` from poisoning an axis the map projects out.
    ///
    /// \param matrix The linear part `A`, shape `(ndim, ndim)`.
    /// \param offset The translation `b`, length `ndim`.
    /// \return The wrap of the transformed box; an empty box maps to
    ///         `AABB::empty(ndim)`.
    /// \throws std::invalid_argument If the shapes are wrong, or if the wrap
    ///         produces NaN, which opposing infinite bounds in one row can do.
    [[nodiscard]] AABB transform(span2d<const T> matrix, std::span<const T> offset) const {
        if (is_empty()) {
            return AABB::empty(ndim());
        }
        if (matrix.extent(0) != ndim() || matrix.extent(1) != ndim()) {
            throw std::invalid_argument("AABB::transform: matrix must be (ndim, ndim).");
        }
        require_len(offset.size(), "AABB::transform: offset");
        std::vector<T> lo(ndim());
        std::vector<T> hi(ndim());
        for (std::size_t i = 0; i < ndim(); ++i) {
            T acc_lo = offset[i];
            T acc_hi = offset[i];
            for (std::size_t j = 0; j < ndim(); ++j) {
                const T a = matrix(i, j);
                if (value_of(a) == T{0}) {
                    continue;
                }
                const T t0 = a * lo_[j];
                const T t1 = a * hi_[j];
                acc_lo += value_of(t0) < value_of(t1) ? t0 : t1;
                acc_hi += value_of(t0) > value_of(t1) ? t0 : t1;
            }
            if (is_nan(acc_lo) || is_nan(acc_hi)) {
                throw std::invalid_argument(
                    "AABB::transform produced NaN bounds; the transform is incompatible with "
                    "this AABB (for example, a singular matrix combined with infinite bounds).");
            }
            lo[i] = acc_lo;
            hi[i] = acc_hi;
        }
        return AABB(Unchecked{}, std::move(lo), std::move(hi));
    }

    /// Value equality: the two corner vectors match entry by entry.
    ///
    /// This is value equality, not geometric equality. Two empty boxes with
    /// different corners are unequal even though both contain no point, which is
    /// the oracle's documented behaviour. Signed zeros compare equal, and the
    /// hash below normalizes them so the two stay consistent.
    ///
    /// \param other The box to compare against.
    /// \return `true` when the dimensions and all corner entries match.
    [[nodiscard]] bool operator==(const AABB& other) const noexcept {
        if (ndim() != other.ndim()) {
            return false;
        }
        for (std::size_t d = 0; d < ndim(); ++d) {
            if (value_of(lo_[d]) != value_of(other.lo_[d])
                || value_of(hi_[d]) != value_of(other.hi_[d])) {
                return false;
            }
        }
        return true;
    }

  private:
    /// Tag for the unchecked constructor.
    ///
    /// It exists so that the unchecked overload cannot be selected by accident:
    /// without it, `AABB(lo, hi)` on two `std::vector<T>` would prefer the
    /// unchecked one over the public span constructor by exact match, and every
    /// derived box in this class would skip validation without a word.
    struct Unchecked {};

    /// Construct from corners already known to satisfy the invariants.
    ///
    /// Used by the factories and by every method that derives a box from one or
    /// two valid boxes, where re-running the NaN scan would establish nothing.
    /// `transform` is the exception and does its own check, because its
    /// arithmetic can produce NaN from finite inputs.
    ///
    /// \param lo Lower corner, moved in.
    /// \param hi Upper corner, moved in.
    AABB(Unchecked, std::vector<T> lo, std::vector<T> hi)
        : lo_(std::move(lo)), hi_(std::move(hi)) {}

    /// Whether a scalar's value is NaN, through the Tier B access point.
    ///
    /// \param x The scalar to test.
    /// \return `true` when the underlying value is NaN.
    [[nodiscard]] static bool is_nan(const T& x) noexcept { return std::isnan(value_of(x)); }

    /// Reject a zero dimension.
    ///
    /// \param ndim The requested dimension.
    /// \param who The caller's name, for the message.
    /// \throws std::invalid_argument If `ndim < 1`.
    static void require_ndim(std::size_t ndim, const char* who) {
        if (ndim < 1) {
            throw std::invalid_argument(std::string(who) + ": ndim must be >= 1; got 0.");
        }
    }

    /// Reject an argument whose length is not this box's dimension.
    ///
    /// \param len The argument's length.
    /// \param who The caller and argument name, for the message.
    /// \throws std::invalid_argument If `len != ndim()`.
    void require_len(std::size_t len, const char* who) const {
        if (len != ndim()) {
            throw std::invalid_argument(std::string(who) + " must have length "
                                        + std::to_string(ndim()) + "; got "
                                        + std::to_string(len) + ".");
        }
    }

    /// Reject a box of a different dimension.
    ///
    /// \param other The other box.
    /// \param who The caller's name, for the message.
    /// \throws std::invalid_argument If the dimensions differ.
    void require_same_ndim(const AABB& other, const char* who) const {
        if (ndim() != other.ndim()) {
            throw std::invalid_argument("AABB." + std::string(who)
                                        + ": dimension mismatch (a.ndim="
                                        + std::to_string(ndim()) + " vs b.ndim="
                                        + std::to_string(other.ndim()) + ").");
        }
    }

    std::vector<T> lo_;  ///< Lower corner, length `ndim`.
    std::vector<T> hi_;  ///< Upper corner, length `ndim`.
};

}  // namespace pantr::geometry

/// Hash an `AABB` consistently with its equality.
///
/// Signed zero is the whole difficulty. `operator==` compares by value, so
/// `-0.0` and `+0.0` are equal, while their bit patterns differ; hashing the raw
/// values would give equal boxes different hashes and break every unordered
/// container. Adding `0.0` maps `-0.0` to `+0.0` and leaves every other value
/// bitwise alone, infinities included. NaN, the other case where bitwise and
/// IEEE comparison disagree, cannot occur -- the constructor rejects it.
///
/// This mirrors what `pantr.geometry.AABB.__hash__` does on the Python side. The
/// two need not produce the same integer, and they do not: what has to hold is
/// that each is consistent with its own equality.
template <pantr::Real T>
struct std::hash<pantr::geometry::AABB<T>> {
    /// Compute the hash.
    ///
    /// \param box The box to hash.
    /// \return A hash equal for boxes that compare equal.
    [[nodiscard]] std::size_t operator()(const pantr::geometry::AABB<T>& box) const noexcept {
        std::size_t seed = box.ndim();
        const auto mix = [&seed](auto value) {
            // +0.0 normalizes the signed zero; see the class comment.
            const std::size_t h = std::hash<decltype(value)>{}(value + decltype(value){0});
            seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        };
        for (std::size_t d = 0; d < box.ndim(); ++d) {
            mix(pantr::value_of(box.lo()[d]));
            mix(pantr::value_of(box.hi()[d]));
        }
        return seed;
    }
};
