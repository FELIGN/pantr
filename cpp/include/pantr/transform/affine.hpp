#pragma once

/// \file
/// The affine map `x -> A x + b`, owned by the C++ core.
///
/// The second domain type moved under the 2026-08-27 amendment to
/// `design/cross_backend_types.md`, after `pantr::geometry::AABB`. It follows
/// that type's shape: validates and throws rather than asserting, is immutable
/// once built, and is wrapped rather than mirrored on the Python side.
///
/// ## Why this port is not justified by speed, and says so
///
/// Measured on the oracle before writing a line: `AffineTransform.__call__` is
/// 20-35% Python overhead at small point counts and essentially 0% at a million
/// points, where it is already a `dgemm`. So a C++ version is Eigen against
/// BLAS, not C++ against Python. What justifies the move is ownership -- a C++
/// consumer with no interpreter needs an affine map -- and that reason is worth
/// stating because the timings alone would argue the other way.
///
/// ## What parity can claim here, operation by operation
///
/// Unlike `AABB`, whose every operation is exact, three of these are not, and
/// the three differ in kind:
///
///  - **`inverse` is a bound, not an equality.** The oracle is `np.linalg.inv`,
///    LAPACK `getrf`/`getri`; this is `Eigen::PartialPivLU`. Both are LU with
///    partial pivoting and both are backward stable, so the two computed
///    inverses differ by roughly `c n rho kappa_inf(A) eps`. That is the same
///    derivation `change_basis.hpp` already carries for its solve, reused rather
///    than reinvented.
///  - **`compose` and `apply` are bounds.** A matrix product is a `dgemm` in
///    numpy and a blocked product in Eigen: the same terms, summed in a
///    different order.
///  - **Normalizing an axis or a normal is a bound, not an equality, and both
///    ways of trying for the equality were measured.** The oracle fuses its
///    multiply-add **inside** OpenBLAS's `ddot`, when it normalizes, and does
///    **not** fuse the array expression that follows. A translation unit has one
///    contraction setting, so it can match one or the other: at `x86-64` and
///    `x86-64-v2` the norm differs in the last bits (`18.330529174821066` against
///    numpy's `18.33052917482107` on one probe vector), and at `x86-64-v3` the
///    norm matches but the outer product then contracts where numpy did not, so
///    the matrices differ anyway. Neither build gives equality. The claim is
///    therefore an unconditional bound, derived in
///    `tests/parity/test_transform_affine.py` from one missing fusion per site.
///
///    Three other explanations were tried and refuted by measurement first, each
///    of which looked convincing: the contraction *setting* (`-ffp-contract=off`
///    changes nothing, because without FMA there is nothing to contract),
///    `sincos` (the extension does import it, but at run time it agrees with a
///    separate `cos`), and the sum's association (real, and the code below still
///    forms the outer product first because it is closer to the oracle -- but
///    **the bound does not require it**, and a test with the association undone
///    still passes, which is stated here rather than left to be discovered).
///
///  - **The trigonometry is a bound, and this was a surprise.** `rotation_2d`
///    does nothing but call `cos` and `sin` and negate, so it looked exact. It
///    is not: measured over 2000 angles, 2 of them give a matrix that differs in
///    the last bit, because **this extension's `cos` and the interpreter's
///    `math.cos` disagree by one ulp on roughly 0.1% of arguments**. Ruled out
///    along the way, each by measurement rather than by argument: numpy's scalar
///    `cos` is not the culprit (it agrees with `math.cos` on 200000 of 200000),
///    and neither is GCC folding the `cos`/`sin` pair into `sincos` (the
///    extension does import `sincos`, but at run time `sincos` and a separate
///    `cos` agree with each other and both differ from the interpreter). What
///    remains is that the two resolve different libm code, which is **not fully
///    pinned** -- it is recorded as an observation, not as a mechanism.
///
///    The consequence is the part that matters: every factory built on `cos` or
///    `sin` inherits a one-ulp bound rather than an equality, and no amount of
///    care on this side removes it.
///
/// ## No cached inverse on this side
///
/// The oracle memoizes `inverse` in a `functools.cached_property`. This type does
/// not: it holds no mutable member, so there is nothing to reason about across
/// threads, and a C++ caller that needs the inverse twice can hold it. The
/// Python wrapper keeps its own cache, where it is unobservable and where it
/// already was.
///
/// ## `about_center` rather than a `center` parameter
///
/// Five of the oracle's factories take an optional `center` and conjugate the
/// map by a translation to and from it. Here that is one method, applied after
/// the fact, because the conjugation is the same three-way product every time
/// and a parameter repeated on five signatures is the same function five times.
/// The Python wrapper keeps `center=`, as `AABB`'s `union` keeps its name over
/// the C++ `merge`.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::transform {

/// An affine map `x -> A x + b` in any spatial dimension `n >= 1`.
///
/// Stores the linear part row-major and the translation alongside it. Immutable
/// once built: every operation returns a new map.
template <Real T>
class AffineTransform {
  public:
    /// Build a map from its linear part and translation, validating both.
    ///
    /// \param matrix The linear part, `(n, n)` with `n >= 1`.
    /// \param offset The translation, length `n`.
    /// \throws std::invalid_argument If `matrix` is not square, `n` is zero, or
    ///         `offset` does not have length `n`.
    AffineTransform(span2d<const T> matrix, std::span<const T> offset)
        : dim_(matrix.extent(0)), matrix_(dim_ * dim_), offset_(offset.begin(), offset.end()) {
        if (matrix.extent(0) != matrix.extent(1)) {
            throw std::invalid_argument("matrix must be a square 2-D array, got shape ("
                                        + std::to_string(matrix.extent(0)) + ", "
                                        + std::to_string(matrix.extent(1)) + ").");
        }
        if (dim_ == 0) {
            throw std::invalid_argument("matrix must be a square 2-D array, got shape (0, 0).");
        }
        if (offset_.size() != dim_) {
            throw std::invalid_argument("translation must have shape (" + std::to_string(dim_)
                                        + ",), got (" + std::to_string(offset_.size()) + ",).");
        }
        for (std::size_t i = 0; i < dim_; ++i) {
            for (std::size_t j = 0; j < dim_; ++j) {
                matrix_[i * dim_ + j] = at(matrix, i, j);
            }
        }
    }

    /// The identity map in `n` dimensions.
    ///
    /// \param n Spatial dimension, `>= 1`.
    /// \return The identity.
    /// \throws std::invalid_argument If `n < 1`.
    [[nodiscard]] static AffineTransform identity(std::size_t n) {
        require_dim(n);
        std::vector<T> mat(n * n, T{0});
        for (std::size_t i = 0; i < n; ++i) {
            mat[i * n + i] = T{1};
        }
        return AffineTransform(Unchecked{}, n, std::move(mat), std::vector<T>(n, T{0}));
    }

    /// A pure translation.
    ///
    /// \param offset The translation, length `n >= 1`.
    /// \return The map `x -> x + offset`.
    /// \throws std::invalid_argument If `offset` is empty.
    [[nodiscard]] static AffineTransform translation(std::span<const T> offset) {
        require_dim(offset.size());
        const std::size_t n = offset.size();
        std::vector<T> mat(n * n, T{0});
        for (std::size_t i = 0; i < n; ++i) {
            mat[i * n + i] = T{1};
        }
        return AffineTransform(Unchecked{}, n, std::move(mat),
                               std::vector<T>(offset.begin(), offset.end()));
    }

    /// An axis-aligned scaling.
    ///
    /// \param factors One factor per axis, length `n >= 1`; each finite and non-zero.
    /// \return The diagonal map.
    /// \throws std::invalid_argument If a factor is zero or not finite, which
    ///         would make the map singular or undefined.
    [[nodiscard]] static AffineTransform scaling(std::span<const T> factors) {
        require_dim(factors.size());
        const std::size_t n = factors.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(value_of(factors[i]))) {
                throw std::invalid_argument("scaling factors must be finite.");
            }
            if (value_of(factors[i]) == T{0}) {
                throw std::invalid_argument(
                    "scaling factors must be non-zero (singular transform).");
            }
        }
        std::vector<T> mat(n * n, T{0});
        for (std::size_t i = 0; i < n; ++i) {
            mat[i * n + i] = factors[i];
        }
        return AffineTransform(Unchecked{}, n, std::move(mat), std::vector<T>(n, T{0}));
    }

    /// A rotation of the plane.
    ///
    /// \param angle The angle in radians; must be finite.
    /// \return The 2x2 rotation.
    /// \throws std::invalid_argument If `angle` is not finite.
    [[nodiscard]] static AffineTransform rotation_2d(T angle) {
        require_finite(angle, "angle");
        using std::cos;
        using std::sin;
        const T c = cos(angle);
        const T s = sin(angle);
        return AffineTransform(Unchecked{}, 2, std::vector<T>{c, -s, s, c},
                               std::vector<T>(2, T{0}));
    }

    /// A rotation about an arbitrary axis in space, by the Rodrigues formula.
    ///
    /// The axis is normalized here rather than by the caller. That normalization
    /// is the one step whose bit-exactness against the oracle depends on the
    /// build contracting a multiply-add; see the file comment.
    ///
    /// \param angle The angle in radians; must be finite.
    /// \param axis The rotation axis, length 3, finite and non-zero.
    /// \return The 3x3 rotation.
    /// \throws std::invalid_argument If `angle` is not finite, `axis` is not
    ///         length 3, or `axis` is zero or not finite.
    [[nodiscard]] static AffineTransform rotation_3d(T angle, std::span<const T> axis) {
        require_finite(angle, "angle");
        if (axis.size() != 3) {
            throw std::invalid_argument("Rotation axis must have shape (3,), got ("
                                        + std::to_string(axis.size()) + ",).");
        }
        const std::array<T, 3> u = normalized(axis, "Rotation axis");

        using std::cos;
        using std::sin;
        const T c = cos(angle);
        const T s = sin(angle);
        const T one_minus_c = T{1} - c;

        // R = I c + (1 - c) u u^T + s [u]x, in the oracle's term order AND its
        // association. Two things here are not free choices:
        //
        //   - `(1 - c) * (u_i u_j)`, not `((1 - c) u_i) u_j`. numpy forms the
        //     outer product first and scales it, and floating-point
        //     multiplication is not associative.
        //   - the scaled outer product is computed into its own statement before
        //     being added. Written as one expression, `-ffp-contract=on` fuses
        //     the multiply and the add into an FMA; numpy cannot fuse there,
        //     because the scaling and the addition are separate array operations
        //     with a materialised array between them. Splitting the statements
        //     asks the compiler not to contract; whether it obliges depends on
        //     the target ISA, which is why the parity claim is a bound rather
        //     than an equality. This form is kept because it is closer to the
        //     oracle, not because the bound needs it.
        std::vector<T> mat(9);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                const T diagonal = (i == j) ? c : T{0};
                const T outer = u[i] * u[j];
                const T scaled = one_minus_c * outer;
                mat[i * 3 + j] = diagonal + scaled;
            }
        }
        // Same reason: each `s * K_ij` is formed before it is added, so the pair
        // is not contracted into an FMA the oracle's array arithmetic cannot make.
        const std::array<T, 6> cross{-u[2], u[1], u[2], -u[0], -u[1], u[0]};
        const std::array<std::size_t, 6> where{1, 2, 3, 5, 6, 7};
        for (std::size_t k = 0; k < 6; ++k) {
            const T term = s * cross[k];
            mat[where[k]] += term;
        }
        return AffineTransform(Unchecked{}, 3, std::move(mat), std::vector<T>(3, T{0}));
    }

    /// A reflection across the hyperplane through the origin with the given normal.
    ///
    /// Householder: `A = I - 2 n n^T`, with `n` normalized here.
    ///
    /// \param normal The plane normal, length `n >= 1`, finite and non-zero.
    /// \return The reflection.
    /// \throws std::invalid_argument If `normal` is zero or not finite.
    [[nodiscard]] static AffineTransform mirror(std::span<const T> normal) {
        require_dim(normal.size());
        const std::size_t n = normal.size();
        const T norm = euclidean_norm(normal, "Mirror normal");
        std::vector<T> unit(n);
        for (std::size_t i = 0; i < n; ++i) {
            unit[i] = normal[i] / norm;
        }
        // Same association discipline as `rotation_3d`: the oracle forms
        // `outer(n, n)`, scales it by two, and subtracts a materialised array,
        // so the port associates the same way rather than to the left. Closer to
        // the oracle, and not required by the bound.
        std::vector<T> mat(n * n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const T diagonal = (i == j) ? T{1} : T{0};
                const T outer = unit[i] * unit[j];
                const T scaled = T{2} * outer;
                mat[i * n + j] = diagonal - scaled;
            }
        }
        return AffineTransform(Unchecked{}, n, std::move(mat), std::vector<T>(n, T{0}));
    }

    /// A shear that adds `factor * x[direction]` to `x[component]`.
    ///
    /// \param n Spatial dimension, `>= 1`.
    /// \param component The axis that is modified.
    /// \param direction The axis whose value drives the shear.
    /// \param factor The shear magnitude; must be finite.
    /// \return The shear.
    /// \throws std::invalid_argument If the two axes coincide, either is out of
    ///         range, or `factor` is not finite.
    [[nodiscard]] static AffineTransform shear(std::size_t n, std::size_t component,
                                               std::size_t direction, T factor) {
        require_dim(n);
        if (component == direction) {
            throw std::invalid_argument("component and direction must differ.");
        }
        if (component >= n) {
            throw std::invalid_argument("component must be in [0, " + std::to_string(n) + "), got "
                                        + std::to_string(component) + ".");
        }
        if (direction >= n) {
            throw std::invalid_argument("direction must be in [0, " + std::to_string(n) + "), got "
                                        + std::to_string(direction) + ".");
        }
        require_finite(factor, "factor");
        std::vector<T> mat(n * n, T{0});
        for (std::size_t i = 0; i < n; ++i) {
            mat[i * n + i] = T{1};
        }
        mat[component * n + direction] = factor;
        return AffineTransform(Unchecked{}, n, std::move(mat), std::vector<T>(n, T{0}));
    }

    /// The spatial dimension.
    ///
    /// \return The number of axes, `>= 1`.
    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }

    /// The linear part, row-major.
    ///
    /// \return A view of the stored matrix, valid while the map lives.
    [[nodiscard]] span2d<const T> matrix() const noexcept {
        return span2d<const T>(matrix_.data(), dim_, dim_);
    }

    /// The translation.
    ///
    /// \return A view of the stored offset, valid while the map lives.
    [[nodiscard]] std::span<const T> offset() const noexcept { return offset_; }

    /// The inverse map.
    ///
    /// Not memoized; see the file comment.
    ///
    /// \return The map `y -> A^-1 (y - b)`.
    /// \throws std::invalid_argument If the linear part is singular.
    [[nodiscard]] AffineTransform inverse() const {
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const Eigen::Map<const Matrix> a(matrix_.data(), static_cast<Eigen::Index>(dim_),
                                         static_cast<Eigen::Index>(dim_));
        const Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> lu(a);

        // PartialPivLU does not report singularity, by design: it is documented as
        // requiring an invertible matrix. So this has to decide, and it decides on
        // the same CRITERION LAPACK uses -- a pivot being exactly zero, which is
        // what `getrf` sets `info > 0` for.
        //
        // **The same criterion is not the same decision, and saying so was the
        // correction to this comment.** Eigen's factorization and LAPACK's are
        // different computations, so their pivots differ in the last bits and one
        // reaches exact zero where the other does not. The two backends therefore
        // disagree about invertibility for some exactly singular matrices, in
        // both directions, reachable with one-digit integers:
        //
        //     [[3, 4, -1], [2, 4, 1], [5, 8, 0]]   (r3 = r1 + r2)
        //         oracle inverts, this refuses
        //     [[-2, -4, 0], [3, 1, -4], [1, -3, -4]]
        //         oracle refuses, this inverts
        //
        // That is a discrete verdict and no tolerance bounds it, which is the
        // shape `design/backend_parity.md` Rule 11 already records for the BVH's
        // tie contract. It is pinned by a parity test rather than papered over,
        // so a future change has to confront it instead of passing quietly.
        //
        // An earlier version tested the DETERMINANT against zero and claimed in a
        // comment that this was the same condition. It is not, and the difference
        // is not academic: the determinant is the product of the pivots, so a
        // product of individually non-zero pivots underflows to exactly zero.
        // `AffineTransform.scaling({1e-300, 1e-300})` has an infinity-norm
        // condition number of exactly 1 and was refused here while the oracle
        // inverted it, and that call is reachable from the public API -- the
        // parity suite's own scaling case uses `1e-300`. Testing the pivots
        // cannot underflow, is scale-invariant, and is what the old comment
        // already said the code did.
        const auto pivots = lu.matrixLU().diagonal();
        for (Eigen::Index i = 0; i < pivots.size(); ++i) {
            if (value_of(pivots[i]) == T{0}) {
                throw std::invalid_argument("Cannot invert a singular affine transformation.");
            }
        }
        const Matrix inv = lu.inverse();

        std::vector<T> inv_mat(dim_ * dim_);
        std::vector<T> inv_off(dim_);
        for (std::size_t i = 0; i < dim_; ++i) {
            T acc{0};
            for (std::size_t j = 0; j < dim_; ++j) {
                const T value = inv(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
                inv_mat[i * dim_ + j] = value;
                acc += value * offset_[j];
            }
            inv_off[i] = -acc;
        }
        return AffineTransform(Unchecked{}, dim_, std::move(inv_mat), std::move(inv_off));
    }

    /// The map `x -> self(other(x))`.
    ///
    /// \param other The inner map, of the same dimension.
    /// \return The composition.
    /// \throws std::invalid_argument If the dimensions differ.
    [[nodiscard]] AffineTransform compose(const AffineTransform& other) const {
        if (dim_ != other.dim_) {
            throw std::invalid_argument("Cannot compose transforms of different dimensions ("
                                        + std::to_string(dim_) + " and "
                                        + std::to_string(other.dim_) + ").");
        }
        std::vector<T> mat(dim_ * dim_);
        std::vector<T> off(dim_);
        for (std::size_t i = 0; i < dim_; ++i) {
            for (std::size_t j = 0; j < dim_; ++j) {
                T acc{0};
                for (std::size_t k = 0; k < dim_; ++k) {
                    acc += matrix_[i * dim_ + k] * other.matrix_[k * dim_ + j];
                }
                mat[i * dim_ + j] = acc;
            }
            T acc{0};
            for (std::size_t k = 0; k < dim_; ++k) {
                acc += matrix_[i * dim_ + k] * other.offset_[k];
            }
            off[i] = acc + offset_[i];
        }
        return AffineTransform(Unchecked{}, dim_, std::move(mat), std::move(off));
    }

    /// Conjugate by a translation, so the linear part acts about `center`.
    ///
    /// `translate(c) . self . translate(-c)`, which is what the oracle's optional
    /// `center` argument builds.
    ///
    /// \param center The point to act about, length `dim()`.
    /// \return The re-centred map.
    /// \throws std::invalid_argument If `center` has the wrong length.
    [[nodiscard]] AffineTransform about_center(std::span<const T> center) const {
        if (center.size() != dim_) {
            throw std::invalid_argument("center must have shape (" + std::to_string(dim_)
                                        + ",), got (" + std::to_string(center.size()) + ",).");
        }
        std::vector<T> negated(dim_);
        for (std::size_t i = 0; i < dim_; ++i) {
            negated[i] = -center[i];
        }
        const auto to_origin = translation(std::span<const T>(negated));
        const auto back = translation(center);
        return back.compose(compose(to_origin));
    }

    /// Apply the map to a set of points.
    ///
    /// \param points The points, `(m, dim())`.
    /// \param out The destination, `(m, dim())`; may not alias `points`.
    /// \throws std::invalid_argument If either shape is wrong.
    void apply(span2d<const T> points, span2d<T> out) const {
        if (points.extent(1) != dim_) {
            throw std::invalid_argument("Points last dimension ("
                                        + std::to_string(points.extent(1))
                                        + ") must match transform dimension ("
                                        + std::to_string(dim_) + ").");
        }
        if (out.extent(0) != points.extent(0) || out.extent(1) != dim_) {
            throw std::invalid_argument("out must have the same shape as points.");
        }
        for (std::size_t p = 0; p < points.extent(0); ++p) {
            for (std::size_t i = 0; i < dim_; ++i) {
                T acc{0};
                for (std::size_t j = 0; j < dim_; ++j) {
                    acc += at(points, p, j) * matrix_[i * dim_ + j];
                }
                at(out, p, i) = acc + offset_[i];
            }
        }
    }

    /// Value equality: same dimension and the same stored entries.
    ///
    /// \param other The map to compare against.
    /// \return `true` when both parts match entry by entry.
    [[nodiscard]] bool operator==(const AffineTransform& other) const noexcept {
        if (dim_ != other.dim_) {
            return false;
        }
        for (std::size_t i = 0; i < matrix_.size(); ++i) {
            if (value_of(matrix_[i]) != value_of(other.matrix_[i])) {
                return false;
            }
        }
        for (std::size_t i = 0; i < dim_; ++i) {
            if (value_of(offset_[i]) != value_of(other.offset_[i])) {
                return false;
            }
        }
        return true;
    }

  private:
    /// Tag for the constructor that skips validation.
    ///
    /// Same reason as `pantr::geometry::AABB`'s: without it the unchecked
    /// overload would win by exact match wherever a factory builds vectors, and
    /// every derived map would skip validation without a word.
    struct Unchecked {};

    /// Construct from parts already known to satisfy the invariants.
    ///
    /// \param n The dimension.
    /// \param matrix The linear part, row-major, moved in.
    /// \param offset The translation, moved in.
    AffineTransform(Unchecked, std::size_t n, std::vector<T> matrix, std::vector<T> offset)
        : dim_(n), matrix_(std::move(matrix)), offset_(std::move(offset)) {}

    /// Reject a zero dimension.
    ///
    /// \param n The requested dimension.
    /// \throws std::invalid_argument If `n < 1`.
    static void require_dim(std::size_t n) {
        if (n < 1) {
            throw std::invalid_argument("dimension must be >= 1; got 0.");
        }
    }

    /// Reject a non-finite scalar argument.
    ///
    /// \param x The value.
    /// \param name Its name, for the message.
    /// \throws std::invalid_argument If `x` is not finite.
    static void require_finite(const T& x, const char* name) {
        if (!std::isfinite(value_of(x))) {
            throw std::invalid_argument(std::string(name) + " must be finite.");
        }
    }

    /// The Euclidean norm, fused so it reproduces the oracle's `ddot`.
    ///
    /// The multiply-add is written so the build can contract it, which is what
    /// makes this bit-exact against `np.linalg.norm`; see the file comment. It is
    /// a conditional claim, not an unconditional one: a build that does not fuse
    /// gives a different last bit and the parity claim degrades to a bound.
    ///
    /// \param v The vector.
    /// \param name Its name, for the message.
    /// \return Its norm.
    /// \throws std::invalid_argument If the norm is zero or not finite.
    static T euclidean_norm(std::span<const T> v, const char* name) {
        T sum{0};
        for (std::size_t i = 0; i < v.size(); ++i) {
            sum += v[i] * v[i];
        }
        using std::sqrt;
        const T norm = sqrt(sum);
        if (value_of(norm) == T{0} || !std::isfinite(value_of(norm))) {
            throw std::invalid_argument(std::string(name)
                                        + " must be a finite non-zero vector.");
        }
        return norm;
    }

    /// A unit vector of length 3, for the Rodrigues formula.
    ///
    /// \param v The axis.
    /// \param name Its name, for the message.
    /// \return The normalized axis.
    /// \throws std::invalid_argument If `v` is zero or not finite.
    static std::array<T, 3> normalized(std::span<const T> v, const char* name) {
        const T norm = euclidean_norm(v, name);
        return {v[0] / norm, v[1] / norm, v[2] / norm};
    }

    std::size_t dim_;          ///< Spatial dimension, `>= 1`.
    std::vector<T> matrix_;    ///< Linear part, row-major, `dim_ * dim_` entries.
    std::vector<T> offset_;    ///< Translation, `dim_` entries.
};

}  // namespace pantr::transform
