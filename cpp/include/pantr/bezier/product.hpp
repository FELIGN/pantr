#pragma once

/// \file
/// The pointwise product of two n-dimensional Béziers, and the composition of one
/// with another.
///
/// Free functions over `Bezier<T>`, as `bezier.hpp` says: the fourth and last of the
/// operation ports over the value type. `compose` composes over
/// `scalar_bernstein_product_1d` where the oracle does and over the n-dimensional
/// product below where the oracle does that instead; neither reimplements the other.
///
/// ## What is exactly representable, and what is exactly computed
///
/// The product of Béziers of degrees `p` and `q` is a polynomial of degree `p + q`,
/// so it is **exactly representable** in the Bernstein basis of that degree: no
/// approximation is made in choosing the output's degree, unlike degree reduction.
/// The composition of a degree-`m` outer map with a degree-`n` inner one is likewise
/// exactly representable at degree `m * n` per direction.
///
/// **The coefficients are not exact.** Each is a binomial-weighted sum formed in
/// floating point, so it carries the roundings of that sum. The Python oracle's
/// docstrings say "the exact pointwise product" and "the exact composition"; read as
/// a statement about the representation that is true, read as a statement about the
/// computed numbers it is false, and it is the second reading a caller comparing two
/// backends will take. Stated here rather than copied, on the ticket's instruction.
/// `tests/parity/test_bezier_product.py` is where the size of the inexactness is
/// measured.
///
/// ## The arithmetic runs at the STORAGE width, and that is the opposite of a kernel
///
/// `design/backend_parity.md` Rule 9: an oracle's accumulation width is a per-kernel
/// fact. Here the oracle for `multiply` is **numpy**, not numba, and numpy computes a
/// `float32` expression in `float32`. Every intermediate in `bernstein_product_1d`
/// and `bernstein_product_nd` below is therefore `T`, never `accumulator_t<T>`. That
/// is the inverse of `kernels_1d.hpp`, whose numba oracle promotes its `float64`
/// coefficient tables against a `float32` array and so accumulates in `double`.
///
/// Widening here would be the same class of error `shape.hpp` records for a parameter
/// it narrowed, seen from the other side: the width belongs to the operation, and
/// reading it off the surrounding module is how a port gets it wrong at `float32`
/// while agreeing at `float64`.
///
/// ### Where widening would actually change the answer, and where it provably would not
///
/// Worth being precise, because two of the mutations
/// `tests/parity/test_bezier_product.py` records **survive the parity suite** and a
/// reader is entitled to know whether that is a gap in the tests or a fact about the
/// arithmetic. It is the second, and the rule is one line: **widening a single
/// operation whose result is immediately narrowed back to `T` cannot change it;
/// widening a chain of two or more can.**
///
///  - A single **product** of two `float32` values is *exact* in `double` -- the
///    result needs at most 48 significand bits and `double` has 53 -- so narrowing it
///    is the one rounding either way. That covers `bernstein_product_nd`'s weighting
///    of its operands and its final reciprocal scaling.
///  - A single **sum** of two `float32` values is not exact in `double`, but double
///    rounding through an intermediate format sufficiently wider than the target is
///    known to equal single rounding, and `double` against `float32` is the instance
///    that result's condition covers with room to spare (Figueroa, *When is double
///    rounding innocuous?*, ACM SIGNUM Newsletter 30(3):21-26, 1995,
///    doi:10.1145/221332.221334 -- the same result `_binomial_tables` rests on, and
///    the one place in this header that leans on a reference rather than on
///    arithmetic anyone can redo). That covers the accumulation `d[k] + term`.
///  - The **chain** `(coefficient * f_i) * g_j` is where the width bites: widened, the
///    intermediate is not narrowed, so the second multiplication receives a different
///    operand. Nine `float32` cases move when it is, which is what pins the claim.
///
/// So the code says `T` throughout because that is what the oracle says and it is the
/// simplest form that is right; the paragraph above is what a future reader needs in
/// order to tell a harmless widening from a defect.
///
/// `compose` mixes the two, because its oracle does. Its Bernstein products go
/// through `scalar_bernstein_product_1d` for a univariate inner map -- a numba kernel,
/// so `accumulator_t<T>` inside it -- and through `bernstein_product_nd` above that,
/// which is numpy and so `T`. Everything else in `compose` -- the `1 - g`, the
/// binomial scaling of a basis, the accumulation of the tensor terms -- is numpy and
/// so `T`.
///
/// ## The binomial tables cross as data
///
/// `multiply`, `compose` and `bernstein_product_nd` take their **binomial and
/// reciprocal-binomial tables as arguments** rather than assembling them. That is the
/// ruling `core/reduction_operator.hpp` records and `degree.hpp` extends, applied
/// again for the same reason: the tables are assembled once from exact arithmetic,
/// rounded to the storage format once, and are an array by the time anything computes
/// with them.
///
/// Here the reason is **different** from the precedent's rather than a stronger form of
/// it, and the distinction is worth keeping straight. `degree.hpp`'s reduction operator
/// cannot be computed natively at all: it is the solution of an exact rational system
/// whose numerators reach 156 bits, so a faithful assembly needs arbitrary precision and
/// `double` loses eleven digits. A binomial coefficient needs no such thing -- it is an
/// integer recurrence with no division, and `core::bincoeff` already computes it exactly.
/// What binds here is **domain**, not precision: `core::bincoeff` stops at
/// `core::kBincoeffMaxN`, which is 61, and **the numpy oracle has no limit at all**,
/// because `math.comb` is arbitrary precision. So `Bezier.multiply` is exact at
/// `p + q = 80` where `scalar_bernstein_product_1d` is outside its int64 envelope and
/// undefined, and `design/cross_backend_types.md` records that difference of domain as
/// one of the three reasons `multiply` keeps reaching the numpy helper rather than the
/// dispatched kernel. Computing the tables from `core::bincoeff` would import the
/// kernel's envelope into an operation that does not have one, which is a numerics
/// change and not a port.
///
/// Widening the native recurrence would move that envelope rather than remove it:
/// `unsigned __int128` reaches about degree 130 and then stops for the same reason. The
/// oracle's domain is unbounded, so a table crossing is what matches it. This is a
/// smaller gap than `degree.hpp`'s, and it is cheaper to close if anyone decides the
/// bound is worth having natively -- which is a decision, not an omission.
///
/// **The consequence, stated plainly:** a C++ caller with no Python cannot multiply
/// or compose unaided, and must supply tables it obtained elsewhere.
/// `product_table_order` and `composition_table_order` tell it how large they have to
/// be, so that the size is defined once rather than once per language. This is the
/// same gap `degree.hpp` records for its reduction operators, and it is tracked
/// rather than hidden.
///
/// ## Which roundings the equality depends on
///
/// The claim these functions carry is an **equality**, not a bound, and it is an
/// equality only because the operation order is reproduced rather than rearranged.
/// Three orderings are load-bearing and none of them is the obvious one:
///
///  - **The 1-D product accumulates in `i`-major order.** The oracle forms a
///    `(p+1, q+1)` table of terms and hands it to `np.add.at`, whose index array is
///    `k_mat.ravel()`; that is row-major, so coefficient `k` accumulates its terms in
///    increasing `i`. Summing in increasing `j`, or over `k`'s own range, is a
///    different number.
///  - **The 1-D product normalizes per term; the n-D one normalizes at the end.** The
///    oracle folds `1 / C(r, k)` into the `(p+1, q+1)` coefficient table before any
///    accumulation, and in the n-D helper multiplies the finished convolution by
///    `inv_w_r` afterwards. The two are not the same computation and neither may be
///    substituted for the other.
///  - **Every product is left-associated exactly as numpy writes it.**
///    `coeff[i] * cq[j] * inv_cr[k]` is `(cp * cq) * inv_cr`, and
///    `coeff * b_f * b_g` is `(coeff * b_f) * b_g`. Reassociating either changes the
///    last bits, which is precisely the mutation that caught an earlier parity test
///    exercising none of the ported code.
///
/// ### Contraction, and where these functions are not what the sibling claims assume
///
/// Every other Bézier claim is conditional on whether the target ISA has a fused
/// multiply-add, because every other Bézier kernel writes its accumulation as one
/// expression, `a + b * c`, which `-ffp-contract=on` may fuse. **These four functions
/// do not.** `bernstein_product_1d` computes the term into a named local and adds it in
/// the next statement, and `-ffp-contract=on` is expression-scoped in the compiler this
/// project builds with, so there is nothing for it to fuse across the statement
/// boundary.
///
/// Measured rather than reasoned, by disassembling an FMA-enabled build and by running
/// the product at all three contraction settings:
///
/// Configure a second build tree with `-DCMAKE_CXX_FLAGS="-mavx2 -mfma"`, then count
/// `vfmadd` per demangled symbol in `objdump -d` of `bezier.cpp.o`: it is zero in all
/// four functions here and nonzero in `scalar_bernstein_product_1d`. (No line
/// continuations in that recipe: a `///` line ending in a backslash is one comment
/// spanning two lines, which this build rejects under `-Werror=comment`, and it did.)
///
/// `off` and `on` agree bit for bit; `=fast` does not, so the site is contractible in
/// principle and the claim keeps its bounded arm for a build that reaches it. What
/// follows for a reader:
///
///  - **`multiply`'s equality survives an FMA host at this project's own flags.** The
///    parity harness selects its arm from the ISA rather than from the contraction
///    scope, so on such a host it asserts the *bounded* claim and observes zero
///    differing elements. That is sound and slack, not wrong, and
///    `tests/parity/test_bezier_product.py` says so rather than presenting the bounded
///    arm as exercised.
///  - **`compose`'s does not**, and the reason is not its own arithmetic. Its univariate
///    branch calls `scalar_bernstein_product_1d`, whose accumulation *is* a single
///    expression and does carry one fused site per instantiation at `-ffp-contract=on`.
///    So the composition inherits a build dependence the product does not have.
///
/// ## Validating rather than asserting
///
/// Like `bezier.hpp`, `evaluate.hpp`, `degree.hpp` and `shape.hpp`: operations on a
/// domain type validate and throw `std::invalid_argument` in a release build as much
/// as a debug one. A caller with no Python cannot be protected by `cpp/bindings/`.
///
/// Two of the oracle's checks have no counterpart here and one has a different type.
/// The dtype check cannot exist: both operands are `Bezier<T>` and the template
/// parameter carries the format, so the mismatch is unsayable. The rational-operand
/// rejection in `compose` is the oracle's `TypeError` and is an
/// `std::invalid_argument` here, which nanobind maps to `ValueError`; the exception
/// *type* stays in parity because Python's Layer 2 refuses a rational operand above
/// the backend branch, so the C++ check is reached only by a caller that has no
/// Python to raise for it.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bezier/axis_layout.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/degree.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bezier {

namespace detail {

/// Refuse two operands the product formula is not defined for.
///
/// The messages are the oracle's, character for character
/// (`_bezier_product._multiply_bezier`), because
/// `tests/parity/test_bezier_product.py` compares them. The oracle's dtype check
/// sits between these two and has no counterpart; see the file comment.
///
/// \param a First operand.
/// \param b Second operand.
/// \throws std::invalid_argument If the dimensions or the ranks differ.
template <Real T>
void require_compatible_operands(const Bezier<T>& a, const Bezier<T>& b) {
    if (a.dim() != b.dim()) {
        throw std::invalid_argument("Operands must have the same dimension. Got "
                                    + std::to_string(a.dim()) + " and "
                                    + std::to_string(b.dim()) + ".");
    }
    if (a.rank() != b.rank()) {
        throw std::invalid_argument("Operands must have the same rank. Got "
                                    + std::to_string(a.rank()) + " and "
                                    + std::to_string(b.rank()) + ".");
    }
}

/// Refuse a binomial table that does not reach the upper index the operation needs.
///
/// \param table The table, `(order + 1, order + 1)` or larger.
/// \param order The largest upper index that will be read.
/// \param what The table's argument name, for the message.
/// \throws std::invalid_argument If either extent is too small.
template <Real T>
void require_table_extents(span2d<const T> table, std::size_t order, const char* what) {
    if (table.extent(0) <= order || table.extent(1) <= order) {
        throw std::invalid_argument(std::string(what) + " must have shape at least ("
                                    + std::to_string(order + 1) + ", "
                                    + std::to_string(order + 1) + "), got ("
                                    + std::to_string(table.extent(0)) + ", "
                                    + std::to_string(table.extent(1)) + ").");
    }
}

/// Row-major strides for a shape, innermost last.
///
/// \param shape The extents.
/// \return One stride per extent, `strides.back() == 1`.
[[nodiscard]] inline std::vector<std::size_t> row_major_strides(
    std::span<const std::size_t> shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t d = shape.size(); d-- > 1;) {
        strides[d - 1] = strides[d] * shape[d];
    }
    return strides;
}

/// Decode a flat row-major index into its multi-index.
///
/// \param flat The flat index, below the product of `shape`.
/// \param shape The extents.
/// \param index Receives one component per extent.
inline void decode_row_major(std::size_t flat, std::span<const std::size_t> shape,
                            std::span<std::size_t> index) {
    for (std::size_t d = shape.size(); d-- > 0;) {
        index[d] = flat % shape[d];
        flat /= shape[d];
    }
}

/// Reproduce `_bernstein_product_coefficients`: the univariate Bernstein product.
///
/// `d_k = (1 / C(r, k)) * sum_{i + j = k} C(p, i) C(q, j) f_i g_j`, formed as the
/// oracle forms it and in the order the oracle forms it: the reciprocal folded into
/// the coefficient before any accumulation, the two control values multiplied in
/// left to right, and the sum for each `k` taken in increasing `i`. See the file
/// comment for why each of those is load-bearing.
///
/// Every intermediate is `T`. The oracle is numpy, which computes a `float32`
/// expression in `float32`.
///
/// \param f Control points of the first curve, `(p + 1, rank)` row-major.
/// \param g Control points of the second curve, `(q + 1, rank)` row-major.
/// \param rank Number of components; the trailing extent of both operands.
/// \param binomials `C(n, k)` in the storage format, at least `(r + 1, r + 1)`.
/// \param inverse_binomials `1 / C(n, k)` likewise.
/// \param out Product control points, `(r + 1, rank)` row-major. Zeroed, then
///        written in full.
///
/// \note No input validation is performed; the shapes are the caller's to establish.
template <Real T>
void bernstein_product_1d(std::span<const T> f, std::span<const T> g, std::size_t rank,
                          span2d<const T> binomials, span2d<const T> inverse_binomials,
                          std::span<T> out) {
    const std::size_t p = (f.size() / rank) - 1;
    const std::size_t q = (g.size() / rank) - 1;
    const std::size_t r = p + q;

    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = T(0);
    }

    for (std::size_t i = 0; i <= p; ++i) {
        for (std::size_t j = 0; j <= q; ++j) {
            // `(cp * cq) * inv_cr`, which is how numpy associates
            // `cp[:, None] * cq[None, :] * inv_cr[k_mat]`.
            const T coefficient = T(at(binomials, p, i) * at(binomials, q, j))
                                  * at(inverse_binomials, r, i + j);
            const std::size_t base = (i + j) * rank;
            for (std::size_t s = 0; s < rank; ++s) {
                // `(coeff * b_f) * b_g`, and then one narrowing add per term: the
                // accumulator IS the output array, exactly as `np.add.at` makes it.
                const T term = T(coefficient * f[(i * rank) + s]) * g[(j * rank) + s];
                out[base + s] = out[base + s] + term;
            }
        }
    }
}

/// Reproduce `_bernstein_product_coefficients_nd`: the tensor-product Bernstein
/// product.
///
/// A different computation from `bernstein_product_1d` rather than a generalisation
/// of it, and the difference is not incidental: this one weights each operand once,
/// convolves, and divides the finished convolution by the product of reciprocal
/// binomials, where the univariate helper folds the reciprocal into every term. The
/// oracle keeps both and so does this.
///
/// Every intermediate is `T`, and every product of per-direction factors is
/// left-associated over ascending direction, which is what `np.multiply.outer`
/// applied in a loop produces.
///
/// \param f Control points of the first patch, row-major with extents `shape_f`.
/// \param shape_f Extents of `f`, the component count last.
/// \param g Control points of the second patch, row-major with extents `shape_g`.
/// \param shape_g Extents of `g`, the component count last and equal to `shape_f`'s.
/// \param binomials `C(n, k)` in the storage format, reaching every `r_d`.
/// \param inverse_binomials `1 / C(n, k)` likewise.
/// \param out Product control points, row-major with extents
///        `(p_d + q_d + 1, ..., rank)`. Zeroed, then written in full.
///
/// \note No input validation is performed; the shapes are the caller's to establish.
template <Real T>
void bernstein_product_nd(std::span<const T> f, std::span<const std::size_t> shape_f,
                          std::span<const T> g, std::span<const std::size_t> shape_g,
                          span2d<const T> binomials, span2d<const T> inverse_binomials,
                          std::span<T> out) {
    const std::size_t ndim = shape_f.size() - 1;
    const std::size_t rank = shape_f.back();

    // `end() - 1` rather than `begin() + ndim`: the latter adds an unsigned index to an
    // iterator, which this build rejects under -Wsign-conversion -Werror. `degree.hpp`
    // drops the component axis the same way.
    std::vector<std::size_t> extents_f(shape_f.begin(), shape_f.end() - 1);
    std::vector<std::size_t> extents_g(shape_g.begin(), shape_g.end() - 1);
    std::vector<std::size_t> extents_h(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        extents_h[d] = extents_f[d] + extents_g[d] - 1;
    }

    const std::vector<std::size_t> strides_h = row_major_strides(extents_h);
    const std::size_t count_f = extent_product(extents_f, 0);
    const std::size_t count_g = extent_product(extents_g, 0);
    const std::size_t count_h = extent_product(extents_h, 0);

    // The weighted operands, `w_f[..., None] * b_f` and its twin. One rounding per
    // stored value, and the weight itself is a left-associated product over ascending
    // direction.
    std::vector<std::size_t> index(ndim);
    std::vector<T> weighted_f(f.size());
    for (std::size_t a = 0; a < count_f; ++a) {
        decode_row_major(a, extents_f, index);
        T weight = at(binomials, extents_f[0] - 1, index[0]);
        for (std::size_t d = 1; d < ndim; ++d) {
            weight = weight * at(binomials, extents_f[d] - 1, index[d]);
        }
        for (std::size_t s = 0; s < rank; ++s) {
            weighted_f[(a * rank) + s] = weight * f[(a * rank) + s];
        }
    }
    std::vector<T> weighted_g(g.size());
    for (std::size_t b = 0; b < count_g; ++b) {
        decode_row_major(b, extents_g, index);
        T weight = at(binomials, extents_g[0] - 1, index[0]);
        for (std::size_t d = 1; d < ndim; ++d) {
            weight = weight * at(binomials, extents_g[d] - 1, index[d]);
        }
        for (std::size_t s = 0; s < rank; ++s) {
            weighted_g[(b * rank) + s] = weight * g[(b * rank) + s];
        }
    }

    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = T(0);
    }

    // The convolution. The oracle adds a whole `(q + 1, ..., rank)` block per
    // multi-index of the first operand, iterating those in row-major order, so each
    // output element accumulates its terms in increasing flat index of `alpha`.
    std::vector<std::size_t> alpha(ndim);
    std::vector<std::size_t> beta(ndim);
    for (std::size_t a = 0; a < count_f; ++a) {
        decode_row_major(a, extents_f, alpha);
        for (std::size_t b = 0; b < count_g; ++b) {
            decode_row_major(b, extents_g, beta);
            std::size_t flat_h = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                flat_h += (alpha[d] + beta[d]) * strides_h[d];
            }
            for (std::size_t s = 0; s < rank; ++s) {
                const T term = weighted_f[(a * rank) + s] * weighted_g[(b * rank) + s];
                out[(flat_h * rank) + s] = out[(flat_h * rank) + s] + term;
            }
        }
    }

    // `inv_w_r[..., None] * h_conv`, applied to the finished convolution.
    for (std::size_t h = 0; h < count_h; ++h) {
        decode_row_major(h, extents_h, index);
        T weight = at(inverse_binomials, extents_h[0] - 1, index[0]);
        for (std::size_t d = 1; d < ndim; ++d) {
            weight = weight * at(inverse_binomials, extents_h[d] - 1, index[d]);
        }
        for (std::size_t s = 0; s < rank; ++s) {
            out[(h * rank) + s] = weight * out[(h * rank) + s];
        }
    }
}

/// Multiply two control nets of equal dimension and component count.
///
/// Dispatches on the dimension exactly as the oracle does: the univariate helper for
/// a curve, the tensor-product one above it. The two are different computations, so
/// the dispatch is part of the parity claim rather than an optimisation.
///
/// \param f First net, row-major with extents `shape_f`.
/// \param shape_f Extents of `f`, components last.
/// \param g Second net, row-major with extents `shape_g`.
/// \param shape_g Extents of `g`, components last.
/// \param binomials `C(n, k)` in the storage format.
/// \param inverse_binomials `1 / C(n, k)` likewise.
/// \param out_shape Receives the product's extents, components last.
/// \return The product net, row-major with extents `out_shape`.
template <Real T>
[[nodiscard]] std::vector<T> multiply_nets(std::span<const T> f,
                                           std::span<const std::size_t> shape_f,
                                           std::span<const T> g,
                                           std::span<const std::size_t> shape_g,
                                           span2d<const T> binomials,
                                           span2d<const T> inverse_binomials,
                                           std::vector<std::size_t>& out_shape) {
    const std::size_t ndim = shape_f.size() - 1;
    const std::size_t rank = shape_f.back();

    out_shape.assign(ndim + 1, rank);
    for (std::size_t d = 0; d < ndim; ++d) {
        out_shape[d] = shape_f[d] + shape_g[d] - 1;
    }

    std::vector<T> out(extent_product(out_shape, 0));
    if (ndim == 1) {
        bernstein_product_1d<T>(f, g, rank, binomials, inverse_binomials, out);
    } else {
        bernstein_product_nd<T>(f, shape_f, g, shape_g, binomials, inverse_binomials, out);
    }
    return out;
}

/// Split a homogeneous net into its numerator and its weight column.
///
/// \param values The net, row-major with `components` per coefficient.
/// \param components The stored component count, weight included.
/// \param numerator Receives the first `components - 1` of each coefficient.
/// \param denominator Receives the last one.
template <Real T>
void split_weight_column(std::span<const T> values, std::size_t components,
                         std::vector<T>& numerator, std::vector<T>& denominator) {
    const std::size_t coefficients = values.size() / components;
    numerator.assign(coefficients * (components - 1), T(0));
    denominator.assign(coefficients, T(0));
    for (std::size_t i = 0; i < coefficients; ++i) {
        for (std::size_t s = 0; s + 1 < components; ++s) {
            numerator[(i * (components - 1)) + s] = values[(i * components) + s];
        }
        denominator[i] = values[(i * components) + components - 1];
    }
}

/// A net in homogeneous form, with a unit weight column added when absent.
///
/// The oracle's `_ensure_rational_ctrl`. A promotion writes exact ones and computes
/// nothing, so it commits no rounding and sits outside the parity claim.
///
/// \param bezier The Bézier.
/// \param shape Receives the homogeneous extents, the component count last.
/// \return The homogeneous net, row-major.
template <Real T>
[[nodiscard]] std::vector<T> homogeneous_net(const Bezier<T>& bezier,
                                             std::vector<std::size_t>& shape) {
    const ControlNet<T>& net = bezier.net();
    shape.assign(net.shape().begin(), net.shape().end());
    if (bezier.is_rational()) {
        return std::vector<T>(net.values().begin(), net.values().end());
    }

    const std::size_t components = net.num_components();
    const std::size_t coefficients = net.size() / components;
    shape.back() = components + 1;

    std::vector<T> promoted(coefficients * (components + 1));
    for (std::size_t i = 0; i < coefficients; ++i) {
        for (std::size_t s = 0; s < components; ++s) {
            promoted[(i * (components + 1)) + s] = net.values()[(i * components) + s];
        }
        promoted[(i * (components + 1)) + components] = T(1);
    }
    return promoted;
}

/// A scalar-valued intermediate of the composition: one coefficient per index.
///
/// `compose` works throughout in the oracle's shape, which keeps the component axis
/// present with extent one -- `(*extents, 1)` -- because both product helpers read the
/// component count off the shape's last entry and the univariate one is handed the
/// column directly.
template <Real T>
struct ScalarNet {
    std::vector<T> values;           ///< Coefficients, row-major, one per index.
    std::vector<std::size_t> shape;  ///< Extents, with a trailing 1 for the component.
};

/// The Bernstein product of two scalar intermediates, dispatched as the oracle
/// dispatches it.
///
/// `_product_fn`: the numba kernel for a univariate inner map, the n-dimensional numpy
/// helper above that. **The two differ by more than shape** -- the kernel accumulates
/// in `accumulator_t<T>` and divides by `C(r, k)`, the helper accumulates in `T` and
/// multiplies by a precomputed reciprocal -- so the branch is part of the claim.
///
/// \param a First operand.
/// \param b Second operand, of the same dimension.
/// \param use_1d_kernel Whether the inner map is univariate.
/// \param binomials `C(n, k)` in the storage format; unused on the univariate branch,
///        which builds its own from `core::bincoeff` exactly as its oracle does.
/// \param inverse_binomials `1 / C(n, k)` likewise.
/// \return The product.
template <Real T>
[[nodiscard]] ScalarNet<T> scalar_net_product(const ScalarNet<T>& a, const ScalarNet<T>& b,
                                              bool use_1d_kernel, span2d<const T> binomials,
                                              span2d<const T> inverse_binomials) {
    ScalarNet<T> out;
    out.shape.assign(a.shape.size(), 1);
    for (std::size_t d = 0; d + 1 < a.shape.size(); ++d) {
        out.shape[d] = a.shape[d] + b.shape[d] - 1;
    }
    out.values.assign(extent_product(out.shape, 0), T(0));

    if (use_1d_kernel) {
        scalar_bernstein_product_1d<T>(a.values, b.values, out.values);
    } else {
        bernstein_product_nd<T>(a.values, a.shape, b.values, b.shape, binomials,
                                inverse_binomials, out.values);
    }
    return out;
}

/// Successive powers of a scalar intermediate: `g^1` through `g^max_power`.
///
/// `_compute_scalar_powers`, and the recurrence is the oracle's:
/// `g^k = g^{k-1} * g`, not a balanced tree. Squaring would form fewer products and a
/// different number.
///
/// \param g The base.
/// \param max_power The highest power, at least 1.
/// \param use_1d_kernel Whether the inner map is univariate.
/// \param binomials `C(n, k)` in the storage format.
/// \param inverse_binomials `1 / C(n, k)` likewise.
/// \return `max_power` nets, entry `k` holding `g^(k + 1)`.
template <Real T>
[[nodiscard]] std::vector<ScalarNet<T>> scalar_net_powers(const ScalarNet<T>& g,
                                                          std::size_t max_power,
                                                          bool use_1d_kernel,
                                                          span2d<const T> binomials,
                                                          span2d<const T> inverse_binomials) {
    std::vector<ScalarNet<T>> powers;
    powers.reserve(max_power);
    powers.push_back(g);
    for (std::size_t k = 1; k < max_power; ++k) {
        powers.push_back(scalar_net_product<T>(powers.back(), g, use_1d_kernel, binomials,
                                               inverse_binomials));
    }
    return powers;
}

/// The Bernstein basis of degree `m` evaluated at a scalar Bézier, in Bézier form.
///
/// `_compute_bernstein_bases`: `B_i^m(g) = C(m, i) g^i (1 - g)^(m - i)`, each returned
/// as a net over the inner map's parametric domain. The binomial scaling is a scalar
/// times an array in the oracle, so it rounds once per stored value and at the storage
/// width.
///
/// \param g The scalar Bézier the basis is evaluated at.
/// \param m The Bernstein degree.
/// \param use_1d_kernel Whether the inner map is univariate.
/// \param binomials `C(n, k)` in the storage format; read at row `m`.
/// \param inverse_binomials `1 / C(n, k)` likewise.
/// \return `m + 1` nets, in the order `B_0^m` through `B_m^m`.
template <Real T>
[[nodiscard]] std::vector<ScalarNet<T>> bernstein_bases_at(const ScalarNet<T>& g, std::size_t m,
                                                           bool use_1d_kernel,
                                                           span2d<const T> binomials,
                                                           span2d<const T> inverse_binomials) {
    if (m == 0) {
        // `B_0^0 = 1`: a degree-zero net, one coefficient, and the oracle's shape for
        // it is all ones rather than the inner map's own extents.
        ScalarNet<T> one;
        one.shape.assign(g.shape.size(), 1);
        one.values.assign(1, T(1));
        return {one};
    }

    ScalarNet<T> one_minus_g;
    one_minus_g.shape = g.shape;
    one_minus_g.values.resize(g.values.size());
    for (std::size_t i = 0; i < g.values.size(); ++i) {
        one_minus_g.values[i] = T(1) - g.values[i];
    }

    const std::vector<ScalarNet<T>> g_powers =
        scalar_net_powers<T>(g, m, use_1d_kernel, binomials, inverse_binomials);
    const std::vector<ScalarNet<T>> one_minus_g_powers =
        scalar_net_powers<T>(one_minus_g, m, use_1d_kernel, binomials, inverse_binomials);

    std::vector<ScalarNet<T>> bases;
    bases.reserve(m + 1);
    bases.push_back(one_minus_g_powers[m - 1]);
    for (std::size_t i = 1; i < m; ++i) {
        ScalarNet<T> scaled = scalar_net_product<T>(g_powers[i - 1], one_minus_g_powers[m - i - 1],
                                                    use_1d_kernel, binomials, inverse_binomials);
        // Read from the table rather than recomputed, and the oracle reaches the same
        // number by a third spelling -- `float(math.comb(m, i))` applied to an array.
        // That the two agree is the double-rounding argument `_binomial_tables`'
        // docstring makes and `test_the_two_spellings_of_a_binomial_agree` checks; the
        // reliance is here, so the pointer is here.
        const T coefficient = at(binomials, m, i);
        for (T& value : scaled.values) {
            value = coefficient * value;
        }
        bases.push_back(std::move(scaled));
    }
    bases.push_back(g_powers[m - 1]);
    return bases;
}

/// One component of a control net, keeping the component axis at extent one.
///
/// The oracle's `_extract_scalar_component`, which is a slice and computes nothing.
///
/// \param net The net.
/// \param component The component index, below `net.num_components()`.
/// \return The scalar net.
template <Real T>
[[nodiscard]] ScalarNet<T> scalar_component(const ControlNet<T>& net, std::size_t component) {
    const std::size_t components = net.num_components();
    const std::size_t coefficients = net.size() / components;

    ScalarNet<T> out;
    out.shape.assign(net.shape().begin(), net.shape().end());
    out.shape.back() = 1;
    out.values.resize(coefficients);
    for (std::size_t i = 0; i < coefficients; ++i) {
        out.values[i] = net.values()[(i * components) + component];
    }
    return out;
}

}  // namespace detail

/// The largest upper binomial index a product forms.
///
/// The size the tables `multiply` needs, defined once so that a caller assembling
/// them in another language does not have to restate the formula. It is
/// `max_d (p_d + q_d)`: the product's own degree per direction, and no reciprocal or
/// coefficient above it is ever read.
///
/// \param a First operand.
/// \param b Second operand.
/// \return The largest `n` for which `C(n, k)` will be read, so the tables must be
///         at least `(n + 1, n + 1)`.
/// \throws std::invalid_argument If the operands are incompatible, with the same
///         message `multiply` would give.
template <Real T>
[[nodiscard]] std::size_t product_table_order(const Bezier<T>& a, const Bezier<T>& b) {
    detail::require_compatible_operands(a, b);

    std::size_t order = 0;
    for (std::size_t d = 0; d < a.dim(); ++d) {
        const std::size_t summed = a.degree(d) + b.degree(d);
        order = (summed > order) ? summed : order;
    }
    return order;
}

/// The pointwise product of two Béziers.
///
/// Degree `p_d + q_d` per direction, which represents the product exactly; the
/// coefficients carry the roundings of a binomial-weighted sum. See the file comment.
///
/// A rational operand makes the result rational: both operands are promoted to
/// homogeneous form and their numerators and weight columns are multiplied
/// independently, which is the product of the two projected mappings because the
/// projection is a quotient.
///
/// \param a First operand.
/// \param b Second operand. Must match `a`'s dimension and rank.
/// \param binomials `C(n, k)` in the storage format, at least
///        `(product_table_order + 1)` square. Entries with `k > n` are never read.
/// \param inverse_binomials `1 / C(n, k)` in the storage format, same shape.
/// \return The product, rational exactly when either operand is.
/// \throws std::invalid_argument If the dimensions or ranks differ, or if a table is
///         too small.
template <Real T>
[[nodiscard]] Bezier<T> multiply(const Bezier<T>& a, const Bezier<T>& b,
                                 span2d<const T> binomials,
                                 span2d<const T> inverse_binomials) {
    const std::size_t order = product_table_order<T>(a, b);
    detail::require_table_extents<T>(binomials, order, "binomials");
    detail::require_table_extents<T>(inverse_binomials, order, "inverse_binomials");

    std::vector<std::size_t> out_shape;

    if (!a.is_rational() && !b.is_rational()) {
        const std::vector<T> values = detail::multiply_nets<T>(
            a.net().values(), a.net().shape(), b.net().values(), b.net().shape(), binomials,
            inverse_binomials, out_shape);
        return Bezier<T>(ControlNet<T>(std::span<const T>(values),
                                       std::span<const std::size_t>(out_shape)),
                         false);
    }

    std::vector<std::size_t> shape_a;
    std::vector<std::size_t> shape_b;
    const std::vector<T> homogeneous_a = detail::homogeneous_net<T>(a, shape_a);
    const std::vector<T> homogeneous_b = detail::homogeneous_net<T>(b, shape_b);

    std::vector<T> numerator_a;
    std::vector<T> denominator_a;
    std::vector<T> numerator_b;
    std::vector<T> denominator_b;
    detail::split_weight_column<T>(homogeneous_a, shape_a.back(), numerator_a, denominator_a);
    detail::split_weight_column<T>(homogeneous_b, shape_b.back(), numerator_b, denominator_b);

    std::vector<std::size_t> numerator_shape_a(shape_a);
    std::vector<std::size_t> numerator_shape_b(shape_b);
    numerator_shape_a.back() -= 1;
    numerator_shape_b.back() -= 1;
    std::vector<std::size_t> weight_shape_a(shape_a);
    std::vector<std::size_t> weight_shape_b(shape_b);
    weight_shape_a.back() = 1;
    weight_shape_b.back() = 1;

    std::vector<std::size_t> numerator_out_shape;
    const std::vector<T> product_numerator = detail::multiply_nets<T>(
        numerator_a, numerator_shape_a, numerator_b, numerator_shape_b, binomials,
        inverse_binomials, numerator_out_shape);
    const std::vector<T> product_weights =
        detail::multiply_nets<T>(denominator_a, weight_shape_a, denominator_b, weight_shape_b,
                                 binomials, inverse_binomials, out_shape);

    // `np.concatenate([h_num, h_den], axis=-1)`.
    const std::size_t rank = numerator_out_shape.back();
    const std::size_t coefficients = product_weights.size();
    out_shape.back() = rank + 1;

    std::vector<T> values(coefficients * (rank + 1));
    for (std::size_t i = 0; i < coefficients; ++i) {
        for (std::size_t s = 0; s < rank; ++s) {
            values[(i * (rank + 1)) + s] = product_numerator[(i * rank) + s];
        }
        values[(i * (rank + 1)) + rank] = product_weights[i];
    }
    return Bezier<T>(
        ControlNet<T>(std::span<const T>(values), std::span<const std::size_t>(out_shape)), true);
}

/// The largest upper binomial index a composition forms.
///
/// Two sources, and the second is conditional:
///
///  - the Bernstein basis of each outer direction needs `C(m_d, i)`, so `max_d m_d`;
///  - an **n-dimensional** inner map routes every product through
///    `bernstein_product_nd`, whose reciprocals reach the composed degree
///    `sum_d m_d * n_s` in direction `s`.
///
/// A **univariate** inner map contributes nothing to the second, because its products
/// go through `scalar_bernstein_product_1d`, which builds its own coefficients from
/// `core::bincoeff`. Including it anyway would be an upper bound rather than a wrong
/// answer, and it is left out for a concrete reason: a degree-1 outer map composed with
/// a very high degree inner one is exempt from the envelope check -- no product is
/// formed at all -- so the composed degree there is unbounded, and asking a caller to
/// tabulate `C(n, k)` up to it would refuse a case both backends handle.
///
/// Unlike `product_table_order` this validates nothing. It reads only each operand's
/// own degrees, so no compatibility between them is required to answer.
///
/// \param outer The outer map.
/// \param inner The inner map.
/// \return The largest `n` for which `C(n, k)` will be read, so the tables must be at
///         least `(n + 1, n + 1)`.
template <Real T>
[[nodiscard]] std::size_t composition_table_order(const Bezier<T>& outer,
                                                  const Bezier<T>& inner) {
    std::size_t order = 0;
    std::size_t total = 0;
    for (std::size_t d = 0; d < outer.dim(); ++d) {
        const std::size_t degree = outer.degree(d);
        order = (degree > order) ? degree : order;
        total += degree;
    }
    if (inner.dim() > 1) {
        for (std::size_t s = 0; s < inner.dim(); ++s) {
            const std::size_t composed = total * inner.degree(s);
            order = (composed > order) ? composed : order;
        }
    }
    return order;
}

/// The composition `outer(inner(t))`.
///
/// Degree `sum_d m_d * n_s` in direction `s`, which represents the composition
/// exactly; the coefficients carry the roundings of the products that build them. See
/// the file comment.
///
/// Each outer direction's Bernstein basis is evaluated at the corresponding component
/// of the inner map, giving `m_d + 1` nets over the inner map's parametric domain;
/// the outer control points then weight the tensor products of those nets.
///
/// \param outer The outer map. Must be non-rational.
/// \param inner The inner map. Must be non-rational and satisfy
///        `inner.rank() == outer.dim()`.
/// \param binomials `C(n, k)` in the storage format, at least
///        `(composition_table_order + 1)` square.
/// \param inverse_binomials `1 / C(n, k)` in the storage format, same shape.
/// \return The composed Bézier, of dimension `inner.dim()` and rank `outer.rank()`.
/// \throws std::invalid_argument If either operand is rational, if the inner map's rank
///         is not the outer map's dimension, if a univariate inner map would drive the
///         composed degree past `core::kBincoeffMaxN`, or if a table is too small.
template <Real T>
[[nodiscard]] Bezier<T> compose(const Bezier<T>& outer, const Bezier<T>& inner,
                                span2d<const T> binomials,
                                span2d<const T> inverse_binomials) {
    // The oracle's order and the oracle's messages, character for character. The two
    // rational rejections are its `TypeError`; see the file comment for why they are an
    // `std::invalid_argument` here without breaking parity.
    if (outer.is_rational()) {
        throw std::invalid_argument(
            "Composition is not supported for rational Béziers (outer is rational).");
    }
    if (inner.is_rational()) {
        throw std::invalid_argument(
            "Composition is not supported for rational Béziers (inner is rational).");
    }
    if (inner.rank() != outer.dim()) {
        throw std::invalid_argument("Inner Bézier rank (" + std::to_string(inner.rank())
                                    + ") must equal outer Bézier parametric dimension ("
                                    + std::to_string(outer.dim()) + ").");
    }

    const std::size_t dim_outer = outer.dim();
    const std::size_t dim_inner = inner.dim();
    const bool use_1d_kernel = dim_inner == 1;

    std::size_t total_degree = 0;
    for (std::size_t d = 0; d < dim_outer; ++d) {
        total_degree += outer.degree(d);
    }

    // The oracle's exemptions, reproduced with its reasoning: only a univariate inner
    // map reaches `scalar_bernstein_product_1d`, and a univariate outer map of degree
    // at most one forms no product at all, so `core::bincoeff` is never called. The
    // message is `_check_bincoeff_envelope`'s.
    if (use_1d_kernel && (dim_outer > 1 || outer.degree(0) > 1)) {
        const std::size_t composed = total_degree * inner.degree(0);
        detail::require_bincoeff_envelope(composed, "Composition to degree "
                                                        + std::to_string(composed)
                                                        + " with a 1D inner Bézier");
    }

    const std::size_t order = composition_table_order<T>(outer, inner);
    detail::require_table_extents<T>(binomials, order, "binomials");
    detail::require_table_extents<T>(inverse_binomials, order, "inverse_binomials");

    std::vector<std::vector<detail::ScalarNet<T>>> all_bases;
    all_bases.reserve(dim_outer);
    for (std::size_t d = 0; d < dim_outer; ++d) {
        all_bases.push_back(detail::bernstein_bases_at<T>(
            detail::scalar_component<T>(inner.net(), d), outer.degree(d), use_1d_kernel,
            binomials, inverse_binomials));
    }

    const std::size_t rank = outer.net().num_components();
    std::vector<std::size_t> result_shape(dim_inner + 1, rank);
    for (std::size_t s = 0; s < dim_inner; ++s) {
        result_shape[s] = (total_degree * inner.degree(s)) + 1;
    }
    const std::size_t coefficients = detail::extent_product(result_shape, 0) / rank;
    std::vector<T> result(coefficients * rank, T(0));

    std::vector<std::size_t> outer_extents(dim_outer);
    for (std::size_t d = 0; d < dim_outer; ++d) {
        outer_extents[d] = outer.degree(d) + 1;
    }
    const std::size_t outer_count = detail::extent_product(outer_extents, 0);

    std::vector<std::size_t> multi_index(dim_outer);
    for (std::size_t c = 0; c < outer_count; ++c) {
        detail::decode_row_major(c, outer_extents, multi_index);

        detail::ScalarNet<T> basis = all_bases[0][multi_index[0]];
        for (std::size_t d = 1; d < dim_outer; ++d) {
            basis = detail::scalar_net_product<T>(basis, all_bases[d][multi_index[d]],
                                                  use_1d_kernel, binomials, inverse_binomials);
        }

        // `result_ctrl += coef * basis`, with `coef` of shape `(rank,)` broadcast
        // against a basis whose component axis has extent one.
        const std::span<const T> outer_values = outer.net().values();
        for (std::size_t h = 0; h < coefficients; ++h) {
            for (std::size_t s = 0; s < rank; ++s) {
                const T weighted = outer_values[(c * rank) + s] * basis.values[h];
                result[(h * rank) + s] = result[(h * rank) + s] + weighted;
            }
        }
    }

    return Bezier<T>(ControlNet<T>(std::span<const T>(result),
                                   std::span<const std::size_t>(result_shape)),
                     false);
}

}  // namespace pantr::bezier
