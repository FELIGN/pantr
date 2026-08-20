#pragma once

/// \file
/// Cox-de Boor tabulation of the cardinal B-spline basis on the central span.
///
/// Port of `_tabulate_cardinal_Bspline_basis_1D_core` in
/// `src/pantr/basis/_basis_core.py`, which stays as the parity oracle. The
/// recurrence is unchanged; what is new is that the accumulation type is stated
/// rather than inherited from a promotion rule (see below).
///
/// ## Why this kernel and not another
///
/// It is the whole of the C++ in this prototype, chosen so the infrastructure is
/// exercised without numerical risk riding along:
///
///   - no linear solve and no BLAS, so nothing here depends on which LAPACK the
///     machine links against;
///   - no transcendental, so no libm difference between platforms;
///   - no domain limit on the kernel -- the degree ceilings in
///     `pantr.change_basis` belong to the builders that solve a Gram system, not
///     to this tabulation;
///   - exactly one site of the form `a * b + c`, so the set of places the two
///     backends could ever disagree is one, and it can be read off the source.
///     Whether that site actually fuses is a property of the BUILD, not of the
///     kernel: see below.
///
/// ## The recurrence, and why its error analysis is easy
///
/// For `u` in `[0, 1]` and the central span of a uniform knot vector with unit
/// spacing, `term = (r + 1 - u) / k` lies in `[0, 1]`, so `term` and `1 - term`
/// are both non-negative and each stage forms a *convex combination* of the
/// previous stage's values. Nothing here can cancel.
///
/// That is what makes the parity bound derivable rather than fitted: a relative
/// error bound on two non-negative summands is inherited by their sum, so
/// relative errors propagate additively through the stages instead of being
/// amplified. The bound itself is derived in
/// tests/parity/test_basis_cardinal_bspline.py, next to the assertion that uses
/// it.
///
/// ## Accumulation type: stated, not inherited
///
/// **Verified against numba 0.65.1**: the Python kernel's intermediates are
/// float64 even when the arrays are float32. `1.0 - u` puts a Python float
/// literal next to a `float32`, and numba's type unification promotes the
/// result to `float64` -- `probe.nopython_signatures` reports
/// `(float32,) -> UniTuple(float64 x 3)` for the expression in isolation. So
/// `one_minus_u`, `inv_k`, `term` and the running `saved` are all float64 there,
/// and only the stores into `out` round to float32.
///
/// That is the right arithmetic and the wrong reason: it is a consequence of
/// literal typing, not a decision. `pantr::accumulator_t`
/// (cpp/include/pantr/core/scalar.hpp) writes the decision down, as a
/// project-wide policy rather than as a property of this kernel.
///
/// What justifies it HERE is parity, not accuracy. A `float` instantiation
/// computed entirely in `float` would be a faithful-looking port that disagrees
/// with the oracle by about one float32 ulp, and the disagreement would be read
/// as a bug in the port. The accuracy argument of design/large_data_fitting.md
/// is a different one and does not reach this kernel -- the trait's own
/// documentation says why, and that note was amended in this branch to agree.

#include <cstddef>
#include <span>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr {

/// Tabulate the cardinal B-spline basis of `degree` at every point of `points`.
///
/// Evaluates the `degree + 1` basis functions active over the central unit span
/// `[0, 1]` of a uniform knot vector with unit spacing, using the Cox-de Boor
/// recurrence specialised to span index 0.
///
/// \param degree Degree of the basis. Must be non-negative.
/// \param points Evaluation points. Values outside `[0, 1]` are evaluated by the
///        same recurrence rather than clamped, matching the Python kernel.
/// \param out Output view of shape `(points.size(), degree + 1)`, written in
///        full.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md, and every guarantee it relies on is established
///       by its Layer 2 caller. Two of them are load-bearing for memory safety
///       rather than merely for the answer: `degree >= 0`, because it is widened
///       to `std::size_t` below and a negative value becomes a loop bound of
///       `SIZE_MAX`; and `out` having exactly `points.size()` rows and
///       `degree + 1` columns, because nothing here bounds-checks a write. Both
///       are checked by `tabulate` in cpp/bindings/pantr_cpp.cpp, which is the
///       Layer 2 caller on the Python side; a C++ caller establishes them
///       itself. Non-finite points are NOT a precondition -- they propagate into
///       non-finite outputs, which is an answer rather than undefined behaviour.
///       For general use call `pantr.basis.tabulate_cardinal_bspline_1d`.
template <Real T>
void tabulate_cardinal_bspline_1d(int degree, std::span<const T> points, span2d<T> out) {
    using Acc = accumulator_t<T>;

    const std::size_t num_pts = points.size();

    if (degree == 0) {
        // The basis is the single constant function.
        for (std::size_t j = 0; j < num_pts; ++j) {
            at(out, j, 0) = T(1);
        }
        return;
    }

    const auto n = static_cast<std::size_t>(degree);

    for (std::size_t j = 0; j < num_pts; ++j) {
        at(out, j, 0) = T(1);

        const Acc u = static_cast<Acc>(points[j]);
        const Acc one_minus_u = Acc(1) - u;

        for (std::size_t k = 1; k <= n; ++k) {
            const Acc inv_k = Acc(1) / Acc(static_cast<double>(k));
            Acc saved = Acc(0);

            for (std::size_t r = 0; r < k; ++r) {
                const Acc nr_old = static_cast<Acc>(at(out, j, r));
                const Acc term = (Acc(static_cast<double>(r)) + one_minus_u) * inv_k;

                // The only `a * b + c` in the kernel, and so the only site where
                // the two backends can disagree at all.
                //
                // Whether it FUSES is decided by the build, not by this line, and
                // both answers are measured rather than assumed:
                //
                //   as shipped   no -march is set (see cmake/PantrCompileOptions
                //                .cmake for why that is deliberate), so the target
                //                is baseline x86-64, which has no FMA instruction.
                //                -ffp-contract=on cannot fuse what the ISA cannot
                //                execute. Verified: `objdump -d` over every object
                //                either preset produces contains no vfmadd, and
                //                this line compiles to mulsd + addsd on GCC 14 and
                //                on Clang 18 alike.
                //   with -mfma   it fuses, and ONLY here -- neither
                //                `nr_old * (1 - term)` below nor `term` above is
                //                in a*b+c form, so neither can.
                //
                // numba emits it unfused in both cases (verified in its LLVM IR:
                // fmul then fadd, never fmuladd). So parity is bit-exact on the
                // shipped build and becomes a one-rounding-per-stage question the
                // day design/simd.md's stage 2 turns the ISA ladder on.
                // tests/parity/test_basis_cardinal_bspline.py selects between
                // the two by reading `pantr._pantr_cpp.__fp_contract__`, and
                // derives the bound for the second.
                at(out, j, r) = static_cast<T>(saved + nr_old * term);

                saved = nr_old * (Acc(1) - term);
            }

            at(out, j, k) = static_cast<T>(saved);
        }
    }
}

}  // namespace pantr
