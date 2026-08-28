/// \file
/// nanobind bindings for the `pantr.bezier` root-finding kernels.
///
/// Six entry points, which are exactly the seam `src/pantr/bezier/_find_roots.py`
/// reaches for. The sixteen kernels behind them are not bound and cannot be: they
/// call each other from inside `nopython` code on the Numba side, and no dispatch
/// can be inserted between two Numba kernels, so the boundary is forced up to here.
/// That is also why `_root_finding_core.py` keeps its kernels and its module path
/// untouched by this port, including the one a downstream consumer imports.
///
/// ## These kernels return a count, and that is deliberate
///
/// `bezier.cpp`'s eight fill the caller's buffer and return `None`. These fill a
/// buffer and return **how much of it is valid**, because the answer's size is not
/// a function of the input: a degree-`n` polynomial has anywhere from zero to `n`
/// roots and the caller cannot size the result in advance. Returning the count is
/// the alternative to returning a fresh array, and it keeps the allocation with the
/// caller where the rest of the tree keeps it.
///
/// The Numba side returns `(array, count)` instead. `_root_finding_backend.py`
/// adapts, in three lines per kernel, rather than the oracle being reshaped to
/// match: the oracle is what parity is measured against and it was cross-checked
/// bit for bit before any of this existed, so changing it to suit the binding would
/// move the thing being measured.
///
/// ## What the checks are for
///
/// nanobind settles dtype, rank and contiguity. What it cannot express is that
/// `out` must be long enough for the *worst case* of a search whose result size is
/// unknown, and each bound below is the kernel's own worst case rather than a
/// guess: `degree` roots from Yuksel, `3 * degree + 4` from clipping before the
/// duplicates are merged, `n_roots` from the merge itself.
///
/// **An empty coefficient array is undefined, not merely wrong.** Every kernel here
/// takes `degree` as `coeff.size() - 1`, and the clipping driver indexes
/// `root_coeff[0]` before any loop bound could stop it. Same class as the floor
/// `bezier.cpp` documents, reached by a different route.
///
/// **A non-positive or non-finite tolerance is refused**, matching
/// `_find_roots._resolve_tol`, which a direct call on the extension bypasses. The
/// iteration terminates on its depth cap regardless, so this is a contract check
/// rather than a safety one, and it is here so the two backends refuse the same
/// inputs rather than one of them returning an answer to a question that was not
/// asked.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "pantr/bezier/root_finding.hpp"
#include "pantr/core/mdspan.hpp"
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

/// Raise unless `coeff` holds at least one coefficient.
///
/// The degree is inferred as `size - 1`, and the clipping driver dereferences
/// `coeff[0]` before any loop bound could stop it.
template <class Arr>
void require_coefficients(const Arr& coeff, const char* what) {
    if (coeff.shape(0) == 0) {
        throw nb::value_error(
            (std::string(what) + " is empty, but its degree is inferred as its length minus one, "
                                 "which needs at least one coefficient")
                .c_str());
    }
}

/// Raise unless `out` can hold `needed` entries along `axis`.
///
/// A lower bound rather than an equality: the caller sizes the row by the degree
/// while a kernel's worst case can be larger, and refusing a buffer that is merely
/// generous would be wrong.
///
/// The axis is explicit because getting it wrong is silent. An earlier version took
/// `shape(0)` implicitly, which is the right axis for the 1-D outputs and the wrong
/// one for `find_roots_batch`'s 2-D `out_roots`, whose first axis is the polynomial
/// count and whose **second** is the row the kernel writes into. The check then
/// duplicated a constraint already enforced two lines above, the row width went
/// unchecked, and an undersized buffer was accepted with the per-row count silently
/// clamped to whatever fitted: a batch with two roots reported one, and with a
/// zero-width row reported none, both without an error.
template <class Arr>
void require_capacity(const Arr& out, std::size_t axis, std::size_t needed, const char* what) {
    if (out.shape(axis) < needed) {
        throw nb::value_error((std::string(what) + " holds " +
                               std::to_string(out.shape(axis)) + " entries along axis " +
                               std::to_string(axis) + ", but this call can produce " +
                               std::to_string(needed))
                                  .c_str());
    }
}

/// Raise unless `tol` is finite and strictly positive.
///
/// Mirrors `_find_roots._resolve_tol`, which a direct call on the extension skips.
/// Written as a conjunction so a NaN is refused rather than admitted.
void require_tolerance(double tol, const char* what) {
    if (!(tol > 0.0 && std::isfinite(tol))) {
        throw nb::value_error(
            (std::string(what) + " must be finite and positive, got " + std::to_string(tol))
                .c_str());
    }
}

/// The largest number of roots Yuksel's decomposition can report.
std::size_t yuksel_capacity(std::size_t coeff_count) {
    return coeff_count > 1 ? coeff_count - 1 : 1;
}

/// The largest number of candidates Bézier clipping can report before the merge.
std::size_t clip_capacity(std::size_t coeff_count) {
    return 3 * (coeff_count - 1) + 4;
}

template <class T>
int bind_yuksel_roots(const_vec<T> coeff, double param_tol, out_vec<double> out) {
    require_coefficients(coeff, "coeff");
    require_tolerance(param_tol, "param_tol");
    require_capacity(out, 0, yuksel_capacity(coeff.shape(0)), "out");
    return pantr::bezier::yuksel_roots<T>(
        std::span<const T>(coeff.data(), coeff.shape(0)), param_tol,
        std::span<double>(out.data(), out.shape(0)));
}

template <class T>
int bind_clip_roots(const_vec<T> coeff, double param_tol, double geom_tol, out_vec<double> out) {
    require_coefficients(coeff, "coeff");
    require_tolerance(param_tol, "param_tol");
    require_tolerance(geom_tol, "geom_tol");
    require_capacity(out, 0, clip_capacity(coeff.shape(0)), "out");
    return pantr::bezier::clip_roots<T>(std::span<const T>(coeff.data(), coeff.shape(0)), param_tol,
                                        geom_tol, std::span<double>(out.data(), out.shape(0)));
}

template <class T>
int bind_dedup_roots(const_vec<T> coeff, const_vec<double> raw_roots, int n_roots,
                     double param_tol, double geom_tol, out_vec<double> out) {
    require_coefficients(coeff, "coeff");
    require_tolerance(param_tol, "param_tol");
    require_tolerance(geom_tol, "geom_tol");
    if (n_roots < 0 || static_cast<std::size_t>(n_roots) > raw_roots.shape(0)) {
        throw nb::value_error(("n_roots is " + std::to_string(n_roots) +
                               ", which is not a valid count for a raw_roots of length " +
                               std::to_string(raw_roots.shape(0)))
                                  .c_str());
    }
    require_capacity(out, 0, static_cast<std::size_t>(n_roots), "out");
    return pantr::bezier::dedup_roots<T>(
        std::span<const double>(raw_roots.data(), raw_roots.shape(0)), n_roots,
        std::span<const T>(coeff.data(), coeff.shape(0)), param_tol, geom_tol,
        std::span<double>(out.data(), out.shape(0)));
}

template <class T>
double bind_solve_monotone_root(const_vec<T> coeff, double param_tol) {
    require_coefficients(coeff, "coeff");
    require_tolerance(param_tol, "param_tol");
    return pantr::bezier::solve_monotone_root<T>(std::span<const T>(coeff.data(), coeff.shape(0)),
                                                 param_tol);
}

template <class T>
void bind_find_roots_batch(const_mat<T> coeffs, double param_tol, double geom_tol,
                           out_mat<double> out_roots, out_vec<std::int64_t> out_counts) {
    require_tolerance(param_tol, "param_tol");
    require_tolerance(geom_tol, "geom_tol");
    if (coeffs.shape(1) == 0) {
        throw nb::value_error("coeffs has no columns, but each row's degree is inferred as its "
                              "length minus one, which needs at least one coefficient");
    }
    if (out_roots.shape(0) != coeffs.shape(0) || out_counts.shape(0) != coeffs.shape(0)) {
        throw nb::value_error(("out_roots and out_counts must have " +
                               std::to_string(coeffs.shape(0)) + " rows, got " +
                               std::to_string(out_roots.shape(0)) + " and " +
                               std::to_string(out_counts.shape(0)))
                                  .c_str());
    }
    // The row the kernel writes into. A degree-n polynomial has at most n roots,
    // and Layer 2 allocates max(n, 1) so a degree-0 batch still has somewhere to
    // write nothing.
    require_capacity(out_roots, 1, yuksel_capacity(coeffs.shape(1)), "out_roots");
    pantr::bezier::find_roots_batch<T>(
        pantr::span2d<const T>(coeffs.data(), coeffs.shape(0), coeffs.shape(1)), param_tol,
        geom_tol, pantr::span2d<double>(out_roots.data(), out_roots.shape(0), out_roots.shape(1)),
        std::span<std::int64_t>(out_counts.data(), out_counts.shape(0)));
}

template <class T>
void bind_solve_monotone_root_batch(const_mat<T> coeffs, double param_tol,
                                    out_vec<double> out_roots) {
    require_tolerance(param_tol, "param_tol");
    if (coeffs.shape(1) == 0) {
        throw nb::value_error("coeffs has no columns, but each row's degree is inferred as its "
                              "length minus one, which needs at least one coefficient");
    }
    if (out_roots.shape(0) != coeffs.shape(0)) {
        throw nb::value_error(("out_roots must have " + std::to_string(coeffs.shape(0)) +
                               " entries, got " + std::to_string(out_roots.shape(0)))
                                  .c_str());
    }
    pantr::bezier::solve_monotone_root_batch<T>(
        pantr::span2d<const T>(coeffs.data(), coeffs.shape(0), coeffs.shape(1)), param_tol,
        std::span<double>(out_roots.data(), out_roots.shape(0)));
}

}  // namespace

void register_bezier_root_finding(nb::module_& m) {
    // `.noconvert()` and keyword-only outputs, for the reasons `basis.cpp` sets out
    // and `bezier.cpp` repeats: without `.noconvert()` nanobind can satisfy a
    // contiguity constraint by filling a temporary and discarding it, so an
    // out-parameter silently does nothing.
    //
    // Two traps closed by the signature rather than by a comment, both found by an
    // audit of this surface against its three siblings.
    //
    // `param_tol` and `geom_tol` are adjacent, same-typed, and nothing orders them,
    // so `require_ordered_bounds` -- the guard `bezier.cpp` uses for `lower`/`upper`
    // -- has no analogue here. Transposed, they return a different and plausible
    // root set. An earlier version left them positional on the grounds that the
    // Numba kernels take them so; that does not bind this signature, since the
    // catalogue already adapts the return shape and can adapt the call too.
    //
    // `dedup_roots` takes the coefficients **first**, as every sibling in this file
    // and every accessor in the four catalogues does. Its `raw_roots` is always
    // `float64` and `coeff` frequently is, so for the `double` overload the two are
    // indistinguishable to the caster: a transposed call type-checked, ran, and
    // merged against the wrong data, returning a coefficient rather than a root. A
    // comment saying so was the whole guard and did not hold. `nb::kw_only()` now
    // sits directly after `coeff`, giving this binding the same shape as
    // `clip_roots` -- one positional argument and nothing else sayable by position.
    m.def("yuksel_roots", &bind_yuksel_roots<double>, nb::arg("coeff").noconvert(),
          nb::arg("param_tol"), nb::kw_only(), nb::arg("out").noconvert());
    m.def("yuksel_roots", &bind_yuksel_roots<float>, nb::arg("coeff").noconvert(),
          nb::arg("param_tol"), nb::kw_only(), nb::arg("out").noconvert());

    m.def("clip_roots", &bind_clip_roots<double>, nb::arg("coeff").noconvert(), nb::kw_only(),
          nb::arg("param_tol"), nb::arg("geom_tol"), nb::arg("out").noconvert());
    m.def("clip_roots", &bind_clip_roots<float>, nb::arg("coeff").noconvert(), nb::kw_only(),
          nb::arg("param_tol"), nb::arg("geom_tol"), nb::arg("out").noconvert());

    m.def("dedup_roots", &bind_dedup_roots<double>, nb::arg("coeff").noconvert(), nb::kw_only(),
          nb::arg("raw_roots").noconvert(), nb::arg("n_roots"), nb::arg("param_tol"),
          nb::arg("geom_tol"), nb::arg("out").noconvert());
    m.def("dedup_roots", &bind_dedup_roots<float>, nb::arg("coeff").noconvert(), nb::kw_only(),
          nb::arg("raw_roots").noconvert(), nb::arg("n_roots"), nb::arg("param_tol"),
          nb::arg("geom_tol"), nb::arg("out").noconvert());

    m.def("solve_monotone_root", &bind_solve_monotone_root<double>, nb::arg("coeff").noconvert(),
          nb::arg("param_tol"));
    m.def("solve_monotone_root", &bind_solve_monotone_root<float>, nb::arg("coeff").noconvert(),
          nb::arg("param_tol"));

    m.def("find_roots_batch", &bind_find_roots_batch<double>, nb::arg("coeffs").noconvert(),
          nb::kw_only(), nb::arg("param_tol"), nb::arg("geom_tol"),
          nb::arg("out_roots").noconvert(), nb::arg("out_counts").noconvert());
    m.def("find_roots_batch", &bind_find_roots_batch<float>, nb::arg("coeffs").noconvert(),
          nb::kw_only(), nb::arg("param_tol"), nb::arg("geom_tol"),
          nb::arg("out_roots").noconvert(), nb::arg("out_counts").noconvert());

    m.def("solve_monotone_root_batch", &bind_solve_monotone_root_batch<double>,
          nb::arg("coeffs").noconvert(), nb::arg("param_tol"), nb::kw_only(),
          nb::arg("out_roots").noconvert());
    m.def("solve_monotone_root_batch", &bind_solve_monotone_root_batch<float>,
          nb::arg("coeffs").noconvert(), nb::arg("param_tol"), nb::kw_only(),
          nb::arg("out_roots").noconvert());
}
