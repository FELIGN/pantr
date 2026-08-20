/// \file
/// nanobind bindings for the pantr C++ prototype.
///
/// ## No docstrings here
///
/// Every user-visible docstring lives in the Python layer -- for this kernel,
/// `src/pantr/basis/` -- and not in a string literal in this file, and that is
/// deliberate. A docstring written in
/// C++ is invisible to ruff's pydocstyle rules, to the doctest runner, and to the
/// docs build, so the project's documentation conventions -- which CLAUDE.md
/// states in full and enforces in CI -- would silently stop applying to exactly
/// the functions a user calls. The thin Python layer owns the contract; this file
/// owns the call.
///
/// ## What the nanobind version question looks like from here
///
/// design/_memory/nanobind-status.md lists the API changes in nanobind 3 that
/// touch decisions already taken: `NB_TRAMPOLINE` losing its `Size` argument,
/// return-value policies and argument annotations becoming compile-time tags,
/// and `nb::gil_scoped_acquire` gaining an `is_valid()` that the callback and
/// plugin seams must consult.
///
/// **None of the three has a site in this file**, and saying so is more useful
/// than pretending otherwise: there is no trampoline because nothing here is a
/// virtual class, no return-value policy because both functions return `void`,
/// and no `gil_scoped_acquire` because nothing calls back into Python.
///
/// The annotations are the one that needs care, because the conclusion is right
/// and the obvious reason for it is wrong. `nb::arg` is *not* a compile-time tag
/// under both: it becomes one in 3.x. Read from the two headers --
///
///   2.14.0  `struct arg` is a plain runtime struct. `noconvert()` clears a
///           `uint8_t convert_` member ON the object and returns `arg&`.
///   3.0.0   `using arg = arg_t<Flags, Locked>`, and `noconvert()` is
///           `constexpr` and returns a DIFFERENT TYPE,
///           `arg_t<Flags & ~convert, Locked>`. The flag moved out of the value
///           and into the type.
///
/// What makes this file portable is therefore not that the two are the same kind
/// of object, but that the SPELLING `nb::arg("out").noconvert()` is valid in
/// both and yields something `m.def` accepts in both.
///
/// So this file compiles unchanged against 2.15.0 and 3.0.0.dev2, and the
/// version choice is decided by what is stable rather than by what the source
/// needs. scripts/ci_local.sh builds it against both.
///
/// The GIL *is* released around the kernel, which is the one threading-adjacent
/// decision this file does make: the kernel touches no Python object, so holding
/// the GIL through it would serialise any caller that threads at the Python
/// level for no reason.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <limits>
#include <span>
#include <string>

#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/core/mdspan.hpp"

namespace nb = nanobind;

namespace {

/// The array shapes the kernel accepts, expressed in the type system.
///
/// nanobind checks dtype, rank, C-contiguity and device before the function body
/// runs and raises a `TypeError` otherwise, so these aliases *are* the input
/// validation for everything they can express. What they cannot express is a
/// relation BETWEEN two arguments -- `out.shape(0) == points.size()` and
/// `out.shape(1) == degree + 1` -- and that is checked in the function body
/// below.
template <class T>
using const_points = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using out_matrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// Tabulate into `out`, after checking what the kernel assumes and never checks.
///
/// **This function is Layer 2's C++ half**, and the checks below belong here for
/// the reason CLAUDE.md gives: a Layer 3 kernel validates nothing, so every
/// guarantee it relies on is established by its caller. The adapter in
/// `pantr.basis._basis_backend` is the Python half of the same layer, but it is
/// not the only caller -- the extension
/// is importable, and `pantr._pantr_cpp.tabulate_cardinal_bspline_1d` is a public
/// attribute of a public module. Measured before these checks existed, one line
/// of Python reached undefined behaviour twice over: a negative `degree` became
/// `SIZE_MAX` at `static_cast<std::size_t>(degree)` and exited with SIGSEGV, and
/// an `out` too small for the degree overran the allocation and corrupted the
/// heap. Neither raised anything a caller could catch.
///
/// `degree` is `unsigned` rather than `int` so that the negative case never
/// reaches this body at all: nanobind's own caster rejects it, raising a
/// `TypeError` before any of pantr's code runs. The kernel's parameter stays
/// `int` -- it is a discrete-structure parameter, and design/automatic
/// _differentiation.md's fourth discipline keeps those as plain integers -- so
/// the one range that `unsigned` admits and `int` does not is refused explicitly
/// rather than wrapped around by the cast.
///
/// The cost is three integer comparisons in front of an
/// `O(points.size() * degree^2)` kernel.
template <class T>
void tabulate(unsigned degree, const_points<T> points, out_matrix<T> out) {
    constexpr unsigned max_degree = static_cast<unsigned>(std::numeric_limits<int>::max());
    if (degree > max_degree) {
        throw nb::value_error(("degree " + std::to_string(degree) +
                               " exceeds the largest degree the kernel can express (" +
                               std::to_string(max_degree) + ")")
                                  .c_str());
    }

    const std::size_t num_pts = points.size();
    const std::size_t num_basis = static_cast<std::size_t>(degree) + 1;
    if (out.shape(0) != num_pts || out.shape(1) != num_basis) {
        throw nb::value_error(("out has shape (" + std::to_string(out.shape(0)) + ", " +
                               std::to_string(out.shape(1)) + "), but degree " +
                               std::to_string(degree) + " at " + std::to_string(num_pts) +
                               " points needs (" + std::to_string(num_pts) + ", " +
                               std::to_string(num_basis) + ")")
                                  .c_str());
    }

    const std::span<const T> pts(points.data(), num_pts);
    const pantr::span2d<T> view(out.data(), num_pts, num_basis);

    // The kernel touches no Python object, so the GIL buys nothing while it
    // runs. Releasing it is what lets a caller thread at the Python level.
    const nb::gil_scoped_release release;
    pantr::tabulate_cardinal_bspline_1d<T>(static_cast<int>(degree), pts, view);
}

}  // namespace

NB_MODULE(_pantr_cpp, m) {
    m.attr("__doc__") = nb::none();  // the docstrings live in the Python layer

    // `.noconvert()` on `out` is a correctness requirement, not a tuning knob.
    //
    // An `nb::ndarray<T, c_contig>` parameter does NOT reject an argument that
    // fails the constraint: nanobind satisfies the constraint by CONVERTING it
    // into a contiguous temporary. For an input that is merely wasteful. For an
    // output it is silently wrong -- the kernel fills the temporary, the
    // temporary is discarded, and the caller's array comes back untouched.
    //
    // Measured before the fix: with PANTR_BACKEND=cpp,
    // `tabulate_cardinal_bspline_1d(2, pts, out=non_contiguous)` returned all
    // zeros while the numba backend returned the right answer, with no error
    // anywhere. An out-parameter that quietly does nothing is the worst failure
    // shape available: no exception, no warning, a plausible-looking array of
    // the right shape and dtype.
    //
    // `points` gets it too. There a conversion is correct, so the argument is
    // weaker -- but this prototype exists to measure, and a silent copy of a
    // 10^6-element array inside the timed region would be attributed to the
    // kernel. Better to refuse and make the caller fix the layout.
    m.def("tabulate_cardinal_bspline_1d", &tabulate<double>, nb::arg("degree"),
          nb::arg("points").noconvert(), nb::arg("out").noconvert());
    m.def("tabulate_cardinal_bspline_1d", &tabulate<float>, nb::arg("degree"),
          nb::arg("points").noconvert(), nb::arg("out").noconvert());

    // Build provenance, so a measurement can name the binary that produced it
    // rather than the source tree it was built from. `fp_contract` is the one
    // that matters for the parity bound: see tests/parity/test_basis_cardinal_bspline.py.
    m.attr("__compiler__") =
#if defined(__clang__)
        "clang " __clang_version__;
#elif defined(__GNUC__)
        "gcc " __VERSION__;
#else
        "unknown";
#endif

    m.attr("__has_std_mdspan__") =
#if defined(PANTR_HAS_STD_MDSPAN)
        true;
#else
        false;
#endif

    m.attr("__fp_contract__") =
#if defined(__FP_FAST_FMA)
        "available";
#else
        "unavailable-on-target-isa";
#endif
}
