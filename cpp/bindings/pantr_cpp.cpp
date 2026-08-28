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
/// virtual class, no return-value policy because every bound function returns
/// either `void` or a plain value (`double`, `int`) that carries no ownership
/// question for nanobind to arbitrate, and no `gil_scoped_acquire` because
/// nothing calls back into Python.
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

#include "pantr/core/error.hpp"
#include "register.hpp"

namespace nb = nanobind;

NB_MODULE(_pantr_cpp, m) {
    m.attr("__doc__") = nb::none();  // the docstrings live in the Python layer

    // One translator for the whole port. nanobind's default maps every unknown
    // exception to RuntimeError, which is the right base for a limit of this
    // build; the named subclass is what lets a caller distinguish it from a bug.
    // cpp/include/pantr/core/error.hpp carries the argument, including why the
    // tag registries deliberately do NOT use a translator.
    nb::exception<pantr::CapacityError>(m, "CapacityError", PyExc_RuntimeError);

    register_basis(m);
    register_quad(m);
    register_change_basis(m);
    register_bezier(m);
    register_bezier_root_finding(m);
    register_grid(m);
    register_geometry(m);
    register_transform(m);
    register_grid_types(m);
    register_grid_tags(m);
    register_grid_bvh(m);
    register_grid_partition(m);
    register_quad_types(m);
    register_bezier_type(m);

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
