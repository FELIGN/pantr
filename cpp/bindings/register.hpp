#pragma once

/// \file
/// Per-package registration entry points, called from the single `NB_MODULE`
/// in `pantr_cpp.cpp`.
///
/// Splitting the module this way keeps each package's bindings -- and the
/// long comments that justify their checks -- in the file for that package,
/// while the module itself stays a short list of what it assembles.

#include <nanobind/nanobind.h>

/// Register the `pantr.basis` kernel bindings on `m`.
void register_basis(nanobind::module_& m);

/// Register the `pantr.quad` kernel bindings on `m`.
void register_quad(nanobind::module_& m);

/// Register the `pantr.change_basis` kernel bindings on `m`.
void register_change_basis(nanobind::module_& m);

/// Register the `pantr.bezier` arithmetic kernel bindings on `m`.
void register_bezier(nanobind::module_& m);

/// Register the `pantr.bezier` root-finding kernel bindings on `m`.
///
/// Separate from `register_bezier` because the two halves of the package are
/// separate ports with separate parity claims, and the file comments that justify
/// each one's checks do not belong in the same file.
void register_bezier_root_finding(nanobind::module_& m);
