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
