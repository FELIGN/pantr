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

/// Register the `pantr.grid` kernel bindings on `m`.
///
/// One entry point rather than three, unlike the two `pantr.bezier` halves below:
/// the three grid headers are one port with one parity claim, and the comments
/// that justify their checks argue from the same place.
void register_grid(nanobind::module_& m);

/// Register the `pantr.geometry` bindings.
///
/// Unlike its siblings this one exposes a class rather than free functions: under
/// the 2026-08-27 amendment to design/cross_backend_types.md the domain types are
/// owned by C++ and Python wraps them.
///
/// \param m The extension module to register into.
void register_geometry(nanobind::module_& m);

/// Register the `pantr.transform` bindings.
///
/// The second type-exposing registration, after geometry.
///
/// \param m The extension module to register into.
void register_transform(nanobind::module_& m);

/// Register the `pantr.bezier` root-finding kernel bindings on `m`.
///
/// Separate from `register_bezier` because the two halves of the package are
/// separate ports with separate parity claims, and the file comments that justify
/// each one's checks do not belong in the same file.
void register_bezier_root_finding(nanobind::module_& m);

// ---------------------------------------------------------------------------
// Reserved slots (FELIGN/pantr#380)
// ---------------------------------------------------------------------------
//
// The thirteen tickets this milestone unblocks all bind a type, and all would
// otherwise collide on this file, on `pantr_cpp.cpp` and on `CMakeLists.txt`.
// Declaring the six entry points here, ahead of any port, means each ticket
// touches only its own `.cpp` file. Every one below is empty today; see the
// file it declares for what it is reserved for.

/// Register the `pantr.grid` tensor-product / hierarchical grid types.
void register_grid_types(nanobind::module_& m);

/// Register `pantr.grid`'s cell and facet tag registries.
void register_grid_tags(nanobind::module_& m);

/// Register `pantr.grid.BVH`.
void register_grid_bvh(nanobind::module_& m);

/// Register `pantr.grid.Partition`.
void register_grid_partition(nanobind::module_& m);

/// Register `pantr.quad.QuadratureRule`.
void register_quad_types(nanobind::module_& m);

/// Register the `pantr.bezier.Bezier` type, distinct from the arithmetic
/// kernels `register_bezier` binds above.
void register_bezier_type(nanobind::module_& m);

/// Register the `pantr.bspline` space types.
///
/// Not one of #380's six reserved slots: those were all filled before this front
/// started, and the milestone's remaining tickets add their own entry point here
/// rather than sharing one. Declared alongside them so a ticket still touches only
/// its own `.cpp`.
void register_bspline_types(nanobind::module_& m);
