#pragma once

/// \file
/// Compatibility forwarder for the old name of `pantr/bezier/kernels_1d.hpp`.
///
/// The kernels moved because this directory is about to gain a `Bezier` *type*,
/// and two headers called `bezier` in one directory is a trap: a reader cannot
/// tell from an include line which of the two a translation unit meant.
///
/// This file stays because the header set is a promise. The top-level
/// `CMakeLists.txt` installs `cpp/include/pantr` wholesale and exports a findable
/// package with `COMPATIBILITY SameMinorVersion`, so an already-installed
/// consumer including this path must keep compiling until someone decides
/// otherwise. `cpp/consumer/main.cpp` includes it for exactly that reason, which
/// is what stops this file from being deleted by accident.
///
/// **It is scaffolding.** What removes it is a deliberate decision to break the
/// installed header set -- a major version, or a note in the release that the old
/// path is gone. Nothing in the tree should include it: in-tree code includes
/// `pantr/bezier/kernels_1d.hpp` directly.
///
/// Note that the namespace moved with the file, from `pantr` to `pantr::bezier`.
/// This header forwards the *path*, not the old spelling of the names.

#include "pantr/bezier/kernels_1d.hpp"
