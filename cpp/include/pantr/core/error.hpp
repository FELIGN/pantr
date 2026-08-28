#pragma once

/// \file
/// The one exception the port raises that is not a defect in the caller's argument.
///
/// ## What nanobind already does, and what it cannot do
///
/// nanobind 2.14.0 ships a default translator, so most of this port needs no
/// exception type of its own. It maps `std::invalid_argument`, `std::domain_error`,
/// `std::length_error` and `std::range_error` to `ValueError`, `std::out_of_range`
/// to `IndexError`, `std::overflow_error` to `OverflowError`, `std::bad_alloc` to
/// `MemoryError`, and everything else -- `std::runtime_error` included -- to
/// `RuntimeError`.
///
/// **There is no path producing `TypeError`**, and the ported types raise several.
/// So the rule for the whole port, stated once here rather than rediscovered by
/// each ticket: **type-kind checks and key lookups stay in the Python wrapper;
/// value and range checks live in the C++ type.** A tag registry's missing name is
/// a `KeyError` raised by the wrapper, which is cheaper than a translator, keeps
/// the message text, and is forced anyway.
///
/// ## Why this one type is different
///
/// `CapacityError` is the failure that is a property of the IMPLEMENTATION rather
/// than of the argument. The BVH traverses with a fixed-size stack, so a tree
/// deeper than that stack is a limit of this build; `ValueError` would blame the
/// caller for a box that is perfectly well formed.

#include <stdexcept>

namespace pantr {

/// A fixed internal limit of the implementation was exceeded.
///
/// Reaches Python as `pantr._pantr_cpp.CapacityError`, a subclass of
/// `RuntimeError`, through the single translator registered in
/// `cpp/bindings/pantr_cpp.cpp`.
class CapacityError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

}  // namespace pantr
