#pragma once

/// \file
/// The Layer 3 precondition macro, and why violating one is not the same in both
/// backends.
///
/// Every kernel in this tree is a transliteration of a Numba kernel whose docstring
/// says, in these words, that no input validation is performed. That sentence is
/// **weaker in Python than it is here**, and the difference is not a matter of degree.
/// A numpy array supports negative indexing and a Python integer does not overflow, so
/// a violated precondition on that side yields a defined wrong answer. On this side the
/// same violation indexes out of bounds or overflows a signed integer, which is
/// undefined behaviour: measured, a heap-buffer-overflow **write** in
/// `tabulate_bernstein_1d` at `degree = -1`, where the oracle returns `[2, 0, 0, 0]`.
///
/// So the contract needs two words rather than one, and `PANTR_PRECONDITION` is how a
/// header says which it means:
///
/// - a **correctness** obligation, whose violation is a wrong answer in both backends,
///   is documented and not asserted;
/// - a **memory-safety** obligation, whose violation is undefined behaviour here and
///   merely wrong there, carries this macro.
///
/// Grepping for it therefore answers "what may I not pass?" without reading the body,
/// which is the property the plain docstring wording had stopped providing.
///
/// **It compiles to nothing in a release build**, by construction rather than by
/// policy: it is `assert`, so `NDEBUG` removes it. That is what lets it sit inside
/// `bvh_query_count`'s traversal loop, which visits a node per iteration and is the one
/// hot path among these sites. Nothing here is a substitute for the validation the
/// bindings already do: `cpp/bindings/` refuses every one of these violations before a
/// Python caller can express it, and that is where a *user* is protected. This macro
/// protects the C++ caller who includes the header directly, in the build where that
/// caller can afford to be protected.
///
/// The corollary is worth stating because it is the limitation: a consumer compiling
/// against these headers in a release build gets no check. The reason that is
/// acceptable here is the same reason `assert` exists at all, and the alternative was
/// measured against `bvh_query_count`'s per-node cost rather than assumed.

#include <cassert>

/// Assert a precondition whose violation is undefined behaviour rather than a wrong answer.
///
/// \param cond The condition the caller is required to establish.
/// \param what A short phrase naming what the caller must hold, shown when it does not.
#define PANTR_PRECONDITION(cond, what) assert((cond) && (what))
