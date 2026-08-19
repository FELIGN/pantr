#pragma once

/// \file
/// The scalar concept the pantr core is generic over, and the Tier B access
/// point every non-differentiable operation goes through.
///
/// ## Why the core is templated at all
///
/// Not for automatic differentiation. design/automatic_differentiation.md
/// concludes that no flavour of the intended work requires AD inside the
/// library: three of the four optimisation flavours are served by linearity plus
/// an analytic rule, and the fourth has few enough parameters for hand-derived
/// formulas. The justification is `float32`, which halves memory and doubles the
/// SIMD lane count, and which the large-image fitting case needs on its own
/// terms. AD compatibility rides along for free, because the discipline that
/// admits a `Dual` type is the same discipline that admits `float`.
///
/// That ordering matters under pressure. If compile times become painful,
/// `float32` support is kept and the AD discipline is what gets relaxed. The
/// reverse would be the wrong trade.
///
/// ## The tiers
///
/// **Tier A** is generic in the scalar: arithmetic only, and every operation is
/// meaningful for a differentiable type. Kernels live here.
///
/// **Tier B** is value-only: comparisons and branches, `floor`, casts to an
/// integer index. These are not differentiable operations and they are not
/// errors -- a knot-span search is genuinely a function of the *value* -- but
/// they must be written so, by passing through `value_of`. A comparison applied
/// to the scalar directly compiles for `double` and then either fails to compile
/// or, worse, compares the wrong component for a `Dual`.
///
/// **Tier C** replaces differentiating an iteration with an analytic derivative
/// rule. No Tier C code exists in this prototype; it is named here so the three
/// are not confused.
///
/// ## The four disciplines
///
/// From design/automatic_differentiation.md, all four cheap up front and
/// expensive to retrofit:
///
///  1. Comparisons and branches on the scalar go through `value_of`.
///  2. No `std::floor` and no integer casts on the scalar.
///  3. Unqualified calls with a using-declaration (`using std::sqrt; sqrt(x);`),
///     never `std::sqrt(x)`, which hard-blocks any AD type by preventing ADL
///     from finding the type's own overload. This prototype's one kernel is
///     arithmetic-only and so has no site that exercises the rule; the guard in
///     scripts/ci_local.sh enforces it for the day one appears.
///  4. Parameters that change discrete structure are value-only. The degree is
///     the example here: it decides how many basis functions exist, so it is an
///     `int` and never a scalar.

#include <concepts>
#include <type_traits>

namespace pantr {

/// Tier B access: the floating-point value underlying a scalar.
///
/// This overload covers the built-in floating-point types, for which the value
/// is the number itself. A differentiable type supplies its own overload in its
/// own namespace, found by argument-dependent lookup through the two-step
/// pattern below.
///
/// Every Tier B operation in the library is written as
///
/// ```cpp
/// using pantr::value_of;          // makes this overload visible for double
/// if (value_of(u) > value_of(u_max)) { ... }   // ADL finds a Dual's overload
/// ```
///
/// rather than as `pantr::value_of(u)`, for the same reason `std::swap` is used
/// that way: qualifying the call suppresses ADL and so silently excludes every
/// user-supplied type.
template <std::floating_point T>
[[nodiscard]] constexpr T value_of(T x) noexcept {
    return x;
}

namespace detail {

using pantr::value_of;

/// Satisfied when `value_of` is reachable for `T` and yields a floating-point
/// value. Kept in a namespace that has the using-declaration in scope so the
/// concept itself is checked through the same two-step lookup the library uses.
template <class T>
concept has_value_of = requires(const T& a) {
    { value_of(a) } -> std::floating_point;
};

/// What the two-step lookup yields for `T`, as a type.
///
/// It exists so that `value_type_t` below can be written without naming
/// `value_of` at its own point of use. Spelling it there as
/// `detail::value_of(x)` is a QUALIFIED call, which suppresses ADL and so never
/// finds a scalar's hidden-friend overload -- the exact mistake this file's
/// header spends ten lines warning about, made in the one place the warning
/// could not be read. The unqualified call has to happen *here*, inside the
/// namespace that has the using-declaration in scope, exactly as
/// `has_value_of` above does it.
template <class T>
using value_of_result_t = decltype(value_of(std::declval<const T&>()));

}  // namespace detail

/// The scalar type the Tier A core is generic over.
///
/// Deliberately *not* requiring `std::regular`, and the omission is the point:
/// `std::regular` demands `operator==`, which would make
/// `if (a == b)` compile on every conforming scalar and so make the Tier B
/// discipline unenforceable at exactly the site it exists to protect. Equality
/// and ordering are reached through `value_of` instead, and a scalar type is
/// free not to provide them at all.
///
/// Satisfied by `float` and `double`. A forward-mode `Dual<T, N>` satisfies it
/// by supplying the five arithmetic operators and a `value_of` overload;
/// cpp/tests/test_scalar_generic.cpp defines one and instantiates the kernel on
/// it, which is what keeps the claim honest rather than aspirational.
template <class T>
concept Real =
    std::copyable<T> && detail::has_value_of<T> &&
    requires(T a, T b) {
        { -a } -> std::convertible_to<T>;
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { a / b } -> std::convertible_to<T>;
    };

/// The floating-point type underlying a `Real`: `double` for `double`, and the
/// value component for a differentiable type.
///
/// `cpp/tests/test_scalar_generic.cpp` pins all three answers as static
/// assertions, and it has to. An alias template is only checked where something
/// instantiates it, and the case this one exists for -- a scalar reached only by
/// ADL -- is exactly the case a `double`-only build never instantiates.
template <Real T>
using value_type_t = std::remove_cvref_t<detail::value_of_result_t<T>>;

}  // namespace pantr
