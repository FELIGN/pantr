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
/// `std::constructible_from<T, double>` is required because the kernels build
/// constants: `T(1)` seeds the recurrence, and `Acc(0)`, `Acc(1)` and
/// `Acc(static_cast<double>(k))` appear inside it. Without it the concept was a
/// strict subset of what a kernel actually needs, so it certified types the
/// kernel then rejected -- measured with a forward-mode `Dual` built by factory
/// functions rather than by a numeric constructor, a common AD shape: it
/// satisfied `Real` and then produced thirty lines of template error from inside
/// a header its author did not write, which is precisely the failure concepts
/// exist to prevent.
///
/// It does not weaken the Tier B discipline. An *explicit* constructor satisfies
/// `constructible_from` without opening an implicit conversion, so
/// `!std::convertible_to<Dual1, double>` and `!std::convertible_to<double,
/// Dual1>` both still hold; cpp/tests/test_scalar_generic.cpp asserts all three.
///
/// Satisfied by `float` and `double`. A forward-mode `Dual<T, N>` satisfies it
/// by supplying the five arithmetic operators, a `value_of` overload and a
/// constructor from `double`; cpp/tests/test_scalar_generic.cpp defines one and
/// instantiates the kernel on it, which is what keeps the claim honest rather
/// than aspirational.
template <class T>
concept Real =
    std::copyable<T> && detail::has_value_of<T> && std::constructible_from<T, double> &&
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

/// The type intermediate quantities are accumulated in: `double` for `float`
/// storage, and the scalar itself otherwise.
///
/// ## Why the decision is keyed on the VALUE type
///
/// The rule is about the format the arithmetic runs in, and that is a property
/// of the scalar's value component rather than of the wrapper carrying it.
/// Keyed on the storage type instead -- as a `template <> struct
/// accumulator<float>` specialisation -- the same question gets two answers by
/// accident: it fires for `float` and silently misses every scalar whose value
/// component is a `float` without being one.
///
/// A scalar of that second shape is rejected here rather than answered wrongly.
/// Widening it means rebinding its value component to `double`, which needs a
/// hook no scalar in this library supplies; the static assertion says so at the
/// trait, instead of letting the kernel fail thirty lines into a template.
///
/// ## What justifies widening at all
///
/// **For the tabulation kernels the reason is PARITY, not accuracy.** numba
/// promotes float32 intermediates to float64 as a side effect of literal
/// typing, so a port that computed in `float` throughout disagrees with the
/// oracle by about one float32 ulp. cpp/include/pantr/basis/cardinal_bspline.hpp
/// records the numba measurement that establishes the promotion, and
/// `test_float32_intermediates_are_float64` in tests/test_cpp_parity.py is what
/// checks the C++ side against it. It is *not* the `sqrt(m) * eps32` argument of
/// design/large_data_fitting.md: that argument is about a length-`m`
/// scalar accumulation, and a Cox-de Boor tabulation stores its state back into
/// the output array at every stage, where the measured float32 error is flat in
/// the degree. That note was amended in this branch to bound its own rule
/// accordingly.
///
/// Both reasons reach the same policy, which is why the policy lives here in the
/// core and not in the header of whichever kernel needed it first.
template <Real T>
struct accumulator {
    static_assert(std::same_as<T, value_type_t<T>> || std::same_as<value_type_t<T>, double>,
                  "a scalar whose value component is not itself double has to be widened "
                  "by rebinding that component, which no scalar here supplies a hook for; "
                  "specialise pantr::accumulator for it before using it");

    /// The accumulation type for `T`.
    using type = std::conditional_t<std::same_as<value_type_t<T>, float>, double, T>;
};

/// The accumulation type of `T`, as an alias.
///
/// `cpp/tests/test_scalar_generic.cpp` pins all three answers -- `double` for
/// `double`, `double` for `float`, and a differentiable scalar for itself -- for
/// the same reason it pins `value_type_t`: nothing checks an alias template
/// until something instantiates it.
template <Real T>
using accumulator_t = typename accumulator<T>::type;

}  // namespace pantr
