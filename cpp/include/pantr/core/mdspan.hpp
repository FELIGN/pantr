#pragma once

/// \file
/// The `<mdspan>` switch: the standard header where it exists, the Kokkos
/// reference implementation where it does not.
///
/// design/toolchain_requirements.md classifies `<mdspan>` as a feature toggle
/// rather than a hard gate, because a drop-in fallback exists and is
/// maintained by the same people who wrote the proposal. `PANTR_HAS_STD_MDSPAN`
/// is set by the configure-time probe in cmake/PantrCompilerProbes.cmake.
///
/// **On this project's toolchain the toggle resolves to Kokkos**, and the
/// reason is an implementation gap rather than a language-mode restriction.
/// Measured here: GCC 14.4.0's libstdc++ ships no `mdspan` header at all, and
/// `__cpp_lib_mdspan` is undefined even under `-std=gnu++23` -- which is that
/// macro's whole job, so its absence in C++23 mode settles that the library has
/// not implemented P0009, rather than that a standard-version gate is hiding a
/// header it does ship. Clang 18 here resolves to that same libstdc++ and not
/// to libc++, so it inherits the absence for the same reason.
///
/// Worth stating precisely, because the plausible explanation is the wrong one:
/// "it is a C++23 header and we build with -std=c++20" predicts that
/// `-std=c++23` makes it appear, and measurably it does not. The standard
/// branch below is kept because it is what the toggle is for once libstdc++
/// implements it, and because it costs three lines.
///
/// ## The one API difference the switch has to absorb
///
/// The two branches disagree on how a pack of indices is spelled. C++23
/// `std::mdspan` uses the multidimensional subscript operator, `view[i, j]`,
/// which is a C++23 *language* feature and does not parse under the C++20
/// baseline at all; Kokkos falls back to `view(i, j)` there. So element access
/// goes through `pantr::at` rather than through either operator directly.
///
/// What `at` uses is a third spelling that both branches accept: a single
/// `std::array` of indices. C++23 specifies that overload, and Kokkos provides
/// it UNCONDITIONALLY -- in `__p0009_bits/mdspan.hpp`, outside the
/// `MDSPAN_USE_BRACKET_OPERATOR` guard that picks between the two spellings
/// above. So one spelling serves both branches and `at` needs no preprocessor
/// branch of its own. It carried one until this was checked against the Kokkos
/// source, on the stated premise that no single spelling could work in both;
/// that premise was false, and the branch it justified was also what kept `at`
/// fixed at rank 2 while `span_nd` had already been generalised.
///
/// The Kokkos half of that claim is compiled and run at ranks 1, 2 and 3 by
/// cpp/tests/test_dependency_smoke.cpp, and separately with
/// `MDSPAN_USE_BRACKET_OPERATOR` forced on and `MDSPAN_USE_PAREN_OPERATOR` forced
/// off, which is the configuration the deleted `#if` existed for. The standard
/// half is not, and cannot be
/// on this toolchain -- no standard library here ships the header, which is why
/// the toggle is OFF -- so it rests on the wording of C++23 [mdspan.mdspan]
/// rather than on a build.

#include <array>
#include <cstddef>

#if defined(PANTR_HAS_STD_MDSPAN)
#  include <mdspan>
#else
#  include <experimental/mdspan>
#endif

namespace pantr {

#if defined(PANTR_HAS_STD_MDSPAN)
namespace md = std;
#else
namespace md = std::experimental;
#endif

/// A view of `Rank` dimensions over contiguous, row-major storage.
///
/// Generic in the rank and fixed in the layout, and the asymmetry is deliberate.
/// The rank is already known to vary: design/bezier_extraction_api.md specifies
/// an mdspan of shape `(n_el, (p+1)^d, rank)` for the extraction sweep, which is
/// rank 3. Naming the rank now costs one line and changes no call site; naming it
/// after five kernels have copied a rank-2 signature costs five signatures.
///
/// The layout stays `layout_right` because the decision that would generalise it
/// has not been taken. design/large_data_fitting.md wants the per-direction
/// kernel to read strided input directly rather than materialise a contiguous
/// transpose, but it offers ping-pong buffers as the alternative and does not
/// settle between them. Templating the kernel on the layout costs readability
/// today and would spend that decision early, so it waits for the kernel that
/// actually needs a stride.
///
/// One consequence is visible now and is the price of waiting: a non-contiguous
/// `out` cannot be described by this alias, so the Python adapter in
/// `pantr.basis._basis_backend` computes into a contiguous buffer and copies
/// back, where the numba kernel fills the caller's array in place.
template <class T, std::size_t Rank>
using span_nd = md::mdspan<T, md::dextents<std::size_t, Rank>, md::layout_right>;

/// A 2D view over contiguous, row-major storage: the shape every tabulation
/// kernel writes into.
template <class T>
using span2d = span_nd<T, 2>;

/// Element access, spelled once so the two branches of the switch stay source
/// compatible, at whatever rank the view has.
///
/// \param view The view to index.
/// \param idx One index per dimension of `view`, outermost first. Supplying the
///        wrong number was already a compile error before the `static_assert`,
///        since every mdspan subscript overload is constrained on the rank; the
///        assertion only replaces a page of template diagnostics with a sentence.
/// \return The element, with the reference type the view's accessor yields.
///
/// \note No bounds checking is performed, in either branch of the switch.
template <class View, class... Idx>
[[nodiscard]] constexpr decltype(auto) at(const View& view, Idx... idx) noexcept {
    static_assert(sizeof...(Idx) == View::rank(), "index count must match the view's rank");
    using I = typename View::index_type;
    return view[std::array<I, sizeof...(Idx)>{static_cast<I>(idx)...}];
}

}  // namespace pantr
