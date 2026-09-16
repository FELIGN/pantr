/// \file
/// nanobind bindings for the knot computations of `pantr/bspline/knots.hpp`.
///
/// One entry point today, `bspline_space_cardinal_intervals_1d`, registered twice --
/// once per storage format, as every other file here registers its pair. Overload
/// resolution separates the two instantiations on the space handle's own class, so
/// neither needs a dtype argument and neither can be reached with a mismatched one.
///
/// ## Why the space crosses and the knot vector does not
///
/// Because a `BsplineSpace1D` is a domain type C++ owns since the 2026-08-27
/// amendment to `design/cross_backend_types.md`, and because the scan's tolerance
/// **is** the space's: `pantr.bspline.BsplineSpace1D.tolerance` is derived once at
/// construction from the knots as supplied, before snapping, and passing the knot
/// vector alone would invite the caller to re-derive it. The wrapper hands over the
/// handle it already holds, and the header reads `knots()`, `degree()` and
/// `tolerance()` off it.
///
/// This is the one place the shape differs from the header it binds.
/// `pantr::bspline::cardinal_intervals` takes the knot span rather than the space,
/// because `space_1d.hpp` includes `knots.hpp` and a signature over the type would
/// close that cycle; the adaptation is the two lines below and nothing else.
///
/// ## Why the result crosses as an `out=` buffer
///
/// Because it is a plain array of a length the caller already knows -- the space's
/// interval count -- which is the line `bspline_structural.cpp` draws between
/// `slice_bspline` and `slice_bspline_point`. Letting Python own the allocation keeps
/// the dtype and the read-only policy where the rest of `pantr`'s `out=` convention
/// keeps them, and this file allocates nothing.
///
/// ## No member is added to the C++ type
///
/// `cpp/include/pantr/bspline/space_1d.hpp` says the type owns no operations and
/// names this scan as the reason the line is drawn where it is. The scan therefore
/// arrives as a free function selected by a per-area catalogue module,
/// `pantr.bspline._knots_backend`, exactly as every other ported operation does.
/// `cpp/bindings/bspline_types.cpp` still binds only genuine members.
///
/// ## What this file validates, and what it leaves alone
///
/// The output length, which the space cannot know, and nothing else: the space
/// arrives already built and already validated, so there is no knot vector to refuse
/// here. `pantr.bspline._knots_backend` validates `out` *above* the branch with the
/// oracle's own helper, so the message a Python caller sees is the oracle's under
/// either backend and this check is the one a caller with no wrapper meets.
///
/// ## The GIL is released through the computation
///
/// The scan reads `T` off the space's own storage and writes `bool` into the
/// caller's buffer, with no Python object in reach and no handle dropped, which is
/// what makes the release safe -- `bspline_structural.cpp` states the converse rule,
/// that dropping the last reference to a space handle with the GIL released would be
/// a crash rather than a slowdown. `bezier_identity_mask` and
/// `lagrange_identity_mask` in `bspline_extraction_operators.cpp` release around a
/// pass of the same shape over the same intervals, and doing otherwise here would be
/// a difference with nothing behind it.

#include <cstddef>
#include <span>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "register.hpp"

namespace nb = nanobind;

namespace {

/// A writeable, contiguous 1-D boolean array as nanobind sees it.
using out_flags = nb::ndarray<bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Which of a space's intervals are cardinal.
///
/// \tparam T The scalar type the space stores.
/// \param space The space to scan.
/// \param out One entry per interval, written in full.
/// \throws nanobind::value_error If `out` does not hold one entry per interval.
template <class T>
void bind_cardinal_intervals(const pantr::bspline::BsplineSpace1D<T>& space, out_flags out) {
    const auto count = static_cast<std::size_t>(space.num_intervals());
    if (out.shape(0) != count) {
        throw nb::value_error(("out has length " + std::to_string(out.shape(0))
                               + ", but this space has " + std::to_string(count) + " intervals")
                                  .c_str());
    }

    const nb::gil_scoped_release release;
    pantr::bspline::cardinal_intervals<T>(space.knots(), space.degree(), space.tolerance(),
                                          std::span<bool>(out.data(), count));
}

}  // namespace

void register_bspline_knots(nb::module_& m) {
    m.def("bspline_space_cardinal_intervals_1d", &bind_cardinal_intervals<double>,
          nb::arg("space"), nb::kw_only(), nb::arg("out").noconvert());
    m.def("bspline_space_cardinal_intervals_1d", &bind_cardinal_intervals<float>,
          nb::arg("space"), nb::kw_only(), nb::arg("out").noconvert());
}
