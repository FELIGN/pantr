#pragma once

/// \file
/// The Bézier value type, and the compatibility forwarder for the old name of
/// `pantr/bezier/kernels_1d.hpp`.
///
/// ## Two things in one file, on purpose
///
/// The kernels that used to live here moved to `kernels_1d.hpp` so that this name
/// could hold the *type*. The include below stays because the header set is a
/// promise: the top-level `CMakeLists.txt` installs `cpp/include/pantr` wholesale
/// and exports a findable package with `COMPATIBILITY SameMinorVersion`, so an
/// already-installed consumer including this path must keep getting the kernels
/// until someone decides otherwise. `cpp/consumer/main.cpp` includes it for
/// exactly that reason, which is what stops the forwarding include from being
/// deleted by accident.
///
/// **The forwarding include is scaffolding.** What removes it is a deliberate
/// decision to break the installed header set -- a major version, or a note in the
/// release that the old path no longer carries the kernels. In-tree code that
/// wants a kernel includes `pantr/bezier/kernels_1d.hpp` directly; in-tree code
/// that wants the type includes this file and gets the kernels it did not ask for
/// as a side effect until that day.
///
/// Note that the namespace moved with the file, from `pantr` to `pantr::bezier`.
/// This header forwards the *path*, not the old spelling of the names.
///
/// ## What this type owns, and what it does not
///
/// It owns the *value*: the control net and the rationality flag, plus the
/// quantities they determine -- `dim`, `degree` and `rank`. It owns no operations.
/// Evaluation, degree elevation and reduction, the shape operations and the
/// product are four separate ports over free functions taking a `const Bezier&`,
/// which is what lets them proceed independently of one another once this type
/// exists.
///
/// ## Validating rather than asserting
///
/// `pantr/core/precondition.hpp` says a Layer 3 kernel asserts, and that the
/// bindings are where a *user* is protected. Both stay true: this is not a kernel
/// but the C++ counterpart of Layer 2, so it validates and throws
/// `std::invalid_argument` in a release build as much as a debug one. A caller
/// with no Python cannot be protected by `cpp/bindings/`.
///
/// ## Parity notes for the Python oracle
///
/// `pantr.bezier.Bezier` is the oracle. Three things it does that this type
/// reproduces, and one it does that this type deliberately does not:
///
///  - **The rank check counts the weight column out, and can report a negative
///    number.** The oracle computes `rank = shape[-1] - 1` for a rational
///    geometry and rejects `rank <= 0`, so a rational net with no components at
///    all is reported as `rank -1`. The arithmetic here is signed for that reason
///    alone.
///  - **The order of the checks is the oracle's:** shape first, in `ControlNet`,
///    rank second. Two simultaneously bad arguments must produce the same message
///    on both sides, and only the order decides which one they get.
///  - **The messages are the oracle's, character for character.**
///    `tests/parity/test_bezier_type.py` asserts it.
///  - **It copies the control points; the oracle does not.** The oracle stores the
///    caller's array and hands the same object back from `control_points`, so a
///    caller can mutate a constructed Bézier through either end. That is a defect
///    of the oracle -- the same shape as FELIGN/pantr#338 -- and this type does not
///    reproduce it. Fixing it on the Python side is FELIGN/pantr#375; until then
///    the two backends differ there, deliberately and in the safe direction.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bezier/control_net.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bezier {

/// A Bézier curve, surface or volume over `[0, 1]^dim`.
///
/// Holds a control net and nothing else but a flag. The polynomial degree in each
/// parametric direction is *inferred*: `degree(d) == net().extent(d) - 1`. For a
/// rational Bézier the last component of every coefficient is a homogeneous
/// weight and the coordinates are stored weighted, so `rank()` is one less than
/// the net's component count.
///
/// Instances are immutable: no operation changes an existing Bézier, and every
/// derived one is returned by value.
template <Real T>
class Bezier {
  public:
    /// Build a Bézier from a control net, validating its rank.
    ///
    /// \param net The control points; moved in.
    /// \param is_rational Whether the last component of each coefficient is a
    ///        homogeneous weight.
    /// \throws std::invalid_argument If the resulting rank is not at least 1.
    Bezier(ControlNet<T> net, bool is_rational)
        : net_(std::move(net)), is_rational_(is_rational) {
        const std::ptrdiff_t rank = signed_rank();
        if (rank <= 0) {
            throw std::invalid_argument("The Bézier must have at least rank one. Got rank "
                                        + std::to_string(rank) + ".");
        }
    }

    /// The control points.
    ///
    /// \return The stored net, valid while the Bézier lives.
    [[nodiscard]] const ControlNet<T>& net() const noexcept { return net_; }

    /// Whether the last component of each coefficient is a homogeneous weight.
    ///
    /// \return `true` for a rational Bézier.
    [[nodiscard]] bool is_rational() const noexcept { return is_rational_; }

    /// The number of parametric directions.
    ///
    /// \return `net().dim()`, at least 1.
    [[nodiscard]] std::size_t dim() const noexcept { return net_.dim(); }

    /// The polynomial degree in every parametric direction.
    ///
    /// \return One degree per direction, `net().extent(d) - 1`. Never empty, and
    ///         no entry underflows, because `ControlNet` rejects a zero extent.
    [[nodiscard]] std::vector<std::size_t> degree() const {
        std::vector<std::size_t> degrees(dim());
        for (std::size_t d = 0; d < dim(); ++d) {
            degrees[d] = net_.shape()[d] - 1;
        }
        return degrees;
    }

    /// The polynomial degree in one parametric direction.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return `net().extent(d) - 1`.
    /// \throws std::out_of_range If `d >= dim()`.
    [[nodiscard]] std::size_t degree(std::size_t d) const { return net_.extent(d) - 1; }

    /// The number of value components a caller sees.
    ///
    /// The weight column of a rational Bézier is not one of them: a rational curve
    /// in the plane stores three components and has rank 2.
    ///
    /// \return `net().num_components()`, less one when rational. At least 1.
    [[nodiscard]] std::size_t rank() const noexcept {
        return static_cast<std::size_t>(signed_rank());
    }

  private:
    /// The rank, before it is known to be positive.
    ///
    /// Split out so that the constructor's check and the public accessor cannot
    /// drift apart: the accessor's cast to `std::size_t` is sound only because the
    /// constructor rejected everything this can return that is not positive.
    ///
    /// \return The component count less the weight column, possibly negative.
    [[nodiscard]] std::ptrdiff_t signed_rank() const noexcept {
        return static_cast<std::ptrdiff_t>(net_.num_components())
               - (is_rational_ ? std::ptrdiff_t{1} : std::ptrdiff_t{0});
    }

    ControlNet<T> net_;  ///< The control points, owned.
    bool is_rational_;   ///< Whether the last component is a homogeneous weight.
};

}  // namespace pantr::bezier
