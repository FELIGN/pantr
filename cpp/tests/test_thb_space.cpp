/// \file
/// The hierarchical B-spline space: what it selects, what it truncates, and what it
/// shares.
///
/// ## What is asserted here and not in `tests/parity/test_thb_spline_space.py`
///
/// The parity file compares this type against the Python oracle over a generative sweep,
/// which is the stronger check for everything both sides can answer. Three kinds of
/// property are not reachable from there and live here:
///
///  - **Lifetime.** `design/bspline_ownership_lifetime.md` F4 records that a value
///    assertion does not detect a broken lifetime -- a scalar read after free returns the
///    right answer often enough that the obvious test passes on a broken design. The
///    check below is that a nested object read *after its owner is destroyed* still
///    reports a count the owner's destructor would have overwritten, and its gate is the
///    `gcc-debug` preset's address sanitizer rather than the assertion.
///  - **Thread safety.** The contribution table is a `pantr::LazySlot`, and
///    `design/bspline_derived_caches.md` F3 measured a bare `mutable std::optional`
///    giving 60 correct answers in 60 unsanitized runs and four ThreadSanitizer reports.
///    So the claim gets a gate rather than a reader's trust, and no assertion on a value
///    can be that gate. From Python the GIL hides it entirely.
///  - **`float` storage.** The binding registers both widths, but the parity file's
///    `float32` coverage is one case; here the whole type is censused at `float` so an
///    instantiation error cannot hide behind the `double` one.
///
/// ## The identity contracts, and why they are compared by address
///
/// `level_space(0)` must hand back the very handle the space was built from, not a copy
/// of what it points at. Every value assertion in this file would pass either way, and
/// the Python contract `thb.level_space(0) is thb.root_space` would then present two
/// Python objects agreeing on identity over two different C++ objects --
/// `design/bspline_ownership_lifetime.md` F6. Comparing addresses is what distinguishes
/// them.
///
/// The same argument runs for the grid, with the opposite conclusion in one place:
/// `refine` and `coarsen` must **never** hand their result the receiver's grid, because a
/// grid carries mutable tag registries and two spaces sharing one would make a tag set
/// through the first visible through the second. That is asserted on the no-op paths,
/// which are the ones where sharing would be the natural implementation.
///
/// ## The truncation is checked against an identity, not against a rerun
///
/// `check_the_truncation_is_a_partition_of_unity` sums every active function's
/// coefficient vector, pushed to the finest level by pure two-scale refinement, and
/// requires the all-ones vector -- Giannelli-Juttler-Speleers (2012) Thm 6. The pushing
/// uses `pantr/bspline/knot_insertion.hpp`, which
/// `cpp/tests/test_bspline_knot_insertion.cpp` pins independently against the cardinal
/// B-spline's binomial stencil, so this is a composition of two checked things rather
/// than the truncation grading itself.
///
/// It carries its discriminator: the same computation on the **untruncated** basis must
/// fail, and does, by a wide margin. Without that, an identity that the truncation is
/// what restores would be consistent with a test that cannot fail.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/thb_space.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

namespace {

using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;
using pantr::bspline::THBSplineSpace;
using pantr::grid::HierarchicalGrid;
using pantr::grid::TensorProductGrid;

/// A clamped uniform knot vector on `[lo, hi]`.
///
/// \tparam T The scalar type.
/// \param degree The polynomial degree.
/// \param num_elements The number of equal spans.
/// \param lo The domain start.
/// \param hi The domain end.
/// \return The knot vector.
template <class T>
std::vector<T> open_knots(std::int64_t degree, std::int64_t num_elements, T lo, T hi) {
    std::vector<T> knots(static_cast<std::size_t>(degree), lo);
    for (std::int64_t i = 0; i <= num_elements; ++i) {
        knots.push_back(lo
                        + static_cast<T>(static_cast<double>(i)
                                         * (static_cast<double>(hi) - static_cast<double>(lo))
                                         / static_cast<double>(num_elements)));
    }
    knots.insert(knots.end(), static_cast<std::size_t>(degree), hi);
    return knots;
}

/// A tensor-product root space with the same direction repeated.
///
/// \tparam T The scalar type.
/// \param dim The number of directions.
/// \param degree The polynomial degree.
/// \param num_elements The number of equal spans per direction.
/// \return A handle on the space.
template <class T>
std::shared_ptr<const BsplineSpace<T>> root_space(std::int64_t dim, std::int64_t degree,
                                                  std::int64_t num_elements) {
    const std::vector<T> knots = open_knots<T>(degree, num_elements, T(0), T(1));
    auto one_d = std::make_shared<const BsplineSpace1D<T>>(
        std::span<const T>(knots), degree, false, KnotSnapping::merge_near_duplicates);
    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions(
        static_cast<std::size_t>(dim), one_d);
    return std::make_shared<const BsplineSpace<T>>(std::move(directions));
}

/// A hierarchy over the unit cube, refined level by level from the origin corner.
///
/// \param dim The number of axes.
/// \param num_elements The root cell count per axis.
/// \param factor The refinement factor per axis.
/// \param levels How many successive corner refinements to apply.
/// \return A handle on the grid.
std::shared_ptr<HierarchicalGrid<double>> corner_hierarchy(std::int64_t dim,
                                                           std::int64_t num_elements,
                                                           std::int64_t factor,
                                                           std::int64_t levels) {
    const auto d = static_cast<std::size_t>(dim);
    std::vector<double> axis;
    for (std::int64_t i = 0; i <= num_elements; ++i) {
        axis.push_back(static_cast<double>(i) / static_cast<double>(num_elements));
    }
    const std::vector<std::vector<double>> breakpoints(d, axis);
    const std::vector<std::int64_t> factors(d, factor);
    auto grid = std::make_shared<HierarchicalGrid<double>>(
        TensorProductGrid<double>(breakpoints), std::span<const std::int64_t>(factors));
    const std::vector<std::int64_t> lo(d, 0);
    const std::vector<std::int64_t> hi(d, 2);
    for (std::int64_t level = 0; level < levels; ++level) {
        grid = std::make_shared<HierarchicalGrid<double>>(
            grid->refine(level, std::span<const std::int64_t>(lo),
                         std::span<const std::int64_t>(hi)));
    }
    return grid;
}

/// A two-dimensional, three-level, degree-2 dyadic space.
///
/// \param truncate Whether the truncated basis is built.
/// \return The space.
THBSplineSpace<double> reference_space(bool truncate) {
    return THBSplineSpace<double>(root_space<double>(2, 2, 4), corner_hierarchy(2, 4, 2, 2),
                                  truncate, {std::nullopt, std::nullopt});
}

/// `gamma_m = m u / (1 - m u)`, the standard accumulation constant.
///
/// \param m The number of roundings.
/// \return The constant, relative.
double gamma_of(std::int64_t m) {
    const double u = 0.5 * std::numeric_limits<double>::epsilon();
    const double mu = static_cast<double>(m) * u;
    return mu / (1.0 - mu);
}

/// How far the active basis is from summing to one, in the finest level's basis.
///
/// Pushes every active function's coefficient vector to the finest level by pure
/// two-scale refinement -- no truncation -- and returns the largest deviation of the sum
/// from one. See the file comment for what makes this an independent check.
///
/// \param space The space to grade.
/// \return `max |sum_i c_i - 1|` over the finest tensor-product basis.
double partition_of_unity_defect(const THBSplineSpace<double>& space) {
    const auto d = static_cast<std::size_t>(space.dim());
    const std::int64_t top = space.num_levels() - 1;

    // The two-scale matrices, one per level transition per direction.
    std::vector<std::vector<std::vector<double>>> oslo;
    std::vector<std::vector<std::int64_t>> cols;
    for (std::int64_t m = 0; m < top; ++m) {
        std::vector<std::vector<double>> per_direction;
        std::vector<std::int64_t> per_direction_cols;
        for (std::size_t k = 0; k < d; ++k) {
            const BsplineSpace1D<double>& old_space =
                space.level_space_ref(m).space_ref(static_cast<std::int64_t>(k));
            const BsplineSpace1D<double>& new_space =
                space.level_space_ref(m + 1).space_ref(static_cast<std::int64_t>(k));
            per_direction.push_back(pantr::bspline::oslo_matrix_1d<double>(
                old_space.degree(), old_space.knots(), new_space.knots()));
            per_direction_cols.push_back(old_space.num_basis());
        }
        oslo.push_back(std::move(per_direction));
        cols.push_back(std::move(per_direction_cols));
    }

    std::vector<std::int64_t> finest(d);
    std::int64_t finest_total = 1;
    for (std::size_t k = 0; k < d; ++k) {
        finest[k] = space.level_space_ref(top).space_ref(static_cast<std::int64_t>(k)).num_basis();
        finest_total *= finest[k];
    }
    std::vector<double> total(static_cast<std::size_t>(finest_total), 0.0);

    for (std::int64_t dof = 0; dof < space.num_total_basis(); ++dof) {
        const std::optional<pantr::bspline::TruncatedView> view = space.truncated(dof);
        std::int64_t start = 0;
        std::vector<std::int64_t> box_lo(d);
        std::vector<std::int64_t> shape(d, 1);
        std::vector<double> coeffs;
        if (view.has_value()) {
            start = view->rep_level;
            for (std::size_t k = 0; k < d; ++k) {
                box_lo[k] = view->box_lo[k];
                shape[k] = view->shape[k];
            }
            coeffs.assign(view->coeffs.begin(), view->coeffs.end());
        } else {
            start = space.dof_level(dof);
            const std::span<const std::int64_t> active =
                space.active_function_indices(start);
            const std::int64_t position =
                dof - space.level_offsets()[static_cast<std::size_t>(start)];
            std::int64_t flat = active[static_cast<std::size_t>(position)];
            for (std::size_t k = d; k > 0; --k) {
                const std::int64_t n = space.level_space_ref(start)
                                           .space_ref(static_cast<std::int64_t>(k - 1))
                                           .num_basis();
                box_lo[k - 1] = flat % n;
                flat /= n;
            }
            coeffs.assign(1, 1.0);
        }

        for (std::int64_t level = start; level < top; ++level) {
            const auto m = static_cast<std::size_t>(level);
            for (std::size_t k = 0; k < d; ++k) {
                const std::vector<double>& alpha = oslo[m][k];
                const std::int64_t num_cols = cols[m][k];
                const std::int64_t rows =
                    static_cast<std::int64_t>(alpha.size()) / num_cols;
                std::int64_t new_lo = -1;
                std::int64_t new_hi = -1;
                for (std::int64_t row = 0; row < rows; ++row) {
                    bool non_zero = false;
                    for (std::int64_t col = box_lo[k]; col < box_lo[k] + shape[k]; ++col) {
                        non_zero =
                            non_zero
                            || alpha[static_cast<std::size_t>(row * num_cols + col)] != 0.0;
                    }
                    if (non_zero) {
                        if (new_lo < 0) {
                            new_lo = row;
                        }
                        new_hi = row + 1;
                    }
                }
                const std::int64_t new_width = new_hi - new_lo;
                std::int64_t outer = 1;
                for (std::size_t a = 0; a < k; ++a) {
                    outer *= shape[a];
                }
                std::int64_t inner = 1;
                for (std::size_t a = k + 1; a < d; ++a) {
                    inner *= shape[a];
                }
                std::vector<double> next(
                    static_cast<std::size_t>(outer * new_width * inner), 0.0);
                for (std::int64_t o = 0; o < outer; ++o) {
                    for (std::int64_t i = 0; i < new_width; ++i) {
                        for (std::int64_t j = 0; j < shape[k]; ++j) {
                            const double a = alpha[static_cast<std::size_t>(
                                (new_lo + i) * num_cols + box_lo[k] + j)];
                            for (std::int64_t n = 0; n < inner; ++n) {
                                next[static_cast<std::size_t>((o * new_width + i) * inner + n)] +=
                                    a
                                    * coeffs[static_cast<std::size_t>((o * shape[k] + j) * inner
                                                                      + n)];
                            }
                        }
                    }
                }
                coeffs.swap(next);
                box_lo[k] = new_lo;
                shape[k] = new_width;
            }
        }

        // Scatter the box into the finest-level accumulator.
        std::vector<std::int64_t> cursor(box_lo.begin(), box_lo.end());
        std::size_t offset = 0;
        for (;;) {
            std::int64_t flat = 0;
            for (std::size_t k = 0; k < d; ++k) {
                flat = flat * finest[k] + cursor[k];
            }
            total[static_cast<std::size_t>(flat)] += coeffs[offset];
            ++offset;
            std::size_t axis = d;
            bool done = false;
            while (axis > 0) {
                --axis;
                ++cursor[axis];
                if (cursor[axis] < box_lo[axis] + shape[axis]) {
                    break;
                }
                cursor[axis] = box_lo[axis];
                if (axis == 0) {
                    done = true;
                }
            }
            if (done) {
                break;
            }
        }
    }

    double worst = 0.0;
    for (const double value : total) {
        worst = std::max(worst, std::abs(value - 1.0));
    }
    return worst;
}

/// The counts and the shape of a hierarchy every other case here builds on.
void check_the_reference_space() {
    const THBSplineSpace<double> space = reference_space(true);
    PANTR_CHECK(space.dim() == 2);
    PANTR_CHECK(space.num_levels() == 3);
    PANTR_CHECK(space.truncate());
    PANTR_CHECK(space.degrees()[0] == 2 && space.degrees()[1] == 2);

    // The per-level counts sum to the total, and the offsets are their prefix sums.
    std::int64_t sum = 0;
    for (const std::int64_t n : space.num_basis_per_level()) {
        sum += n;
    }
    PANTR_CHECK(sum == space.num_total_basis());
    PANTR_CHECK(space.level_offsets().size() == 4);
    PANTR_CHECK(space.level_offsets()[0] == 0);
    PANTR_CHECK(space.level_offsets()[3] == space.num_total_basis());
    for (std::int64_t level = 0; level < space.num_levels(); ++level) {
        const auto l = static_cast<std::size_t>(level);
        PANTR_CHECK(static_cast<std::int64_t>(space.active_function_indices(level).size())
                    == space.num_basis_per_level()[l]);
        // Sorted and distinct, which the accessor's contract promises.
        const std::span<const std::int64_t> active = space.active_function_indices(level);
        PANTR_CHECK(std::is_sorted(active.begin(), active.end()));
        PANTR_CHECK(std::adjacent_find(active.begin(), active.end()) == active.end());
    }
    PANTR_CHECK(space.to_string()
                == "THBSplineSpace(dim=2, degrees=(2, 2), num_levels=3, num_total_basis="
                       + std::to_string(space.num_total_basis()) + ", truncate=True)");

    // A one-direction space's `repr` uses Python's one-element tuple spelling.
    const THBSplineSpace<double> flat(root_space<double>(1, 2, 4), corner_hierarchy(1, 4, 2, 1),
                                      true, {std::nullopt});
    PANTR_CHECK_MSG(flat.to_string().find("degrees=(2,)") != std::string::npos,
                    "a one-direction repr must spell its tuple Python's way: " + flat.to_string());
}

/// Every dof is assigned to the level whose offset range contains it.
void check_dof_levels_partition_the_basis() {
    const THBSplineSpace<double> space = reference_space(true);
    std::vector<std::int64_t> seen(static_cast<std::size_t>(space.num_levels()), 0);
    for (std::int64_t dof = 0; dof < space.num_total_basis(); ++dof) {
        const std::int64_t level = space.dof_level(dof);
        PANTR_CHECK(level >= 0 && level < space.num_levels());
        PANTR_CHECK(dof >= space.level_offsets()[static_cast<std::size_t>(level)]);
        PANTR_CHECK(dof < space.level_offsets()[static_cast<std::size_t>(level) + 1]);
        ++seen[static_cast<std::size_t>(level)];
    }
    for (std::int64_t level = 0; level < space.num_levels(); ++level) {
        PANTR_CHECK(seen[static_cast<std::size_t>(level)]
                    == space.num_basis_per_level()[static_cast<std::size_t>(level)]);
    }
}

/// The contribution table's three views agree with each other and with `active_basis`.
void check_the_contribution_table() {
    const THBSplineSpace<double> space = reference_space(true);
    const auto d = static_cast<std::size_t>(space.dim());
    std::int64_t widest = 0;
    for (std::int64_t cid = 0; cid < space.grid_ref().num_cells(); ++cid) {
        const pantr::bspline::CellContributions c = space.contributions(cid);
        PANTR_CHECK(c.dofs.size() == c.levels.size());
        PANTR_CHECK(c.multi_indices.size() == c.dofs.size() * d);
        PANTR_CHECK(std::is_sorted(c.dofs.begin(), c.dofs.end()));
        // `active_basis` is the same view, so it must be the same values.
        const std::span<const std::int64_t> basis = space.active_basis(cid);
        PANTR_CHECK(basis.size() == c.dofs.size());
        PANTR_CHECK(std::equal(basis.begin(), basis.end(), c.dofs.begin()));
        for (std::int64_t i = 0; i < c.size(); ++i) {
            const auto e = static_cast<std::size_t>(i);
            PANTR_CHECK(c.levels[e] == space.dof_level(c.dofs[e]));
            // Every recorded function is one the level's Kraft selection admits.
            const std::span<const std::int64_t> active =
                space.active_function_indices(c.levels[e]);
            const auto owning = static_cast<std::size_t>(c.levels[e]);
            const std::int64_t position = c.dofs[e] - space.level_offsets()[owning];
            std::int64_t flat = 0;
            for (std::size_t k = 0; k < d; ++k) {
                flat = flat * space.level_space_ref(c.levels[e])
                                  .space_ref(static_cast<std::int64_t>(k))
                                  .num_basis()
                       + c.multi_indices[e * d + k];
            }
            PANTR_CHECK(active[static_cast<std::size_t>(position)] == flat);
        }
        widest = std::max(widest, c.size());
    }
    PANTR_CHECK_MSG(widest == space.max_active_per_cell(),
                    "max_active_per_cell must be the same sweep's answer");
    // The vacuity guard: a grid with no cells would satisfy every loop above.
    PANTR_CHECK_MSG(space.grid_ref().num_cells() > 0 && widest > 0,
                    "this hierarchy has no cell with an active function, so the table was "
                    "never actually inspected");
}

/// The truncated basis sums to one; the untruncated one does not.
void check_the_truncation_is_a_partition_of_unity() {
    const THBSplineSpace<double> truncated = reference_space(true);
    const THBSplineSpace<double> plain = reference_space(false);

    // Bound: at most `prod(degree + 1) * num_levels` functions overlap one finest
    // function, each carrying `gamma_N` of its own value with `N` its chain length.
    std::int64_t widest = 1;
    for (std::int64_t dof = 0; dof < truncated.num_total_basis(); ++dof) {
        const std::optional<pantr::bspline::TruncatedView> view = truncated.truncated(dof);
        if (view.has_value()) {
            for (const std::int64_t n : view->shape) {
                widest = std::max(widest, n);
            }
        }
    }
    std::int64_t overlapping = truncated.num_levels();
    for (const std::int64_t degree : truncated.degrees()) {
        overlapping *= degree + 1;
    }
    const double bound = static_cast<double>(overlapping)
                         * gamma_of((truncated.num_levels() - 1) * truncated.dim() * widest);

    const double defect = partition_of_unity_defect(truncated);
    PANTR_CHECK_MSG(defect <= bound, "the truncated basis is not a partition of unity: defect "
                                         + std::to_string(defect) + " against "
                                         + std::to_string(bound));
    PANTR_CHECK_MSG(bound < 1.0e-10,
                    "the vacuity guard: this bound is loose enough to accept a basis that is "
                    "not a partition of unity at all");

    const double plain_defect = partition_of_unity_defect(plain);
    PANTR_CHECK_MSG(plain_defect > 0.1,
                    "the discriminator: the UNtruncated basis summed to within "
                        + std::to_string(plain_defect)
                        + " of one, so the identity above cannot tell the two bases apart");
    PANTR_CHECK_MSG(truncated.num_truncated() > 0,
                    "this hierarchy truncated nothing, so the identity says nothing about the "
                    "truncation");
    PANTR_CHECK_MSG(plain.num_truncated() == 0, "an untruncated space must store no coefficients");
}

/// Every truncation coefficient is non-negative, which the bound's derivation rests on.
///
/// The parity claim's relative form -- `gamma_n * s` rather than `gamma_n * sum|terms|` --
/// is licensed by there being no cancellation. That is a property of the values, so it is
/// checked rather than assumed.
void check_the_coefficients_never_go_negative() {
    std::int64_t inspected = 0;
    for (const std::int64_t factor : {2, 3}) {
        for (const std::int64_t degree : {1, 2, 3}) {
            const THBSplineSpace<double> space(root_space<double>(2, degree, 4),
                                               corner_hierarchy(2, 4, factor, 2), true,
                                               {std::nullopt, std::nullopt});
            for (std::int64_t dof = 0; dof < space.num_total_basis(); ++dof) {
                const std::optional<pantr::bspline::TruncatedView> view = space.truncated(dof);
                if (!view.has_value()) {
                    continue;
                }
                for (const double c : view->coeffs) {
                    PANTR_CHECK_MSG(c >= 0.0, "a truncation coefficient went negative, so the "
                                              "parity bound's no-cancellation premise is false");
                    ++inspected;
                }
            }
        }
    }
    PANTR_CHECK_MSG(inspected > 0, "no coefficient was inspected, so this asserts nothing");
}

/// `level_space(0)` is the handle the space was built from, by address.
void check_the_root_space_is_shared_not_copied() {
    const std::shared_ptr<const BsplineSpace<double>> root = root_space<double>(2, 2, 4);
    const THBSplineSpace<double> space(root, corner_hierarchy(2, 4, 2, 2), true,
                                       {std::nullopt, std::nullopt});
    PANTR_CHECK_MSG(space.root_space().get() == root.get(),
                    "the root space must be shared, not copied");
    PANTR_CHECK_MSG(space.level_space(0).get() == root.get(),
                    "level 0 must be the root handle itself: `thb.level_space(0) is "
                    "thb.root_space` is the oracle's contract");
    PANTR_CHECK_MSG(space.level_space(1).get() != root.get(),
                    "and the finer levels must be different objects, or the assertion above "
                    "would hold for a type returning one space for every level");
    PANTR_CHECK(&space.root_space_ref() == root.get());
}

/// A level space read after the owning THB space is destroyed is still valid.
///
/// `design/bspline_ownership_lifetime.md` F4: asserted on a **count** the destroyed
/// space's storage would have been reused for, rather than on the pointer being non-null,
/// because a scalar read after free returns the right answer often enough that the obvious
/// version of this test passes on a broken design. Its real gate is the `gcc-debug`
/// preset's address sanitizer.
void check_a_level_space_outlives_the_owner() {
    std::shared_ptr<const BsplineSpace<double>> escapee;
    std::int64_t expected = 0;
    {
        const THBSplineSpace<double> space = reference_space(true);
        escapee = space.level_space(2);
        expected = escapee->num_total_basis();
    }
    PANTR_CHECK(escapee != nullptr);
    PANTR_CHECK_MSG(escapee->num_total_basis() == expected,
                    "a level space handed out must survive its owner's destruction");
    PANTR_CHECK(expected > 0);
}

/// Refinement and coarsening never hand their result the receiver's grid.
///
/// Two spaces sharing one grid would make a tag set through the first visible through the
/// second. The no-op paths are where sharing would be the natural implementation, so they
/// are what is checked.
void check_the_operations_detach_the_grid() {
    const THBSplineSpace<double> space = reference_space(true);
    const std::vector<std::int64_t> nothing;

    const THBSplineSpace<double> refined =
        space.refine(std::span<const std::int64_t>(nothing), 2);
    PANTR_CHECK_MSG(refined.grid().get() != space.grid().get(),
                    "a refinement that refined nothing must still get a grid of its own");
    const THBSplineSpace<double> coarsened =
        space.coarsen(std::span<const std::int64_t>(nothing), 2);
    PANTR_CHECK_MSG(coarsened.grid().get() != space.grid().get(),
                    "a coarsening that coarsened nothing must still get a grid of its own");

    // The cell decomposition is unchanged either way, which is what makes those two
    // spaces equivalent rather than merely different objects.
    PANTR_CHECK(refined.grid()->num_cells() == space.grid()->num_cells());
    PANTR_CHECK(refined.num_total_basis() == space.num_total_basis());
    PANTR_CHECK(coarsened.num_total_basis() == space.num_total_basis());

    // The root space, by contrast, IS shared with the result: it is immutable, and
    // rebuilding it would break the identity contract one level up.
    PANTR_CHECK(refined.root_space().get() == space.root_space().get());
}

/// Refining then coarsening the children recovers the original space, ungraded.
///
/// The oracle documents this as the exact inverse when the admissibility guard is off,
/// and it is the one property of the pair that a count alone can state.
void check_refine_and_coarsen_invert_each_other() {
    const THBSplineSpace<double> space = reference_space(true);

    // Marked at the CURRENT deepest level, so the refinement opens a level that did not
    // exist and the cells of that new level are exactly the children it created. Marking
    // a level-0 cell instead would put the children among cells the reference hierarchy
    // already had, and "the deepest cells" would then name the wrong set -- which is what
    // a first version of this test did.
    const std::int64_t deepest = space.grid()->max_level();
    std::vector<std::int64_t> mark;
    for (std::int64_t cid = 0; cid < space.grid()->num_cells(); ++cid) {
        if (space.grid()->cell_level(cid) == deepest) {
            mark.push_back(cid);
            break;
        }
    }
    PANTR_CHECK_MSG(!mark.empty(), "the reference hierarchy has no cell at its own top level");

    const THBSplineSpace<double> refined =
        space.refine(std::span<const std::int64_t>(mark), std::nullopt);
    PANTR_CHECK(refined.grid()->max_level() == deepest + 1);
    PANTR_CHECK(refined.grid()->num_cells() > space.grid()->num_cells());

    std::vector<std::int64_t> children;
    for (std::int64_t cid = 0; cid < refined.grid()->num_cells(); ++cid) {
        if (refined.grid()->cell_level(cid) == deepest + 1) {
            children.push_back(cid);
        }
    }
    const THBSplineSpace<double> back =
        refined.coarsen(std::span<const std::int64_t>(children), std::nullopt);
    PANTR_CHECK_MSG(back.grid()->num_cells() == space.grid()->num_cells(),
                    "an ungraded coarsening of exactly what a refinement created must undo it");
    PANTR_CHECK(back.num_levels() == space.num_levels());
    PANTR_CHECK(back.num_total_basis() == space.num_total_basis());
    if (back.num_levels() != space.num_levels()) {
        return;
    }
    for (std::int64_t level = 0; level < space.num_levels(); ++level) {
        const std::span<const std::int64_t> before = space.active_function_indices(level);
        const std::span<const std::int64_t> after = back.active_function_indices(level);
        PANTR_CHECK(before.size() == after.size());
        PANTR_CHECK(before.size() == after.size()
                    && std::equal(before.begin(), before.end(), after.begin()));
    }
}

/// A `float` root space over the `double` grid, which is the oracle's shipped pairing.
void check_float_storage() {
    const THBSplineSpace<float> space(root_space<float>(2, 2, 4), corner_hierarchy(2, 4, 2, 2),
                                      true, {std::nullopt, std::nullopt});
    PANTR_CHECK(space.dim() == 2);
    PANTR_CHECK(space.num_levels() == 3);
    PANTR_CHECK(space.num_total_basis() > 0);
    PANTR_CHECK(space.num_truncated() > 0);
    // The coefficients are `double` whatever the root space stores, which is the oracle's
    // shape: `_build_oslo_matrices` widens the kernel's output before anything uses it.
    static_assert(
        std::is_same_v<
            std::remove_cvref_t<decltype(*space.truncated(0)->coeffs.begin())>, const double>
            || std::is_same_v<std::remove_cvref_t<decltype(space.truncated(0)->coeffs)>,
                              std::span<const double>>,
        "a truncation coefficient is a double at every storage width");
    // The domain is the root space's own storage, so it is `float` here.
    PANTR_CHECK(space.domain().size() == 4);
    PANTR_CHECK(static_cast<double>(space.domain()[0]) == 0.0);
}

/// What the constructor refuses, and with which message.
void check_refusals() {
    const std::shared_ptr<const BsplineSpace<double>> root = root_space<double>(2, 2, 4);
    const std::shared_ptr<HierarchicalGrid<double>> grid = corner_hierarchy(2, 4, 2, 1);

    bool threw = false;
    try {
        static_cast<void>(THBSplineSpace<double>(nullptr, grid, true, {std::nullopt}));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "root_space must not be null.";
    }
    PANTR_CHECK_MSG(threw, "a null root space must be refused");

    threw = false;
    try {
        static_cast<void>(THBSplineSpace<double>(root, nullptr, true, {std::nullopt}));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "grid must not be null.";
    }
    PANTR_CHECK_MSG(threw, "a null grid must be refused");

    threw = false;
    try {
        static_cast<void>(THBSplineSpace<double>(root, corner_hierarchy(1, 4, 2, 1), true,
                                                 {std::nullopt, std::nullopt}));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "grid.ndim (1) must equal root_space.dim (2).";
    }
    PANTR_CHECK_MSG(threw, "a dimension mismatch must be refused with the oracle's message");

    threw = false;
    try {
        static_cast<void>(THBSplineSpace<double>(root, corner_hierarchy(2, 5, 2, 1), true,
                                                 {std::nullopt, std::nullopt}));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what())
                == "grid root cells_per_axis (5, 5) must match root_space.num_intervals (4, 4).";
    }
    PANTR_CHECK_MSG(threw, "a root knot-span mismatch must be refused, naming both tuples");

    threw = false;
    try {
        static_cast<void>(THBSplineSpace<double>(root, grid, true, {std::nullopt}));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what())
                == "regularity must be a scalar or length-2 sequence; got length 1.";
    }
    PANTR_CHECK_MSG(threw, "a mis-shaped regularity must be refused");

    threw = false;
    try {
        static_cast<void>(THBSplineSpace<double>(root, grid, true, {2, std::nullopt}));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what())
                == "regularity[0]=2 must be in [-1, degree[0]-1=1]; got 2.";
    }
    PANTR_CHECK_MSG(threw, "a regularity at or above the degree must be refused");

    const THBSplineSpace<double> space = reference_space(true);
    threw = false;
    try {
        static_cast<void>(space.level_space(space.num_levels()));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a level past the top must be refused");

    threw = false;
    try {
        static_cast<void>(space.contributions(space.grid_ref().num_cells()));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a cell id past the end must be refused");

    threw = false;
    try {
        static_cast<void>(space.truncated(space.num_total_basis()));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a dof past the end must be refused");

    threw = false;
    const std::vector<std::int64_t> nothing;
    try {
        static_cast<void>(space.refine(std::span<const std::int64_t>(nothing), 1));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what())
                == "admissible_class must be an integer >= 2 or None; got 1.";
    }
    PANTR_CHECK_MSG(threw, "an admissibility class below the definition's minimum must be "
                           "refused with the oracle's message");

    // Every offending id, not the first: the oracle collects them all before raising,
    // and a caller debugging a list of them would otherwise get one per attempt.
    const std::int64_t past_the_end = space.grid_ref().num_cells();
    const std::vector<std::int64_t> bad = {past_the_end + 5, -1, past_the_end};
    for (const bool coarsening : {false, true}) {
        threw = false;
        try {
            if (coarsening) {
                static_cast<void>(space.coarsen(std::span<const std::int64_t>(bad), 2));
            } else {
                static_cast<void>(space.refine(std::span<const std::int64_t>(bad), 2));
            }
        } catch (const std::out_of_range& e) {
            threw = std::string(e.what())
                    == "cell_ids must lie in [0, " + std::to_string(past_the_end)
                           + "); got out-of-range id(s): [-1, " + std::to_string(past_the_end)
                           + ", " + std::to_string(past_the_end + 5) + "].";
        }
        PANTR_CHECK_MSG(threw, std::string(coarsening ? "coarsen" : "refine")
                                   + " must name every out-of-range id, sorted, as the "
                                     "oracle does");
    }
}

/// A factor of 1 on one axis leaves that direction alone at every level.
///
/// `HierarchicalGrid` admits it and the oracle's `_build_level_spaces` skips the axis
/// rather than subdividing it, so the level spaces share that direction's 1D space.
void check_an_unrefined_direction() {
    const auto root = root_space<double>(2, 2, 4);
    const std::vector<double> axis = {0.0, 0.25, 0.5, 0.75, 1.0};
    const std::vector<std::vector<double>> breakpoints(2, axis);
    const std::vector<std::int64_t> factors = {2, 1};
    auto grid = std::make_shared<HierarchicalGrid<double>>(
        TensorProductGrid<double>(breakpoints), std::span<const std::int64_t>(factors));
    const std::vector<std::int64_t> lo = {0, 0};
    const std::vector<std::int64_t> hi = {2, 2};
    grid = std::make_shared<HierarchicalGrid<double>>(
        grid->refine(0, std::span<const std::int64_t>(lo), std::span<const std::int64_t>(hi)));

    const THBSplineSpace<double> space(root, grid, true, {std::nullopt, std::nullopt});
    PANTR_CHECK(space.num_levels() == 2);
    PANTR_CHECK_MSG(space.level_space_ref(1).space(1).get()
                        == space.level_space_ref(0).space(1).get(),
                    "an axis whose factor is 1 must keep the same 1D space at every level");
    PANTR_CHECK_MSG(space.level_space_ref(1).space(0).get()
                        != space.level_space_ref(0).space(0).get(),
                    "and the refined axis must not");
    PANTR_CHECK(space.level_space_ref(1).space_ref(0).num_intervals() == 8);
    PANTR_CHECK(space.level_space_ref(1).space_ref(1).num_intervals() == 4);
}

/// Copying a space leaves the copy's contribution table cold, and correct.
///
/// `pantr::LazySlot` starts cold on copy on purpose: the memo is a function of its
/// owner's state, and one that travelled would have to be proved still to describe the
/// target.
void check_copy_leaves_the_memo_cold() {
    const THBSplineSpace<double> space = reference_space(true);
    const std::int64_t expected = space.max_active_per_cell();

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)  -- copying IS the test
    const THBSplineSpace<double> copy = space;
    PANTR_CHECK(copy.max_active_per_cell() == expected);
    PANTR_CHECK(copy.num_total_basis() == space.num_total_basis());
    // The copy shares the nested handles, which is what makes it cheap and what the
    // immutability of both makes safe.
    PANTR_CHECK(copy.root_space().get() == space.root_space().get());
    PANTR_CHECK(copy.grid().get() == space.grid().get());
}

/// Every accessor is safe to call concurrently, the lazy table included.
///
/// The only gate that can see a broken memo is a ThreadSanitizer build:
/// `design/bspline_derived_caches.md` F3 measured 60 correct answers in 60 unsanitized
/// runs from a shape that TSan reported four races in. So this asserts values *and* is
/// meant to be run under `--preset gcc-tsan`.
void check_concurrent_reads() {
    const THBSplineSpace<double> space = reference_space(true);
    constexpr int num_threads = 8;
    std::atomic<bool> go{false};
    std::vector<std::int64_t> widths(num_threads, -1);
    std::vector<std::int64_t> totals(num_threads, -1);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::int64_t sum = 0;
            for (std::int64_t cid = 0; cid < space.grid_ref().num_cells(); ++cid) {
                sum += static_cast<std::int64_t>(space.active_basis(cid).size());
            }
            widths[static_cast<std::size_t>(t)] = space.max_active_per_cell();
            totals[static_cast<std::size_t>(t)] = sum;
        });
    }
    go.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (int t = 0; t < num_threads; ++t) {
        const auto i = static_cast<std::size_t>(t);
        PANTR_CHECK_MSG(widths[i] == widths[0], "every thread must read one table");
        PANTR_CHECK_MSG(totals[i] == totals[0], "and one set of contribution lists");
    }
    PANTR_CHECK_MSG(totals[0] > 0, "the vacuity guard: the table was empty, so the threads "
                                   "contended over nothing");
}

}  // namespace

int main() {
    check_the_reference_space();
    check_dof_levels_partition_the_basis();
    check_the_contribution_table();
    check_the_truncation_is_a_partition_of_unity();
    check_the_coefficients_never_go_negative();
    check_the_root_space_is_shared_not_copied();
    check_a_level_space_outlives_the_owner();
    check_the_operations_detach_the_grid();
    check_refine_and_coarsen_invert_each_other();
    check_float_storage();
    check_refusals();
    check_an_unrefined_direction();
    check_copy_leaves_the_memo_cold();
    check_concurrent_reads();
    return pantr::test::summary("test_thb_space");
}
