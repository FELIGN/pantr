/// \file
/// The tensor-product spanwise element extraction: what it compacts, what it
/// decompresses, and what it refuses.
///
/// ## What is asserted, and against what
///
/// Nothing here carries a numerical tolerance, and that is a property of the type
/// rather than of the cases. This type performs **no arithmetic at all**: the
/// compaction selects rows and copies them, the decompression copies them back and
/// writes exact `1` and `0`, and every other quantity is a count or a flag. So `==`
/// is the right comparison for every assertion below, and the operator values are
/// small integers chosen so that a transposition or a wrong row is visible by
/// reading a failure message rather than by decoding one.
///
/// The operators are supplied rather than built. That is the type's own seam --
/// `cpp/include/pantr/bspline/spanwise_extraction.hpp` states why -- so the cases
/// here are free to use operator entries no real extraction would produce, which is
/// exactly what makes a misplaced row detectable.
///
/// ## The case set is asymmetric wherever a symmetry would hide a transposition
///
/// Both multi-direction cases differ between directions in the element count, the
/// operator shape **and** the identity pattern at once. A symmetric case would pass
/// under an axis swap, an off-by-one in the direction index, and a reduction that
/// returned its first argument.
///
/// `check_rectangular_operators` is the one case with `n_out != n_in`, and it
/// exists because `numpy.eye(n_out, n_in)` is what the oracle fills an identity
/// element with. A square-only case set would admit an implementation that wrote a
/// square identity and then read it at the wrong stride.
///
/// ## The four cases that exist because getting them wrong is silent
///
/// **`check_the_all_identity_sentinel`.** A direction with no non-identity element
/// still gets one compact row, of zeros. The oracle does this so that
/// `compact[idx_map[e]]` is in range for a numba kernel that computed the index
/// before it checked the mask. No caller ever reads the row, so an implementation
/// that allocated zero rows would pass every value assertion here and fail only
/// inside a kernel, out of process, as an out-of-bounds read.
///
/// **`check_the_space_is_shared_not_copied`.** Compared by address, because the
/// Python identity contract `extraction.space is space` rests on the C++ type
/// storing the handle rather than a copy of what it points at, and every value
/// assertion in this file would pass either way.
///
/// **`check_the_dense_block_is_the_memo_itself`.** The accessor must return a view
/// *of* the memo, never of a copy: `design/bspline_derived_caches.md` assigns
/// `ops_1d` that shape, and `tests/test_spanwise_element_extraction.py`'s
/// `np.shares_memory` assertion is what reads it from Python. Two calls returning
/// equal values over different addresses would satisfy every other check here.
///
/// **`check_concurrent_reads`.** The header claims the memo is filled at most once
/// and published atomically. Here that is a real claim rather than a structural one,
/// because there *is* a memo; the gate is a sanitizer build rather than an assertion
/// on a value, for the reason `design/bspline_derived_caches.md` F3 records --
/// the unsynchronised shape gave 60 correct answers in 60 unsanitized runs.

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/bspline/spanwise_extraction.hpp"

namespace {

using pantr::span_nd;
using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::ExtractionTarget;
using pantr::bspline::KnotSnapping;
using pantr::bspline::SpanwiseElementExtraction;

using Extraction = SpanwiseElementExtraction<double>;
using Input = Extraction::DirectionInput;

/// Build a shared univariate space from a knot vector and a degree.
///
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \return A handle on the space.
std::shared_ptr<const BsplineSpace1D<double>> one_d(const std::vector<double>& knots,
                                                    std::int64_t degree) {
    return std::make_shared<const BsplineSpace1D<double>>(
        std::span<const double>(knots), degree, false, KnotSnapping::merge_near_duplicates);
}

/// An open uniform knot vector of the given degree over `n_intervals` unit spans.
///
/// \param degree The polynomial degree.
/// \param n_intervals The interval count; at least one.
/// \return The knot vector.
std::vector<double> open_knots(std::int64_t degree, std::int64_t n_intervals) {
    std::vector<double> knots;
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(0.0);
    }
    for (std::int64_t i = 1; i < n_intervals; ++i) {
        knots.push_back(static_cast<double>(i));
    }
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(static_cast<double>(n_intervals));
    }
    return knots;
}

/// Heap-allocated identity flags a `std::span<const bool>` can view.
///
/// `std::vector<bool>` is a bitset with no `bool` objects to point at, and a `char`
/// buffer reinterpreted as `bool*` is not something the language permits. A plain
/// array is the one spelling that is both legal and short, and it is what the type
/// under test uses for the same reason.
struct Flags {
    std::unique_ptr<bool[]> values;  ///< The flags.
    std::size_t size = 0;            ///< How many.

    /// The flags as a span.
    ///
    /// \return A view of `size` flags.
    [[nodiscard]] std::span<const bool> view() const {
        return std::span<const bool>(values.get(), size);
    }
};

/// Copy a flag list into heap-allocated `bool`s.
///
/// \param flags The flags, in element order.
/// \return The allocation and its size.
Flags flags_of(std::initializer_list<bool> flags) {
    Flags out;
    out.size = flags.size();
    out.values = std::make_unique<bool[]>(out.size);
    std::size_t i = 0;
    for (const bool flag : flags) {
        out.values[i++] = flag;
    }
    return out;
}

/// Operator storage plus the flags, so a case can keep both alive together.
struct Block {
    std::vector<double> values;  ///< Row-major `(n_elements, n_out, n_in)`.
    Flags mask;                  ///< `n_elements` identity flags.
    std::int64_t n_elements = 0;
    std::int64_t n_out = 0;
    std::int64_t n_in = 0;

    /// The dense operator view this block backs.
    ///
    /// \return A `(n_elements, n_out, n_in)` view of `values`.
    [[nodiscard]] span_nd<const double, 3> operators() const {
        return span_nd<const double, 3>(values.data(), static_cast<std::size_t>(n_elements),
                                        static_cast<std::size_t>(n_out),
                                        static_cast<std::size_t>(n_in));
    }

    /// This block as a constructor argument.
    ///
    /// \return The direction input.
    [[nodiscard]] Input input() const { return Input{operators(), mask.view()}; }
};

/// Build a block whose element `e` carries the constant matrix `base + e`.
///
/// A distinct constant per element is what makes a misplaced row nameable: a failure
/// reports the element it actually read rather than a plausible-looking matrix.
///
/// \param identity The per-element identity flags; its length is the element count.
/// \param n_out The operator row count.
/// \param n_in The operator column count.
/// \param base The value element 0's entries take.
/// \return The block.
Block block_of(std::initializer_list<bool> identity, std::int64_t n_out, std::int64_t n_in,
               double base) {
    Block b;
    b.n_elements = static_cast<std::int64_t>(identity.size());
    b.n_out = n_out;
    b.n_in = n_in;
    b.mask = flags_of(identity);
    b.values.reserve(identity.size() * static_cast<std::size_t>(n_out * n_in));
    for (std::size_t e = 0; e < identity.size(); ++e) {
        for (std::int64_t k = 0; k < n_out * n_in; ++k) {
            b.values.push_back(base + static_cast<double>(e));
        }
    }
    return b;
}

/// The value at `(element, row, col)` of a 3D view.
///
/// \param view The view.
/// \param e The element.
/// \param i The row.
/// \param j The column.
/// \return The entry.
double entry(span_nd<const double, 3> view, std::size_t e, std::size_t i, std::size_t j) {
    return pantr::at(view, e, i, j);
}

/// Two directions that differ in every quantity a reduction could confuse.
void check_two_directions() {
    // Direction 0: 3 elements, 2x2 operators, only element 1 is the identity.
    // Direction 1: 2 elements, 3x3 operators, neither is the identity.
    const std::vector<double> k0 = open_knots(1, 3);
    const std::vector<double> k1 = open_knots(2, 2);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(k0, 1), one_d(k1, 2)});

    const Block b0 = block_of({false, true, false}, 2, 2, 10.0);
    const Block b1 = block_of({false, false}, 3, 3, 100.0);
    const std::vector<Input> inputs{b0.input(), b1.input()};
    const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                         std::span<const Input>(inputs));

    PANTR_CHECK(ext.dim() == 2);
    PANTR_CHECK(ext.target() == ExtractionTarget::bezier);
    PANTR_CHECK(ext.lagrange_variant() == "equispaces");
    PANTR_CHECK(ext.num_intervals()[0] == 3 && ext.num_intervals()[1] == 2);
    PANTR_CHECK(ext.num_total_intervals() == 6);
    PANTR_CHECK(ext.input_shape_per_dir()[0] == 2 && ext.input_shape_per_dir()[1] == 3);
    PANTR_CHECK(ext.output_shape_per_dir()[0] == 2 && ext.output_shape_per_dir()[1] == 3);

    // Direction 0 keeps elements 0 and 2, in that order, at compact rows 0 and 1.
    const span_nd<const double, 3> c0 = ext.compact_ops(0);
    PANTR_CHECK(c0.extent(0) == 2 && c0.extent(1) == 2 && c0.extent(2) == 2);
    PANTR_CHECK(entry(c0, 0, 0, 0) == 10.0);
    PANTR_CHECK(entry(c0, 1, 1, 1) == 12.0);
    PANTR_CHECK(ext.idx_map(0)[0] == 0 && ext.idx_map(0)[2] == 1);
    PANTR_CHECK_MSG(ext.idx_map(0)[1] == 0,
                    "an identity element's index map entry is the oracle's unused zero");
    PANTR_CHECK(ext.is_identity_mask(0)[0] == false && ext.is_identity_mask(0)[1] == true
                && ext.is_identity_mask(0)[2] == false);

    // Direction 1 keeps both of its elements.
    const span_nd<const double, 3> c1 = ext.compact_ops(1);
    PANTR_CHECK(c1.extent(0) == 2 && c1.extent(1) == 3 && c1.extent(2) == 3);
    PANTR_CHECK(entry(c1, 0, 2, 2) == 100.0 && entry(c1, 1, 0, 2) == 101.0);
    PANTR_CHECK(ext.idx_map(1)[0] == 0 && ext.idx_map(1)[1] == 1);

    // One identity element in direction 0, none in direction 1.
    PANTR_CHECK(ext.num_identity_elements() == 0);
    PANTR_CHECK(!ext.is_identity());
}

/// Three directions, again asymmetric, to catch a reduction that stops at two.
void check_three_directions() {
    const std::vector<double> k0 = open_knots(1, 2);
    const std::vector<double> k1 = open_knots(2, 3);
    const std::vector<double> k2 = open_knots(1, 4);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(k0, 1), one_d(k1, 2),
                                                                   one_d(k2, 1)});

    const Block b0 = block_of({true, false}, 2, 2, 1.0);          // 1 identity of 2
    const Block b1 = block_of({true, true, false}, 3, 3, 10.0);   // 2 identities of 3
    const Block b2 = block_of({true, true, true, false}, 2, 2, 100.0);  // 3 of 4
    const std::vector<Input> inputs{b0.input(), b1.input(), b2.input()};
    const Extraction ext(space, ExtractionTarget::lagrange, "gauss_lobatto_legendre",
                         std::span<const Input>(inputs));

    PANTR_CHECK(ext.dim() == 3);
    PANTR_CHECK(ext.target() == ExtractionTarget::lagrange);
    PANTR_CHECK(ext.lagrange_variant() == "gauss_lobatto_legendre");
    PANTR_CHECK_MSG(ext.num_identity_elements() == 1 * 2 * 3,
                    "the count is the product of the per-direction identity counts");
    PANTR_CHECK(!ext.is_identity());
    // Each direction keeps exactly one compact row, and it is its last element.
    for (std::int64_t d = 0; d < 3; ++d) {
        PANTR_CHECK(ext.compact_ops(d).extent(0) == 1);
        const std::int64_t last = ext.num_intervals()[static_cast<std::size_t>(d)] - 1;
        PANTR_CHECK(ext.idx_map(d)[static_cast<std::size_t>(last)] == 0);
    }
}

/// A direction with no non-identity element still carries one compact row, of zeros.
void check_the_all_identity_sentinel() {
    const std::vector<double> knots = open_knots(1, 2);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(knots, 1)});

    const Block b = block_of({true, true}, 2, 2, 7.0);
    const std::vector<Input> inputs{b.input()};
    const Extraction ext(space, ExtractionTarget::cardinal, "", std::span<const Input>(inputs));

    const span_nd<const double, 3> compact = ext.compact_ops(0);
    PANTR_CHECK_MSG(compact.extent(0) == 1,
                    "the sentinel row exists so a kernel's precomputed index is in range");
    PANTR_CHECK(entry(compact, 0, 0, 0) == 0.0 && entry(compact, 0, 1, 1) == 0.0);
    PANTR_CHECK(ext.idx_map(0)[0] == 0 && ext.idx_map(0)[1] == 0);
    PANTR_CHECK(ext.num_identity_elements() == 2);
    PANTR_CHECK(ext.is_identity());

    // Every element decompresses to the identity, which is what the mask says.
    const span_nd<const double, 3> dense = ext.ops_1d(0);
    PANTR_CHECK(dense.extent(0) == 2);
    for (std::size_t e = 0; e < 2; ++e) {
        PANTR_CHECK(entry(dense, e, 0, 0) == 1.0 && entry(dense, e, 1, 1) == 1.0);
        PANTR_CHECK(entry(dense, e, 0, 1) == 0.0 && entry(dense, e, 1, 0) == 0.0);
    }
}

/// The dense block reads identities as `eye` and everything else from compact storage.
void check_the_dense_block() {
    const std::vector<double> knots = open_knots(1, 3);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(knots, 1)});

    const Block b = block_of({false, true, false}, 2, 2, 10.0);
    const std::vector<Input> inputs{b.input()};
    const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                         std::span<const Input>(inputs));

    const span_nd<const double, 3> dense = ext.ops_1d(0);
    PANTR_CHECK(dense.extent(0) == 3 && dense.extent(1) == 2 && dense.extent(2) == 2);
    PANTR_CHECK(entry(dense, 0, 0, 0) == 10.0 && entry(dense, 0, 1, 0) == 10.0);
    PANTR_CHECK_MSG(entry(dense, 1, 0, 0) == 1.0 && entry(dense, 1, 0, 1) == 0.0
                        && entry(dense, 1, 1, 0) == 0.0 && entry(dense, 1, 1, 1) == 1.0,
                    "the identity element reads as the identity matrix");
    PANTR_CHECK_MSG(entry(dense, 2, 0, 0) == 12.0,
                    "element 2 reads element 2's operator, not compact row 1's neighbour");
}

/// `numpy.eye(n_out, n_in)` is rectangular, and so is what an identity element reads.
void check_rectangular_operators() {
    const std::vector<double> knots = open_knots(1, 2);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(knots, 1)});

    // Two rows, three columns: the identity has ones at (0,0) and (1,1) and nothing
    // in column 2.
    const Block b = block_of({true, false}, 2, 3, 5.0);
    const std::vector<Input> inputs{b.input()};
    const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                         std::span<const Input>(inputs));

    PANTR_CHECK(ext.input_shape_per_dir()[0] == 3 && ext.output_shape_per_dir()[0] == 2);
    const span_nd<const double, 3> dense = ext.ops_1d(0);
    PANTR_CHECK(dense.extent(1) == 2 && dense.extent(2) == 3);
    PANTR_CHECK(entry(dense, 0, 0, 0) == 1.0 && entry(dense, 0, 1, 1) == 1.0);
    PANTR_CHECK_MSG(entry(dense, 0, 0, 2) == 0.0 && entry(dense, 0, 1, 2) == 0.0,
                    "the rectangular identity has no entry outside the main diagonal");
    PANTR_CHECK(entry(dense, 1, 1, 2) == 6.0);
}

/// The accessor returns a view of the memo, not of a fresh copy.
void check_the_dense_block_is_the_memo_itself() {
    const std::vector<double> knots = open_knots(1, 2);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(knots, 1)});

    const Block b = block_of({false, false}, 2, 2, 3.0);
    const std::vector<Input> inputs{b.input()};
    const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                         std::span<const Input>(inputs));

    const double* first = ext.ops_1d(0).data_handle();
    const double* second = ext.ops_1d(0).data_handle();
    PANTR_CHECK_MSG(first == second,
                    "two reads of ops_1d view one block, which is what makes the Python "
                    "np.shares_memory assertion true");
}

/// The space is stored by handle, and outlives the extraction that shared it.
void check_the_space_is_shared_not_copied() {
    const std::vector<double> knots = open_knots(1, 2);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(knots, 1)});

    const Block b = block_of({false, false}, 2, 2, 1.0);
    const std::vector<Input> inputs{b.input()};
    {
        const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                             std::span<const Input>(inputs));
        PANTR_CHECK_MSG(ext.space().get() == space.get(),
                        "the extraction shares the caller's space rather than copying it");
        PANTR_CHECK(&ext.space_ref() == space.get());
        PANTR_CHECK(space.use_count() >= 2);
    }
    PANTR_CHECK_MSG(space->dim() == 1, "the space is intact after the extraction is gone");
}

/// A space handed out survives the extraction it came from.
void check_a_space_outlives_the_extraction() {
    std::shared_ptr<const BsplineSpace<double>> escaped;
    const std::vector<double> knots = open_knots(1, 2);
    const Block b = block_of({false, false}, 2, 2, 1.0);
    const std::vector<Input> inputs{b.input()};
    {
        const auto space = std::make_shared<const BsplineSpace<double>>(
            std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(knots, 1)});
        const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                             std::span<const Input>(inputs));
        escaped = ext.space();
    }
    PANTR_CHECK(escaped != nullptr && escaped->dim() == 1);
    PANTR_CHECK(escaped->num_intervals()[0] == 2);
}

/// A dimensionless space is legal, and its empty products are the oracle's.
void check_no_directions() {
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{});
    const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                         std::span<const Input>{});

    PANTR_CHECK(ext.dim() == 0);
    PANTR_CHECK(ext.num_total_intervals() == 1);
    PANTR_CHECK_MSG(ext.num_identity_elements() == 1, "the empty product is one");
    PANTR_CHECK_MSG(ext.is_identity(), "every one of no directions is all-identity");
}

/// `float` storage behaves as `double` storage does.
void check_float_storage() {
    using Ext32 = SpanwiseElementExtraction<float>;
    const std::vector<float> knots{0.0F, 0.0F, 1.0F, 2.0F, 2.0F};
    const auto one = std::make_shared<const BsplineSpace1D<float>>(
        std::span<const float>(knots), 1, false, KnotSnapping::merge_near_duplicates);
    const auto space = std::make_shared<const BsplineSpace<float>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<float>>>{one});

    const Flags mask = flags_of({false, true});
    const std::vector<float> values{2.0F, 2.0F, 2.0F, 2.0F, 9.0F, 9.0F, 9.0F, 9.0F};
    const std::vector<Ext32::DirectionInput> inputs{Ext32::DirectionInput{
        span_nd<const float, 3>(values.data(), 2, 2, 2), mask.view()}};
    const Ext32 ext(space, ExtractionTarget::bezier, "equispaces",
                    std::span<const Ext32::DirectionInput>(inputs));

    PANTR_CHECK(ext.compact_ops(0).extent(0) == 1);
    PANTR_CHECK(pantr::at(ext.compact_ops(0), 0U, 0U, 0U) == 2.0F);
    PANTR_CHECK(pantr::at(ext.ops_1d(0), 1U, 0U, 0U) == 1.0F);
    PANTR_CHECK(ext.num_identity_elements() == 1);
}

/// Every refusal the type owns, and the message it carries.
void check_refusals() {
    const std::vector<double> knots = open_knots(1, 2);
    const auto one = one_d(knots, 1);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one});
    const Block good = block_of({false, false}, 2, 2, 1.0);

    bool threw = false;
    try {
        const std::vector<Input> inputs{good.input()};
        const Extraction ext(nullptr, ExtractionTarget::bezier, "", std::span<const Input>(inputs));
        (void)ext.dim();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a null space is refused");

    threw = false;
    try {
        const Extraction ext(space, ExtractionTarget::bezier, "", std::span<const Input>{});
        (void)ext.dim();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "one bundle per direction is required");

    threw = false;
    try {
        // Three operators for a two-element direction.
        const Block wrong = block_of({false, false, false}, 2, 2, 1.0);
        const std::vector<Input> inputs{wrong.input()};
        const Extraction ext(space, ExtractionTarget::bezier, "", std::span<const Input>(inputs));
        (void)ext.dim();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "the operator count must match the element count");

    threw = false;
    try {
        // A mask of the wrong length against operators of the right count.
        const Flags mask = flags_of({false});
        const std::vector<Input> inputs{Input{good.operators(), mask.view()}};
        const Extraction ext(space, ExtractionTarget::bezier, "", std::span<const Input>(inputs));
        (void)ext.dim();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "the mask length must match the element count");

    threw = false;
    try {
        const Block flat = block_of({false, false}, 2, 0, 1.0);
        const std::vector<Input> inputs{flat.input()};
        const Extraction ext(space, ExtractionTarget::bezier, "", std::span<const Input>(inputs));
        (void)ext.dim();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "operators with no columns are refused");

    const std::vector<Input> inputs{good.input()};
    const Extraction ext(space, ExtractionTarget::bezier, "", std::span<const Input>(inputs));
    threw = false;
    try {
        (void)ext.compact_ops(1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a direction outside [0, dim) is refused");
    threw = false;
    try {
        (void)ext.ops_1d(-1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a negative direction is refused");
}

/// `is_extraction_target` accepts exactly the three members.
void check_the_target_range() {
    PANTR_CHECK(pantr::bspline::is_extraction_target(0));
    PANTR_CHECK(pantr::bspline::is_extraction_target(1));
    PANTR_CHECK(pantr::bspline::is_extraction_target(2));
    PANTR_CHECK(!pantr::bspline::is_extraction_target(-1));
    PANTR_CHECK(!pantr::bspline::is_extraction_target(3));
}

/// Many threads first-touching the memo agree, and race nothing.
void check_concurrent_reads() {
    const std::vector<double> knots = open_knots(2, 5);
    const auto space = std::make_shared<const BsplineSpace<double>>(
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{one_d(knots, 2)});
    const Block b = block_of({false, true, false, true, false}, 3, 3, 1.0);
    const std::vector<Input> inputs{b.input()};
    const Extraction ext(space, ExtractionTarget::bezier, "equispaces",
                         std::span<const Input>(inputs));

    constexpr int kThreads = 8;
    std::atomic<int> agreed{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&ext, &agreed] {
            const span_nd<const double, 3> dense = ext.ops_1d(0);
            const bool ok = dense.extent(0) == 5 && entry(dense, 0, 0, 0) == 1.0
                            && entry(dense, 1, 0, 0) == 1.0 && entry(dense, 1, 0, 1) == 0.0
                            && entry(dense, 4, 2, 2) == 5.0;
            if (ok) {
                agreed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    PANTR_CHECK(agreed.load() == kThreads);
}

}  // namespace

int main() {
    check_two_directions();
    check_three_directions();
    check_the_all_identity_sentinel();
    check_the_dense_block();
    check_rectangular_operators();
    check_the_dense_block_is_the_memo_itself();
    check_the_space_is_shared_not_copied();
    check_a_space_outlives_the_extraction();
    check_no_directions();
    check_float_storage();
    check_refusals();
    check_the_target_range();
    check_concurrent_reads();
    return pantr::test::summary("test_spanwise_extraction");
}
