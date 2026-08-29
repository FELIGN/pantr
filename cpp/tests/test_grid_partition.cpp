/// \file
/// Properties of the cell-ownership partition.
///
/// ## Why these cases and not others
///
/// A partition holds integers and answers two questions about them, so there is no
/// floating point, no rounding and no tolerance anywhere below. What it can get
/// wrong is small and entirely at the edges, which is what these groups are:
///
///  - **The two range checks, on both sides.** `-1` is a legal owner and `n_parts`
///    is not, so the interesting values are `-2`, `-1`, `n_parts - 1` and
///    `n_parts`; the same for `owned_cells`' `rank`. An off-by-one on either
///    boundary passes every ordinary case, which is why they are checked one at a
///    time rather than by a sweep.
///  - **The empty partition.** Zero cells is legal, and it is the input that would
///    make a `min`/`max` over the owners read past the end. `owned_cells` on it
///    must answer with an empty list rather than refuse.
///  - **Ordering and completeness of `owned_cells`.** The oracle promises ascending
///    ids, and the distributed-space machinery indexes with the result.
///  - **Validation at all.** This type is the C++ counterpart of Layer 2: a caller
///    with no Python is protected by these throws and by nothing else.

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/grid/partition.hpp"

namespace {

using pantr::grid::Partition;

/// Build a partition from a vector, for brevity below.
///
/// \param owners Per-cell owner ranks.
/// \param n_parts Number of parts.
/// \return The partition.
Partition part(const std::vector<std::int32_t>& owners, std::int64_t n_parts) {
    return Partition(std::span<const std::int32_t>(owners), n_parts);
}

/// Whether calling `fn` throws `std::invalid_argument`.
///
/// \tparam Fn The callable's type.
/// \param fn The call to attempt.
/// \return `true` when it threw.
template <class Fn>
bool rejects(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

/// The constructor accepts what the oracle accepts and refuses what it refuses.
void test_construction_validates() {
    const std::vector<std::int32_t> ok{0, 1, 0, -1, 1};
    PANTR_CHECK(part(ok, 2).n_cells() == 5);
    PANTR_CHECK(part(ok, 2).n_parts() == 2);

    PANTR_CHECK(rejects([&] { return part(ok, 0); }));
    PANTR_CHECK(rejects([&] { return part(ok, -3); }));
    // `n_parts` itself is out of range as an owner, and `-2` is one below the
    // inactive marker: the two ends of the same interval.
    PANTR_CHECK(rejects([] { return part({0, 2}, 2); }));
    PANTR_CHECK(rejects([] { return part({-2, 0}, 2); }));
    // ... and the two values just inside it are accepted, so the guard cannot pass
    // merely by refusing everything.
    PANTR_CHECK(part({-1, 1}, 2).n_cells() == 2);
}

/// A zero-cell partition is legal, and nothing reads past its end.
void test_the_empty_partition_is_legal() {
    const Partition empty = part({}, 1);
    PANTR_CHECK(empty.n_cells() == 0);
    PANTR_CHECK(empty.cell_owner().empty());
    PANTR_CHECK(empty.owned_cells(0).empty());
}

/// `owned_cells` returns exactly the cells of one rank, ascending.
void test_owned_cells_is_ascending_and_complete() {
    const Partition p = part({0, 1, 0, -1, 1, 0}, 2);
    PANTR_CHECK(p.owned_cells(0) == std::vector<std::int64_t>({0, 2, 5}));
    PANTR_CHECK(p.owned_cells(1) == std::vector<std::int64_t>({1, 4}));

    // A rank that owns nothing is an ordinary answer, not an error.
    PANTR_CHECK(part({0, 0, -1}, 3).owned_cells(2).empty());
}

/// `owned_cells` refuses a rank outside `[0, n_parts)`, on both sides.
void test_owned_cells_validates_its_rank() {
    const Partition p = part({0, 1}, 2);
    PANTR_CHECK(rejects([&] { return p.owned_cells(2); }));
    PANTR_CHECK(rejects([&] { return p.owned_cells(-1); }));
    PANTR_CHECK(p.owned_cells(1) == std::vector<std::int64_t>({1}));
}

/// The stored owners are the caller's values, copied rather than aliased.
void test_the_owners_are_copied() {
    std::vector<std::int32_t> owners{0, 1};
    const Partition p = part(owners, 2);
    owners[0] = 1;
    PANTR_CHECK(p.cell_owner()[0] == 0);
}

/// The representation names the two counts and not the owners.
void test_to_string_names_the_counts() {
    PANTR_CHECK_MSG(part({0, 1, -1}, 2).to_string() == "Partition(n_cells=3, n_parts=2)",
                    part({0, 1, -1}, 2).to_string());
}

}  // namespace

int main() {
    test_construction_validates();
    test_the_empty_partition_is_legal();
    test_owned_cells_is_ascending_and_complete();
    test_owned_cells_validates_its_rank();
    test_the_owners_are_copied();
    test_to_string_names_the_counts();
    return pantr::test::summary("test_grid_partition");
}
