/// \file
/// Properties of the cell and facet tag registries.
///
/// ## Why these cases and not others
///
/// There is no floating point here at all, so nothing in this file is about
/// rounding and no assertion carries a tolerance. What the registries can get
/// wrong is structural, and the groups below are the four places the port could
/// have got it wrong without any test in the Python suite noticing:
///
///  - **Lifetime.** A handle taken from `get` must survive a `set` that replaces
///    the same name, and a `remove` that deletes it. The Python oracle gets this
///    from reference counting and cannot fail it; this side gets it from a
///    `shared_ptr` and would otherwise hand a caller freed memory. Nothing in the
///    Python suite reaches it except through the binding, so it is checked here
///    where the ownership actually lives.
///  - **Insertion order across a replacement.** `names()` is public, so the order
///    is a contract. A `std::unordered_map` would pass every value assertion in
///    the suite and fail this one.
///  - **Validation.** The type is the C++ counterpart of Layer 2: a caller with no
///    Python is protected by these throws and by nothing else.
///  - **`scatter`'s width behaviour.** The overflow check fires below eight bytes
///    and, reproducing the oracle, does not fire at eight -- so a negative value
///    scattered into a `uint64` destination wraps rather than throwing. That is
///    asserted rather than left implicit, because it is the one place this file
///    pins behaviour the header itself calls a contract gap.

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/tags.hpp"

namespace {

using pantr::grid::CellTags;
using pantr::grid::FacetTags;

/// View a vector as a read-only span, for brevity below.
///
/// \param v The vector.
/// \return A span over it.
std::span<const std::int64_t> as_span(const std::vector<std::int64_t>& v) {
    return {v.data(), v.size()};
}

/// View a flat vector as an `(M, 2)` key block.
///
/// \param v The flattened rows.
/// \return A 2D view over them.
pantr::span2d<const std::int64_t> as_keys(const std::vector<std::int64_t>& v) {
    return pantr::span2d<const std::int64_t>(v.data(), v.size() / 2, 2);
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

/// Whether calling `fn` throws `std::out_of_range`.
///
/// \tparam Fn The callable's type.
/// \param fn The call to attempt.
/// \return `true` when it threw.
template <class Fn>
bool reports_missing(Fn&& fn) {
    try {
        fn();
    } catch (const std::out_of_range&) {
        return true;
    }
    return false;
}

/// Whether calling `fn` throws `std::overflow_error`.
///
/// \tparam Fn The callable's type.
/// \param fn The call to attempt.
/// \return `true` when it threw.
template <class Fn>
bool overflows(Fn&& fn) {
    try {
        fn();
    } catch (const std::overflow_error&) {
        return true;
    }
    return false;
}

/// The constructors reject what the oracle rejects.
void test_construction_validates() {
    PANTR_CHECK(rejects([] { return CellTags(-1); }));
    PANTR_CHECK(rejects([] { return FacetTags(-1, 4); }));
    PANTR_CHECK(rejects([] { return FacetTags(4, 0); }));
    PANTR_CHECK(CellTags(0).num_cells() == 0);
    PANTR_CHECK(FacetTags(3, 4).facets_per_cell() == 4);
}

/// A cell tag stores its pair sorted by id, and reads it back.
void test_cell_set_sorts_by_id() {
    CellTags tags(10);
    const std::vector<std::int64_t> ids{4, 1, 7};
    const std::vector<std::int64_t> values{2, 1, 2};
    tags.set("location", as_span(ids), as_span(values));

    const auto tag = tags.get("location");
    PANTR_CHECK(tag->ids == std::vector<std::int64_t>({1, 4, 7}));
    PANTR_CHECK(tag->values == std::vector<std::int64_t>({1, 2, 2}));
    PANTR_CHECK(tags.size() == 1);
    PANTR_CHECK(tags.contains("location"));
    PANTR_CHECK(!tags.contains("other"));
}

/// An empty tag is legal and stores two empty arrays.
void test_an_empty_tag_is_legal() {
    CellTags tags(4);
    tags.set("none", {}, {});
    PANTR_CHECK(tags.get("none")->ids.empty());

    FacetTags facets(4, 4);
    facets.set("none", pantr::span2d<const std::int64_t>(nullptr, 0, 2), {});
    PANTR_CHECK(facets.get("none")->rows() == 0);
}

/// Replacing a tag keeps its position among the names.
///
/// The failure this catches is a registry that appends on replace: every value
/// assertion in the suite still passes, and `names()` silently reorders.
void test_replacing_a_tag_keeps_its_position() {
    CellTags tags(10);
    const std::vector<std::int64_t> one{1};
    const std::vector<std::int64_t> two{2};
    tags.set("a", as_span(one), as_span(one));
    tags.set("b", as_span(one), as_span(one));
    tags.set("c", as_span(one), as_span(one));
    tags.set("b", as_span(two), as_span(two));

    PANTR_CHECK(tags.names() == std::vector<std::string>({"a", "b", "c"}));
    PANTR_CHECK(tags.get("b")->ids == std::vector<std::int64_t>({2}));

    tags.remove("a");
    PANTR_CHECK(tags.names() == std::vector<std::string>({"b", "c"}));
}

/// A handle taken from `get` outlives a replacement and a removal.
///
/// This is what the `shared_ptr` is for; without it the binding's zero-copy views
/// would dangle the moment a caller re-tagged the same name.
void test_a_handle_outlives_a_replacement() {
    CellTags tags(10);
    const std::vector<std::int64_t> first{1, 2};
    const std::vector<std::int64_t> second{5};
    tags.set("a", as_span(first), as_span(first));

    const std::shared_ptr<const CellTags::Tag> held = tags.get("a");
    tags.set("a", as_span(second), as_span(second));
    PANTR_CHECK(held->ids == std::vector<std::int64_t>({1, 2}));
    PANTR_CHECK(tags.get("a")->ids == std::vector<std::int64_t>({5}));

    tags.remove("a");
    PANTR_CHECK(held->ids == std::vector<std::int64_t>({1, 2}));
    PANTR_CHECK(!tags.contains("a"));
}

/// `set` rejects an out-of-range id, a duplicate, and a length mismatch.
void test_cell_set_validates() {
    CellTags tags(5);
    const std::vector<std::int64_t> oob{5};
    const std::vector<std::int64_t> negative{-1};
    const std::vector<std::int64_t> dup{1, 1};
    const std::vector<std::int64_t> three{0, 1, 2};
    const std::vector<std::int64_t> two{1, 2};
    const std::vector<std::int64_t> one{1};

    PANTR_CHECK(rejects([&] { tags.set("a", as_span(oob), as_span(one)); }));
    PANTR_CHECK(rejects([&] { tags.set("a", as_span(negative), as_span(one)); }));
    PANTR_CHECK(rejects([&] { tags.set("a", as_span(dup), as_span(two)); }));
    PANTR_CHECK(rejects([&] { tags.set("a", as_span(three), as_span(two)); }));
    PANTR_CHECK(tags.size() == 0);
}

/// A missing name is `std::out_of_range`, which the wrapper turns into `KeyError`.
void test_a_missing_name_is_reported() {
    CellTags tags(4);
    PANTR_CHECK(reports_missing([&] { return tags.get("nope"); }));
    PANTR_CHECK(reports_missing([&] { tags.remove("nope"); }));

    FacetTags facets(4, 4);
    PANTR_CHECK(reports_missing([&] { return facets.get("nope"); }));
}

/// `scatter` writes the stored values at the stored ids and touches nothing else.
void test_cell_scatter_leaves_the_rest_alone() {
    CellTags tags(6);
    const std::vector<std::int64_t> ids{0, 4};
    const std::vector<std::int64_t> values{1, 2};
    tags.set("location", as_span(ids), as_span(values));

    std::vector<std::int64_t> out(6, -1);
    tags.scatter<std::int64_t>("location", std::span<std::int64_t>(out));
    PANTR_CHECK(out == std::vector<std::int64_t>({1, -1, -1, -1, 2, -1}));

    std::vector<std::int64_t> wrong_size(5, 0);
    PANTR_CHECK(rejects(
        [&] { tags.scatter<std::int64_t>("location", std::span<std::int64_t>(wrong_size)); }));
}

/// The overflow check fires below eight bytes and, as the oracle does, not at eight.
///
/// The second half pins the contract gap `detail::require_representable` documents:
/// a negative value scattered into a `uint64` destination wraps. Asserted rather
/// than left implicit, so that closing the gap has to change a test that says why.
void test_scatter_overflow_follows_the_oracle_width_rule() {
    CellTags tags(4);
    const std::vector<std::int64_t> ids{0, 1};
    const std::vector<std::int64_t> big{200, 1};
    tags.set("big", as_span(ids), as_span(big));

    std::vector<std::int8_t> narrow(4, 0);
    PANTR_CHECK(
        overflows([&] { tags.scatter<std::int8_t>("big", std::span<std::int8_t>(narrow)); }));

    std::vector<std::int16_t> wide(4, 0);
    tags.scatter<std::int16_t>("big", std::span<std::int16_t>(wide));
    PANTR_CHECK(wide == std::vector<std::int16_t>({200, 1, 0, 0}));

    const std::vector<std::int64_t> one{0};
    const std::vector<std::int64_t> minus_one{-1};
    tags.set("neg", as_span(one), as_span(minus_one));
    std::vector<std::uint64_t> unsigned_out(4, 0);
    tags.scatter<std::uint64_t>("neg", std::span<std::uint64_t>(unsigned_out));
    PANTR_CHECK_MSG(unsigned_out[0] == std::numeric_limits<std::uint64_t>::max(),
                    "the oracle skips its range check at eight bytes, so -1 wraps here too");
}

/// A facet tag sorts lexicographically by `(cell_id, local_facet_id)`.
void test_facet_set_sorts_lexicographically() {
    FacetTags tags(10, 4);
    const std::vector<std::int64_t> keys{3, 1, 0, 0, 3, 0};
    const std::vector<std::int64_t> values{7, 5, 6};
    tags.set("bc", as_keys(keys), as_span(values));

    const auto tag = tags.get("bc");
    PANTR_CHECK(tag->keys == std::vector<std::int64_t>({0, 0, 3, 0, 3, 1}));
    PANTR_CHECK(tag->values == std::vector<std::int64_t>({5, 6, 7}));
}

/// The facet registry rejects a bad width, an out-of-range component, a duplicate.
void test_facet_set_validates() {
    FacetTags tags(4, 4);
    const std::vector<std::int64_t> bad_cid{4, 0};
    const std::vector<std::int64_t> bad_lfid{0, 4};
    const std::vector<std::int64_t> dup{0, 0, 0, 0};
    const std::vector<std::int64_t> one{1};
    const std::vector<std::int64_t> two{1, 2};
    const std::vector<std::int64_t> three_wide{0, 0, 0};

    PANTR_CHECK(rejects([&] { tags.set("a", as_keys(bad_cid), as_span(one)); }));
    PANTR_CHECK(rejects([&] { tags.set("a", as_keys(bad_lfid), as_span(one)); }));
    PANTR_CHECK(rejects([&] { tags.set("a", as_keys(dup), as_span(two)); }));
    PANTR_CHECK(rejects([&] {
        tags.set("a", pantr::span2d<const std::int64_t>(three_wide.data(), 1, 3), as_span(one));
    }));
    PANTR_CHECK(tags.size() == 0);
}

/// `scatter` lands each facet value at its own `(cell, facet)` slot.
void test_facet_scatter_lands_at_the_key() {
    FacetTags tags(3, 4);
    const std::vector<std::int64_t> keys{0, 0, 2, 3};
    const std::vector<std::int64_t> values{1, 2};
    tags.set("bc", as_keys(keys), as_span(values));

    std::vector<std::int64_t> storage(3 * 4, -1);
    tags.scatter<std::int64_t>("bc", pantr::span2d<std::int64_t>(storage.data(), 3, 4));
    PANTR_CHECK(storage[0] == 1);
    PANTR_CHECK(storage[2 * 4 + 3] == 2);
    PANTR_CHECK(storage[1] == -1);
}

/// Both registries print their counts and their tag names, with Python's quoting.
void test_to_string_names_the_tags() {
    CellTags cells(6);
    const std::vector<std::int64_t> one{1};
    cells.set("a", as_span(one), as_span(one));
    PANTR_CHECK_MSG(cells.to_string() == "CellTags(num_cells=6, tags=['a'])", cells.to_string());

    FacetTags facets(3, 4);
    PANTR_CHECK_MSG(facets.to_string() == "FacetTags(num_cells=3, facets_per_cell=4, tags=[])",
                    facets.to_string());
}

}  // namespace

int main() {
    test_construction_validates();
    test_cell_set_sorts_by_id();
    test_an_empty_tag_is_legal();
    test_replacing_a_tag_keeps_its_position();
    test_a_handle_outlives_a_replacement();
    test_cell_set_validates();
    test_a_missing_name_is_reported();
    test_cell_scatter_leaves_the_rest_alone();
    test_scatter_overflow_follows_the_oracle_width_rule();
    test_facet_set_sorts_lexicographically();
    test_facet_set_validates();
    test_facet_scatter_lands_at_the_key();
    test_to_string_names_the_tags();
    return pantr::test::summary("test_grid_tags");
}
