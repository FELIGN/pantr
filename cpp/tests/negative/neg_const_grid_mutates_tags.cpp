/// \file
/// A const grid may read its tag registries and may not write them.
///
/// This is the const polarity the ticket's mechanism 3 was about. Holding the sizes in
/// the mixin rather than in `Derived` is what lets `cell_tags()` be an ordinary pair of
/// overloads, so the guarantee costs no `const_cast` and is the compiler's rather than
/// a convention's.

#include "_fixture.hpp"

void write_through_a_const_grid() {
    const GoodGrid g(4);
    const std::vector<std::int64_t> ids = {0};
    const std::vector<std::int64_t> values = {1};
    g.cell_tags().set("a", ids, values);
}
