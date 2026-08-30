/// \file
/// Nothing may hand out the BVH cache slot itself.
///
/// The ticket asked for "a const grid invalidating its cache". This ticket ships no
/// `invalidate_caches()` -- FELIGN/pantr#378 makes refinement return a new grid, so
/// there is nothing to invalidate -- and the property that survives is the one the
/// mechanism existed to protect: the slot is private, `cell_bvh()` is the only way to
/// it, and it yields a reference through which nothing can be replaced.

#include "_fixture.hpp"

void take_the_cache() {
    const GoodGrid g(4);
    pantr::grid::BVH<double>& tree = g.cell_bvh();
    static_cast<void>(tree);
}
