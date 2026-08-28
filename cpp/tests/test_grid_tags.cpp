/// \file
/// Reserved slot for the cell and facet tag registries tests -- no assertions yet.
///
/// Registered ahead of the port so the ticket that writes these tests does not
/// edit `cpp/tests/CMakeLists.txt`, which thirteen tickets in this milestone
/// would otherwise collide on. It passes because it checks nothing, and says so
/// on stdout: a green ctest line from this file means "not written yet", never
/// "the port is correct".

#include <cstdio>

#include "check.hpp"

int main() {
    std::puts("PLACEHOLDER: test_grid_tags asserts nothing yet");
    return pantr::test::summary("test_grid_tags");
}
