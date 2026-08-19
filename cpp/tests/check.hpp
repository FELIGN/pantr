#pragma once

/// \file
/// A three-macro test harness.
///
/// Deliberately not GoogleTest or Catch2. This tree is a prototype whose first
/// requirement is that it be cheap to abandon, and a test framework is a
/// FetchContent entry, a configure-time cost and a compile-time cost bought for
/// assertions that fit in twenty lines. If the port outlives the prototype, this
/// is the first thing to replace.

#include <cstdio>
#include <cstdlib>
#include <string>

namespace pantr::test {

inline int failures = 0;

inline void report(bool ok, const char* expr, const char* file, int line, const std::string& note) {
    if (ok) {
        return;
    }
    ++failures;
    std::fprintf(stderr, "FAIL %s:%d\n  %s\n", file, line, expr);
    if (!note.empty()) {
        std::fprintf(stderr, "  %s\n", note.c_str());
    }
}

inline int summary(const char* name) {
    if (failures == 0) {
        std::printf("PASS %s\n", name);
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "%d failure(s) in %s\n", failures, name);
    return EXIT_FAILURE;
}

}  // namespace pantr::test

#define PANTR_CHECK(expr) ::pantr::test::report((expr), #expr, __FILE__, __LINE__, {})

#define PANTR_CHECK_MSG(expr, note) ::pantr::test::report((expr), #expr, __FILE__, __LINE__, (note))
