#pragma once

/// \file
/// Turn a fired precondition into a clean exit, so ctest can judge it by its message.
///
/// `PANTR_PRECONDITION` is `assert`, which prints to stderr and raises `SIGABRT`. ctest
/// reports a signalled process as "Subprocess aborted" and fails it outright:
/// `PASS_REGULAR_EXPRESSION` does not override a signal, only an exit code. So a test
/// that expects an assertion has to survive it.
///
/// Catching `SIGABRT` and exiting zero leaves the pass criterion as **exit cleanly AND
/// print this message**, which is exactly the pair that separates a fired precondition
/// from the defect it replaced. Against the unfixed headers these same programs also
/// end abnormally, because AddressSanitizer reports the overflow, but its output does
/// not contain the precondition's text, so the match fails and the test fails with it.
/// A weaker criterion -- `WILL_FAIL`, or any check of the exit status alone -- passes
/// against both and would therefore pin nothing.
///
/// The handler runs after the C library has already written the message, which is what
/// makes this safe: nothing is suppressed, only the signal is.

#include <csignal>
#include <cstdlib>

namespace pantr::test {

/// Install a `SIGABRT` handler that exits successfully.
inline void expect_a_precondition_to_fire() {
    std::signal(SIGABRT, [](int) { std::_Exit(EXIT_SUCCESS); });
}

}  // namespace pantr::test
