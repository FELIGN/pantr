# Negative translation units

Each file here is a program that **must not compile**, and each is registered with ctest
by `pantr_add_negative_test` in `cpp/tests/CMakeLists.txt`. The test command builds the
translation unit; the pass criterion is the **diagnostic's own text**, not a non-zero
exit.

That distinction is the whole test, and it is the same one the precondition tests make.
A negative translation unit rejected for the wrong reason -- a typo, a missing include,
a header that moved -- still fails to compile, so a test that only checked the exit code
would pass while asserting nothing. Matching the message is what ties each file to the
mechanism it was written for.

Every one is built by whichever compiler configured the build directory, so
`scripts/ci_local.sh` runs them on g++ 14, clang++ 18 and clang++ 10, the floor toolchain.
Not on g++ 10: FELIGN/pantr#376 raised the floor to libstdc++ 11, which g++ 10 does not meet,
so it is now refused at configure time rather than built.
