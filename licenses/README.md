# Third-party licences shipped in the pantr wheel

`pantr` itself is MIT; `LICENSE` at the repository root is its text.

**Since 2026-08-21 the compiled extension statically contains Eigen**, which is
primarily MPL-2.0. Before that date Eigen was fetched only to build the C++ tests
and no Eigen code reached a wheel, which is why this directory did not exist. The
change-of-basis port needs a dense solve (`Eigen::PartialPivLU`) and `bezier` will
need a truncated SVD, so the header is now included by
`cpp/include/pantr/change_basis/change_basis.hpp` and its object code is linked in:
measured at roughly 115 KB, about 46% of the extension's sized symbols.

MPL-2.0 section 3.2 requires the Executable Form to carry the licence and to tell
recipients how to obtain the Source Code Form. `eigen/` holds Eigen's own licence
files, copied verbatim from the pinned revision, and `pyproject.toml` ships this
directory in the wheel.

**Source for the bundled Eigen.** <https://gitlab.com/libeigen/eigen> at commit
`bc3b39870ecb690a623a3f49149a358b95c5781d`, tag 5.0.1, which is the revision
`cmake/PantrDependencies.cmake` pins.

**Open, and not settled here.** Whether `pyproject.toml`'s `license = "MIT"` should
change, or whether shipping these files alongside an MIT declaration is the right
expression of "MIT project, MPL-2.0 dependency statically linked", is a question for
whoever owns licensing. Shipping the licence text is required either way, so it is
done; the declaration is left alone deliberately rather than by oversight.
