# Third-party licences in pantr's binary distributions

`pantr` itself is MIT, and `LICENSE` at the repository root is its text. Nothing
below changes that: the MPL-2.0 is copyleft **per file**, and no file of pantr's is
covered by it.

## What carries Eigen, and what does not

**The sdist does not contain Eigen.** `cmake/PantrDependencies.cmake` fetches it at
build time, so a source distribution is MIT alone and this directory is
informational there.

**A built wheel does.** Eigen is header-only, so its object code is compiled into
the extension: measured at roughly 115 KB, about 46% of the installed `.so`'s sized
symbols. That began on 2026-08-21, when the change-of-basis port took Eigen as a
dependency of the shipped extension for its dense solve (`Eigen::PartialPivLU`);
before then Eigen was fetched only to build the C++ tests and no Eigen code reached
a wheel, which is why this directory did not exist.

## Why MIT and MPL-2.0 combine here

Read against the text in `eigen/COPYING.MPL2` rather than from reputation:

- **§3.3** permits distributing a Larger Work "under terms of Your choice", provided
  the licence's requirements are met for the Covered Software. The wheel is that
  Larger Work.
- **§1.7** defines a Larger Work as Covered Software combined with other material
  "in a separate file or files, that is not Covered Software". pantr's sources are
  separate files.
- **§1.10** defines Modifications as a file altering Covered Software, or a new file
  *containing* Covered Software. pantr contains no Eigen source; two `#include`
  lines in `cpp/include/pantr/change_basis/change_basis.hpp` reference it. So no
  pantr file is a Modification and none falls under the MPL.
- **§3.2(b)** allows the Executable Form to be sublicensed under different terms, so
  long as they do not limit the recipient's rights in the Source Code Form. MIT does
  not.

## What is therefore required, and is done

**§3.2(a)**: distributing the Executable Form obliges us to keep the Source Code
Form available and to tell recipients how to obtain it. `eigen/` holds Eigen's own
licence files, copied verbatim from the pinned revision, and `pyproject.toml`'s
`license-files` ships this directory.

**Source for the bundled Eigen.** <https://gitlab.com/libeigen/eigen> at commit
`bc3b39870ecb690a623a3f49149a358b95c5781d`, tag 5.0.1, the revision
`cmake/PantrDependencies.cmake` pins. A public repository at a pinned commit is the
standard reading of §3.2(a)'s "reasonable means in a timely manner"; it is a reading
rather than a certainty.

**If Eigen is ever patched in place** rather than consumed as an upstream pin, the
patched files stay MPL-2.0 and must ship as source. Nothing does that today.

## Left alone deliberately

`pyproject.toml` declares `license = "MIT"`, which is accurate for pantr's own code
and is what the MPL requires nothing of. An argument exists for `MIT AND MPL-2.0` on
a wheel that carries both compiled in, and an argument against, since it would
suggest to downstream tooling that pantr propagates copyleft when its file-level
scope means it does not. That is a metadata-precision question rather than a
compliance one, and it is not a port's to settle.

None of this is legal advice.
