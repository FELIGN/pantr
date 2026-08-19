---
name: naming-and-licensing
description: Two sibling libraries are private and unpublished; never name them outside this machine, and never let an MIT library depend on them
metadata:
  type: project
---

**Never name them.** Two sibling libraries (a certified polynomial root finder, and an
unfitted-quadrature consumer of pantr) are **private and unpublished**. Keep their identity
out of anything that leaves the machine: pantr's code, docs, commit messages, published
text. Refer to "a downstream consumer", "a sibling project", "the root-finding library".
Their practices may be copied and their mechanisms reimplemented from the facts; their
**verbatim code and comments may not**, and the root finder is a third party's copyright.

**Licence plan** (not yet applied; both still carry MIT files today): pantr and QUGaR are
**MIT**; the two private libraries will be **hybrid**, free for academic use and paid for
commercial. tIGArx stays **LGPL-3.0** and is not being relicensed.

**The constraint is the direction of dependency.** MIT depending on hybrid is forbidden,
because it would force the MIT library to be effectively hybrid. Hybrid depending on MIT is
fine.

**The seam pattern is what keeps the direction safe, and it is used three times:** an MIT
library defines the interface and always ships a self-sufficient implementation of its own;
the higher-capability provider is installed separately and selected by the user. pantr does
this for the root solver and for graph partitioning; QUGaR does it for quadrature, where the
shipped provider is Algoim (BSD 3-Clause). This is not only a licence workaround: it is what
keeps each library usable with no dependencies.

Moving code from a private library into pantr is a **relicensing act**, not a refactor:
one-way, since MIT cannot be pulled back. Record each move deliberately.
