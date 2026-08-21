---
name: expired-guard-rationales
description: A guard whose stated reason has expired is not a constraint; check the reason before defending the guard
metadata:
  type: feedback
---

**A comment explaining why a guard exists is dated evidence, not a standing constraint.**
Before treating a fence, an `if`, or a deliberate omission as something to design around,
read its stated reason and ask whether that reason still holds under the current decision.

**Why.** In pantr's C++ port, `cpp/tests/CMakeLists.txt` fenced Eigen off from
`pantr::core` so that "a stray `#include` cannot compile without anyone deciding to take
the dependency on". That reason was sound while the dependency was untaken. Once Eigen
became a deliberate dependency of two Stage 1 modules, the fence protected against nothing,
and I still built a recommendation around preserving it. Pablo saw through it in one
question. Same error I had just flagged in `design/large_data_fitting.md:164`, whose
"zero new dependency (Eigen is already required)" was false for the pip path.

**How to apply.** When a design note or a build guard supplies a *reason*, verify the
reason against the code before quoting the conclusion. `pyproject.toml` settings and CMake
`option()` defaults are the ground truth about what ships; a design note is someone's claim
about it. And when Pablo asks "why not X?" twice about the same choice, the question is
usually load-bearing: stop generating options and re-derive the premise.

See [[reviewing-my-own-measurements]] and [[decisions-pointer]].
