---
name: proto-cpp-pr-mechanics
description: What bites on every PR into proto/cpp, including that its CI is two jobs rather than eight
metadata:
  type: project
---

The port's PRs target **`proto/cpp`**, not `main`. Several consequences, all observed rather
than predicted, and all silent.

**`Closes #N` does nothing.** GitHub resolves closing keywords only for pull requests merged
into the **default branch**, which here is `main`. PR #342 carried `Closes #340`, merged into
`proto/cpp`, and #340 stayed open. **Close the issue by hand after every port PR**, with the
verification record in the closing comment, since the PR body is not where anyone reads it
from the issue. This will recur for #338, #339 and #341.

**A branch cut from `proto/cpp` carries whatever is sitting unpushed there.** PR #342 was
announced as one commit and merged two: the memory-notes commit was already on the local
`proto/cpp` when the fix branch was cut from it. Same shape as the cache-keying fix, which was
planned as its own PR and shipped inside #337 for the same reason. **Before branching, check
that the base is level with `origin`**, not just that the working tree is clean; `git status`
says nothing about it. `git log --oneline origin/<base>..HEAD` does.

**CI runs only on pull requests against `proto/cpp`.** A branch push triggers nothing, so an
experiment that needs a CI run needs a PR with a non-empty diff. And `gh run view --log` shows
only the **latest attempt**: re-running a failed job destroys the log that explained it, so
capture what matters first.

**And it is a REDUCED CI, which is the one that has actually cost something.** `ci.yaml` owns
lint, type-check, the whole suite under Python, the docs build, MPI and the adversarial sweep,
and its `on:` block names **`main` only**, for both `push` and `pull_request`. What runs on a
`proto/cpp` PR is `cpp.yaml`: the GCC 14 build plus one parity job, which runs `tests/parity`
and then the whole suite under `PANTR_BACKEND=cpp`. Two jobs, not eight. So **ruff, mypy, the
Python-backend suite and the docs build are the developer's to run**, and `scripts/ci_local.sh`
is the only thing that does. Confirmed by a docs regression that shipped in #344 and would have
turned `main` red at merge time: a `:meth:` reference to a private method, invisible for
two PRs. Note the parity job's own comment says the Python leg is covered by `ci.yaml`'s tests
job; on this branch that job does not run at all.

See [[active-task]].
