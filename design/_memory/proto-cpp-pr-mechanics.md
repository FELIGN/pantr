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

**`gh pr merge --delete-branch` fails when the base is checked out elsewhere.** It reports
`fatal: 'proto/cpp' is already checked out at ...` and looks like the merge failed. It did not:
only the local branch cleanup did. Check `gh pr view <n> --json state` before reacting, then
fast-forward the main checkout and delete the branch by hand.

**Verify the rebase changed no content**, since these merge by rebase: compare
`git rev-parse HEAD^{tree}` before and after. Identical tree hashes are what license saying the
merged code is the code that passed the suite.

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

**A documentation-only commit may be pushed directly to `proto/cpp`, without the full check
suite.** Ruled by Pablo on 2026-08-21 for `c37d374`, two markdown files under
`design/_memory/`. **The exception is bounded and the bound is the point**: nothing under
`src/`, nothing under `docs/`, no config, no test. The moment a commit touches any of those,
`CLAUDE.md`'s non-negotiable rule applies in full. Ask rather than widen it: the rule exists
because "this cannot break anything" is what one thinks right before breaking something.
