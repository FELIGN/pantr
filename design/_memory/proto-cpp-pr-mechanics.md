---
name: proto-cpp-pr-mechanics
description: Two things that bite on every PR into proto/cpp, because it is not the default branch
metadata:
  type: project
---

The port's PRs target **`proto/cpp`**, not `main`. Two consequences, both observed rather than
predicted, and both silent.

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

See [[active-task]].
