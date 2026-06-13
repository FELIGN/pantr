---
name: new-feature
description: End-to-end workflow for implementing a new feature, bug fix, or refactor — from creating a worktree and branch through implementation, testing, validation, PR creation, and user notification. Use this skill whenever the user asks you to implement, add, build, create, or fix something that will result in a PR. Trigger on requests like "add support for X", "implement Y", "fix bug Z", "refactor W", or any task that involves writing code that should be merged. Even if the user doesn't mention PRs or branches, if the task involves code changes that need review, use this skill.
---

# New Feature Workflow

This skill orchestrates the full lifecycle of a code change — from branch creation to PR. Follow these phases in order.

## Phase 1: Worktree and branch

Create an isolated worktree to work in. Use the `EnterWorktree` tool — it will create a new worktree under `.claude/worktrees/` with a fresh branch based on HEAD.

After entering the worktree, rename the branch to follow the convention:

```bash
git branch -m <type>/<short-description>
```

Where `<type>` matches the conventional commit type (`feat`, `fix`, `refactor`, etc.) and `<short-description>` is a kebab-case summary (e.g., `feat/periodic-bspline-product`, `fix/bernstein-derivative-at-boundary`).

## Phase 2: Plan

Before writing code, understand the task and form a plan:

1. Read the relevant existing code to understand the architecture and conventions
2. Identify which layer(s) the change touches (Layer 1 / 2 / 3 — see CLAUDE.md)
3. Outline the approach — what files to create or modify, what the public API looks like
4. Share the plan with the user and get confirmation before proceeding

For non-trivial features, use `EnterPlanMode` to align on the approach.

## Phase 3: Implement

Write the code following the project's architecture and conventions:

- **Layer 3 (kernels)**: Pure Numba computation, no validation, `@nb_jit(nopython=True, cache=True)`
- **Layer 2 (helpers)**: Input validation, array allocation, calls Layer 3
- **Layer 1 (public API)**: Lightweight validation, delegates to Layer 2
- Follow strict mypy typing, Google-style docstrings, and all conventions in CLAUDE.md

### Commit conventions

Commit regularly as you complete logical units of work. Use **conventional commits**:

```
<type>(<scope>): <imperative summary>

Optional body explaining why, not what.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`
**Scope**: the module or area affected (e.g., `bspline`, `basis`, `quad`, `docs`)

Examples:
- `feat(bspline): add exact 1D B-spline product via Bezier multiplication`
- `fix(basis): correct off-by-one in Cox-de Boor recursion`
- `refactor(bspline): move to_open_bspline to Layer 2`
- `test(bspline): add boundary case tests for derivative evaluation`

Keep commits focused — one logical change per commit. Do not bundle unrelated changes.

## Phase 4: Test

Write tests for the new code:

- Place tests in the appropriate file under `tests/` (follow existing naming: `test_<module>.py`)
- Cover normal cases, edge cases, and error cases
- Test public API (Layer 1) primarily; test Layer 2 directly only if it has complex validation logic
- Use `pytest --no-cov -v` to run tests during development
- Commit tests in a dedicated commit (e.g., `test(bspline): add tests for exact product`)

## Phase 5: Pre-PR validation

Before creating the PR, run the full check suite. Invoke the `pre-pr-checks` skill (or run the checks manually):

1. `ruff check .` — lint
2. `ruff format --check .` — formatting
3. `mypy --config-file mypy.ini src tests` — type checking
4. `pytest --no-cov -v` — tests
5. `NUMBA_DISABLE_JIT=1 sphinx-build -M html docs/ docs/_build -W --keep-going -j auto` — docs

Fix any issues and commit the fixes before proceeding. All checks must pass.

**Always run every check on the whole repo, not single files.** Running
`ruff check <one_file>` while skipping `ruff format --check .` (or vice versa) has
let real failures through. Run all five commands, every push — no shortcuts.

**Local green ≠ CI green.** The local environment can differ from CI in ways that
hide failures; the local suite passing is necessary but not sufficient. Known traps:

- **mypy runs on a Python matrix (3.10–3.14).** CI's 3.10 job uses *older* numpy
  stubs than a typical local env, so code that type-checks locally can fail on 3.10.
  `mypy.ini` pins `python_version = 3.10`, but the *numpy stub version* is whatever is
  installed locally. Be defensive with numpy typing: no bare `np.ndarray`; cast
  `tensordot`/`einsum`/`linspace` results to `float64` where the dtype matters; and
  prefer a covariant `Sequence[...]` parameter over an invariant `list[...]` when the
  element dtype may be inferred differently across stub versions.
- **Headless GL.** The CI test job runs without a display. pyvista/VTK tests must not
  call `.show()` or `pantr.viz.plot()` (they force a render and segfault headless) —
  build the scene with `to_plotter()` instead. The full suite runs under coverage with
  `NUMBA_DISABLE_JIT=1`; a segfault there crashes the whole run.
- If a dependency the suite needs (e.g. `pyvista`, `mpi4py`) is missing locally, the
  relevant tests silently skip — install it so they actually run before you rely on a
  green local result.

## Phase 5.5: Automated review (fresh Sonnet reviewers)

After checks pass and before pushing, run a fresh-context review of the branch diff.

**Carve-out — skip this phase entirely** if the diff is trivial: docs-only,
formatting-only, or comment-only changes (`git diff main...HEAD` touches no
executable logic). For these, proceed straight to Phase 6.

Otherwise, launch the pr-review-toolkit agents **in parallel** via the Task tool,
each with **`model: sonnet`** (the orchestrator stays on Opus; reviewers get clean,
uncontaminated context):

- `pr-review-toolkit:code-reviewer` — always
- `pr-review-toolkit:pr-test-analyzer` — if test files changed
- `pr-review-toolkit:comment-analyzer` — if comments/docstrings changed
- `pr-review-toolkit:silent-failure-hunter` — if error handling changed
- `pr-review-toolkit:type-design-analyzer` — if new types added

Tell each agent to review the branch diff against `main` (`git diff main...HEAD`).

Aggregate findings into **Critical / Important / Suggestion** buckets, then:

1. Fix every **Critical** and **Important** finding. Commit the fixes
   (`fix(<scope>): address review findings`).
2. Re-run Phase 5 checks — they must pass.
3. Re-run the review on the new diff. Repeat until no Critical/Important findings
   remain, or two consecutive rounds surface nothing new (**hard cap: 3 rounds**).
4. List any unaddressed **Suggestions** in the PR body under a "Review notes"
   heading — do not silently drop them.

This phase is **unattended** — do not pause for approval. Proceed to Phase 6 when clean.

> Note: findings come from fresh Sonnet reviewers, but fixes are applied by this
> (Opus) orchestrator, which carries implementation context. The re-review in
> step 3 is what guards against the implementer rationalizing away a real finding.

## Phase 6: PR and notify

Once all checks pass:

1. Push the branch:
   ```bash
   git push -u origin HEAD
   ```

2. Create the PR using `gh pr create`:
   - Title: concise, under 70 characters
   - Body: summary of changes, test plan, and the generated-by footer
   - Target the `main` branch

   ```bash
   gh pr create --title "<type>(<scope>): <summary>" --body "$(cat <<'EOF'
   ## Summary
   <bullet points describing the changes>

   ## Test plan
   - [ ] All existing tests pass
   - [ ] New tests added for <feature>
   - [ ] Pre-PR checks pass (ruff, mypy, pytest, docs)

   Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

3. **Verify CI is green before declaring done.** Pushing is not the finish line —
   local checks can pass while CI fails (see the local-vs-CI traps in Phase 5). After
   creating the PR, watch the checks to completion:

   ```bash
   gh pr checks <pr-number> --watch    # or poll `gh pr checks <pr-number>`
   ```

   If any job fails, read its log (`gh run view --job <id> --log`), fix the cause,
   commit, and push again — then re-watch. Do not report the PR as ready until every
   required check passes. Avoid pushing twice in quick succession: a second push can
   cancel the first run's in-progress jobs (showing a spurious "failure"); wait for the
   run to settle, or confirm a job was *canceled* (not a real failure) before reacting.

4. **Notify the user**: Tell them the PR is ready *and CI is green*, provide the URL,
   and give a brief summary of what was done.

## Important notes

- If the user asks you to **plan only** (not implement), stop after Phase 2 and present the plan.
- If at any point you're unsure about a design decision, ask the user rather than guessing.
- Do not push or create a PR without running all checks first (every command, whole repo).
- Do not push or create a PR without running Phase 5.5 review first (unless the diff hit the trivial carve-out).
- The work is not done until **CI is green**, not merely when local checks pass — always confirm with `gh pr checks` before declaring the PR ready.
- Keep the user informed at natural milestones (plan ready, implementation done, tests passing, PR created, CI green).
