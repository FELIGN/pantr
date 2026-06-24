# Releasing PaNTr

Maintainer runbook for cutting a release. Publishing is automated by
[`.github/workflows/release.yaml`](.github/workflows/release.yaml): pushing a
`v*` tag builds the distributions and publishes them to TestPyPI and then PyPI
via [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC —
no API tokens are stored anywhere), then cuts a GitHub Release.

The version is single-sourced from `__version__` in `src/pantr/__init__.py`
(hatchling reads it as the package version). **The git tag must be
`v<version>`** — the pipeline rebuilds the wheel and fails the release if the
built version does not match the tag.

## One-time setup

These are dashboard actions, done once for the project. They cannot be committed
to the repository.

### PyPI and TestPyPI trusted publishers

On both [pypi.org](https://pypi.org/manage/account/publishing/) **and**
[test.pypi.org](https://test.pypi.org/manage/account/publishing/) (enable 2FA on
each account first), add a *pending publisher*:

| Field             | Value          |
| ----------------- | -------------- |
| PyPI Project Name | `pantr`        |
| Owner             | `FELIGN`       |
| Repository name   | `pantr`        |
| Workflow name     | `release.yaml` |
| Environment name  | `pypi` on pypi.org, `testpypi` on test.pypi.org |

The first successful publish creates the project; the pending publisher then
becomes a regular trusted publisher.

### GitHub environments

In *Settings → Environments*, create two environments named `pypi` and
`testpypi` (the names must match the pending publishers above). Add yourself as a
*required reviewer* on each so a tag push pauses for a one-click approval before
anything is uploaded.

### Read the Docs

Confirm the GitHub webhook is connected (RTD → *Admin → Integrations*; a recent
delivery should show `200`). RTD reads `.readthedocs.yaml` from the repo, so no
other configuration is needed. Optionally add an *Automation Rule* matching the
`v*` tag pattern to **activate** the new version and **set it as default** — with
that rule in place the per-release RTD steps below become automatic.

## Cutting a release

1. **Land all changes on `main`** with CI green.
2. **Bump the version** in `src/pantr/__init__.py` (`__version__`).
3. **Update the changelog**: add the new `## <version> (<date>)` section to
   `docs/changelog.md` (Keep a Changelog style — `### Added` / `### Changed` /
   …). This exact section becomes the GitHub Release notes.
4. Open a PR with the bump + changelog, run the checks, and merge it.
5. *(Optional) Rehearse.* Trigger the workflow manually from *Actions → Release →
   Run workflow*. This builds and publishes to **TestPyPI only** — production
   PyPI is never touched on a manual run. Verify the install:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ "pantr==<version>"
   ```

6. **Tag and push** from `main`:

   ```bash
   git checkout main && git pull
   git tag v<version>
   git push origin v<version>
   ```

7. **Approve** the `testpypi` and then `pypi` environments when the run pauses
   (*Actions* tab). The pipeline then publishes to PyPI and cuts the GitHub
   Release.
8. **On Read the Docs** (unless an Automation Rule handles it): *Versions* →
   activate `v<version>`; *Admin → Settings* → set it as the default version.

## Verify

- PyPI project page shows the new version, and `pip install "pantr==<version>"`
  works from a clean environment.
- The GitHub Release exists with the changelog section as its notes and the
  sdist + wheel attached.
- `https://pantr.readthedocs.io` serves the new version's docs.

## How the pipeline is wired

`release.yaml` runs four jobs:

- **build** — `python -m build` (sdist + wheel) and `twine check`; on a tag,
  asserts the built version equals the tag.
- **testpypi** — publishes to TestPyPI (`skip-existing`, so re-runs are safe).
  Runs on both a tag push and a manual dispatch.
- **pypi** — publishes to PyPI. Gated to tag pushes only.
- **github-release** — creates the GitHub Release with notes extracted from the
  matching section of `docs/changelog.md`.

Permissions are least-privilege: the publish jobs request only the OIDC
`id-token` they need, and the third-party publish action is pinned to a commit
SHA.
