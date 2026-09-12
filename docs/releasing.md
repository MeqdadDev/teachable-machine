## Versioning

This project follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **MAJOR** — a breaking change: a public method/parameter is removed or its
  behavior changes incompatibly, the minimum supported Python version is
  raised, or a dependency's required version range changes in a way that
  drops support for environments that previously worked.
- **MINOR** — a backwards-compatible addition: a new method, parameter, or
  supported model/classifier type.
- **PATCH** — a backwards-compatible bug fix: a crash, incorrect result, or
  compatibility break (e.g. with a newer TensorFlow/Keras/Pillow release) is
  fixed without changing the public API.

When a change could plausibly be read as more than one of these (e.g. a bug
fix that also happens to add a new optional parameter), pick the highest
level that applies.

### Where the version lives

The version is defined in exactly one place: `__version__` in
[`src/__init__.py`](https://github.com/MeqdadDev/teachable-machine/blob/main/src/__init__.py).
`setup.py` reads it from there at build time — don't hardcode a version
in `setup.py` or anywhere else, and don't bump it in more than one file.
(This package previously had the version duplicated between `setup.py`
and `src/__init__.py`, and the two drifted out of sync — `setup.py` said
`1.3.0` while `src/__init__.py` still said `1.2`, a version that was never
actually shipped as `v1.3.0` in a release. Keeping a single source of
truth is what this rule exists to prevent.)

## Changelog

Every user-facing change (fix, feature, or breaking change) is recorded in
[`CHANGELOG.md`](https://github.com/MeqdadDev/teachable-machine/blob/main/CHANGELOG.md)
at the repo root, mirrored verbatim to `docs/changelog.md` (the two files
must stay identical — that's what the docs site's Changelog page renders).

- Add the entry **in the same PR** as the change, under the version header
  for the *next* release (create that header, e.g. `## [1.4.0]`, if the PR
  is the first change since the last release).
- Group entries under `### Added`, `### Changed`, `### Fixed`, or
  `### Removed`, following [Keep a Changelog](https://keepachangelog.com/).
- Link the relevant GitHub issue/PR where useful.
- Once that version is actually released (see below), add the release date
  next to its header, e.g. `## [1.3.0] - 2026-09-12`.

## Release process

1. **Confirm the changelog is accurate.** The unreleased version's section
   at the top of `CHANGELOG.md` (and `docs/changelog.md`) should already
   describe everything merged since the last release, since each PR added
   its own entry — read through it and fix anything missing before
   continuing.
2. **Confirm the version bump matches semver** (see above) given everything
   in that section. Adjust the header if the accumulated changes turned out
   to need a higher bump than originally expected (e.g. a breaking change
   landed alongside what were otherwise just fixes).
3. **Bump `__version__`** in `src/__init__.py` to that version.
4. **Add the release date** to the changelog header, e.g.
   `## [1.4.0] - 2026-09-12`.
5. Commit both (e.g. `Release v1.4.0`) directly to `main` (or via a small
   PR, same as any other change) once everything above is in place.
6. **Tag and push:**
   ```bash
   git tag v1.4.0
   git push origin v1.4.0
   ```
7. **Create a GitHub Release from that tag** (Releases -> Draft a new
   release -> pick the tag). Use the changelog section for that version as
   the release notes. Publishing the release triggers
   [`publish.yml`](https://github.com/MeqdadDev/teachable-machine/blob/main/.github/workflows/publish.yml),
   which builds the package and publishes it to PyPI via
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no
   token needed) — this only works once a trusted publisher is registered
   for this repo/workflow on the [PyPI project's publishing settings](https://pypi.org/manage/project/teachable_machine/settings/publishing/).
8. **Verify:**
   - The [PyPI project page](https://pypi.org/project/teachable-machine/)
     shows the new version.
   - `pip install --upgrade teachable-machine` picks it up.
   - The [docs site](https://meqdaddev.github.io/teachable-machine/) — which
     [redeploys automatically](https://github.com/MeqdadDev/teachable-machine/blob/main/.github/workflows/docs.yml)
     on every push to `main` that touches `docs/`, `src/`, or `mkdocs.yml` —
     reflects the release (Changelog page, and any docstring changes via
     `mkdocstrings` on the Explanation page).
