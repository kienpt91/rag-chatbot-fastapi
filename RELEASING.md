# Releasing

A release is a git tag `vMAJOR.MINOR.PATCH` pushed to GitHub. The `release` workflow (`.github/workflows/release.yml`)
does the rest. **Nothing is published to a package registry.**

## Cut a release

1. On `main` with a green CI, set the version in `pyproject.toml` and `src/rag_chatbot/_version.py` (a test keeps them
   equal), add the section to `CHANGELOG.md`, commit:

   ```sh
   git commit -am "Release 0.2.0"
   ```

2. Tag that commit and push the tag:

   ```sh
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin main v0.2.0
   ```

The workflow runs only for a pushed tag that starts with `v`. It stops before building anything unless

- the tag is exactly `v` plus the version in `pyproject.toml` (`scripts/check-release-tag.sh`), and
- the tagged commit is in the history of the default branch.

Then the `verify` job repeats what CI does (every pinned development file at least seven days old, format check,
lint, strict type check, the whole suite with the coverage thresholds), and the `release` job builds the sdist and
the wheel **in the container pinned by digest** (`scripts/build-dist.sh`: `python -m build --no-isolation`, with the
hash-checked `setuptools`, `SOURCE_DATE_EPOCH` set to the commit time), writes `SHA256SUMS`, creates a build-provenance
attestation for the files, and creates the GitHub release with the built-in `GITHUB_TOKEN` (no other secret is
involved). The notes are the subjects of the commits since the previous tag (`scripts/release-notes.sh`). A tag with
a `-` in it (`v0.2.0-rc.1`) is a pre-release.

To try the build without releasing, run the `release` workflow by hand (Actions, release, Run workflow): it builds and
keeps the files as a workflow artifact for seven days, never a release or an attestation. Locally:
`sh scripts/py.sh install && sh scripts/build-dist.sh` puts the files in `out/dist`.

## What a release contains

- `rag_chatbot-VERSION-py3-none-any.whl` and `rag_chatbot-VERSION.tar.gz`
- `SHA256SUMS`

## Verify and install a download

```sh
sha256sum -c SHA256SUMS            # macOS: shasum -a 256 -c SHA256SUMS
gh attestation verify rag_chatbot-0.1.0-py3-none-any.whl --repo OWNER/rag-chatbot-fastapi
pip install ./rag_chatbot-0.1.0-py3-none-any.whl     # no dependencies are installed
```

The checksum shows the file is the one listed; the attestation shows it was built by this repository's workflow
from the tagged commit. Do both.

## Updating the pinned tools

- **A development package:** edit its pin in `requirements-dev.in` (or `requirements.in`), then
  `PY_NET=bridge sh scripts/py.sh 'python scripts/lock-deps.py 2026-10-29 requirements-dev.in' > requirements-dev.txt` (the argument is the
  cutoff: today minus seven days), review the diff, `sh scripts/py.sh install`, `sh scripts/verify.sh`. `scripts/lock-deps.py`
  refuses a file uploaded after the cutoff; CI runs `scripts/check-dep-age.py` and fails while the `.in` and `.txt` disagree.
- **The build backend:** `requirements-build.txt` has no input file. Change the pin in `pyproject.toml` (`[build-system]`) and regenerate: `printf 'setuptools==NEW\n' > out/build.in; PY_NET=bridge sh scripts/py.sh 'python scripts/lock-deps.py CUTOFF out/build.in' > requirements-build.txt` (a test keeps the two equal).
- **The runtime packages:** the same with `requirements.in` and `requirements.txt`; also update the exact pins in `pyproject.toml` (a test keeps them equal to the lock).
- **The Python image:** pick a tag of the official `python` image whose index was pushed at least seven days ago, take
  the digest of the multi-arch index, update the digests and dates in `scripts/py.sh`, `scripts/eval-real.sh` and the `Dockerfile`, run the
  suite on both versions (`PY_VERSION=3.12`, `3.13`).
- **A GitHub Action:** use the full commit SHA of a release at least seven days old, with the version in a comment.

## Publishing to a registry later

A deliberate manual step that is not wired up: it needs a long-lived secret or a trusted-publisher setup, and a
decision about the name on the registry. If you add it, make it a job that `needs: release`, runs in a GitHub
environment that requires your approval, has only `contents: read` and `id-token: write`, and publishes the files the
release job built (never a rebuild). Pin and review any action you add the same way the existing ones are.

## The database image

The pgvector image is pinned by digest in `scripts/py.sh` and `docker-compose.yml` (0.8.6 on PostgreSQL 16, created 2026-08-13). Update both together, with a tag at least seven days old, and run `sh scripts/verify.sh`.
