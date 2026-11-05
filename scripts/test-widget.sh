#!/bin/sh
# Runs the widget tests in a throwaway, offline, read-only container.
# Image: Node 22.22.3 (created 2026-06-11), pinned by digest and older than seven days.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)

docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --memory 1g --cpus 2 -v "$here:/work:ro" -w /work \
  node@sha256:e21fc383b50d5347dc7a9f1cae45b8f4e2f0d39f7ade28e4eef7d2934522b752 node --test --test-reporter=spec "tests/widget/*.test.mjs"
