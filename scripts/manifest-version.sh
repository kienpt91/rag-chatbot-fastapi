#!/bin/sh
# Prints the version of the package in the current directory, read from pyproject.toml ([project] table).
# No tools beyond awk, so it runs the same in the CI runner and in a bare container.
# usage: manifest-version.sh
set -eu
if [ ! -f pyproject.toml ]; then
  echo "no pyproject.toml in $(pwd)" >&2
  exit 1
fi
# the first `version = "..."` line inside the [project] table; dependency tables and tool tables do not count
version=$(awk '/^\[/ { in_project = ($0 == "[project]") } in_project && /^version[ \t]*=/ { gsub(/^[^"]*"|".*$/, ""); print; exit }' pyproject.toml)
if [ -z "$version" ]; then
  echo "no version found in the [project] table of pyproject.toml" >&2
  exit 1
fi
printf '%s\n' "$version"
