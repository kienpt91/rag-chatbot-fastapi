#!/usr/bin/env python3
"""Dependency age policy: every file pinned in requirements.txt and requirements-dev.txt must be at least seven days
old on PyPI.

    python scripts/check-dep-age.py [CUTOFF_ISO_DATE] [--index URL] [--lock FILE --input FILE]

Without a cutoff it is seven days before now. Without --lock and --input it checks both locks of the project
(requirements.txt against requirements.in, requirements-dev.txt against requirements-dev.in, requirements-build.txt alone). For each ``name==version`` of a lock and each
``--hash=sha256:...`` it asks the index's JSON API for the files of that exact release, finds the file with that
hash and compares its upload time with the cutoff. It also fails when

* a lock and its input (``requirements.in``) do not pin exactly the same packages and versions,
* a pin has no hash,
* a hash is not a file of that release, or the file was yanked.

Exit status: 0 all old enough, 1 a file is too new, unknown or yanked (or lock and input disagree), 2 usage or network
error (also: the index has no such release, so nothing could be checked). Standard library only; needs the network
(it is the one script that does). ``--index`` exists for the tests: it must be https, or http on loopback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = "https://pypi.org"
# `name==version`, optionally `name[extra,other]==version` (pip-compile writes the extras into a lock)
PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?==(?P<version>[A-Za-z0-9.!+_-]+)\s*\\?$"
)
HASH = re.compile(r"--hash=sha256:(?P<digest>[0-9a-f]{64})")


def parse(text: str) -> dict[tuple[str, str], list[str]]:
    pins: dict[tuple[str, str], list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pin = PIN.match(stripped)
        if pin is not None:
            current = pins.setdefault((pin["name"], pin["version"]), [])
            continue
        digest = HASH.search(stripped)
        if digest is not None and current is not None:
            current.append(digest["digest"])
            continue
        raise ValueError(f"cannot read line: {stripped!r}")
    return pins


def declared_pins(text: str) -> set[tuple[str, str]]:
    """The (normalised name, version) pairs of an input file, extras ignored; blank lines and comments skipped."""
    pins: set[tuple[str, str]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pin = PIN.match(stripped)
        if pin is None:
            raise ValueError(f"cannot read line: {stripped!r}")
        pins.add((normalise(pin["name"]), pin["version"]))
    return pins


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def index_is_acceptable(index: str) -> bool:
    parts = urlsplit(index)
    host = parts.hostname or ""
    return parts.scheme == "https" or (parts.scheme == "http" and host in ("127.0.0.1", "localhost", "::1"))


def check_lock(lock: Path, input_file: Path | None, index: str, cutoff: datetime) -> tuple[int, int, list[str]]:
    """Returns (status, files checked, problems). Status 0 fine, 1 a problem, 2 the index could not be asked."""
    pins = parse(lock.read_text())
    if not pins:
        print(f"{lock.name}: the lock pins nothing", file=sys.stderr)
        return 2, 0, []
    declared = declared_pins(input_file.read_text()) if input_file is not None else None
    if declared is not None and declared != {(normalise(name), version) for name, version in pins}:
        print(
            f"{input_file.name if input_file else 'the input'} and {lock.name} disagree: run scripts/lock-deps.py",
            file=sys.stderr,
        )
        return 1, 0, []
    too_new: list[str] = []
    files_checked = 0
    for (name, version), digests in sorted(pins.items()):
        if not digests:
            too_new.append(f"{name}=={version}: no hash pinned")
            continue
        try:
            with urllib.request.urlopen(f"{index.rstrip('/')}/pypi/{name}/{version}/json", timeout=30) as response:
                release = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            print(f"{name}=={version}: lookup failed: {error}", file=sys.stderr)
            return 2, files_checked, too_new
        by_hash = {f["digests"]["sha256"]: f for f in release["urls"]}
        for digest in digests:
            found = by_hash.get(digest)
            if found is None:
                too_new.append(f"{name}=={version}: hash {digest[:12]} is not a file of this release on the index")
                continue
            files_checked += 1
            if found.get("yanked"):
                too_new.append(f"{found['filename']}: yanked")
            uploaded = datetime.fromisoformat(found["upload_time_iso_8601"].replace("Z", "+00:00"))
            if uploaded > cutoff:
                too_new.append(f"{found['filename']}: uploaded {uploaded.isoformat()}")
    print(
        f"{lock.name}: checked {files_checked} files of {len(pins)} pinned packages against cutoff {cutoff.isoformat()}"
    )
    return (1 if too_new else 0), files_checked, too_new


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="check-dep-age.py", add_help=True)
    parser.add_argument("cutoff", nargs="?")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--lock")
    parser.add_argument("--input")
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as error:
        return 2 if error.code else 0
    if (args.lock is None) != (args.input is None):
        print("--lock and --input go together", file=sys.stderr)
        return 2
    try:
        cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else datetime.now(UTC) - timedelta(days=7)
    except ValueError:
        print("invalid cutoff date", file=sys.stderr)
        return 2
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    if not index_is_acceptable(args.index):
        print("the index must be an https URL (or http on loopback)", file=sys.stderr)
        return 2
    pairs = (
        [(Path(args.lock), Path(args.input))]
        if args.lock
        else [
            (ROOT / "requirements.txt", ROOT / "requirements.in"),
            (ROOT / "requirements-dev.txt", ROOT / "requirements-dev.in"),
        ]
    )
    problems: list[str] = []
    worst = 0
    for lock, input_file in pairs:
        status, _, too_new = check_lock(lock, input_file, args.index, cutoff)
        if status == 2:
            return 2
        worst = max(worst, status)
        problems.extend(too_new)
    if problems:
        print("Not acceptable:")
        for line in problems:
            print(f"  {line}")
        return 1
    if worst:
        return worst
    print("No file is newer than the cutoff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
