"""The logic of the small scripts that are not part of the package: lock generation, the age check, the dependency files."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("wheel", "interpreter", "arch", "expected"),
    [
        ("pkg-1.0-py3-none-any.whl", "cp313", "x86_64", True),
        ("pkg-1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "cp313", "x86_64", True),
        ("pkg-1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "cp313", "aarch64", False),
        ("pkg-1.0-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl", "cp312", "aarch64", True),
        ("pkg-1.0-py3-none-macosx_11_0_arm64.whl", "cp313", "aarch64", False),
        ("pkg-1.0.tar.gz", "cp313", "x86_64", False),
    ],
)
def test_wheel_selection_for_the_hash_lock(wheel: str, interpreter: str, arch: str, expected: bool) -> None:
    assert load("lock-deps").covers(wheel, interpreter, arch) is expected


def test_the_age_check_reads_pins_and_hashes() -> None:
    check = load("check-dep-age")
    digest = "a" * 64
    pins = check.parse(
        f"# c\nfoo==1.0 \\\n    --hash=sha256:{digest} \\\n    --hash=sha256:{'b' * 64}\nbar-baz==2.1.0 \\\n    --hash=sha256:{digest}\n"
    )
    assert pins == {("foo", "1.0"): [digest, "b" * 64], ("bar-baz", "2.1.0"): [digest]}
    with pytest.raises(ValueError, match="cannot read line"):
        check.parse("foo>=1.0\n")


@pytest.mark.parametrize(
    ("lock", "source"), [("requirements.txt", "requirements.in"), ("requirements-dev.txt", "requirements-dev.in")]
)
def test_each_lock_pins_exactly_what_its_input_pins_and_every_pin_has_hashes(lock: str, source: str) -> None:
    check = load("check-dep-age")
    pins = check.parse((ROOT / lock).read_text())
    declared = {
        (check.normalise(n), v)
        for n, v in (
            line.split("==", 1)
            for line in (ROOT / source).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    }
    assert declared == {(check.normalise(n), v) for n, v in pins} and all(pins.values())


def test_the_two_locks_agree_on_the_packages_they_share() -> None:
    check = load("check-dep-age")
    runtime = {check.normalise(n): v for n, v in check.parse((ROOT / "requirements.txt").read_text())}
    dev = {check.normalise(n): v for n, v in check.parse((ROOT / "requirements-dev.txt").read_text())}
    assert {n for n in runtime.keys() & dev.keys() if runtime[n] != dev[n]} == set()


def test_pyproject_pins_the_top_level_runtime_packages_exactly_as_the_lock_does() -> None:
    check = load("check-dep-age")
    runtime = {check.normalise(n): v for n, v in check.parse((ROOT / "requirements.txt").read_text())}
    block = re.search(r"^dependencies = \[(.*?)\]", (ROOT / "pyproject.toml").read_text(), re.S | re.M)
    assert block is not None
    declared = re.findall(r'"([a-z_-]+)==([0-9.]+)"', block.group(1))
    assert declared and all(runtime[check.normalise(n)] == v for n, v in declared)
