#!/usr/bin/env python3
"""Run the test suite under coverage and enforce the thresholds, per layer.

    python scripts/coverage_check.py [pytest arguments...]

Line coverage (statements executed) of the package, measured over the whole suite, including the threads of the
runner and the real child processes the adapters start (the user programs themselves are not measured):

    domain >= 95%    application >= 90%    overall >= 85%

Prints the real numbers, writes them to out/coverage-summary.txt and exits 1 below a threshold.
Needs the database container (scripts/verify.sh sets it up); the suite refuses to skip the database tests then.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = {"domain": 95.0, "application": 90.0, "overall": 85.0}
REPORT = ROOT / "out" / "coverage.json"


def layer_of(path: str) -> str:
    parts = Path(path).parts
    try:
        index = parts.index("rag_chatbot")
    except ValueError:
        return "other"
    nxt = parts[index + 1] if len(parts) > index + 1 else ""
    return nxt if nxt in ("domain", "application", "infrastructure", "composition") else "entry"


def main(argv: list[str]) -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pytest", "-q", "--cov=rag_chatbot", "--cov-branch",
        f"--cov-report=json:{REPORT}", "--cov-report=", *argv,
    ]  # fmt: skip
    status = subprocess.run(command, cwd=ROOT, check=False).returncode
    if status != 0:
        print("tests failed: no coverage verdict", file=sys.stderr)
        return status
    files = json.loads(REPORT.read_text())["files"]
    covered: dict[str, int] = {}
    total: dict[str, int] = {}
    branch_covered: dict[str, int] = {}
    branch_total: dict[str, int] = {}
    for path, data in files.items():
        layer = layer_of(path)
        summary = data["summary"]
        covered[layer] = covered.get(layer, 0) + summary["covered_lines"]
        total[layer] = total.get(layer, 0) + summary["num_statements"]
        branch_covered[layer] = branch_covered.get(layer, 0) + summary.get("covered_branches", 0)
        branch_total[layer] = branch_total.get(layer, 0) + summary.get("num_branches", 0)
    lines = ["layer            lines   covered/statements   branches"]
    failed: list[str] = []
    for layer in ("domain", "application", "infrastructure", "composition", "entry"):
        if layer not in total:
            continue
        pct = 100.0 * covered[layer] / total[layer] if total[layer] else 100.0
        bpct = 100.0 * branch_covered[layer] / branch_total[layer] if branch_total[layer] else 100.0
        lines.append(f"{layer:<15} {pct:6.1f}%   {covered[layer]:>6}/{total[layer]:<6}        {bpct:5.1f}%")
        if layer in THRESHOLDS and pct < THRESHOLDS[layer]:
            failed.append(f"{layer} {pct:.1f}% < {THRESHOLDS[layer]:.0f}%")
    all_covered, all_total = sum(covered.values()), sum(total.values())
    overall = 100.0 * all_covered / all_total
    all_bc, all_bt = sum(branch_covered.values()), sum(branch_total.values())
    lines.append(
        f"{'overall':<15} {overall:6.1f}%   {all_covered:>6}/{all_total:<6}        {100.0 * all_bc / all_bt:5.1f}%"
    )
    if overall < THRESHOLDS["overall"]:
        failed.append(f"overall {overall:.1f}% < {THRESHOLDS['overall']:.0f}%")
    text = "\n".join(lines) + "\n"
    (ROOT / "out" / "coverage-summary.txt").write_text(text)
    print(text, end="")
    if failed:
        print("coverage below threshold: " + "; ".join(failed), file=sys.stderr)
        return 1
    print("coverage thresholds met (domain >= 95, application >= 90, overall >= 85)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
