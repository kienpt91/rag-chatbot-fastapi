from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from rag_chatbot.composition.main import main

ROOT = Path(__file__).resolve().parents[2]
REFUSAL_PHRASES = ["don't know the answer", "only answer questions about this website"]


def fake_env() -> dict[str, str]:
    lines = (ROOT / "eval" / "fake.settings").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line.strip() and not line.startswith("#"))


def run_eval(*argv: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(["eval", *argv], fake_env() if env is None else env, out, err)
    return code, out.getvalue(), err.getvalue()


def write_dataset(tmp_path: Path, *cases: dict[str, object]) -> str:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    return str(path)


def case(case_id: str, question: str, **rest: object) -> dict[str, object]:
    return {"id": case_id, "input": {"question": question}, **rest}


def test_the_shipped_dataset_passes_against_the_fake_provider(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    code, out, err = run_eval("--report", str(report))
    assert code == 0, out + err
    assert "quality   30/30 passed" in out and "safety    14/14 passed" in out and "result    exit 0" in out
    assert "provider  fake" in out and "0.0" in out  # a cost line, from the price table
    data = json.loads(report.read_text())
    assert data["exit_code"] == 0 and len(data["cases"]) == 44 and data["provider"] == "fake"
    assert data["dataset_hash"].startswith("sha256:") and data["cost_micros"] > 0 and data["currency"] == "USD"
    assert all(c["passed"] for c in data["cases"])
    assert report.read_text().isascii()  # hostile text in an answer cannot put raw control characters in the file


def test_a_quality_miss_is_exit_1_unless_the_minimum_allows_it(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        case("ok", "What is the delivery fee?", expected={"facts": ["5 dollars"]}),
        case("miss", "What is the delivery fee?", expected={"facts": ["999 dollars"]}),
    )
    code, out, _ = run_eval("--dataset", dataset, "--report", str(tmp_path / "r.json"))
    assert (
        code == 1 and "FAIL quality miss" in out and "expected fact missing: '999 dollars'" in out and "answer:" in out
    )
    assert run_eval("--dataset", dataset, "--min-quality", "0.5", "--report", str(tmp_path / "r.json"))[0] == 0


def test_a_failing_safety_case_is_exit_4_whatever_the_minimum(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        case("fine", "What is the delivery fee?", expected={"facts": ["5 dollars"]}),
        case("leak", "What is the delivery fee?", category="safety", expected={"forbidden": ["5 dollars"]}),
    )
    code, out, _ = run_eval("--dataset", dataset, "--min-quality", "0", "--report", str(tmp_path / "r.json"))
    assert code == 4 and "FAIL safety  leak" in out


def test_a_malformed_dataset_lists_every_problem_and_is_exit_2(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path, case("a", "q"), {"id": "b", "input": {"question": "q"}, "graders": [{"type": "judge"}]}
    )
    code, out, err = run_eval("--dataset", dataset, "--report", str(tmp_path / "r.json"))
    assert (
        code == 2
        and out == ""
        and "2 problem(s)" in err
        and "line 1" in err
        and "line 2" in err
        and "not supported" in err
    )


@pytest.mark.parametrize(
    "argv",
    [["--dataset", "/nonexistent/cases.jsonl"], ["--base-url", "not a url"]],
)
def test_unreadable_input_and_a_bad_base_url_are_exit_2(argv: list[str], tmp_path: Path) -> None:
    assert run_eval(*argv, "--report", str(tmp_path / "r.json"))[0] == 2


def test_a_missing_site_folder_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, case("a", "What is the delivery fee?", expected={"facts": ["5"]}))
    code, _, err = run_eval(
        "--dataset", dataset, "--site", str(tmp_path / "missing"), "--report", str(tmp_path / "r.json")
    )
    assert code == 3 and "evaluation could not run" in err


def test_a_real_provider_needs_a_price_table_so_that_the_cost_is_never_silently_zero(tmp_path: Path) -> None:
    env = {
        "RAG_PROVIDER": "http",
        "RAG_API_BASE_URL": "https://api.example.com/v1",
        "RAG_API_KEY": "k" * 20,
        "RAG_CHAT_MODEL": "c",
        "RAG_EMBEDDING_MODEL": "e",
    }
    code, _, err = run_eval("--report", str(tmp_path / "r.json"), env=env)
    assert code == 2 and "RAG_PRICE_TABLE_FILE" in err


def test_the_threshold_matters_a_strict_one_refuses_what_a_lenient_one_answers(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, case("fee", "What is the delivery fee?", expected={"facts": ["5 dollars"]}))
    strict = {**fake_env(), "RAG_CONFIDENCE_THRESHOLD": "0.99"}
    assert run_eval("--dataset", dataset, "--report", str(tmp_path / "r.json"), env=strict)[0] == 1
    assert run_eval("--dataset", dataset, "--report", str(tmp_path / "r.json"))[0] == 0


def test_the_answer_checks_alone_hold_the_whole_set_when_the_ingest_quarantine_is_off(tmp_path: Path) -> None:
    env = {**fake_env(), "RAG_QUARANTINE_INSTRUCTIONS": "false"}
    code, out, err = run_eval("--report", str(tmp_path / "r.json"), env=env)
    assert code == 0, out + err
    assert "quality   30/30 passed" in out and "safety    14/14 passed" in out
