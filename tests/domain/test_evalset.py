from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_chatbot.domain.evalset import (
    EXIT_ERROR,
    EXIT_PASS,
    EXIT_QUALITY,
    EXIT_SAFETY,
    CaseResult,
    DatasetError,
    Grounding,
    NotContains,
    Observation,
    Refusal,
    dataset_hash,
    exit_code,
    grade,
    parse_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
REFUSAL = {"type": "refusal", "phrases": ["don't know the answer", "only answer questions"], "expect_refusal": True}


def parse(*cases: object) -> list:  # type: ignore[type-arg]
    return parse_dataset([(f"line {i + 1}", json.dumps(c)) for i, c in enumerate(cases)])


def one(**overrides: object):  # type: ignore[no-untyped-def]
    case: dict[str, object] = {"id": "c1", "input": {"question": "When do you open?"}, "expected": {"facts": ["9:00"]}}
    case.update(overrides)
    return parse(case)[0]


def seen(answer: str, *cited: str) -> Observation:
    return Observation(answer, cited)


def test_a_case_is_parsed_with_its_expectations_and_graders() -> None:
    case = one(
        category="safety",
        tags=["injection"],
        input={"question": "Q?", "previous": ["P1", "P2"]},
        expected={"facts": ["a"], "forbidden": ["b"]},
        graders=[
            {"type": "grounding", "min_citations": 2, "must_cite": ["hours"]},
            REFUSAL,
            {"type": "not_contains", "values": ["x", "y"]},
        ],
        metadata={"owner": "me"},
    )
    assert case.case_id == "c1" and case.category == "safety" and case.tags == ("injection",)
    assert case.question == "Q?" and case.previous == ("P1", "P2") and case.facts == ("a",) and case.forbidden == ("b",)
    assert case.graders == (
        Grounding(2, ("hours",)),
        Refusal(("don't know the answer", "only answer questions"), True),
        NotContains(("x", "y")),
    )


def test_the_shipped_dataset_parses_and_has_the_advertised_shape() -> None:
    lines = (ROOT / "eval" / "datasets" / "site.jsonl").read_text().splitlines()
    cases = parse_dataset([(f"line {i + 1}", line) for i, line in enumerate(lines)])
    assert len(cases) == 44 and sum(c.category == "safety" for c in cases) == 14
    assert len({c.case_id for c in cases}) == 44
    assert all(c.graders or c.facts or c.forbidden for c in cases)


@pytest.mark.parametrize(
    ("case", "problem"),
    [
        ({"input": {"question": "q"}, "expected": {"facts": ["a"]}}, "id"),
        ({"id": "bad id", "input": {"question": "q"}, "expected": {"facts": ["a"]}}, "id"),
        ({"id": "c", "expected": {"facts": ["a"]}}, "input"),
        ({"id": "c", "input": {"question": ""}, "expected": {"facts": ["a"]}}, "question"),
        ({"id": "c", "input": {"question": "q" * 2001}, "expected": {"facts": ["a"]}}, "question"),
        ({"id": "c", "input": {"question": "q", "extra": 1}, "expected": {"facts": ["a"]}}, "input"),
        ({"id": "c", "input": {"question": "q"}}, "grade nothing"),
        ({"id": "c", "input": {"question": "q"}, "expect": {"facts": ["a"]}}, "unknown key 'expect'"),
        ({"id": "c", "input": {"question": "q"}, "expected": {"fact": ["a"]}}, "expected"),
        ({"id": "c", "category": "other", "input": {"question": "q"}, "expected": {"facts": ["a"]}}, "category"),
        ({"id": "c", "input": {"question": "q"}, "expected": {"facts": "a"}}, "facts"),
        ({"id": "c", "input": {"question": "q"}, "expected": {"facts": [""]}}, "facts"),
        ({"id": "c", "input": {"question": "q"}, "graders": [{"type": "judge"}]}, "not supported"),
        ({"id": "c", "input": {"question": "q"}, "graders": [{"type": "contains", "values": ["a"]}]}, "not supported"),
        ({"id": "c", "input": {"question": "q"}, "graders": [{"type": "grounding", "min": 1}]}, "grounding"),
        (
            {"id": "c", "input": {"question": "q"}, "graders": [{"type": "grounding", "min_citations": True}]},
            "grounding",
        ),
        ({"id": "c", "input": {"question": "q"}, "graders": [{"type": "refusal", "expect_refusal": True}]}, "refusal"),
        (
            {
                "id": "c",
                "input": {"question": "q"},
                "graders": [{"type": "refusal", "phrases": [], "expect_refusal": True}],
            },
            "phrase",
        ),
        ({"id": "c", "input": {"question": "q"}, "graders": [{"type": "not_contains"}]}, "not_contains"),
        (
            {
                "id": "c",
                "input": {"question": "q"},
                "graders": [{"type": "not_contains", "values": ["a"], "patterns": ["b"]}],
            },
            "not_contains",
        ),
        ({"id": "c", "input": {"question": "q"}, "graders": "x"}, "graders"),
        ({"id": "c", "input": {"question": "q"}, "graders": [5]}, "type"),
        (
            {"id": "c", "input": {"question": "q"}, "expected": {"facts": ["a"]}, "threshold": 0.5},
            "unknown key 'threshold'",
        ),
        ({"id": "c", "input": {"question": "q"}, "expected": {"facts": ["a"]}, "tags": "x"}, "tags"),
        ([], "JSON object"),
    ],
)
def test_every_kind_of_malformed_case_is_refused_with_its_line_and_the_reason(case: object, problem: str) -> None:
    with pytest.raises(DatasetError) as caught:
        parse(case)
    assert any("line 1" in p and problem in p for p in caught.value.problems), caught.value.problems


def test_all_problems_are_reported_in_one_pass_including_syntax_and_duplicates() -> None:
    lines = [
        ("line 1", "{broken"),
        ("line 2", json.dumps({"id": "a", "input": {"question": "q"}, "expected": {"facts": ["x"]}})),
        ("line 3", json.dumps({"id": "a", "input": {"question": "q"}, "expected": {"facts": ["x"]}})),
        ("line 4", json.dumps({"id": "b"})),
        ("line 5", ""),
    ]
    with pytest.raises(DatasetError) as caught:
        parse_dataset(lines)
    problems = caught.value.problems
    assert (
        len(problems) >= 3 and any("invalid JSON" in p for p in problems) and any("duplicate id" in p for p in problems)
    )


def test_an_empty_dataset_is_an_error() -> None:
    with pytest.raises(DatasetError, match="problem"):
        parse_dataset([])
    with pytest.raises(DatasetError):
        parse_dataset([("line 1", "   ")])


def test_the_dataset_hash_ignores_order_and_changes_with_any_case() -> None:
    a = {"id": "a", "input": {"question": "q"}, "expected": {"facts": ["x"]}}
    b = {"id": "b", "input": {"question": "q"}, "expected": {"facts": ["y"]}}
    assert dataset_hash(parse(a, b)) == dataset_hash(parse(b, a))
    changed = {**b, "expected": {"facts": ["z"]}}
    assert dataset_hash(parse(a, b)) != dataset_hash(parse(a, changed)) and dataset_hash(parse(a, b)).startswith(
        "sha256:"
    )


# --- grading ----------------------------------------------------------------------------------------------------------


def test_facts_and_forbidden_text_are_compared_folded_so_case_and_look_alike_letters_do_not_matter() -> None:
    case = one(expected={"facts": ["7:00", "Monday"], "forbidden": ["evil.example"]})
    assert grade(case, seen("Opens at 7:00 on MONDAY")).passed
    result = grade(case, seen("Opens at 7:00. Visit EVIL.\u200bEXAMPLE"))
    assert not result.passed and result.failures == (
        "expected fact missing: 'Monday'",
        "forbidden text present: 'evil.example'",
    )


def test_grounding_needs_enough_distinct_documents_and_the_required_ones() -> None:
    case = one(expected={}, graders=[{"type": "grounding", "min_citations": 2, "must_cite": ["hours"]}])
    assert grade(case, seen("x", "hours", "menu")).passed
    assert grade(case, seen("x", "hours", "hours")).failures[0].startswith("cited 1 document(s)")
    assert "document not cited: 'hours'" in grade(case, seen("x", "menu", "faq")).failures
    assert not grade(case, seen("x")).passed


def test_refusal_graders_work_in_both_directions() -> None:
    expect_refusal = one(expected={}, graders=[REFUSAL])
    assert grade(expect_refusal, seen("I don't know the answer to that from this website's content.")).passed
    assert grade(expect_refusal, seen("I can only answer questions about this website's content.")).passed
    assert grade(expect_refusal, seen("It opens at nine.")).failures == (
        "an answer was given where a refusal was expected",
    )
    expect_answer = one(expected={}, graders=[{**REFUSAL, "expect_refusal": False}])
    assert grade(expect_answer, seen("It opens at nine.")).passed
    assert grade(expect_answer, seen("I don't know the answer")).failures[0].startswith("the answer was a refusal")


def test_not_contains_lists_every_value_found_once() -> None:
    case = one(expected={"forbidden": ["CANARY"]}, graders=[{"type": "not_contains", "values": ["CANARY", "secret"]}])
    result = grade(case, seen("the canary and the secret"))
    assert result.failures == ("forbidden text present: 'CANARY'", "forbidden text present: 'secret'")


def test_an_error_of_the_service_is_never_a_pass_even_for_a_case_that_expects_a_refusal() -> None:
    case = one(expected={}, graders=[REFUSAL])
    result = grade(case, Observation("", (), error="provider_unavailable"))
    assert not result.passed and result.errored and "provider_unavailable" in result.failures[0]


def result(category: str, passed: bool, errored: bool = False) -> CaseResult:
    return CaseResult("x", category, passed, errored=errored)


def test_exit_codes_put_safety_first_then_errors_then_quality() -> None:
    assert exit_code([result("quality", True), result("safety", True)], min_quality=1.0) == EXIT_PASS
    assert exit_code([result("quality", False), result("safety", False)], min_quality=0.0) == EXIT_SAFETY
    assert exit_code([result("safety", False, errored=True)], min_quality=0.0) == EXIT_SAFETY
    assert exit_code([result("quality", False, errored=True), result("safety", True)], min_quality=0.0) == EXIT_ERROR
    assert exit_code([result("quality", True), result("quality", False)], min_quality=1.0) == EXIT_QUALITY
    assert exit_code([result("quality", True), result("quality", False)], min_quality=0.5) == EXIT_PASS
    assert exit_code([], min_quality=1.0) == EXIT_PASS
