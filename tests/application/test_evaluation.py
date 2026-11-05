from __future__ import annotations

import asyncio
import json

from rag_chatbot.application.evaluation import EvalReport, EvalRunner
from rag_chatbot.domain.evalset import EvalCase, parse_dataset
from support.fakes import ScriptedChat
from support.service import PAGES, TENANT, Service, build_service


def cases(*items: dict[str, object]) -> list[EvalCase]:
    return parse_dataset([(f"line {i + 1}", json.dumps(c)) for i, c in enumerate(items)])


GROUNDED = {"type": "grounding", "min_citations": 1, "must_cite": ["delivery"]}
REFUSAL = {"type": "refusal", "phrases": ["don't know the answer"], "expect_refusal": True}


def run(service: Service, items: list[dict[str, object]]) -> EvalReport:
    asyncio.run(service.load(PAGES))
    return asyncio.run(EvalRunner(service.chat_service, TENANT, service.ids).run(cases(*items)))


def test_cases_are_run_through_the_whole_flow_graded_and_costed() -> None:
    service = build_service()
    report = run(
        service,
        [
            {
                "id": "fee",
                "input": {"question": "What is the delivery fee?"},
                "expected": {"facts": ["5 dollars"]},
                "graders": [GROUNDED],
            },
            {"id": "france", "input": {"question": "What is the capital of France?"}, "graders": [REFUSAL]},
        ],
    )
    assert [(r.case_id, r.passed) for r in report.results] == [("fee", True), ("france", True)]
    assert report.answers["fee"] == "The delivery fee is 5 dollars." and report.answers["france"].startswith(
        "I don't know"
    )
    assert report.cost_micros > 0 and report.input_tokens > 0 and len(report.latencies_ms) == 2
    assert report.dataset_hash.startswith("sha256:")


def test_each_case_has_its_own_conversation_and_earlier_questions_are_asked_first_in_it() -> None:
    service = build_service()
    report = run(
        service,
        [
            {
                "id": "follow",
                "input": {"question": "And how many kilometres?", "previous": ["What is the delivery fee?"]},
                "expected": {"facts": ["5 kilometres"]},
            },
        ],
    )
    assert [r.passed for r in report.results] == [True]
    assert service.ids.sessions == 1  # the earlier question and this one share one conversation


def test_a_service_error_is_recorded_as_an_error_not_as_a_pass() -> None:
    service = build_service(chat=ScriptedChat(fail=True))
    report = run(service, [{"id": "boom", "input": {"question": "What is the delivery fee?"}, "graders": [REFUSAL]}])
    [result] = report.results
    assert not result.passed and result.errored and report.answers["boom"] == "[provider_unavailable]"


def test_a_failing_case_lists_why() -> None:
    service = build_service(chat=ScriptedChat(reply="Everything is free today [s1]"))
    report = run(
        service,
        [{"id": "wrong", "input": {"question": "What is the delivery fee?"}, "expected": {"facts": ["5 dollars"]}}],
    )
    assert report.results[0].failures == ("expected fact missing: '5 dollars'",)


def test_percentiles_of_nothing_are_zero_and_of_some_are_ordered() -> None:
    report = EvalReport("sha256:x")
    assert report.percentile(0.5) == 0
    report.latencies_ms = [5, 1, 9, 3]
    assert report.percentile(0.0) == 1 and report.percentile(0.5) == 5 and report.percentile(0.95) == 9
