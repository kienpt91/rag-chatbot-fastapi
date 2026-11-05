from __future__ import annotations

import pytest

from rag_chatbot.domain.guard import GuardResult, check_answer
from rag_chatbot.domain.prompt import system_prompt

CANARY = "CANARY-0123456789ab"
RULES = system_prompt("Maple Bakery", CANARY)


def check(raw: str) -> GuardResult:
    return check_answer(
        raw, aliases={"s1": "ck_0"}, system_prompt=RULES, canary=CANARY, allowed_hosts=["www.acme.example"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        "See //evil.example/p.png?c=1 now [s1]",
        "![x](//evil.example/p.png?c=1) [s1]",
        "[click here](//evil.example) [s1]",
        "![x](evil.example/p.png) [s1]",
        "![x](https://www.acme.example/p.png) [s1]",  # an image has no business in a plain-text answer
        "[x](evil.example/pay) [s1]",
        "[x](/relative/path) [s1]",
        "[x](https://evil.example/pay) [s1]",
        "write to mailto:boss@evil.example [s1]",
        "call tel:+15550100 [s1]",
        "send sms:+15550100 [s1]",
    ],
)
def test_links_without_a_scheme_markdown_links_images_and_other_schemes_are_refused(answer: str) -> None:
    assert check(answer).reason == "unverified_link"


@pytest.mark.parametrize(
    "answer",
    [
        "Open from 10//12 and 7:00 [s1]",
        "See //www.acme.example/hours [s1]",
        "[hours](https://www.acme.example/hours) [s1]",
        "[hours](//www.acme.example/hours) [s1]",
        "We open at nine (see the hours page) [s1]",
    ],
)
def test_ordinary_text_and_the_sites_own_links_pass(answer: str) -> None:
    assert check(answer).ok


def test_a_citation_marker_followed_by_a_bracket_target_is_not_a_free_pass() -> None:
    assert check("Hours [s1](//evil.example)").reason == "unverified_link"
