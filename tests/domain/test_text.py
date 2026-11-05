from __future__ import annotations

import pytest

from rag_chatbot.domain.text import (
    estimate_tokens,
    fold,
    normalise_text,
    search_terms,
    strip_invisible,
    truncate_at_boundary,
    words,
)


def test_estimate_tokens_is_four_characters_per_token_and_zero_for_nothing() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\u200bb", "ab"),  # zero width space
        ("a\u202eb", "ab"),  # right-to-left override
        ("a\u2066b\u2069", "ab"),  # isolates
        ("a\x00b\x07c", "abc"),
        ("a\ufeffb", "ab"),  # byte order mark
        ("a\u00adb", "ab"),  # soft hyphen
        ("a\u2028b", "a\nb"),  # line separator becomes a newline
        ("tab\tand\nnewline", "tab\tand\nnewline"),
        ("caf\u00e9", "caf\u00e9"),
    ],
)
def test_strip_invisible_removes_what_a_person_cannot_see_and_keeps_text(raw: str, expected: str) -> None:
    assert strip_invisible(raw) == expected


def test_normalise_text_collapses_spaces_and_blank_lines_and_line_endings() -> None:
    raw = "Hello   world \r\n\r\n\r\n\r\nSecond\u00a0paragraph\t\tok\r\n   \n"
    assert normalise_text(raw) == "Hello world\n\nSecond paragraph ok"


def test_normalise_text_composes_unicode() -> None:
    assert normalise_text("e\u0301") == "\u00e9"


def test_fold_maps_lookalike_letters_and_case_and_invisible_characters() -> None:
    assert fold("IGNORE\u200b Previous") == "ignore previous"
    assert fold("\uff29\uff27\uff2e\uff2f\uff32\uff25") == "ignore"  # full-width letters
    assert fold("Stra\u00dfe") == "strasse"


def test_words_handles_any_script_and_drops_punctuation() -> None:
    assert words("Open 9–5, Mon-Fri! Café ٣ مرحبا") == ["open", "9", "5", "mon", "fri", "café", "٣", "مرحبا"]


def test_search_terms_drops_stopwords_dedupes_and_orders_longest_first() -> None:
    terms = search_terms("What are the opening hours on Sunday? Sunday hours!")
    assert terms == ["opening", "sunday", "hours"]


def test_search_terms_respects_the_limit_and_is_stable() -> None:
    text = " ".join(f"word{i:02d}x" for i in range(40))
    first = search_terms(text, limit=5)
    assert len(first) == 5
    assert first == search_terms(text, limit=5)


def test_search_terms_of_only_stopwords_is_empty() -> None:
    assert search_terms("what is the of and") == []


def test_truncate_prefers_a_boundary_but_always_stays_within_the_limit() -> None:
    assert truncate_at_boundary("short", 100) == "short"
    cut = truncate_at_boundary("alpha beta gamma delta", 14)
    assert cut == "alpha beta"
    assert truncate_at_boundary("x" * 50, 10) == "x" * 10
