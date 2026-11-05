from __future__ import annotations

import pytest

from rag_chatbot.domain.redaction import MAX_SCAN_CHARS, luhn_valid, redact


@pytest.mark.parametrize(
    "text",
    ["write to jane.doe+news@example.co.uk please", "JANE@EXAMPLE.COM", "a_b-c.d@sub.domain.example.org."],
)
def test_emails_are_redacted(text: str) -> None:
    result = redact(text)
    assert "@" not in result.text and "[email]" in result.text and result.counts["emails"] == 1


@pytest.mark.parametrize(
    "number",
    [
        "+84 912 345 678",
        "(415) 555-2671",
        "415.555.2671",
        "0912345678",
        "+1-202-555-0143",
        "020 7946 0958",
        "+442079460958",
    ],
)
def test_phone_numbers_are_redacted(number: str) -> None:
    result = redact(f"call me on {number} tomorrow")
    assert "[phone]" in result.text and result.text.endswith("tomorrow") and result.counts == {"phones": 1}


@pytest.mark.parametrize(
    "number", ["4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111", "378282246310005"]
)
def test_card_numbers_that_pass_luhn_are_redacted_as_cards(number: str) -> None:
    result = redact(f"my card is {number}, thanks")
    assert result.text == "my card is [card], thanks" and result.counts == {"cards": 1}


def test_a_long_number_that_fails_luhn_is_not_a_card() -> None:
    result = redact("order 4111111111111112 shipped")
    assert "[card]" not in result.text


@pytest.mark.parametrize(
    "text",
    ["2026-11-05", "version 1.2.3", "open 9-5", "price 1,299.00", "12345", "room 101", "in 2026 we", "ref A1B2C3D4E5"],
)
def test_ordinary_numbers_dates_and_prices_are_left_alone(text: str) -> None:
    assert redact(text).text == text


def test_secrets_of_this_service_and_bearer_tokens_are_redacted() -> None:
    key = "ragk_" + "A" * 43
    token = "ragt1." + "B" * 40 + "." + "C" * 43
    text = f"key {key} token {token} header Bearer {'x' * 30}"
    result = redact(text)
    assert result.text == "key [secret] token [secret] header [secret]"
    assert result.counts == {"secrets": 3}


def test_a_jwt_and_a_private_key_header_are_redacted() -> None:
    jwt = "eyJ" + "a" * 20 + ".eyJ" + "b" * 20 + "." + "c" * 20
    assert redact(f"t={jwt}").text == "t=[secret]"
    header = "-" * 5 + "BEGIN RSA PRIVATE KEY" + "-" * 5  # built here so the repository's own secret scan stays quiet
    assert "[secret]" in redact(header).text


def test_kinds_can_be_switched_off() -> None:
    text = "mail a@b.example or call +84 912 345 678"
    assert redact(text, kinds={"emails"}).text == "mail [email] or call +84 912 345 678"
    assert redact(text, kinds=set()).text == text
    assert redact(text, kinds={"phones"}).text == "mail a@b.example or call [phone]"


def test_phones_off_still_redacts_cards() -> None:
    assert redact("4111111111111111", kinds={"cards"}).text == "[card]"
    assert redact("4111111111111111", kinds={"phones"}).text == "4111111111111111"  # too long for a phone number


def test_redaction_is_idempotent() -> None:
    once = redact("a@b.example +84 912 345 678 4111111111111111").text
    assert redact(once).text == once


def test_text_beyond_the_scan_limit_is_dropped_not_left_unscanned() -> None:
    text = "x" * MAX_SCAN_CHARS + " tail@example.com"
    result = redact(text)
    assert "tail@example.com" not in result.text and result.text.endswith("[truncated]")


@pytest.mark.parametrize(
    "hostile",
    ["1" + " 1" * 40000, "a" * 100000, "@" * 50000, "." * 50000, "1-" * 50000, "(" * 50000 + "1", "a@" + "b." * 30000],
)
def test_hostile_inputs_are_processed_without_blowing_up(hostile: str) -> None:
    redact(hostile)  # no assertion: the pytest hang detector is the bound; the patterns have no nested quantifiers


def test_luhn() -> None:
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")
