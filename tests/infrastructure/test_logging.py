from __future__ import annotations

import io
import json
import logging

import pytest

from rag_chatbot.infrastructure.http.logging import JsonFormatter, configure_logging, ip_hash, request_id_var


def render(message: str, formatter: JsonFormatter | None = None, **extra: object) -> dict[str, object]:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, (), None)
    record.__dict__.update(extra)
    parsed = json.loads((formatter or JsonFormatter()).format(record))
    assert isinstance(parsed, dict)
    return parsed


def test_a_record_has_a_timestamp_level_logger_message_and_request_id() -> None:
    token = request_id_var.set("r_abc12345")
    try:
        entry = render("hello")
    finally:
        request_id_var.reset(token)
    assert entry["level"] == "info" and entry["logger"] == "test" and entry["msg"] == "hello"
    assert entry["request_id"] == "r_abc12345" and str(entry["ts"]).endswith("+00:00")


def test_extra_fields_are_included_and_every_string_is_redacted() -> None:
    entry = render(
        "mail jane@example.org", tenant="acme", note="call +1 555 0177 or 4111 1111 1111 1111", n=3, ok=True, none=None
    )
    assert entry["msg"] == "mail [email]" and entry["tenant"] == "acme"
    assert (
        entry["note"] == "call [phone] or [card]" and entry["n"] == 3 and entry["ok"] is True and entry["none"] is None
    )


def test_objects_are_never_logged_only_their_type() -> None:
    entry = render("x", body={"message": "secret"}, items=[1, 2], blob=b"bytes")
    assert entry["body"] == "<dict>" and entry["items"] == "<list>" and entry["blob"] == "<bytes>"


def test_secrets_are_redacted_even_when_the_operator_switched_personal_data_off() -> None:
    formatter = JsonFormatter(redaction_kinds=frozenset())
    entry = render("key ragk_" + "A" * 43 + " and jane@example.org", formatter)
    assert entry["msg"] == "key [secret] and jane@example.org"


def test_long_values_are_cut() -> None:
    assert len(str(render("x", note="a" * 5000)["note"])) <= 500


def test_exceptions_are_logged_by_class_and_redacted_message_without_a_traceback() -> None:
    try:
        raise ValueError("bad value for jane@example.org")
    except ValueError as error:
        record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "failed", (), (type(error), error, error.__traceback__)
        )
    entry = json.loads(JsonFormatter().format(record))
    assert (
        entry["error"] == "ValueError"
        and entry["error_detail"] == "bad value for [email]"
        and "Traceback" not in json.dumps(entry)
    )


def test_output_is_ascii_one_line_json_even_for_hostile_text() -> None:
    line = JsonFormatter().format(
        logging.LogRecord("t", logging.INFO, __file__, 1, "line\nbreak \u202e \u2028 \x00 emoji \U0001f600", (), None)
    )
    assert "\n" not in line and line.isascii() and json.loads(line)["msg"].startswith("line")


def test_configure_logging_replaces_handlers_and_sends_everything_through_the_formatter() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    try:
        logging.getLogger("anything").warning("careful jane@example.org")
        logging.getLogger("anything").info("quiet")
        configure_logging("INFO", stream=stream)  # a second call does not duplicate handlers
        logging.getLogger("anything").info("loud")
        lines = stream.getvalue().splitlines()
        assert len(lines) == 2 and "[email]" in lines[0] and json.loads(lines[1])["msg"] == "loud"
    finally:
        logging.getLogger().handlers.clear()


@pytest.mark.parametrize("address", ["203.0.113.5", "2001:db8::1", "unknown"])
def test_the_address_hash_is_short_stable_salted_and_not_the_address(address: str) -> None:
    first = ip_hash(address, "salt")
    assert first == ip_hash(address, "salt") and first != ip_hash(address, "other") and len(first) == 10
    assert address not in first
