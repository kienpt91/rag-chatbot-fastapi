from __future__ import annotations

from decimal import Decimal

import pytest

from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.model import Usage
from rag_chatbot.domain.pricing import format_micros, parse_price_table, usd_to_micros
from rag_chatbot.domain.ratelimit import RateLimiter

TABLE = {
    "currency": "usd",
    "models": {
        "chat-small": {"input_per_mtok": 0.15, "output_per_mtok": 0.6},
        "embed-small": {"input_per_mtok": "0.02"},
    },
}


def test_cost_is_tokens_times_price_per_million_rounded_up_to_a_micro() -> None:
    table = parse_price_table(TABLE)
    assert table.currency == "USD"
    cost = table.cost("chat-small", Usage(1000, 500))
    assert cost.micros == 150 + 300 and not cost.estimated
    assert table.cost("embed-small", Usage(10)).micros == 1  # 0.2 micros rounds up: nothing is ever free
    assert table.cost("chat-small", Usage()).micros == 0


def test_estimated_usage_marks_the_cost_as_estimated() -> None:
    assert parse_price_table(TABLE).cost("chat-small", Usage(10, 10, estimated=True)).estimated


def test_an_unpriced_model_is_a_configuration_error_not_a_free_call() -> None:
    with pytest.raises(ConfigurationError, match="no entry"):
        parse_price_table(TABLE).cost("mystery", Usage(1, 1))


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"currency": "USD"},
        {"currency": "USD", "models": {}},
        {"currency": "", "models": {"m": {"input_per_mtok": 1}}},
        {"currency": "US1", "models": {"m": {"input_per_mtok": 1}}},
        {"currency": "USD", "models": {"m": {}}},
        {"currency": "USD", "models": {"m": {"input_per_mtok": -1}}},
        {"currency": "USD", "models": {"m": {"input_per_mtok": "NaN"}}},
        {"currency": "USD", "models": {"m": {"input_per_mtok": True}}},
        {"currency": "USD", "models": {"m": {"input_per_mtok": 1e9}}},
        {"currency": "USD", "models": {"m": {"input_per_mtok": 1, "per_image": 2}}},
        {"currency": "USD", "models": {"m": {"input_per_mtok": "abc"}}},
        {"currency": "USD", "models": {"m": {"input_per_mtok": 1}}, "extra": 1},
        {"currency": "USD", "models": {"": {"input_per_mtok": 1}}},
    ],
)
def test_malformed_price_tables_are_refused(data: object) -> None:
    with pytest.raises(ConfigurationError):
        parse_price_table(data)


def test_budgets_convert_exactly() -> None:
    assert usd_to_micros("5.00") == 5_000_000
    assert usd_to_micros(0.000001) == 1
    assert format_micros(1_234_567, "USD") == "1.234567 USD"
    assert Decimal(usd_to_micros("0.1")) == Decimal(100_000)
    with pytest.raises(ConfigurationError, match="daily"):
        usd_to_micros("-1", "daily")


def test_a_bucket_allows_its_capacity_then_refuses_and_says_when_to_retry() -> None:
    limiter = RateLimiter(capacity=3, refill_per_s=1.0)
    assert all(limiter.acquire("k", 100.0).allowed for _ in range(3))
    denied = limiter.acquire("k", 100.0)
    assert not denied.allowed and denied.retry_after_s == pytest.approx(1.0)
    assert not limiter.acquire("k", 100.5).allowed
    assert limiter.acquire("k", 101.0).allowed


def test_refill_never_exceeds_capacity() -> None:
    limiter = RateLimiter(capacity=2, refill_per_s=10.0)
    assert limiter.acquire("k", 0.0).allowed and limiter.acquire("k", 0.0).allowed
    assert limiter.acquire("k", 1000.0).allowed and limiter.acquire("k", 1000.0).allowed
    assert not limiter.acquire("k", 1000.0).allowed


def test_keys_are_independent() -> None:
    limiter = RateLimiter(capacity=1, refill_per_s=0.1)
    assert limiter.acquire("a", 0.0).allowed and not limiter.acquire("a", 0.0).allowed
    assert limiter.acquire("b", 0.0).allowed


def test_memory_is_bounded_and_the_least_recently_used_key_is_dropped() -> None:
    limiter = RateLimiter(capacity=1, refill_per_s=0.001, max_keys=100)
    for i in range(10_000):
        limiter.acquire(f"ip-{i}", float(i))
    assert len(limiter) == 100
    assert not limiter.acquire("ip-9999", 10_000.0).allowed  # a recent key is remembered
    assert limiter.acquire("ip-0", 10_000.0).allowed  # an old one was evicted and starts fresh


def test_a_busy_key_is_not_evicted_by_idle_ones() -> None:
    limiter = RateLimiter(capacity=1, refill_per_s=0.001, max_keys=3)
    limiter.acquire("busy", 0.0)
    for i in range(10):
        limiter.acquire("busy", float(i + 1))
        limiter.acquire(f"idle-{i}", float(i + 1))
    assert not limiter.acquire("busy", 20.0).allowed


def test_a_clock_that_goes_backwards_does_not_mint_tokens() -> None:
    limiter = RateLimiter(capacity=1, refill_per_s=1.0)
    assert limiter.acquire("k", 100.0).allowed
    assert not limiter.acquire("k", 50.0).allowed


@pytest.mark.parametrize("kwargs", [{"capacity": 0.5}, {"refill_per_s": 0}, {"max_keys": 0}])
def test_nonsense_limits_are_refused(kwargs: dict[str, float]) -> None:
    base = {"capacity": 1.0, "refill_per_s": 1.0, "max_keys": 1}
    with pytest.raises(ConfigurationError):
        RateLimiter(**{**base, **kwargs})  # type: ignore[arg-type]
