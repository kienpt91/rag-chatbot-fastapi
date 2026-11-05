"""The price table and the cost of a request.

The operator supplies the table (prices change; the service never guesses them). Prices are per million tokens in one
currency; money is counted in millionths of that currency as an integer, rounded up, so nothing is ever free and
sums are exact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation

from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.model import Cost, Usage

MICROS = Decimal(1_000_000)
_MAX_PRICE = Decimal(100_000)  # per million tokens: a typo guard, not a market claim


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok: Decimal
    output_per_mtok: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class PriceTable:
    currency: str
    models: Mapping[str, ModelPrice]

    def price_of(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError:
            raise ConfigurationError(f"the price table has no entry for the model {model!r}") from None

    def cost(self, model: str, usage: Usage) -> Cost:
        price = self.price_of(model)
        total = (
            Decimal(usage.input_tokens) * price.input_per_mtok + Decimal(usage.output_tokens) * price.output_per_mtok
        )
        # (tokens x price per million tokens) is already in millionths of the currency
        return Cost(micros=int(total.to_integral_value(rounding=ROUND_CEILING)), estimated=usage.estimated)


def _decimal(value: object, where: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigurationError(f"{where} must be a number")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ConfigurationError(f"{where} must be a number") from None
    if not number.is_finite() or number < 0 or number > _MAX_PRICE:
        raise ConfigurationError(f"{where} must be between 0 and {_MAX_PRICE}")
    return number


def parse_price_table(data: object) -> PriceTable:
    """`{"currency": "USD", "models": {"name": {"input_per_mtok": 0.15, "output_per_mtok": 0.6}}}`, strictly."""
    if not isinstance(data, dict) or set(data) - {"currency", "models"}:
        raise ConfigurationError("price table: expected an object with 'currency' and 'models'")
    currency = data.get("currency")
    if not isinstance(currency, str) or not (2 <= len(currency) <= 8) or not currency.isalpha():
        raise ConfigurationError("price table: 'currency' must be a short currency code")
    models_raw = data.get("models")
    if not isinstance(models_raw, dict) or not models_raw or len(models_raw) > 200:
        raise ConfigurationError("price table: 'models' must be a non-empty object")
    models: dict[str, ModelPrice] = {}
    for name, entry in models_raw.items():
        if not isinstance(name, str) or not name or len(name) > 200:
            raise ConfigurationError("price table: model names must be non-empty strings")
        if (
            not isinstance(entry, dict)
            or set(entry) - {"input_per_mtok", "output_per_mtok"}
            or "input_per_mtok" not in entry
        ):
            raise ConfigurationError(f"price table: {name!r} needs 'input_per_mtok' (and optionally 'output_per_mtok')")
        models[name] = ModelPrice(
            _decimal(entry["input_per_mtok"], f"price table: {name}.input_per_mtok"),
            _decimal(entry.get("output_per_mtok", 0), f"price table: {name}.output_per_mtok"),
        )
    return PriceTable(currency.upper(), models)


def usd_to_micros(amount: str | float | int, where: str = "amount") -> int:
    """A budget in currency units (a string such as "5.00") as millionths."""
    value = _decimal(amount, where)
    return int((value * MICROS).to_integral_value(rounding=ROUND_CEILING))


def format_micros(micros: int, currency: str) -> str:
    return f"{Decimal(micros) / MICROS:.6f} {currency}"
