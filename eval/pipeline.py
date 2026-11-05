#!/usr/bin/env python3
"""Wires this service to `llm-eval` (the sibling project llm-eval-harness) through its `command` provider.

The harness starts this program for every case run: one JSON request on stdin, one JSON answer on stdout. The program
builds the same retrieval and chat flow the service runs (in memory, over the sample site in examples/site), asks the
case's question, and reports the answer, the documents it cited and what it cost. Which provider answers is decided by
the RAG_* environment variables, exactly as for the service itself: with `RAG_PROVIDER=fake` (see eval/llm-eval.json)
no hosted model is called.

    cd eval && PYTHONPATH=/path/to/llm-eval-harness/src python3 -m llm_eval_harness run

The built-in runner (`rag-chatbot eval`) grades the same dataset without needing the harness; the harness adds
repeats, statistics, baselines and its CI gate (docs/evaluation.md).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_chatbot.application.chat import ChatRequest  # noqa: E402
from rag_chatbot.composition import wiring  # noqa: E402
from rag_chatbot.composition.commands import FAKE_PRICES  # noqa: E402
from rag_chatbot.composition.config import parse_settings  # noqa: E402
from rag_chatbot.domain.tenants import Tenant  # noqa: E402
from rag_chatbot.infrastructure.sources.folder import FolderSource  # noqa: E402

BASE_URL = "https://www.maple-street-bakery.example"


async def answer(question: str, previous: list[str]) -> dict[str, object]:
    settings = parse_settings({"RAG_PROVIDER": "fake", **os.environ})
    prices = wiring.load_prices(settings) if settings.price_table_file else FAKE_PRICES
    providers = wiring.build_providers(settings)
    stores = wiring.memory_stores(settings)
    tenant = Tenant("eval", "Maple Street Bakery", frozenset({BASE_URL}))
    await wiring.build_ingest_service(settings, providers, stores, prices).run(
        tenant.tenant_id, FolderSource(ROOT / "examples" / "site", base_url=BASE_URL)
    )
    service = wiring.build_chat_service(settings, providers, stores, prices)
    session: str | None = None
    for earlier in previous:
        session = (await service.handle(ChatRequest(tenant, earlier, session, None, "r_pipeline"))).session_id
    reply = await service.handle(ChatRequest(tenant, question, session, None, "r_pipeline"))
    documents = list(dict.fromkeys(c.document_id for c in reply.answer.citations))
    return {
        "text": reply.answer.text,
        "citations": documents,
        "retrieved": documents,
        "tool_calls": [],  # the model has no tools: there is nothing to report
        "usage": {
            "model": settings.chat_model,
            "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens,
        },
    }


def main() -> None:
    request = json.load(sys.stdin)  # {"protocol": 1, "case": {...}, "repeat": 0, "attempt": 1, "seed": null}
    case_input = request["case"]["input"]
    result = asyncio.run(answer(case_input["question"], list(case_input.get("previous", []))))
    json.dump(result, sys.stdout)  # nothing else on stdout; logs go to stderr


if __name__ == "__main__":
    main()
