from __future__ import annotations

import asyncio

from rag_chatbot.application.retrieval import Retriever, search_query
from rag_chatbot.application.settings import ChatSettings
from rag_chatbot.domain.model import ChatTurn
from support.service import build_service


def test_a_short_follow_up_is_searched_together_with_the_previous_question() -> None:
    history = [ChatTurn("user", "What is the delivery fee?"), ChatTurn("assistant", "Five dollars.")]
    assert search_query("And on Sundays?", history) == "What is the delivery fee? And on Sundays?"


def test_a_long_question_stands_on_its_own() -> None:
    history = [ChatTurn("user", "What is the delivery fee?")]
    question = "Could you please tell me whether the bakery opens early on Saturday mornings?"
    assert search_query(question, history) == question


def test_a_short_question_without_history_stands_on_its_own() -> None:
    assert search_query("Opening hours?", []) == "Opening hours?"
    assert search_query("Opening hours?", [ChatTurn("assistant", "Hello")]) == "Opening hours?"


def test_the_most_recent_visitor_question_is_the_one_joined() -> None:
    history = [
        ChatTurn("user", "first"),
        ChatTurn("assistant", "a"),
        ChatTurn("user", "second"),
        ChatTurn("assistant", "b"),
    ]
    assert search_query("and?", history) == "second and?"


def test_retrieval_fuses_both_searches_and_reports_the_best_similarity_as_confidence() -> None:
    service = build_service()
    asyncio.run(service.load())
    retriever = Retriever(service.chunks, service.embedder, ChatSettings(canary="CANARY-0123456789ab"))
    outcome = asyncio.run(retriever.retrieve("acme", "What is the delivery fee?"))
    assert outcome.retrieval.chunks[0].chunk.document_id == "delivery"
    assert outcome.retrieval.confidence == max(c.similarity for c in outcome.retrieval.chunks) > 0.3
    assert outcome.embedding_usage.input_tokens > 0 and outcome.embedding_model == "fake-embedding-1"
    best = outcome.retrieval.chunks[0]
    assert best.vector_rank == 1 and best.keyword_rank == 1  # found by meaning and by words


def test_a_question_of_only_stopwords_still_searches_by_meaning() -> None:
    service = build_service()
    asyncio.run(service.load())
    retriever = Retriever(service.chunks, service.embedder, ChatSettings(canary="CANARY-0123456789ab"))
    outcome = asyncio.run(retriever.retrieve("acme", "what is the"))
    assert all(c.keyword_rank is None for c in outcome.retrieval.chunks)


def test_the_context_budget_limits_the_chunks_handed_on() -> None:
    service = build_service()
    asyncio.run(service.load())
    settings = ChatSettings(canary="CANARY-0123456789ab", context_chunks=1)
    outcome = asyncio.run(
        Retriever(service.chunks, service.embedder, settings).retrieve("acme", "bakery opens delivery menu")
    )
    assert len(outcome.retrieval.chunks) == 1
