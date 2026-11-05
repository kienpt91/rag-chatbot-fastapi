"""The chat use case: one visitor message in, one checked answer out.

The order of the steps is the design:

1. cheap refusals first (empty, too long, an obvious attempt to rewrite the rules), before any money is spent;
2. the daily budget, fail closed;
3. the session (a visitor can only continue a conversation that belongs to the same tenant);
4. redaction of personal data, so the model and the database never see it;
5. retrieval, and a refusal when it found nothing good enough, *without calling the chat model*;
6. the model, which is given the rules, fenced sources and no tools;
7. the checks on its answer (`domain.guard`), and a refusal when it fails them;
8. what happened is stored and its cost recorded, whatever the outcome.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from rag_chatbot.application.ports import (
    ChatProvider,
    Clock,
    IdGenerator,
    SessionStore,
    UsageEntry,
    UsageLedger,
)
from rag_chatbot.application.retrieval import RetrievalOutcome, Retriever, search_query
from rag_chatbot.application.settings import BLOCKED_MESSAGE, ChatSettings
from rag_chatbot.domain.errors import (
    BudgetExhaustedError,
    ChatbotError,
    InvalidRequestError,
    SessionFullError,
    SessionNotFoundError,
)
from rag_chatbot.domain.guard import GuardResult, check_answer, host_of, scrub
from rag_chatbot.domain.history import trim_history
from rag_chatbot.domain.model import (
    Answer,
    ChatReply,
    ChatTurn,
    Citation,
    Cost,
    RefusalReason,
    Usage,
)
from rag_chatbot.domain.pricing import PriceTable
from rag_chatbot.domain.prompt import build_prompt
from rag_chatbot.domain.redaction import redact
from rag_chatbot.domain.scope import check_message
from rag_chatbot.domain.tenants import Tenant, linkable_hosts
from rag_chatbot.domain.text import estimate_tokens, truncate_at_boundary

log = logging.getLogger("rag_chatbot.chat")
SNIPPET_CHARS = 240


@dataclass(frozen=True, slots=True)
class ChatRequest:
    tenant: Tenant
    message: str
    session_id: str | None
    owner_id: str | None  # the end user's id when the credential carries one (row-level security)
    request_id: str


def _refusal_reason(guard_reason: str | None) -> RefusalReason:
    return "no_answer" if guard_reason == "no_answer" else "unsafe_answer"


class ChatService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        chat: ChatProvider,
        sessions: SessionStore,
        ledger: UsageLedger,
        prices: PriceTable,
        embedding_model: str,
        clock: Clock,
        ids: IdGenerator,
        settings: ChatSettings,
    ) -> None:
        self._retriever = retriever
        self._chat = chat
        self._sessions = sessions
        self._ledger = ledger
        self._prices = prices
        self._embedding_model = embedding_model
        self._clock = clock
        self._ids = ids
        self._settings = settings

    # --- steps -------------------------------------------------------------------------------------------------

    async def _check_budget(self, tenant: Tenant) -> None:
        cap = tenant.daily_budget_micros
        if cap is None:
            cap = self._settings.default_daily_budget_micros
        if cap is None:
            return
        spent = await self._ledger.spent_micros(tenant.tenant_id, self._clock.now().date())
        if spent >= cap:
            log.warning("daily budget exhausted", extra={"tenant": tenant.tenant_id, "event": "budget_exhausted"})
            raise BudgetExhaustedError(f"tenant {tenant.tenant_id}")

    async def _session(self, request: ChatRequest) -> tuple[str, bool]:
        """The session id to use, and whether it was just created."""
        if request.session_id is None:
            session_id = self._ids.new_session_id()
            await self._sessions.create_session(request.tenant.tenant_id, session_id, request.owner_id)
            return session_id, True
        info = await self._sessions.get_session(request.tenant.tenant_id, request.session_id)
        if info is None:
            raise SessionNotFoundError(request.session_id)
        if info.message_count >= self._settings.max_messages_per_session:
            raise SessionFullError(request.session_id)
        return request.session_id, False

    def _citations(
        self, guard: GuardResult, outcome: RetrievalOutcome, aliases: dict[str, str], allowed_hosts: set[str]
    ) -> tuple[Citation, ...]:
        by_id = {c.chunk.chunk_id: c.chunk for c in outcome.retrieval.chunks}
        citations: list[Citation] = []
        for alias in guard.cited_aliases:
            chunk = by_id[aliases[alias]]
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    url=chunk.url,
                    snippet=truncate_at_boundary(scrub(" ".join(chunk.text.split()), allowed_hosts), SNIPPET_CHARS),
                )
            )
        return tuple(citations)

    async def _finish(
        self,
        request: ChatRequest,
        session_id: str,
        stored_question: str,
        answer: Answer,
        *,
        usage: Usage,
        embedding_tokens: int,
        cost: Cost,
        started: float,
        outcome: str,
    ) -> ChatReply:
        tenant_id = request.tenant.tenant_id
        await self._sessions.append_messages(
            tenant_id,
            session_id,
            [ChatTurn("user", stored_question), ChatTurn("assistant", answer.text)],
            grounded=answer.grounded,
            chunk_ids=[c.chunk_id for c in answer.citations],
        )
        latency_ms = max(0, int((self._clock.monotonic() - started) * 1000))
        await self._ledger.record(
            UsageEntry(
                tenant_id=tenant_id,
                session_id=session_id,
                request_id=request.request_id,
                at=self._clock.now(),
                chat_model=self._chat.model,
                embedding_model=self._embedding_model,
                usage=usage,
                embedding_tokens=embedding_tokens,
                cost_micros=cost.micros,
                cost_estimated=cost.estimated,
                latency_ms=latency_ms,
                outcome=outcome,
            )
        )
        return ChatReply(session_id, answer, usage, cost, latency_ms)

    # --- the use case ------------------------------------------------------------------------------------------

    async def handle(self, request: ChatRequest) -> ChatReply:
        started = self._clock.monotonic()
        settings = self._settings
        tenant = request.tenant
        refusal_text = tenant.refusal_message or settings.refusal_message
        threshold = (
            tenant.confidence_threshold if tenant.confidence_threshold is not None else settings.confidence_threshold
        )

        rejected = check_message(request.message, settings.blocked_phrases)
        if rejected in ("empty", "too_long"):
            raise InvalidRequestError(f"message: {rejected}")
        await self._check_budget(tenant)
        session_id, _ = await self._session(request)

        # personal data never reaches the model or the database; the visitor's own wording is otherwise kept
        redacted = (
            redact(request.message, settings.redaction_kinds).text if settings.redaction_kinds else request.message
        )
        if rejected is not None:
            log.info("message refused before retrieval", extra={"tenant": tenant.tenant_id, "reason": rejected})
            answer = Answer(BLOCKED_MESSAGE, grounded=False, refusal_reason="out_of_scope")
            return await self._finish(
                request,
                session_id,
                redacted,
                answer,
                usage=Usage(),
                embedding_tokens=0,
                cost=Cost(),
                started=started,
                outcome="blocked",
            )

        history: Sequence[ChatTurn] = ()
        if request.session_id is not None:
            history = await self._sessions.recent_messages(tenant.tenant_id, session_id, settings.history_turns * 2)
        history = trim_history(
            history, budget_tokens=settings.history_token_budget, max_turns=settings.history_turns * 2
        )

        outcome = await self._retriever.retrieve(tenant.tenant_id, search_query(redacted, history))
        embedding_cost = self._prices.cost(self._embedding_model, outcome.embedding_usage)
        embedding_tokens = outcome.embedding_usage.input_tokens

        if not outcome.retrieval.chunks or outcome.retrieval.confidence < threshold:
            answer = Answer(
                refusal_text,
                grounded=False,
                refusal_reason="low_confidence",
                confidence=outcome.retrieval.confidence,
            )
            return await self._finish(
                request, session_id, redacted, answer, usage=Usage(), embedding_tokens=embedding_tokens,
                cost=embedding_cost, started=started, outcome="refused",
            )  # fmt: skip

        prompt = build_prompt(
            site_name=tenant.name,
            question=redacted,
            history=history,
            retrieval=outcome.retrieval,
            nonce=self._ids.new_nonce(),
            canary=settings.canary,
            max_source_chars=settings.context_chars,
            max_sentences=settings.max_sentences,
        )
        try:
            completion = await self._chat.complete(
                prompt.system, prompt.messages, max_output_tokens=settings.max_output_tokens
            )
        except ChatbotError:
            await self._record_failure(request, session_id, embedding_tokens, embedding_cost, started)
            raise
        usage = completion.usage
        if usage.input_tokens == 0 and usage.output_tokens == 0:
            # the provider reported nothing: estimate, and say so in the accounting
            usage = Usage(
                estimate_tokens(prompt.system) + sum(estimate_tokens(m.content) for m in prompt.messages),
                estimate_tokens(completion.text),
                estimated=True,
            )
        cost = self._prices.cost(self._chat.model, usage)
        total = Cost(cost.micros + embedding_cost.micros, cost.estimated or embedding_cost.estimated)

        aliases = {alias: item.chunk.chunk_id for alias, item in prompt.aliases.items()}
        allowed_hosts = set(linkable_hosts(tenant))
        allowed_hosts |= {h for item in prompt.aliases.values() if (h := host_of(item.chunk.url))}
        guard = check_answer(
            completion.text,
            aliases=aliases,
            system_prompt=prompt.system,
            canary=settings.canary,
            allowed_hosts=allowed_hosts,
        )
        if not guard.ok:
            log.warning(
                "answer refused by the checks",
                extra={"tenant": tenant.tenant_id, "reason": guard.reason, "event": "answer_blocked"},
            )
            reason = _refusal_reason(guard.reason)
            answer = Answer(
                refusal_text, grounded=False, refusal_reason=reason, confidence=outcome.retrieval.confidence
            )
            return await self._finish(
                request, session_id, redacted, answer, usage=usage, embedding_tokens=embedding_tokens, cost=total,
                started=started, outcome="refused" if reason == "no_answer" else "blocked",
            )  # fmt: skip
        citations = self._citations(guard, outcome, aliases, allowed_hosts)
        answer = Answer(guard.text, grounded=True, citations=citations, confidence=outcome.retrieval.confidence)
        return await self._finish(
            request, session_id, redacted, answer, usage=usage, embedding_tokens=embedding_tokens, cost=total,
            started=started, outcome="answered",
        )  # fmt: skip

    async def _record_failure(
        self, request: ChatRequest, session_id: str, embedding_tokens: int, cost: Cost, started: float
    ) -> None:
        """The chat model failed after the question was embedded: the embedding was paid for, so it is recorded."""
        await self._ledger.record(
            UsageEntry(
                tenant_id=request.tenant.tenant_id,
                session_id=session_id,
                request_id=request.request_id,
                at=self._clock.now(),
                chat_model=self._chat.model,
                embedding_model=self._embedding_model,
                usage=Usage(),
                embedding_tokens=embedding_tokens,
                cost_micros=cost.micros,
                cost_estimated=cost.estimated,
                latency_ms=max(0, int((self._clock.monotonic() - started) * 1000)),
                outcome="error",
            )
        )
