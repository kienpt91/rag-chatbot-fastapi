"""Running an evaluation set through the real chat flow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_chatbot.application.chat import ChatRequest, ChatService
from rag_chatbot.application.ports import IdGenerator
from rag_chatbot.domain.errors import ChatbotError
from rag_chatbot.domain.evalset import CaseResult, EvalCase, Observation, dataset_hash, grade
from rag_chatbot.domain.tenants import Tenant


@dataclass(slots=True)
class EvalReport:
    dataset_hash: str
    results: list[CaseResult] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)  # case id -> what the visitor would have seen
    cost_micros: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: list[int] = field(default_factory=list)

    def percentile(self, fraction: float) -> int:
        if not self.latencies_ms:
            return 0
        ordered = sorted(self.latencies_ms)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


class EvalRunner:
    """One fresh conversation per case; the case's `previous` questions are asked first, in the same conversation."""

    def __init__(self, chat: ChatService, tenant: Tenant, ids: IdGenerator) -> None:
        self._chat = chat
        self._tenant = tenant
        self._ids = ids

    async def run(self, cases: Sequence[EvalCase]) -> EvalReport:
        report = EvalReport(dataset_hash(cases))
        for case in cases:
            observation = await self._observe(case, report)
            report.results.append(grade(case, observation))
            report.answers[case.case_id] = observation.answer if not observation.error else f"[{observation.error}]"
        return report

    async def _observe(self, case: EvalCase, report: EvalReport) -> Observation:
        session_id: str | None = None
        try:
            for earlier in case.previous:
                reply = await self._chat.handle(self._request(earlier, session_id))
                session_id = reply.session_id
            reply = await self._chat.handle(self._request(case.question, session_id))
        except ChatbotError as error:
            return Observation("", (), error=error.code)
        report.cost_micros += reply.cost.micros
        report.input_tokens += reply.usage.input_tokens
        report.output_tokens += reply.usage.output_tokens
        report.latencies_ms.append(reply.latency_ms)
        cited = tuple(dict.fromkeys(c.document_id for c in reply.answer.citations))
        return Observation(reply.answer.text, cited)

    def _request(self, message: str, session_id: str | None) -> ChatRequest:
        return ChatRequest(self._tenant, message, session_id, None, self._ids.new_request_id())
