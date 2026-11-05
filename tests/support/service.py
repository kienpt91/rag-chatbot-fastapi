"""A fully wired chat service over in-memory stores, the deterministic fake embedder and a scripted chat model."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_chatbot.application.chat import ChatRequest, ChatService
from rag_chatbot.application.ingest import IngestReport, IngestService
from rag_chatbot.application.retrieval import Retriever
from rag_chatbot.application.settings import ChatSettings, IngestSettings
from rag_chatbot.domain.model import ChatReply, RawPage
from rag_chatbot.domain.pricing import PriceTable, parse_price_table
from rag_chatbot.domain.tenants import Tenant
from rag_chatbot.infrastructure.memory import InMemoryChunkStore, InMemorySessionStore, InMemoryUsageLedger
from rag_chatbot.infrastructure.providers.fake import FakeChatProvider, FakeEmbeddingProvider
from support.fakes import FakeClock, RecordingEmbedder, ScriptedChat, SequentialIds

DIMENSIONS = 256
PRICES = parse_price_table(
    {
        "currency": "USD",
        "models": {
            "fake-embedding-1": {"input_per_mtok": 100},
            "scripted-chat": {"input_per_mtok": 1000, "output_per_mtok": 3000},
            "fake-chat-1": {"input_per_mtok": 1000, "output_per_mtok": 3000},
        },
    }
)
CANARY = "CANARY-0123456789ab"
TENANT = Tenant("acme", "Maple Bakery", frozenset({"https://www.acme.example"}))

PAGES = [
    RawPage(
        "hours.html",
        "html",
        "<html><head><title>Opening hours</title></head><body><main><h1>Opening hours</h1>"
        "<p>The bakery opens at 7:00 in the morning from Monday to Friday.</p>"
        "<p>On Sunday the bakery is closed.</p></main></body></html>",
        "https://www.acme.example/hours",
    ),
    RawPage(
        "menu.md",
        "markdown",
        "---\ntitle: Menu and prices\n---\n# Menu\n\nA sourdough loaf costs 6.50 dollars.\n\nA croissant costs 3.50 dollars.\n",
        "https://www.acme.example/menu",
    ),
    RawPage(
        "delivery.txt",
        "text",
        "We deliver within 5 kilometres of the shop. The delivery fee is 5 dollars.",
        "https://www.acme.example/delivery",
    ),
]


class ListSource:
    def __init__(self, pages: list[RawPage]) -> None:
        self._pages = pages

    async def pages(self):  # type: ignore[no-untyped-def]
        for page in self._pages:
            yield page


@dataclass
class Service:
    chat_service: ChatService
    ingest: IngestService
    chunks: InMemoryChunkStore
    sessions: InMemorySessionStore
    ledger: InMemoryUsageLedger
    embedder: RecordingEmbedder
    chat: ScriptedChat
    clock: FakeClock
    ids: SequentialIds
    settings: ChatSettings
    prices: PriceTable = field(default=PRICES)

    async def load(self, pages: list[RawPage] | None = None, tenant: str = "acme") -> IngestReport:
        return await self.ingest.run(tenant, ListSource(pages if pages is not None else PAGES))

    async def ask(
        self,
        message: str,
        *,
        session_id: str | None = None,
        tenant: Tenant = TENANT,
        owner_id: str | None = None,
    ) -> ChatReply:
        return await self.chat_service.handle(
            ChatRequest(tenant, message, session_id, owner_id, self.ids.new_request_id())
        )


def build_service(
    settings: ChatSettings | None = None, chat: ScriptedChat | None = None, *, gullible: bool = False
) -> Service:
    settings = settings or ChatSettings(canary=CANARY, confidence_threshold=0.15)
    chunks = InMemoryChunkStore(dimensions=DIMENSIONS)
    sessions = InMemorySessionStore()
    ledger = InMemoryUsageLedger()
    embedder = RecordingEmbedder(FakeEmbeddingProvider(DIMENSIONS))
    if chat is None:
        fake = FakeChatProvider(gullible=gullible, model="scripted-chat")
        chat = ScriptedChat(reply=lambda system, messages: _fake_reply(fake, system, messages))
    clock = FakeClock()
    ids = SequentialIds()
    service = ChatService(
        retriever=Retriever(chunks, embedder, settings),
        chat=chat,
        sessions=sessions,
        ledger=ledger,
        prices=PRICES,
        embedding_model=embedder.model,
        clock=clock,
        ids=ids,
        settings=settings,
    )
    ingest = IngestService(chunks, embedder, PRICES, IngestSettings())
    return Service(service, ingest, chunks, sessions, ledger, embedder, chat, clock, ids, settings)


def _fake_reply(fake: FakeChatProvider, system: str, messages: object) -> str:
    return fake._answer(system, messages[-1].content)  # type: ignore[index,attr-defined]
