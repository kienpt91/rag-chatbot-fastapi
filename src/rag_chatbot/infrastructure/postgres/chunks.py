"""The chunk store: documents, chunks and their embeddings in PostgreSQL with pgvector."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence

from rag_chatbot.domain.errors import InvalidRequestError, StorageError
from rag_chatbot.domain.model import Chunk, SearchHit
from rag_chatbot.infrastructure.postgres.pool import Pool, storage_errors, tenant_transaction

_TERM = re.compile(r"^[^\W_]+$")
# The inner query orders by the distance alone, which is the one form the HNSW index can answer; the outer one only
# makes ties deterministic. (ORDER BY distance, chunk_id in one query would silently never use the index.)
_VECTOR_SQL = (
    "SELECT chunk_id, document_id, ordinal, title, url, content, 1 - dist FROM ("
    "SELECT chunk_id, document_id, ordinal, title, url, content, embedding <=> %s::vector AS dist "
    "FROM chunks WHERE tenant_id = %s ORDER BY embedding <=> %s::vector LIMIT %s"
    ") nearest ORDER BY dist, chunk_id"
)
_KEYWORD_SQL = (
    "SELECT chunk_id, document_id, ordinal, title, url, content, 1 - (embedding <=> %s::vector) "
    "FROM chunks WHERE tenant_id = %s AND search @@ to_tsquery('simple', %s) "
    "ORDER BY ts_rank_cd(search, to_tsquery('simple', %s)) DESC, chunk_id LIMIT %s"
)


def vector_literal(vector: Sequence[float]) -> str:
    if not all(math.isfinite(x) for x in vector):
        raise StorageError("an embedding contains a value that is not a finite number")
    return "[" + ",".join(format(x, ".9g") for x in vector) + "]"


def tsquery_of(terms: Sequence[str]) -> str:
    """`'sunday' | 'opening'`: any of the words. Each term is a run of letters and digits (checked), quoted."""
    if not terms or not all(_TERM.match(t) for t in terms):
        raise InvalidRequestError("search terms must be words")
    return " | ".join(f"'{t}'" for t in terms)


def _hit(tenant_id: str, row: tuple[object, ...]) -> SearchHit:
    chunk_id, document_id, ordinal, title, url, content, similarity = row
    value = float(similarity) if isinstance(similarity, (int, float)) else 0.0  # NaN for a zero vector
    value = 0.0 if math.isnan(value) else min(1.0, max(0.0, value))
    return SearchHit(
        Chunk(str(chunk_id), tenant_id, str(document_id), int(str(ordinal)), str(title), str(url), str(content)),
        value,
    )


class PostgresChunkStore:
    def __init__(self, pool: Pool, *, dimensions: int) -> None:
        self._pool = pool
        self._dimensions = dimensions

    def _check(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dimensions:
            raise StorageError(f"an embedding has {len(vector)} values, the column holds {self._dimensions}")

    async def known_chunk_ids(self, tenant_id: str, document_id: str) -> frozenset[str]:
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute(
                    "SELECT chunk_id FROM chunks WHERE tenant_id = %s AND document_id = %s", (tenant_id, document_id)
                )
                return frozenset(str(row[0]) for row in await cursor.fetchall())

    async def upsert_document(
        self,
        tenant_id: str,
        document_id: str,
        *,
        url: str,
        title: str,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, tuple[float, ...]],
    ) -> None:
        for chunk in chunks:
            if chunk.tenant_id != tenant_id or chunk.document_id != document_id:
                raise StorageError("a chunk does not belong to the document it is stored with")
        for vector in embeddings.values():
            self._check(vector)
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                await conn.execute(
                    "INSERT INTO documents (tenant_id, document_id, url, title) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (tenant_id, document_id) DO UPDATE "
                    "SET url = EXCLUDED.url, title = EXCLUDED.title, updated_at = now()",
                    (tenant_id, document_id, url, title),
                )
                keep = [c.chunk_id for c in chunks]
                await conn.execute(
                    "DELETE FROM chunks WHERE tenant_id = %s AND document_id = %s AND NOT (chunk_id = ANY(%s))",
                    (tenant_id, document_id, keep),
                )
                cursor = await conn.execute(
                    "SELECT chunk_id FROM chunks WHERE tenant_id = %s AND document_id = %s", (tenant_id, document_id)
                )
                existing = {str(row[0]) for row in await cursor.fetchall()}
                new = [c for c in chunks if c.chunk_id not in existing]
                missing = [c.chunk_id for c in new if c.chunk_id not in embeddings]
                if missing:
                    raise StorageError(f"{len(missing)} new chunk(s) have no embedding")
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "INSERT INTO chunks (tenant_id, chunk_id, document_id, ordinal, title, url, content, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector) "
                        "ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET "
                        "ordinal = EXCLUDED.ordinal, title = EXCLUDED.title, url = EXCLUDED.url",
                        [
                            (
                                tenant_id,
                                c.chunk_id,
                                document_id,
                                c.ordinal,
                                c.title,
                                c.url,
                                c.text,
                                vector_literal(embeddings[c.chunk_id]),
                            )
                            for c in new
                        ],
                    )
                    await cur.executemany(
                        "UPDATE chunks SET ordinal = %s, title = %s, url = %s WHERE tenant_id = %s AND chunk_id = %s",
                        [(c.ordinal, c.title, c.url, tenant_id, c.chunk_id) for c in chunks if c.chunk_id in existing],
                    )

    async def prune_documents(self, tenant_id: str, keep_document_ids: Collection[str]) -> int:
        keep = list(keep_document_ids)
        if not keep:
            raise InvalidRequestError("refusing to remove every document of a tenant")
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute(
                    "DELETE FROM documents WHERE tenant_id = %s AND NOT (document_id = ANY(%s))", (tenant_id, keep)
                )
                return cursor.rowcount

    async def vector_search(self, tenant_id: str, embedding: Sequence[float], limit: int) -> list[SearchHit]:
        self._check(embedding)
        literal = vector_literal(embedding)
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                # a tenant is a small slice of the index: without iterative scans the approximate search could
                # return fewer than `limit` of its chunks (or none) because it filters after reading the index
                await conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
                cursor = await conn.execute(_VECTOR_SQL, (literal, tenant_id, literal, limit))
                return [_hit(tenant_id, row) for row in await cursor.fetchall()]

    async def keyword_search(
        self, tenant_id: str, terms: Sequence[str], embedding: Sequence[float], limit: int
    ) -> list[SearchHit]:
        self._check(embedding)
        query = tsquery_of(terms)
        literal = vector_literal(embedding)
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute(_KEYWORD_SQL, (literal, tenant_id, query, query, limit))
                return [_hit(tenant_id, row) for row in await cursor.fetchall()]

    async def count_chunks(self, tenant_id: str) -> int:
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute("SELECT count(*) FROM chunks WHERE tenant_id = %s", (tenant_id,))
                row = await cursor.fetchone()
                return int(row[0]) if row else 0
