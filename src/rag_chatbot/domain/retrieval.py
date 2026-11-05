"""Combining the vector and the keyword search, and deciding whether retrieval found anything worth answering from.

Reciprocal rank fusion (Cormack et al., 2009) merges two ranked lists without having to compare their scores, which
are not comparable (a cosine similarity and a text-search rank). A chunk's fused score is the sum of
`1 / (k + rank)` over the lists it appears in.

*Confidence* is deliberately not the fused score. The fused score only says how a chunk ranks among the others, and
the best of a bad lot still ranks first. Confidence is the cosine similarity of the question and the best chunk
that is going to be shown to the model: a number that means the same for every question, which an operator can
set a threshold on and tune with the evaluation set. A keyword hit with a low similarity does not raise it: that
is how an off-topic question that happens to share a common word with the site gets through.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_chatbot.domain.model import Retrieval, RetrievedChunk, SearchHit

RRF_K = 60


def fuse(
    vector_hits: Sequence[SearchHit], keyword_hits: Sequence[SearchHit], *, k: int = RRF_K
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion. Deterministic: ties are broken by similarity, then by chunk id."""
    ranks: dict[str, dict[str, int]] = {}
    hits: dict[str, SearchHit] = {}
    for name, ranked in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(ranked, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id in ranks and name in ranks[chunk_id]:
                continue  # a duplicate in one list counts once, at its best rank
            ranks.setdefault(chunk_id, {})[name] = rank
            previous = hits.get(chunk_id)
            if previous is None or hit.similarity > previous.similarity:
                hits[chunk_id] = hit
    fused: list[RetrievedChunk] = []
    for chunk_id, by_list in ranks.items():
        hit = hits[chunk_id]
        score = sum(1.0 / (k + rank) for rank in by_list.values())
        fused.append(
            RetrievedChunk(
                chunk=hit.chunk,
                similarity=min(1.0, max(0.0, hit.similarity)),
                vector_rank=by_list.get("vector"),
                keyword_rank=by_list.get("keyword"),
                fused_score=score,
            )
        )
    fused.sort(key=lambda r: (-r.fused_score, -r.similarity, r.chunk.chunk_id))
    return fused


def select_context(
    fused: Sequence[RetrievedChunk], *, max_chunks: int, max_chars: int, relative_cutoff: float = 0.0
) -> list[RetrievedChunk]:
    """The best chunks that fit the context budget. The best chunk is always kept (truncated by the prompt builder
    if it alone is over budget); later ones are skipped when they would pass the budget.

    `relative_cutoff` drops chunks that are much less similar to the question than the most similar one
    (similarity below `cutoff x best`): they add cost, and they add risk, because every chunk in the context is text
    somebody wrote on a website. The best chunk is judged by similarity, not by its rank in the fusion."""
    best_similarity = max((c.similarity for c in fused), default=0.0)
    chosen: list[RetrievedChunk] = []
    used = 0
    for item in fused:
        if len(chosen) >= max_chunks:
            break
        if item.similarity < relative_cutoff * best_similarity:
            continue
        size = len(item.chunk.text)
        if chosen and used + size > max_chars:
            continue
        chosen.append(item)
        used += size
    return chosen


def confidence_of(chosen: Sequence[RetrievedChunk]) -> float:
    return max((c.similarity for c in chosen), default=0.0)


def build_retrieval(
    fused: Sequence[RetrievedChunk], *, max_chunks: int, max_chars: int, relative_cutoff: float = 0.0
) -> Retrieval:
    chosen = select_context(fused, max_chunks=max_chunks, max_chars=max_chars, relative_cutoff=relative_cutoff)
    return Retrieval(chunks=tuple(chosen), confidence=confidence_of(chosen))
