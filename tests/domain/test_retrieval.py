from __future__ import annotations

from rag_chatbot.domain.retrieval import RRF_K, build_retrieval, confidence_of, fuse, select_context
from support.builders import hit, retrieved


def test_a_chunk_found_by_both_searches_outranks_one_found_by_one() -> None:
    fused = fuse([hit("a", 0.9), hit("b", 0.8)], [hit("b", 0.8), hit("c", 0.4)])
    assert [r.chunk.chunk_id for r in fused] == ["b", "a", "c"]
    b = fused[0]
    assert b.vector_rank == 2 and b.keyword_rank == 1
    assert abs(b.fused_score - (1 / (RRF_K + 2) + 1 / (RRF_K + 1))) < 1e-12


def test_fusion_with_one_empty_list_keeps_the_other_order() -> None:
    assert [r.chunk.chunk_id for r in fuse([hit("a", 0.9), hit("b", 0.5)], [])] == ["a", "b"]
    assert [r.chunk.chunk_id for r in fuse([], [hit("x", 0.2), hit("y", 0.1)])] == ["x", "y"]
    assert fuse([], []) == []


def test_ties_are_broken_by_similarity_then_id_so_the_order_is_deterministic() -> None:
    fused = fuse([hit("b", 0.5)], [hit("a", 0.9)])  # both rank 1 in their list: same fused score
    assert [r.chunk.chunk_id for r in fused] == ["a", "b"]
    fused = fuse([hit("b", 0.5)], [hit("a", 0.5)])
    assert [r.chunk.chunk_id for r in fused] == ["a", "b"]


def test_a_duplicate_in_one_list_counts_once_at_its_best_rank() -> None:
    fused = fuse([hit("a", 0.9), hit("a", 0.9), hit("b", 0.1)], [])
    assert len(fused) == 2 and fused[0].vector_rank == 1 and fused[1].vector_rank == 3


def test_similarity_is_clamped_to_the_unit_interval() -> None:
    fused = fuse([hit("a", 1.7)], [hit("b", -0.4)])
    assert {r.chunk.chunk_id: r.similarity for r in fused} == {"a": 1.0, "b": 0.0}


def test_context_selection_respects_the_chunk_and_character_budgets() -> None:
    items = [retrieved(f"c{i}", 0.9 - i / 100, "x" * 100) for i in range(10)]
    assert len(select_context(items, max_chunks=3, max_chars=10_000)) == 3
    assert len(select_context(items, max_chunks=10, max_chars=250)) == 2


def test_the_best_chunk_is_kept_even_when_it_alone_is_over_budget() -> None:
    items = [retrieved("big", 0.9, "x" * 500), retrieved("small", 0.8, "y" * 10)]
    chosen = select_context(items, max_chunks=5, max_chars=100)
    assert [c.chunk.chunk_id for c in chosen] == ["big"]


def test_a_chunk_that_does_not_fit_is_skipped_but_smaller_later_ones_still_can_be_used() -> None:
    items = [retrieved("a", 0.9, "x" * 60), retrieved("b", 0.8, "y" * 60), retrieved("c", 0.7, "z" * 30)]
    assert [c.chunk.chunk_id for c in select_context(items, max_chunks=5, max_chars=100)] == ["a", "c"]


def test_confidence_is_the_best_similarity_not_the_fused_score() -> None:
    assert confidence_of([retrieved("a", 0.2), retrieved("b", 0.71)]) == 0.71
    assert confidence_of([]) == 0.0


def test_a_keyword_only_hit_with_low_similarity_does_not_raise_confidence() -> None:
    result = build_retrieval(fuse([], [hit("kw", 0.05)]), max_chunks=5, max_chars=1000)
    assert result.chunks and result.confidence == 0.05


def test_build_retrieval_exposes_the_chunk_ids() -> None:
    result = build_retrieval(fuse([hit("a", 0.9), hit("b", 0.8)], []), max_chunks=1, max_chars=1000)
    assert result.chunk_ids == frozenset({"a"})


def test_chunks_much_less_similar_than_the_best_are_left_out_of_the_context() -> None:
    items = [retrieved("a", 0.80), retrieved("b", 0.45), retrieved("c", 0.39), retrieved("d", 0.10)]
    kept = select_context(items, max_chunks=5, max_chars=10_000, relative_cutoff=0.5)
    assert [c.chunk.chunk_id for c in kept] == ["a", "b"]  # 0.39 < 0.5 x 0.80 and 0.10 too
    assert len(select_context(items, max_chunks=5, max_chars=10_000, relative_cutoff=0.0)) == 4


def test_the_cutoff_is_relative_to_the_most_similar_chunk_not_to_the_first_by_fusion_rank() -> None:
    items = [retrieved("keyword_only", 0.10), retrieved("best", 0.70), retrieved("close", 0.40)]
    kept = select_context(items, max_chunks=5, max_chars=10_000, relative_cutoff=0.5)
    assert [c.chunk.chunk_id for c in kept] == [
        "best",
        "close",
    ]  # even the top of the fusion goes if it is a weak match
    assert [c.chunk.chunk_id for c in select_context(items, max_chunks=5, max_chars=10_000, relative_cutoff=0.6)] == [
        "best"
    ]


def test_build_retrieval_passes_the_cutoff_and_reports_the_confidence_of_what_was_kept() -> None:
    result = build_retrieval(
        fuse([hit("a", 0.9), hit("b", 0.2)], []), max_chunks=5, max_chars=1000, relative_cutoff=0.5
    )
    assert result.chunk_ids == frozenset({"a"}) and result.confidence == 0.9
