from __future__ import annotations

import itertools

import pytest

from rag_chatbot.domain.chunking import ChunkingConfig, chunk_document, chunk_id_for, split_text
from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.model import SourceDocument

SMALL = ChunkingConfig(target_tokens=50, max_tokens=80, overlap_tokens=10, min_tokens=10)


def doc(text: str, document_id: str = "faq") -> SourceDocument:
    return SourceDocument("acme", document_id, "https://acme.example/faq", "FAQ", text)


def paragraphs(n: int, size: int = 120) -> str:
    return "\n\n".join(f"Paragraph {i} " + ("lorem ipsum " * (size // 12)).strip() + "." for i in range(n))


def test_empty_and_blank_text_make_no_chunks() -> None:
    assert chunk_document(doc("")) == []
    assert chunk_document(doc("  \n\n \t ")) == []


def test_a_short_page_is_one_chunk() -> None:
    chunks = chunk_document(doc("We open at nine.\n\nWe close at five."))
    assert [c.text for c in chunks] == ["We open at nine.\n\nWe close at five."]
    assert chunks[0].ordinal == 0 and chunks[0].tenant_id == "acme" and chunks[0].title == "FAQ"


def test_every_chunk_is_within_the_hard_limit_and_no_text_is_lost() -> None:
    text = paragraphs(60)
    chunks = split_text(text, SMALL)
    assert len(chunks) > 5
    assert all(len(c) <= SMALL.max_chars for c in chunks)
    joined = "\n".join(chunks)
    for i in range(60):
        assert f"Paragraph {i} " in joined


def test_neighbouring_chunks_overlap_by_whole_paragraphs_when_they_are_short() -> None:
    text = "\n\n".join(f"Short line number {i} here." for i in range(80))
    chunks = split_text(text, SMALL)
    assert len(chunks) > 3
    for before, after in itertools.pairwise(chunks):
        assert before.split("\n\n")[-1] in after


def test_long_paragraphs_overlap_by_their_last_sentences() -> None:
    sentences = [f"Sentence {i} says something quite specific." for i in range(40)]
    text = "\n\n".join(" ".join(sentences[i : i + 4]) for i in range(0, 40, 4))
    chunks = split_text(text, ChunkingConfig(target_tokens=50, max_tokens=80, overlap_tokens=20, min_tokens=10))
    assert len(chunks) > 3
    carried = 0
    for before, after in itertools.pairwise(chunks):
        last_sentence = before.split(". ")[-1].strip()
        if last_sentence in after:
            carried += 1
    assert carried >= len(chunks) // 2


def test_no_overlap_when_it_is_zero() -> None:
    config = ChunkingConfig(target_tokens=50, max_tokens=80, overlap_tokens=0, min_tokens=10)
    chunks = split_text(paragraphs(30, 100), config)
    for before, after in itertools.pairwise(chunks):
        assert before.split("\n\n")[-1] not in after


def test_chunks_start_with_the_heading_of_their_section() -> None:
    text = "# Shop\n\n## Opening hours\n\n" + paragraphs(12, 150)
    chunks = split_text(text, SMALL)
    assert len(chunks) > 3
    assert all(c.startswith("## Opening hours") for c in chunks[1:])


def test_a_new_section_starts_a_new_chunk_and_does_not_overlap_backwards() -> None:
    text = (
        "## Delivery\n\n" + "We deliver within the city every weekday morning. " * 4 + "\n\n"
        "## Returns\n\n" + "Unopened goods can be returned within thirty days. " * 4
    )
    chunks = split_text(text, SMALL)
    returns = [c for c in chunks if "returned" in c]
    assert returns and all(c.startswith("## Returns") for c in returns)
    assert not any("deliver" in c and "returned" in c for c in chunks)


def test_a_trailing_heading_joins_the_previous_chunk() -> None:
    chunks = split_text("Some text about bread.\n\n## Contact", SMALL)
    assert len(chunks) == 1 and chunks[0].endswith("## Contact")


def test_a_giant_sentence_and_a_giant_word_are_cut_to_the_limit() -> None:
    sentence = " ".join(["word"] * 2000) + "."
    assert all(len(c) <= SMALL.max_chars for c in split_text(sentence, SMALL))
    blob = "x" * 100_000
    pieces = split_text(blob, SMALL)
    assert len(pieces) > 100 and all(len(c) <= SMALL.max_chars for c in pieces)
    assert "".join(pieces).count("x") >= 100_000 - 500  # overlap never drops text, hard cuts lose nothing


def test_a_very_long_heading_cannot_break_the_limit() -> None:
    chunks = split_text("# " + "h" * 5000 + "\n\nbody text here.", SMALL)
    assert all(len(c) <= SMALL.max_chars for c in chunks)


def test_ids_are_stable_unique_and_content_addressed() -> None:
    text = paragraphs(20)
    first = chunk_document(doc(text), SMALL)
    second = chunk_document(doc(text), SMALL)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(c.chunk_id.startswith("ck_") and len(c.chunk_id) == 27 for c in first)


def test_ids_differ_by_tenant_and_document_and_text() -> None:
    base = chunk_id_for("acme", "faq", "text", 0)
    assert base != chunk_id_for("other", "faq", "text", 0)
    assert base != chunk_id_for("acme", "menu", "text", 0)
    assert base != chunk_id_for("acme", "faq", "text!", 0)
    assert base != chunk_id_for("acme", "faq", "text", 1)


def test_an_edit_changes_only_the_chunks_it_touches() -> None:
    parts = [f"Section {i} " + "alpha beta gamma delta " * 12 for i in range(12)]
    before = chunk_document(doc("\n\n".join(parts)), SMALL)
    edited = list(parts)
    edited[10] = "Section 10 was rewritten completely."
    after = chunk_document(doc("\n\n".join(edited)), SMALL)
    unchanged = {c.chunk_id for c in before} & {c.chunk_id for c in after}
    assert len(unchanged) >= len(before) - 4


def test_inserting_a_paragraph_at_the_top_keeps_the_ids_of_later_chunks_that_do_not_overlap_it() -> None:
    config = ChunkingConfig(target_tokens=50, max_tokens=80, overlap_tokens=0, min_tokens=10)
    parts = [f"Part {i} " + "alpha beta gamma delta " * 9 for i in range(12)]
    before = {c.chunk_id for c in chunk_document(doc("\n\n".join(parts)), config)}
    after = {c.chunk_id for c in chunk_document(doc("New intro paragraph.\n\n" + "\n\n".join(parts)), config)}
    assert len(before & after) >= len(before) - 3  # only the chunks whose packing the insertion shifted differ


def test_identical_chunks_in_one_document_get_different_ids() -> None:
    config = ChunkingConfig(target_tokens=20, max_tokens=30, overlap_tokens=0, min_tokens=5)
    line = "Repeat this exact paragraph of filler text now."
    chunks = chunk_document(doc("\n\n".join([line] * 6)), config)
    assert len(chunks) >= 2
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_the_ordinal_counts_chunks_in_order() -> None:
    chunks = chunk_document(doc(paragraphs(20)), SMALL)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens": 10, "max_tokens": 5},
        {"min_tokens": 0},
        {"overlap_tokens": 220},
        {"overlap_tokens": -1},
        {"min_tokens": 300},
    ],
)
def test_nonsense_configuration_is_refused(kwargs: dict[str, int]) -> None:
    with pytest.raises(ConfigurationError):
        ChunkingConfig(**kwargs)


def test_the_token_estimate_of_a_chunk_follows_its_length() -> None:
    chunk = chunk_document(doc("x" * 400))[0]
    assert chunk.token_estimate == 100
