"""Text extraction: a `RawPage` (HTML, Markdown or plain text) becomes plain text ready for chunking.

All three readers produce the same shape, which the chunker relies on:

* paragraphs are separated by one blank line;
* a heading is its own paragraph, written `## Opening hours` (1-6 `#`);
* a list is one paragraph with one `- item` per line (ordered lists too);
* a table is one paragraph with one line per row, cells joined by ` | `;
* preformatted and code blocks keep their line breaks, `<br>` is a line break inside a paragraph;
* inline whitespace is collapsed, there is no trailing whitespace and no HTML entity is left;
* the text is NFC-normalised and free of invisible and dangerous characters (control, format, bidirectional,
  zero-width, private-use and surrogate code points).

The page's content is data: nothing here interprets or filters what it says. What is not visible to a visitor
(hidden elements, comments, scripts) is dropped so that it can never reach an index or a prompt.
"""

from __future__ import annotations

from rag_chatbot.domain.htmlextract import extract_html
from rag_chatbot.domain.model import RawPage
from rag_chatbot.domain.textextract import extract_markdown, extract_plain
from rag_chatbot.domain.textnorm import MAX_INPUT_CHARS, MAX_LINKS, MAX_NESTING_DEPTH, MAX_TEXT_CHARS, ExtractedPage

__all__ = [
    "MAX_INPUT_CHARS",
    "MAX_LINKS",
    "MAX_NESTING_DEPTH",
    "MAX_TEXT_CHARS",
    "ExtractedPage",
    "extract_html",
    "extract_markdown",
    "extract_page",
    "extract_plain",
]


def extract_page(page: RawPage) -> ExtractedPage:
    """Extract by `page.kind`; links of an HTML page are resolved against `page.url` ("" keeps only absolute ones)."""
    if page.kind == "html":
        return extract_html(page.content, base_url=page.url)
    if page.kind == "markdown":
        return extract_markdown(page.content)
    return extract_plain(page.content)
