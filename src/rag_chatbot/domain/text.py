"""Small text helpers shared by chunking, retrieval, prompt building and the answer checks."""

from __future__ import annotations

import re
import unicodedata

# Characters that carry no visible text but can hide or reorder it: other control characters, zero-width and
# bidirectional marks, word joiners, byte order marks, surrogates and private-use code points.
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Zl", "Zp"})
_KEEP_CONTROLS = frozenset({"\n", "\t"})
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SPACES = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")

# Function words that match almost every English passage. A keyword query made of them finds noise, so they are
# not search terms. Replace the set for another language (see docs/adapting.md).
STOPWORDS_EN = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "may",
        "me",
        "more",
        "most",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "she",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "up",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def estimate_tokens(text: str) -> int:
    """A cheap, provider-independent estimate: four characters per token, at least one for non-empty text."""
    return (len(text) + 3) // 4 if text else 0


def strip_invisible(text: str) -> str:
    """Remove NUL, control characters (except newline and tab) and format characters; line and paragraph
    separators become newlines. Does not change what a person sees."""
    out: list[str] = []
    for ch in text:
        if ch in _KEEP_CONTROLS:
            out.append(ch)
            continue
        category = unicodedata.category(ch)
        if category in ("Zl", "Zp"):
            out.append("\n")
        elif category not in _INVISIBLE_CATEGORIES:
            out.append(ch)
    return "".join(out)


def normalise_text(text: str) -> str:
    """NFC, no invisible characters, `\\r\\n` as `\\n`, trailing spaces removed, at most one blank line in a row."""
    cleaned = strip_invisible(unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")))
    lines = [_SPACES.sub(" ", line).strip() for line in cleaned.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if line:
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank == 1 and out:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def fold(text: str) -> str:
    """A form for comparing text: NFKC, case-folded, invisible characters dropped, spaces collapsed. NFKC maps
    full-width and other look-alike letters onto the ones a pattern is written with."""
    folded = unicodedata.normalize("NFKC", strip_invisible(text)).casefold()
    return _SPACES.sub(" ", folded.replace("\n", " ")).strip()


def words(text: str) -> list[str]:
    """Lower-case word tokens (letters and digits of any script), in order."""
    return _WORD.findall(fold(text))


def search_terms(text: str, *, stopwords: frozenset[str] = STOPWORDS_EN, limit: int = 12) -> list[str]:
    """The distinct, meaningful words of a question, longest first within the limit, in a stable order."""
    seen: dict[str, None] = {}
    for word in words(text):
        if len(word) >= 2 and word not in stopwords and word not in seen:
            seen[word] = None
    ordered = sorted(seen, key=lambda w: (-len(w), w))[:limit]
    return ordered


def truncate_at_boundary(text: str, limit: int) -> str:
    """Cut `text` to at most `limit` characters, at a line or word boundary when one is near."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for separator in ("\n", " "):
        index = cut.rfind(separator)
        if index >= limit * 0.6:
            return cut[:index].rstrip()
    return cut.rstrip()
