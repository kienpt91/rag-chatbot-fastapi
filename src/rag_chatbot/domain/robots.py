"""robots.txt: a parser for the group that applies to our crawler and a matcher for paths (RFC 9309).

The file is untrusted input from a website, so everything here is bounded: the text is cut at 500 KiB, at most 1,000
rules are kept, a rule longer than 4,096 characters is ignored and the wildcard matcher never backtracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag_chatbot.domain.urlpolicy import normalise_percent_encoding

MAX_ROBOTS_BYTES = 500 * 1024  # RFC 9309: a crawler may ignore everything after the first 500 KiB
MAX_RULES = 1_000
MAX_PATTERN_CHARS = 4_096  # longer than any URL we request, so a plain rule of that size can never match
MAX_CRAWL_DELAY_S = 60.0

_DIGITS = frozenset("0123456789")


@dataclass(frozen=True, slots=True)
class RobotsRules:
    """The rules of the one user-agent group chosen for our agent."""

    allow: tuple[str, ...]
    disallow: tuple[str, ...]
    crawl_delay_s: float | None

    def allows(self, path_and_query: str) -> bool:
        """The longest matching rule wins, Allow wins a tie, and a path no rule matches is allowed.

        Both the rule and the path are compared in their percent-encoding-normalised form, so `/a%2Fb` is not
        `/a/b` while `/%61` is `/a`. An empty Allow or Disallow rule matches nothing."""
        target = _normalise_target(path_and_query)
        best_length = -1
        best_allows = True
        for patterns, is_allow in ((self.allow, True), (self.disallow, False)):
            for raw in patterns:
                pattern = normalise_percent_encoding(raw)
                if not pattern or not _matches(pattern, target):
                    continue
                length = len(pattern)
                if length > best_length or (length == best_length and is_allow and not best_allows):
                    best_length = length
                    best_allows = is_allow
        return best_allows


ALLOW_ALL = RobotsRules(allow=(), disallow=(), crawl_delay_s=None)
DISALLOW_ALL = RobotsRules(allow=(), disallow=("/",), crawl_delay_s=None)


def parse_robots(text: str, user_agent: str) -> RobotsRules:
    """The rules for `user_agent`: the group whose agent token is the most specific match for our product token
    (case-insensitive, exact or prefix), else the `*` group, else everything allowed. Groups of the same agent are
    merged. Comments, unknown fields and malformed lines are ignored."""
    token = _agent_token(user_agent)
    accumulators: dict[int, _Accumulator] = {}
    group_best = -1  # specificity of the current group for us; -1 when the group is not ours
    in_rules = False  # a rule line was seen since the last user-agent line, so the next agent starts a new group
    for line in _lines(_cap(text)):
        field, value = _split_field(line)
        if field == "user-agent":
            if in_rules:
                group_best = -1
                in_rules = False
            group_best = max(group_best, _specificity(value, token))
            if group_best >= 0:
                accumulators.setdefault(group_best, _Accumulator())
        elif field in ("allow", "disallow", "crawl-delay"):
            in_rules = True
            if group_best >= 0:
                accumulators[group_best].add(field, value)
    if not accumulators:
        return ALLOW_ALL
    return accumulators[max(accumulators)].to_rules()


def rules_for_status(status: int, text: str, user_agent: str) -> RobotsRules:
    """RFC 9309: a 2xx answer is parsed; a 4xx answer (except 429) means there is no robots.txt, so everything is
    allowed; any other outcome (5xx, 429, redirects we could not resolve) means the site is unreachable and we
    stay out."""
    if 200 <= status < 300:
        return parse_robots(text, user_agent)
    if 400 <= status < 500 and status != 429:
        return ALLOW_ALL
    return DISALLOW_ALL


class _Accumulator:
    """The merged rules of every group that applies to one specificity level."""

    def __init__(self) -> None:
        self._allow: list[str] = []
        self._disallow: list[str] = []
        self._delay: float | None = None

    def add(self, field: str, value: str) -> None:
        if field == "crawl-delay":
            delay = _parse_delay(value)
            if delay is not None:
                self._delay = delay if self._delay is None else max(self._delay, delay)
            return
        pattern = normalise_percent_encoding(value)
        # An empty rule is legal ("Disallow:" allows everything) and adds nothing; a pattern must anchor at the path.
        if not pattern or len(pattern) > MAX_PATTERN_CHARS or pattern[0] not in "/*":
            return
        if len(self._allow) + len(self._disallow) >= MAX_RULES:
            return
        (self._allow if field == "allow" else self._disallow).append(pattern)

    def to_rules(self) -> RobotsRules:
        return RobotsRules(tuple(self._allow), tuple(self._disallow), self._delay)


def _cap(text: str) -> str:
    """The first 500 KiB of the file, without a trailing partial line (a cut-off rule is not a rule)."""
    head = text[:MAX_ROBOTS_BYTES]  # a character is at least one byte, so this already bounds the work
    encoded = head.encode("utf-8", "replace")
    if len(head) == len(text) and len(encoded) <= MAX_ROBOTS_BYTES:
        return text
    cut = encoded[:MAX_ROBOTS_BYTES].decode("utf-8", "ignore")
    last_break = max(cut.rfind("\n"), cut.rfind("\r"))
    return cut[: last_break + 1] if last_break >= 0 else ""


def _lines(text: str) -> list[str]:
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _split_field(line: str) -> tuple[str, str]:
    comment = line.find("#")
    if comment >= 0:
        line = line[:comment]
    field, separator, value = line.partition(":")
    if not separator:
        return "", ""
    return field.strip().lower(), value.strip()


def _agent_token(value: str) -> str:
    """The product token of a user-agent string: `RagBot/1.0 (+https://x)` -> `ragbot`."""
    token: list[str] = []
    for char in value.strip().lower():
        if char.isascii() and (char.isalnum() or char in "-_"):
            token.append(char)
        else:
            break
    return "".join(token)


def _specificity(group_agent: str, our_token: str) -> int:
    """0 for `*`, the length of the agent token for a named match, -1 when the group is not for us."""
    if group_agent.strip() == "*":
        return 0
    agent = _agent_token(group_agent)
    if agent and our_token.startswith(agent):
        return len(agent)
    return -1


def _parse_delay(value: str) -> float | None:
    """A plain non-negative decimal (no signs, exponents, underscores, NaN or infinity), capped at 60 seconds."""
    digits = value.replace(".", "", 1)
    if not digits or not _DIGITS.issuperset(digits):
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if not math.isfinite(seconds):
        return None
    return min(seconds, MAX_CRAWL_DELAY_S)


def _normalise_target(path_and_query: str) -> str:
    fragment = path_and_query.find("#")
    if fragment >= 0:
        path_and_query = path_and_query[:fragment]
    target = normalise_percent_encoding(path_and_query)
    return target if target.startswith("/") else "/" + target


def _matches(pattern: str, target: str) -> bool:
    """Does `pattern` (a prefix pattern; `*` matches any run, a final `$` anchors the end) match `target`?

    The pattern is cut at its stars into literal pieces. The first piece must open the target, the middle pieces
    are located left to right at their earliest position (which leaves the most room for the rest, so greedy is
    exact) and the last piece must close the target when anchored. Each piece is searched once from where the
    previous one ended, so the work is linear in the target for every piece; nothing backtracks."""
    anchored = pattern.endswith("$")
    if anchored:
        pattern = pattern[:-1]
    pieces = pattern.split("*")
    first = pieces[0]
    if not target.startswith(first):
        return False
    position = len(first)
    if len(pieces) == 1:
        return target == first if anchored else True
    for piece in pieces[1:-1]:
        found = target.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    last = pieces[-1]
    if anchored:
        return target.endswith(last) and len(target) - len(last) >= position
    return not last or target.find(last, position) >= 0
