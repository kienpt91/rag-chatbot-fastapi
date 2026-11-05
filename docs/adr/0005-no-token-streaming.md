# 5. No token streaming

Status: accepted

## Context

Server-sent events would let the widget show words as they are generated.

## Decision

Not implemented. Every answer is checked as a whole (citations, leaks, links) before anyone sees it, so streaming
raw model text would show unchecked text, and streaming the checked text afterwards would only simulate it.

## Consequences

Latency to the first word equals latency to the whole answer; the widget shows a waiting indicator. If a client
needs streaming, the checks have to be reworked into an incremental form first; that is a design change, not a switch.
