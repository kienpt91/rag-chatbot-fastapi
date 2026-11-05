# 8. Quarantine and hidden text at ingest

Status: accepted

## Context

The answer checks (ADR 0004) stop an instruction from steering the answer, but not a harmless-looking claim made by
a sentence somebody wrote on a page, and an independent review showed that what the extractor treats as hidden can
differ from what a browser hides.

## Decision

Two filters run before anything is chunked or embedded. The extractor drops text a browser would not show: hidden
elements, a longer list of hiding styles (off-screen boxes, zero-size boxes, scaled-to-zero, clipped, one-pixel fonts,
text coloured like its background), image alt text, script content under the browser's double-escaped rules, and the
same hidden-element rules for raw HTML inside Markdown. Then `domain/quarantine.py` removes, sentence by sentence,
wording that addresses an assistant (override and extraction phrases, role hijacks, `REPLY:` directives, "the assistant
must ...", "tell the visitors that ..."), keeps the rest of the page, and returns what it removed; the ingest report
prints it. `--exclude` matches crawled URL paths, so user-written sections can be kept out entirely.

## Consequences

Both are filters, not proofs: a rephrased or translated instruction, or text hidden by a stylesheet the extractor
cannot see, passes; an honest sentence that matches is lost and is listed so the operator notices. The guard and the
fenced prompt do not depend on either filter, and the evaluation set passes with the quarantine switched off too.
