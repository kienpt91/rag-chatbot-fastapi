# 4. Answers are checked, and a failed check refuses

Status: accepted

## Context

A language model that was fooled by a page or a visitor produces an answer that looks fine.

## Decision

The model gets no tools and sees untrusted text fenced with a per-request nonce. Its answer must cite sources it
was shown, must not contain the rules, the internal marker or an address outside the site's own hosts, and loses its
markup. Anything else becomes a refusal that is not stored, cited or displayed. The set of linkable hosts is the
tenant's origins plus declared link hosts, not "any address that appears in a page": a poisoned review that advertises
a phishing address would otherwise pass. Citation snippets are scrubbed by the same rule.

## Consequences

The checks cannot judge truth: a correctly cited sentence that a customer wrote is passed on. The remedy is
operational (`ingest --exclude`), and SECURITY.md says so. A retrieved poisoned chunk can turn a good answer into a
refusal; the system fails closed.
