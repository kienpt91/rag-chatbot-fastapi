# 2. Hybrid retrieval and what "confidence" means

Status: accepted

## Context

A site chatbot must answer only from the site and say "I don't know" otherwise. That needs a number that
means the same for every question, which an operator can put a threshold on.

## Decision

Retrieval runs a vector search (pgvector, cosine) and a keyword search (`tsvector`, any of the question's terms)
and merges them with reciprocal rank fusion, which does not need the two scores to be comparable. **Confidence is not
the fused score** (the best of a bad lot still ranks first): it is the cosine similarity of the best chunk that goes to
the model. Below the threshold the model is not called at all. A keyword hit with low similarity does not raise
confidence, because that is how an off-topic question that shares a common word with the site gets through. Chunks much
less similar than the best (`RAG_CONTEXT_RELATIVE_CUTOFF`) are not given to the model: they cost tokens and each one is
text someone wrote on a website.

## Consequences

The threshold is a property of the embedding model: the default (0.30) is a starting point, and the fake provider
needs a much lower one. docs/adapting.md describes tuning it with the evaluation set. The filtered HNSW search needs
iterative scans (pgvector 0.8), and the SQL keeps the index usable by ordering by distance alone in the inner query:
both are covered by tests that were checked to fail without them.
