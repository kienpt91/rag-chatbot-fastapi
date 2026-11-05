# 6. Dependencies and the standard library

Status: accepted

## Context

Every dependency is code that runs with the keys and the customers' conversations.

## Decision

Runtime: FastAPI, uvicorn, pydantic (required by FastAPI) and psycopg 3 with its pool, plus their pinned
closure; everything else is the standard library: the HTML extractor, the crawler and fetcher, the provider clients
(`http.client` in a worker thread, so no HTTP client package is needed at run time), the redactor, the rate limiter,
the token signing (`hmac`). All packages are pinned by exact version and sha256 hash, none younger than seven days at
the cutoff, checked in CI by a script that is itself tested. The widget has no dependencies and no build step.

## Consequences

The standard-library HTML tokenizer is our own, because `html.parser` rescans the remainder of the input for every
stray `<` (quadratic on `<a<a<a...`). The price is more code of our own to test; coverage and the hostile-input tests
are the answer to that.
