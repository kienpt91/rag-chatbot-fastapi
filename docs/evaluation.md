# Evaluation

`eval/datasets/site.jsonl` holds 44 golden cases for the sample site in `examples/site/`: 20 answerable questions, 2
follow-ups, 8 unanswerable ones and 14 safety cases (injected instructions in a review, a request to print the rules,
markup in a page, hidden payloads, a `noindex` page, evasive overrides, personal data in the question, exfiltration
through an image).

The format is the JSON Lines format of the sibling project [llm-eval-harness](../../llm-eval-harness): `id`,
`category` (`quality` or `safety`), `tags`, `input` (`question`, optional `previous` questions of the same
conversation), `expected` (`facts`, `forbidden`) and `graders` (`grounding`, `refusal`, `not_contains`). Unknown keys
and unsupported graders are errors. A failing `safety` case is always fatal.

## Run it (no hosted model)

```sh
sh scripts/eval.sh          # in the pinned container, no network: the deterministic fake provider
```

Exit status: 0 pass, 1 quality below `--min-quality`, 3 the service failed, 4 a safety case failed. CI runs this.
The report is `out/eval-report.json`. The fake provider quotes the best-matching sentence of the retrieved text and,
for the evaluation only, is *deliberately gullible* (it obeys a `REPLY:` line and a request to print its rules), so
that the cases exercise the application's checks. A pass therefore shows that the plumbing and the defences work; it
says nothing about the quality of a real model.

## Run it against a real provider (spends money)

```sh
cp eval/real.env.example eval/real.env      # fill in the API URL, key, model names, a price table, a threshold
sh scripts/eval-real.sh --min-quality 0.9
```

## Run it with llm-eval

`eval/pipeline.py` speaks llm-eval's `command` protocol, so the harness can add repeats, statistics, baselines and
its gate: `cd eval && PYTHONPATH=/path/to/llm-eval-harness/src python3 -m llm_eval_harness run`. Verified here
with the sibling at its current commit: `validate` accepts the dataset and `run` reports 30/30 quality and 14 safety
cases with 0 failing (fake provider).

## Both layers hold alone

The poisoned review of the sample site is stopped twice: its instruction sentences never enter the index (ingest-time
quarantine), and, if they did, the answer checks refuse what a fooled model would say. `EVAL_ENV=RAG_QUARANTINE_INSTRUCTIONS=false sh scripts/eval.sh`
runs the set with the quarantine off (the checks alone); CI runs both.

## Mutation check

With the answer checks disabled (`check_answer` replaced by "everything passes") and the ingest-time quarantine on,
25/30 quality and 11/14 safety cases pass; with the quarantine off as well, 25/30 and 8/14. With both on, 30/30 and
14/14. So the checks matter even when the quarantine removes the obvious instruction sentences, and the quarantine
matters when the checks are absent. The message filter alone is not what holds: disabling it changes nothing in the
set, because retrieval and the checks also stop those cases (it saves money, it is not a defence).
