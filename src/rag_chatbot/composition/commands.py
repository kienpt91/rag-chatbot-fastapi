"""The commands of the command line. Each takes the settings, its parsed arguments and two text streams, and returns
the exit status: 0 success, 1 the checked thing failed (an evaluation, a pruning refusal), 2 usage or configuration
error, 3 the service or a provider failed, 4 a safety case failed."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import TextIO

import uvicorn

from rag_chatbot.application.evaluation import EvalRunner
from rag_chatbot.application.ingest import IngestReport
from rag_chatbot.composition import wiring
from rag_chatbot.composition.config import Settings
from rag_chatbot.domain.auth import API_KEY_PREFIX, TokenClaims, hash_api_key, mint_token, validate_claims_for_minting
from rag_chatbot.domain.errors import ChatbotError, ConfigurationError
from rag_chatbot.domain.evalset import DatasetError, EvalCase, exit_code, parse_dataset
from rag_chatbot.domain.pricing import PriceTable, format_micros, parse_price_table
from rag_chatbot.domain.tenants import Tenant, TenantDirectory, normalise_origin, parse_tenants
from rag_chatbot.domain.urlpolicy import HostPolicy, normalise_host
from rag_chatbot.infrastructure.http.app import create_app
from rag_chatbot.infrastructure.http.logging import configure_logging
from rag_chatbot.infrastructure.postgres.conversations import PostgresSessionStore
from rag_chatbot.infrastructure.postgres.migrations import migrate
from rag_chatbot.infrastructure.postgres.pool import open_pool
from rag_chatbot.infrastructure.sources import SourceReport
from rag_chatbot.infrastructure.sources.crawler import Crawler
from rag_chatbot.infrastructure.sources.filtered import FilteredSource, ReportingSource
from rag_chatbot.infrastructure.sources.folder import FolderSource
from rag_chatbot.infrastructure.sources.http_fetcher import HttpFetcher
from rag_chatbot.infrastructure.system import RandomIds, SystemClock

USER_AGENT = "rag-chatbot-fastapi (+operator crawl; honours robots.txt)"
# a run that skipped pages for these reasons has not seen the whole site: it must not delete what it did not see
INCOMPLETE = frozenset({"limit", "depth", "total_bytes", "too_large", "truncated", "status", "error", "redirect"})
FAKE_PRICES = parse_price_table(
    {
        "currency": "USD",
        "models": {
            "fake-chat-1": {"input_per_mtok": 1, "output_per_mtok": 3},
            "fake-embedding-1": {"input_per_mtok": 0.1},
        },
    }
)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    temporary.replace(path)


# --- serve and migrate ----------------------------------------------------------------------------------------------


def serve(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    settings.require("database_url", "tenants_file", "price_table_file", "provider")
    configure_logging(settings.log_level, redaction_kinds=settings.chat.redaction_kinds)
    asyncio.run(_serve(settings))
    return 0


async def _serve(settings: Settings) -> None:
    assert settings.database_url is not None
    pool = await open_pool(
        settings.database_url,
        min_size=1,
        max_size=settings.db_pool_max,
        statement_timeout_ms=settings.db_statement_timeout_ms,
    )
    try:
        deps = await wiring.build_app_deps(settings, pool)
        config = uvicorn.Config(
            create_app(deps),
            host=settings.host,
            port=settings.port,
            log_config=None,  # our JSON logging, configured above
            access_log=False,  # the request log line replaces it and carries no path or address
            proxy_headers=False,  # X-Forwarded-For is read by the application, only for the proxies it is told to trust
            server_header=False,
            lifespan="off",
            timeout_keep_alive=5,
        )
        await uvicorn.Server(config).serve()
    finally:
        await pool.close()


def migrate_command(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    dsn = settings.migrate_database_url or settings.database_url
    if not dsn:
        raise ConfigurationError("missing configuration: RAG_MIGRATE_DATABASE_URL (or RAG_DATABASE_URL)")
    applied = migrate(dsn, settings.migrations_dir)
    out.write("applied: " + ", ".join(applied) + "\n" if applied else "the database is up to date\n")
    return 0


# --- ingestion ------------------------------------------------------------------------------------------------------


def build_source(args: Namespace) -> FilteredSource:
    """The content source the arguments describe, with the --exclude patterns applied."""
    if bool(args.folder) == bool(args.crawl):
        raise ConfigurationError("give exactly one of --folder and --crawl")
    inner: ReportingSource
    if args.folder:
        inner = FolderSource(Path(args.folder), base_url=args.base_url or "")
    else:
        hosts = {h for raw in args.allow_host or [] if (h := normalise_host(raw)) is not None}
        if not hosts:
            raise ConfigurationError("a crawl needs --allow-host for every host it may visit")
        inner = Crawler(
            HttpFetcher(user_agent=USER_AGENT, allow_private_addresses=args.allow_private_addresses),
            start_urls=args.crawl,
            policy=HostPolicy(
                frozenset(hosts), allow_http=args.allow_http, allowed_ports=frozenset(args.allow_port or ())
            ),
            user_agent=USER_AGENT,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            delay_s=args.delay,
        )
    return FilteredSource(inner, args.exclude or [])


async def _ingest(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    tenants = wiring.load_tenants(settings)
    if tenants.get(args.tenant) is None:
        raise ConfigurationError(f"the tenant {args.tenant!r} is not in the tenants file")
    prices = wiring.load_prices(settings)
    wiring.check_prices(prices, settings)
    providers = wiring.build_providers(settings)
    source = build_source(args)  # validates the arguments before anything is opened
    assert settings.database_url is not None
    pool = await open_pool(
        settings.database_url, min_size=1, max_size=4, statement_timeout_ms=settings.db_statement_timeout_ms
    )
    try:
        service = wiring.build_ingest_service(settings, providers, wiring.postgres_stores(pool, settings), prices)
        report = await service.run(args.tenant, source)
        _print_ingest(out, report, source.report, source.excluded, prices)
        if args.prune:
            incomplete = sorted(set(source.report.skipped) & INCOMPLETE) + (["errors"] if source.report.errors else [])
            if incomplete:
                err.write(f"not pruned: the run did not see the whole site ({', '.join(incomplete)})\n")
                return 1
            if not report.document_ids:
                err.write("not pruned: the run indexed nothing\n")
                return 1
            out.write(f"pruned {await service.prune(args.tenant, report)} document(s) that are no longer on the site\n")
        return 0
    finally:
        await pool.close()


def _print_ingest(
    out: TextIO, report: IngestReport, source_report: SourceReport, excluded: int, prices: PriceTable
) -> None:
    out.write(f"pages read      {report.pages_read}\n")
    out.write(f"documents       {report.documents_indexed} indexed\n")
    skipped = dict(report.documents_skipped)
    for reason, count in source_report.skipped.items():
        skipped[f"source:{reason}"] = count
    if excluded:
        skipped["excluded"] = excluded
    if skipped:
        out.write("skipped         " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())) + "\n")
    out.write(
        f"chunks          {report.chunks_total} ({report.chunks_embedded} embedded, {report.chunks_reused} reused unchanged)\n"
    )
    if report.quarantined_total:
        documents = {q.document_id for q in report.quarantined}
        out.write(
            f"quarantined     {report.quarantined_total} instruction-like sentence(s), kept out of the index (first {len(documents)} document(s) below)\n"
        )
        for item in report.quarantined[:20]:
            out.write(f"  quarantined   {item.document_id}: {json.dumps(item.sentence[:120], ensure_ascii=True)}\n")
    if report.chunks_dropped:
        out.write(f"chunks dropped  {report.chunks_dropped} (over the per-document cap)\n")
    out.write(
        f"embedding       {report.embedding_usage.input_tokens} tokens, {format_micros(report.cost_micros, prices.currency)}\n"
    )
    for message in source_report.errors:
        out.write(f"source error    {message}\n")


def ingest(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    settings.require("database_url", "tenants_file", "price_table_file", "provider")
    configure_logging(settings.log_level, redaction_kinds=settings.chat.redaction_kinds)
    return asyncio.run(_ingest(settings, args, out, err))


# --- evaluation -----------------------------------------------------------------------------------------------------


def _eval_prices(settings: Settings) -> PriceTable:
    if settings.price_table_file is not None:
        return wiring.load_prices(settings)
    if settings.provider == "http":
        settings.require("price_table_file")
    return FAKE_PRICES


async def _evaluate(
    settings: Settings, args: Namespace, cases: list[EvalCase], out: TextIO
) -> tuple[int, dict[str, object]]:
    prices = _eval_prices(settings)
    wiring.check_prices(prices, settings)
    providers = wiring.build_providers(settings)
    stores = wiring.memory_stores(settings)
    origin = normalise_origin(args.base_url)
    if origin is None:
        raise ConfigurationError("--base-url must be an origin or a URL on one, like https://www.example.com")
    tenant = Tenant("eval", args.site_name, frozenset({origin}))
    await wiring.build_ingest_service(settings, providers, stores, prices).run(
        tenant.tenant_id, FolderSource(Path(args.site), base_url=args.base_url.rstrip("/"))
    )
    service = wiring.build_chat_service(settings, providers, stores, prices)
    started = time.monotonic()
    report = await EvalRunner(service, tenant, RandomIds()).run(cases)
    quality = [r for r in report.results if r.category == "quality"]
    safety = [r for r in report.results if r.category == "safety"]
    for result in report.results:
        if not result.passed:
            out.write(f"FAIL {result.category:<7} {result.case_id}\n")
            for failure in result.failures:
                out.write(f"       {failure}\n")
            out.write(f"       answer: {json.dumps(report.answers[result.case_id], ensure_ascii=True)[:300]}\n")
    out.write(f"dataset   {args.dataset}  {report.dataset_hash[:19]}…  ({len(cases)} cases)\n")
    out.write(f"provider  {settings.provider}  chat={settings.chat_model}  embedding={settings.embedding_model}\n")
    out.write(f"quality   {sum(r.passed for r in quality)}/{len(quality)} passed\n")
    out.write(f"safety    {sum(r.passed for r in safety)}/{len(safety)} passed\n")
    out.write(f"tokens    in {report.input_tokens}  out {report.output_tokens}\n")
    out.write(f"cost      {format_micros(report.cost_micros, prices.currency)}\n")
    out.write(
        f"latency   p50 {report.percentile(0.5)} ms  p95 {report.percentile(0.95)} ms  (run took {time.monotonic() - started:.1f} s)\n"
    )
    code = exit_code(report.results, min_quality=args.min_quality)
    out.write(f"result    exit {code}\n")
    payload: dict[str, object] = {
        "dataset": args.dataset,
        "dataset_hash": report.dataset_hash,
        "provider": settings.provider,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "confidence_threshold": settings.chat.confidence_threshold,
        "exit_code": code,
        "cost_micros": report.cost_micros,
        "currency": prices.currency,
        "cases": [
            {
                "id": r.case_id,
                "category": r.category,
                "passed": r.passed,
                "failures": list(r.failures),
                "answer": report.answers[r.case_id],
            }
            for r in report.results
        ],
    }
    return code, payload


def evaluate(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    # the answers the checks blocked are in the report; the log lines of each one would only bury the summary
    configure_logging("ERROR", redaction_kinds=settings.chat.redaction_kinds, stream=err)
    try:
        text = Path(args.dataset).read_text(encoding="utf-8")
        cases = parse_dataset([(f"line {i + 1}", line) for i, line in enumerate(text.splitlines())])
        code, payload = asyncio.run(_evaluate(settings, args, cases, out))
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return code
    except DatasetError as error:
        err.write(f"{args.dataset}: {len(error.problems)} problem(s)\n")
        for problem in error.problems[:50]:
            err.write(f"  {problem}\n")
        return 2
    except OSError as error:
        raise ConfigurationError(f"cannot read the dataset or the site: {error.__class__.__name__}") from None
    except ConfigurationError:
        raise  # a mistake in the settings or the arguments: the caller reports it and exits with 2
    except ChatbotError as error:
        err.write(f"evaluation could not run: {error.code}: {error.detail or error.public_message}\n")
        return 3


# --- tenants, tokens, retention ---------------------------------------------------------------------------------------


def new_api_key() -> tuple[str, str]:
    key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_api_key(key)


def new_token_secret() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


def init_tenant(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    path = Path(args.out)
    existing: list[object] = []
    if path.exists():
        if not args.add:
            raise ConfigurationError(f"{path} exists; use --add to add this tenant to it")
        loaded = wiring.read_json_file(path, "the tenants file")
        directory = parse_tenants(loaded)
        if directory.get(args.id) is not None:
            raise ConfigurationError(f"the tenant {args.id!r} is already in {path}")
        assert isinstance(loaded, dict)
        existing = list(loaded["tenants"])
    key, digest = new_api_key()
    entry: dict[str, object] = {
        "id": args.id,
        "name": args.name,
        "allowed_origins": args.origin or [],
        "api_key_sha256": [digest],
        "token_secrets": [new_token_secret()],
    }
    parse_tenants({"tenants": [entry]})  # validate before anything is written
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps({"tenants": [*existing, entry]}, indent=2) + "\n")
    out.write(key + "\n")
    err.write(
        f"wrote {path} (mode 0600). The API key above is shown once and is not stored: only its hash is in the file. "
        "Keep it on your server; never put it in a web page.\n"
    )
    return 0


def new_key(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    key, digest = new_api_key()
    out.write(key + "\n")
    err.write(f"add this to the tenant's api_key_sha256 list: {digest}\n")
    return 0


def embed_token(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    tenants = wiring.load_tenants(settings)
    tenant = tenants.get(args.tenant)
    if tenant is None or not tenant.token_secrets:
        raise ConfigurationError(f"the tenant {args.tenant!r} is not in the tenants file or has no token secret")
    now = int(SystemClock().now().timestamp())
    claims = TokenClaims(tenant.tenant_id, now + args.ttl, args.subject, args.origin)
    validate_claims_for_minting(claims, now)
    out.write(mint_token(tenant.token_secrets[0], claims) + "\n")
    return 0


def purge(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    settings.require("database_url", "tenants_file")
    return asyncio.run(_purge(settings, args, out))


async def _purge(settings: Settings, args: Namespace, out: TextIO) -> int:
    assert settings.database_url is not None
    tenants: TenantDirectory = wiring.load_tenants(settings)
    chosen = [args.tenant] if args.tenant else sorted(tenants.tenants)
    for tenant_id in chosen:
        if tenants.get(tenant_id) is None:
            raise ConfigurationError(f"the tenant {tenant_id!r} is not in the tenants file")
    pool = await open_pool(settings.database_url, min_size=1, max_size=2)
    try:
        store = PostgresSessionStore(pool)
        cutoff = SystemClock().now() - timedelta(days=args.older_than_days)
        for tenant_id in chosen:
            removed = await store.purge_sessions(tenant_id, cutoff)
            out.write(
                f"{tenant_id}: removed {removed} conversation(s) last active before {cutoff.date().isoformat()}\n"
            )
    finally:
        await pool.close()
    return 0


def check_config(settings: Settings, args: Namespace, out: TextIO, err: TextIO) -> int:
    """Validates what is configured and says what it found, without connecting to anything or printing a secret."""
    out.write(f"provider        {settings.provider or '(not set)'}\n")
    if settings.provider == "http":
        settings.require("provider")
        out.write(f"api             {settings.api_base_url} (key set)\n")
    out.write(
        f"models          chat={settings.chat_model or '-'} embedding={settings.embedding_model or '-'} ({settings.embedding_dimensions} dimensions)\n"
    )
    out.write(f"threshold       {settings.chat.confidence_threshold}\n")
    out.write(f"redaction       {', '.join(sorted(settings.chat.redaction_kinds))}\n")
    out.write(f"database        {'set' if settings.database_url else '(not set)'}\n")
    if settings.tenants_file:
        tenants = wiring.load_tenants(settings)
        for tenant in tenants.tenants.values():
            out.write(
                f"tenant          {tenant.tenant_id}: {len(tenant.allowed_origins)} origin(s), {len(tenant.api_key_sha256)} api key(s), "
                f"{len(tenant.token_secrets)} token secret(s)\n"
            )
    if settings.price_table_file:
        prices = wiring.load_prices(settings)
        if settings.chat_model and settings.embedding_model:
            wiring.check_prices(prices, settings)
        out.write(f"price table     {len(prices.models)} model(s) in {prices.currency}\n")
    out.write("configuration ok\n")
    return 0
