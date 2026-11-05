"""The command line: `rag-chatbot COMMAND`, see `rag-chatbot --help`."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TextIO

from rag_chatbot import __version__
from rag_chatbot.composition import commands
from rag_chatbot.composition.config import Settings, parse_settings
from rag_chatbot.domain.errors import ChatbotError, ConfigurationError, InvalidRequestError

Handler = Callable[[Settings, argparse.Namespace, TextIO, TextIO], int]

EPILOG = """\
Configuration is read from RAG_* environment variables (docs/configuration.md). Exit status: 0 success, 1 the checked
thing failed, 2 usage or configuration error, 3 the service or a provider failed, 4 a safety case failed.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-chatbot",
        description="A website chatbot that answers only from the site's own content.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"rag-chatbot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("serve", help="run the web service").set_defaults(handler=commands.serve)
    sub.add_parser("migrate", help="apply the SQL migrations").set_defaults(handler=commands.migrate_command)
    sub.add_parser("check-config", help="validate the configuration and say what it found").set_defaults(
        handler=commands.check_config
    )

    ingest = sub.add_parser("ingest", help="read a folder or crawl a site into the knowledge base of a tenant")
    ingest.add_argument("--tenant", required=True)
    ingest.add_argument("--folder", help="a folder of .html, .md and .txt files")
    ingest.add_argument(
        "--base-url", help="with --folder: the public address the folder is served from (for citations)"
    )
    ingest.add_argument("--crawl", action="append", metavar="URL", help="a start URL; repeat for several")
    ingest.add_argument(
        "--allow-host", action="append", metavar="HOST", help="a host the crawl may visit, exactly; repeat"
    )
    ingest.add_argument(
        "--allow-port",
        action="append",
        type=int,
        metavar="PORT",
        help="a port other than 80/443 the crawl may use; repeat",
    )
    ingest.add_argument("--allow-http", action="store_true", help="allow plain http (the default is https only)")
    ingest.add_argument(
        "--allow-private-addresses", action="store_true", help="allow hosts on private networks (a staging site)"
    )
    ingest.add_argument("--max-pages", type=int, default=200)
    ingest.add_argument("--max-depth", type=int, default=3)
    ingest.add_argument("--delay", type=float, default=1.0, help="seconds between requests to one host")
    ingest.add_argument(
        "--exclude", action="append", metavar="GLOB", help="leave out pages whose location matches; repeat"
    )
    ingest.add_argument(
        "--prune", action="store_true", help="afterwards remove documents that are no longer in the source"
    )
    ingest.set_defaults(handler=commands.ingest)

    evaluate = sub.add_parser("eval", help="run the evaluation set through the whole chat flow (in memory)")
    evaluate.add_argument("--dataset", default="eval/datasets/site.jsonl")
    evaluate.add_argument("--site", default="examples/site", help="the folder of the sample site")
    evaluate.add_argument("--base-url", default="https://www.maple-street-bakery.example")
    evaluate.add_argument("--site-name", default="Maple Street Bakery")
    evaluate.add_argument("--min-quality", type=float, default=1.0, help="the share of quality cases that must pass")
    evaluate.add_argument("--report", default="out/eval-report.json")
    evaluate.set_defaults(handler=commands.evaluate)

    tenant = sub.add_parser("init-tenant", help="create a tenants file (or add a tenant) with a fresh key and secret")
    tenant.add_argument("--id", required=True)
    tenant.add_argument("--name", required=True)
    tenant.add_argument("--origin", action="append", metavar="ORIGIN", help="an allowed web origin; repeat")
    tenant.add_argument("--out", default="tenants.json")
    tenant.add_argument("--add", action="store_true", help="add to an existing file")
    tenant.set_defaults(handler=commands.init_tenant)

    sub.add_parser("new-api-key", help="print a new API key and the hash to put in the tenants file").set_defaults(
        handler=commands.new_key
    )

    token = sub.add_parser("embed-token", help="mint a signed token for the website widget")
    token.add_argument("--tenant", required=True)
    token.add_argument("--ttl", type=int, default=900, help="lifetime in seconds")
    token.add_argument("--subject", help="the end user's id, a lower-case uuid (for row-level security)")
    token.add_argument("--origin", help="bind the token to one origin")
    token.set_defaults(handler=commands.embed_token)

    purge = sub.add_parser("purge", help="delete conversations that have been inactive for a number of days")
    purge.add_argument("--older-than-days", type=int, required=True)
    purge.add_argument("--tenant", help="one tenant (the default is all of them)")
    purge.set_defaults(handler=commands.purge)
    return parser


def main(argv: Sequence[str], env: Mapping[str, str], out: TextIO, err: TextIO) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exit_:
        return 0 if exit_.code in (0, None) else 2
    handler: Handler = args.handler
    try:
        settings = parse_settings(env)
        return handler(settings, args, out, err)
    except (ConfigurationError, InvalidRequestError) as error:
        err.write(f"error: {error.detail or error.public_message}\n")
        return 2
    except ChatbotError as error:
        err.write(f"error: {error.code}: {error.detail or error.public_message}\n")
        return 3


def run() -> None:
    sys.exit(main(sys.argv[1:], os.environ, sys.stdout, sys.stderr))
