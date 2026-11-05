from __future__ import annotations

import io
import json
import stat
from pathlib import Path

import pytest

from rag_chatbot.composition.main import main
from rag_chatbot.domain.auth import authenticate, hash_api_key
from rag_chatbot.domain.tenants import parse_tenants

ROOT = Path(__file__).resolve().parents[2]


def run(*argv: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), env or {}, out, err)
    return code, out.getvalue(), err.getvalue()


# --- the command line itself ----------------------------------------------------------------------------------------


def test_version_help_and_unknown_commands() -> None:
    assert run("--version")[0] == 0
    assert run("--help")[0] == 0
    assert run("nonsense")[0] == 2
    assert run()[0] == 2
    assert run("ingest")[0] == 2  # --tenant is required


# --- tenants, keys and tokens -----------------------------------------------------------------------------------------


def test_init_tenant_writes_a_valid_private_file_and_shows_the_key_once(tmp_path: Path) -> None:
    path = tmp_path / "tenants.json"
    code, out, err = run(
        "init-tenant",
        "--id",
        "maple",
        "--name",
        "Maple Bakery",
        "--origin",
        "https://www.maple.example",
        "--out",
        str(path),
    )
    assert code == 0
    key = out.strip()
    assert key.startswith("ragk_") and "shown once" in err and key not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    directory = parse_tenants(json.loads(path.read_text()))
    tenant = directory.get("maple")
    assert tenant is not None and tenant.api_key_sha256 == {hash_api_key(key)} and len(tenant.token_secrets) == 1
    assert authenticate(directory, key, None, 1).tenant.tenant_id == "maple"


def test_init_tenant_refuses_to_overwrite_and_can_add(tmp_path: Path) -> None:
    path = tmp_path / "tenants.json"
    assert run("init-tenant", "--id", "a", "--name", "A", "--out", str(path))[0] == 0
    code, _, err = run("init-tenant", "--id", "b", "--name", "B", "--out", str(path))
    assert code == 2 and "--add" in err
    assert run("init-tenant", "--id", "b", "--name", "B", "--out", str(path), "--add")[0] == 0
    assert set(parse_tenants(json.loads(path.read_text())).tenants) == {"a", "b"}
    code, _, err = run("init-tenant", "--id", "b", "--name", "B", "--out", str(path), "--add")
    assert code == 2 and "already" in err


@pytest.mark.parametrize(
    "args",
    [
        ["--id", "Bad Id", "--name", "x"],
        ["--id", "a", "--name", "x", "--origin", "http://insecure.example"],
        ["--id", "a", "--name", ""],
    ],
)
def test_init_tenant_validates_before_writing(tmp_path: Path, args: list[str]) -> None:
    path = tmp_path / "t.json"
    code, out, _ = run("init-tenant", *args, "--out", str(path))
    assert code == 2 and out == "" and not path.exists()


def test_new_api_key_prints_a_key_and_the_hash_to_store() -> None:
    code, out, err = run("new-api-key")
    assert code == 0 and out.strip().startswith("ragk_") and hash_api_key(out.strip()) in err


def test_embed_token_mints_a_token_that_authenticates(tmp_path: Path) -> None:
    path = tmp_path / "tenants.json"
    run("init-tenant", "--id", "maple", "--name", "M", "--origin", "https://www.maple.example", "--out", str(path))
    env = {"RAG_TENANTS_FILE": str(path)}
    code, out, _ = run(
        "embed-token", "--tenant", "maple", "--ttl", "120", "--origin", "https://www.maple.example", env=env
    )
    assert code == 0
    import time

    directory = parse_tenants(json.loads(path.read_text()))
    principal = authenticate(directory, out.strip(), "https://www.maple.example", int(time.time()))
    assert principal.method == "embed_token" and principal.tenant.tenant_id == "maple"
    assert run("embed-token", "--tenant", "nobody", env=env)[0] == 2
    assert run("embed-token", "--tenant", "maple", "--ttl", "0", env=env)[0] == 2
    assert run("embed-token", "--tenant", "maple", "--subject", "not-a-uuid", env=env)[0] == 2
    assert run("embed-token", "--tenant", "maple")[0] == 2  # no tenants file configured


# --- check-config -----------------------------------------------------------------------------------------------------


def test_check_config_says_what_it_found_and_prints_no_secret(tmp_path: Path) -> None:
    tenants = tmp_path / "tenants.json"
    run("init-tenant", "--id", "maple", "--name", "M", "--out", str(tenants))
    env = {
        "RAG_PROVIDER": "http",
        "RAG_API_BASE_URL": "https://api.example.com/v1",
        "RAG_API_KEY": "super-secret-key-1234",
        "RAG_CHAT_MODEL": "your-chat-model",
        "RAG_EMBEDDING_MODEL": "your-embedding-model",
        "RAG_DATABASE_URL": "postgresql://u@h/db",
        "RAG_TENANTS_FILE": str(tenants),
        "RAG_PRICE_TABLE_FILE": str(ROOT / "examples" / "prices.example.json"),
    }
    code, out, err = run("check-config", env=env)
    assert (
        code == 0
        and "configuration ok" in out
        and "tenant          maple: 0 origin(s), 1 api key(s), 1 token secret(s)" in out
    )
    assert "super-secret" not in out + err and "postgresql://" not in out + err and "(key set)" in out


def test_check_config_fails_when_a_configured_model_has_no_price(tmp_path: Path) -> None:
    env = {
        "RAG_PROVIDER": "fake",
        "RAG_CHAT_MODEL": "unpriced-model",
        "RAG_PRICE_TABLE_FILE": str(ROOT / "examples" / "prices.example.json"),
    }
    code, _, err = run("check-config", env=env)
    assert code == 2 and "unpriced-model" in err


@pytest.mark.parametrize(
    ("env", "needle"),
    [
        ({"RAG_PROVIDER": "bogus"}, "RAG_PROVIDER"),
        ({"RAG_TENANTS_FILE": "/nonexistent/tenants.json"}, "cannot be read"),
        ({"RAG_PROVIDER": "http"}, "RAG_API_BASE_URL"),
    ],
)
def test_check_config_reports_configuration_problems_as_exit_2(env: dict[str, str], needle: str) -> None:
    code, _, err = run("check-config", env=env)
    assert code == 2 and needle in err


def test_the_shipped_examples_are_valid_configuration() -> None:
    from rag_chatbot.domain.pricing import parse_price_table

    parse_tenants(json.loads((ROOT / "examples" / "tenants.example.json").read_text()))
    parse_price_table(json.loads((ROOT / "examples" / "prices.example.json").read_text()))
