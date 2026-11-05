"""The architecture rules, as pure functions over source text (AST based), so that each rule can be tested with a snippet.

Layers, from the inside out: domain <- application <- infrastructure <- composition. A layer imports only from itself
and from the layers inside it. Beyond that:

* domain is pure: standard-library modules that compute, nothing else: no files, no network, no clock, no randomness, no
  environment, no logging, no third-party package.
* application orchestrates over ports: the same modules plus `logging` and `datetime` (as a type: it never asks for
  the time, it is given a clock); no input or output of its own.
* infrastructure implements the ports. Third-party packages are confined to the modules that exist for them (the web
  framework to infrastructure/http, the database driver to infrastructure/postgres). Raw network access lives in two
  modules.
* composition wires everything and is the only place that reads the environment and starts the server.
* nowhere: dynamic code execution, process creation, pickle-like loaders, wildcard imports, `print`.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

PACKAGE = "rag_chatbot"
LAYERS = ("domain", "application", "infrastructure", "composition")

ALLOWED_INTERNAL: dict[str, tuple[str, ...]] = {
    "domain": ("domain",),
    "application": ("domain", "application"),
    "infrastructure": ("domain", "application", "infrastructure"),
    "composition": ("domain", "application", "infrastructure", "composition"),
    "entry": ("composition",),
}

PURE_STDLIB = frozenset(
    {
        "__future__", "abc", "base64", "binascii", "collections", "collections.abc", "dataclasses", "decimal", "enum",
        "fractions", "functools", "hashlib", "hmac", "html", "ipaddress", "itertools", "json", "math", "operator",
        "posixpath", "re", "string", "typing", "unicodedata", "urllib.parse",
    }
)  # fmt: skip
APPLICATION_EXTRA = frozenset({"datetime", "logging"})

# third-party packages, and the only modules that may import them
THIRD_PARTY: dict[str, tuple[str, ...]] = {
    "fastapi": ("infrastructure/http/",),
    "starlette": ("infrastructure/http/",),
    "pydantic": ("infrastructure/http/",),
    "psycopg": ("infrastructure/postgres/",),
    "psycopg_pool": ("infrastructure/postgres/",),
    "uvicorn": ("composition/commands.py",),
}

NETWORK_MODULES = frozenset(
    {"socket", "ssl", "http", "http.client", "http.server", "urllib.request", "ftplib", "smtplib", "xmlrpc"}
)
NETWORK_FILES = frozenset({"infrastructure/sources/http_fetcher.py", "infrastructure/providers/http.py"})
PROCESS_MODULES = frozenset(
    {"subprocess", "multiprocessing", "pty", "selectors", "select", "threading", "concurrent.futures"}
)
UNSAFE_MODULES = frozenset(
    {
        "pickle", "marshal", "shelve", "ctypes", "importlib", "code", "codeop", "runpy", "imp", "builtins", "posix", "nt",
        "pkgutil", "timeit", "cProfile", "profile", "pdb", "trace", "gc", "inspect",
    }
)  # fmt: skip
ASYNC_MODULES = frozenset({"asyncio"})  # allowed from infrastructure outwards: a pure layer never schedules anything

PROCESS_ATTRS = frozenset({"create_subprocess_exec", "create_subprocess_shell"})
NETWORK_ATTRS = frozenset(
    {"open_connection", "start_server", "open_unix_connection", "start_unix_server", "create_server"}
)
REFLECTION_CALLS = frozenset({"vars", "globals", "locals"})
OPERATOR_ESCAPES = frozenset({"attrgetter", "methodcaller"})
FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
FORBIDDEN_ATTRS = frozenset(
    {
        "__builtins__", "__globals__", "__code__", "__subclasses__", "__mro__", "__bases__", "__loader__", "__import__",
        "__dict__", "f_globals", "f_locals", "f_builtins", "gi_frame", "cr_frame", "modules",
    }
)  # fmt: skip
REFLECTION = frozenset({"getattr", "setattr", "delattr"})
ENV_NAMES = frozenset({"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"})
SHELL_NAMES = frozenset(
    {"system", "popen", "startfile", "execv", "execve", "execvp", "execl", "spawnv", "fork", "posix_spawn"}
)
CLOCK_ATTRS = frozenset({"now", "today", "utcnow"})  # reading the time: through the Clock port only
IO_BUILTINS_PURE = frozenset({"open", "input", "breakpoint"})


@dataclass(frozen=True)
class Import:
    module: str
    internal: bool
    line: int


def layer_of(relpath: str) -> str:
    first = PurePosixPath(relpath).parts[0]
    return first if first in LAYERS else "entry"


def _package_of(relpath: str) -> list[str]:
    return [PACKAGE, *PurePosixPath(relpath).parts[:-1]]


def imports_of(tree: ast.AST, relpath: str) -> list[Import]:
    found: list[Import] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(Import(alias.name, alias.name.split(".")[0] == PACKAGE, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _package_of(relpath)
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                module = ".".join([*base, *(node.module.split(".") if node.module else [])])
                found.append(Import(module, True, node.lineno))
                continue
            module = node.module or ""
            found.append(Import(module, module.split(".")[0] == PACKAGE, node.lineno))
            if module.split(".")[0] == PACKAGE:  # `from pkg import sub` may import a submodule
                found.extend(Import(f"{module}.{alias.name}", True, node.lineno) for alias in node.names)
    return found


def _internal_layer(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == PACKAGE and parts[1] in LAYERS:
        return parts[1]
    return "entry" if parts and parts[0] == PACKAGE else None


def _within(module: str, allowed: frozenset[str]) -> bool:
    return any(module == a or module.startswith(a + ".") for a in allowed)


def _third_party_allowed(top: str, relpath: str) -> bool:
    return any(
        relpath.startswith(prefix) if prefix.endswith("/") else relpath == prefix for prefix in THIRD_PARTY.get(top, ())
    )


def _is_log_record_vars(call: ast.Call) -> bool:
    """`vars(record)` and `vars(logging.makeLogRecord(...))`: the one use the logging formatter needs."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "vars" and len(call.args) == 1):
        return False
    arg = call.args[0]
    if isinstance(arg, ast.Name):
        return arg.id == "record"
    return isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "makeLogRecord"


def _is_clock_class(node: ast.expr) -> bool:
    """`datetime.now()`, `date.today()`, `datetime.datetime.now()`: the standard library reading the clock. A call on a
    value that was passed in (`clock.now()`, `self._clock.now()`) is the Clock port doing its job."""
    if isinstance(node, ast.Name):
        return node.id in {"datetime", "date"}
    return isinstance(node, ast.Attribute) and node.attr in {"datetime", "date"}


def violations(relpath: str, source: str) -> list[str]:
    layer = layer_of(relpath)
    tree = ast.parse(source)
    found: list[str] = []
    for imp in imports_of(tree, relpath):
        top = imp.module.split(".")[0]
        where = f"{relpath}:{imp.line}"
        if imp.internal:
            target = _internal_layer(imp.module)
            if target is not None and target != "entry" and target not in ALLOWED_INTERNAL[layer]:
                found.append(f"{where}: {layer} must not import {target} ({imp.module})")
            continue
        if top not in sys.stdlib_module_names:
            if top not in THIRD_PARTY:
                found.append(f"{where}: {imp.module} is not in the standard library and is not a declared dependency")
            elif not _third_party_allowed(top, relpath):
                found.append(f"{where}: {top} may only be imported by {', '.join(THIRD_PARTY[top])}")
            continue
        if layer == "domain" and not _within(imp.module, PURE_STDLIB):
            found.append(f"{where}: domain must be pure; {imp.module} is not on the allow-list")
        if layer == "application" and not _within(imp.module, PURE_STDLIB | APPLICATION_EXTRA):
            found.append(
                f"{where}: application reaches the outside world only through ports; {imp.module} is not on the allow-list"
            )
        if _within(imp.module, NETWORK_MODULES) and relpath not in NETWORK_FILES:
            found.append(f"{where}: {imp.module} may only be imported by {', '.join(sorted(NETWORK_FILES))}")
        if _within(imp.module, PROCESS_MODULES):
            found.append(f"{where}: {imp.module} (processes and threads) is not used by the service")
        if _within(imp.module, ASYNC_MODULES) and layer in ("domain", "application"):
            found.append(f"{where}: {layer} must not schedule anything ({imp.module})")
        if _within(imp.module, UNSAFE_MODULES):
            found.append(f"{where}: {imp.module} can execute or load arbitrary code")
    found.extend(_syntax_violations(relpath, layer, tree))
    return found


def _syntax_violations(relpath: str, layer: str, tree: ast.AST) -> list[str]:
    found: list[str] = []
    os_names = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(alias.asname or alias.name for alias in node.names if alias.name == "os")

    def os_name(name: str, where: str) -> None:
        if name in SHELL_NAMES:
            found.append(f"{where}: os.{name} runs or replaces a process")
        elif name in ENV_NAMES and layer != "composition":
            found.append(f"{where}: only the composition layer reads the environment (os.{name})")

    calls = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        where = f"{relpath}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names):
                found.append(f"{where}: wildcard import")
            if node.level == 0 and node.module == "os":
                for alias in node.names:
                    os_name(alias.name, where)
            if node.level == 0 and node.module == "sys" and any(a.name == "modules" for a in node.names):
                found.append(f"{where}: from sys import modules reaches every other module")
            if node.level == 0 and node.module == "operator":
                found.extend(
                    f"{where}: operator.{a.name} looks attributes up by name"
                    for a in node.names
                    if a.name in OPERATOR_ESCAPES
                )
            if node.level == 0 and node.module == "asyncio":
                found.extend(
                    f"{where}: asyncio.{a.name} creates a process" for a in node.names if a.name in PROCESS_ATTRS
                )
                if relpath not in NETWORK_FILES:
                    found.extend(
                        f"{where}: asyncio.{a.name} is raw network access"
                        for a in node.names
                        if a.name in NETWORK_ATTRS
                    )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_CALLS:
                found.append(f"{where}: reference to {node.id} (model output and page text are never executed)")
            if node.id == "print":
                found.append(f"{where}: print() (use logging, or write to the stream a command was given)")
            if node.id in IO_BUILTINS_PURE and layer in ("domain", "application") and id(node) in calls:
                found.append(f"{where}: {node.id}() in the {layer} layer")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                found.append(f"{where}: forbidden attribute {node.attr}")
            if node.attr in PROCESS_ATTRS:
                found.append(f"{where}: {node.attr} creates a process")
            if node.attr in NETWORK_ATTRS and relpath not in NETWORK_FILES:
                found.append(
                    f"{where}: {node.attr} is raw network access, allowed only in {', '.join(sorted(NETWORK_FILES))}"
                )
            if node.attr in OPERATOR_ESCAPES:
                found.append(f"{where}: {node.attr} looks attributes up by name")
            if isinstance(node.value, ast.Name) and node.value.id in os_names:
                os_name(node.attr, where)
            if (
                node.attr in CLOCK_ATTRS
                and layer in ("domain", "application")
                and id(node) in calls
                and _is_clock_class(node.value)
            ):
                found.append(
                    f"{where}: {node.attr}() on the datetime classes in the {layer} layer: the time comes from the Clock port"
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in REFLECTION_CALLS and not _is_log_record_vars(node):
                found.append(f"{where}: {node.func.id}() exposes the namespace of a module or frame")
            if isinstance(node.func, ast.Name) and node.func.id in REFLECTION and len(node.args) >= 2:
                name = node.args[1]
                if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                    found.append(f"{where}: {node.func.id} with a computed name")
                elif name.value in FORBIDDEN_ATTRS or name.value in SHELL_NAMES:
                    found.append(f"{where}: forbidden attribute {name.value} through {node.func.id}")
            for keyword in node.keywords:
                if keyword.arg == "shell" and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                ):
                    found.append(f"{where}: shell= must be the literal False")
    return found
