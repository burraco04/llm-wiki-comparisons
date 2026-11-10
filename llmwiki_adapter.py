"""Trace the public llmw2 API without replacing its decisions or writing steps.

The adapter records effective configuration and the actual HTTP payloads.  A
separate record is written for every HTTP attempt, including library retries.
No request or response headers are persisted.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class AdapterCapabilityError(RuntimeError):
    """The installed library cannot supply the required benchmark evidence."""


_SECRET = re.compile(r"(^|_)(api_?key|authorization|password|secret|access_token|refresh_token|token|cookie)$", re.I)
_URL = re.compile(r"(?:git\+)?https?://[^\s\[\]\"<>]+")


def _secret_key(key: object) -> bool:
    return bool(_SECRET.search(str(key).replace("-", "_")))


def _clean_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        if not parts.scheme or not parts.netloc:
            return value
        query = [(k, "<redacted>" if _secret_key(k) or k.lower() == "key" else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, urlencode(query), parts.fragment))
    except ValueError:
        return "<invalid URL>"


def _sanitize(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {str(k): "<redacted>" if _secret_key(k) else _sanitize(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, secrets) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        out = _URL.sub(lambda match: _clean_url(match.group(0)), value)
        for secret in secrets:
            out = out.replace(secret, "<redacted>")
        return out
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump(mode="json"), secrets)
    return str(value)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class HTTPRecorder:
    """Durable HTTP-attempt records, scoped to an ingestion or query operation."""

    def __init__(self, path: Path, secrets: tuple[str, ...] = ()) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("xb")
        self._lock = threading.RLock()
        self._offsets: list[int] = []
        self._pending: dict[str, dict[str, Any]] = {}
        self._clients: list[Any] = []
        self._closed = False
        self._secrets = secrets
        self._outer: dict[str, str] = {}
        self._scope: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("llmw2_benchmark_operation", default=None)
        self._op: contextvars.ContextVar[str] = contextvars.ContextVar("llmw2_benchmark_op", default="")
        self.trace_io_seconds = 0.0

    def _append(self, event: dict[str, Any]) -> None:
        started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot write to a closed llmw2 trace")
            # Preserve lone surrogate escapes from a malformed model reply;
            # tracing must not fail before the library can validate that reply.
            data = (json.dumps(_sanitize(event, self._secrets), ensure_ascii=False) + "\n").encode("utf-8", errors="backslashreplace")
            self._offsets.append(self._stream.tell())
            self._stream.write(data)
            self._stream.flush()
            self.trace_io_seconds += time.perf_counter() - started

    def mark(self) -> int:
        with self._lock:
            return len(self._offsets)

    def since(self, mark: int) -> list[dict[str, Any]]:
        with self._lock:
            end = len(self._offsets)
            if not 0 <= mark <= end:
                raise ValueError("trace mark is outside the recorded event range")
            if mark == end:
                return []
            offset = self._offsets[mark]
        with self.path.open("rb") as stream:
            stream.seek(offset)
            return [json.loads(stream.readline()) for _ in range(end - mark)]

    @contextmanager
    def operation(self, phase: str, item_id: str) -> Iterator[None]:
        scope = {"phase": str(phase), "item_id": str(item_id)}
        with self._lock:
            previous = self._outer
            self._outer = scope
        token = self._scope.set(scope)
        try:
            yield
        finally:
            self._scope.reset(token)
            with self._lock:
                self._outer = previous

    def _request(self, role: str, request: Any) -> None:
        data = request.read()
        try:
            body = json.loads(data)
        except (ValueError, UnicodeError):
            body = data.decode("utf-8", errors="replace")
        identifier = uuid.uuid4().hex
        with self._lock:
            scope = self._scope.get() or dict(self._outer)
            self._pending[identifier] = {
                "event": "http_call", "call_id": identifier, "role": role,
                "op": self._op.get(), **scope,
                "started_at_utc": _utc(), "method": request.method,
                "url": _clean_url(str(request.url)), "request": body,
                "_started": time.perf_counter(),
            }
        request.extensions["llmw2_benchmark_call_id"] = identifier

    def _finish(self, request: Any, *, response: Any = None, error: Exception | None = None) -> None:
        identifier = request.extensions.get("llmw2_benchmark_call_id")
        with self._lock:
            event = self._pending.pop(identifier, None)
        if event is None:
            return
        started = event.pop("_started")
        event["status_code"] = response.status_code if response is not None else None
        event["response"] = None
        event["error"] = None
        if response is not None:
            try:
                raw = response.read()
            except Exception as exc:
                event["error"] = {"type": type(exc).__name__, "message": str(exc)}
                event["response_read_failed"] = True
                event["seconds"] = time.perf_counter() - started
                self._append(event)
                raise
            try:
                event["response"] = json.loads(raw)
            except (ValueError, UnicodeError):
                event["response"] = raw.decode("utf-8", errors="replace")
            if response.status_code >= 400:
                event["error"] = {"type": "HTTPStatusError", "message": f"HTTP {response.status_code}"}
        if error is not None:
            event["error"] = {"type": type(error).__name__, "message": str(error)}
        event["seconds"] = time.perf_counter() - started
        self._append(event)

    def client(self, role: str, timeout: float) -> Any:
        import httpx

        recorder = self

        class RecordedTransport(httpx.BaseTransport):
            def __init__(self) -> None:
                self.inner = httpx.HTTPTransport()

            def handle_request(self, request: Any) -> Any:
                try:
                    return self.inner.handle_request(request)
                except Exception as exc:
                    recorder._finish(request, error=exc)
                    raise

            def close(self) -> None:
                self.inner.close()

        client = httpx.Client(
            transport=RecordedTransport(), timeout=timeout,
            event_hooks={"request": [functools.partial(self._request, role)], "response": [lambda response: self._finish(response.request, response=response)]},
        )
        self._clients.append(client)
        return client

    def bind_op(self, model: Any, method_name: str) -> None:
        original = getattr(model, method_name, None)
        if not callable(original):
            raise AdapterCapabilityError(f"llmw2 model has no public {method_name} method; exact step-labelled tracing is unsupported")
        signature = inspect.signature(original)
        if "op" not in signature.parameters and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            raise AdapterCapabilityError(f"llmw2 {method_name} does not expose an op argument; exact step-labelled tracing is unsupported")

        @functools.wraps(original)
        def traced(*args: Any, **kwargs: Any) -> Any:
            arguments = signature.bind_partial(*args, **kwargs).arguments
            token = self._op.set(str(arguments.get("op", "")))
            try:
                return original(*args, **kwargs)
            finally:
                self._op.reset(token)

        setattr(model, method_name, traced)

    def close(self) -> None:
        if self._closed:
            return
        for client in self._clients:
            client.close()
        with self._lock:
            pending = list(self._pending.items())
        for identifier, _ in pending:
            request = type("PendingRequest", (), {"extensions": {"llmw2_benchmark_call_id": identifier}})()
            self._finish(request, error=RuntimeError("operation closed before a response was received"))
        with self._lock:
            self._stream.close()
            self._closed = True


def _library_provenance(package: Any) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    candidates = set(importlib.metadata.packages_distributions().get("llmw2", [])) | {"llmw2", "llmwiki2"}
    for name in sorted(candidates):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        entry: dict[str, Any] = {"name": distribution.metadata.get("Name", name), "version": distribution.version}
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            try:
                entry["direct_url"] = json.loads(direct_url)
            except ValueError:
                entry["direct_url_error"] = "invalid JSON metadata"
        versions[name] = entry
    package_file = Path(package.__file__).resolve()
    hashes = {
        path.relative_to(package_file.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_file.parent.rglob("*.py")) if "__pycache__" not in path.parts
    }
    return {"module_version": getattr(package, "__version__", None), "distributions": versions, "source_hashes": hashes}


def _resolved_provenance(wiki: Any, package: Any, provenance: dict[str, Any]) -> None:
    cfg = wiki.cfg
    provenance["configuration"] = cfg.model_dump(mode="json")
    provenance["thresholds"] = cfg.crow.model_dump(mode="json")
    provenance["effective_values"] = {
        name: getattr(cfg, name) for name in ("effective_read_chars", "effective_note_chars", "note_ceiling") if hasattr(cfg, name)
    }
    names = wiki.prompts.names()
    provenance["prompts"] = {name: wiki.prompts.load(name) for name in names}
    provenance["prompts_fingerprint"] = wiki.prompts.fingerprint()
    provenance["steps"] = {"name": wiki.steps.name, "hash": wiki.steps.hash, "decision_hash": wiki.librarian().hash}
    provenance["feature_flags"] = {
        name: getattr(cfg, name) for name in (
            "mode", "summarize", "max_depth", "max_steps", "max_links", "max_notes", "concurrency",
            "note_length", "revisions", "answer_conflicts", "relate_rare_values", "strict_new_folders",
        ) if hasattr(cfg, name)
    }
    constants, unavailable_constants = {}, {}
    for module_name, fields in {
        "config": ("PROMPT_ROOM", "NOTE_CAP", "NOTE_RATIO", "NOTE_MAX"),
        "models.llm": ("CHARS_PER_TOKEN", "REPLY_FACTOR", "MIN_REPLY_TOKENS", "REASONING_ROOM"),
        "models.classifier": ("MAX_OPTIONS",),
        "agents.state": ("STATE_VERSION",),
        "agents.steps_classic": ("SUMMARY_VERSION", "DECIDE_CHARS"),
        "agents.steps_crow": ("FALLBACK_MENU_CHARS",),
    }.items():
        try:
            module = importlib.import_module(f"{package.__name__}.{module_name}")
        except Exception as exc:
            for field in fields:
                unavailable_constants[f"{module_name}.{field}"] = {"type": type(exc).__name__, "message": str(exc)}
            continue
        for field in fields:
            try:
                constants[f"{module_name}.{field}"] = getattr(module, field)
            except Exception as exc:
                unavailable_constants[f"{module_name}.{field}"] = {"type": type(exc).__name__, "message": str(exc)}
    provenance["implementation_constants"] = constants
    provenance["implementation_constants_unavailable"] = unavailable_constants
    provenance["trace_capabilities"] = {
        "actual_http_payloads": True, "http_attempts_separate_from_usage_calls": True,
        "model_op": True, "full_resolved_configuration": True,
        "classifier_enabled": wiki.classifier is not None,
    }


def create_wiki(bundle: Path, settings: dict[str, str], events_path: Path) -> tuple[Any, HTTPRecorder, dict[str, Any]]:
    """Create one native library mode with isolated settings and complete tracing.

    The installed release chooses its normal defaults.  They are exported after
    resolution; this adapter does not substitute another release's defaults.
    """
    environment = dict(os.environ)
    environment.update(settings)
    mode = settings.get("OKF_MODE")
    if mode not in {"classic", "crow"}:
        raise ValueError("create_wiki requires explicit OKF_MODE=classic or OKF_MODE=crow")
    environment["OKF_USAGE_LOG"] = settings.get("OKF_USAGE_LOG", str(Path(events_path).parent / "usage.jsonl"))
    secrets = tuple(sorted({str(value) for key, value in environment.items() if _secret_key(key) and value and value not in {"EMPTY", "none", "None"}}, key=len, reverse=True))
    recorder = HTTPRecorder(Path(events_path), secrets)
    config_path = Path(events_path).parent / "configuration.json"
    provenance: dict[str, Any] = {"mode": mode, "created_at_utc": _utc(), "status": "initializing", "bundle": str(bundle)}
    wiki = None

    def persist() -> None:
        clean = _sanitize(provenance, secrets)
        config_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", errors="backslashreplace")

    try:
        package = importlib.import_module("llmw2")
        provenance["library"] = _library_provenance(package)
        Wiki, WikiConfig = package.Wiki, package.WikiConfig
        from llmw2.models.classifier import Classifier
        from llmw2.models.llm import LLM
        from llmw2.models.usage import UsageTracker

        for target, arguments in ((WikiConfig.from_env, ("env",)), (Wiki, ("llm", "classifier")), (LLM, ("client",)), (Classifier, ("client",))):
            parameters = inspect.signature(target).parameters
            if any(name not in parameters for name in arguments):
                raise AdapterCapabilityError(f"installed llmw2 lacks the public injected-client configuration API: {target.__qualname__}; use a compatible release and record it")
        cfg = WikiConfig.from_env(env=environment, bundle=Path(bundle), mode=mode)
        if not hasattr(cfg, "model_dump") or not hasattr(cfg, "crow"):
            raise AdapterCapabilityError("installed llmw2 does not expose its resolved configuration")
        missing = {f"tau_{name}" for name in ("route", "path", "ing", "cons", "link", "fold", "ret")} - set(cfg.crow.model_dump())
        if missing:
            raise AdapterCapabilityError(f"installed llmw2 does not expose required thresholds: {sorted(missing)}")
        usage_path = Path(environment["OKF_USAGE_LOG"])
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        tracker = UsageTracker(usage_path)
        llm = LLM(cfg.llm, tracker, client=recorder.client("llm", cfg.llm.timeout))
        recorder.bind_op(llm, "chat")
        classifier = None
        if mode == "crow":
            classifier = Classifier(cfg.classifier, tracker, client=recorder.client("classifier", cfg.classifier.timeout))
            recorder.bind_op(classifier, "ask")
        wiki = Wiki(cfg, llm=llm, classifier=classifier)
        _resolved_provenance(wiki, package, provenance)
        provenance["status"] = "ready"
        persist()
        return wiki, recorder, _sanitize(provenance, secrets)
    except Exception as exc:
        provenance["status"] = "initialization_error"
        provenance["error"] = {"type": type(exc).__name__, "message": str(exc)}
        persist()
        recorder._append({"event": "initialization_error", "role": "adapter", "error": provenance["error"], "timestamp": _utc()})
        if wiki is not None:
            wiki.close()
        recorder.close()
        error = AdapterCapabilityError(f"cannot initialize traceable llmw2 {mode} mode: {_sanitize(str(exc), secrets)}")
        error.provenance = _sanitize(provenance, secrets)
        raise error from exc


def note_snapshot(wiki: Any, paths: list[str]) -> list[dict[str, Any]]:
    """Read selected notes without changing their content or leaving the bundle."""
    root = Path(wiki.cfg.bundle).resolve()
    snapshots = []
    for rel in paths:
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"selected note path escapes its bundle: {rel!r}")
        text = path.read_text(encoding="utf-8")
        note = wiki.store.read(rel)
        snapshots.append({
            "path": rel, "text": text, "full_text": note.full_text(), "sources": _sanitize(note.frontmatter.get("sources", [])),
            "frontmatter": _sanitize(note.frontmatter), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return snapshots
