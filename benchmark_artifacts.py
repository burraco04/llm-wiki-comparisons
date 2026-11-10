"""Portable artifact and reporting helpers for the offline benchmark runner.

These functions record inputs and measurements; they do not select data, call
models, or change the experiment. Values missing from runtime provenance remain
explicitly unavailable rather than being guessed from another installation.
"""
from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import math
import numbers
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DEFAULT_PACKAGES = (
    "llmwiki2", "llmw2", "vllm", "openai", "datasets",
    "sentence-transformers", "rank-bm25", "nltk", "scikit-learn",
    "jsonschema", "numpy", "rouge-score",
)
MEAN_METRICS = (
    "em", "token_f1", "rouge_l", "seconds_wall", "llm_calls",
    "input_tokens", "output_tokens", "classifier_calls",
    "classifier_input_tokens", "classifier_output_tokens",
    "total_input_tokens", "total_output_tokens",
)
RESOURCE_METRICS = MEAN_METRICS[3:]
_SECRET_KEY = re.compile(
    r"(^|_)(api_?key|access_?token|refresh_?token|authorization|password|"
    r"passwd|secret|credential|credentials|cookie)(_|$)", re.I
)
_APPEND_LOCK = threading.Lock()


def json_safe(value: Any) -> Any:
    """Convert dataclasses, model results, paths and numeric scalars to JSON."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(v) for v in sorted(value, key=str)]
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json"))
        except (TypeError, ValueError):
            return json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _secret_key(key: str) -> bool:
    normalized = re.sub(r"[-.\s]+", "_", key)
    # A plain "key" can be a typed-decision option identifier, not a credential.
    return normalized.lower() == "token" or bool(_SECRET_KEY.search(normalized))


def _redact_url(value: str) -> str:
    if not re.match(r"^(?:https?|ssh|git\+https?|git\+ssh)://", value, re.I):
        return value
    try:
        parts = urlsplit(value)
        netloc = parts.netloc.rsplit("@", 1)[-1]
        query = [
            (key, "[REDACTED]" if _secret_key(key) else item)
            for key, item in parse_qsl(parts.query, keep_blank_values=True)
        ]
        fragment = parts.fragment
        if "=" in fragment:
            fragment = urlencode([
                (key, "[REDACTED]" if _secret_key(key) else item)
                for key, item in parse_qsl(fragment, keep_blank_values=True)
            ])
        return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), fragment))
    except ValueError:
        return "[UNPARSEABLE URL]"


def redact_secrets(value: Any) -> Any:
    """Remove credential fields and URL credentials while retaining token counts."""
    value = json_safe(value)
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _secret_key(key) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _redact_url(value)
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    """Write a complete UTF-8 JSON artifact atomically, preserving existing files on error."""
    content = json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _atomic_bytes(Path(path), content.encode("utf-8"))


def append_jsonl(path: Path, value: Any) -> None:
    """Append and flush one complete JSON record for durable query checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def _write_jsonl(path: Path, values: Iterable[Any]) -> None:
    content = "".join(
        json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False) + "\n"
        for value in values
    )
    _atomic_bytes(path, content.encode("utf-8"))


def _module_version(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                if "__version__" in names:
                    value = ast.literal_eval(node.value)
                    return str(value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "__version__":
                    return str(ast.literal_eval(node.value))
    except (OSError, SyntaxError, ValueError, TypeError):
        pass
    return None


def _module_provenance(name: str) -> dict[str, Any]:
    loaded = sys.modules.get(name)
    module_path = getattr(loaded, "__file__", None)
    try:
        if not module_path:
            spec = importlib.util.find_spec(name)
            module_path = spec.origin if spec is not None else None
        if not module_path or module_path in {"built-in", "frozen"}:
            return {"status": "unavailable"}
        path = Path(module_path).resolve()
        version = getattr(loaded, "__version__", None) or _module_version(path)
        files = sorted(path.parent.rglob("*.py")) if path.name == "__init__.py" else [path]
        digest = hashlib.sha256()
        for source in files:
            relative = source.relative_to(path.parent).as_posix()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
        return {
            "status": "recorded", "path": str(path), "module_version": version,
            "source_sha256": digest.hexdigest(), "source_files": len(files),
            "source_hash_scope": "relative paths and contents of all package Python files",
        }
    except (ImportError, ModuleNotFoundError, OSError, ValueError) as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def collect_package_provenance(packages: Iterable[str] | None = None) -> dict[str, Any]:
    """Record installed distributions and library source identity without importing it."""
    requested = tuple(DEFAULT_PACKAGES if packages is None else packages)
    if "llmwiki2" in requested or "llmw2" in requested:
        requested = tuple(dict.fromkeys((*requested, "llmwiki2", "llmw2")))
    distributions: dict[str, Any] = {}
    for name in requested:
        try:
            distribution = metadata.distribution(name)
            item: dict[str, Any] = {
                "status": "installed",
                "name": distribution.metadata.get("Name", name),
                "version": distribution.version,
            }
            direct_url = distribution.read_text("direct_url.json")
            if direct_url:
                try:
                    item["direct_url"] = redact_secrets(json.loads(direct_url))
                except (json.JSONDecodeError, TypeError):
                    item["direct_url_status"] = "invalid JSON"
            distributions[name] = item
        except metadata.PackageNotFoundError:
            distributions[name] = {"status": "not installed"}
        except (OSError, ValueError) as exc:
            distributions[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    modules = {"llmw2": _module_provenance("llmw2")} if {"llmw2", "llmwiki2"} & set(requested) else {}
    return redact_secrets({"distributions": distributions, "modules": modules})


def _linux_cpu() -> dict[str, Any]:
    result: dict[str, Any] = {
        "logical_cpus": os.cpu_count(), "model": platform.processor() or None,
    }
    path = Path("/proc/cpuinfo")
    if path.is_file():
        try:
            records = []
            for block in path.read_text(encoding="utf-8").split("\n\n"):
                fields = dict(
                    line.split(":", 1) for line in block.splitlines() if ":" in line
                )
                records.append({key.strip(): value.strip() for key, value in fields.items()})
            if records:
                result["model"] = records[0].get("model name") or records[0].get("Hardware") or result["model"]
            cores = {
                (record.get("physical id", "0"), record["core id"])
                for record in records if "core id" in record
            }
            sockets = {record["physical id"] for record in records if "physical id" in record}
            if cores:
                result["physical_cores"] = len(cores)
            if sockets:
                result["sockets"] = len(sockets)
            result["source"] = "/proc/cpuinfo and os.cpu_count"
        except OSError as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _memory_snapshot() -> dict[str, Any]:
    path = Path("/proc/meminfo")
    if path.is_file():
        try:
            fields = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                values = value.split()
                if values:
                    fields[key] = int(values[0]) * (1024 if len(values) > 1 and values[1] == "kB" else 1)
            result = {
                "status": "recorded", "source": "/proc/meminfo",
                "total_bytes": fields.get("MemTotal"),
                "available_bytes": fields.get("MemAvailable"),
                "swap_total_bytes": fields.get("SwapTotal"),
            }
            for candidate in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
                limit_path = Path(candidate)
                if limit_path.is_file():
                    value = limit_path.read_text(encoding="utf-8").strip()
                    result["cgroup_limit_bytes"] = int(value) if value.isdigit() else value
                    result["cgroup_limit_source"] = candidate
                    break
            return result
        except (OSError, ValueError) as exc:
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            return {
                "status": "recorded", "source": "GlobalMemoryStatusEx",
                "total_bytes": status.total_phys, "available_bytes": status.avail_phys,
            }
        except (AttributeError, OSError) as exc:
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "unavailable", "reason": "No supported operating-system memory interface"}


def _os_snapshot() -> dict[str, Any]:
    result = {
        "system": platform.system(), "kernel": platform.release(),
        "machine": platform.machine(), "platform": platform.platform(),
    }
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        try:
            result["distribution"] = {
                key: value.strip('"')
                for key, value in (
                    line.split("=", 1)
                    for line in os_release.read_text(encoding="utf-8").splitlines()
                    if "=" in line and not line.startswith("#")
                )
                if key in {"NAME", "VERSION", "VERSION_ID", "PRETTY_NAME", "ID"}
            }
        except OSError as exc:
            result["distribution_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi not found"}
    command = [
        executable, "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        devices = []
        for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
            if len(row) == 5:
                index, uuid, name, memory, driver = (value.strip() for value in row)
                devices.append({
                    "index": int(index), "uuid": uuid, "name": name,
                    "memory_total_mib": int(memory), "driver_version": driver,
                })
        if not devices:
            return {"status": "unavailable", "reason": "nvidia-smi returned no parseable devices"}
        return {"status": "recorded", "devices": devices, "source": "nvidia-smi device inventory"}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def runtime_snapshot() -> dict[str, Any]:
    """Record process-visible resources and optional launcher metadata, not GPU peaks."""
    supplied = os.getenv("BENCHMARK_RUNTIME_METADATA")
    launcher: dict[str, Any] = {"status": "unavailable", "reason": "BENCHMARK_RUNTIME_METADATA not set"}
    if supplied:
        path = Path(supplied)
        try:
            launcher = {
                "status": "recorded", "path": str(path),
                "data": redact_secrets(json.loads(path.read_text(encoding="utf-8"))),
            }
        except (OSError, ValueError) as exc:
            launcher = {"status": "unavailable", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return redact_secrets({
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Resources visible to the benchmark process; launcher metadata is separate",
        "platform": {"python": platform.python_version(), "executable": sys.executable},
        "cpu": _linux_cpu(), "memory": _memory_snapshot(), "os": _os_snapshot(),
        "gpus": _gpu_snapshot(), "launcher_metadata": launcher,
        "visible_devices": {
            name: os.getenv(name)
            for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        },
        "measurement_scope": "Inventory and startup snapshot; no peak-memory or energy measurement",
    })


def _length_summary(values: Iterable[int]) -> dict[str, Any]:
    values = list(values)
    return {
        "count": len(values), "min": min(values) if values else None,
        "mean": mean(values) if values else None,
        "max": max(values) if values else None, "total": sum(values),
    }


def _support_count(row: Mapping[str, Any]) -> int | None:
    field = "evidence_titles" if "evidence_titles" in row else "evidence_list" if "evidence_list" in row else None
    if field is None:
        return None
    titles = set()
    for item in row.get(field) or []:
        title = item if isinstance(item, str) else item.get("title") or item.get("source") if isinstance(item, Mapping) else None
        if title:
            titles.add(str(title))
    return len(titles)


def dataset_statistics(
    docs: Iterable[Mapping[str, Any]],
    questions: Iterable[Mapping[str, Any]],
    full_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the given selection without reordering or selecting any records."""
    docs, questions = list(docs), list(questions)
    supports = [str(n) if (n := _support_count(row)) is not None else "unknown" for row in questions]
    types = [str(row.get("question_type") or "unknown") for row in questions]
    cross: dict[str, Counter[str]] = defaultdict(Counter)
    for kind, support in zip(types, supports):
        cross[kind][support] += 1
    result = {
        "documents": len(docs), "questions": len(questions),
        "question_types": dict(sorted(Counter(types).items())),
        "supporting_document_counts": dict(sorted(Counter(supports).items())),
        "question_type_by_supporting_documents": {
            kind: dict(sorted(counts.items())) for kind, counts in sorted(cross.items())
        },
        "document_words": _length_summary(len(str(d.get("text", "")).split()) for d in docs),
        "document_characters": _length_summary(len(str(d.get("text", ""))) for d in docs),
        "question_words": _length_summary(len(str(q.get("question", q.get("query", ""))).split()) for q in questions),
        "question_characters": _length_summary(len(str(q.get("question", q.get("query", "")))) for q in questions),
        "length_units": {
            "words": "Whitespace-separated units, as in str.split()",
            "characters": "Unicode code points, as in len(str); not model tokens",
        },
        "support_count_scope": "Distinct annotated source titles, not retrieval steps",
    }
    if full_counts is not None:
        result["full_dataset_counts"] = json_safe(full_counts)
    return result


def save_dataset(
    run_dir: Path, docs: Iterable[Mapping[str, Any]],
    questions: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save exact selected input records and their descriptive statistics."""
    docs, questions = list(docs), list(questions)
    directory = Path(run_dir) / "dataset"
    _write_jsonl(directory / "documents.jsonl", docs)
    _write_jsonl(directory / "questions.jsonl", questions)
    statistics = dataset_statistics(docs, questions)
    write_json(directory / "statistics.json", statistics)
    if metadata is not None:
        write_json(directory / "metadata.json", redact_secrets(metadata))
    return statistics


def save_prompt_snapshot(
    run_dir: Path, prompt_dir: Path, direct_prompts: Mapping[str, str],
    schemas: Mapping[str, Any],
) -> dict[str, Any]:
    """Snapshot exact prompt overrides, direct templates and JSON schemas."""
    destination = Path(run_dir) / "prompts"
    source = Path(prompt_dir).resolve()
    overrides = []
    for path in sorted(source.rglob("*.md")) if source.is_dir() else []:
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(source)
        except ValueError:
            overrides.append({"relative_path": path.relative_to(source).as_posix(), "status": "skipped outside prompt directory"})
            continue
        relative = path.relative_to(source)
        content = path.read_bytes()
        _atomic_bytes(destination / "overrides" / relative, content)
        overrides.append({
            "relative_path": relative.as_posix(), "status": "recorded",
            "snapshot_path": (Path("prompts") / "overrides" / relative).as_posix(),
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        })
    write_json(destination / "direct_prompts.json", direct_prompts)
    write_json(destination / "schemas.json", schemas)
    manifest = {
        "source_directory": str(source),
        "source_directory_status": "recorded" if source.is_dir() else "unavailable",
        "overrides": overrides,
        "direct_prompts": {
            name: {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(), "characters": len(value)}
            for name, value in direct_prompts.items()
        },
        "schemas": {
            name: hashlib.sha256(json.dumps(json_safe(schema), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
            for name, schema in schemas.items()
        },
        "scope": "Project templates and overrides; complete requests are recorded separately by the runner",
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def _number(value: Any) -> float:
    try:
        result = float(value or 0)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _normalized_usage(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    usage = json_safe(row.get("usage") or {})
    usage = usage if isinstance(usage, Mapping) else {}
    writing = usage.get("llm") or {}
    classifier = usage.get("classifier") or {}
    for flat, nested, key in (
        ("llm_calls", writing, "calls"), ("input_tokens", writing, "input_tokens"),
        ("output_tokens", writing, "output_tokens"), ("classifier_calls", classifier, "calls"),
        ("classifier_input_tokens", classifier, "input_tokens"),
        ("classifier_output_tokens", classifier, "output_tokens"),
    ):
        if flat not in value:
            value[flat] = nested.get(key, 0) if isinstance(nested, Mapping) else 0
    value["total_input_tokens"] = _number(value.get("input_tokens")) + _number(value.get("classifier_input_tokens"))
    value["total_output_tokens"] = _number(value.get("output_tokens")) + _number(value.get("classifier_output_tokens"))
    return value


def _is_failed(row: Mapping[str, Any]) -> bool:
    return bool(row.get("error") or row.get("exception")) or row.get("status") in {"error", "failed"}


def _error_type(row: Mapping[str, Any]) -> str:
    explicit = row.get("error_type") or row.get("exception_type")
    if explicit:
        return str(explicit)
    for field in ("error", "exception"):
        nested = row.get(field)
        if isinstance(nested, Mapping) and nested.get("type"):
            return str(nested["type"])
    return "unspecified"


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Average over every recorded question, including failed questions scored zero."""
    rows = list(rows)
    normalized = [_normalized_usage(row) for row in rows]
    totals = {key: sum(_number(row.get(key)) for row in normalized) for key in RESOURCE_METRICS}
    failures_known = any("failed_requests" in row for row in rows)
    errors = sum(_is_failed(row) for row in rows)
    error_types = Counter(_error_type(row) for row in rows if _is_failed(row))
    return {
        "questions": len(rows), "errors": errors,
        "successful_questions": len(rows) - errors,
        "metric_denominator": len(rows),
        "failed_requests": sum(int(_number(row.get("failed_requests"))) for row in rows) if failures_known else None,
        "failed_requests_recorded": failures_known,
        "error_types": dict(sorted(error_types.items())),
        **{
            key: sum(_number(row.get(key)) for row in normalized) / len(rows) if rows else 0.0
            for key in MEAN_METRICS
        },
        "totals": totals,
        "available_values": {
            key: sum(key in row and row[key] is not None for row in rows)
            for key in ("em", "token_f1", "rouge_l", "seconds_wall")
        },
        "aggregation": "Per-question arithmetic mean, including failed questions; no category balancing",
    }


def summary_by_group(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, Any]:
    """Describe each type or support-count group without changing the overall mean."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    support_fields = {"supporting_documents", "supporting_document_count", "support_count", "evidence_count"}
    for row in rows:
        value = row.get(field)
        if value is None and field in support_fields:
            value = _support_count(row)
        groups[str(value) if value is not None and value != "" else "unknown"].append(row)
    return {value: summarize_rows(group) for value, group in sorted(groups.items())}


def buildcost_summary(
    ingestion: Iterable[Mapping[str, Any]], indexing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep construction costs separate from per-query means and token counts."""
    ingestion = list(ingestion)
    normalized = [_normalized_usage(row) for row in ingestion]
    stage_seconds: Counter[str] = Counter()
    for row in ingestion:
        for stage, value in (row.get("timing") or {}).items():
            stage_seconds[str(stage)] += _number(value)
    indexing = dict(indexing or {})
    indexing_seconds = {
        key: _number(value) for key, value in indexing.items()
        if key == "seconds" or key.endswith("_seconds") or key.endswith("_seconds_wall")
    }
    ingestion_total = sum(_number(row.get("seconds_wall")) for row in ingestion)
    usage = {
        "llm": {
            "calls": sum(_number(row.get("llm_calls")) for row in normalized),
            "input_tokens": sum(_number(row.get("input_tokens")) for row in normalized),
            "output_tokens": sum(_number(row.get("output_tokens")) for row in normalized),
        },
        "classifier": {
            "calls": sum(_number(row.get("classifier_calls")) for row in normalized),
            "input_tokens": sum(_number(row.get("classifier_input_tokens")) for row in normalized),
            "output_tokens": sum(_number(row.get("classifier_output_tokens")) for row in normalized),
        },
    }
    return {
        "documents": len(ingestion), "errors": sum(_is_failed(row) for row in ingestion),
        "ingestion_seconds_total": ingestion_total,
        "ingestion_seconds_mean": ingestion_total / len(ingestion) if ingestion else 0.0,
        "stage_seconds_total": dict(sorted(stage_seconds.items())),
        "usage": usage, "indexing": json_safe(indexing),
        "indexing_seconds_total": sum(indexing_seconds.values()),
        "construction_seconds_total": ingestion_total + sum(indexing_seconds.values()),
        "total_input_tokens": usage["llm"]["input_tokens"] + usage["classifier"]["input_tokens"],
        "total_output_tokens": usage["llm"]["output_tokens"] + usage["classifier"]["output_tokens"],
        "scope": "Recorded ingestion calls and index preparation only; excludes query processing",
    }
