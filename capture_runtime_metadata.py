"""Capture bounded, non-secret build/host/container provenance using stdlib only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


PACKAGES = (
    "vllm", "torch", "torchvision", "torchaudio", "torchcodec", "transformers",
    "tokenizers", "huggingface-hub", "numpy", "openai", "datasets", "llmwiki2",
    "llmw2", "rank-bm25", "rouge-score", "sentence-transformers", "jsonschema",
    "nltk", "scikit-learn", "pydantic", "fastapi",
)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def command(argv: list[str]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False,
                                env={**os.environ, "LC_ALL": "C"})
        if result.returncode:
            return None, f"command exited with status {result.returncode}"
        return result.stdout.strip(), None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, type(exc).__name__


def machine() -> dict:
    release = {}
    for line in (read_text(Path("/etc/os-release")) or "").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            release[key] = value.strip('"')
    cpu, error = command(["lscpu", "--json"])
    keep = {"Architecture", "CPU(s)", "Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core"}
    cpu_info = {}
    if cpu:
        try:
            cpu_info = {row["field"].rstrip(":"): row["data"] for row in json.loads(cpu)["lscpu"]
                        if row["field"].rstrip(":") in keep}
        except (ValueError, KeyError, TypeError):
            error = "unrecognized lscpu JSON"
    mem = {}
    for line in (read_text(Path("/proc/meminfo")) or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "SwapTotal"):
            mem[key] = value.strip()
    gpu_text, gpu_error = command([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpu_keys = ("index", "uuid", "name", "total_memory_mib", "free_memory_mib_at_capture", "driver_version")
    gpus = [dict(zip(gpu_keys, (cell.strip() for cell in row)))
            for row in csv.reader((gpu_text or "").splitlines())]
    return {
        "os_release": release, "kernel": platform.release(), "architecture": platform.machine(),
        "cpu": cpu_info, "cpu_query_error": error, "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "memory": mem, "gpus": gpus, "gpu_query_error": gpu_error,
        "cgroup_limits": {
            "memory_max_bytes": read_text(Path("/sys/fs/cgroup/memory.max")),
            "memory_limit_v1_bytes": read_text(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
            "cpu_max": read_text(Path("/sys/fs/cgroup/cpu.max")),
        },
    }


def versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "executable": sys.executable, "packages": found}


def load_json(path: str) -> dict | None:
    if not path:
        return None
    text = read_text(Path(path))
    if text is None:
        return {"unavailable": True}
    try:
        return json.loads(text)
    except ValueError:
        return {"unavailable": True, "error": "invalid JSON"}


def build_hashes() -> dict:
    root = Path(__file__).resolve().parent
    files = [root / name for name in (
        "benchmark_llm_wiki.py", "benchmark_artifacts.py", "llmwiki_adapter.py",
        "start_vllm_services.sh", "capture_runtime_metadata.py",
    )]
    files += sorted((root / "prompts_benchmark").rglob("*.md"))
    files.append(Path("/tmp/requirements.txt"))
    return {str(path.relative_to(root)) if root in path.parents else str(path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build", "host", "container"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-metadata", default="")
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--llm-argc", type=int, default=0)
    parser.add_argument("--llm-device", default="0")
    parser.add_argument("--classifier-device", default="1")
    parser.add_argument("service_commands", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    result = {"schema_version": 1, "phase": args.phase,
              "captured_at_utc": datetime.now(timezone.utc).isoformat()}
    if args.phase == "build":
        result.update(versions=versions(), sha256=build_hashes())
    elif args.phase == "host":
        result.update(machine=machine(), allocation={
            "gpu_uuids": args.gpu,
            "cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
            "memory_per_node_mib": os.getenv("SLURM_MEM_PER_NODE"),
            "memory_per_cpu_mib": os.getenv("SLURM_MEM_PER_CPU"),
        })
    else:
        commands = args.service_commands
        if commands[:1] == ["--"]:
            commands = commands[1:]
        if not 0 < args.llm_argc < len(commands):
            parser.error("container capture requires both service commands and --llm-argc")
        result.update(
            host=load_json(args.host_metadata),
            container={"machine": machine(), "versions": versions()},
            build=load_json("/workdir/image_build_metadata.json"),
            image={"tag": os.getenv("BENCHMARK_IMAGE"), "id": os.getenv("BENCHMARK_IMAGE_ID")},
            services={
                "llm": {"device": args.llm_device, "argv": commands[:args.llm_argc]},
                "classifier": {"device": args.classifier_device, "argv": commands[args.llm_argc:]},
            },
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
