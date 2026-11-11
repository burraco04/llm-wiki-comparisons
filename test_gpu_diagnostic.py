"""Offline checks for the Bash script submitted directly to Slurm (Python 3.8+)."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH.is_file() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required (Git Bash is supported on Windows)")
class GpuDiagnosticTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="gpu-diagnostic-", dir=scratch)
        self.work = Path(temporary.name).resolve()
        self.assertEqual(self.work.parent, scratch.resolve())
        self.addCleanup(temporary.cleanup)
        shutil.copyfile(ROOT / "gpu_diagnostic.sh", self.work / "gpu_diagnostic.sh")
        mockbin = self.work / "mockbin"
        mockbin.mkdir()
        query = mockbin / "nvidia-smi"
        with query.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$QUERY_LOG"
if [[ "$1" == -i && "$2" == "${FAIL_GPU:-}" ]]; then
  printf 'fixture GPU query failed\\n' >&2
  exit 42
fi
printf 'fixture GPU query succeeded\\n'
''')
        query.chmod(0o755)
        self.trace = self.work / "queries.txt"
        keep = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "SYSTEMDRIVE", "HOME", "USERPROFILE")
        self.environment = {key: value for key, value in os.environ.items() if key.upper() in keep}
        self.environment.update({"QUERY_LOG": self.trace.as_posix(), "LC_ALL": "C"})

    def run_diagnostic(self, **environment: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, "-c", 'cd "$1"; export PATH="$PWD/mockbin:$PATH"; bash gpu_diagnostic.sh',
             "offline-gpu-diagnostic", self.work.as_posix()],
            env={**self.environment, **environment}, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )

    def allocated_queries(self) -> list[str]:
        return [line for line in self.trace.read_text(encoding="utf-8").splitlines() if line.startswith("-i ")]

    def test_uses_bash_and_checks_only_allocated_gpu_indices(self) -> None:
        script = (ROOT / "gpu_diagnostic.sh").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/bin/bash\n"))
        result = self.run_diagnostic(CUDA_VISIBLE_DEVICES="0,2", SLURM_JOB_GPUS="1,3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.allocated_queries(), [
            "-i 0 --query-gpu=uuid,memory.free,memory.total --format=csv",
            "-i 2 --query-gpu=uuid,memory.free,memory.total --format=csv",
        ])

    def test_slurm_fallback_accepts_gpu_uuids(self) -> None:
        result = self.run_diagnostic(SLURM_JOB_GPUS="GPU-first,GPU-second")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Checking allocated GPU GPU-first", result.stdout)
        self.assertIn("Checking allocated GPU GPU-second", result.stdout)

    def test_failure_keeps_status_and_checks_remaining_gpu(self) -> None:
        result = self.run_diagnostic(CUDA_VISIBLE_DEVICES="0,2", FAIL_GPU="0")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertIn("fixture GPU query failed", result.stderr)
        self.assertIn("Checking allocated GPU 2", result.stdout)
        self.assertEqual(len(self.allocated_queries()), 2)

    def test_missing_allocation_fails_without_querying_arbitrary_gpu(self) -> None:
        result = self.run_diagnostic()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no allocated GPUs", result.stderr)
        self.assertEqual(self.allocated_queries(), [])


if __name__ == "__main__":
    unittest.main()
