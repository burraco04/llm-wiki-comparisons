"""Offline regression checks for launcher failures before GPU/model startup.

Run with ``python -m unittest -v test_startup_scripts``. Every external GPU,
container, metadata and HTTP operation is replaced in a disposable directory.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parent


def bash_executable() -> str | None:
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        if git_bash.is_file():
            return str(git_bash)
    return shutil.which("bash")


BASH = bash_executable()

MOCKS = {
    "nvidia-smi": r"""#!/usr/bin/env bash
set -eu
field= gpu=
while (( $# )); do
  case "$1" in
    -i) gpu="$2"; shift ;;
    --query-gpu=*) field="${1#--query-gpu=}" ;;
  esac
  shift
done
if [[ "$field" == "${MOCK_NVIDIA_FAIL_FIELD:-}" ]]; then
  # Real nvidia-smi failures can report their diagnostic on captured stdout.
  printf 'GPU query unavailable in offline fixture\n'
  exit 42
fi
case "$field" in
  uuid)
    case "$gpu" in
      0) printf 'GPU-00000000-0000-0000-0000-000000000000\n' ;;
      1) printf 'GPU-00000000-0000-0000-0000-000000000001\n' ;;
      GPU-*) printf '%s\n' "$gpu" ;;
      *) exit 43 ;;
    esac ;;
  memory.free) printf '24121\n' ;;
  memory.total) printf '24576\n' ;;
  *) exit 44 ;;
esac
""",
    "docker": r"""#!/usr/bin/env bash
set -eu
# Record only the operation, never environment values or credentials.
printf '%s\n' "${1:-}" >> "$MOCK_DOCKER_LOG"
case "${1:-}" in
  image) printf 'sha256:offline-fixture\n' ;;
  inspect) printf 'running\n' ;;
  run) printf 'offline-container\n' ;;
  exec) printf 'MOCK benchmark executed\n' ;;
esac
""",
    "python3": r"""#!/usr/bin/env bash
set -eu
if [[ "${MOCK_METADATA_FAIL:-0}" == 1 ]]; then
  printf 'MOCK host metadata capture failed\n' >&2
  exit 37
fi
while (( $# )); do
  if [[ "$1" == --output ]]; then
    printf '{}\n' > "$2"
    break
  fi
  shift
done
""",
    "curl": "#!/usr/bin/env bash\nprintf '{\"data\":[]}\\n'\n",
    "sleep": "#!/usr/bin/env bash\nexit 0\n",
}


@unittest.skipUnless(BASH, "Bash is required (Git Bash is supported on Windows)")
class StartupScriptsTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="startup-scripts-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name).resolve()
        # TemporaryDirectory cleanup is restricted to this verified scratch root.
        self.assertTrue(self.work.is_relative_to(scratch.resolve()))
        mockbin = self.work / "mockbin"
        mockbin.mkdir()
        for name, source in MOCKS.items():
            target = mockbin / name
            target.write_text(textwrap.dedent(source), encoding="utf-8", newline="\n")
            target.chmod(0o755)
        for name in ("run_with_vllm.sh", "start_vllm.sh", "capture_runtime_metadata.py"):
            shutil.copyfile(ROOT / name, self.work / name)
            (self.work / name).chmod(0o755)
        self.trace = self.work / "docker-operations.txt"
        # Inherit only process-launch essentials; do not forward personal/model keys.
        keep = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "SYSTEMDRIVE", "HOME", "USERPROFILE")
        self.environment = {key: value for key, value in os.environ.items() if key.upper() in keep}
        self.environment.update({
            "LC_ALL": "C",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SLURM_JOB_GPUS": "0,1",
            "SLURM_SUBMIT_DIR": self.work.as_posix(),
            "SLURM_JOB_ID": "offline-test",
            "HF_CACHE": (self.work / "hf-cache").as_posix(),
            "RESULTS_DIR": (self.work / "results").as_posix(),
            "LOG_DIR": (self.work / "logs").as_posix(),
            "VLLM_IMAGE": "offline-fixture-image",
            "MOCK_DOCKER_LOG": self.trace.as_posix(),
        })

    def run_bash(self, command: str, **environment: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, "-c", 'cd "$1"; export PATH="$PWD/mockbin:$PATH"; ' + command,
             "offline-startup-test", self.work.as_posix()],
            env={**self.environment, **environment},
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )

    def run_launcher(self, **environment: str) -> subprocess.CompletedProcess[str]:
        return self.run_bash(
            "bash run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- python benchmark_llm_wiki.py",
            **environment,
        )

    def docker_operations(self) -> list[str]:
        return self.trace.read_text(encoding="utf-8").splitlines() if self.trace.exists() else []

    def test_original_unguarded_query_reproduces_cleanup_only_output(self) -> None:
        # Preserve the exact failure mechanism, independent of future launcher edits.
        baseline = self.work / "unguarded-baseline.sh"
        baseline.write_text(textwrap.dedent(r"""\
            set -euo pipefail
            trap 'echo ">>> stopping vLLM"' EXIT
            FREE0=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
            echo 'unexpected: query succeeded'
            """), encoding="utf-8", newline="\n")
        result = self.run_bash("bash unguarded-baseline.sh", MOCK_NVIDIA_FAIL_FIELD="memory.free")
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(result.stdout.strip(), ">>> stopping vLLM")
        self.assertEqual(result.stderr, "")

    def test_gpu_query_failure_is_reported_without_starting_container(self) -> None:
        result = self.run_launcher(MOCK_NVIDIA_FAIL_FIELD="memory.free")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertIn("nvidia-smi", result.stderr)
        self.assertIn("memory.free", result.stderr)
        self.assertIn("GPU query unavailable in offline fixture", result.stderr)
        self.assertNotIn("run", self.docker_operations())
        self.assertNotIn("exec", self.docker_operations())

    def test_healthy_fake_services_reach_benchmark_exec(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("run", self.docker_operations())
        self.assertIn("exec", self.docker_operations())
        self.assertIn("MOCK benchmark executed", result.stdout)
        self.assertIn(">>> done", result.stdout)

    def test_metadata_failure_stops_before_container_and_keeps_error(self) -> None:
        result = self.run_launcher(MOCK_METADATA_FAIL="1")
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertIn("MOCK host metadata capture failed", result.stderr)
        self.assertIn("metadata", result.stdout + result.stderr)
        self.assertNotIn("run", self.docker_operations())
        self.assertNotIn("exec", self.docker_operations())


if __name__ == "__main__":
    unittest.main()
