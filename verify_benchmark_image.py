"""Reject outdated benchmark images without starting models or requesting GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
APPLICATION_FILES = (
    "benchmark_llm_wiki.py", "benchmark_artifacts.py", "llmwiki_adapter.py",
    "capture_runtime_metadata.py", "start_vllm_services.sh",
)
REBUILD_MESSAGE = (
    "Rebuild the benchmark image from this checkout (docker build -t "
    "llm-wiki-benchmark:structured-wiki-v3 .), then set VLLM_IMAGE to its tag."
)

# This runs even in an old image: it imports only the Python standard library,
# rather than importing the benchmark or relying on a recent metadata helper.
IMAGE_PAYLOAD = r'''
import hashlib
import json
from pathlib import Path
import sys

def file_hash(path):
    try:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None

def inventory(root, requirements, files):
    result = {name: file_hash(root / name) for name in files}
    for path in sorted((root / "prompts_benchmark").rglob("*.md")):
        result[path.relative_to(root).as_posix()] = file_hash(path)
    result["requirements.txt"] = file_hash(requirements)
    return result

def main(root=Path("/workdir"), requirements=Path("/tmp/requirements.txt")):
    request = json.load(sys.stdin)
    print(json.dumps(inventory(root, requirements, request["files"]), sort_keys=True))

if __name__ == "__main__":
    main()
'''


class ImageVerificationError(RuntimeError):
    pass


def source_manifest(root: Path) -> dict:
    paths = {name: root / name for name in (*APPLICATION_FILES, "requirements.txt")}
    prompt_dir = root / "prompts_benchmark"
    if not prompt_dir.is_dir():
        raise ImageVerificationError("Local prompts_benchmark directory is missing.")
    paths.update({path.relative_to(root).as_posix(): path
                  for path in sorted(prompt_dir.rglob("*.md"))})
    hashes = {}
    for name, path in paths.items():
        try:
            hashes[name] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        except OSError:
            raise ImageVerificationError("Local source file is missing or unreadable: " + repr(name)) from None
    return hashes


def docker_command(argv: list, *, input_text: str | None = None, timeout: int = 60):
    try:
        result = subprocess.run(
            argv, input=input_text, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError:
        raise ImageVerificationError("Docker executable was not found.") from None
    except subprocess.TimeoutExpired:
        raise ImageVerificationError("Docker image verification timed out.") from None
    except OSError as exc:
        raise ImageVerificationError("Docker could not be executed (" + type(exc).__name__ + ").") from None
    if result.returncode:
        # Do not echo arbitrary container stdout or file contents into the log.
        operation = "image inspect" if argv[1:3] == ["image", "inspect"] else "run"
        raise ImageVerificationError(
            "Docker " + operation + " failed (exit=" + str(result.returncode)
            + "); check that the image exists locally and Docker is accessible."
        )
    return result.stdout


def verify_image(image: str, root: Path = ROOT) -> str:
    expected = source_manifest(root)
    image_id = docker_command(
        ["docker", "image", "inspect", "--format", "{{.Id}}", "--", image], timeout=20,
    ).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ImageVerificationError("Docker returned an invalid immutable image ID.")
    output = docker_command(
        ["docker", "run", "--rm", "-i", "--network", "none",
         "--env", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "python",
         image_id, "-c", IMAGE_PAYLOAD],
        input_text=json.dumps({"files": list(APPLICATION_FILES)}),
    )
    try:
        actual = json.loads(output)
    except ValueError:
        raise ImageVerificationError("Docker returned an invalid source inventory.") from None
    if not isinstance(actual, dict) or not all(
        isinstance(name, str) and (value is None or isinstance(value, str)
                                  and re.fullmatch(r"[0-9a-f]{64}", value))
        for name, value in actual.items()
    ):
        raise ImageVerificationError("Docker returned an invalid source inventory.")
    missing = sorted(name for name in expected if actual.get(name) is None)
    changed = sorted(name for name in expected if name in actual
                     and actual[name] is not None and actual[name] != expected[name])
    extra = sorted(set(actual) - set(expected))
    details = []
    for label, names in (("missing/unreadable", missing), ("different", changed), ("extra", extra)):
        if names:
            details.append(label + ": " + ", ".join(repr(name) for name in names))
    if details:
        raise ImageVerificationError("Image sources do not match this checkout (" + "; ".join(details) + ").")
    return image_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Existing local image tag or immutable image ID; never pulled.")
    args = parser.parse_args(argv)
    print(">>> verifying benchmark image against local sources", file=sys.stderr)
    try:
        image_id = verify_image(args.image)
    except ImageVerificationError as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        print(REBUILD_MESSAGE, file=sys.stderr)
        return 2
    print(">>> benchmark image sources match", file=sys.stderr)
    print(image_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
