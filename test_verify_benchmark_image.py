"""Offline image/source verification, using the real stdlib container payload."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import verify_benchmark_image as verify


ROOT = Path(__file__).resolve().parent
IMAGE_ID = "sha256:" + "a" * 64
REAL_RUN = subprocess.run


class VerifyImageTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="verify-image-", dir=scratch)
        self.work = Path(self.temporary.name).resolve()
        self.assertEqual(self.work.parent, scratch.resolve())
        self.addCleanup(self.temporary.cleanup)
        self.local = self.work / "local"
        self.image = self.work / "image"
        for tree in (self.local, self.image):
            (tree / "prompts_benchmark" / "nested").mkdir(parents=True)
            for name in (*verify.APPLICATION_FILES, "requirements.txt",
                         "prompts_benchmark/writer.md", "prompts_benchmark/nested/answer.md"):
                (tree / name).write_bytes(("fixture " + name + "\n").encode("utf-8"))
        self.calls = []

    def docker(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if argv[1:3] == ["image", "inspect"]:
            self.assertEqual(argv[-1], "fixture:tag")
            return subprocess.CompletedProcess(argv, 0, IMAGE_ID + "\n", "")
        self.assertEqual(argv[:9], ["docker", "run", "--rm", "-i", "--network", "none",
                                    "--env", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint"])
        self.assertEqual(argv[9:12], ["python", IMAGE_ID, "-c"])
        # Execute the exact image payload, with only its filesystem roots supplied
        # by the test. Its main still consumes the launcher's real stdin request.
        wrapper = (
            "namespace = {'__name__': 'offline_image_payload'}\n"
            "exec(" + repr(argv[12]) + ", namespace)\n"
            "namespace['main'](namespace['Path'](" + repr(str(self.image)) + "), "
            "namespace['Path'](" + repr(str(self.image / "requirements.txt")) + "))\n"
        )
        return REAL_RUN([sys.executable, "-c", wrapper], **kwargs)

    def verify(self):
        with patch.object(verify.subprocess, "run", side_effect=self.docker):
            return verify.verify_image("fixture:tag", self.local)

    def test_match_returns_pinned_id_and_runs_without_gpu_or_network(self):
        self.assertEqual(self.verify(), IMAGE_ID)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(call[1]["timeout"] <= 60 for call in self.calls))
        self.assertNotIn("--gpus", self.calls[1][0])
        self.assertNotIn("--pull", self.calls[1][0])

    def test_only_crlf_is_normalized(self):
        for path in self.local.rglob("*"):
            if path.is_file():
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(self.verify(), IMAGE_ID)
        (self.image / "benchmark_llm_wiki.py").write_bytes(b"different\r")
        with self.assertRaisesRegex(verify.ImageVerificationError, "different.*benchmark_llm_wiki.py"):
            self.verify()

    def test_missing_application_file_is_reported_without_contents(self):
        (self.image / "llmwiki_adapter.py").unlink()
        with self.assertRaisesRegex(verify.ImageVerificationError, "missing/unreadable.*llmwiki_adapter.py"):
            self.verify()

    def test_different_nested_prompt_and_extra_prompt_are_detected(self):
        (self.image / "prompts_benchmark/nested/answer.md").write_text("SECRET_CONTENT", encoding="utf-8")
        (self.image / "prompts_benchmark/old.md").write_text("SECRET_CONTENT", encoding="utf-8")
        with self.assertRaises(verify.ImageVerificationError) as error:
            self.verify()
        message = str(error.exception)
        self.assertIn("different: 'prompts_benchmark/nested/answer.md'", message)
        self.assertIn("extra: 'prompts_benchmark/old.md'", message)
        self.assertNotIn("SECRET_CONTENT", message)

    def test_missing_prompt_is_detected(self):
        (self.image / "prompts_benchmark/writer.md").unlink()
        with self.assertRaisesRegex(verify.ImageVerificationError, "missing/unreadable.*writer.md"):
            self.verify()

    def test_requirements_change_is_detected(self):
        (self.image / "requirements.txt").write_text("old-dependency==1\n", encoding="utf-8")
        with self.assertRaisesRegex(verify.ImageVerificationError, "different.*requirements.txt"):
            self.verify()

    def test_missing_local_file_stops_before_docker(self):
        (self.local / "benchmark_artifacts.py").unlink()
        with self.assertRaisesRegex(verify.ImageVerificationError, "Local source file.*benchmark_artifacts.py"):
            self.verify()
        self.assertEqual(self.calls, [])

    def test_missing_image_stops_before_run(self):
        with patch.object(verify.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "No such image")) as run:
            with self.assertRaisesRegex(verify.ImageVerificationError, "image inspect failed"):
                verify.verify_image("missing", self.local)
        self.assertEqual(run.call_count, 1)

    def test_invalid_image_id_is_rejected(self):
        with patch.object(verify.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "fixture:tag\n", "")) as run:
            with self.assertRaisesRegex(verify.ImageVerificationError, "invalid immutable image ID"):
                verify.verify_image("fixture:tag", self.local)
        self.assertEqual(run.call_count, 1)

    def test_container_failure_and_timeout_are_reported(self):
        cases = [
            subprocess.CompletedProcess([], 17, "SECRET_CONTENT", ""),
            subprocess.TimeoutExpired(["docker"], 60),
        ]
        for failure in cases:
            with self.subTest(failure=type(failure).__name__):
                with patch.object(verify.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, IMAGE_ID, ""), failure,
                ]):
                    with self.assertRaises(verify.ImageVerificationError) as error:
                        verify.verify_image("fixture:tag", self.local)
                self.assertNotIn("SECRET_CONTENT", str(error.exception))

    def test_cli_success_stdout_is_only_id(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(verify, "verify_image", return_value=IMAGE_ID), redirect_stdout(output), redirect_stderr(errors):
            result = verify.main(["fixture:tag"])
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), IMAGE_ID + "\n")
        self.assertIn("sources match", errors.getvalue())

    def test_cli_failure_has_rebuild_instruction_and_empty_stdout(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(verify, "verify_image", side_effect=verify.ImageVerificationError("outdated")), redirect_stdout(output), redirect_stderr(errors):
            result = verify.main(["fixture:tag"])
        self.assertEqual(result, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("outdated", errors.getvalue())
        self.assertIn("Rebuild", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
