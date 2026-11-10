"""Offline adapter tests; mock transports never contact a model endpoint."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import httpx
except ImportError:
    httpx = None

import llmwiki_adapter as adapter


class Workspace:
    def setUp(self):
        root = Path(__file__).parent / "tmp" / "adapter_tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(root.resolve()))
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            recorder.close()
        self.temporary.cleanup()

    def recorder(self):
        recorder = adapter.HTTPRecorder(self.directory / "calls.jsonl", ("super-secret-key",))
        self.recorders.append(recorder)
        return recorder


@unittest.skipUnless(httpx is not None, "httpx is not installed in this environment")
class AdapterTests(Workspace, unittest.TestCase):
    def test_actual_requests_retry_responses_and_secret_redaction(self):
        attempts = []

        def handle(request):
            attempts.append(json.loads(request.content))
            return httpx.Response(400 if len(attempts) == 1 else 200, json={"attempt": len(attempts)}, headers={"x-secret": "super-secret-key"})

        recorder = self.recorder()
        with patch.object(httpx, "HTTPTransport", return_value=httpx.MockTransport(handle)):
            client = recorder.client("llm", 1)
        model = SimpleNamespace()

        def chat(messages, *, op=""):
            first = client.post("http://user:password@example.invalid/v1/chat?api_key=super-secret-key&keep=yes", json={"messages": messages, "api_key": "super-secret-key"})
            if first.status_code == 400:
                return client.post("http://example.invalid/v1/chat", json={"messages": messages + [{"role": "user", "content": "retry"}]}).json()
            return first.json()

        model.chat = chat
        recorder.bind_op(model, "chat")
        mark = recorder.mark()
        with recorder.operation("query", "q-1"):
            self.assertEqual(model.chat([{"role": "user", "content": "exact context"}], op="researcher/answer"), {"attempt": 2})
        events = recorder.since(mark)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["status_code"], 400)
        self.assertEqual(events[0]["error"]["type"], "HTTPStatusError")
        self.assertEqual(events[1]["request"]["messages"][-1]["content"], "retry")
        self.assertTrue(all(e["op"] == "researcher/answer" and e["item_id"] == "q-1" for e in events))
        text = recorder.path.read_text(encoding="utf-8")
        self.assertNotIn("super-secret-key", text)
        self.assertNotIn("user:password", text)
        self.assertNotIn("x-secret", text)
        self.assertIn("keep=yes", events[0]["url"])
        self.assertGreater(recorder.trace_io_seconds, 0)

    def test_transport_failure_is_recorded_and_reraised(self):
        def handle(request):
            raise httpx.ReadTimeout("timeout", request=request)

        recorder = self.recorder()
        with patch.object(httpx, "HTTPTransport", return_value=httpx.MockTransport(handle)):
            client = recorder.client("classifier", 1)
        with recorder.operation("ingestion", "doc-1"), self.assertRaises(httpx.ReadTimeout):
            client.post("http://example.invalid/systemone", json={"state": "source"})
        event, = recorder.since(0)
        self.assertEqual(event["role"], "classifier")
        self.assertEqual(event["phase"], "ingestion")
        self.assertEqual(event["error"]["type"], "ReadTimeout")
        self.assertIsNone(event["status_code"])

    def test_response_body_failure_keeps_the_attempt(self):
        class InterruptedBody(httpx.SyncByteStream):
            def __iter__(self):
                raise httpx.ReadError("incomplete response from http://user:password@example.invalid/v1")
                yield b""  # makes this a streaming iterator

        recorder = self.recorder()
        transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=InterruptedBody()))
        with patch.object(httpx, "HTTPTransport", return_value=transport):
            client = recorder.client("llm", 1)
        with self.assertRaises(httpx.ReadError):
            client.post("http://example.invalid/chat", json={"messages": []})
        event, = recorder.since(0)
        self.assertTrue(event["response_read_failed"])
        self.assertEqual(event["error"]["type"], "ReadError")
        self.assertNotIn("user:password", event["error"]["message"])

    def test_operation_scope_reaches_library_worker_threads(self):
        recorder = self.recorder()
        with patch.object(httpx, "HTTPTransport", return_value=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))):
            client = recorder.client("llm", 1)
        model = SimpleNamespace()
        model.chat = lambda messages, *, op="": client.post("http://example.invalid/chat", json={"messages": messages})
        recorder.bind_op(model, "chat")
        with recorder.operation("ingestion", "doc-thread"):
            worker = threading.Thread(target=lambda: model.chat([], op="librarian/summarize"))
            worker.start()
            worker.join()
        event, = recorder.since(0)
        self.assertEqual((event["item_id"], event["op"]), ("doc-thread", "librarian/summarize"))

    def test_escaped_surrogates_are_recorded_without_changing_the_response(self):
        recorder = self.recorder()
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b'{"reply":"\\ud800"}'))
        with patch.object(httpx, "HTTPTransport", return_value=transport):
            client = recorder.client("llm", 1)
        response = client.post("http://example.invalid/chat", json={"messages": []})
        self.assertEqual(response.json()["reply"], "\ud800")
        event, = recorder.since(0)
        self.assertEqual(event["response"]["reply"], "\ud800")
        self.assertIn(b"\\ud800", recorder.path.read_bytes())

    def test_initialization_failure_is_durable_and_actionable(self):
        with patch.object(adapter.importlib, "import_module", side_effect=ImportError("llmw2 unavailable")):
            with self.assertRaisesRegex(adapter.AdapterCapabilityError, "cannot initialize traceable") as caught:
                adapter.create_wiki(self.directory / "bundle", {"OKF_MODE": "classic"}, self.directory / "calls.jsonl")
        config = json.loads((self.directory / "configuration.json").read_text())
        self.assertEqual(config["status"], "initialization_error")
        self.assertEqual(caught.exception.provenance["status"], "initialization_error")
        event = json.loads((self.directory / "calls.jsonl").read_text())
        self.assertEqual(event["event"], "initialization_error")

    def test_note_snapshots_preserve_content_and_reject_bundle_escape(self):
        root = self.directory / "bundle"
        path = root / "topic" / "note.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Original\n\nUnchanged body.", encoding="utf-8")
        note = SimpleNamespace(frontmatter={"sources": [{"path": "raw/source.md"}]}, full_text=lambda: "Title: Original\n\nUnchanged body.")
        wiki = SimpleNamespace(cfg=SimpleNamespace(bundle=root), store=SimpleNamespace(read=lambda rel: note))
        snapshot, = adapter.note_snapshot(wiki, ["topic/note.md"])
        self.assertEqual(snapshot["text"], path.read_text())
        self.assertEqual(snapshot["sources"], [{"path": "raw/source.md"}])
        with self.assertRaisesRegex(ValueError, "escapes"):
            adapter.note_snapshot(wiki, ["../outside.md"])


@unittest.skipUnless(httpx is not None and importlib.util.find_spec("llmw2"), "real llmw2 and httpx are not installed in this environment")
class InstalledLibraryTests(Workspace, unittest.TestCase):
    def settings(self, mode):
        return {
            "OKF_MODE": mode, "OKF_LLM_PROVIDER": "custom", "OKF_LLM_BASE_URL": "http://example.invalid/v1",
            "OKF_LLM_MODEL": "fake-writer", "OKF_LLM_API_KEY": "super-secret-key", "OKF_LLM_READ_CHARS": "10000",
            "OKF_CLASSIFIER_PROVIDER": "custom", "OKF_CLASSIFIER_BASE_URL": "http://example.invalid/v1",
            "OKF_CLASSIFIER_MODEL": "fake-classifier", "OKF_CLASSIFIER_API_KEY": "another-secret-key",
        }

    def test_real_modes_export_effective_config_without_environment_leak(self):
        original = dict(os.environ)

        def handle(request):
            if request.url.path.endswith("/systemone"):
                return httpx.Response(200, json={"answers": {"q": {"noul": 0.9}}, "usage": {"input_tokens": 5, "output_tokens": 1}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "exact answer"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})

        for mode in ("classic", "crow"):
            directory = self.directory / mode
            with patch.object(httpx, "HTTPTransport", side_effect=lambda: httpx.MockTransport(handle)):
                wiki, recorder, provenance = adapter.create_wiki(directory / "bundle", self.settings(mode), directory / "calls.jsonl")
            self.recorders.append(recorder)
            self.assertEqual(provenance["configuration"]["mode"], mode)
            self.assertEqual(provenance["configuration"]["llm"]["api_key"], "<redacted>")
            self.assertEqual(set(provenance["thresholds"]) & {"tau_route", "tau_path", "tau_ing", "tau_cons", "tau_fold", "tau_ret", "tau_link"}, {"tau_route", "tau_path", "tau_ing", "tau_cons", "tau_fold", "tau_ret", "tau_link"})
            self.assertEqual(provenance["effective_values"]["effective_read_chars"], 10000)
            self.assertIn("researcher/answer", provenance["prompts"])
            self.assertIsNotNone(provenance["library"]["module_version"])
            self.assertEqual(wiki.classifier is not None, mode == "crow")
            with recorder.operation("query", "q-real"):
                self.assertEqual(wiki.llm.chat([{"role": "user", "content": "actual prompt"}], op="researcher/answer"), "exact answer")
                if mode == "crow":
                    self.assertEqual(wiki.classifier.nouls("state", {"q": "relevant?"}, op="retrieve_note"), {"q": 0.9})
            events = recorder.since(0)
            self.assertEqual(events[0]["request"]["messages"][0]["content"], "actual prompt")
            self.assertEqual(events[0]["op"], "researcher/answer")
            self.assertEqual(len(events), 1 if mode == "classic" else 2)
            wiki.close()
            recorder.close()
        self.assertEqual(dict(os.environ), original)

    def test_real_json_retry_keeps_both_actual_prompts(self):
        from pydantic import BaseModel

        class Reply(BaseModel):
            answer: str

        calls = []

        def handle(request):
            calls.append(json.loads(request.content))
            content = "invalid JSON" if len(calls) == 1 else '{"answer":"correct"}'
            return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

        with patch.object(httpx, "HTTPTransport", side_effect=lambda: httpx.MockTransport(handle)):
            wiki, recorder, _ = adapter.create_wiki(self.directory / "bundle", self.settings("classic"), self.directory / "calls.jsonl")
        self.recorders.append(recorder)
        with recorder.operation("query", "q-retry"):
            self.assertEqual(wiki.llm.json("system", "original prompt", Reply, op="researcher/answer").answer, "correct")
        events = recorder.since(0)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event["op"] == "researcher/answer" for event in events))
        self.assertIn("previous reply", events[1]["request"]["messages"][-1]["content"].lower())
        self.assertIn("response_format", events[1]["request"])
        wiki.close()

    def test_optional_constant_module_may_move_without_blocking_tracing(self):
        original_import = adapter.importlib.import_module

        def import_module(name, *args, **kwargs):
            if name == "llmw2.agents.state":
                raise ModuleNotFoundError("optional constants moved in this release")
            return original_import(name, *args, **kwargs)

        with patch.object(adapter.importlib, "import_module", side_effect=import_module), \
             patch.object(httpx, "HTTPTransport", side_effect=lambda: httpx.MockTransport(lambda request: httpx.Response(200, json={}))):
            wiki, recorder, provenance = adapter.create_wiki(self.directory / "bundle", self.settings("classic"), self.directory / "calls.jsonl")
        self.recorders.append(recorder)
        self.assertEqual(provenance["status"], "ready")
        self.assertNotIn("agents.state.STATE_VERSION", provenance["implementation_constants"])
        self.assertEqual(provenance["implementation_constants_unavailable"]["agents.state.STATE_VERSION"]["type"], "ModuleNotFoundError")
        self.assertTrue(provenance["trace_capabilities"]["actual_http_payloads"])
        wiki.close()

    def test_real_researcher_answer_keeps_selected_note_and_exact_context(self):
        from dataclasses import replace
        from llmw2.bundle.tree import Folder, Note

        response = {"choices": [{"message": {"content": "answer [topic/note.md]"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
        with patch.object(httpx, "HTTPTransport", side_effect=lambda: httpx.MockTransport(lambda request: httpx.Response(200, json=response))):
            wiki, recorder, _ = adapter.create_wiki(self.directory / "bundle", self.settings("classic"), self.directory / "calls.jsonl")
        self.recorders.append(recorder)
        note = Note(self.directory / "bundle/topic/note.md", "topic/note.md", {"title": "Useful note", "description": "summary"}, "unique context evidence")
        root = Folder(self.directory / "bundle", "", notes=[note])
        wiki.steps = replace(wiki.steps, select=lambda context, tree, question: [note])
        with patch.object(wiki.store, "load", return_value=root), recorder.operation("query", "q-context"):
            answer = wiki.ask("What does the note say?")
        self.assertEqual(answer.notes, ["topic/note.md"])
        self.assertEqual(answer.citations, ["topic/note.md"])
        event, = recorder.since(0)
        self.assertEqual(event["op"], "researcher/answer")
        actual_prompt = event["request"]["messages"][-1]["content"]
        self.assertIn("What does the note say?", actual_prompt)
        self.assertIn("### topic/note.md", actual_prompt)
        self.assertIn(note.full_text(), actual_prompt)
        self.assertEqual(event["item_id"], "q-context")
        wiki.close()


if __name__ == "__main__":
    unittest.main()
