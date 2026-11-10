"""Offline tests of the note-wiki benchmark orchestration and saved evidence."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_llm_wiki as bench
import llmwiki_adapter


DOCS = [
    {"id": "source-a", "title": "Source A", "text": "Original evidence A", "meta": {}},
    {"id": "source-b", "title": "Source B", "text": "Original evidence B", "meta": {}},
]
QUESTIONS = [
    {"qa_id": "q-1", "question": "First question?", "reference": "correct", "question_type": "inference_query", "evidence_titles": ["Source A", "Source B"]},
    {"qa_id": "q-2", "question": "Second question?", "reference": "correct", "question_type": "comparison_query", "evidence_titles": ["Source A", "Source B"]},
]
EXACT_CONTEXT = "Question-specific context sent to the real answering API."


def usage():
    return {role: {"calls": 0, "input_tokens": 0, "output_tokens": 0} for role in ("llm", "classifier")}


class FakeRecorder:
    def __init__(self, path, mode):
        self.path, self.mode = path, mode
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        self.events = []
        self.scope = {}
        self.operations = []
        self.trace_io_seconds = 0.0
        self.closed = False

    @contextmanager
    def operation(self, phase, item_id):
        previous = self.scope
        self.scope = {"phase": phase, "item_id": item_id}
        self.operations.append(dict(self.scope))
        try:
            yield
        finally:
            self.scope = previous

    def mark(self):
        return len(self.events)

    def since(self, mark):
        return copy.deepcopy(self.events[mark:])

    def add(self, role, op, request, *, error=None):
        event = {"call_id": f"{self.mode}-{len(self.events)}", "event": "http_call", "role": role,
                 "op": op, **self.scope, "request": copy.deepcopy(request),
                 "response": {"content": "correct"} if error is None else {"error": error},
                 "status_code": 200 if error is None else 503, "seconds": 0.01, "error": error}
        self.events.append(event)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        self.trace_io_seconds += 0.0001

    def close(self):
        self.closed = True


class FakeResponse(SimpleNamespace):
    def to_dict(self):
        return copy.deepcopy(vars(self))


class FakeWiki:
    def __init__(self, bundle, mode, recorder, *, query_error=False, construction_error=False, mutate=False, no_notes=False):
        self.cfg = SimpleNamespace(bundle=bundle, mode=mode)
        self.recorder = recorder
        self.counts = usage()
        self.usage = SimpleNamespace(to_dict=lambda: copy.deepcopy(self.counts))
        self.ingested, self.asked = [], []
        self.query_error, self.construction_error, self.mutate = query_error, construction_error, mutate
        self.no_notes = no_notes
        self.closed = False

    def init(self):
        self.cfg.bundle.mkdir(parents=True)
        (self.cfg.bundle / "index.md").write_text("# Folder index", encoding="utf-8")

    def call(self, role, op, body, *, error=None):
        self.recorder.add(role, op, body, error=error)
        self.counts[role]["calls"] += 1
        self.counts[role]["input_tokens"] += 10
        self.counts[role]["output_tokens"] += 2

    def ingest(self, text, *, title):
        self.ingested.append((title, text))
        failed = self.construction_error and len(self.ingested) == 2
        self.call("llm", "librarian/summarize", {"messages": [{"role": "user", "content": text}]}, error={"type": "APIError"} if failed else None)
        if failed:
            raise RuntimeError("ingestion API failed")
        if self.cfg.mode == "crow":
            self.call("classifier", "route", {"state": text, "questions": {"q": "folder?"}})
        notes = self.cfg.bundle / "topic"
        notes.mkdir(exist_ok=True)
        (notes / "note.md").write_text("# Note\n\nStored note evidence.", encoding="utf-8")
        (self.cfg.bundle / "log.md").write_text(f"Created note from {title}", encoding="utf-8")
        decisions = [{"step": "route", "decider": "classifier" if self.cfg.mode == "crow" else "llm",
                      "choice": "/topic", "confidence": 0.9, "fallback": False, "scores": {"/topic": 0.9}}]
        return FakeResponse(action="created", note="topic/note.md", title=title, folder="/topic",
                            created_folders=["/topic"], related=[], decisions=decisions,
                            usage={"llm": {"calls": 1, "input_tokens": 10, "output_tokens": 2},
                                   "classifier": {"calls": int(self.cfg.mode == "crow"), "input_tokens": 10 * int(self.cfg.mode == "crow"), "output_tokens": 2 * int(self.cfg.mode == "crow")}})

    def ask(self, question):
        self.asked.append(question)
        if self.cfg.mode == "crow":
            self.call("classifier", "retrieve_note", {"state": question, "questions": {"q": "relevant?"}})
        if self.no_notes:
            # A legitimate empty selection can follow an LLM navigation call.
            self.call("llm", "researcher/navigate", {"messages": [{"role": "user", "content": question}]})
            return FakeResponse(text="The wiki has no notes relevant to this question.", notes=[], citations=[],
                                decisions=[{"step": "navigate", "decider": "llm", "choice": "select 0"}],
                                usage={"llm": {"calls": 1, "input_tokens": 10, "output_tokens": 2},
                                       "classifier": {"calls": int(self.cfg.mode == "crow"), "input_tokens": 10 * int(self.cfg.mode == "crow"), "output_tokens": 2 * int(self.cfg.mode == "crow")}})
        failed = self.query_error and len(self.asked) == 1
        self.call("llm", "researcher/answer", {"messages": [{"role": "system", "content": "Use notes."},
                  {"role": "user", "content": f"{question}\n\n{EXACT_CONTEXT}"}]},
                  error={"type": "APIError", "message": "temporary failure"} if failed else None)
        if failed:
            raise RuntimeError("query API failed")
        # Audit logging is not knowledge ingestion and must not invalidate frozen knowledge.
        with (self.cfg.bundle / "log.md").open("a", encoding="utf-8") as stream:
            stream.write("\nQuery completed")
        if self.mutate:
            (self.cfg.bundle / "topic" / "note.md").write_text("Changed knowledge after a question", encoding="utf-8")
        decisions = [{"step": "retrieve_note", "decider": "classifier" if self.cfg.mode == "crow" else "llm",
                      "choice": "topic/note.md", "confidence": 0.95, "fallback": False,
                      "scores": {"topic/note.md": 0.95}}]
        return FakeResponse(text="correct", citations=["topic/note.md"], notes=["topic/note.md"],
                            decisions=decisions, usage={"llm": {"calls": 1, "input_tokens": 10, "output_tokens": 2},
                            "classifier": {"calls": int(self.cfg.mode == "crow"), "input_tokens": 10 * int(self.cfg.mode == "crow"), "output_tokens": 2 * int(self.cfg.mode == "crow")}})

    def close(self):
        self.closed = True


class NoteWikiPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).parent / "tmp" / "pipeline_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(temporary_root.resolve()))
        self.prompt_dir = self.directory / "prompts"
        self.prompt_dir.mkdir()
        self.instances, self.creations = [], []
        self.progress = []

    def tearDown(self):
        self.temporary.cleanup()

    def factory(self, **options):
        def create(bundle, settings, events_path):
            self.creations.append((bundle, copy.deepcopy(settings), events_path))
            recorder = FakeRecorder(events_path, settings["OKF_MODE"])
            wiki = FakeWiki(bundle, settings["OKF_MODE"], recorder, **options)
            self.instances.append((wiki, recorder))
            provenance = {"mode": settings["OKF_MODE"], "configuration": {"mode": settings["OKF_MODE"], "bundle": str(bundle)},
                          "library": {"module_version": "fake-version"}, "thresholds": {"tau_route": 0.6}}
            (events_path.parent / "configuration.json").write_text(json.dumps(provenance), encoding="utf-8")
            return wiki, recorder, provenance
        return create

    def snapshots(self, wiki, paths):
        return [{"path": rel, "text": (wiki.cfg.bundle / rel).read_text(encoding="utf-8"),
                 "full_text": "Snapshot is distinct from the actual rendered HTTP context.", "sources": ["raw/source.md"]}
                for rel in paths]

    def run_mode(self, mode="crow", *, create=None, snapshot=None):
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://writer.invalid/v1"}), \
             patch.object(llmwiki_adapter, "create_wiki", side_effect=create or self.factory()), \
             patch.object(llmwiki_adapter, "note_snapshot", side_effect=snapshot or self.snapshots):
            return bench.run_llmwiki2("fake-writer", copy.deepcopy(DOCS), copy.deepcopy(QUESTIONS),
                                     self.directory, self.prompt_dir, mode=mode, progress=self.progress.append)

    def query_artifact(self, mode, question="q-1"):
        return json.loads((self.directory / f"llmwiki2_{mode}" / "queries" / f"{question}.json").read_text(encoding="utf-8"))

    def test_classic_and_crow_use_independent_bundles_and_identical_inputs(self):
        original_environment = dict(os.environ)
        classic = self.run_mode("classic")
        crow = self.run_mode("crow")
        self.assertEqual(dict(os.environ), original_environment)
        self.assertEqual([entry[1]["OKF_MODE"] for entry in self.creations], ["classic", "crow"])
        self.assertNotEqual(self.creations[0][0], self.creations[1][0])
        self.assertNotEqual(self.creations[0][1]["OKF_USAGE_LOG"], self.creations[1][1]["OKF_USAGE_LOG"])
        self.assertEqual(self.creations[0][1]["OKF_PROMPTS_DIR"], self.creations[1][1]["OKF_PROMPTS_DIR"])
        ignored = {"OKF_MODE", "OKF_USAGE_LOG"}
        self.assertEqual({k: v for k, v in self.creations[0][1].items() if k not in ignored},
                         {k: v for k, v in self.creations[1][1].items() if k not in ignored})
        self.assertEqual(self.instances[0][0].ingested, self.instances[1][0].ingested)
        self.assertEqual(self.instances[0][0].asked, self.instances[1][0].asked)
        for result, mode in ((classic, "classic"), (crow, "crow")):
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["knowledge_frozen"])
            self.assertFalse(result["writeback_enabled"])
            self.assertEqual(result["evaluated_questions"], len(QUESTIONS))
            self.assertEqual(result["configuration"]["mode"], mode)
            self.assertTrue((self.directory / f"llmwiki2_{mode}" / "results.json").is_file())
        self.assertEqual(classic["usage_total"]["classifier"]["calls"], 0)
        self.assertGreater(crow["usage_total"]["classifier"]["calls"], 0)
        self.assertTrue(all(recorder.closed for _, recorder in self.instances))

    def test_query_error_keeps_full_denominator_and_the_following_query(self):
        result = self.run_mode(create=self.factory(query_error=True))
        rows = result["answers"]["llmwiki2_crow"]
        self.assertEqual(result["status"], "complete_with_query_errors")
        self.assertEqual((result["expected_questions"], result["evaluated_questions"], result["query_errors"]), (2, 2, 1))
        self.assertEqual([row["qa_id"] for row in rows], ["q-1", "q-2"])
        self.assertEqual([row["em"] for row in rows], [0, 1])
        self.assertEqual(bench.summarize(rows)["em"], 0.5)
        self.assertEqual((rows[0]["prediction"], rows[0]["raw_prediction"]), ("", ""))
        self.assertIsNotNone(rows[0]["error"])
        artifact = self.query_artifact("crow")
        self.assertEqual(artifact["answer_requests"][0]["error"]["type"], "APIError")

    def test_actual_answer_request_notes_decisions_and_usage_are_preserved(self):
        result = self.run_mode()
        row = result["answers"]["llmwiki2_crow"][0]
        artifact = self.query_artifact("crow")
        self.assertEqual(row["notes"], ["topic/note.md"])
        self.assertEqual(row["citations"], ["topic/note.md"])
        self.assertEqual(row["decisions"][0]["scores"], {"topic/note.md": 0.95})
        self.assertEqual(row["response"]["notes"], row["notes"])
        self.assertEqual(artifact["selected_note_snapshots"][0]["path"], "topic/note.md")
        request = artifact["answer_requests"][0]["request"]
        self.assertEqual(request["messages"][-1]["content"], f"{QUESTIONS[0]['question']}\n\n{EXACT_CONTEXT}")
        self.assertNotIn("Snapshot is distinct", request["messages"][-1]["content"])
        self.assertEqual([call["role"] for call in artifact["model_calls"]], ["classifier", "llm"])
        self.assertEqual(row["model_call_ids"], [call["call_id"] for call in artifact["model_calls"]])
        self.assertEqual(row["usage"]["llm"]["calls"], 1)
        self.assertGreater(row["trace_io_seconds"], 0)
        self.assertEqual(result["ingestion"][0]["response"]["folder"], "/topic")
        self.assertEqual(result["ingestion"][0]["decisions"][0]["step"], "route")

    def test_construction_failure_retains_partial_artifacts_and_does_not_ask(self):
        result = self.run_mode(create=self.factory(construction_error=True))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["construction_status"], "failed")
        self.assertEqual((result["evaluated_questions"], result["expected_questions"]), (0, 2))
        self.assertEqual([record["status"] for record in result["ingestion"]], ["complete", "failed"])
        self.assertEqual(self.instances[0][0].asked, [])
        self.assertIsNone(result["knowledge_frozen"])
        artifact = self.directory / "llmwiki2_crow" / "ingestion" / "001-source-b.json"
        self.assertTrue(artifact.is_file())
        self.assertTrue(json.loads(artifact.read_text())["model_calls"][0]["error"])

    def test_knowledge_mutation_invalidates_the_comparison(self):
        result = self.run_mode(create=self.factory(mutate=True))
        self.assertEqual(result["status"], "invalid_knowledge_changed")
        self.assertFalse(result["knowledge_frozen"])
        self.assertEqual(result["knowledge_changed_files"], ["topic/note.md"])
        self.assertEqual(result["evaluated_questions"], 2)

    def test_snapshot_error_is_explicit_and_does_not_skip_later_queries(self):
        count = 0

        def snapshot(wiki, paths):
            nonlocal count
            count += 1
            if count == 1:
                raise OSError("snapshot could not be read")
            return self.snapshots(wiki, paths)

        result = self.run_mode(snapshot=snapshot)
        rows = result["answers"]["llmwiki2_crow"]
        self.assertEqual(result["status"], "complete_with_trace_errors")
        self.assertEqual(result["evaluated_questions"], 2)
        self.assertEqual(result["query_errors"], 0)
        self.assertEqual(result["trace_errors"], 1)
        self.assertEqual(rows[0]["trace_status"], "note_snapshot_error")
        self.assertEqual(rows[0]["trace_error"]["type"], "OSError")
        self.assertEqual(rows[1]["trace_status"], "complete")
        self.assertEqual(rows[1]["em"], 1)
        self.assertEqual(self.query_artifact("crow")["selected_note_snapshots"], [])

    def test_no_selected_notes_can_follow_navigation_without_an_answer_call(self):
        for mode in ("classic", "crow"):
            result = self.run_mode(mode, create=self.factory(no_notes=True))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["trace_errors"], 0)
            rows = result["answers"][f"llmwiki2_{mode}"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["trace_status"] == "complete" for row in rows))
            self.assertTrue(all(row["no_answer_request_reason"] for row in rows))
            artifact = self.query_artifact(mode)
            self.assertEqual(artifact["answer_requests"], [])
            self.assertIn("researcher/navigate", [call["op"] for call in artifact["model_calls"]])


if __name__ == "__main__":
    unittest.main()
