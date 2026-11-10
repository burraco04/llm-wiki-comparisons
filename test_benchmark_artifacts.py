"""Artifact regressions using recorded-like rows; no model, GPU or network calls."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import benchmark_artifacts as artifacts


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        # Work inside the workspace so these offline checks also work in sandboxes.
        self.directory = tempfile.TemporaryDirectory(
            prefix="artifact-test-", dir=Path(__file__).resolve().parent,
        )
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_failed_question_keeps_denominator_and_separate_usage_totals(self):
        rows = [
            {
                "em": 1.0, "token_f1": 0.8, "rouge_l": 0.6, "seconds_wall": 4.0,
                "question_type": "inference_query", "evidence_titles": ["a", "a", "b"],
                "usage": {
                    "llm": {"calls": 2, "input_tokens": 10, "output_tokens": 4},
                    "classifier": {"calls": 3, "input_tokens": 20, "output_tokens": 2},
                },
                "failed_requests": 0,
            },
            {
                "em": 0.0, "token_f1": 0.0, "rouge_l": 0.0, "seconds_wall": 2.0,
                "question_type": "comparison_query", "evidence_titles": ["a", "b", "c"],
                "error": "request timed out", "error_type": "TimeoutError",
                "llm_calls": 1, "input_tokens": 5, "output_tokens": 0,
                "classifier_calls": 1, "classifier_input_tokens": 7,
                "classifier_output_tokens": 0, "failed_requests": 1,
            },
        ]
        result = artifacts.summarize_rows(rows)
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["metric_denominator"], 2)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["successful_questions"], 1)
        self.assertEqual(result["error_types"], {"TimeoutError": 1})
        self.assertEqual(result["failed_requests"], 1)
        self.assertTrue(result["failed_requests_recorded"])
        self.assertEqual(result["em"], 0.5)
        self.assertEqual(result["token_f1"], 0.4)
        self.assertEqual(result["seconds_wall"], 3.0)
        self.assertEqual(result["totals"]["llm_calls"], 3)
        self.assertEqual(result["totals"]["classifier_calls"], 4)
        self.assertEqual(result["totals"]["input_tokens"], 15)
        self.assertEqual(result["totals"]["classifier_input_tokens"], 27)
        self.assertEqual(result["totals"]["total_input_tokens"], 42)
        self.assertEqual(result["totals"]["total_output_tokens"], 6)
        self.assertEqual(result["total_input_tokens"], 21)
        groups = artifacts.summary_by_group(rows, "supporting_documents")
        self.assertEqual(groups["2"]["questions"], 1)
        self.assertEqual(groups["3"]["errors"], 1)
        self.assertEqual(artifacts.summary_by_group(rows, "question_type")["inference_query"]["em"], 1)
        # Aggregation does not rewrite the caller's measured row.
        self.assertNotIn("total_input_tokens", rows[0])

    def test_unknown_request_failures_and_empty_means_are_explicit(self):
        result = artifacts.summarize_rows([{"em": 1}])
        self.assertIsNone(result["failed_requests"])
        self.assertFalse(result["failed_requests_recorded"])
        self.assertEqual(result["available_values"]["seconds_wall"], 0)
        empty = artifacts.summarize_rows([])
        self.assertEqual(empty["questions"], 0)
        self.assertEqual(empty["metric_denominator"], 0)
        self.assertEqual(empty["em"], 0)
        self.assertEqual(empty["totals"]["total_input_tokens"], 0)

    def test_nested_error_types_are_counted_with_explicit_override_precedence(self):
        rows = [
            {"error": {"type": "TimeoutError", "message": "wait exceeded"}},
            {"error": {"type": "TimeoutError"}, "error_type": "TransportError"},
            {"exception": {"type": "ValueError", "message": "bad reply"}},
            {"error": {"message": "unknown failure"}},
        ]
        result = artifacts.summarize_rows(rows)
        self.assertEqual(result["errors"], 4)
        self.assertEqual(result["metric_denominator"], 4)
        self.assertEqual(result["error_types"], {
            "TimeoutError": 1, "TransportError": 1, "ValueError": 1, "unspecified": 1,
        })

    def test_construction_excludes_query_measurements_and_non_time_index_counts(self):
        ingestion = [
            {
                "seconds_wall": 3, "llm_calls": 2, "input_tokens": 12, "output_tokens": 6,
                "timing": {"extract": 2, "write": 1},
            },
            {
                "seconds_wall": 5, "error": "ingestion failed",
                "usage": {
                    "llm": {"calls": 1, "input_tokens": 5, "output_tokens": 1},
                    "classifier": {"calls": 4, "input_tokens": 20, "output_tokens": 2},
                },
                "timing": {"extract": 4, "write": 1},
            },
        ]
        result = artifacts.buildcost_summary(
            ingestion, {"raw_seconds": 1, "wiki_seconds": 2, "wiki_paragraphs": 100},
        )
        self.assertEqual(result["documents"], 2)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["ingestion_seconds_total"], 8)
        self.assertEqual(result["ingestion_seconds_mean"], 4)
        self.assertEqual(result["stage_seconds_total"], {"extract": 6, "write": 2})
        self.assertEqual(result["indexing_seconds_total"], 3)
        self.assertEqual(result["construction_seconds_total"], 11)
        self.assertEqual(result["usage"]["llm"]["calls"], 3)
        self.assertEqual(result["usage"]["classifier"]["calls"], 4)
        self.assertEqual(result["total_input_tokens"], 37)
        self.assertEqual(result["total_output_tokens"], 9)

    def test_dataset_support_counts_are_distinct_and_lengths_have_defined_units(self):
        docs = [{"text": "One two"}, {"text": "Café\nthree four"}]
        questions = [
            {"question": "Which company?", "question_type": "inference_query", "evidence_titles": ["a", "a", "b"]},
            {"question": "When?", "question_type": "temporal_query", "evidence_titles": ["a", "c"]},
        ]
        result = artifacts.dataset_statistics(docs, questions, {"documents": 609})
        self.assertEqual(result["documents"], 2)
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["supporting_document_counts"], {"2": 2})
        self.assertEqual(result["question_type_by_supporting_documents"]["inference_query"], {"2": 1})
        self.assertEqual(result["document_words"]["mean"], 2.5)
        self.assertEqual(result["document_characters"]["min"], 7)
        self.assertEqual(result["document_characters"]["max"], 15)
        self.assertEqual(result["question_words"]["mean"], 1.5)
        self.assertEqual(result["full_dataset_counts"]["documents"], 609)
        empty = artifacts.dataset_statistics([], [{"question": "Unknown"}])
        self.assertEqual(empty["supporting_document_counts"], {"unknown": 1})
        self.assertIsNone(empty["document_words"]["mean"])

    def test_saved_dataset_is_exact_and_repeat_save_does_not_duplicate_records(self):
        docs = [{"id": "one", "text": "First\nCafé", "meta": {"source": "article"}}]
        questions = [{"question": "Who?", "reference": "Answer", "evidence_titles": ["one"]}]
        artifacts.save_dataset(self.root, docs, questions, {"api_key": "PRIVATE", "revision": "abc"})
        artifacts.save_dataset(self.root, docs, questions)
        records = (self.root / "dataset" / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0]), docs[0])
        self.assertEqual(json.loads((self.root / "dataset" / "questions.jsonl").read_text(encoding="utf-8")), questions[0])
        metadata = json.loads((self.root / "dataset" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["api_key"], "[REDACTED]")
        self.assertEqual(metadata["revision"], "abc")
        self.assertEqual(json.loads((self.root / "dataset" / "statistics.json").read_text(encoding="utf-8"))["questions"], 1)

    def test_prompt_snapshot_preserves_override_bytes_and_template_placeholders(self):
        source = self.root / "source"
        (source / "researcher").mkdir(parents=True)
        content = b"Question: {{question}}\r\nNotes: {{notes}}\r\n"
        (source / "researcher" / "answer.md").write_bytes(content)
        direct = {"ANSWER": "Context:\n{context}\nQuestion: {question}"}
        schemas = {"ANSWER": {"type": "object", "properties": {"answer": {"type": "string"}}}}
        result = artifacts.save_prompt_snapshot(self.root / "run", source, direct, schemas)
        saved = self.root / "run" / "prompts"
        self.assertEqual((saved / "overrides" / "researcher" / "answer.md").read_bytes(), content)
        self.assertEqual(json.loads((saved / "direct_prompts.json").read_text(encoding="utf-8")), direct)
        self.assertEqual(json.loads((saved / "schemas.json").read_text(encoding="utf-8")), schemas)
        self.assertEqual(result["overrides"][0]["bytes"], len(content))
        self.assertEqual(result["overrides"][0]["relative_path"], "researcher/answer.md")
        self.assertEqual(len(result["overrides"][0]["sha256"]), 64)

    def test_json_safe_and_atomic_failure_preserve_existing_output(self):
        @dataclasses.dataclass
        class Result:
            path: Path
            value: float

        path = self.root / "result.json"
        artifacts.write_json(path, Result(Path("wiki/note.md"), math.inf))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"path": str(Path("wiki/note.md")), "value": None})
        original = path.read_bytes()
        with patch.object(artifacts.os, "replace", side_effect=OSError("simulated replacement failure")):
            with self.assertRaises(OSError):
                artifacts.write_json(path, {"new": True})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        checkpoint = self.root / "rows.jsonl"
        artifacts.append_jsonl(checkpoint, {"answer": "Café\nsecond line"})
        artifacts.append_jsonl(checkpoint, {"answer": "two"})
        rows = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["answer"], "Café\nsecond line")

    def test_redaction_keeps_measurements_but_removes_keys_and_url_credentials(self):
        value = {
            "api_key": "PRIVATE-KEY", "nested": {"Authorization": "Bearer PRIVATE-AUTH", "token": "PRIVATE-TOKEN"},
            "input_tokens": 100, "max_tokens": 50,
            "decision": {"key": "note-a"},
            "url": "https://person:PRIVATE-PASSWORD@example.org/repo?access_token=PRIVATE-QUERY&ref=main",
        }
        result = artifacts.redact_secrets(value)
        serialized = json.dumps(result)
        self.assertNotIn("PRIVATE-", serialized)
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["max_tokens"], 50)
        self.assertEqual(result["decision"]["key"], "note-a")
        url = urlsplit(result["url"])
        self.assertEqual(url.netloc, "example.org")
        self.assertEqual(parse_qs(url.query)["access_token"], ["[REDACTED]"])
        self.assertEqual(parse_qs(url.query)["ref"], ["main"])

    def test_provenance_finds_distribution_alias_and_tracks_actual_package_sources(self):
        package = self.root / "llmw2"
        package.mkdir()
        init = package / "__init__.py"
        init.write_text('__version__ = "0.6.0"\n', encoding="utf-8")
        other = package / "config.py"
        other.write_text("tau = 0.5\n", encoding="utf-8")
        module = SimpleNamespace(__file__=str(init), __version__="0.6.0")
        distribution = SimpleNamespace(
            metadata={"Name": "llmw2"}, version="0.6.0",
            read_text=lambda name: json.dumps({
                "url": "https://person:PRIVATE@example.org/repo?token=PRIVATE",
                "vcs_info": {"commit_id": "abc", "vcs": "git"},
            }) if name == "direct_url.json" else None,
        )

        def find(name):
            if name == "llmw2":
                return distribution
            raise artifacts.metadata.PackageNotFoundError(name)

        with patch.object(artifacts.metadata, "distribution", side_effect=find), patch.dict(sys.modules, {"llmw2": module}):
            first = artifacts.collect_package_provenance(["llmwiki2"])
            other.write_text("tau = 0.6\n", encoding="utf-8")
            second = artifacts.collect_package_provenance(["llmwiki2"])
        self.assertEqual(first["distributions"]["llmwiki2"]["status"], "not installed")
        self.assertEqual(first["distributions"]["llmw2"]["version"], "0.6.0")
        self.assertEqual(first["modules"]["llmw2"]["module_version"], "0.6.0")
        self.assertEqual(first["modules"]["llmw2"]["source_files"], 2)
        self.assertNotEqual(first["modules"]["llmw2"]["source_sha256"], second["modules"]["llmw2"]["source_sha256"])
        self.assertNotIn("PRIVATE", json.dumps(first))
        self.assertEqual(first["distributions"]["llmw2"]["direct_url"]["vcs_info"]["commit_id"], "abc")

    def test_runtime_metadata_unavailability_is_explicit_and_supplied_secrets_are_removed(self):
        path = self.root / "launcher.json"
        path.write_text(json.dumps({"host": {"cpu": "declared"}, "api_key": "PRIVATE"}), encoding="utf-8")
        with patch.dict(os.environ, {"BENCHMARK_RUNTIME_METADATA": str(path)}), patch.object(artifacts.shutil, "which", return_value=None):
            result = artifacts.runtime_snapshot()
        self.assertEqual(result["launcher_metadata"]["status"], "recorded")
        self.assertEqual(result["launcher_metadata"]["data"]["host"]["cpu"], "declared")
        self.assertEqual(result["gpus"]["status"], "unavailable")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("peak_gpu_memory_bytes", result)
        self.assertIn("kernel", result["os"])


if __name__ == "__main__":
    unittest.main()
