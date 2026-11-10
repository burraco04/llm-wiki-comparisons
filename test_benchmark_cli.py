"""Offline end-to-end orchestration: real exports, fake data/model executions."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_llm_wiki as bench


DOCS = [{"id": "doc", "title": "Document", "text": "Original source text.", "meta": {}}]
QUESTIONS = [{"qa_id": "q1", "question": "What happened?", "reference": "A result",
              "question_type": "inference_query", "evidence_titles": ["Document"]}]


class SilentClient:
    def __init__(self, *args, **kwargs):
        pass

    def flush_events(self):
        pass


def dataset(top_docs, cap, *, revision=None, metadata=None):
    metadata.update({"requested_revision": revision, "test_input": True})
    return copy.deepcopy(DOCS), copy.deepcopy(QUESTIONS)


def direct(llm, docs, questions, run_dir, *, progress=None):
    result = {"answers": {}, "systems": {}, "ingestion": [{"seconds_wall": 1, "llm_calls": 1}],
              "indexing": {"raw_seconds": .1, "wiki_seconds": .2}}
    for name in bench.DIRECT_SYSTEMS:
        result["answers"][name] = [{**questions[0], "system": name, "prediction": "A result", "error": None,
                                     "em": 1., "token_f1": 1., "rouge_l": 1., "seconds_wall": .1,
                                     "llm_calls": 1, "input_tokens": 10, "output_tokens": 2}]
        result["systems"][name] = {"status": "complete"}
        progress({"component": name, "state": "complete", "completed_questions": 1})
    return result


def note_wiki(model, docs, questions, run_dir, prompt_dir, *, mode, progress=None):
    system = f"llmwiki2_{mode}"
    progress({"component": system, "state": "complete", "completed_questions": 1})
    usage = {"llm": {"calls": 1, "input_tokens": 10, "output_tokens": 2},
             "classifier": {"calls": int(mode == "crow"), "input_tokens": 5 if mode == "crow" else 0,
                            "output_tokens": 1 if mode == "crow" else 0}}
    return {"status": "complete", "mode": mode, "knowledge_frozen": True, "usage_total": usage,
            "answers": {system: [{**questions[0], "error": None, "usage": usage, "em": 1.,
                                  "token_f1": 1., "rouge_l": 1., "seconds_wall": .1}]},
            "ingestion": [{"seconds_wall": 2, "usage": usage}],
            "configuration": {"library": {"module_version": "test", "source_hashes": {"config.py": "same"}},
                              "configuration": {"mode": mode, "bundle": str(run_dir / system),
                                                "usage_log": str(run_dir / system / "usage.jsonl"),
                                                "summarize": True, "crow": {"tau_route": .6}},
                              "prompts_fingerprint": "same", "feature_flags": {"summarize": True}}}


class CLITests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent / "tmp" / "cli_tests"
        root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:1/v1"}).start()
        patch.object(bench, "make_dataset", side_effect=dataset).start()
        patch.object(bench, "VLLM", SilentClient).start()
        patch.object(bench, "run_structured_wiki", side_effect=direct).start()
        self.library = patch.object(bench, "run_llmwiki2", side_effect=note_wiki).start()
        patch.object(bench, "runtime_snapshot", return_value={"scope": "offline test"}).start()
        patch.object(bench, "collect_package_provenance", return_value={"test": True}).start()

    def run_cli(self, *extra):
        result = bench.main(["--model", "test-model", "--output", str(self.directory), *extra])
        runs = list((self.directory / bench.BENCHMARK_VERSION / "test-model").glob("*/*"))
        self.assertEqual(len(runs), 1)
        return result, runs[0]

    def read(self, run, name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    def test_default_cli_exports_all_six_and_verified_pair(self):
        result, run = self.run_cli("--dataset-revision", "fixed-revision")
        self.assertEqual(result, 0)
        manifest = self.read(run, "benchmark_manifest.json")
        self.assertEqual(manifest["wiki_modes"], ["classic", "crow"])
        self.assertEqual(manifest["dataset"]["requested_revision"], "fixed-revision")
        self.assertEqual([call.kwargs["mode"] for call in self.library.call_args_list], ["classic", "crow"])
        summaries = self.read(run, "summary.json")
        self.assertEqual(len(summaries), 6)
        self.assertEqual({row["system"] for row in summaries}, set(manifest["expected_systems"]))
        self.assertTrue(all(row["eligible_for_complete_comparison"] for row in summaries))
        self.assertTrue(self.read(run, "paired_comparison.json")["paired_comparison_complete"])
        self.assertTrue(self.read(run, "run_status.json")["all_systems_complete"])
        for name in ("full_results.json", "summary.csv", "construction_summary.json",
                     "summary_by_type.json", "summary_by_support_count.json", "dataset/statistics.json",
                     "dataset/documents.jsonl", "prompts/direct_prompts.json"):
            self.assertTrue((run / name).is_file(), name)

    def test_one_mode_failure_keeps_other_results_and_null_missing_accuracy(self):
        def fail_classic(*args, mode, **kwargs):
            if mode == "classic":
                raise RuntimeError("classic initialization failed")
            return note_wiki(*args, mode=mode, **kwargs)
        self.library.side_effect = fail_classic
        result, run = self.run_cli()
        self.assertEqual(result, 2)
        rows = {row["system"]: row for row in self.read(run, "summary.json")}
        self.assertIsNone(rows["llmwiki2_classic"]["em"])
        self.assertEqual(rows["llmwiki2_classic"]["status"], "failed")
        self.assertEqual(rows["llmwiki2_crow"]["questions"], 1)
        self.assertEqual(rows["wiki_hybrid"]["em"], 1)
        self.assertFalse(self.read(run, "run_status.json")["all_systems_complete"])
        self.assertFalse(self.read(run, "paired_comparison.json")["paired_comparison_complete"])

    def test_single_mode_is_explicit_diagnostic_not_a_complete_pair(self):
        result, run = self.run_cli("--wiki-modes", "crow")
        self.assertEqual(result, 0)
        self.assertEqual(len(self.read(run, "summary.json")), 5)
        self.assertFalse(self.read(run, "paired_comparison.json")["paired_comparison_complete"])


if __name__ == "__main__":
    unittest.main()
