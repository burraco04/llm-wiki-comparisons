"""Offline regression tests; no model downloads, endpoint or reference leakage."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import benchmark_llm_wiki as bench


def extraction(entity_id="acme", description="Acme develops retrieval systems."):
    return {
        "summary": "Acme develops systems for knowledge retrieval.",
        "key_points": ["Acme builds retrieval systems."],
        "facts_and_data": ["Acme uses compiled knowledge pages."],
        "entities": [{"id": entity_id, "name": "Acme", "type": "organization", "description": description}],
        "concepts": [{"id": "retrieval", "name": "Retrieval", "description": "Retrieval selects relevant evidence."}],
        "relationships": [{"source_id": entity_id, "target_id": "retrieval", "relation": "develops"}],
    }


DOCS = [
    {"id": "article-one", "title": "Article One", "text": "Acme develops retrieval systems with compiled knowledge pages.", "meta": {}},
    {"id": "article-two", "title": "Article Two", "text": "Acme Corporation also uses retrieval across multiple source documents.", "meta": {}},
]
RECONCILIATION = {"merge_ids": [{"new_id": "acme-corp", "existing_id": "acme"}], "conflicts": []}
ROUTING = {"source_pages": ["sources/article-one"], "entity_pages": ["entities/acme"], "concept_pages": ["concepts/retrieval"]}


def client_with_replies(replies):
    client = bench.VLLM("fake-model", "http://localhost:1/v1", "EMPTY", 1)
    queue = iter(replies)

    def create(**kwargs):
        value = next(queue)
        if isinstance(value, Exception):
            raise value
        text = value if isinstance(value, str) else json.dumps(value, indent=2)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


class FakeEncoder:
    def __init__(self, *args):
        pass

    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts])


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).parent / "tmp" / "benchmark_tests"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def test_multiline_response_and_json_are_preserved(self):
        llm = client_with_replies(["First line\nSecond line", extraction()])
        self.assertEqual(llm.ask("test"), "First line\nSecond line")
        data = llm.ask_json("test", bench.INGEST_SCHEMA, 4096, "ingest")
        self.assertEqual(data["entities"][0]["name"], "Acme")
        self.assertEqual(llm.snapshot(), (2, 20, 10))

    def test_invalid_json_retries_and_then_fails_explicitly(self):
        llm = client_with_replies(['{"summary": "truncated', extraction()])
        self.assertEqual(llm.ask_json("test", bench.INGEST_SCHEMA, 4096, "ingest")["entities"][0]["id"], "acme")
        self.assertEqual(llm.json_failures, 1)
        llm = client_with_replies([{}, {}])
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            bench.StructuredWiki().ingest(llm, DOCS[0])

    def test_merge_provenance_valid_edges_and_backlinks(self):
        second = extraction("acme-corp", "Acme also retrieves evidence from multiple documents.")
        second["relationships"].append({"source_id": "missing", "target_id": "retrieval", "relation": "invented"})
        llm = client_with_replies([extraction(), second, RECONCILIATION])
        wiki = bench.StructuredWiki()
        wiki.ingest(llm, DOCS[0])
        row = wiki.ingest(llm, DOCS[1])
        self.assertEqual(wiki.stats()["entities"], 1)
        self.assertEqual(wiki.stats()["concepts"], 1)
        self.assertEqual(wiki.nodes["acme"]["sources"], ["sources/article-one", "sources/article-two"])
        self.assertEqual(len(wiki.nodes["acme"]["claims"]), 2)
        self.assertEqual(row["rejected_relations"], 1)
        self.assertTrue(all(e["source"] in wiki.nodes and e["target"] in wiki.nodes for e in wiki.edges))
        for pid, page in wiki.pages.items():
            for target in page["links"]:
                self.assertIn(pid, wiki.pages[target]["links"])
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            wiki.save(Path(directory))
            saved = Path(directory, "wiki/entities/acme.md").read_text(encoding="utf-8")
            self.assertIn("sources/article-two", saved)
            self.assertIn("multiple documents", saved)
            self.assertTrue(Path(directory, "wiki/knowledge_graph.json").exists())

    def test_unknown_routes_fall_back_without_invented_pages(self):
        llm = client_with_replies([extraction(), {"source_pages": ["invented"], "entity_pages": [], "concept_pages": []}, {"search_terms": ["Acme"]}])
        wiki = bench.StructuredWiki()
        wiki.ingest(llm, DOCS[0])
        ids, terms, diag = bench.prepare_query(llm, "Which systems does Acme develop?", wiki)
        self.assertTrue(diag["routing_fallback"])
        self.assertTrue(ids)
        self.assertTrue(all(pid in wiki.pages for pid in ids))
        self.assertIn("acme", terms)

    def test_full_context_budget_includes_graph_and_labels(self):
        wiki = bench.StructuredWiki()
        wiki.ingest(client_with_replies([extraction()]), DOCS[0])
        hits = [{"title": "entities/acme", "page_id": "entities/acme", "text": "Evidence. " * 200, "branch": "wiki"}]
        context, diag = bench.pack_wiki_context(wiki, hits, 1, budget=400)
        self.assertLessEqual(len(context), 400)
        self.assertEqual(diag["chars"], len(context))
        self.assertIn("Acme [develops] Retrieval", context)
        self.assertEqual(diag["kg_triples"], 1)
        self.assertLessEqual(len(bench.context_from_hits(hits, budget=100)), 100)

    def test_all_strategies_run_with_frozen_wiki_and_correct_call_counts(self):
        replies = [extraction(), extraction("acme-corp"), RECONCILIATION, "Acme"]
        for _ in range(3):
            replies += [copy.deepcopy(ROUTING), {"search_terms": ["Acme", "retrieval"]}, "Acme"]
        llm = client_with_replies(replies)
        questions = [{"qa_id": "q1", "question": "Who develops retrieval systems?", "reference": "Acme"}]
        fake_module = SimpleNamespace(SentenceTransformer=FakeEncoder)
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory, patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            output = bench.run_structured_wiki(llm, DOCS, questions, Path(directory))
            self.assertFalse(output["writeback_enabled"])
            self.assertEqual(output["wiki_stats"]["entities"], 1)
            self.assertEqual(output["answers"]["rag_semantic"][0]["llm_calls"], 1)
            for name in ("wiki_routed", "wiki_bm25", "wiki_hybrid"):
                row = output["answers"][name][0]
                self.assertEqual(row["llm_calls"], 3)
                self.assertEqual(row["em"], 1)
                self.assertGreater(row["retrieval"]["selected"], 0)
            self.assertEqual(len(Path(directory, "structured_wiki", "answers.jsonl").read_text().splitlines()), 4)
            for name, rows in output["answers"].items():
                row = rows[0]
                self.assertEqual(row["context"], row["answer_prompt"].split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0])
                self.assertTrue(row["retrieval"]["selected_passages"])
                self.assertTrue(Path(directory, "structured_wiki", "queries", name, "q1.json").is_file())

    def test_failed_structured_construction_keeps_dense_baseline_independent(self):
        llm = client_with_replies([{}, {}, "Acme"])
        questions = [{"qa_id": "q1", "question": "Who develops retrieval systems?", "reference": "Acme"}]
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory, patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=FakeEncoder)}):
            output = bench.run_structured_wiki(llm, DOCS, questions, Path(directory))
            self.assertEqual(output["construction_status"], "failed")
            self.assertEqual(output["answers"]["rag_semantic"][0]["em"], 1)
            for name in ("wiki_routed", "wiki_bm25", "wiki_hybrid"):
                self.assertEqual(output["systems"][name]["status"], "failed")
                self.assertEqual(output["answers"][name], [])

    def test_reference_answers_never_enter_model_requests(self):
        replies = [extraction(), extraction("acme-corp"), RECONCILIATION, "Acme"]
        for _ in range(3):
            replies += [copy.deepcopy(ROUTING), {"search_terms": ["Acme", "retrieval"]}, "Acme"]
        llm = client_with_replies(replies)
        questions = [{"qa_id": "q1", "question": "Who develops retrieval systems?", "reference": "PRIVATE_REFERENCE_SENTINEL"}]
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory, patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=FakeEncoder)}):
            bench.run_structured_wiki(llm, DOCS, questions, Path(directory))
        for event in llm.events:
            self.assertNotIn("PRIVATE_REFERENCE_SENTINEL", json.dumps(event["request"]))
        self.assertTrue(any(event.get("item_id") == "q1" for event in llm.events))

    def test_classifier_usage_is_included_in_summary(self):
        row = {"usage": {"llm": {"calls": 1, "input_tokens": 100, "output_tokens": 5},
                         "classifier": {"calls": 3, "input_tokens": 200, "output_tokens": 2}}}
        summary = bench.summarize([row])
        self.assertEqual(summary["llm_calls"], 1)
        self.assertEqual(summary["classifier_calls"], 3)
        self.assertEqual(summary["total_input_tokens"], 300)
        self.assertEqual(summary["total_output_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
