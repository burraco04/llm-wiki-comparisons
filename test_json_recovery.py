"""Offline output-limit recovery regressions using a fake chat endpoint."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest

from openai import APIConnectionError

import benchmark_llm_wiki as bench
from test_benchmark_llm_wiki import DOCS, extraction


def reply(value, *, finish="stop", prompt_tokens=2000, completion_tokens=7):
    raw = value if isinstance(value, str) else json.dumps(value)
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason=finish)],
    )


def fake_client(replies, *, context_tokens=32768):
    client = bench.VLLM("fake-model", "http://localhost:1/v1", "EMPTY", 1,
                        context_tokens=context_tokens)
    pending = iter(replies)
    requests = []

    def create(**request):
        requests.append(copy.deepcopy(request))
        result = next(pending)
        if isinstance(result, Exception):
            raise result
        return result

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, requests


class JSONRecoveryTests(unittest.TestCase):
    def test_truncated_output_grows_budget_and_accounts_for_every_attempt(self):
        client, requests = fake_client([
            reply('{"summary": "partial', finish="length", completion_tokens=4096),
            reply('{"summary": "still partial', finish="length", completion_tokens=8192),
            reply(extraction(), completion_tokens=1500),
        ])

        result = client.ask_json("Original full source", bench.INGEST_SCHEMA, 4096, "ingest_extract")

        self.assertEqual(result, extraction())
        self.assertEqual([request["max_tokens"] for request in requests], [4096, 8192, 16384])
        self.assertEqual(client.snapshot(), (3, 6000, 4096 + 8192 + 1500))
        self.assertEqual(client.json_failures, 2)
        self.assertEqual(len(client.events), 3)
        self.assertTrue(all("json_attempt" in event and "json_validation" in event
                            for event in client.events))
        self.assertIn("json_retry", client.events[0])
        self.assertIn("json_retry", client.events[1])
        self.assertTrue(all("Original full source" in request["messages"][0]["content"]
                            for request in requests))

    def test_malformed_stop_retries_at_same_budget_and_fails_closed(self):
        client, requests = fake_client([reply('{"summary": "partial'), reply({})])

        with self.assertRaises(bench.JSONGenerationError) as caught:
            client.ask_json("source", bench.INGEST_SCHEMA, 4096, "ingest_extract")

        self.assertIsInstance(caught.exception, ValueError)
        self.assertFalse(caught.exception.truncated)
        self.assertEqual([request["max_tokens"] for request in requests], [4096, 4096])
        self.assertEqual(client.json_failures, 2)

    def test_retry_clips_to_available_context_then_exhausts_explicitly(self):
        client, requests = fake_client([
            reply('{"summary": "partial', finish="length", completion_tokens=4096),
            reply('{"summary": "still partial', finish="length", completion_tokens=7872),
        ], context_tokens=10000)

        with self.assertRaises(bench.JSONGenerationError) as caught:
            client.ask_json("source", bench.INGEST_SCHEMA, 4096, "ingest_extract")

        self.assertTrue(caught.exception.truncated)
        self.assertEqual([request["max_tokens"] for request in requests], [4096, 7872])
        self.assertTrue(all(request["max_tokens"] + 2000 <= 10000 for request in requests))

    def test_three_truncations_exhaust_the_growth_limit(self):
        client, requests = fake_client([
            reply('{"summary": "partial', finish="length", completion_tokens=limit)
            for limit in (4096, 8192, 16384)
        ])

        with self.assertRaises(bench.JSONGenerationError) as caught:
            client.ask_json("source", bench.INGEST_SCHEMA, 4096, "ingest_extract")

        self.assertTrue(caught.exception.truncated)
        self.assertEqual([request["max_tokens"] for request in requests], [4096, 8192, 16384])
        self.assertEqual(client.json_failures, 3)

    def test_closed_schema_valid_object_is_accepted_after_length_finish(self):
        expected = extraction()
        client, requests = fake_client([reply(json.dumps(expected) + "\n\t ", finish="length")])

        self.assertEqual(client.ask_json("source", bench.INGEST_SCHEMA, 4096, "ingest_extract"), expected)
        self.assertEqual(len(requests), 1)
        self.assertEqual(client.json_failures, 0)

    def test_transport_failure_propagates_without_json_retry(self):
        failure = APIConnectionError(request=SimpleNamespace(
            method="POST", url="http://localhost:1/v1/chat/completions"))
        client, requests = fake_client([failure])

        with self.assertRaises(APIConnectionError) as caught:
            client.ask_json("source", bench.INGEST_SCHEMA, 4096, "ingest_extract")

        self.assertIs(caught.exception, failure)
        self.assertEqual(len(requests), 1)
        self.assertEqual(client.json_failures, 0)
        self.assertEqual(client.events[0]["status"], "failed")

    def test_sectioned_recovery_builds_valid_source_nodes_and_graph(self):
        expected = extraction()
        source = copy.deepcopy(DOCS[0])
        source["text"] += " SOURCE_FULL_TEXT_SENTINEL"
        source["reference"] = "PRIVATE_ANSWER_SENTINEL"
        facts = {key: expected[key] for key in ("summary", "key_points", "facts_and_data")}
        nodes = {key: expected[key] for key in ("entities", "concepts")}
        relations = {"relationships": expected["relationships"]}
        truncated = '{"summary": "PARTIAL_OUTPUT_SENTINEL'
        client, requests = fake_client([
            reply(truncated, finish="length", completion_tokens=limit)
            for limit in (4096, 8192, 16384)
        ] + [reply(facts), reply(nodes), reply(relations)])
        wiki = bench.StructuredWiki()

        row = wiki.ingest(client, source)

        self.assertEqual(row["extraction_mode"], "sectioned_after_truncation")
        self.assertEqual(row["llm_calls"], 6)
        self.assertEqual([event["stage"] for event in client.events], [
            "ingest_extract", "ingest_extract", "ingest_extract",
            "ingest_extract_facts", "ingest_extract_nodes", "ingest_extract_relations",
        ])
        self.assertEqual(wiki.sources["sources/article-one"]["extracted"], expected)
        self.assertEqual(wiki.sources["sources/article-one"]["text"], source["text"])
        self.assertEqual(set(wiki.nodes), {"acme", "retrieval"})
        self.assertEqual(wiki.edges, [{"source": "acme", "target": "retrieval",
                                      "relation": "develops", "source_doc": "sources/article-one"}])
        self.assertIn("Acme develops systems", wiki.pages["sources/article-one"]["text"])
        self.assertNotIn("PARTIAL_OUTPUT_SENTINEL", json.dumps(wiki.sources))
        for request in requests:
            prompt = request["messages"][0]["content"]
            self.assertIn(source["text"], prompt)
            self.assertNotIn("PRIVATE_ANSWER_SENTINEL", prompt)
            self.assertNotIn("PARTIAL_OUTPUT_SENTINEL", prompt)
        relation_prompt = requests[-1]["messages"][0]["content"]
        for node_id in wiki.nodes:
            self.assertIn(node_id, relation_prompt)

    def test_successful_single_extraction_is_recorded_without_recovery(self):
        client, requests = fake_client([reply(extraction())])

        row = bench.StructuredWiki().ingest(client, copy.deepcopy(DOCS[0]))

        self.assertEqual(row["extraction_mode"], "single")
        self.assertEqual(len(requests), 1)

    def test_malformed_nontruncated_extraction_does_not_start_section_recovery(self):
        client, requests = fake_client([reply({}), reply({})])
        wiki = bench.StructuredWiki()

        with self.assertRaises(bench.JSONGenerationError) as caught:
            wiki.ingest(client, copy.deepcopy(DOCS[0]))

        self.assertFalse(caught.exception.truncated)
        self.assertEqual(len(requests), 2)
        self.assertEqual(wiki.sources, {})
        self.assertEqual(wiki.nodes, {})
        self.assertEqual(wiki.edges, [])

    def test_invalid_section_aborts_without_committing_a_source_only_wiki(self):
        expected = extraction()
        facts = {key: expected[key] for key in ("summary", "key_points", "facts_and_data")}
        client, requests = fake_client([
            reply('{"summary": "partial', finish="length", completion_tokens=limit)
            for limit in (4096, 8192, 16384)
        ] + [reply(facts), reply({}), reply({})])
        wiki = bench.StructuredWiki()

        with self.assertRaises(bench.JSONGenerationError):
            wiki.ingest(client, copy.deepcopy(DOCS[0]))

        self.assertEqual(len(requests), 6)
        self.assertEqual(wiki.sources, {})
        self.assertEqual(wiki.nodes, {})
        self.assertEqual(wiki.edges, [])


if __name__ == "__main__":
    unittest.main()
