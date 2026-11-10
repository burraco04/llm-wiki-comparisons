"""Compare Dense RAG, a structured wiki, and independent classic/CROW note wikis.

The writing model is exposed through ``LLM_BASE_URL``.  In CROW mode llmwiki2
uses the local System One endpoint exposed by the second vLLM server through
``OKF_CLASSIFIER_BASE_URL``; no external classifier API key is required.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
import re
import string
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from openai import BadRequestError, OpenAI
from jsonschema import ValidationError, validate
from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from benchmark_artifacts import (
    append_jsonl, buildcost_summary, collect_package_provenance, json_safe,
    redact_secrets, runtime_snapshot, save_dataset, save_prompt_snapshot,
    summarize_rows, summary_by_group, write_json,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BENCHMARK_VERSION = "structured_wiki_v3"
CONTEXT_BUDGET = 12000
STEMMER = SnowballStemmer("english")
DIRECT_SYSTEMS = ("rag_semantic", "wiki_routed", "wiki_bm25", "wiki_hybrid")
WIKI_MODES = ("classic", "crow")


def error_record(exc: Exception) -> dict[str, Any]:
    return redact_secrets({"type": type(exc).__name__, "message": str(exc),
                           "step": getattr(exc, "step", None),
                           "status_code": getattr(exc, "status_code", getattr(exc, "status", None)),
                           "traceback": traceback.format_exc()})


def relative_file(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def report_progress(callback, **update) -> None:
    if callback is not None:
        callback(update)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "unknown"


def normalize(text: str) -> str:
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def score(prediction: str, reference: str) -> dict[str, float]:
    p, r = normalize(prediction), normalize(reference)
    cp, cr = Counter(p.split()), Counter(r.split())
    common = sum((cp & cr).values())
    f1 = 0.0 if not p or not r or not common else 2 * (common / len(p.split())) * (common / len(r.split())) / ((common / len(p.split())) + (common / len(r.split())))
    return {"em": float(p == r), "token_f1": f1, "rouge_l": ROUGE.score(reference, prediction)["rougeL"].fmeasure}


def evidence_titles(row: dict[str, Any]) -> list[str]:
    result = []
    for item in row.get("evidence_list", []) or []:
        title = item if isinstance(item, str) else (item.get("title") or item.get("source")) if isinstance(item, dict) else None
        if title:
            result.append(title)
    return list(dict.fromkeys(result))


def make_dataset(top_docs: int, max_questions: int, *, revision: str | None = None,
                 metadata: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    options = {"revision": revision} if revision else {}
    queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train", **options)
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train", **options)
    valid = [dict(q) for q in queries if q.get("question_type") != "null_query"]
    counts, qdocs = Counter(), {}
    for i, row in enumerate(valid):
        docs = set(evidence_titles(row))
        if docs:
            qdocs[i] = docs; counts.update(sorted(docs))
    selected = [title for title, _ in counts.most_common(top_docs)]
    selected_set = set(selected)
    text, source_metadata = defaultdict(list), {}
    for row in corpus:
        if row.get("title") in selected_set and row.get("body"):
            text[row["title"]].append(row["body"])
            source_metadata.setdefault(row["title"], {k: str(row.get(k, "")) for k in ("source", "url", "published_at")})
    docs = [{"id": slug(title), "title": title, "text": "\n\n".join(text[title]), "meta": source_metadata.get(title, {})} for title in selected]
    questions = []
    for i in qdocs:
        if qdocs[i].issubset(selected_set):
            row = valid[i]
            questions.append({"qa_id": f"mhop_{i}", "question": row["query"], "reference": str(row.get("answer", "")), "question_type": row.get("question_type", ""), "evidence_titles": evidence_titles(row)})
    if metadata is not None:
        configs = {}
        for name, data in (("MultiHopRAG", queries), ("corpus", corpus)):
            info = data.info
            checksums = getattr(info, "download_checksums", None) or {}
            resolved = sorted(set(re.findall(r"/resolve/([0-9a-f]{40})/", "\n".join(checksums))))
            configs[name] = {"rows": len(data), "fingerprint": getattr(data, "_fingerprint", None),
                             "builder_version": str(getattr(info, "version", "")),
                             "resolved_repository_revisions": resolved,
                             "download_checksums": json_safe(checksums)}
        metadata.update({"name": "yixuantt/MultiHopRAG", "requested_revision": revision,
                         "split": "train", "configurations": configs,
                         "full_counts": {"documents": len(corpus), "questions": len(queries),
                                         "question_types": dict(Counter(queries["question_type"])),
                                         "non_null_with_evidence": len(qdocs),
                                         "supporting_document_counts": dict(Counter(len(titles) for titles in qdocs.values()))},
                         "eligible_questions_before_cap": len(questions),
                         "selection": "non-null queries; documents by evidence frequency; dataset question order; no shuffle"})
    return docs, questions[:max_questions]


class VLLM:
    def __init__(self, model: str, base_url: str, api_key: str, timeout: float, events_path: Path | None = None):
        self.model, self.client = model, OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout, max_retries=0)
        self.calls = self.input_tokens = self.output_tokens = 0
        self.structured_output = True
        self.json_failures = 0
        self.events: list[dict[str, Any]] = []
        self.events_path, self._flushed = events_path, 0
        self._operation: dict[str, Any] = {}

    @contextmanager
    def operation(self, phase: str, item_id: str, system: str = "structured_wiki"):
        previous = self._operation
        self._operation = {"phase": phase, "item_id": item_id, "system": system}
        try:
            yield
        finally:
            self._operation = previous

    def flush_events(self) -> None:
        """Flush outside the measured document/query interval."""
        if self.events_path is not None:
            for event in self.events[self._flushed:]:
                append_jsonl(self.events_path, event)
        self._flushed = len(self.events)

    def ask(self, prompt: str, max_tokens: int = 96, *, response_format=None, stage: str = "answer") -> str:
        started = time.perf_counter()
        options = {"response_format": response_format} if response_format else {}
        request = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": max_tokens, **options}
        event = {"call_id": len(self.events) + 1, **self._operation, "stage": stage,
                 "request": request, "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            result = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.events.append({**event, "seconds_wall": time.perf_counter() - started,
                                "error": error_record(exc), "status": "failed"})
            raise
        usage = result.usage
        self.calls += 1; self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0; self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        choice = result.choices[0]
        self.events.append({**event, "seconds_wall": time.perf_counter() - started, "status": "complete",
                            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                            "finish_reason": choice.finish_reason,
                            "response": (choice.message.content or ""),
                            "usage": json_safe(usage)})
        return (choice.message.content or "").strip()

    def ask_json(self, prompt: str, schema: dict, max_tokens: int, stage: str) -> dict[str, Any]:
        """Validate complete JSON responses; retry once, then report the failure.

        Only an explicit server rejection of JSON-schema support disables it.
        Transport/authentication errors propagate instead of masquerading as bad JSON.
        """
        for attempt in range(2):
            response_format = {"type": "json_schema", "json_schema": {
                "name": "wiki_response", "strict": True, "schema": schema,
            }} if self.structured_output else None
            request = prompt + "\n\nReturn one JSON object matching this schema:\n" + json.dumps(schema)
            if attempt:
                request += "\nThe previous reply was invalid or incomplete. Return complete JSON only."
            try:
                raw = self.ask(request, max_tokens, response_format=response_format, stage=stage)
            except BadRequestError as exc:
                message = str(exc).lower()
                if not self.structured_output or not any(x in message for x in ("response_format", "json_schema", "structured output", "structured_outputs")):
                    raise
                self.structured_output = False
                raw = self.ask(request, max_tokens, stage=stage)
            data = parse_json_object(raw)
            try:
                validate(data, schema)
                return data
            except ValidationError:
                self.json_failures += 1
        raise ValueError(f"Invalid JSON after two attempts during {stage}")

    def snapshot(self) -> tuple[int, int, int]: return self.calls, self.input_tokens, self.output_tokens
    def delta(self, before: tuple[int, int, int]) -> dict[str, int]:
        return {"llm_calls": self.calls - before[0], "input_tokens": self.input_tokens - before[1], "output_tokens": self.output_tokens - before[2]}


ANSWER = """Answer using only the context. Give a short direct answer (entity, name, number, date, phrase, Yes/No), or unanswerable. Output only the answer.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
EXTRACT = """Extract structured knowledge from this document. Do not invent facts.
Return a 2-3 sentence summary, 3-5 key_points, 15-25 short facts_and_data,
entities (named people, organizations, places, events or products discussed in detail),
concepts (central abstract ideas), and relationships between those extracted items.
Each entity/concept has id, name and description. Entities also have type.
IDs must be lowercase hyphenated names, at most 50 characters. Reuse no example
values. Skip passing mentions. Every relationship has source_id, target_id and
relation (a short verb phrase). Only use IDs extracted from this document.
Title: {title}
Document:
{text}"""
RECONCILE = """Compare the new entities and concepts with the existing wiki nodes.
Suggest merge_ids only for the SAME real-world thing under different IDs, never
for merely related things. Each suggestion has new_id and existing_id.
Report conflicts only for incompatible factual claims, not additional details or
paraphrases. Each conflict has existing_id, existing_description, new_description.
Use empty lists when nothing needs merging or flagging. Do not resolve conflicts
by inventing a preferred version.
Existing nodes:
{existing}
New items from {title}:
{incoming}"""
ROUTE = """You are the router of a wiki knowledge base. Select only page ids that are useful for answering the question.\n\nQuestion: {question}\n\nWiki index:\n{index}\n\nValid page ids: {valid_ids}\n\nReturn ONLY JSON in this form: {{\"source_pages\": [], \"entity_pages\": [], \"concept_pages\": []}}"""
SEARCH_TERMS = """Generate 8-15 short search terms for paragraph-level keyword matching.\n\nQuestion: {question}\n\nSelected wiki pages:\n{pages}\n\nReturn ONLY JSON in this form: {{\"search_terms\": []}}"""


def object_schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def array_schema(item: dict) -> dict:
    return {"type": "array", "items": item}


TEXT = {"type": "string"}
NODE_FIELDS = {"id": TEXT, "name": TEXT, "description": TEXT}
INGEST_SCHEMA = object_schema({
    "summary": TEXT, "key_points": array_schema(TEXT), "facts_and_data": array_schema(TEXT),
    "entities": array_schema(object_schema({**NODE_FIELDS, "type": TEXT})),
    "concepts": array_schema(object_schema(NODE_FIELDS)),
    "relationships": array_schema(object_schema({"source_id": TEXT, "target_id": TEXT, "relation": TEXT})),
})
RECONCILE_SCHEMA = object_schema({
    "merge_ids": array_schema(object_schema({"new_id": TEXT, "existing_id": TEXT})),
    "conflicts": array_schema(object_schema({"existing_id": TEXT, "existing_description": TEXT, "new_description": TEXT})),
})
ROUTE_SCHEMA = object_schema({key: array_schema(TEXT) for key in ("source_pages", "entity_pages", "concept_pages")})
TERMS_SCHEMA = object_schema({"search_terms": array_schema(TEXT)})


def safe_id(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^\w-]+", "-", value.lower())).strip("-")[:50].rstrip("-")


def tokenize(text: str) -> list[str]:
    return [STEMMER.stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def code_search_terms(question: str) -> list[str]:
    words = re.findall(r"\w+", question.lower())
    return list(dict.fromkeys([w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
                             + [f"{a} {b}" for a, b in zip(words, words[1:]) if a not in ENGLISH_STOP_WORDS and b not in ENGLISH_STOP_WORDS]))


def term_score(text: str, terms: list[str]) -> tuple[int, int]:
    low, stems = text.lower(), set(tokenize(text))
    exact = sum(t in low for t in terms)
    fuzzy = sum(t in low or (bool(tokenize(t)) and set(tokenize(t)).issubset(stems)) for t in terms)
    return fuzzy, exact


def split_paragraphs(text: str, min_len: int = 30) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text)
            if len(p.strip()) >= min_len and not p.strip().startswith(("#", "- [[", "---"))]


class StructuredWiki:
    """Batch adaptation of notebook passes 1/2, deterministic writers and graph.

    No questions or reference answers are supplied at ingestion. Query-time
    write-back is deliberately disabled, as in the notebook's batch evaluation.
    Page IDs include their type so that a source and an entity cannot collide.
    """

    def __init__(self):
        self.pages: dict[str, dict[str, Any]] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []
        self.sources: dict[str, dict[str, Any]] = {}
        self.log: list[dict[str, Any]] = []
        self.aliases: dict[str, str] = {}

    def ingest(self, llm: VLLM, doc: dict[str, Any]) -> dict[str, Any]:
        started, before = time.perf_counter(), llm.snapshot()
        timing = {}
        t = time.perf_counter()
        data = llm.ask_json(EXTRACT.format(title=doc["title"], text=doc["text"]), INGEST_SCHEMA, 4096, "ingest_extract")
        timing["pass1_llm"] = time.perf_counter() - t
        # Invalid extraction aborts this wiki's construction instead of substituting a
        # source-only wiki. Raw model output is never truncated to its first line.
        if not data["summary"].strip():
            raise ValueError(f"Empty extracted summary for {doc['title']}")
        items, local_ids = [], set()
        for kind, field in (("entity", "entities"), ("concept", "concepts")):
            for value in data[field]:
                nid = safe_id(value["id"] or value["name"])
                if not nid or nid in local_ids or not value["description"].strip() or not value["name"].strip():
                    continue
                local_ids.add(nid)
                items.append({**value, "id": nid, "node_type": kind})
        rec = {"merge_ids": [], "conflicts": []}
        t = time.perf_counter()
        if self.nodes and items:
            existing = [{k: n[k] for k in ("id", "name", "node_type", "description")} for n in self.nodes.values()]
            rec = llm.ask_json(RECONCILE.format(existing=json.dumps(existing, ensure_ascii=False), title=doc["title"],
                                             incoming=json.dumps(items, ensure_ascii=False)), RECONCILE_SCHEMA, 1024, "ingest_reconcile")
        timing["pass2_llm"] = time.perf_counter() - t
        t = time.perf_counter()
        merge_map = {}
        incoming = {x["id"]: x for x in items}
        for merge in rec["merge_ids"]:
            new, old = safe_id(merge["new_id"]), safe_id(merge["existing_id"])
            if (new in incoming and old in self.nodes and new not in self.nodes and new != old
                    and incoming[new]["node_type"] == self.nodes[old]["node_type"]):
                merge_map[new] = old
        source_id = "sources/" + doc["id"]
        source = {"id": source_id, "title": doc["title"], "text": doc["text"], "meta": doc.get("meta", {}),
                  "sha256": hashlib.sha256(doc["text"].encode("utf-8")).hexdigest(), "extracted": data}
        self.sources[source_id] = source
        created, updated, resolved = [], [], {}
        for item in items:
            original = item["id"]
            nid = merge_map.get(original, self.aliases.get(original, original))
            # Same-name entity/concept classification may vary across documents;
            # retain the existing canonical type and all source-specific claims.
            if nid not in self.nodes:
                self.nodes[nid] = {"id": nid, "name": item["name"], "node_type": item["node_type"],
                                   "description": item["description"], "claims": [], "sources": [], "conflicts": []}
                created.append(nid)
            else:
                updated.append(nid)
            node = self.nodes[nid]
            claim = {"source": source_id, "description": item["description"]}
            if claim not in node["claims"]:
                node["claims"].append(claim)
            if source_id not in node["sources"]:
                node["sources"].append(source_id)
            if len(item["description"]) > len(node["description"]):
                node["description"] = item["description"]
            resolved[original] = nid
            self.aliases[original] = nid
        rejected_relations = 0
        for rel in data["relationships"]:
            a, b = safe_id(rel["source_id"]), safe_id(rel["target_id"])
            if a not in resolved or b not in resolved or not rel["relation"].strip():
                rejected_relations += 1
                continue
            edge = {"source": resolved[a], "target": resolved[b], "relation": rel["relation"].strip(), "source_doc": source_id}
            if edge["source"] == edge["target"]:
                rejected_relations += 1
            elif edge not in self.edges:
                self.edges.append(edge)
        conflicts = 0
        for conflict in rec["conflicts"]:
            nid = safe_id(conflict["existing_id"])
            a, b = conflict["existing_description"].strip(), conflict["new_description"].strip()
            # A conflict must refer to an item actually observed in this source.
            if nid in resolved.values() and a and b and normalize(a) != normalize(b):
                entry = {**conflict, "source": source_id}
                if entry not in self.nodes[nid]["conflicts"]:
                    self.nodes[nid]["conflicts"].append(entry)
                    conflicts += 1
        self.rebuild_pages()
        timing["write_code"] = time.perf_counter() - t
        row = {"document": doc["title"], "seconds_wall": time.perf_counter() - started, **llm.delta(before),
               "timing": timing, "created": created, "updated": list(dict.fromkeys(updated)),
               "merges": merge_map, "conflicts": conflicts, "rejected_relations": rejected_relations,
               "entities_extracted": sum(i["node_type"] == "entity" for i in items),
               "concepts_extracted": sum(i["node_type"] == "concept" for i in items)}
        self.log.append(row)
        return row

    def node_page_id(self, nid: str) -> str:
        return ("entities/" if self.nodes[nid]["node_type"] == "entity" else "concepts/") + nid

    def rebuild_pages(self) -> None:
        for sid, source in self.sources.items():
            data = source["extracted"]
            links = [self.node_page_id(nid) for nid, n in self.nodes.items() if sid in n["sources"]]
            body = f"# {source['title']}\n\n## Core Content\n\n{data['summary']}\n\n"
            body += "## Key Points\n\n" + "\n".join("- " + p for p in dict.fromkeys(data["key_points"]))
            body += "\n\n## Facts and Data\n\n" + "\n".join("- " + f for f in dict.fromkeys(data["facts_and_data"]))
            self.pages[sid] = {"id": sid, "title": source["title"], "type": "source", "text": body,
                               "summary": data["summary"], "sources": [sid], "links": sorted(set(links))}
        for nid, node in self.nodes.items():
            pid = self.node_page_id(nid)
            links = set(node["sources"])
            relations = []
            for edge in self.edges:
                if nid in (edge["source"], edge["target"]):
                    other = edge["target"] if nid == edge["source"] else edge["source"]
                    links.add(self.node_page_id(other))
                    relations.append(f"- {self.nodes[edge['source']]['name']} [{edge['relation']}] {self.nodes[edge['target']]['name']} (from [[{edge['source_doc']}]])")
            body = f"# {node['name']}\n\n{node['description']}\n\n## References\n\n"
            body += "\n".join(f"- From [[{c['source']}]]: {c['description']}" for c in node["claims"])
            if relations:
                body += "\n\n## Relations\n\n" + "\n".join(relations)
            if node["conflicts"]:
                body += "\n\n## Contradictions (require review)\n\n" + "\n".join(
                    f"- Existing: {c['existing_description']} | [[{c['source']}]]: {c['new_description']}" for c in node["conflicts"])
            self.pages[pid] = {"id": pid, "title": node["name"], "type": node["node_type"], "text": body,
                               "summary": node["description"], "sources": node["sources"], "links": sorted(links)}
        # Every recorded link has a reverse link; relation direction stays in the graph.
        for pid, page in list(self.pages.items()):
            for target in list(page["links"]):
                if pid not in self.pages[target]["links"]:
                    self.pages[target]["links"].append(pid)

    def index_text(self) -> str:
        return "\n".join(f"- {pid}: {p['summary'][:280]}" for pid, p in self.pages.items())

    def save(self, bundle: Path) -> None:
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "wiki").mkdir(exist_ok=True)
        raw_dir = bundle / "documents"
        raw_dir.mkdir(exist_ok=True)
        for sid, source in self.sources.items():
            (raw_dir / (sid.split("/", 1)[1] + ".md")).write_text(source["text"], encoding="utf-8")
        for pid, page in self.pages.items():
            path = bundle / "wiki" / (pid + ".md")
            path.parent.mkdir(parents=True, exist_ok=True)
            fm = {k: page[k] for k in ("title", "type", "sources")}
            if page["type"] == "source":
                fm.update({"raw_file": "../../documents/" + pid.split("/", 1)[1] + ".md", "sha256": self.sources[pid]["sha256"],
                           "metadata": self.sources[pid]["meta"]})
            header = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fm.items()) + "\n---\n\n"
            links = "\n\n## Related\n\n" + "\n".join(f"- [[{target}]]" for target in sorted(page["links"]))
            path.write_text(header + page["text"] + links + "\n", encoding="utf-8")
        graph = {"nodes": list(self.nodes.values()), "edges": self.edges, "aliases": self.aliases, "stats": self.stats()}
        (bundle / "wiki" / "knowledge_graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        (bundle / "wiki" / "index.md").write_text("# Wiki index\n\n" + self.index_text(), encoding="utf-8")
        (bundle / "wiki" / "log.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=2), encoding="utf-8")
        (bundle / "compilation.json").write_text(json.dumps(list(self.sources.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(self) -> dict[str, int]:
        return {"sources": len(self.sources), "entities": sum(n["node_type"] == "entity" for n in self.nodes.values()),
                "concepts": sum(n["node_type"] == "concept" for n in self.nodes.values()),
                "kg_edges": len(self.edges), "wiki_links": sum(len(p["links"]) for p in self.pages.values())}


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("llmwiki2", "vllm", "openai", "datasets", "sentence-transformers", "rank-bm25", "nltk", "scikit-learn", "jsonschema", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed in client environment"
    try:
        result["llmw2"] = version("llmw2")
    except PackageNotFoundError:
        result["llmw2"] = "not installed in client environment"
    return result


def paragraphs(docs: list[dict[str, Any]], words: int = 200) -> list[dict[str, Any]]:
    output = []
    for doc in docs:
        tokens = doc["text"].split()
        for start in range(0, len(tokens), words):
            output.append({"doc_id": doc["id"], "title": doc["title"],
                           "chunk_id": f"raw/{doc['id']}/{start // words}",
                           "start_word": start, "end_word": min(start + words, len(tokens)),
                           "text": " ".join(tokens[start:start + words])})
    return output


def context_from_hits(hits: list[dict[str, Any]], budget: int = 12000, *, trace: list | None = None) -> str:
    out, used = [], 0
    for hit in hits:
        block = f"[{hit['title']}]\n{hit['text']}"
        remaining = budget - used - (len("\n\n---\n\n") if out else 0)
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining > len(hit["title"]) + 20:
                out.append(block[:remaining])
                if trace is not None:
                    trace.append({**hit, "context_block": block[:remaining], "truncated": True})
            break
        used += len(block) + (len("\n\n---\n\n") if out else 0)
        out.append(block)
        if trace is not None:
            trace.append({**hit, "context_block": block, "truncated": False})
    return "\n\n---\n\n".join(out)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the small JSON objects requested by routed retrieval, tolerating fences."""
    cleaned = re.sub(r"```(?:json)?|```", "", str(text), flags=re.I).strip()
    match = re.search(r"\{", cleaned)
    if not match:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[match.start():])
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def prepare_query(llm: VLLM, question: str, wiki: StructuredWiki) -> tuple[list[str], list[str], dict]:
    """Notebook calls 1a/1b, shared by the three wiki retrieval strategies."""
    timing, fallback = {}, False
    t = time.perf_counter()
    try:
        routed = llm.ask_json(ROUTE.format(question=question, index=wiki.index_text(), valid_ids=", ".join(wiki.pages)),
                              ROUTE_SCHEMA, 512, "query_route")
    except ValueError:
        routed = {}
    ids = []
    for key, kind in (("source_pages", "source"), ("entity_pages", "entity"), ("concept_pages", "concept")):
        ids.extend(pid for pid in routed.get(key, []) if pid in wiki.pages and wiki.pages[pid]["type"] == kind)
    ids = list(dict.fromkeys(ids))
    if not ids:
        fallback = True
        terms = code_search_terms(question)
        ranked = sorted(wiki.pages, key=lambda pid: term_score(wiki.pages[pid]["text"], terms), reverse=True)
        counts = Counter()
        for pid in ranked:
            kind = wiki.pages[pid]["type"]
            if term_score(wiki.pages[pid]["text"], terms)[0] and counts[kind] < 3:
                ids.append(pid)
                counts[kind] += 1
    timing["routing_llm"] = time.perf_counter() - t
    t = time.perf_counter()
    selected = []
    for pid in ids:
        selected.append(f"[{pid}]\n{wiki.pages[pid]['text'][:2500]}")
        if pid in wiki.sources:
            selected.append(f"[raw:{pid}]\n{wiki.sources[pid]['text'][:2500]}")
    try:
        data = llm.ask_json(SEARCH_TERMS.format(question=question, pages="\n\n".join(selected)),
                           TERMS_SCHEMA, 384, "query_search_terms") if selected else {}
    except ValueError:
        data = {}
    terms = list(dict.fromkeys([t.strip().lower() for t in data.get("search_terms", []) if t.strip()]
                               + code_search_terms(question)))
    timing["search_terms_llm"] = time.perf_counter() - t
    return ids, terms, {"routed_ids": ids, "terms": terms, "routing_fallback": fallback, "timing": timing}


def wiki_items(wiki: StructuredWiki) -> list[dict[str, str]]:
    items = []
    for pid, page in wiki.pages.items():
        items.extend({"title": pid, "page_id": pid, "text": p, "branch": "wiki",
                      "item_id": f"wiki:{pid}:{i}"} for i, p in enumerate(split_paragraphs(page["text"], 20)))
    for sid, source in wiki.sources.items():
        items.extend({"title": "raw:" + sid, "page_id": sid, "text": p, "branch": "raw",
                      "item_id": f"raw:{sid}:{i}"} for i, p in enumerate(split_paragraphs(source["text"])))
    return items


def pack_wiki_context(wiki: StructuredWiki, hits: list[dict[str, str]], candidates: int, budget: int = CONTEXT_BUDGET) -> tuple[str, dict]:
    """Bound the entire context, including labels, separators and graph triples."""
    selected, matched, parts, used, seen = [], set(), [], 0, set()
    candidate_page_ids = {h["page_id"] for h in hits}
    candidate_nodes = {nid for nid in wiki.nodes if wiki.node_page_id(nid) in candidate_page_ids}
    has_edges = any(e["source"] in candidate_nodes or e["target"] in candidate_nodes for e in wiki.edges)
    passage_budget = budget - budget // 4 if has_edges else budget
    for hit in hits:
        key = (hit["title"], hit["text"])
        if key in seen:
            continue
        seen.add(key)
        block = f"[{hit['title']}]\n{hit['text']}"
        remaining = passage_budget - used - (2 if parts else 0)
        if remaining < len(hit["title"]) + 24:
            break
        parts.append(block[:remaining])
        used += len(parts[-1]) + (2 if len(parts) > 1 else 0)
        selected.append({**hit, "context_block": parts[-1], "truncated": len(block) > remaining})
        matched.add(hit["page_id"])
        if len(block) > remaining:
            break
    node_ids = {nid for nid in wiki.nodes if wiki.node_page_id(nid) in matched}
    triples, graph, graph_records = [], "", []
    for edge in wiki.edges:
        if edge["source"] not in node_ids and edge["target"] not in node_ids:
            continue
        line = f"{wiki.nodes[edge['source']]['name']} [{edge['relation']}] {wiki.nodes[edge['target']]['name']} (from {edge['source_doc']})"
        if line in triples:
            continue
        proposed = "[Knowledge graph]\n" + "\n".join(triples + [line])
        if used + 2 + len(proposed) > budget:
            continue
        triples.append(line)
        graph_records.append({**edge, "context_line": line})
        graph = proposed
        if len(triples) == 6:
            break
    context = "\n\n".join(parts)
    if graph:
        context += "\n\n" + graph
    return context, {"candidates": candidates, "selected": len(selected), "chars": len(context),
                     "matched_pages": list(dict.fromkeys(h["page_id"] for h in selected)),
                     "kg_triples": len(triples), "graph_relations": graph_records,
                     "selected_passages": selected,
                     "branches": dict(Counter(h["branch"] for h in selected))}


def routed_retrieve(wiki: StructuredWiki, ids: list[str], terms: list[str], items: list[dict[str, str]]) -> tuple[str, dict]:
    """Original routed branches, global keyword ranking and raw-text safety net."""
    other_pages = sorted((pid for pid in wiki.pages if pid not in ids),
                         key=lambda pid: term_score(wiki.pages[pid]["text"], terms), reverse=True)[:3]
    candidates = []
    for hit in items:
        fuzzy, exact = term_score(hit["text"], terms)
        routed = hit["page_id"] in ids
        if hit["branch"] == "raw":
            eligible = fuzzy > 0 if routed else fuzzy >= 2
            branch = "routed_raw" if routed else "other_raw"
        else:
            eligible = fuzzy > 0 and (routed or hit["page_id"] in other_pages)
            branch = "routed_wiki" if routed else "other_wiki"
        if eligible:
            candidates.append((fuzzy, exact, {**hit, "branch": branch,
                                               "matched_terms": fuzzy, "direct_matches": exact}))
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return pack_wiki_context(wiki, [h for _, _, h in candidates], len(candidates))


def run_structured_wiki(llm: VLLM, docs: list[dict[str, Any]], questions: list[dict[str, Any]],
                        run_dir: Path, *, progress=None) -> dict[str, Any]:
    """Build once, retain actual evidence, and evaluate the four direct configurations."""
    from sentence_transformers import SentenceTransformer

    out = run_dir / "structured_wiki"
    out.mkdir(parents=True, exist_ok=True)
    wiki, ingest = StructuredWiki(), []
    bundle = run_dir / "structured_wiki_bundle"
    construction_error = None
    report_progress(progress, component="structured_wiki_build", state="running", completed_documents=0)
    for number, doc in enumerate(docs):
        mark, before, started = len(llm.events), llm.snapshot(), time.perf_counter()
        try:
            with llm.operation("ingestion", doc["id"]):
                record = {**wiki.ingest(llm, doc), "document_id": doc["id"], "status": "complete"}
        except Exception as exc:
            construction_error = error_record(exc)
            record = {"document": doc["title"], "document_id": doc["id"], "status": "failed",
                      "seconds_wall": time.perf_counter() - started, **llm.delta(before),
                      "error": construction_error}
        record["model_call_ids"] = [e["call_id"] for e in llm.events[mark:]]
        ingest.append(record)
        # Serialization and trace flushing follow the construction timer.
        llm.flush_events()
        wiki.save(bundle)
        append_jsonl(out / "ingestion.jsonl", record)
        write_json(out / "ingestion.json", ingest)
        write_json(out / "ingestion" / f"{number:03d}-{slug(doc['id'])}.json",
                   {**record, "wiki_stats": wiki.stats(),
                    "source": wiki.sources.get("sources/" + doc["id"])})
        report_progress(progress, component="structured_wiki_build",
                        state="failed" if construction_error else "running",
                        completed_documents=sum(x["status"] == "complete" for x in ingest),
                        error=construction_error)
        if construction_error:
            break
        print(f"Ingested {doc['title']}: {wiki.stats()}", flush=True)
    if not construction_error:
        report_progress(progress, component="structured_wiki_build", state="complete",
                        completed_documents=len(docs))

    raw = paragraphs(docs)
    if not raw:
        raise ValueError("No original-source chunks; check the input corpus")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    started = time.perf_counter()
    embeddings = encoder.encode([x["text"] for x in raw], convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
    raw_index_seconds = time.perf_counter() - started
    items, wiki_emb, wiki_bm25, wiki_index_seconds = [], None, None, None
    if construction_error is None:
        started = time.perf_counter()
        items = wiki_items(wiki)
        if not items:
            construction_error = {"type": "EmptyWiki", "message": "No wiki retrieval paragraphs were produced"}
        else:
            wiki_emb = encoder.encode([x["text"] for x in items], convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
            wiki_bm25 = BM25Okapi([tokenize(x["text"]) or ["__empty__"] for x in items])
            wiki_index_seconds = time.perf_counter() - started
    indexing = {"raw_seconds": raw_index_seconds, "wiki_seconds": wiki_index_seconds,
                "wiki_paragraphs": len(items), "raw_chunks": len(raw),
                "embedding_model": EMBEDDING_MODEL,
                "encoder_max_seq_length": getattr(encoder, "max_seq_length", None),
                "normalized_embeddings": True, "raw_embedding_shape": list(embeddings.shape),
                "wiki_embedding_shape": list(wiki_emb.shape) if wiki_emb is not None else None}
    write_json(out / "retrieval_index.json", {"indexing": indexing, "raw_chunks": raw, "wiki_paragraphs": items})
    np.save(out / "raw_embeddings.npy", embeddings)
    if wiki_emb is not None:
        np.save(out / "wiki_embeddings.npy", wiki_emb)
    write_json(out / "indexing.json", indexing)

    result, systems = {}, {}
    for name in DIRECT_SYSTEMS:
        if name != "rag_semantic" and construction_error:
            result[name] = []
            systems[name] = {"status": "failed", "error": construction_error, "evaluated_questions": 0,
                             "expected_questions": len(questions)}
            report_progress(progress, component=name, state="failed", error=construction_error,
                            completed_questions=0, expected_questions=len(questions))
            continue
        rows = []
        report_progress(progress, component=name, state="running", completed_questions=0,
                        expected_questions=len(questions))
        for qa in questions:
            before, mark, started = llm.snapshot(), len(llm.events), time.perf_counter()
            diagnostics, context, prompt, prediction, raw_prediction, error = {}, "", None, "", "", None
            try:
                with llm.operation("query", qa["qa_id"], name):
                    if name == "rag_semantic":
                        query_vector = encoder.encode([qa["question"]], convert_to_numpy=True,
                                                      normalize_embeddings=True)[0]
                        similarities = embeddings @ query_vector
                        ranks = np.argsort(-similarities, kind="stable")[:3]
                        hits = [{**raw[i], "rank": number + 1, "similarity": float(similarities[i])}
                                for number, i in enumerate(ranks)]
                        selected = []
                        context = context_from_hits(hits, CONTEXT_BUDGET, trace=selected)
                        diagnostics = {"candidates": len(raw), "selected": len(selected),
                                       "chars": len(context), "top_k": 3, "selected_passages": selected}
                    else:
                        ids, terms, diagnostics = prepare_query(llm, qa["question"], wiki)
                        t = time.perf_counter()
                        if name == "wiki_routed":
                            context, retrieval = routed_retrieve(wiki, ids, terms, items)
                        else:
                            lexical = np.asarray(wiki_bm25.get_scores(tokenize(qa["question"] + " " + " ".join(terms))))
                            lexical_rank = np.argsort(-lexical, kind="stable")
                            fused = np.zeros(len(items))
                            fused[lexical_rank] += 1 / (60 + np.arange(1, len(items) + 1))
                            semantic, semantic_rank = None, None
                            if name == "wiki_hybrid":
                                query_vector = encoder.encode([qa["question"]], convert_to_numpy=True,
                                                              normalize_embeddings=True)[0]
                                semantic = wiki_emb @ query_vector
                                semantic_rank = np.argsort(-semantic, kind="stable")
                                fused[semantic_rank] += 1 / (60 + np.arange(1, len(items) + 1))
                            order = np.argsort(-fused, kind="stable")
                            lexical_positions = {int(index): rank + 1 for rank, index in enumerate(lexical_rank)}
                            semantic_positions = ({int(index): rank + 1 for rank, index in enumerate(semantic_rank)}
                                                  if semantic_rank is not None else {})
                            hits = [{**items[i], "rank": rank + 1, "bm25_score": float(lexical[i]),
                                     "bm25_rank": lexical_positions[int(i)], "rrf_score": float(fused[i]),
                                     "dense_score": float(semantic[i]) if semantic is not None else None,
                                     "dense_rank": semantic_positions.get(int(i))}
                                    for rank, i in enumerate(order)
                                    if name == "wiki_hybrid" or lexical[i] > 0]
                            context, retrieval = pack_wiki_context(wiki, hits, len(items))
                        diagnostics.update(retrieval)
                        diagnostics["timing"]["retrieval_code"] = time.perf_counter() - t
                    t = time.perf_counter()
                    prompt = ANSWER.format(context=context, question=qa["question"])
                    raw_prediction = llm.ask(prompt, max_tokens=96, stage=f"{name}_answer")
                    prediction = repair_answer(raw_prediction)
                    diagnostics.setdefault("timing", {})["answer_llm"] = time.perf_counter() - t
            except Exception as exc:
                error = error_record(exc)
            elapsed = time.perf_counter() - started
            calls = llm.events[mark:]
            row = {**qa, "system": name, "prediction": prediction, "raw_prediction": raw_prediction,
                   "seconds_wall": elapsed, "retrieval": diagnostics, "error": error,
                   "context": context, "answer_prompt": prompt,
                   "model_call_ids": [event["call_id"] for event in calls],
                   "request_attempts": len(calls), "failed_requests": sum(e.get("status") == "failed" for e in calls),
                   **llm.delta(before),
                   **(score(prediction, qa["reference"]) if error is None
                      else {"em": 0.0, "token_f1": 0.0, "rouge_l": 0.0})}
            # Every selected evaluation question remains in the denominator on a query error.
            rows.append(row)
            llm.flush_events()
            append_jsonl(out / "answers.jsonl", row)
            write_json(out / "queries" / name / f"{slug(qa['qa_id'])}.json", row)
            report_progress(progress, component=name, state="running", completed_questions=len(rows),
                            expected_questions=len(questions), query_errors=sum(bool(x["error"]) for x in rows))
        result[name] = rows
        systems[name] = {"status": "complete_with_query_errors" if any(r["error"] for r in rows) else "complete",
                         "evaluated_questions": len(rows), "expected_questions": len(questions),
                         "query_errors": sum(bool(r["error"]) for r in rows)}
        write_json(out / f"{name}.json", {"status": systems[name], "answers": rows})
        report_progress(progress, component=name, state=systems[name]["status"],
                        completed_questions=len(rows), expected_questions=len(questions),
                        query_errors=systems[name]["query_errors"])
    output = {"ingestion": ingest, "answers": result, "systems": systems, "wiki_stats": wiki.stats(),
              "construction_status": "failed" if construction_error else "complete",
              "construction_error": construction_error, "writeback_enabled": False, "indexing": indexing,
              "json_failures": llm.json_failures, "model_calls_file": relative_file(llm.events_path, run_dir)
              if llm.events_path is not None else None,
              "construction_cost": buildcost_summary(ingest, indexing),
              "model_calls": llm.events}
    write_json(out / "results.json", output)
    return output


# Compatibility for callers of the earlier runner; new artifacts use structured_wiki.
run_notebook_wiki = run_structured_wiki


def repair_answer(text: str) -> str:
    text = re.sub(r"```.*?```", "", str(text), flags=re.S).strip()
    match = re.search(r"(?:final answer|short answer|the answer is|answer)\s*[:\-]?\s*(.+)", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip().split("\n")[0].strip(" .")


def usage_snapshot(wiki: Any) -> dict[str, Any]:
    usage = getattr(wiki, "usage", None)
    return json_safe(usage.to_dict()) if usage is not None and callable(getattr(usage, "to_dict", None)) else {}


def usage_difference(before: Any, after: Any) -> Any:
    if isinstance(after, dict):
        return {key: usage_difference(before.get(key, {}) if isinstance(before, dict) else {}, value)
                for key, value in after.items()}
    if isinstance(after, (int, float)) and not isinstance(after, bool):
        return max(0, after - (before if isinstance(before, (int, float)) else 0))
    return after


def response_dict(response: Any) -> dict[str, Any]:
    convert = getattr(response, "to_dict", None)
    if callable(convert):
        return json_safe(convert())
    return {key: json_safe(getattr(response, key)) for key in
            ("text", "citations", "notes", "decisions", "action", "note", "title", "folder", "related")
            if hasattr(response, key)}


def knowledge_digest(bundle: Path) -> dict[str, str]:
    """Content hashes, not access times; query/audit logs are not wiki knowledge."""
    return {path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.rglob("*.md")) if path != bundle / "log.md"}


def run_llmwiki2(model: str, docs: list[dict[str, Any]], questions: list[dict[str, Any]],
                 run_dir: Path, prompt_dir: Path, *, mode: str = "crow", progress=None) -> dict[str, Any]:
    """Build a fresh independent library wiki with explicit mode and resolved configuration."""
    from llmwiki_adapter import create_wiki, note_snapshot

    if mode not in WIKI_MODES:
        raise ValueError(f"Unknown note-wiki mode: {mode}")
    system = f"llmwiki2_{mode}"
    out = run_dir / system
    out.mkdir(parents=True, exist_ok=False)
    bundle = out / "bundle"
    settings = {
        "OKF_LLM_PROVIDER": "custom", "OKF_LLM_BASE_URL": os.environ["LLM_BASE_URL"],
        "OKF_LLM_MODEL": model, "OKF_LLM_TIMEOUT": os.getenv("OKF_LLM_TIMEOUT", "900"),
        "OKF_LLM_READ_CHARS": os.getenv("OKF_LLM_READ_CHARS", "10000"), "OKF_MODE": mode,
        "OKF_USAGE_LOG": str((out / "usage.jsonl").resolve()),
        "OKF_CLASSIFIER_PROVIDER": "custom",
        "OKF_CLASSIFIER_BASE_URL": os.getenv("OKF_CLASSIFIER_BASE_URL", "http://localhost:8001/v1"),
        "OKF_CLASSIFIER_MODEL": os.getenv("OKF_CLASSIFIER_MODEL", "LiquidAI/d1-3B"),
        "OKF_CLASSIFIER_TIMEOUT": os.getenv("OKF_CLASSIFIER_TIMEOUT", "60"),
    }
    if prompt_dir.is_dir():
        settings["OKF_PROMPTS_DIR"] = str(prompt_dir.resolve())
    wiki = recorder = None
    ingestion, rows, construction_error = [], [], None
    output = {"mode": mode, "system": system, "bundle": relative_file(bundle, run_dir),
              "answers": {system: rows}, "ingestion": ingestion, "writeback_enabled": False}
    report_progress(progress, component=system, state="building", completed_documents=0,
                    expected_documents=len(docs), completed_questions=0, expected_questions=len(questions))
    try:
        wiki, recorder, provenance = create_wiki(bundle, settings, out / "model_calls.jsonl")
        output["configuration"] = provenance
        with recorder.operation("initialization", mode):
            wiki.init()
        for number, doc in enumerate(docs):
            before, mark, started = usage_snapshot(wiki), recorder.mark(), time.perf_counter()
            trace_before = recorder.trace_io_seconds
            try:
                with recorder.operation("ingestion", doc["id"]):
                    response = wiki.ingest(doc["text"], title=doc["title"])
                elapsed = time.perf_counter() - started
                data = response_dict(response)
                usage = data.get("usage") or usage_difference(before, usage_snapshot(wiki))
                record = {"document": doc["title"], "document_id": doc["id"], "status": "complete",
                          "seconds_wall": elapsed, "usage": usage, "response": data,
                          "action": data.get("action"), "note": data.get("note"),
                          "decisions": data.get("decisions", [])}
            except Exception as exc:
                construction_error = error_record(exc)
                record = {"document": doc["title"], "document_id": doc["id"], "status": "failed",
                          "seconds_wall": time.perf_counter() - started,
                          "usage": usage_difference(before, usage_snapshot(wiki)), "error": construction_error}
            calls = recorder.since(mark)
            record.update({"trace_io_seconds": recorder.trace_io_seconds - trace_before,
                           "model_call_ids": [event["call_id"] for event in calls if "call_id" in event],
                           "failed_requests": sum(bool(event.get("error")) for event in calls)})
            ingestion.append(record)
            append_jsonl(out / "ingestion.jsonl", record)
            write_json(out / "ingestion" / f"{number:03d}-{slug(doc['id'])}.json",
                       {**record, "model_calls": calls})
            report_progress(progress, component=system, state="failed" if construction_error else "building",
                            completed_documents=sum(x["status"] == "complete" for x in ingestion),
                            expected_documents=len(docs), error=construction_error)
            if construction_error:
                break
        output["construction_status"] = "failed" if construction_error else "complete"
        output["construction_error"] = construction_error
        if construction_error is None:
            before_queries = knowledge_digest(bundle)
            write_json(out / "knowledge_before_queries.json", before_queries)
            report_progress(progress, component=system, state="running", completed_questions=0,
                            expected_questions=len(questions))
            for qa in questions:
                before, mark, started = usage_snapshot(wiki), recorder.mark(), time.perf_counter()
                trace_before = recorder.trace_io_seconds
                raw_prediction, prediction, error, data = "", "", None, {}
                try:
                    with recorder.operation("query", qa["qa_id"]):
                        response = wiki.ask(qa["question"])
                    raw_prediction = response.text
                    prediction = repair_answer(raw_prediction)
                    data = response_dict(response)
                except Exception as exc:
                    error = error_record(exc)
                elapsed = time.perf_counter() - started
                trace_seconds = recorder.trace_io_seconds - trace_before
                # Artifact reads/writes below are outside the measured ask interval.
                calls = recorder.since(mark)
                usage = data.get("usage") or usage_difference(before, usage_snapshot(wiki))
                notes = data.get("notes", []) or []
                answer_requests = [{"call_id": event.get("call_id"), "op": event.get("op"),
                                    "request": event.get("request"), "response": event.get("response"),
                                    "error": event.get("error")} for event in calls
                                   if event.get("role") == "llm" and "answer" in event.get("op", "").lower()]
                no_answer_call = error is None and "notes" in data and not notes and not answer_requests
                row = {**qa, "system": system, "mode": mode, "raw_prediction": raw_prediction,
                       "prediction": prediction, "citations": data.get("citations", []),
                       "notes": notes, "decisions": data.get("decisions", []), "response": data,
                       "seconds_wall": elapsed, "trace_io_seconds": trace_seconds,
                       "error": error, "usage": usage, "request_attempts": len(calls),
                       "failed_requests": sum(bool(event.get("error")) for event in calls),
                       "model_call_ids": [event["call_id"] for event in calls if "call_id" in event],
                       "trace_status": "complete" if answer_requests or error or no_answer_call else "missing_answer_request",
                       "no_answer_request_reason": "No notes selected; navigation may call models, but no answer-model call was made" if no_answer_call else None,
                       "trace_file": relative_file(out / "queries" / f"{slug(qa['qa_id'])}.json", run_dir),
                       **(score(prediction, qa["reference"]) if error is None
                          else {"em": 0.0, "token_f1": 0.0, "rouge_l": 0.0})}
                try:
                    snapshots = note_snapshot(wiki, notes)
                except Exception as exc:
                    snapshots = []
                    row["trace_status"] = "note_snapshot_error"
                    row["trace_error"] = error_record(exc)
                write_json(out / "queries" / f"{slug(qa['qa_id'])}.json",
                           {**row, "selected_note_snapshots": snapshots,
                            "answer_requests": answer_requests, "model_calls": calls})
                rows.append(row)
                append_jsonl(out / "answers.jsonl", row)
                report_progress(progress, component=system, state="running", completed_questions=len(rows),
                                expected_questions=len(questions), query_errors=sum(bool(x["error"]) for x in rows))
            after_queries = knowledge_digest(bundle)
            output["knowledge_frozen"] = before_queries == after_queries
            output["knowledge_changed_files"] = sorted(key for key in before_queries.keys() | after_queries.keys()
                                                       if before_queries.get(key) != after_queries.get(key))
            write_json(out / "knowledge_after_queries.json", after_queries)
        else:
            output["knowledge_frozen"] = None
        output["usage_total"] = usage_snapshot(wiki)
    except Exception as exc:
        construction_error = error_record(exc)
        output.update({"construction_status": "failed", "construction_error": construction_error,
                       "usage_total": usage_snapshot(wiki) if wiki is not None else {}})
    finally:
        try:
            close = getattr(wiki, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            output["cleanup_error"] = error_record(exc)
        finally:
            if recorder is not None:
                recorder.close()
    query_errors = sum(bool(row["error"]) for row in rows)
    trace_errors = sum(row.get("trace_status") != "complete" for row in rows)
    classifier_calls = output.get("usage_total", {}).get("classifier", {}).get("calls", 0)
    state = ("failed" if construction_error else
             "invalid_knowledge_changed" if output.get("knowledge_frozen") is False else
             "invalid_classic_classifier_usage" if mode == "classic" and classifier_calls else
             "complete_with_cleanup_errors" if output.get("cleanup_error") else
             "complete_with_trace_errors" if trace_errors else
             "complete_with_query_errors" if query_errors else "complete")
    output.update({"status": state, "expected_questions": len(questions), "evaluated_questions": len(rows),
                   "query_errors": query_errors, "trace_errors": trace_errors,
                   "construction_cost": buildcost_summary(ingestion),
                   "latency_scope": "wiki.ask plus response processing; includes synchronous HTTP trace capture; excludes artifact exports",
                   "model_calls_file": relative_file(out / "model_calls.jsonl", run_dir)})
    write_json(out / "ingestion.json", ingestion)
    write_json(out / "results.json", output)
    report_progress(progress, component=system, state=state, completed_questions=len(rows),
                    expected_questions=len(questions), query_errors=query_errors, trace_errors=trace_errors,
                    error=construction_error)
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_rows(rows)


def comparison_report(wikis: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    classic, crow = wikis.get("classic"), wikis.get("crow")
    report = {"modes_requested": manifest["wiki_modes"],
              "same_documents_and_order": manifest["documents"],
              "document_sha256": manifest["document_sha256"],
              "questions_sha256": manifest["questions_sha256"],
              "native_library_modes": True,
              "interpretation": "Independent constructions; mode-specific library features may also differ. "
                                "Resolved flags and actual decision/request traces are retained.",
              "both_available": classic is not None and crow is not None}
    if not classic or not crow:
        return {**report, "paired_comparison_complete": False, "reason": "Both modes were not completed"}
    left, right = classic.get("configuration", {}), crow.get("configuration", {})
    ignored = {"bundle", "mode", "usage_log"}
    cfg_left = {k: v for k, v in left.get("configuration", {}).items() if k not in ignored}
    cfg_right = {k: v for k, v in right.get("configuration", {}).items() if k not in ignored}
    differences = {k: {"classic": cfg_left.get(k), "crow": cfg_right.get(k)}
                   for k in cfg_left.keys() | cfg_right.keys() if cfg_left.get(k) != cfg_right.get(k)}
    library_equal = bool(left.get("library")) and left.get("library") == right.get("library")
    prompts_equal = bool(left.get("prompts_fingerprint")) and left.get("prompts_fingerprint") == right.get("prompts_fingerprint")
    complete = {"complete", "complete_with_query_errors"}
    report.update({"same_library_provenance": library_equal, "same_resolved_prompts": prompts_equal,
                   "common_configuration_equal": bool(cfg_left) and bool(cfg_right) and not differences,
                   "common_configuration_differences": differences,
                   "feature_flags": {"classic": left.get("feature_flags"), "crow": right.get("feature_flags")},
                   "status": {"classic": classic.get("status"), "crow": crow.get("status")},
                   "knowledge_frozen": {"classic": classic.get("knowledge_frozen"), "crow": crow.get("knowledge_frozen")},
                   "classic_classifier_calls": classic.get("usage_total", {}).get("classifier", {}).get("calls", 0)})
    expected_ids = [question["qa_id"] for question in manifest["questions"]]
    left_rows = classic.get("answers", {}).get("llmwiki2_classic", [])
    right_rows = crow.get("answers", {}).get("llmwiki2_crow", [])
    report["same_complete_question_order"] = ([row["qa_id"] for row in left_rows] == expected_ids
                                              and [row["qa_id"] for row in right_rows] == expected_ids)
    report["paired_comparison_complete"] = (
        classic.get("status") in complete and crow.get("status") in complete
        and report["common_configuration_equal"] and library_equal and prompts_equal
        and report["classic_classifier_calls"] == 0 and report["same_complete_question_order"]
    )
    left_summary, right_summary = summarize(left_rows), summarize(right_rows)
    report["crow_minus_classic"] = {key: right_summary[key] - left_summary[key]
                                    for key in ("em", "token_f1", "rouge_l", "seconds_wall", "llm_calls",
                                                "classifier_calls", "total_input_tokens", "total_output_tokens")}
    report["deltas_valid_for_complete_comparison"] = report["paired_comparison_complete"]
    return report


def save_run(run_dir: Path, model: str, structured: dict[str, Any],
             wikis: dict[str, dict[str, Any]], manifest: dict[str, Any],
             run_status: dict[str, Any] | None = None) -> None:
    """Write six distinct summaries without turning missing runs into zero accuracy."""
    run_dir.mkdir(parents=True, exist_ok=True)
    classifier_model = manifest.get("classifier_model")
    summaries, construction, by_type, by_support = [], {}, {}, {}
    for system in manifest["expected_systems"]:
        if system in DIRECT_SYSTEMS:
            rows = structured.get("answers", {}).get(system, [])
            system_state = structured.get("systems", {}).get(system, {}).get("status", "pending")
            implementation = "dense_rag" if system == "rag_semantic" else "structured_wiki"
            if system == "rag_semantic":
                cost = buildcost_summary([], {"raw_seconds": structured.get("indexing", {}).get("raw_seconds")})
                shared_group = None
            else:
                cost = buildcost_summary(structured.get("ingestion", []),
                                         {"wiki_seconds": structured.get("indexing", {}).get("wiki_seconds")})
                shared_group = "structured_wiki"
        else:
            mode = system.removeprefix("llmwiki2_")
            record = wikis.get(mode, {})
            rows = record.get("answers", {}).get(system, [])
            system_state = record.get("status", "pending")
            implementation, shared_group = "llmwiki2", None
            cost = record.get("construction_cost") or buildcost_summary(record.get("ingestion", []))
        summary = {"model": model, "classifier_model": classifier_model if system == "llmwiki2_crow" else None,
                   "implementation": implementation, "system": system, "status": system_state,
                   "expected_questions": len(manifest["questions"]), **summarize(rows)}
        summary["eligible_for_complete_comparison"] = (
            system_state in {"complete", "complete_with_query_errors"}
            and [row["qa_id"] for row in rows] == [question["qa_id"] for question in manifest["questions"]]
        )
        if not rows:
            for metric in ("em", "token_f1", "rouge_l", "seconds_wall"):
                summary[metric] = None
        summaries.append(summary)
        construction[system] = {**cost, "shared_construction_group": shared_group,
                                "scope": ("One structured build and common wiki index, shared by three configurations; do not sum it three times"
                                          if shared_group else cost.get("scope"))}
        construction[system]["partial_or_pending"] = system_state not in {"complete", "complete_with_query_errors"}
        if construction[system]["partial_or_pending"] and not cost.get("documents") and not cost.get("indexing_seconds_total"):
            for key in ("construction_seconds_total", "ingestion_seconds_total", "ingestion_seconds_mean", "indexing_seconds_total"):
                construction[system][key] = None
        by_type[system] = summary_by_group(rows, "question_type")
        by_support[system] = summary_by_group(rows, "supporting_documents")
    payload = {"benchmark_version": BENCHMARK_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
               "model": model, "classifier_model": classifier_model, "manifest": manifest,
               "status": run_status, "structured_wiki": structured, "llmwiki2": wikis}
    write_json(run_dir / "full_results.json", payload)
    write_json(run_dir / "summary.json", summaries)
    write_json(run_dir / "construction_summary.json", construction)
    write_json(run_dir / "summary_by_type.json", by_type)
    write_json(run_dir / "summary_by_support_count.json", by_support)
    write_json(run_dir / "paired_comparison.json", comparison_report(wikis, manifest))
    csv_rows = []
    for summary in summaries:
        flat = {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}
        flat.update({f"sum_{key}": value for key, value in summary["totals"].items()})
        cost = construction[summary["system"]]
        flat.update({"construction_seconds": cost.get("construction_seconds_total"),
                     "ingestion_seconds": cost.get("ingestion_seconds_total"),
                     "indexing_seconds": cost.get("indexing_seconds_total"),
                     "shared_construction_group": cost.get("shared_construction_group")})
        csv_rows.append(flat)
    fields = list(dict.fromkeys(key for row in csv_rows for key in row))
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)


def source_hashes() -> dict[str, str]:
    names = ("benchmark_llm_wiki.py", "benchmark_artifacts.py", "llmwiki_adapter.py",
             "capture_runtime_metadata.py", "requirements.txt", "start_vllm_services.sh")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names if (ROOT / name).is_file()}


def recover_structured_outputs(run_dir: Path) -> dict[str, Any]:
    """Recover completed checkpoints if a later construction/index/export step fails."""
    out = run_dir / "structured_wiki"
    recovered: dict[str, Any] = {"answers": {}, "systems": {}, "ingestion": [], "indexing": {}}
    for name, key in (("ingestion.json", "ingestion"), ("indexing.json", "indexing")):
        path = out / name
        if path.is_file():
            recovered[key] = json.loads(path.read_text(encoding="utf-8"))
    path = out / "answers.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            recovered["answers"].setdefault(row["system"], []).append(row)
    for system in DIRECT_SYSTEMS:
        path = out / f"{system}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            recovered["systems"][system] = payload["status"]
    return recovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"), help="Hugging Face model served by vLLM (defaults to LLM_MODEL).")
    parser.add_argument("--top-docs", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=25)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--prompt-dir", type=Path, default=Path(os.getenv("OKF_PROMPTS_DIR", ROOT / "prompts_benchmark")))
    parser.add_argument("--wiki-modes", nargs="+", choices=WIKI_MODES, default=list(WIKI_MODES),
                        help="Independent note-wiki modes, in execution order (default: classic crow).")
    parser.add_argument("--dataset-revision", default=None,
                        help="Optional Hugging Face revision; actual dataset fingerprints/checksums are always saved.")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required (or set LLM_MODEL)")
    if not os.getenv("LLM_BASE_URL"):
        parser.error("LLM_BASE_URL is required (for example http://localhost:8000/v1)")
    if args.top_docs < 1 or args.max_questions < 1:
        parser.error("--top-docs and --max-questions must be positive")
    if len(args.wiki_modes) != len(set(args.wiki_modes)):
        parser.error("--wiki-modes cannot contain duplicate modes")
    model = args.model
    run_label = f"docs_{args.top_docs}_questions_{args.max_questions}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output / BENCHMARK_VERSION / slug(model) / run_label / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "components": {}, "run_directory": str(run_dir), "last_update": None}

    def progress(update):
        component = update.pop("component")
        status["components"].setdefault(component, {}).update(update)
        status["last_update"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "run_status.json", status)

    write_json(run_dir / "run_status.json", status)
    print(f"Run directory: {run_dir}", flush=True)
    manifest: dict[str, Any] = {"benchmark_version": BENCHMARK_VERSION, "schema_version": 3,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_sha256": source_hashes(),
        "package_versions": package_versions(), "package_provenance": collect_package_provenance(),
        "top_docs": args.top_docs, "max_questions": args.max_questions, "wiki_modes": args.wiki_modes,
        "expected_systems": [*DIRECT_SYSTEMS, *(f"llmwiki2_{mode}" for mode in args.wiki_modes)],
        "execution_order": [*DIRECT_SYSTEMS, *(f"llmwiki2_{mode}" for mode in args.wiki_modes)],
        "classifier_model": os.getenv("OKF_CLASSIFIER_MODEL", "LiquidAI/d1-3B"),
        "embedding_model": EMBEDDING_MODEL, "context_budget_chars": CONTEXT_BUDGET,
        "rag_top_k": 3, "rag_chunk_words": 200,
        "wiki_search": "compiled-page and raw-source paragraphs; BM25 or RRF(k=60)",
        "wiki_query_preparation": "LLM page selection and term expansion for all three structured-wiki strategies",
        "llm_settings": {"temperature": 0, "sdk_transport_retries": 0, "json_validation_attempts": 2,
                         "extract_max_tokens": 4096, "reconcile_max_tokens": 1024,
                         "route_max_tokens": 512, "search_terms_max_tokens": 384, "answer_max_tokens": 96},
        "writeback_enabled": False, "replicates": 1, "randomized_order": False,
        "direct_retrieval_settings": {"raw_chunk_words": 200, "chunk_overlap_words": 0, "dense_top_k": 3,
            "raw_paragraph_min_chars": 30, "wiki_paragraph_min_chars": 20,
            "context_budget_chars": CONTEXT_BUDGET, "graph_reserved_fraction": 0.25, "graph_max_relations": 6,
            "reconciliation_candidates": "all stored nodes", "index_description_chars": 280,
            "fallback_pages_per_type": 3, "expansion_excerpt_chars": 2500,
            "expansion_term_prompt_target": [8, 15], "other_wiki_pages": 3,
            "selected_term_min_matches": 1, "other_raw_min_matches": 2,
            "question_unigram_min_chars": 3, "question_bigrams": "adjacent non-stopwords",
            "stopwords": sorted(ENGLISH_STOP_WORDS), "stemmer": "Snowball English",
            "bm25_package_defaults": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
            "rrf_k": 60, "fusion_lists": "all indexed paragraphs", "normalize_embeddings": True},
        "latency_scope": "Retrieval plus generation, excluding per-query result export. Library timing includes HTTP trace capture.",
        "relevant_environment": redact_secrets({key: value for key, value in os.environ.items()
                                  if key.startswith(("OKF_", "VLLM_", "BENCHMARK_"))
                                  or key in {"LLM_MODEL", "LLM_BASE_URL"}})}
    write_json(run_dir / "benchmark_manifest.json", manifest)
    write_json(run_dir / "runtime_environment.json", runtime_snapshot())
    prompt_manifest = save_prompt_snapshot(
        run_dir, args.prompt_dir,
        {"ANSWER": ANSWER, "EXTRACT": EXTRACT, "RECONCILE": RECONCILE, "ROUTE": ROUTE, "SEARCH_TERMS": SEARCH_TERMS},
        {"INGEST_SCHEMA": INGEST_SCHEMA, "RECONCILE_SCHEMA": RECONCILE_SCHEMA,
         "ROUTE_SCHEMA": ROUTE_SCHEMA, "TERMS_SCHEMA": TERMS_SCHEMA})
    manifest["prompt_snapshot"] = prompt_manifest
    dataset_metadata: dict[str, Any] = {}
    try:
        docs, questions = make_dataset(args.top_docs, args.max_questions, revision=args.dataset_revision,
                                       metadata=dataset_metadata)
        if not questions or any(not doc["text"].strip() for doc in docs):
            raise ValueError("Selected data has no questions or contains empty source documents")
        manifest.update({"actual_questions": len(questions), "documents": [doc["title"] for doc in docs],
                         "document_order": [doc["id"] for doc in docs],
                         "document_sha256": {doc["id"]: hashlib.sha256(doc["text"].encode("utf-8")).hexdigest() for doc in docs},
                         "questions": questions,
                         "questions_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                         "dataset": dataset_metadata})
        manifest["dataset_statistics"] = save_dataset(run_dir, docs, questions, dataset_metadata)
        write_json(run_dir / "benchmark_manifest.json", manifest)
        progress({"component": "dataset", "state": "complete", "documents": len(docs), "questions": len(questions)})
    except Exception as exc:
        status.update({"status": "failed", "error": error_record(exc), "completed_at": datetime.now(timezone.utc).isoformat()})
        write_json(run_dir / "run_status.json", status)
        raise

    structured: dict[str, Any] = {"answers": {}, "systems": {}, "ingestion": []}
    wikis: dict[str, dict[str, Any]] = {}
    save_run(run_dir, model, structured, wikis, manifest, status)
    llm = VLLM(model, os.environ["LLM_BASE_URL"], os.getenv("VLLM_API_KEY", "EMPTY"),
               float(os.getenv("VLLM_TIMEOUT_SECONDS", "1200")), run_dir / "structured_wiki" / "model_calls.jsonl")
    try:
        structured = run_structured_wiki(llm, docs, questions, run_dir, progress=progress)
    except Exception as exc:
        error = error_record(exc)
        structured = recover_structured_outputs(run_dir)
        for system in DIRECT_SYSTEMS:
            if system not in structured["systems"]:
                structured["systems"][system] = {"status": "failed", "error": error}
                progress({"component": system, "state": "failed", "error": error})
        structured["fatal_error"] = error
    finally:
        llm.flush_events()
    save_run(run_dir, model, structured, wikis, manifest, status)
    for mode in args.wiki_modes:
        try:
            wikis[mode] = run_llmwiki2(model, docs, questions, run_dir, args.prompt_dir, mode=mode, progress=progress)
        except Exception as exc:
            error = error_record(exc)
            wikis[mode] = {"mode": mode, "status": "failed", "construction_error": error,
                           "answers": {f"llmwiki2_{mode}": []}, "ingestion": []}
            progress({"component": f"llmwiki2_{mode}", "state": "failed", "error": error})
        save_run(run_dir, model, structured, wikis, manifest, status)
    summaries = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    states = [row["status"] for row in summaries]
    healthy = {"complete", "complete_with_query_errors"}
    systems_complete = all(state in healthy for state in states) and all(row["eligible_for_complete_comparison"] for row in summaries)
    pair_required = set(args.wiki_modes) == set(WIKI_MODES)
    pair_valid = comparison_report(wikis, manifest)["paired_comparison_complete"]
    finished = systems_complete and (not pair_required or pair_valid)
    status.update({"status": ("complete_with_query_errors" if "complete_with_query_errors" in states else "complete")
                   if finished else "incomplete", "completed_at": datetime.now(timezone.utc).isoformat(),
                   "expected_systems": manifest["expected_systems"], "all_systems_complete": systems_complete,
                   "paired_comparison_required": pair_required, "paired_comparison_complete": pair_valid,
                   "comparison_valid": finished})
    write_json(run_dir / "run_status.json", status)
    save_run(run_dir, model, structured, wikis, manifest, status)
    print(f"{status['status']}: {run_dir}", flush=True)
    return 0 if finished else 2


if __name__ == "__main__":
    raise SystemExit(main())
