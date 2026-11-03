"""Run the MultiHopRAG benchmark against the notebook LLM-wiki and llmwiki2.

The writing model is exposed through ``LLM_BASE_URL``.  In CROW mode llmwiki2
uses the local System One endpoint exposed by the second vLLM server through
``OKF_CLASSIFIER_BASE_URL``; no external classifier API key is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import string
import time
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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BENCHMARK_VERSION = "structured_wiki_v2"
CONTEXT_BUDGET = 12000
STEMMER = SnowballStemmer("english")


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


def make_dataset(top_docs: int, max_questions: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    valid = [dict(q) for q in queries if q.get("question_type") != "null_query"]
    counts, qdocs = Counter(), {}
    for i, row in enumerate(valid):
        docs = set(evidence_titles(row))
        if docs:
            qdocs[i] = docs; counts.update(sorted(docs))
    selected = [title for title, _ in counts.most_common(top_docs)]
    selected_set = set(selected)
    text, metadata = defaultdict(list), {}
    for row in corpus:
        if row.get("title") in selected_set and row.get("body"):
            text[row["title"]].append(row["body"])
            metadata.setdefault(row["title"], {k: str(row.get(k, "")) for k in ("source", "url", "published_at")})
    docs = [{"id": slug(title), "title": title, "text": "\n\n".join(text[title]), "meta": metadata.get(title, {})} for title in selected]
    questions = []
    for i in qdocs:
        if qdocs[i].issubset(selected_set):
            row = valid[i]
            questions.append({"qa_id": f"mhop_{i}", "question": row["query"], "reference": str(row.get("answer", "")), "question_type": row.get("question_type", ""), "evidence_titles": evidence_titles(row)})
    return docs, questions[:max_questions]


class VLLM:
    def __init__(self, model: str, base_url: str, api_key: str, timeout: float):
        self.model, self.client = model, OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)
        self.calls = self.input_tokens = self.output_tokens = 0
        self.structured_output = True
        self.json_failures = 0
        self.events: list[dict[str, Any]] = []

    def ask(self, prompt: str, max_tokens: int = 96, *, response_format=None, stage: str = "answer") -> str:
        started = time.perf_counter()
        options = {"response_format": response_format} if response_format else {}
        result = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens, **options)
        usage = result.usage
        self.calls += 1; self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0; self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        choice = result.choices[0]
        self.events.append({"stage": stage, "seconds_wall": time.perf_counter() - started,
                            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                            "finish_reason": choice.finish_reason})
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
        # Invalid extraction aborts the run instead of silently substituting a
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
    return result


def paragraphs(docs: list[dict[str, Any]], words: int = 200) -> list[dict[str, str]]:
    output = []
    for doc in docs:
        tokens = doc["text"].split()
        for start in range(0, len(tokens), words):
            output.append({"doc_id": doc["id"], "title": doc["title"], "text": " ".join(tokens[start:start + words])})
    return output


def context_from_hits(hits: list[dict[str, Any]], budget: int = 12000) -> str:
    out, used = [], 0
    for hit in hits:
        block = f"[{hit['title']}]\n{hit['text']}"
        remaining = budget - used - (len("\n\n---\n\n") if out else 0)
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining > len(hit["title"]) + 20:
                out.append(block[:remaining])
            break
        used += len(block) + (len("\n\n---\n\n") if out else 0)
        out.append(block)
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
        items.extend({"title": pid, "page_id": pid, "text": p, "branch": "wiki"} for p in split_paragraphs(page["text"], 20))
    for sid, source in wiki.sources.items():
        items.extend({"title": "raw:" + sid, "page_id": sid, "text": p, "branch": "raw"} for p in split_paragraphs(source["text"]))
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
        selected.append(hit)
        matched.add(hit["page_id"])
        if len(block) > remaining:
            break
    node_ids = {nid for nid in wiki.nodes if wiki.node_page_id(nid) in matched}
    triples, graph = [], ""
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
        graph = proposed
        if len(triples) == 6:
            break
    context = "\n\n".join(parts)
    if graph:
        context += "\n\n" + graph
    return context, {"candidates": candidates, "selected": len(selected), "chars": len(context),
                     "matched_pages": list(dict.fromkeys(h["page_id"] for h in selected)),
                     "kg_triples": len(triples), "branches": dict(Counter(h["branch"] for h in selected))}


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
            candidates.append((fuzzy, exact, {**hit, "branch": branch}))
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return pack_wiki_context(wiki, [h for _, _, h in candidates], len(candidates))


def run_notebook_wiki(llm: VLLM, docs: list[dict[str, Any]], questions: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Structured notebook wiki, immutable throughout the batch evaluation."""
    from sentence_transformers import SentenceTransformer

    wiki, ingest = StructuredWiki(), []
    bundle = run_dir / "notebook_wiki_bundle"
    for doc in docs:
        ingest.append(wiki.ingest(llm, doc))
        wiki.save(bundle)
        print(f"Ingested {doc['title']}: {wiki.stats()}", flush=True)
    (run_dir / "notebook_ingestion.json").write_text(json.dumps(ingest, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = paragraphs(docs)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    started = time.perf_counter()
    embeddings = encoder.encode([x["text"] for x in raw], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    raw_index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    items = wiki_items(wiki)
    if not raw or not items:
        raise ValueError("No source/retrieval passages; check the input corpus")
    wiki_emb = encoder.encode([x["text"] for x in items], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    wiki_bm25 = BM25Okapi([tokenize(x["text"]) or ["__empty__"] for x in items])
    wiki_index_seconds = time.perf_counter() - started
    result = {}
    for name in ("rag_semantic", "wiki_routed", "wiki_bm25", "wiki_hybrid"):
        rows = []
        for qa in questions:
            before, started = llm.snapshot(), time.perf_counter()
            if name == "rag_semantic":
                q = encoder.encode([qa["question"]], convert_to_numpy=True, normalize_embeddings=True)[0]
                ranks = np.argsort(-(embeddings @ q), kind="stable")[:3]
                context = context_from_hits([raw[i] for i in ranks], CONTEXT_BUDGET)
                diagnostics = {"selected": len(ranks), "chars": len(context), "top_k": 3}
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
                    if name == "wiki_hybrid":
                        q = encoder.encode([qa["question"]], convert_to_numpy=True, normalize_embeddings=True)[0]
                        semantic_rank = np.argsort(-(wiki_emb @ q), kind="stable")
                        fused[semantic_rank] += 1 / (60 + np.arange(1, len(items) + 1))
                    order = np.argsort(-fused, kind="stable")
                    hits = [items[i] for i in order if name == "wiki_hybrid" or lexical[i] > 0]
                    context, retrieval = pack_wiki_context(wiki, hits, len(items))
                diagnostics.update(retrieval)
                diagnostics["timing"]["retrieval_code"] = time.perf_counter() - t
            t = time.perf_counter()
            raw_prediction = llm.ask(ANSWER.format(context=context, question=qa["question"]), max_tokens=96, stage=f"{name}_answer")
            prediction = repair_answer(raw_prediction)
            diagnostics.setdefault("timing", {})["answer_llm"] = time.perf_counter() - t
            rows.append({**qa, "prediction": prediction, "raw_prediction": raw_prediction,
                         "seconds_wall": time.perf_counter() - started, "retrieval": diagnostics,
                         **llm.delta(before), **score(prediction, qa["reference"])})
            # Durable checkpoint: an interrupted model/system run retains completed answers.
            with (run_dir / "notebook_answers.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"system": name, **rows[-1]}, ensure_ascii=False) + "\n")
        result[name] = rows
    return {"ingestion": ingest, "answers": result, "wiki_stats": wiki.stats(), "writeback_enabled": False,
            "indexing": {"raw_seconds": raw_index_seconds, "wiki_seconds": wiki_index_seconds,
                         "wiki_paragraphs": len(items), "raw_chunks": len(raw)},
            "json_failures": llm.json_failures, "model_calls": llm.events}


def repair_answer(text: str) -> str:
    text = re.sub(r"```.*?```", "", str(text), flags=re.S).strip()
    match = re.search(r"(?:final answer|short answer|the answer is|answer)\s*[:\-]?\s*(.+)", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip().split("\n")[0].strip(" .")


def run_llmwiki2(model: str, docs: list[dict[str, Any]], questions: list[dict[str, Any]], run_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    try:
        from llmw2 import Wiki
    except ImportError as exc:
        raise RuntimeError("llmwiki2 is required; install requirements.txt") from exc
    bundle = run_dir / "llmwiki2_bundle"
    bundle.mkdir(parents=True, exist_ok=False)
    settings = {
        "OKF_LLM_PROVIDER": "custom",
        "OKF_LLM_BASE_URL": os.environ["LLM_BASE_URL"],
        "OKF_LLM_MODEL": model,
        "OKF_LLM_TIMEOUT": os.getenv("OKF_LLM_TIMEOUT", "900"),
        "OKF_LLM_READ_CHARS": os.getenv("OKF_LLM_READ_CHARS", "10000"),
        "OKF_MODE": os.getenv("OKF_MODE", "crow"),
        "OKF_USAGE_LOG": str((run_dir / "llmwiki2_calls.jsonl").resolve()),
        "OKF_CLASSIFIER_PROVIDER": "custom",
        "OKF_CLASSIFIER_BASE_URL": os.getenv("OKF_CLASSIFIER_BASE_URL", "http://localhost:8001/v1"),
        "OKF_CLASSIFIER_MODEL": os.getenv("OKF_CLASSIFIER_MODEL", "LiquidAI/d1-3B"),
        "OKF_CLASSIFIER_TIMEOUT": os.getenv("OKF_CLASSIFIER_TIMEOUT", "60"),
    }
    if prompt_dir.is_dir():
        settings["OKF_PROMPTS_DIR"] = str(prompt_dir.resolve())
    os.environ.update(settings)
    wiki = Wiki.from_env(bundle=bundle); wiki.init(); ingestion, rows = [], []
    for doc in docs:
        started = time.perf_counter(); response = wiki.ingest(doc["text"], title=doc["title"]); usage = response.usage.to_dict()
        ingestion.append({"document": doc["title"], "seconds_wall": time.perf_counter() - started, "action": str(response.action), "note": response.note, "usage": usage})
    for qa in questions:
        started, error = time.perf_counter(), None
        try:
            response = wiki.ask(qa["question"]); raw = response.text; prediction = repair_answer(raw); citations = response.citations; usage = response.usage.to_dict()
        except Exception as exc:
            raw = prediction = ""; citations = []; usage = {}; error = repr(exc)
        rows.append({**qa, "raw_prediction": raw, "prediction": prediction, "citations": citations, "seconds_wall": time.perf_counter() - started, "error": error, "usage": usage, **score(prediction, qa["reference"])})
        with (run_dir / "llmwiki2_answers.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    return {"ingestion": ingestion, "answers": {"llmwiki2": rows}, "usage_total": wiki.usage.to_dict()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("em", "token_f1", "rouge_l", "seconds_wall", "llm_calls", "input_tokens", "output_tokens",
               "classifier_calls", "classifier_input_tokens", "classifier_output_tokens", "total_input_tokens", "total_output_tokens")
    normalized = []
    for row in rows:
        value = dict(row)
        if "usage" in row:
            usage = row["usage"]
            writing, classifier = usage.get("llm", {}), usage.get("classifier", {})
            value.update(llm_calls=writing.get("calls", 0), input_tokens=writing.get("input_tokens", 0),
                         output_tokens=writing.get("output_tokens", 0), classifier_calls=classifier.get("calls", 0),
                         classifier_input_tokens=classifier.get("input_tokens", 0), classifier_output_tokens=classifier.get("output_tokens", 0))
        value["total_input_tokens"] = value.get("input_tokens", 0) + value.get("classifier_input_tokens", 0)
        value["total_output_tokens"] = value.get("output_tokens", 0) + value.get("classifier_output_tokens", 0)
        normalized.append(value)
    return {"questions": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
            **{key: sum(float(r.get(key, 0) or 0) for r in normalized) / max(1, len(rows)) for key in metrics}}


def save_run(run_dir: Path, model: str, notebook: dict[str, Any], wiki2: dict[str, Any], manifest: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    classifier_model = manifest.get("classifier_model", "LiquidAI/d1-3B")
    payload = {"benchmark_version": BENCHMARK_VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "model": model, "classifier_model": classifier_model, "dataset": manifest, "tutorial_llm_wiki": notebook, "llmwiki2": wiki2}
    (run_dir / "full_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    for system, rows in notebook["answers"].items(): summaries.append({"model": model, "classifier_model": classifier_model, "implementation": "tutorial_llm_wiki", "system": system, **summarize(rows)})
    for system, rows in wiki2["answers"].items(): summaries.append({"model": model, "classifier_model": classifier_model, "implementation": "llmwiki2", "system": system, **summarize(rows)})
    (run_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"), help="Hugging Face model served by vLLM (defaults to LLM_MODEL).")
    parser.add_argument("--top-docs", type=int, default=3); parser.add_argument("--max-questions", type=int, default=25)
    parser.add_argument("--output", type=Path, default=OUT); parser.add_argument("--prompt-dir", type=Path, default=Path(os.getenv("OKF_PROMPTS_DIR", ROOT / "prompts_benchmark")))
    args = parser.parse_args()
    if not args.model: parser.error("--model is required (or set LLM_MODEL)")
    if not os.getenv("LLM_BASE_URL"): parser.error("LLM_BASE_URL is required (for example http://localhost:8000/v1)")
    if args.top_docs < 1 or args.max_questions < 1:
        parser.error("--top-docs and --max-questions must be positive")
    docs, questions = make_dataset(args.top_docs, args.max_questions)
    if not questions or any(not d["text"].strip() for d in docs):
        raise ValueError("The selected dataset has no questions or contains empty source documents")
    classifier_model = os.getenv("OKF_CLASSIFIER_MODEL", "LiquidAI/d1-3B")
    manifest = {"benchmark_version": BENCHMARK_VERSION, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "package_versions": package_versions(),
                "top_docs": args.top_docs, "max_questions": args.max_questions, "actual_questions": len(questions),
                "classifier_model": classifier_model, "embedding_model": EMBEDDING_MODEL,
                "context_budget_chars": CONTEXT_BUDGET, "rag_top_k": 3, "rag_chunk_words": 200,
                "wiki_search": "paragraphs of compiled pages and raw sources; BM25 or RRF(k=60)",
                "wiki_query_preparation": "LLM routing and term expansion for all three wiki strategies",
                "llm_settings": {"temperature": 0, "extract_max_tokens": 4096, "reconcile_max_tokens": 1024,
                                 "route_max_tokens": 512, "search_terms_max_tokens": 384, "answer_max_tokens": 96},
                "writeback_enabled": False, "documents": [d["title"] for d in docs],
                "document_sha256": {d["id"]: hashlib.sha256(d["text"].encode("utf-8")).hexdigest() for d in docs}, "questions": questions}
    model = args.model
    run_label = f"docs_{args.top_docs}_questions_{args.max_questions}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output / BENCHMARK_VERSION / slug(model) / run_label / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    llm = VLLM(model, os.environ["LLM_BASE_URL"], os.getenv("VLLM_API_KEY", "EMPTY"), float(os.getenv("VLLM_TIMEOUT_SECONDS", "1200")))
    notebook = run_notebook_wiki(llm, docs, questions, run_dir)
    (run_dir / "notebook_results.json").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    wiki2 = run_llmwiki2(model, docs, questions, run_dir, args.prompt_dir)
    save_run(run_dir, model, notebook, wiki2, manifest)
    print(f"Completed {model}: {run_dir}")


if __name__ == "__main__": main()
