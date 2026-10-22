"""Run the MultiHopRAG benchmark against the notebook LLM-wiki and llmwiki2.

Required environment variables: VLLM_BASE_URL.  llmwiki2 additionally needs
OPENROUTER_API_KEY when OKF_CLASSIFIER_PROVIDER=openrouter (the default).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import string
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

MODELS = ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3.5-4B")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
        title = item if isinstance(item, str) else item.get("title") or item.get("source") if isinstance(item, dict) else None
        if title:
            result.append(title)
    return list(dict.fromkeys(result))


def make_dataset(top_docs: int, max_questions: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    valid = [dict(q) for q in queries if q.get("question_type") != "null_query"]
    counts, qdocs = Counter(), {}
    for i, row in enumerate(valid):
        docs = set(evidence_titles(row))
        if docs:
            qdocs[i] = docs; counts.update(docs)
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

    def ask(self, prompt: str, max_tokens: int = 96) -> str:
        result = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens)
        usage = result.usage
        self.calls += 1; self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0; self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        return (result.choices[0].message.content or "").strip().split("\n")[0].strip(' "')

    def snapshot(self) -> tuple[int, int, int]: return self.calls, self.input_tokens, self.output_tokens
    def delta(self, before: tuple[int, int, int]) -> dict[str, int]:
        return {"llm_calls": self.calls - before[0], "input_tokens": self.input_tokens - before[1], "output_tokens": self.output_tokens - before[2]}


ANSWER = """Answer using only the context. Give a short direct answer (entity, name, number, date, phrase, Yes/No), or unanswerable. Output only the answer.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
EXTRACT = """Extract a compact factual wiki note from the document. Include title, a concise summary, named entities, and relations. Do not invent facts.\n\nTitle: {title}\nDocument:\n{text}"""
ROUTE = """You are the router of a wiki knowledge base. Select only page ids that are useful for answering the question.\n\nQuestion: {question}\n\nWiki index:\n{index}\n\nValid page ids: {valid_ids}\n\nReturn ONLY JSON in this form: {{\"source_pages\": [], \"entity_pages\": [], \"concept_pages\": []}}"""
SEARCH_TERMS = """Generate 8-15 short search terms for paragraph-level keyword matching.\n\nQuestion: {question}\n\nSelected wiki pages:\n{pages}\n\nReturn ONLY JSON in this form: {{\"search_terms\": []}}"""


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
        text = hit["text"]
        if used + len(text) > budget: break
        out.append(f"[{hit['title']}]\n{text}"); used += len(text)
    return "\n\n---\n\n".join(out)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the small JSON objects requested by routed retrieval, tolerating fences."""
    cleaned = re.sub(r"```(?:json)?|```", "", str(text), flags=re.I).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def routed_retrieve(llm: VLLM, question: str, docs: list[dict[str, Any]], wiki_notes: list[dict[str, Any]], budget: int = 12000) -> tuple[str, dict[str, Any]]:
    """Method A from tutorial_LLM_wiki_v3: LLM routing -> LLM terms -> ranked paragraphs.

    The notebook has source/entity/concept/analysis pages.  The compact runner
    represents the same branches with raw source documents and generated wiki
    notes, while retaining the two LLM decisions and deterministic safety net.
    """
    pages = {d["id"]: d["text"] for d in docs}
    pages.update({f"wiki/{slug(n['title'])}": n["text"] for n in wiki_notes})
    valid_ids = sorted(pages)
    index = "\n".join(f"- {pid}: {text[:280]}" for pid, text in pages.items())
    routed = parse_json_object(llm.ask(ROUTE.format(question=question, index=index, valid_ids=", ".join(valid_ids)), max_tokens=256))
    selected_ids = []
    for key in ("source_pages", "entity_pages", "concept_pages", "analysis_pages"):
        selected_ids.extend(x for x in routed.get(key, []) if isinstance(x, str) and x in pages)
    selected_ids = list(dict.fromkeys(selected_ids))
    if not selected_ids:
        # Same deterministic fallback used by the notebook when routing returns no valid id.
        q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
        selected_ids = sorted(pages, key=lambda pid: len(q_words & set(re.findall(r"[a-z0-9]+", pages[pid].lower()))), reverse=True)[:3]
    selected_text = "\n\n".join(f"[{pid}]\n{pages[pid][:2500]}" for pid in selected_ids)
    terms_data = parse_json_object(llm.ask(SEARCH_TERMS.format(question=question, pages=selected_text), max_tokens=200))
    terms = [str(x).lower().strip() for x in terms_data.get("search_terms", []) if str(x).strip()]
    terms.extend(re.findall(r"[a-z0-9]+", question.lower()))
    terms = list(dict.fromkeys(terms))

    candidates = []
    for doc in docs:
        for paragraph in re.split(r"\n\n+", doc["text"]):
            paragraph = paragraph.strip()
            if len(paragraph) < 30:
                continue
            low = paragraph.lower()
            exact = sum(term in low for term in terms)
            if exact:
                candidates.append((exact, int(doc["id"] in selected_ids), {"title": doc["title"], "text": paragraph}))
    for note in wiki_notes:
        low = note["text"].lower()
        exact = sum(term in low for term in terms)
        if exact:
            candidates.append((exact, int(f"wiki/{slug(note['title'])}" in selected_ids), note))
    candidates.sort(key=lambda row: (row[1], row[0]), reverse=True)
    hits = [row[2] for row in candidates]
    return context_from_hits(hits, budget), {"routed_ids": selected_ids, "terms": terms, "candidates": len(candidates), "selected": len(hits)}


def run_notebook_wiki(llm: VLLM, docs: list[dict[str, Any]], questions: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Compact, non-notebook execution of tutorial_LLM_wiki_v3's ingest/RAG/wiki tests."""
    raw = paragraphs(docs)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = encoder.encode([x["text"] for x in raw], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", x["text"].lower()) for x in raw])
    ingest = []
    wiki_notes = []
    for doc in docs:
        before, started = llm.snapshot(), time.perf_counter()
        note = llm.ask(EXTRACT.format(title=doc["title"], text=doc["text"]), max_tokens=700)
        wiki_notes.append({"title": doc["title"], "text": note})
        ingest.append({"document": doc["title"], "seconds_wall": time.perf_counter() - started, **llm.delta(before)})
    wiki_emb = encoder.encode([x["text"] for x in wiki_notes], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    wiki_bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", x["text"].lower()) for x in wiki_notes])
    systems = {"rag_semantic": (raw, embeddings, None), "wiki_bm25": (wiki_notes, None, wiki_bm25), "wiki_hybrid": (wiki_notes, wiki_emb, wiki_bm25)}
    result = {}
    routed_rows = []
    for qa in questions:
        before, started = llm.snapshot(), time.perf_counter()
        context, diagnostics = routed_retrieve(llm, qa["question"], docs, wiki_notes)
        prediction = llm.ask(ANSWER.format(context=context, question=qa["question"]), max_tokens=64)
        routed_rows.append({**qa, "prediction": prediction, "seconds_wall": time.perf_counter() - started, "retrieval": diagnostics, **llm.delta(before), **score(prediction, qa["reference"])})
    result["wiki_routed"] = routed_rows
    for name, (items, emb, lexical) in systems.items():
        rows = []
        for qa in questions:
            before, started = llm.snapshot(), time.perf_counter()
            if emb is not None:
                q = encoder.encode([qa["question"]], convert_to_numpy=True, normalize_embeddings=True)[0]
                semantic = emb @ q
            else: semantic = np.zeros(len(items))
            if lexical is not None:
                lexical_scores = np.asarray(lexical.get_scores(re.findall(r"[a-z0-9]+", qa["question"].lower())))
                lexical_scores = lexical_scores / max(float(lexical_scores.max()), 1e-9)
            else: lexical_scores = np.zeros(len(items))
            ranks = np.argsort(-(semantic / max(float(semantic.max()), 1e-9) + lexical_scores))[:5]
            prediction = llm.ask(ANSWER.format(context=context_from_hits([items[i] for i in ranks]), question=qa["question"]), max_tokens=64)
            rows.append({**qa, "prediction": prediction, "seconds_wall": time.perf_counter() - started, **llm.delta(before), **score(prediction, qa["reference"])})
        result[name] = rows
    return {"ingestion": ingest, "answers": result}


def repair_answer(text: str) -> str:
    text = re.sub(r"```.*?```", "", str(text), flags=re.S).strip()
    match = re.search(r"(?:final answer|short answer|the answer is|answer)\s*[:\-]?\s*(.+)", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip().split("\n")[0].strip(" .")


def run_llmwiki2(model: str, docs: list[dict[str, Any]], questions: list[dict[str, Any]], run_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    try:
        from llmw2 import Wiki
    except ImportError as exc:
        raise RuntimeError("llmwiki2 is required; install requirements.txt") from exc
    bundle = run_dir / "llmwiki2_bundle"; shutil.rmtree(bundle, ignore_errors=True); bundle.mkdir(parents=True)
    settings = {"OKF_LLM_PROVIDER": "custom", "OKF_LLM_BASE_URL": os.environ["VLLM_BASE_URL"], "OKF_LLM_MODEL": model, "OKF_LLM_TIMEOUT": os.getenv("OKF_LLM_TIMEOUT", "900"), "OKF_LLM_READ_CHARS": os.getenv("OKF_LLM_READ_CHARS", "10000"), "OKF_MODE": os.getenv("OKF_MODE", "crow"), "OKF_USAGE_LOG": str((run_dir / "llmwiki2_calls.jsonl").resolve())}
    if prompt_dir.is_dir():
        settings["OKF_PROMPTS_DIR"] = str(prompt_dir.resolve())
    os.environ.update(settings)
    if not os.getenv("OKF_CLASSIFIER_PROVIDER"): os.environ["OKF_CLASSIFIER_PROVIDER"] = "openrouter"
    if not os.getenv("OKF_CLASSIFIER_MODEL"): os.environ["OKF_CLASSIFIER_MODEL"] = "inception/mercury-decide:free"
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
    return {"ingestion": ingestion, "answers": {"llmwiki2": rows}, "usage_total": wiki.usage.to_dict()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("em", "token_f1", "rouge_l", "seconds_wall", "llm_calls", "input_tokens", "output_tokens")
    return {"questions": len(rows), **{key: sum(float(r.get(key, 0) or 0) for r in rows) / max(1, len(rows)) for key in metrics}}


def save_run(run_dir: Path, model: str, notebook: dict[str, Any], wiki2: dict[str, Any], manifest: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "model": model, "dataset": manifest, "tutorial_llm_wiki": notebook, "llmwiki2": wiki2}
    (run_dir / "full_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    for system, rows in notebook["answers"].items(): summaries.append({"model": model, "implementation": "tutorial_llm_wiki", "system": system, **summarize(rows)})
    for system, rows in wiki2["answers"].items(): summaries.append({"model": model, "implementation": "llmwiki2", "system": system, **summarize(rows)})
    (run_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS)); parser.add_argument("--top-docs", type=int, default=3); parser.add_argument("--max-questions", type=int, default=25)
    parser.add_argument("--output", type=Path, default=OUT); parser.add_argument("--prompt-dir", type=Path, default=Path(os.getenv("OKF_PROMPTS_DIR", ROOT / "prompts_benchmark")))
    args = parser.parse_args()
    if not os.getenv("VLLM_BASE_URL"): parser.error("VLLM_BASE_URL is required (for example http://localhost:8001/v1)")
    docs, questions = make_dataset(args.top_docs, args.max_questions)
    manifest = {"top_docs": args.top_docs, "max_questions": args.max_questions, "documents": [d["title"] for d in docs], "questions": questions}
    for model in args.models:
        run_dir = args.output / slug(model); run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        llm = VLLM(model, os.environ["VLLM_BASE_URL"], os.getenv("VLLM_API_KEY", "EMPTY"), float(os.getenv("VLLM_TIMEOUT_SECONDS", "1200")))
        notebook = run_notebook_wiki(llm, docs, questions, run_dir)
        wiki2 = run_llmwiki2(model, docs, questions, run_dir, args.prompt_dir)
        save_run(run_dir, model, notebook, wiki2, manifest)
        print(f"Completed {model}: {run_dir}")


if __name__ == "__main__": main()
