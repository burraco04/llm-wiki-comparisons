"""Rebuild dataset and subset statistics from the archived MultiHop-RAG release."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from audit_results import PROTOCOL, REPO_ROOT, RUNS

REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"


def evidence_titles(row):
    result = []
    for item in row.get("evidence_list", []) or []:
        title = item if isinstance(item, str) else (item.get("title") or item.get("source")) if isinstance(item, dict) else None
        if title:
            result.append(title)
    return list(dict.fromkeys(result))


def summary(values):
    return {"min": min(values), "mean": mean(values), "max": max(values), "total": sum(values)}


def ordered(counter):
    return dict(sorted(counter.items(), key=lambda kv: str(kv[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/multihoprag")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/analysis/dataset_statistics.json")
    args = parser.parse_args()
    query_file = args.data_dir / "MultiHopRAG.json"
    corpus_file = args.data_dir / "corpus.json"
    queries = json.loads(query_file.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
    valid = [dict(q) for q in queries if q.get("question_type") != "null_query"]
    counts, qdocs = Counter(), {}
    for i, row in enumerate(valid):
        docs = set(evidence_titles(row))
        if docs:
            qdocs[i] = docs
            counts.update(sorted(docs))
    report = {
        "dataset": "yixuantt/MultiHopRAG", "revision": REVISION,
        "query_config": "MultiHopRAG", "corpus_config": "corpus", "split": "train",
        "query_rows": len(queries), "corpus_rows": len(corpus),
        "unique_corpus_titles": len({article["title"] for article in corpus}),
        "full_question_types": ordered(Counter(q["question_type"] for q in queries)),
        "nonnull_questions": len(valid), "nonnull_with_evidence": len(qdocs),
        "json_sha256": {"questions": hashlib.sha256(query_file.read_bytes()).hexdigest(), "corpus": hashlib.sha256(corpus_file.read_bytes()).hexdigest()},
        "subsets": {},
        "length_units": {"words": "Python str.split() whitespace-separated units", "characters": "Python len(str), Unicode code points"},
    }
    for top_docs, cap in ((3, 25), (8, 150)):
        selected = [title for title, _ in counts.most_common(top_docs)]
        selected_set = set(selected)
        text = defaultdict(list)
        for row in corpus:
            if row.get("title") in selected_set and row.get("body"):
                text[row["title"]].append(row["body"])
        doc_texts = ["\n\n".join(text[t]) for t in selected]
        eligible = []
        for i in qdocs:
            if qdocs[i].issubset(selected_set):
                row = valid[i]
                eligible.append({"qa_id": f"mhop_{i}", "question": row["query"], "reference": str(row.get("answer", "")), "question_type": row.get("question_type", ""), "evidence_titles": evidence_titles(row)})
        questions = eligible[:cap]
        key = f"docs_{top_docs}_questions_{cap}"
        entry = {
            "requested_docs": top_docs, "question_cap": cap, "actual_docs": len(selected),
            "eligible_questions_before_cap": len(eligible), "actual_questions": len(questions),
            "question_types": ordered(Counter(q["question_type"] for q in questions)),
            "supporting_document_counts": ordered(Counter(len(q["evidence_titles"]) for q in questions)),
            "documents": [{"title": t, "associated_question_count": counts[t], "corpus_rows": len(text[t]), "words": len(s.split()), "characters": len(s), "sha256": hashlib.sha256(s.encode("utf-8")).hexdigest()} for t, s in zip(selected, doc_texts)],
            "document_words": summary([len(s.split()) for s in doc_texts]),
            "document_characters": summary([len(s) for s in doc_texts]),
            "question_words": summary([len(q["question"].split()) for q in questions]),
            "question_characters": summary([len(q["question"]) for q in questions]),
            "question_ids": [q["qa_id"] for q in questions],
            "verified_manifests": [],
        }
        for model, subset, run_id in RUNS:
            if subset != key:
                continue
            relative = Path(PROTOCOL) / model / subset / run_id / "benchmark_manifest.json"
            manifest = json.loads((args.results_dir / relative).read_text(encoding="utf-8"))
            if manifest["documents"] != selected or manifest["questions"] != questions:
                raise ValueError(f"Selected inputs differ from {relative}")
            expected_hashes = [manifest["document_sha256"][doc_id] for doc_id in manifest["document_order"]]
            if [doc["sha256"] for doc in entry["documents"]] != expected_hashes:
                raise ValueError(f"Document text differs from {relative}")
            entry["verified_manifests"].append(relative.as_posix())
        report["subsets"][key] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified {len(queries)} questions and {len(corpus)} articles against the selected inputs: {args.output}")


if __name__ == "__main__":
    main()
