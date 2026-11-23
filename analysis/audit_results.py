"""Recompute the statistics for the four archived structured_wiki_v4 runs."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "structured_wiki_v4"
# The exact run directories keep the report stable when new experiments are added.
RUNS = (
    ("Qwen_Qwen2.5-3B-Instruct", "docs_3_questions_25", "20261116T110157_135698Z"),
    ("Qwen_Qwen2.5-3B-Instruct", "docs_8_questions_150", "20261116T113159_994748Z"),
    ("Qwen_Qwen3.5-4B", "docs_3_questions_25", "20261116T111556_037302Z"),
    ("Qwen_Qwen3.5-4B", "docs_8_questions_150", "20261116T122712_405157Z"),
)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    remainder = position - lower
    return (ordered[lower] * (1 - remainder)
            + ordered[min(lower + 1, len(ordered) - 1)] * remainder)


def audit(root):
    runs = []
    for model, subset, run_id in RUNS:
        path = root / model / subset / run_id / "full_results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        construction = json.loads((path.parent / "construction_summary.json").read_text(encoding="utf-8"))
        arrays = dict(result["structured_wiki"]["answers"])
        for mode in result["llmwiki2"].values():
            arrays.update(mode["answers"])
        manifest = result["manifest"]
        assert result["benchmark_version"] == PROTOCOL
        assert set(arrays) == {row["system"] for row in summary} == set(manifest["expected_systems"])
        expected_ids = [q["qa_id"] for q in manifest["questions"]]
        systems = {}
        for row in summary:
            key = row["system"]
            answers = arrays.get(key, [])
            assert len(answers) == row["questions"]
            assert row["status"] == "complete"
            assert [a["qa_id"] for a in answers] == expected_ids
            assert len({a["qa_id"] for a in answers}) == len(answers)
            times = [a["seconds_wall"] for a in answers]
            data = {k: row[k] for k in (
                "status", "questions", "errors", "em", "token_f1", "rouge_l",
                "seconds_wall", "llm_calls", "classifier_calls", "total_input_tokens",
                "total_output_tokens", "totals")}
            if times:
                assert abs(statistics.mean(times) - row["seconds_wall"]) < 1e-9
                for metric in ("em", "token_f1", "rouge_l"):
                    assert abs(statistics.mean(a[metric] for a in answers) - row[metric]) < 1e-9
                data.update(median=statistics.median(times), p95=percentile(times, .95),
                            maximum=max(times), exact_answers=sum(a["em"] for a in answers))
                categories = defaultdict(list)
                for answer in answers:
                    categories[answer["question_type"]].append(answer)
                data["by_type"] = {category: {"correct": sum(a["em"] for a in group),
                                              "n": len(group)}
                                   for category, group in categories.items()}
                empty_notes = [a for a in answers if a.get("no_answer_request_reason")]
                data["no_notes"] = len(empty_notes)
                data["false_overlap_abstentions"] = sum(a["token_f1"] > 0 for a in empty_notes)
            data["construction"] = construction[key]
            systems[key] = data
        pairs = {}
        for first, second in (("wiki_routed", "wiki_bm25"),
                              ("wiki_routed", "wiki_hybrid"),
                              ("wiki_routed", "llmwiki2_crow"),
                              ("llmwiki2_crow", "llmwiki2_classic")):
            a = {row["qa_id"]: row for row in arrays.get(first, [])}
            b = {row["qa_id"]: row for row in arrays.get(second, [])}
            if not a or not b:
                continue
            assert a.keys() == b.keys()
            counts = [0, 0, 0, 0]
            for key in a:
                left, right = bool(a[key]["em"]), bool(b[key]["em"])
                counts[0 if left and right else 1 if left else 2 if right else 3] += 1
            pairs[first + "/" + second] = counts
        runs.append({"source": path.parent.relative_to(root).as_posix(), "model": result["model"],
                     "documents": manifest["top_docs"], "questions": manifest["actual_questions"],
                     "benchmark_version": result["benchmark_version"],
                     "code_sha256": manifest["code_sha256"],
                     "questions_sha256": manifest["questions_sha256"],
                     "systems": systems, "paired": pairs,
                     "structured_knowledge": result["structured_wiki"]["wiki_stats"]})
    assert len(runs) == 4
    total = sum(s["questions"] for run in runs for s in run["systems"].values())
    assert total == 2100
    return {"quantile_definition": "Linear interpolation at (n-1)*p (type 7)",
            "records": total, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results" / PROTOCOL,
                        help="Directory containing the two model folders")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/analysis/results_v4_audit.json",
                        help="Destination JSON report")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run without python -O so the numerical checks are enabled")
    result = audit(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified {result['records']} records across {len(result['runs'])} conditions: {args.output}")


if __name__ == "__main__":
    main()
