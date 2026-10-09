"""
evaluate_retrieval.py — measure retrieval quality against the golden set.
 
Sweeps k and search_type (similarity vs MMR) against the current index and reports
recall@k, hit@k, and MRR. Entirely local (bge-m3 + Chroma) — no LLM, no API cost.
 
Scoring uses SECTION-PREFIX matching: a retrieved chunk counts as relevant for a
label L when its number == L or starts with "L." — so a subsection chunk (e.g.
"2.3.1") credits a section-level label ("2.3"), matching how the golden set is
labelled.
 
Metrics (averaged over answerable questions):
  hit@k     — fraction of questions with >=1 relevant chunk in the top-k (did it work)
  recall@k  — fraction of a question's relevant SECTIONS covered by the top-k
  MRR@k     — mean reciprocal rank of the first relevant chunk (how high it ranks)
"""
 
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for config + retrieval
from config import CFG  # noqa: E402
from retrieval import Retriever  # noqa: E402

GOLDEN_PATH = CFG["paths"]["golden"]
REPORT_DIR = CFG["paths"]["retrieval_reports"]
K_VALUES = CFG["eval"]["retrieval"]["k_values"]
SEARCH_TYPES = CFG["eval"]["retrieval"]["search_types"]
APP_K = CFG["retrieval"]["k"]   # pick the best search_type at the k the app uses
 
 
def _is_relevant(chunk_number, labels):
    """
    A chunk is relevant to a label if its number matches the label or starts with it + dot.
    """
    num = str(chunk_number)
    
    return any(num == lbl or num.startswith(f"{lbl}.") for lbl in labels)
 
 
def _eval_query(hits, labels, k_values):
    """
    Per-query metrics at each k, from a single ranked hit list (length >= max k).
    """
    first_relevant_rank = None
    label_first_rank = {lbl: None for lbl in labels}
    
    for rank, hit in enumerate(hits, start=1):
        num = str(hit.get("number"))
        if first_relevant_rank is None and _is_relevant(num, labels):
            first_relevant_rank = rank
        for lbl in labels:
            if label_first_rank[lbl] is None and (num == lbl or num.startswith(f"{lbl}.")):
                label_first_rank[lbl] = rank
 
    out = {}
    for k in k_values:
        hit_k = first_relevant_rank is not None and first_relevant_rank <= k
        covered = sum(1 for r in label_first_rank.values() if r is not None and r <= k)
        out[k] = {
            "hit": 1.0 if hit_k else 0.0,
            "recall": covered / len(labels) if labels else 0.0,
            "mrr": (1.0 / first_relevant_rank) if hit_k else 0.0,
        }
    
    return out
 
 
def evaluate(retriever, golden, k_values=K_VALUES, search_types=SEARCH_TYPES):
    """
    Evaluate the retriever against the golden set, sweeping k and search_type.
    """
    answerable = [r for r in golden if r.get("type") == "answerable" and r.get("relevant_sections")]
    max_k = max(k_values)
    fetch_k = max(20, max_k * 4)
 
    results = {}   # (search_type, k) -> aggregated metrics
    per_lang = {}  # (search_type, k, language) -> list of per-query metric dicts
 
    for search_type in search_types:
        agg = {k: {"hit": [], "recall": [], "mrr": []} for k in k_values}
        
        for row in answerable:
            hits = retriever.retrieve(row["question"], k=max_k,
                                      search_type=search_type, fetch_k=fetch_k)
            q_metrics = _eval_query(hits, row["relevant_sections"], k_values)
            
            for k in k_values:
                for m in ("hit", "recall", "mrr"):
                    agg[k][m].append(q_metrics[k][m])
                    per_lang.setdefault((search_type, k, row.get("language", "?")),
                                        {"hit": [], "recall": [], "mrr": []})
                    per_lang[(search_type, k, row.get("language", "?"))][m].append(q_metrics[k][m])
        
        for k in k_values:
            results[(search_type, k)] = {
                "n": len(answerable),
                **{m: sum(v) / len(v) if v else 0.0 for m, v in agg[k].items()},
            }
    
    return results, per_lang
 
 
def _print_table(results):
    """
    Print aggregated results in a simple table.
    """
    print(f"\n{'search_type':<12} {'k':>3} {'hit@k':>8} {'recall@k':>10} {'MRR':>8}")
    print("-" * 46)
    
    for (st, k), m in results.items():
        print(f"{st:<12} {k:>3} {m['hit']:>8.3f} {m['recall']:>10.3f} {m['mrr']:>8.3f}")
 
 
def _print_language_breakdown(per_lang, k=5, search_type="similarity"):
    """
    Print results broken down by language (for a specific k and search_type).
    """
    print(f"\nBy language @ k={k} ({search_type}):")
    print(f"  {'lang':<6} {'n':>4} {'hit':>8} {'recall':>8} {'MRR':>8}")
    
    for lang in sorted({l for (st, kk, l) in per_lang if st == search_type and kk == k}):
        v = per_lang[(search_type, k, lang)]
        n = len(v["hit"])
        print(f"  {lang:<6} {n:>4} {sum(v['hit'])/n:>8.3f} "
              f"{sum(v['recall'])/n:>8.3f} {sum(v['mrr'])/n:>8.3f}")
 
 
def _save_csv(results, path):
    """
    Save aggregated results to CSV for easier sharing and plotting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["search_type", "k", "n", "hit@k", "recall@k", "MRR"])
        for (st, k), m in results.items():
            w.writerow([st, k, m["n"], f"{m['hit']:.4f}", f"{m['recall']:.4f}", f"{m['mrr']:.4f}"])


def _save_lang_csv(per_lang, path):
    """
    Save the per-language breakdown (every search_type and k) for plotting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["search_type", "language", "k", "n", "hit@k", "recall@k", "MRR"])
        for (st, k, lang), v in sorted(per_lang.items()):
            n = len(v["hit"])
            w.writerow([st, lang, k, n, f"{sum(v['hit'])/n:.4f}",
                        f"{sum(v['recall'])/n:.4f}", f"{sum(v['mrr'])/n:.4f}"])

 
def main():
    # Load the golden set and evaluate the retriever.
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    retriever = Retriever()
    results, per_lang = evaluate(retriever, golden)
    
    # Print overall results and language breakdown.
    _print_table(results)
    _print_language_breakdown(per_lang, k=APP_K, search_type="similarity")
 
    # Best search_type by recall at the app's k (skipped if that k isn't in the sweep)
    candidates = [(st, k) for (st, k) in results if k == APP_K]
    if candidates:
        best = max(candidates, key=lambda key: results[key]["recall"])
        print(f"\nbest @k={APP_K} by recall: {best[0]} (recall {results[best]['recall']:.3f})")
    
    # Save results to CSV for sharing/plotting.
    out = REPORT_DIR / "retrieval_eval.csv"
    _save_csv(results, out)
    out_lang = REPORT_DIR / "retrieval_eval_by_lang.csv"
    _save_lang_csv(per_lang, out_lang)
    print(f"\nsaved -> {out}, {out_lang.name}")
 
 
if __name__ == "__main__":
    main()