"""
plot_evals.py — draw the evaluation charts from the saved reports.

Reads what each eval script wrote to its reports dir and writes the charts next to it:

  evals/retrieval/  from evaluate_retrieval.py (retrieval_eval.csv, retrieval_eval_by_lang.csv)
    retrieval_eval.png               recall@k and MRR@k vs k, similarity vs MMR
    retrieval_eval_by_language.png   hit@1, hit@k and MRR@k, Portuguese vs English
  evals/generation/  from evaluate_generation.py (generation_eval.json, ragas_per_row.csv
                     + the golden set)
    generation_eval.png              RAGAS + refusal bars, and retrieval distance by type
    generation_eval_by_language.png  the same scores split Portuguese vs English

The k marked on the retrieval charts is the app's retrieval.k. Charts whose inputs are
missing are skipped. No models are loaded, so it runs in seconds. matplotlib lives in
the notebooks group:
  uv run --group notebooks python src/evaluation/plot_evals.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for config
from config import CFG, ROOT  # noqa: E402

RETRIEVAL_DIR = CFG["paths"]["retrieval_reports"]
GENERATION_DIR = CFG["paths"]["generation_reports"]
GOLDEN_PATH = CFG["paths"]["golden"]
APP_K = CFG["retrieval"]["k"]
SEARCH_TYPE_LABELS = {"similarity": "Similarity", "mmr": "MMR"}

NEGATIVE_TYPES = ("out_of_scope", "unanswerable_on_topic")
METRICS = [("faithfulness", "Faithfulness"),
           ("answer_relevancy", "Answer\nrelevancy"),
           ("semantic_similarity", "Semantic\nsimilarity")]
DISTANCE_ROWS = [("answerable", "Answerable", "tab:blue"),
                 ("unanswerable_on_topic", "Unanswerable\non-topic", "tab:orange"),
                 ("out_of_scope", "Out of scope", "tab:green")]
LANGUAGES = [("pt", "Portuguese", "tab:blue"), ("en", "English", "tab:orange")]
DISTANCE_CUTOFF = 0.50   # illustrative only; the app does not filter on distance


def load_reports():
    summary = json.loads((GENERATION_DIR / "generation_eval.json").read_text(encoding="utf-8"))
    per_row = pd.read_csv(GENERATION_DIR / "ragas_per_row.csv")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return summary, per_row, golden


def refusal_by_language(summary, golden):
    """
    {lang: (declined, n)} over the negatives. The report only lists the leaked ids, so
    the per-language split comes from the golden set's language tags.
    """
    leaked = {i for t in NEGATIVE_TYPES for i in summary["refusal"][t]["leaked"]}
    out = {}
    for lang, _, _ in LANGUAGES:
        ids = [r["id"] for r in golden if r["type"] in NEGATIVE_TYPES and r["language"] == lang]
        out[lang] = (sum(i not in leaked for i in ids), len(ids))
    return out


def _label_bars(ax, bars, fmt="{:.2f}", **kw):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", **kw)


def plot_overall(summary, per_row, path):
    judge = summary.get("judge", "local Ollama")
    refusal = summary["refusal"]
    declined = sum(refusal[t]["declined"] for t in NEGATIVE_TYPES)
    n_neg = sum(refusal[t]["n"] for t in NEGATIVE_TYPES)

    scores = [per_row[m].mean() for m, _ in METRICS] + [declined / n_neg]
    labels = [lbl for _, lbl in METRICS] + ["Refusal rate"]
    counts = [per_row[m].notna().sum() for m, _ in METRICS] + [n_neg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    bars = ax1.bar(labels, scores, width=0.6,
                   color=["tab:blue"] * len(METRICS) + ["tab:orange"])
    _label_bars(ax1, bars, fontweight="bold")
    for b, n in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, 0.03, f"n={n}",
                 ha="center", color="white", fontsize=9)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("score")
    ax1.set_title("Answer quality  (RAGAS + refusal)")

    dist = summary["distance"]
    for y, (key, label, color) in enumerate(DISTANCE_ROWS):
        d = dist[key]
        ax2.plot([d["min"], d["max"]], [y, y], color=color, alpha=0.35, linewidth=6)
        ax2.plot([d["min"], d["max"]], [y, y], "o", color=color)
        ax2.plot(d["median"], y, "D", color=color, markersize=10,
                 label="median" if y == 0 else None)
    ax2.axvline(DISTANCE_CUTOFF, color="red", linestyle="--",
                label=f"example cut-off = {DISTANCE_CUTOFF:.2f}")
    ax2.set_yticks(range(len(DISTANCE_ROWS)), [lbl for _, lbl, _ in DISTANCE_ROWS])
    ax2.set_ylim(-0.6, len(DISTANCE_ROWS) - 0.4)
    ax2.set_xlabel("best retrieval distance  (cosine, lower = closer)")
    ax2.set_title("Retrieval distance by question type")
    ax2.legend(loc="lower right")

    fig.suptitle(f"Generation quality — grounded answers & hallucination resistance (judge: {judge})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_by_language(summary, per_row, golden, path):
    judge = summary.get("judge", "local Ollama")
    refusal = refusal_by_language(summary, golden)
    labels = [lbl for _, lbl in METRICS] + ["Refusal\nrate"]
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for offset, (lang, name, color) in zip((-width / 2, width / 2), LANGUAGES):
        rows = per_row[per_row["language"] == lang]
        declined, n_neg = refusal[lang]
        scores = [rows[m].mean() for m, _ in METRICS] + [declined / n_neg if n_neg else 0]
        bars = ax.bar([i + offset for i in x], scores, width,
                      color=color, label=f"{name} (n={len(rows)})")
        _label_bars(ax, bars, fontsize=10)

    neg_note = " · ".join(f"{lang} n={refusal[lang][1]}" for lang, _, _ in LANGUAGES)
    ax.text(len(labels) - 1, 0.05, neg_note, ha="center", fontsize=9, color="0.3")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("score")
    ax.set_title("Answer quality by language (RAGAS + refusal)")
    ax.legend(loc="upper center", ncol=2)
    fig.suptitle(f"Generation — Portuguese vs. English (judge: {judge})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_retrieval(results, path):
    n = int(results["n"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (col, title) in zip(axes, [("recall@k", "Recall@k"), ("MRR", "MRR@k")]):
        for st, rows in results.groupby("search_type", sort=False):
            # categorical x so the uneven sweep (…, 7, 10) stays evenly spaced
            ax.plot(rows["k"].astype(str), rows[col], "o-", label=SEARCH_TYPE_LABELS.get(st, st))
        ks = [str(k) for k in results["k"].drop_duplicates()]
        if str(APP_K) in ks:
            ax.axvline(ks.index(str(APP_K)), color="red", linestyle="--",
                       label=f"app k = {APP_K}")
        ax.set_ylim(0.75, 1.02)
        ax.set_xlabel("k (retrieved chunks)")
        ax.set_ylabel("score")
        ax.set_title(title)
        ax.legend(loc="lower right")
    fig.suptitle(f"Retrieval quality vs. k — similarity vs. MMR ({n} golden questions)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_retrieval_by_language(by_lang, path, search_type="similarity"):
    rows = by_lang[by_lang["search_type"] == search_type]
    metrics = [(1, "hit@k", "Hit@1"), (APP_K, "hit@k", f"Hit@{APP_K}"),
               (APP_K, "MRR", f"MRR@{APP_K}")]
    x = range(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    total = 0
    for offset, (lang, name, color) in zip((-width / 2, width / 2), LANGUAGES):
        lang_rows = rows[rows["language"] == lang].set_index("k")
        if lang_rows.empty:
            continue
        n = int(lang_rows["n"].iloc[0])
        total += n
        scores = [lang_rows.loc[k, col] for k, col, _ in metrics]
        bars = ax.bar([i + offset for i in x], scores, width, color=color, label=f"{name} (n={n})")
        _label_bars(ax, bars, fontsize=10)
    ax.set_xticks(list(x), [lbl for _, _, lbl in metrics])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("score")
    ax.set_title(f"Retrieval quality by language ({search_type})")
    ax.legend(loc="upper center", ncol=2)
    fig.suptitle(f"Retrieval — Portuguese vs. English ({total} answerable questions)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    saved = []

    retrieval_csv = RETRIEVAL_DIR / "retrieval_eval.csv"
    by_lang_csv = RETRIEVAL_DIR / "retrieval_eval_by_lang.csv"
    if retrieval_csv.exists():
        out = RETRIEVAL_DIR / "retrieval_eval.png"
        plot_retrieval(pd.read_csv(retrieval_csv), out)
        saved.append(out)
    if by_lang_csv.exists():
        out = RETRIEVAL_DIR / "retrieval_eval_by_language.png"
        plot_retrieval_by_language(pd.read_csv(by_lang_csv), out)
        saved.append(out)

    if (GENERATION_DIR / "generation_eval.json").exists() and (GENERATION_DIR / "ragas_per_row.csv").exists():
        summary, per_row, golden = load_reports()
        out, out_lang = GENERATION_DIR / "generation_eval.png", GENERATION_DIR / "generation_eval_by_language.png"
        plot_overall(summary, per_row, out)
        plot_by_language(summary, per_row, golden, out_lang)
        saved += [out, out_lang]

    if not saved:
        print("nothing saved (no reports found)")
    for path in saved:
        print(f"saved -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
