"""
evaluate_generation.py — grade the answers the LLM writes (the generation half).
 
Retrieval is covered by evaluate_retrieval.py; this grades generation. All local
(Ollama judge + bge-m3), so free. Runs three checks, in order:
 
  1. Refusal eval on the 50 negatives (27 out_of_scope + 23 unanswerable_on_topic)
     — does the assistant correctly decline instead of inventing an answer?
     unanswerable_on_topic is the hard case (on-topic context is retrieved but holds
     no answer), so its refusal rate is the real hallucination-resistance number.
     A yes/no Ollama judge decides.
  2. Retrieval distance per question type (free, deterministic) — shows that a
     distance cutoff could screen out_of_scope questions but NOT
     unanswerable_on_topic, which is why check 1 exists.
  3. RAGAS on the 70 answerable questions — faithfulness + answer_relevancy (vs the
     retrieved context) and semantic_similarity vs the golden reference_answer.
     factual_correctness is available but disabled by default; see RAGAS_METRICS.
 
Generated answers are cached to REPORT_DIR/answers_cache.json, keyed by question and
tagged with a signature of the generation setup (model + system prompt). Re-runs reuse
the cache and skip the slow generation step; if the signature changes (you switch model
or edit the prompt) the cache is discarded and answers regenerate. Set USE_CACHE=False
to force regeneration. Drop SAMPLE to a small number while developing.
"""
 
import csv
import hashlib
import json
import statistics
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import Faithfulness, ResponseRelevancy, FactualCorrectness, SemanticSimilarity
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
 
from generation import Generator, SYSTEM_PROMPT

GOLDEN_PATH = Path("data/golden/golden.json")
REPORT_DIR = Path("evals/reports")
ANSWERS_CACHE = REPORT_DIR / "answers_cache.json"
USE_CACHE = True                # reuse cached answers when the generation signature matches
 
JUDGE_MODEL = "qwen2.5:14b"      # local Ollama judge, same family as generation
JUDGE_NUM_PREDICT = 4096         # RAGAS claim-decomposition JSON is long (esp. in PT);
                                 # 1024 truncated it mid-object and broke the parser.
NEGATIVE_TYPES = ("out_of_scope", "unanswerable_on_topic")
SAMPLE = None                     # set to e.g. 5 to only run the first few questions

# Ollama processes requests serially, so high RAGAS concurrency just makes jobs
# queue and time out (that's what produced the nan faithfulness). Keep workers
# low and the per-job timeout high. Drop workers to 1 if you still see timeouts.
RAGAS_MAX_WORKERS = 2
RAGAS_TIMEOUT = 600             # seconds per metric job (faithfulness is the slow one)
RAGAS_SAMPLE = None             # cap how many answerable answers RAGAS scores (None = all);
                                # each answer x metric is a slow local-judge job, so a
                                # subset (e.g. 25) keeps the run to ~1h without losing signal.
 
# Comment a line to skip that metric. semantic_similarity needs reference_answer.
# factual_correctness is disabled: its claim_decomposition step needs strict JSON the
# local 14B judge can't produce reliably (RagasOutputParserException, even with a large
# num_predict). Re-enable it if you switch to a stronger judge (e.g. a hosted GPT-4-class).
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    # "factual_correctness",
    "semantic_similarity",
]
 
REFUSAL_SYSTEM = (
    "You decide whether an assistant DECLINED to answer a question. It was told to "
    "answer only from a thesis and to say the thesis doesn't cover the question when "
    "the answer isn't there. Answer 'yes' if it declined (said it couldn't find the "
    "answer / the thesis doesn't cover it / refused). Answer 'no' if it gave a real "
    "answer. Reply with exactly one word: yes or no."
)
 
 
def get_judge():
    """
    Local Ollama judge, temperature 0 so judging is deterministic.
    """
    
    return ChatOllama(model=JUDGE_MODEL, temperature=0, num_predict=JUDGE_NUM_PREDICT)
 
 
def _generation_signature(generator):
    """
    Fingerprint of what produces an answer — the model and the system prompt. If either
    changes, cached answers are stale and must be regenerated.
    """
    model = getattr(generator.llm, "model", type(generator.llm).__name__)
    return hashlib.sha256(f"{model}||{SYSTEM_PROMPT}".encode("utf-8")).hexdigest()[:16]


def generate_answers(generator, golden, cache_path=ANSWERS_CACHE, use_cache=USE_CACHE):
    """
    Run the generator over every question once; return records aligned to golden.

    Answers are cached per question in cache_path so re-runs skip the slow generation
    step. The cache is tagged with the generation signature; if it no longer matches
    (model or prompt changed) the cache is dropped and everything regenerates.
    """
    signature = _generation_signature(generator)
    answers = {}   # question -> {"answer", "error", "sources"}
    if use_cache and cache_path.exists():
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        if blob.get("signature") == signature:
            answers = blob.get("answers", {})
        else:
            print("generation signature changed — ignoring stale answer cache.")

    records, generated = [], 0
    for i, row in enumerate(golden, start=1):
        q = row["question"]
        cached = answers.get(q) if use_cache else None
        if cached is not None:
            print(f"Answer {i}/{len(golden)} (cached)")
            result = cached
        else:
            print(f"Generating answer {i}/{len(golden)}...")
            result = generator.answer(q)
            answers[q] = {"answer": result["answer"], "error": result.get("error"),
                          "sources": result.get("sources", [])}
            generated += 1

        records.append({
            "id": row["id"],
            "question": q,
            "type": row["type"],
            "language": row["language"],
            "reference_answer": row.get("reference_answer"),
            "answer": result["answer"],
            "error": result.get("error"),
            "sources": result.get("sources", []),
        })

    if use_cache and generated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"signature": signature, "answers": answers}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"cached {len(answers)} answers -> {cache_path} ({generated} newly generated)")

    return records
 
 
# --- check 1: refusal on the negatives ------------------------------------- #
 
def is_refusal(judge, question, answer):
    """
    Ask the judge whether the answer declined the question.
    """
    
    human = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nDid the assistant decline? (yes/no)"
    reply = judge.invoke([SystemMessage(content=REFUSAL_SYSTEM), HumanMessage(content=human)])
    
    return reply.content.strip().lower().startswith("y")
 
 
def run_refusal(records, judge):
    """
    Refusal rate per negative type. 'leaked' = ids answered that should've been declined.
    """
    out = {t: {"n": 0, "declined": 0, "leaked": []} for t in NEGATIVE_TYPES}
    
    for r in records:
        print(f"Running refusal evaluation for answer {r['id']}...")
        if r["type"] not in NEGATIVE_TYPES or r["error"]:
            continue
        bucket = out[r["type"]]
        bucket["n"] += 1
        if is_refusal(judge, r["question"], r["answer"]):
            bucket["declined"] += 1
        else:
            bucket["leaked"].append(r["id"])
    
    return out
 
 
# --- check 2: retrieval distance by type ----------------------------------- #
 
def distance_by_type(records):
    """
    Smallest cosine distance per question, grouped by type (lower = closer).
    """
 
    buckets = {}
    for r in records:
        scores = [s["score"] for s in r["sources"] if s.get("score") is not None]
        if scores:
            buckets.setdefault(r["type"], []).append(min(scores))
    return {
        t: {"n": len(v), "min": min(v), "median": statistics.median(v),
            "mean": statistics.fmean(v), "max": max(v)}
        for t, v in buckets.items()
    }
 
 
# --- check 3: RAGAS on the answerable -------------------------------------- #
 
def run_ragas(records, embeddings, judge):
    """
    Score answerable answers with RAGAS. Returns a per-row DataFrame, or None.
    """
 
    available = {
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "factual_correctness": FactualCorrectness,
        "semantic_similarity": SemanticSimilarity,
    }
    metrics = [available[name]() for name in RAGAS_METRICS]
 
    answerable = [r for r in records if r["type"] == "answerable" and not r["error"]]
    if RAGAS_SAMPLE:
        answerable = answerable[:RAGAS_SAMPLE]
    if not answerable:
        return None
 
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "retrieved_contexts": [s.get("raw_text") or s.get("text") for s in r["sources"]],
            "response": r["answer"],
            "reference": r["reference_answer"],
        }
        for r in answerable
    ])
 
    print(f"\n[ragas] scoring {len(answerable)} answers with {RAGAS_METRICS} (slow, local judge)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(embeddings),   # reuse the loaded bge-m3
        run_config=RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=RAGAS_TIMEOUT),
    )
 
    df = result.to_pandas()
    df.insert(0, "language", [r["language"] for r in answerable])
    df.insert(0, "id", [r["id"] for r in answerable])
    
    return df
 
 
# --- printing + saving ----------------------------------------------------- #
 
def print_refusal(refusal):
    print("\n=== Refusal on negatives ===")
    for t in NEGATIVE_TYPES:
        b = refusal[t]
        rate = b["declined"] / b["n"] if b["n"] else 0.0
        print(f"  {t:<22} {b['declined']}/{b['n']}  rate {rate:.3f}")
    leaked = [i for t in NEGATIVE_TYPES for i in refusal[t]["leaked"]]
    if leaked:
        print(f"  leaked (answered instead of declined): {', '.join(leaked)}")
 
 
def print_distance(stats):
    print("\n=== Best retrieval distance by type (lower = closer) ===")
    print(f"  {'type':<22} {'n':>3} {'min':>6} {'median':>7} {'mean':>6} {'max':>6}")
    for t in ("answerable", *NEGATIVE_TYPES):
        if t in stats:
            s = stats[t]
            print(f"  {t:<22} {s['n']:>3} {s['min']:>6.3f} {s['median']:>7.3f} "
                  f"{s['mean']:>6.3f} {s['max']:>6.3f}")
 
 
def print_ragas(df):
    if df is None:
        print("\n=== RAGAS: no answerable rows ===")
        return
    # a metric whose jobs all failed has no column in the result — skip it
    # instead of crashing on a KeyError.
    present = [name for name in RAGAS_METRICS if name in df.columns]
    missing = [name for name in RAGAS_METRICS if name not in df.columns]
    print(f"\n=== RAGAS (n={len(df)}) ===")
    for name in present:
        print(f"  {name:<22} {df[name].mean():.3f}")
    if missing:
        print(f"  (no results for: {', '.join(missing)} — all jobs failed/timed out)")
    print("  by language:")
    for lang, sub in df.groupby("language"):
        scores = "  ".join(f"{name}={sub[name].mean():.3f}" for name in present)
        print(f"    {lang} (n={len(sub)}): {scores}")
 
 
def _safe_mean(series):
    """Column mean, but return None (not NaN) so the JSON stays valid."""
    m = series.mean()
    return None if m != m else float(m)   # m != m is True only for NaN
 
 
def save_report(refusal, stats, df):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"refusal": refusal, "distance": stats}
    if df is not None:
        present = [name for name in RAGAS_METRICS if name in df.columns]
        summary["ragas"] = {name: _safe_mean(df[name]) for name in present}
        df[["id", "language", *present]].to_csv(REPORT_DIR / "ragas_per_row.csv", index=False)
    (REPORT_DIR / "generation_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {REPORT_DIR}/generation_eval.json")
 
 
def main():
 
    # Load golden dataset
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if SAMPLE:
        golden = golden[:SAMPLE]
 
    # Create generator object and generate answers. Pin the local backend so the
    # eval always runs on Ollama (same model as HF deploy) regardless of LLM_BACKEND.
    generator = Generator(backend="ollama")
    print(f"generating answers for {len(golden)} questions...")
    records = generate_answers(generator, golden)
 
    # Load judge model
    judge = get_judge()
    
    # Generate refusal evaluation
    refusal = run_refusal(records, judge)
    print_refusal(refusal)
 
    # Generate retrievals
    stats = distance_by_type(records)
    print_distance(stats)
    
    # Generate RAGAS evaluation
    df = run_ragas(records, generator.retriever.embeddings, judge)
    print_ragas(df)
 
    # Save final report
    save_report(refusal, stats, df)
 
 
if __name__ == "__main__":
    main()