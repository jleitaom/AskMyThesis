"""
Answer questions about the thesis, grounded in retrieved chunks.

Two interchangeable backends, selected by the LLM_BACKEND env var. Both run the SAME
model, so the local evaluation genuinely predicts production behaviour:
  - "ollama" (default) — local qwen2.5:7b via Ollama. Free, offline; this is what the
    evaluation scripts use so heavy RAGAS runs don't burn API quota.
  - "hf" — the same model on Hugging Face Inference Providers, via
    HuggingFaceEndpoint(provider="auto") wrapped in ChatHuggingFace. Used for
    deployment (no local weights load). Auth uses HUGGINGFACEHUB_API_TOKEN (from .env).

Grounding contract (in the system prompt):
  - answer ONLY from the provided context
  - if the context doesn't cover it, say so — never invent or use outside knowledge
  - cite the section number(s) used
  - answer in the SAME language as the question (PT or EN)
"""

# IMPORTS ---------------------------------------------------------------------------------

import os

from dotenv import load_dotenv
from langdetect import detect, DetectorFactory, LangDetectException
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval import Retriever

load_dotenv()   # pick up HUGGINGFACEHUB_API_TOKEN from .env for the "hf" backend
DetectorFactory.seed = 0   # make langdetect deterministic

# CONFIGURATION --------------------------------------------------------------------------

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")   # "ollama" (local/eval) or "hf" (deploy)
OLLAMA_MODEL = "qwen2.5:7b"              # ollama tag; use "qwen2.5:3b" if RAM is tight
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # same model as OLLAMA_MODEL, on HF Inference
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a question-answering assistant for a specific master's thesis. "
    "Detect the language of the question (European Portuguese or English) and write your "
    "ENTIRE reply in that language — this applies to refusals too. "
    "Answer ONLY using the provided context excerpts. "
    "If the context does NOT contain the answer, reply with one brief plain sentence stating that "
    "the thesis/document does not cover the question, and nothing more. Use no outside knowledge and invent nothing. "
)

# UTILITY FUNCTIONS -----------------------------------------------------------------------

def _get_llm(backend=None):
    """
    Build the chat model for the selected backend. Both are deterministic (greedy)
    so eval runs are reproducible.
    """
    backend = backend or LLM_BACKEND

    if backend == "ollama":
        # Local, offline, free — used for evaluation.
        return ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=MAX_NEW_TOKENS)

    if backend == "hf":
        # Same model on HF Inference Providers — used for deployment. ChatHuggingFace
        # wraps the raw endpoint so it accepts SystemMessage/HumanMessage lists.
        endpoint = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            task="text-generation",
            provider="auto",            # let HF route to an available provider
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,            # greedy -> deterministic, good for eval
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Unknown LLM_BACKEND {backend!r}; use 'ollama' or 'hf'")


def _build_context(hits):
    """
    Format retrieved chunks into a labelled context block the model can cite.
    """
    blocks = []

    for hit in hits:
        label = f"[{hit.get('number')}] {hit.get('title')}"
        body = hit.get("raw_text") or hit.get("text")
        blocks.append(f"{label}\n{body}")

    return "\n\n---\n\n".join(blocks)


def _reply_language(query):
    """
    Detect the query's language and return the name to instruct the model with. The
    thesis context is Portuguese, which pulls a small model toward PT even for English
    questions (worst on refusals, where there's no answer to anchor the language) — so
    we detect deterministically and force the reply language instead of trusting the
    model. Anything that isn't Portuguese is treated as English (the app is PT/EN only).
    """
    try:
        return "European Portuguese" if detect(query) == "pt" else "English"
    except LangDetectException:
        return "English"


def _build_messages(query, hits):
    """
    Build the system + human messages for the LLM, including the retrieved context and a
    hard directive (next to the question, where it's most salient) fixing the reply language.
    """
    context = _build_context(hits)
    directive = f"Write your ENTIRE reply in {_reply_language(query)}, including any refusal."
    human = f"Context excerpts from the thesis:\n\n{context}\n\n{directive}\n\nQuestion: {query}"

    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]

def _error_catch(exc):
    """
    Map an LLM-call exception to a (code, user-facing message) pair, so the app
    shows a clean message instead of a stack trace. Messages are bilingual (PT/EN)
    since the app answers in either language.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc).lower()

    if status == 402 or any(w in text for w in ("exceeded", "credit", "payment required", "quota")):
        return ("quota_exceeded",
                "O limite de utilização mensal foi atingido. Tente novamente mais tarde.\n"
                "The monthly usage limit has been reached. Please try again later.")
    if status == 429 or any(w in text for w in ("rate limit", "too many requests")):
        return ("rate_limited",
                "O assistente está ocupado. Aguarde um momento e tente novamente.\n"
                "The assistant is busy right now. Please wait a moment and try again.")
    if status == 503 or any(w in text for w in ("loading", "unavailable", "503")):
        return ("model_unavailable",
                "O modelo está a iniciar. Tente novamente dentro de alguns segundos.\n"
                "The model is starting up. Please retry in a few seconds.")

    return ("error",
            "Ocorreu um erro ao contactar o modelo de linguagem. Tente novamente.\n"
            "Something went wrong reaching the language model. Please try again.")


class Generator:
    """
    Retrieve + generate. Loads the retriever (with its local bge-m3) and the LLM client once; reuse across queries.

    backend: "ollama" | "hf" | None (None -> LLM_BACKEND env, default "ollama").
    """

    def __init__(self, retriever=None, backend=None):
        self.retriever = retriever or Retriever()
        self.llm = _get_llm(backend)

    def answer(self, query, k=5, search_type="similarity"):
        """
        Retrieve top-k chunks and generate an answer grounded in them.
        """
        hits = self.retriever.retrieve(query, k=k, search_type=search_type)

        try:
            response = self.llm.invoke(_build_messages(query, hits))

        except Exception as exc:
            code, message = _error_catch(exc)
            return {"query": query, "answer": message, "sources": hits, "error": code}

        return {"query": query, "answer": response.content, "sources": hits, "error": None}


def main():
    # Create a generator object (loads retriever + LLM once)
    generator = Generator()

    # Smoke test: a quick query should return an answer + cited sections
    for query in ["Qual é o objetivo do estudo?",
                  "What software was used to simulate the processes?"]:

        result = generator.answer(query, k=5)
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}\n")

        cited = ", ".join(f"[{s['number']}] {s['title']}" for s in result["sources"])
        print(f"sources: {cited}\n{'-'*60}")


if __name__ == "__main__":
    main()
