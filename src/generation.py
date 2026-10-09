"""
Answer questions about the thesis, grounded in retrieved chunks.

Generation runs locally on Ollama (qwen2.5:7b by default): free, offline and greedy,
and the app and the evaluation use the exact same model, so the evaluation genuinely
predicts what the app does.

Grounding contract (in the system prompt):
  - answer ONLY from the provided context
  - if the context doesn't cover it, say so — never invent or use outside knowledge
  - answer in the SAME language as the question (PT or EN)
The sections used are returned as sources and shown under the answer in the app.
"""

# IMPORTS ---------------------------------------------------------------------------------

import httpx
from langdetect import detect, DetectorFactory, LangDetectException
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config import CFG
from retrieval import Retriever

DetectorFactory.seed = 0   # make langdetect deterministic

# CONFIGURATION --------------------------------------------------------------------------

MODEL = CFG["generation"]["model"]   # Ollama model tag
MAX_NEW_TOKENS = CFG["generation"]["max_new_tokens"]
TEMPERATURE = CFG["generation"]["temperature"]   # 0 = greedy

SYSTEM_PROMPT = (
    "You are a question-answering assistant for a specific master's thesis. "
    "Detect the language of the question (European Portuguese or English) and write your "
    "ENTIRE reply in that language — this applies to refusals too. "
    "Answer ONLY using the provided context excerpts. "
    "If the context does NOT contain the answer, reply with one brief plain sentence stating that "
    "the thesis/document does not cover the question, and nothing more. Use no outside knowledge and invent nothing. "
)

# UTILITY FUNCTIONS -----------------------------------------------------------------------

def _get_llm():
    """
    Build the local Ollama chat model. Greedy (temperature 0) so runs are reproducible.
    """
    return ChatOllama(model=MODEL, temperature=TEMPERATURE, num_predict=MAX_NEW_TOKENS)


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
    if isinstance(exc, httpx.ConnectError):
        return ("ollama_unreachable",
                "Não foi possível contactar o Ollama. Confirme que está a correr (`ollama serve`).\n"
                "Could not reach Ollama. Make sure it is running (`ollama serve`).")
    if getattr(exc, "status_code", None) == 404 and "not found" in str(exc).lower():
        return ("model_missing",
                f"O modelo {MODEL} não está instalado. Execute `ollama pull {MODEL}`.\n"
                f"The model {MODEL} is not installed. Run `ollama pull {MODEL}`.")

    return ("error",
            "Ocorreu um erro ao contactar o modelo de linguagem. Tente novamente.\n"
            "Something went wrong reaching the language model. Please try again.")


class Generator:
    """
    Retrieve + generate. Loads the retriever (with its local bge-m3) and the LLM client once; reuse across queries.
    """

    def __init__(self, retriever=None):
        self.retriever = retriever or Retriever()
        self.llm = _get_llm()

    def answer(self, query, k=None, search_type=None):
        """
        Retrieve top-k chunks and generate an answer grounded in them. Unset k /
        search_type fall back to the retrieval config.
        """
        hits = self.retriever.retrieve(query, k=k, search_type=search_type)

        try:
            response = self.llm.invoke(_build_messages(query, hits))

        except Exception as exc:
            code, message = _error_catch(exc)
            return {"query": query, "answer": message, "sources": hits, "error": code}

        return {"query": query, "answer": response.content, "sources": hits, "error": None}

    def stream(self, query, k=None, search_type=None):
        """
        Streaming version of answer() for the app. Retrieves right away and returns
        (result, chunks): chunks is an iterator of answer text that calls the LLM lazily,
        and result (same keys as answer()) is filled in as it's consumed. On an LLM error
        the friendly message is streamed as well and result["error"] is set.
        """
        hits = self.retriever.retrieve(query, k=k, search_type=search_type)
        result = {"query": query, "answer": "", "sources": hits, "error": None}

        def chunks():
            try:
                for chunk in self.llm.stream(_build_messages(query, hits)):
                    result["answer"] += chunk.content
                    yield chunk.content
            except Exception as exc:
                code, message = _error_catch(exc)
                message = f"\n\n{message}" if result["answer"] else message   # after partial text
                result["answer"] += message
                result["error"] = code
                yield message

        return result, chunks()


def main():
    # Create a generator object (loads retriever + LLM once)
    generator = Generator()

    # Smoke test: a quick query should return an answer + cited sections
    for query in ["Qual é o objetivo do estudo?",
                  "What software was used to simulate the processes?"]:

        result = generator.answer(query)
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}\n")

        cited = ", ".join(f"[{s['number']}] {s['title']}" for s in result["sources"])
        print(f"sources: {cited}\n{'-'*60}")


if __name__ == "__main__":
    main()
