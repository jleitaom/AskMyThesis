"""
Query the persisted Chroma index.

Chroma rewrites its files whenever a client opens them, even for read-only queries,
which would leave the git-tracked index dirty after every run. So the Retriever opens
a temporary copy of the index and the files under data/chroma are never touched.
"""

# IMPORTS ---------------------------------------------------------------------------------

import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CFG

# CONFIGURATION --------------------------------------------------------------------------

CHROMA_DIR = CFG["paths"]["chroma_dir"]
COLLECTION_NAME = CFG["index"]["collection"]
EMBEDDING_MODEL = CFG["embedding"]["model"]
K = CFG["retrieval"]["k"]
SEARCH_TYPE = CFG["retrieval"]["search_type"]
FETCH_K = CFG["retrieval"]["fetch_k"]
 
# UTILITY FUNCTIONS -----------------------------------------------------------------------

def _get_embeddings(model=EMBEDDING_MODEL):
    """
    Get embeddings for the given model, normalized so cosine similarity behaves
    """
    return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})


def _empty_index_message():
    return (f"Collection '{COLLECTION_NAME}' at {CHROMA_DIR} is missing or empty — "
            f"build it with `python src/indexing.py` from the repo root.")


def _copy_index(tmp_dir):
    """
    Copy the persisted index into tmp_dir and return the copy's path, so opening it
    can't modify the tracked files.
    """
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(_empty_index_message())
    copy_dir = Path(tmp_dir) / "chroma"
    shutil.copytree(CHROMA_DIR, copy_dir)

    return copy_dir

 
def _load_store(embeddings, persist_dir, model=EMBEDDING_MODEL):
    """
    Open the collection at persist_dir and fail loudly if it's empty or was built
    with a different embedding model.
    """
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    collection = store._collection
    if collection.count() == 0:
        raise RuntimeError(_empty_index_message())
    stamped = (collection.metadata or {}).get("embedding_model")
    if stamped and stamped != model:
        raise RuntimeError(
            f"Embedding model mismatch: index built with {stamped!r} but querying "
            f"With {model!r}. Rebuild the index or set embedding.model in the config to match."
        )
    
    return store
 
 
def _format(doc, score):
    """
    Flatten a (Document, score)
    """
    return {"score": score, "text": doc.page_content, **doc.metadata}
 
 
class Retriever:
    """
    Loads the model + index once, then answers queries. The index is opened from a
    temporary copy that is deleted when the Retriever is garbage-collected.
    """

    def __init__(self, embedding_model=EMBEDDING_MODEL):
        self.embeddings = _get_embeddings(embedding_model)
        self._tmp = tempfile.TemporaryDirectory(prefix="askmythesis-chroma-")
        self.store = _load_store(self.embeddings, _copy_index(self._tmp.name), embedding_model)
 
    def retrieve(self, query, k=None, search_type=None, fetch_k=None):
        """
        Return the top-k chunks as dicts. Unset arguments fall back to the config.

        search_type:
          "similarity" — plain cosine top-k, includes a distance score.
          "mmr"        — Maximal Marginal Relevance
        """
        k = k or K
        search_type = search_type or SEARCH_TYPE
        fetch_k = fetch_k or FETCH_K

        if search_type == "similarity":
            pairs = self.store.similarity_search_with_score(query, k=k)
            return [_format(doc, float(score)) for doc, score in pairs]
        
        if search_type == "mmr":
            docs = self.store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
            return [_format(doc, None) for doc in docs]
        
        raise ValueError(f"Unknown search_type {search_type!r}; use 'similarity' or 'mmr'")
 
 
def main():
    # Create retriever object (loads bge-m3 + the index once)
    retriever = Retriever()

    # Smoke test: a PT and an EN query, in both search modes
    for query in ["Qual é o objetivo do estudo?",
                  "What software was used to simulate the processes?"]:
        print(f"Q: {query}")
        for search_type in ("similarity", "mmr"):
            print(f"  {search_type}:")
            for hit in retriever.retrieve(query, search_type=search_type):
                score = f"distance {hit['score']:.3f}" if hit["score"] is not None else "(no score)"
                print(f"    [{hit.get('number')}] {hit.get('title')}  {score}")
        print("-" * 60)
 
 
if __name__ == "__main__":
    main()