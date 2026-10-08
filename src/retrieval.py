"""
Query the persisted Chroma index.
"""

# IMPORTS ---------------------------------------------------------------------------------

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
 
 
def _load_store(embeddings, model=EMBEDDING_MODEL):
    """
    Open the persisted collection and fail loudly if it's empty or was built
    with a different embedding model.
    """
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    collection = store._collection
    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' at {CHROMA_DIR} is empty — "
            f"Build it with `python -m src.indexing` from the repo root."
        )
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
    Loads the model + index once, then answers queries
    """
 
    def __init__(self, embedding_model=EMBEDDING_MODEL):
        self.embeddings = _get_embeddings(embedding_model)
        self.store = _load_store(self.embeddings, embedding_model)
 
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
    # Create retriever object
    retriever = Retriever()
 
 
if __name__ == "__main__":
    main()