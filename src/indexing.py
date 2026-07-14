"""
Embed chunks with bge-m3 and persist a Chroma collection.
"""

# IMPORTS ---------------------------------------------------------------------------------

import json
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# CONFIGURATION --------------------------------------------------------------------------

CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "thesis"
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 500  # stamped into collection metadata for provenance
 
# UTILITY FUNCTIONS -----------------------------------------------------------------------

def _get_embeddings(model=EMBEDDING_MODEL):
    """
    Get embeddings for the given model, normalized so cosine similarity behaves
    """
    return HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True},
    )
 
 
def _clean_metadata(chunk):
    """
    Return a dict of metadata for a chunk, excluding the text itself
    """
    return {k: v for k, v in chunk.items() if k != "text" and v is not None}
 
 
def _to_documents(chunks):
    """
    Convert chunk dicts to LangChain Documents (+ their ids)
    """
    documents, ids = [], []
    
    for chunk in chunks:
        documents.append(Document(
            page_content=chunk["text"],          # title-prepended text gets embedded
            metadata=_clean_metadata(chunk),     # raw_text + scalar fields
        ))
        ids.append(chunk["chunk_id"])
    
    return documents, ids
 
 
def build_index(chunks, embeddings=None):
    """
    Wipe any existing collection and build a fresh persisted Chroma index
    """
    embeddings = embeddings or _get_embeddings()
 
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
 
    documents, ids = _to_documents(chunks)
    
    return Chroma.from_documents(
        documents=documents,
        ids=ids,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
        },
    )
 
 
def main():
    # Load chunks.json
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    
    # Build the Chroma index
    store = build_index(chunks)
    print(f"indexed {len(chunks)} chunks -> {CHROMA_DIR} (collection '{COLLECTION_NAME}')")
        
    print("Chroma index built.")
 
 
if __name__ == "__main__":
    main()