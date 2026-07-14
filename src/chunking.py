"""
Turn cleaned thesis sections into embeddable chunks.
"""

# IMPORTS ---------------------------------------------------------------------------------

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# CONFIGURATION --------------------------------------------------------------------------

CLEANED_SECTIONS_PATH = Path("data/processed/cleaned_sections.json")
CHUNK_OUTPUT_PATH = Path("data/processed/chunks.json")
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

# UTILITY FUNCTIONS -----------------------------------------------------------------------

def _get_token_counter(model=EMBEDDING_MODEL):
    """
    Return a function that counts tokens for the given model
    """
    tokenizer = AutoTokenizer.from_pretrained(model)

    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    return count_tokens



def _split(text, chunk_size, overlap, count_tokens):
    """
    Split text into ~chunk_size-token pieces on natural boundaries, with overlap
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

def _build_heading(section):
    """
    Join chapter title and title with ' > ', skipping any that are missing
    """
    parts = (section.get("chapter_title"), section.get("title"))
    
    return " > ".join(part for part in parts if part)


def _build_section_id(section):
    """
    Build a filesystem-friendly id from the section number or title.
    """
    raw = section.get("number") or section.get("title")
    
    return str(raw).replace(" ", "_")


def _make_chunk(section, raw_text, text, token_count, *, chunk_id):
    """
    Assemble the chunk dict that gets embedded and returned
    """
    
    return {
        "chunk_id": chunk_id,
        "text": text,             # embedded (title-prepended)
        "raw_text": raw_text,     # shown to the user
        "token_count": token_count,
        "number": section.get("number"),
        "title": section.get("title"),
        "chapter_title": section.get("chapter_title"),
        "section_number": section.get("section_number"),
        "section_title": section.get("section_title"),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        "level": section.get("level"),
    }

def chunk_sections(sections, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                   min_tokens=50, prepend_title=True, count_tokens=None):
    """
    Return a flat list of chunk dicts ready to embed
    """
    count_tokens = count_tokens or _get_token_counter()
 
    chunks = []
    for section in sections:
        heading = _build_heading(section)
        section_id = _build_section_id(section)
        pieces = _split(section["text"], chunk_size, overlap, count_tokens)
 
        for i, piece in enumerate(pieces):
            text = f"{heading}\n\n{piece}" if (prepend_title and heading) else piece
            token_count = count_tokens(text)
            if token_count < min_tokens:
                continue
 
            chunks.append(_make_chunk(
                section, piece, text, token_count,
                chunk_id=f"{section_id}__{i:03d}",
            ))
 
    return chunks

def main():
    # Read the cleaned sections from the JSON file
    sections = json.loads(CLEANED_SECTIONS_PATH.read_text(encoding="utf-8"))

    # Create chunks from sections
    chunks = chunk_sections(sections)

    # Quick sanity checks
    assert chunks, "no chunks produced"
    assert all(chunk["raw_text"].strip() for chunk in chunks), "empty chunk"
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks), "duplicate ids"

    # Stats for eyeballing the distribution and tuning chunk_size if needed
    token_counts = [chunk["token_count"] for chunk in chunks]
    sections_covered = len({chunk["number"] for chunk in chunks})
    print(
        f"{len(chunks)} chunks | "
        f"tokens: mean {sum(token_counts)//len(token_counts)}, "
        f"min {min(token_counts)}, max {max(token_counts)} | "
        f"sections covered {sections_covered}"
    )

    # Save the chunks to a JSON file
    CHUNK_OUTPUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {CHUNK_OUTPUT_PATH}")   


if __name__ == "__main__":
    main()