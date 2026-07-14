"""
Stage 2 of Preprocessing: Clean the extracted text.

What this does:
- Remove page numbers (lines that are just a number)
- Remove figure/table caption lines (e.g., "Figura 3 - ...")
- Fix hyphenation across line breaks:
  - True line-wrap hyphenation (lowercase continuation, NOT a clitic).    "melho-\nria"  ->  "melhoria"
  - Portuguese clitic pronoun (lowercase continuation IS a clitic).  "esperam-\nse"  ->  "esperam-se"   (keep the hyphen!)
  - Compound term wrap (uppercase continuation).  "Just-in-\nTime"  ->  "Just-in-Time"

Output: cleaned_sections.json
"""

# IMPORTS ---------------------------------------------------------------------------------

import json
import re
from pathlib import Path
import unicodedata



# CONFIGURATION --------------------------------------------------------------------------

INPUT_PATH = Path("data/processed/extracted_sections.json")
OUTPUT_PATH = Path("data/processed/cleaned_sections.json")



# CLEANING PATTERNS -----------------------------------------------------------------------------

# A line that is JUST a page number — "32" or "117"
# any pure-digit short line inside the body is a page footer.
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")

# End-of-line hyphenation: a word that ends with "-" right before a newline
# Example matches:    "melho-\nria"   -> "melhoria"
#                     "implementa-\nção"  -> "implementação"
HYPHEN_BREAK_RE = re.compile(r"(\w+)-\n([a-záéíóúâêôãõç]+)")

# Compound term wrap: a hyphenated compound that wraps at the hyphen,
# where the second part starts with uppercase (like "Just-in-\nTime" or "On-\nShelf"). 
COMPOUND_WRAP_RE = re.compile(r"(\w+)-\n\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇ]\w+)")

# Portuguese clitic pronouns
#   "esperam-\nse"  must become  "esperam-se"  (not "esperamse"), etc.
PT_CLITICS = {
    "se", "me", "te", "nos", "vos", "lhe", "lhes",
    "o", "a", "os", "as",
    "lo", "la", "los", "las",
    "no", "na",
}

# Figure/table caption lines: "Figura 3 - Processo de simulação ..."
# The dash after the number is what separates a CAPTION from a
# reference in prose ("como mostra a Figura 3, ..."), which we keep.
CAPTION_RE = re.compile(
    r"^\s*(Figura|Tabela|Gráfico|Equação)\s+\d+\s*[-–—]",
    re.IGNORECASE,
)


# CLEANING FUNCTIONS -----------------------------------------------------------------------------

def _remove_page_numbers(text):
    """
    Remove lines that are just page numbers.
    """
    lines = text.split("\n")
    kept = [line for line in lines if not PAGE_NUMBER_RE.match(line)]

    return "\n".join(kept)

def _remove_captions(text):
    """
    Drop caption lines ("Figura 3 - ..."). Must run BEFORE paragraph
    reconstruction, while captions are still on their own lines.
    """
    lines = text.split("\n")
    kept = [line for line in lines if not CAPTION_RE.match(line)]
    return "\n".join(kept)

def _replace_lowercase_break(match):
    """
    Helper for HYPHEN_BREAK_RE: decide whether to keep the hyphen or join the word
    """
    before = match.group(1)
    after = match.group(2)
    # Is the continuation a Portuguese clitic? Check the *whole*
    # continuation word, not just a prefix — "se" is a clitic but
    # "selecionar" is not, even though both start with "se".
    # We extract the word that follows: continuation might be just
    # the clitic or the clitic followed by punctuation/space.
    # Simplest check: is `after` exactly a clitic?
    if after in PT_CLITICS:
        # Real grammatical hyphen: keep it, just drop the newline
        return f"{before}-{after}"
    # Compound connectors: the middle links of hyphenated compounds
    # like "mão-DE-obra", "dia-A-dia", "fim-DE-semana". If the fragment
    # before the break IS one of these, the hyphen is orthographic
    # (part of the word), not a line-wrap artifact — keep it.
    if before.lower() in {"de", "a", "e", "do", "da"}:
        return f"{before}-{after}"
    # Real line-wrap hyphenation: join the word
    return f"{before}{after}"

def _fix_hyphenation(text):
    """
    Handle three kinds of hyphens across line breaks:

    1. True line-wrap hyphenation (lowercase continuation, NOT a clitic).
       "melho-\nria"  ->  "melhoria"

    2. Portuguese clitic pronoun (lowercase continuation IS a clitic).
       "esperam-\nse"  ->  "esperam-se"   (keep the hyphen!)

    3. Compound term wrap (uppercase continuation).
       "Just-in-\nTime"  ->  "Just-in-Time"

    We use a single regex and decide what to do based on whether the
    continuation is in the clitic set.
    """

    # Case 1+2: lowercase continuation (regular word OR clitic)
    text = HYPHEN_BREAK_RE.sub(_replace_lowercase_break, text)
    # Case 3: uppercase continuation -> keep hyphen, drop newline+space
    text = COMPOUND_WRAP_RE.sub(r"\1-\2", text)
    
    return text


def _reconstruct_paragraphs(text):
    """
    Within each paragraph, collapse line breaks to single spaces.
    Paragraphs are separated by blank lines (one or more).

    Algorithm:
    1. Split on \n\n+ (one or more blank lines) to get paragraphs
    2. Within each paragraph, replace remaining \n with single space
    3. Rejoin with \n\n

    This reconstructs the original sentence flow while preserving
    paragraph structure.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paragraphs = []
    
    for para in paragraphs:
        # Collapse internal line breaks to spaces
        merged = re.sub(r"\s*\n\s*", " ", para).strip()
        if merged:  # skip empty
            cleaned_paragraphs.append(merged)
    
    return "\n\n".join(cleaned_paragraphs)


def _normalize_whitespace(text):
    """
    Collapse multiple spaces, trim each line.
    """
    # Multiple spaces -> single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Trim trailing whitespace per line
    text = "\n".join(line.strip() for line in text.split("\n"))
    
    # Collapse 3+ newlines to exactly 2 (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def clean_text(text):
    """
    Apply all cleaning steps in the correct order.
    """
    # 1. Unicode normalization to ensure consistent character forms (mathematical symbols, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2a. Remove page numbers — they can interfere with hyphenation fixes and paragraph merging, so do this early.
    text = _remove_page_numbers(text)

    # 2b. Remove caption lines — same logic: whole isolated lines, must go before line-merging fuses them into sentences
    text = _remove_captions(text)

    # 3. Fix hyphenation — needs the \n positions to find end-of-line hyphens
    text = _fix_hyphenation(text)

    # 4. Reconstruct paragraphs — collapses within-paragraph line breaks
    text = _reconstruct_paragraphs(text)

    # 5. Final whitespace normalization
    text = _normalize_whitespace(text)

    return text



# MAIN -----------------------------------------------------------------------------

def main():
    # Load extracted sections from JSON
    with open(INPUT_PATH, encoding="utf-8") as f:
        sections = json.load(f)

    # Apply cleaning to each section's text
    for s in sections:
        s["text"] = clean_text(s["text"])

    # Save the extracted sections to a JSON file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
