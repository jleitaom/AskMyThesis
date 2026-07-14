"""
Stage 1 of Preprocessing: Extract structured text from the thesis PDF.

What this does:
- Reads the thesis PDF file (data/raw/thesis.pdf)
- Extracts text from each page, ignoring tables
- Detects headings (chapters, sections, subsections) based on numbering patterns

Output: extracted_sections.json
"""

# IMPORTS ---------------------------------------------------------------------------------

import json
import re
import pdfplumber
from pathlib import Path



# CONFIGURATION --------------------------------------------------------------------------

PDF_PATH = Path("data/raw/thesis.pdf")
OUTPUT_PATH = Path("data/processed/extracted_sections.json")

PDF_PAGE_START = 20 # 14
PDF_PAGE_END = 137  # 137

# METADATA INJECTION  --------------------------------------------------------------------------

# Front-matter metadata (title page, pages 1-2) lives BEFORE PDF_PAGE_START and
# has no numbered heading, so the heading-based extractor never captures it.
# Inject it as one synthetic section so questions like "who wrote this thesis?"
# / "quem é o autor?" have something to retrieve. Written bilingually (PT/EN) to
# match the assistant's bilingual scope.
METADATA_SECTION_TEXT = (
    "Autor / Author: José Luís Leitão de Matos.\n"
    "Título / Title: Análise e melhoria de fluxos em operações intralogísticas "
    "de retalho através da aplicação de conceitos Lean e simulação discreta.\n"
    "Orientadores / Supervisors: Professor Doutor Bruno Samuel Ferreira Gonçalves; "
    "Professor Doutor Rui Manuel de Sá Pereira de Lima.\n"
    "Instituição / Institution: Universidade do Minho, Escola de Engenharia.\n"
    "Grau / Degree: Dissertação de Mestrado em Engenharia e Gestão de Operações, "
    "Ramo de Especialização em Gestão Industrial.\n"
    "Data / Date: Janeiro de 2025.\n\n"
    "Esta dissertação de mestrado foi escrita por José Luís Leitão de Matos e "
    "orientada pelos Professores Doutores Bruno Samuel Ferreira Gonçalves e Rui "
    "Manuel de Sá Pereira de Lima, na Universidade do Minho. "
    "This master's thesis was written by José Luís Leitão de Matos and supervised "
    "by Professor Bruno Samuel Ferreira Gonçalves and Professor Rui Manuel de Sá "
    "Pereira de Lima at the University of Minho."
)



# DETECTING PATTERNS -----------------------------------------------------------------

# Subsection: "X.X.X Title"
SUBSECTION_RE = re.compile(
    r"^\s*(\d+\.\d+\.\d+)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{2,100})$",
    re.MULTILINE,
)

# Section: "X.X   Title"
SECTION_RE = re.compile(
    r"^\s*(\d+\.\d+)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{2,100})$",
    re.MULTILINE,
)

# Chapter: "X. Title"
CHAPTER_RE = re.compile(
    r"^\s*(\d+)\.\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{2,200})$",
    re.MULTILINE,
)



# FUNCTIONS -----------------------------------------------------------------------------

# Extract all the lines from the PDF, keeping track of page numbers
def extract_pages_with_lines(pdf_path, start, end):
    """
    Pulls text from each page
    Returns a list of (page number, text) tuples
    """
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start - 1, end):
            page = pdf.pages[i]

            # Find table regions on this page (bounding boxes)
            table_bboxes = [t.bbox for t in page.find_tables()]

            if table_bboxes:
                # Keep only characters whose center falls OUTSIDE
                # every table box -> tables vanish from the prose
                def outside_tables(obj, bboxes=table_bboxes):
                    cx = (obj["x0"] + obj["x1"]) / 2
                    cy = (obj["top"] + obj["bottom"]) / 2
                    return not any(
                        x0 <= cx <= x1 and top <= cy <= bottom
                        for (x0, top, x1, bottom) in bboxes
                    )
                page = page.filter(outside_tables)

            text = page.extract_text() or ""
            lines = text.split("\n")
            pages.append((i + 1, lines))  # store as 1-indexed page number
    
    return pages

def classify_heading(line):
    """
    Look at one line and decide if it's a heading.
    Returns (level, number, title) or None
    Level can be: 'chapter', 'section', 'subsection'
    """
    stripped = line.strip()

    # Subsection check first (most specific): X.Y.Z  Title
    m = re.match(r"^(\d+\.\d+\.\d+)\s+(\S.{2,150})$", stripped)
    if m:
        return ("subsection", m.group(1), m.group(2).strip())

    # Section: X.Y  Title
    m = re.match(r"^(\d+\.\d+)\s+(\S.{2,150})$", stripped)
    if m:
        return ("section", m.group(1), m.group(2).strip())

    # Chapter: X.  Title  (only single-digit chapter numbers in your thesis)
    m = re.match(r"^(\d+)\.\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇa-záéíóúâêôãõç\s\-,]{2,150})$", stripped)
    if m:
        # Filter out things that look like list items: "X. Title:"
        # Heuristic: if title is short and ends with ":", probably a list item
        title = m.group(2).strip()
        if title.endswith(":") and len(title) < 30:
            return None
        return ("chapter", m.group(1), title)

    return None

def find_wrapped_continuation(flat_lines, i, level):
    """
    Look ahead from position i to find a wrapped title continuation.
    Returns (continuation_text, new_index) if a continuation is found,
    else (None, i).

    A "continuation" is a line that looks like the second half of a heading
    that wrapped onto a new line. The heuristic differs by level:
      - Chapters: ALL CAPS (matching the styled chapter-title look)
      - Sections/subsections: starts with an uppercase letter (Title Case)
    Common rejection rules apply to all levels: must be short, must not
    start with a digit (which would indicate a new numbered heading), and
    must not end with a period (which would indicate a sentence).
    """
    # Skip blank lines after the heading
    j = i + 1
    while j < len(flat_lines) and flat_lines[j][1].strip() == "":
        j += 1

    if j >= len(flat_lines):
        return None, i

    next_line = flat_lines[j][1].strip()

    # Common rejection rules for any heading level
    if not next_line:
        return None, i
    if re.match(r"^\d", next_line):  # next line starts a new numbered heading
        return None, i
    if next_line.endswith("."):
        return None, i

    # Level-specific checks
    if level == "chapter":
        # Chapter titles in this thesis are ALL CAPS
        if len(next_line) >= 80:
            return None, i
        if next_line != next_line.upper():
            return None, i
    else:
        # Section/subsection titles are Title Case — first char uppercase.
        # Use a tighter length cap to reduce false positives from body text.
        if len(next_line) >= 60:
            return None, i
        if not next_line[0].isupper():
            return None, i

    return next_line, j


def build_sections(pages):
    """
    Walks through all lines from all pages, detecting headings to establish
    section boundaries. Returns a list of section dicts.

    Handles wrapped headings at all levels: if a title wraps to two lines
    (like "4. DESCRIÇÃO ... DO MODELO DE\n    SIMULAÇÃO"), the next line is
    appended to the title if it looks like a continuation.
    """
    # Flatten everything into (page, line) pairs first so we can look ahead
    flat_lines = []
    for pdf_page_num, lines in pages:
        for line in lines:
            flat_lines.append((pdf_page_num, line))

    sections = []
    current = None
    current_chapter = None
    current_section = None

    i = 0
    while i < len(flat_lines):
        pdf_page_num, line = flat_lines[i]
        heading = classify_heading(line)

        if heading is not None:
            level, number, title = heading

            # Look ahead for a wrapped title continuation, regardless of level
            if i + 1 < len(flat_lines):
                continuation, new_i = find_wrapped_continuation(
                    flat_lines, i, level
                )
                if continuation:
                    title = f"{title} {continuation}"
                    i = new_i  # skip past the continuation line

            # Close the previous section
            if current is not None:
                current["page_end"] = pdf_page_num
                sections.append(current)

            # Update parent context
            if level == "chapter":
                current_chapter = (number, title)
                current_section = None
            elif level == "section":
                current_section = (number, title)

            # Open new section
            current = {
                "level": level,
                "number": number,
                "title": title,
                "chapter_number": current_chapter[0] if current_chapter else None,
                "chapter_title": current_chapter[1] if current_chapter else None,
                "section_number": current_section[0] if current_section and level == "subsection" else None,
                "section_title": current_section[1] if current_section and level == "subsection" else None,
                "page_start": pdf_page_num,
                "page_end": pdf_page_num,
                "text": "",
            }
        else:
            if current is not None:
                current["text"] += line + "\n"

        i += 1

    if current is not None:
        current["page_end"] = flat_lines[-1][0]
        sections.append(current)

    return sections



# MAIN -----------------------------------------------------------------------------

def build_metadata_section():
    """
    Synthetic front-matter section carrying author/title/supervisor metadata that
    lives on the title page (before PDF_PAGE_START) and would otherwise be lost.
    """
    return {
        "level": "metadata",
        "number": "0",
        "title": "Metadados da Dissertação / Thesis Metadata",
        "chapter_number": None,
        "chapter_title": "Metadados da Dissertação / Thesis Metadata",
        "section_number": None,
        "section_title": None,
        "page_start": 1,
        "page_end": 2,
        "text": METADATA_SECTION_TEXT,
    }


def main():
    # Get raw text lines from each page
    pages = extract_pages_with_lines(PDF_PATH, PDF_PAGE_START, PDF_PAGE_END)

    # Build sections based on detected headings
    sections = build_sections(pages)

    # Prepend front-matter metadata (author, title, supervisors) that the
    # heading-based extractor cannot capture.
    sections.insert(0, build_metadata_section())

    # Save the extracted sections to a JSON file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()